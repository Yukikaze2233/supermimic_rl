"""Exercise nominal backend contracts without importing Isaac or running Warp."""

from copy import deepcopy
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from supermimic_rl.adapters.kamino import (
    KaminoBackend, KaminoHealth, KaminoSettings, NewtonFlatContactLoads,
    NewtonNominalWheelMaterial, NewtonWorldBinding, _module_bindings,
    make_nominal_dynamics_type,
)


class Array:
    def __init__(self, values):
        self.value = np.asarray(values)

    def numpy(self):
        return self.value.copy()

    def assign(self, value):
        self.value[...] = value

    def zero_(self):
        self.value[...] = 0


class Robot:
    body_names = ["base_link", "left", "right"]
    joint_names = ["left_axis", "right_axis"]
    num_instances = 2

    def __init__(self):
        self.data = SimpleNamespace(
            body_mass=torch.tensor([[2., 1., 1.], [2., 1., 1.]]),
            body_com_pos_b=torch.tensor([[[.1, .2, .3], [0., 0., 0.], [0., 0., 0.]]]).repeat(2, 1, 1),
            body_inertia=torch.eye(3).flatten().repeat(2, 3, 1),
        )

    @property
    def root_view(self):
        raise AssertionError("Newton nominal code must not use a PhysX view")


class UpstreamDynamics:
    @property
    def receipt(self):
        return deepcopy(self._receipt)

    def base_com_position(self, position, quaternion):
        return "upstream implementation retained"


def model_and_manager():
    # Body and shape orders are intentionally different from world/robot order.
    model = SimpleNamespace(
        world_count=2, articulation_count=2, body_count=6, joint_count=54,
        articulation_start=Array([0, 27, 54]), articulation_end=Array([21, 48]),
        body_label=["/e1/right", "/e0/base_link", "/e0/left", "/e0/right", "/e1/left", "/e1/base_link"],
        body_world=Array([1, 0, 0, 0, 1, 1]),
        shape_body=Array([-1, 0, 2, 3, 4, 2, 1, 5]),
        shape_material_mu=Array(np.array([.7, .8, .3, .4, .6, .3, .2, .9], dtype=np.float32)),
        shape_material_restitution=Array(np.array([0, .08, .03, .04, .06, .03, .02, .09], dtype=np.float32)),
        joint_armature=Array(np.array([.1, .2, .1, .2], dtype=np.float32)),
        joint_damping=Array(np.array([.3, .4, .3, .4], dtype=np.float32)),
        joint_friction=Array(np.array([.5, .6, .5, .6], dtype=np.float32)),
    )
    status = np.zeros(2, dtype=[("iterations", "i4"), ("r_p", "f4"), ("r_d", "f4"),
                               ("r_c", "f4"), ("converged", "?")])
    status["iterations"] = [12, 200]
    status["converged"] = [True, False]
    status["r_p"] = [.1, .2]
    notifications = []
    manager = SimpleNamespace(backend=SimpleNamespace(model=model),
                              _solver=SimpleNamespace(status=Array(status), notify_model_changed=notifications.append))
    return model, manager, notifications


@pytest.mark.parametrize("values", [{"padmm_iterations": 0}, {"max_contacts_per_world": True},
                                    {"robot_friction": float("nan")}, {"ground_friction": -1.},
                                    {"integrator": "unsupported"}, {"use_cuda_graph": 1},
                                    {"rotation_correction": "twopi"}, {"rotation_correction": None}])
def test_settings_reject_invalid_native_contract(values):
    with pytest.raises(ValueError):
        KaminoSettings(**values)


def test_native_rotation_correction_is_explicitly_continuous():
    assert KaminoSettings().rotation_correction == "continuous"
    assert KaminoSettings(rotation_correction="continuous").rotation_correction == "continuous"


def test_nominal_dynamics_reads_owned_core_properties_and_retains_upstream_methods():
    robot = Robot()
    binding = SimpleNamespace(native_body_names=robot.body_names)
    cls = make_nominal_dynamics_type(UpstreamDynamics)
    state = cls(robot, binding, 2, 4., 7, SimpleNamespace(nominal_fraction=1.), device="cpu")
    assert torch.equal(state.mass_kg_by_environment, torch.tensor([4., 4.]))
    assert state.base_com_position(None, None) == "upstream implementation retained"
    assert torch.equal(state.base_com_delta_local_m, torch.zeros(2, 3))
    robot.data.body_mass.zero_()
    assert state.mass_kg_by_body.sum().item() == 8.
    assert state.receipt["randomization_active"] is False
    assert state.receipt["property_writes_performed"] is False
    receipt = state.receipt
    receipt["body_order"].clear()
    assert len(state.receipt["body_order"]) == 3


def test_nominal_dynamics_rejects_dr_wrong_mass_and_wrong_order():
    cls, robot = make_nominal_dynamics_type(UpstreamDynamics), Robot()
    binding = SimpleNamespace(native_body_names=robot.body_names)
    with pytest.raises(ValueError, match="nominal"):
        cls(robot, binding, 2, 4., 7, SimpleNamespace(nominal_fraction=.2), device="cpu")
    with pytest.raises(ValueError, match="mass"):
        cls(robot, binding, 2, 5., 7, None, device="cpu")
    with pytest.raises(ValueError, match="bound core"):
        cls(robot, SimpleNamespace(native_body_names=list(reversed(robot.body_names))),
            2, 4., 7, None, device="cpu")


def test_material_ownership_uses_world_ids_and_all_shapes():
    model, manager, _ = model_and_manager()
    material = NewtonNominalWheelMaterial(Robot(), [1, 2], num_envs=2, device="cpu", manager=manager)
    assert torch.allclose(material.features(), torch.tensor([[.3, .3, .03, .4, .4, .04],
                                                            [.6, .6, .06, .8, .8, .08]]))
    model.shape_material_mu.value[5] = .9
    with pytest.raises(ValueError, match="share one"):
        material.refresh()
    with pytest.raises(ValueError, match="randomization"):
        material.randomize_startup(None)


def test_material_rejects_missing_world_owner():
    model, manager, _ = model_and_manager()
    model.body_world.value[4] = 0
    with pytest.raises(ValueError, match="unique"):
        NewtonNominalWheelMaterial(Robot(), [1, 2], num_envs=2, device="cpu", manager=manager)


def test_flat_contact_projects_core_without_losing_auxiliary_shape_alignment():
    forces = torch.zeros(2, 4, 2, 3)
    forces[..., 2] = torch.tensor([[[1., 2.], [100., 200.], [3., 4.], [5., 6.]]]).repeat(2, 1, 1)
    sensor = SimpleNamespace(body_names=["right", "gimbal", "base_link", "left"],
                             data=SimpleNamespace(normal_force_matrix_w=forces))
    binding = SimpleNamespace(native_body_names=["base_link", "left", "right"], wheel_body_indices=[1, 2])
    loads = NewtonFlatContactLoads(sensor, binding, ["e0", "e1"], device="cpu")
    normal, support = loads.read(1, .005)
    assert normal.tolist() == [[7., 11., 3.], [7., 11., 3.]]
    assert support.tolist() == [[11., 3.], [11., 3.]]
    loads.reset_rows(torch.tensor([1]), 1)
    assert loads.valid.tolist() == [True, False]
    assert loads.read(1, .005)[0][1].eq(0).all()
    loads.read(2, .005)
    assert loads.valid.all()
    forces[0, 0, 0, 2] = float("nan")
    with pytest.raises(ValueError, match="packet"):
        loads.read(3, .005)


def test_world_binding_rejects_missing_loop_and_extra_articulation():
    model, manager, _ = model_and_manager()
    world = NewtonWorldBinding(2, manager)
    world.finish_warmup()
    assert world.receipt["verified"]
    assert world.receipt["loop_joints_per_world"] == [6, 6]
    model.articulation_end.value[0] += 1
    with pytest.raises(ValueError, match="six"):
        world.finish_warmup()
    model.articulation_count = 3
    with pytest.raises(ValueError, match="one full articulation"):
        world.finish_warmup()


def test_health_records_nonconvergence_but_rejects_nonfinite_status():
    _, manager, _ = model_and_manager()
    health = KaminoHealth(manager)
    assert health.summary["converged_fraction"] is None
    health.check()
    manager._solver.status.value["converged"][:] = True
    health.check()
    assert health.summary["checks"] == 2
    assert health.summary["solver_world_samples"] == 4
    assert health.summary["converged_world_samples"] == 3
    assert health.summary["converged_fraction"] == .75
    assert health.summary["max_iterations"] == 200
    manager._solver.status.value["r_p"][0] = float("nan")
    with pytest.raises(FloatingPointError):
        health.check()
    assert health.summary["checks"] == 2
    health.close()
    with pytest.raises(RuntimeError, match="closed"):
        health.check()


def test_constructor_module_bindings_restore_after_failure():
    module = SimpleNamespace(material="physx", sensor="physx")
    with pytest.raises(RuntimeError):
        with _module_bindings(module, {"material": "newton", "sensor": "newton"}):
            assert module.material == "newton"
            raise RuntimeError("construction failed")
    assert vars(module) == {"material": "physx", "sensor": "physx"}


def make_backend():
    model, manager, notifications = model_and_manager()
    backend = KaminoBackend.__new__(KaminoBackend)
    backend.settings, backend.device = KaminoSettings(), "cpu"
    backend.manager = manager
    backend.physics_configuration = object()
    backend._actuator_receipt = {}
    backend._closed = False
    backend.health = KaminoHealth(manager)
    backend._world_binding = NewtonWorldBinding(2, manager)
    return backend, model, notifications


def test_prepare_scene_preserves_requested_response_and_uses_newton_binding(monkeypatch):
    backend, _, _ = make_backend()
    assets, sim = ModuleType("isaaclab.assets"), ModuleType("isaaclab.sim")
    assets.AssetBaseCfg = sim.GroundPlaneCfg = sim.RigidBodyMaterialCfg = SimpleNamespace
    monkeypatch.setitem(sys.modules, "isaaclab.assets", assets)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sim)
    actuator = SimpleNamespace(armature={"a": .1}, friction={"a": .2}, dynamic_friction={"a": .3},
                               viscous_friction={"a": .4}, damping=0.)
    robot = SimpleNamespace(actuators={"leg": actuator})
    cfg = SimpleNamespace(declarative_scene=True, course_manifest=None, static_plane_height_m=0.,
                          height_scanner=None, terrain_scanner=None, forward_scanners=(),
                          startup_dynamics_spec=SimpleNamespace(nominal_fraction=1.),
                          wheel_material_spec=None, episode_randomization_spec=None,
                          sim=SimpleNamespace(), scene=SimpleNamespace(num_envs=2, robot=robot), robot_cfg=robot)
    backend.prepare_scene(cfg)
    assert cfg.sim.physics is backend.physics_configuration
    assert cfg.sim.use_newton_actuators is False
    assert cfg._native_clone_binding is backend._world_binding
    assert actuator.dynamic_friction == {"a": 0.}
    assert actuator.armature == {"a": .1} and actuator.friction == {"a": .2}
    for record in backend._actuator_receipt.values():
        assert record["leg"]["dynamic_friction"] == {"a": .3}
    cfg.course_manifest = "rough-course.json"
    with pytest.raises(ValueError, match="flat plane"):
        backend.prepare_scene(cfg)


def test_finalize_records_actual_properties_and_notifies_material_update(monkeypatch):
    backend, model, notifications = make_backend()
    module = ModuleType("newton")
    module.ModelFlags = SimpleNamespace(SHAPE_PROPERTIES="shape_properties")
    monkeypatch.setitem(sys.modules, "newton", module)
    robot = Robot()
    env = SimpleNamespace(robot=robot, num_envs=2,
        _wheel_materials=NewtonNominalWheelMaterial(robot, [1, 2], num_envs=2, device="cpu", manager=backend.manager),
        contact_sensor=SimpleNamespace(kamino_contact_partner_receipt={"shape_ids": [0]}))
    result = backend.finalize_environment(env)
    assert notifications == ["shape_properties"]
    assert np.allclose(model.shape_material_mu.numpy(), .7)
    assert np.all(model.shape_material_restitution.numpy() == 0)
    assert result["identity"]["backend"] == "newton_kamino"
    assert result["readback"]["mechanics"]["mass_kg"]["row0"] == [2., 1., 1.]
    assert result["readback"]["world_binding"]["verified"]
    identity = deepcopy(result["identity"])
    robot.data.body_mass[0, 0] = 3.
    changed = backend.finalize_environment(env)
    assert changed["identity"] == identity
    assert changed["readback"]["mechanics"]["mass_kg"]["row0"] == [3., 1., 1.]
