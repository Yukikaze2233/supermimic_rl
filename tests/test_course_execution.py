"""CPU coverage of the real full-articulation effort boundary and reset owner."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.course.execution import (
    configure_identified_actuators,
    make_environment,
)
from supermimic_rl.course.identified_execution import ACTION_ORDER, IdentifiedExecution


ROOT = Path(__file__).resolve().parents[1]


def execution(count=2, mode="software_dynamic", dtype=torch.float32):
    return IdentifiedExecution(
        count, "cpu", .005, friction_mode=mode, dtype=dtype,
        asset_directory=ROOT / "assets/full_vehicle_v2p0p1_source_audited_v2",
        bundle_directory=ROOT / "configs/identified/20261031",
    )


def actuator(names, **kwargs):
    return SimpleNamespace(joint_names_expr=names, stiffness=0., damping=0.,
                           armature=0., friction=0., dynamic_friction=0.,
                           viscous_friction=0., effort_limit=100., effort_limit_sim=100., **kwargs)


def config():
    groups = {
        "legs": actuator(list(ACTION_ORDER[:4])),
        "wheels": actuator(list(ACTION_ORDER[4:])),
        "pressure": actuator(["L_spring_slide", "R_spring_slide"]),
        "gimbal": actuator(["M_joint1", "M_joint2"]),
    }
    groups["pressure"].damping = 2.
    groups["gimbal"].effort_limit = 4.5
    return SimpleNamespace(response_spec=None, robot_cfg=SimpleNamespace(actuators=groups))


def test_actuator_configuration_replaces_active_physics_without_touching_auxiliary():
    cfg, model = config(), execution()
    cfg.scene = SimpleNamespace(robot=deepcopy(cfg.robot_cfg))
    pressure, gimbal = deepcopy(cfg.robot_cfg.actuators["pressure"]), deepcopy(cfg.robot_cfg.actuators["gimbal"])
    receipt = configure_identified_actuators(cfg, model, backend="newton")
    assert receipt["native_readback_required"] is True
    assert receipt["declarative_scene_copy_updated"] is True
    assert receipt["modified_actuator_groups"] == ["legs", "wheels"]
    assert cfg.robot_cfg.actuators["pressure"] == pressure
    assert cfg.robot_cfg.actuators["gimbal"] == gimbal
    for name in ("legs", "wheels"):
        group = cfg.robot_cfg.actuators[name]
        for joint in group.joint_names_expr:
            assert group.armature[joint] == model.native_parameters()[joint]["armature"]
            assert group.friction[joint] == group.dynamic_friction[joint] == group.viscous_friction[joint] == 0
            assert group.effort_limit[joint] == model.native_effort_limits()[joint]
            assert group.effort_limit_sim[joint] == group.effort_limit[joint]
        assert group.stiffness == group.damping == 0
        assert vars(cfg.scene.robot.actuators[name]) == vars(group)


def test_unsupported_native_friction_and_duplicate_owners_fail_before_config_mutation():
    cfg = config()
    before = deepcopy(cfg)
    with pytest.raises(ValueError, match="separate native dynamic friction"):
        configure_identified_actuators(cfg, execution(mode="native_floor"), backend="newton")
    assert cfg == before
    cfg.robot_cfg.actuators["extra"] = actuator(["L_joint1"])
    before = deepcopy(cfg)
    with pytest.raises(ValueError, match="exactly one"):
        configure_identified_actuators(cfg, execution(), backend="newton")
    assert cfg == before


class FakeRobot:
    def __init__(self, model, names):
        self.body_names = tuple(reversed(list(model._bodies))) + ("M_link2", "M_link1")
        count, device, dtype = model.count, model.device, model.dtype
        pose = torch.zeros(count, len(self.body_names), 7, device=device, dtype=dtype)
        pose[..., 6] = 1.
        element = model.mount_force_element(self.body_names)
        desired = torch.tensor([[.18, 0., 0.], [0., 0., 0.], [.2, .4, 0.], [0., .4, 0.]], dtype=dtype)
        for index, point, target in zip(element.body_ids, element._points, desired):
            pose[:, index, :3] = target - point
        self.data = SimpleNamespace(joint_pos=torch.zeros(count, len(names), dtype=dtype),
                                    joint_vel=torch.ones(count, len(names), dtype=dtype),
                                    body_link_pose_w=pose, body_com_pose_w=pose.clone())
        self.commands, self.wrenches = [], []
        self.instantaneous_wrench_composer = SimpleNamespace(add_forces_and_torques_index=self.add_wrench)

    def add_wrench(self, **kwargs):
        self.wrenches.append(kwargs)

    def set_joint_effort_target(self, target):
        self.commands.append(target.clone())


def base_environment_type(model):
    class Base:
        def __init__(self):
            self.num_envs, self._device, self.physics_dt = model.count, model.device, model.dt
            names = tuple(reversed(list(model._joints))) + ("M_joint2", "M_joint1")
            self._binding = SimpleNamespace(native_joint_names=names)
            self._active_indices = torch.tensor([names.index(name) for name in ACTION_ORDER])
            self._spring_indices = torch.tensor([names.index(name) for name in ("L_spring_slide", "R_spring_slide")])
            self.robot = FakeRobot(model, names)
            self.scene = {"robot": self.robot}
            self._control = SimpleNamespace(drive=None, controller=SimpleNamespace(damping=torch.tensor([2.5] * 4 + [.6] * 2)))
            self._communication, self._motor_envelope = None, None
            self._recovery_preparation = SimpleNamespace(passive=torch.zeros(self.num_envs, dtype=torch.bool))
            self._last_effort = torch.zeros(self.num_envs, 6)
            self._spring_strength = torch.ones(self.num_envs, 1)
            self._pressure = self._build_pressure()
            self._spring_force = self._build_spring_force()
            self.reset_rows = []
            self.break_before_write = False
            self.double_write = False

        @staticmethod
        def _as_torch(tensor):
            return tensor

        def _apply_action(self):
            if self.break_before_write:
                raise RuntimeError("injected parent failure")
            raw_motor = torch.ones(self.num_envs, 6)
            raw_motor[self._recovery_preparation.passive] = 0.
            mount = self._spring_force.evaluate(self.robot.data.body_link_pose_w,
                                               self.robot.data.body_com_pose_w[..., :3],
                                               self._spring_strength)
            self.robot.instantaneous_wrench_composer.add_forces_and_torques_index(
                forces=mount.forces_world, torques=mount.torques_about_com_world,
                body_ids=list(self._spring_force.body_ids), is_global=True)
            command = torch.zeros(self.num_envs, len(self._binding.native_joint_names))
            command.index_copy_(1, self._active_indices, raw_motor)
            command[:, -2:] = torch.tensor([.75, -1.25])
            self.robot.set_joint_effort_target(command)
            if self.double_write:
                self.robot.set_joint_effort_target(command)
            self._last_effort.copy_(raw_motor)

        def _reset_idx(self, ids):
            self.reset_rows.append(ids.clone())
            self._last_effort[ids] = 0

    return Base


def test_parent_control_is_reused_while_native_active_axes_and_motor_diagnostics_are_correct():
    model = execution()
    environment = make_environment(base_environment_type(model), model)()
    original = environment.robot
    assert environment._control.controller.damping[4:].tolist() == [.5, .5]
    environment._apply_action()
    assert environment.robot is original is environment.scene["robot"]
    assert len(original.commands) == len(original.wrenches) == 1
    actual = original.commands[-1]
    torch.testing.assert_close(actual[:, environment._active_indices], environment._last_native_effort)
    torch.testing.assert_close(environment._last_native_effort,
                               environment._last_effort + environment._last_passive_effort)
    torch.testing.assert_close(environment._last_effort[:, 4:], torch.tensor([[0., .3], [0., .3]]))
    torch.testing.assert_close(actual[:, -2:], torch.tensor([[.75, -1.25], [.75, -1.25]]))
    assert not actual[:, environment._spring_indices].any()
    assert "gas_spring_model.json" in environment._spring_source_sha256
    assert "fit_10mpa.json" not in environment._spring_source_sha256


def test_passive_preparation_and_row_reset_clear_only_owned_execution_memory():
    model = execution()
    environment = make_environment(base_environment_type(model), model)()
    for _ in range(4):
        environment._apply_action()
    environment._recovery_preparation.passive[0] = True
    environment._apply_action()
    assert not environment._last_effort[0].any()
    assert (environment._last_passive_effort[0] < 0).all()
    assert not model.state_memory()["wheel_torque_history_nm"][0].any()
    previous_other = model.state_memory()["wheel_torque_history_nm"][1].clone()
    environment._reset_idx(torch.tensor([0]))
    torch.testing.assert_close(model.state_memory()["wheel_torque_history_nm"][1], previous_other)
    assert not environment._last_native_effort[0].any()
    assert environment.reset_rows[0].tolist() == [0]


@pytest.mark.parametrize("failure", ["break_before_write", "double_write"])
def test_proxy_restores_true_robot_on_parent_failure(failure):
    model = execution()
    environment = make_environment(base_environment_type(model), model)()
    original = environment.robot
    setattr(environment, failure, True)
    with pytest.raises(RuntimeError):
        environment._apply_action()
    assert environment.robot is original is environment.scene["robot"]


def test_slider_diagnostics_use_the_same_force_and_analytic_potential():
    model = execution()
    environment = make_environment(base_environment_type(model), model)()
    positions = model.spring_zero[None] - torch.tensor([[.03, .04], [.05, .06]])
    result = environment._pressure.evaluate(positions, strength=torch.full((2, 1), .8))
    torch.testing.assert_close(result.slider_effort_n, model.slider_effort(positions) * .8)
    assert result.within_fitted_curve.all()
    assert result.within_mechanical_stroke.all()
    assert torch.isfinite(result.model_pressure_potential_j).all()
    precise = execution(count=1, dtype=torch.float64)
    compression = torch.tensor([-.02, .019, .03, .05, .07, .11], dtype=torch.float64)
    delta = 1e-7
    derivative = (precise.spring_potential_j(compression + delta)
                  - precise.spring_potential_j(compression - delta)) / (2 * delta)
    torch.testing.assert_close(derivative, precise.physical_spring_force(compression), atol=1e-6, rtol=1e-7)
    assert precise.spring_potential_j(torch.tensor([0.], dtype=torch.float64)).item() == 0.
