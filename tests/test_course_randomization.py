"""Requested DR distributions, native readback, and sensor/control integration."""
from dataclasses import replace
from pathlib import Path
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from supermimic_rl.adapters.mechanics import MechanicalParameters
from supermimic_rl.course.randomization import (
    CourseRandomization, RequestedRandomizationConfig, install_identified_feedback,
    make_randomized_environment, make_sensor_model_type, make_startup_dynamics_type, sample_mechanics,
    sensor_to_body_rotation,
)


UPSTREAM_ROOT = Path(os.environ.get(
    "SUPERMIMIC_UPSTREAM_ROOT",
    str(Path(__file__).resolve().parents[2] / "transformer_rl-worktrees/full-vehicle-gimbal"),
))
UPSTREAM = UPSTREAM_ROOT / "src"
if not (UPSTREAM / "transformer_rl/formal/lab_env/sensors.py").exists():
    pytest.skip("Pinned upstream sensor/control code is required", allow_module_level=True)
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from transformer_rl.formal.control.actuation import OutputFeedbackController  # noqa: E402
from transformer_rl.formal.control.observation import MeasuredState  # noqa: E402
from transformer_rl.formal.control.sensors import SensorModelConfig  # noqa: E402
from transformer_rl.formal.lab_env.sensors import LabSensorModel  # noqa: E402


def randomized(count=128, seed=23, **kwargs):
    model = CourseRandomization(count, seed=seed, config=RequestedRandomizationConfig(**kwargs))
    model.reset(torch.ones(count, dtype=torch.bool), 5000)
    return model


def state(count, timestamp=0.):
    return MeasuredState(
        torch.zeros(count, 6), torch.zeros(count, 6), torch.zeros(count, 18), torch.zeros(count, 18),
        torch.zeros(count, 3), torch.ones(count, 3), torch.tensor([[0., 0., -1.]]).expand(count, -1),
        torch.full((count,), .32), torch.zeros(count, 2), torch.full((count, 2), .4),
        torch.zeros(count, 2), torch.full((count,), timestamp, dtype=torch.float64),
    )


def mechanics(count=128):
    names = ("L_link3", "base_link", "R_link3")
    mass = torch.ones(count, 3)
    com = torch.zeros(count, 3, 7)
    com[..., 6] = 1
    inertia = torch.diag(torch.tensor([.02, .03, .04])).flatten().expand(count, 3, 9).clone()
    return MechanicalParameters(names, mass, com, inertia, "fixture", "link", "link")


def test_shared_nominal_lanes_and_independent_biases_use_requested_si_bounds():
    model = randomized(1024)
    nominal = model.nominal
    assert .20 < nominal.float().mean() < .30
    assert torch.equal(model.motor_strength[nominal], torch.ones_like(model.motor_strength[nominal]))
    for value in (model.encoder_bias_rad, model.policy_encoder_bias_rad,
                  model.gyro_bias_rad_s, model.imu_rpy_rad):
        assert value[nominal].count_nonzero() == 0
    assert model.motor_strength.min() >= .84 and model.motor_strength.max() <= 1.16
    assert (model.motor_strength > 1).any() and (model.motor_strength < 1).any()
    assert model.encoder_bias_rad[:, 4:].count_nonzero() == 0
    assert model.policy_encoder_bias_rad[:, 4:].count_nonzero() == 0
    assert model.encoder_bias_rad.abs().max() <= .004
    assert model.policy_encoder_bias_rad.abs().max() <= .004
    assert not torch.equal(model.encoder_bias_rad, model.policy_encoder_bias_rad)
    assert model.gyro_bias_rad_s.abs().max() <= .00525
    assert torch.all(model.imu_rpy_rad.abs() <= torch.tensor([.8, .8, .8 / 3]) * torch.pi / 180)
    assert model.strength.eq(1).all() and model.no_load_scale.eq(1).all()
    assert model.time_constant_scale.eq(1).all()
    assert model.config.provenance()["source_sha256"]["contract"].startswith("5e2e6452")


def test_partial_reset_preserves_other_rows_and_cold_restore_retains_rng():
    model = randomized(4)
    saved = model.state_dict()
    continued = randomized(4)
    continued.restore_after_physical_reset(saved)
    mask = torch.tensor([False, True, False, True])
    before = model.context_parameters()
    before = {name: value.clone() for name, value in before.items()}
    for item in (model, continued):
        item.reset(mask, 3000)
    for name, actual in model.context_parameters().items():
        torch.testing.assert_close(actual[~mask], before[name][~mask], rtol=0, atol=0)
        torch.testing.assert_close(actual, continued.context_parameters()[name], rtol=0, atol=0)


def test_mechanical_draw_preserves_nominal_spd_and_body_name_identity():
    model, source = randomized(), mechanics()
    mass, com, inertia, scale, independent = sample_mechanics(source, model, 17)
    assert scale.min() >= .92 and scale.max() <= 1.08
    assert independent.min() >= .94 and independent.max() <= 1.06
    torch.testing.assert_close(inertia, source.inertia_kg_m2 * (scale * independent)[..., None])
    assert (torch.linalg.eigvalsh(inertia.reshape(128, 3, 3, 3)) > 0).all()
    assert torch.equal(mass[model.nominal], source.mass_kg[model.nominal])
    assert torch.equal(inertia[model.nominal], source.inertia_kg_m2[model.nominal])
    assert torch.equal(com[:, [0, 2]], source.com_pose_local_xyzw[:, [0, 2]])
    assert torch.all(com[:, 1, :3].abs() <= torch.tensor([.012, .0064, .008]))
    order = [2, 0, 1]
    reordered = replace(source, body_names=tuple(source.body_names[i] for i in order),
                        mass_kg=source.mass_kg[:, order], com_pose_local_xyzw=source.com_pose_local_xyzw[:, order],
                        inertia_kg_m2=source.inertia_kg_m2[:, order])
    alternate = sample_mechanics(reordered, model, 17)
    for actual, expected in zip(alternate, (mass, com, inertia, scale, independent)):
        torch.testing.assert_close(actual, expected[:, order], rtol=0, atol=0)


def test_fixed_imu_rotation_and_independent_policy_bias_are_applied_once():
    model = randomized(2, nominal_fraction=0.)
    measurement = state(2)
    biased = replace(measurement, policy_position_rad=measurement.policy_position_rad + model.encoder_bias_rad)
    result = model.transform_measurement(biased)
    torch.testing.assert_close(result.policy_position_rad, model.policy_encoder_bias_rad, rtol=0, atol=0)
    rotation = sensor_to_body_rotation(model.imu_rpy_rad)
    torch.testing.assert_close(rotation @ rotation.transpose(-1, -2), torch.eye(3).expand(2, -1, -1))
    torch.testing.assert_close(result.gravity_body, (rotation.transpose(-1, -2) @ measurement.gravity_body[..., None]).squeeze(-1))
    torch.testing.assert_close(result.angular_velocity_body_rad_s,
        (rotation.transpose(-1, -2) @ measurement.angular_velocity_body_rad_s[..., None]).squeeze(-1) + model.gyro_bias_rad_s)


def test_native_sensor_gaussian_after_delay_matches_generator_and_keeps_reset_cache():
    model = randomized(4, nominal_fraction=0.)
    sensor_type = make_sensor_model_type(LabSensorModel, model)
    config = SensorModelConfig(policy_dt_s=.02, nominal_episode_fraction=.25,
                               observation_delays_s=(.01,), extra_delay_probability=1., shared_delay=True)
    sensor = sensor_type(4, 53, config)
    mask, now = torch.ones(4, dtype=torch.bool), torch.zeros(4, dtype=torch.float64)
    source = state(4)
    source = replace(source, policy_position_rad=model.encoder_bias_rad.clone())
    sensor.reset(mask, now)
    assert sensor.lags.eq(2).all()  # No second independent nominal mask suppresses lag.
    sensor.seed_after_reset(mask, source)
    first = sensor(source)
    assert first.sampled_at_s.eq(0).all()
    assert sensor.noise_distribution == 1
    for tick in range(1, 5):
        source = replace(source, policy_velocity_rad_s=torch.full((4, 6), float(tick)),
                         timestamp_s=torch.full((4,), tick * .005, dtype=torch.float64))
        sensor.push(source, tick)
    generator = torch.Generator().set_state(sensor.generator.get_state())
    noise = torch.randn(4, 18, generator=generator) * sensor._halfwidth
    observed = sensor(source)
    torch.testing.assert_close(observed.policy_velocity_rad_s, torch.full((4, 6), 2.) + noise[:, 6:12])
    torch.testing.assert_close(observed.angular_velocity_body_rad_s,
        model.transform_measurement(source).angular_velocity_body_rad_s + noise[:, 12:15])
    torch.testing.assert_close(observed.gravity_body.norm(dim=-1), torch.ones(4))
    assert torch.equal(sensor.generator.get_state(), generator.get_state())
    continuing = observed.policy_velocity_rad_s[1:].clone()
    rng_before = sensor.generator.get_state().clone()
    partial = torch.tensor([True, False, False, False])
    model.reset(partial, 5000)
    sensor.reset(partial, source.timestamp_s)
    new_source = replace(source, policy_position_rad=model.encoder_bias_rad.clone())
    sensor.seed_after_reset(partial, new_source)
    torch.testing.assert_close(sensor(new_source).policy_velocity_rad_s[1:], continuing, rtol=0, atol=0)
    assert torch.equal(rng_before, sensor.generator.get_state())


def test_feedback_reuses_controller_and_real_validation_with_wheel_p_and_fixed_caps():
    controller = OutputFeedbackController((0.,) * 6, 2, pair_domain=None)
    held_reference = controller
    install_identified_feedback(controller)
    q, dq = torch.zeros(2, 6), torch.zeros(2, 6)
    now = torch.zeros(2, dtype=torch.float64)
    actions = torch.tensor([[.1] * 6, [3.] * 6])
    targets = controller.decode(actions, q, now)
    result = controller.effort(targets, q, dq, now, strength=torch.full((2, 6), 1.16))
    assert held_reference is controller
    torch.testing.assert_close(result.capped_effort[0], torch.tensor([4.64] * 4 + [.58, .58]))
    torch.testing.assert_close(result.capped_effort[1], torch.tensor([40.] * 4 + [4.5, 4.5]))
    weakened = controller.effort(targets, q, dq, now, strength=torch.full((2, 6), .84))
    torch.testing.assert_close(weakened.capped_effort[1], torch.tensor([33.6] * 4 + [3.78, 3.78]))
    with pytest.raises(ValueError, match="twice"):
        install_identified_feedback(controller)
    with pytest.raises(ValueError, match="future"):
        controller.effort(replace(targets, controller_issued_at_s=now + 1), q, dq, now)


class FakeRobot:
    def __init__(self, source, ignore_inertia=False):
        self.body_names = source.body_names
        self.joint_names = ("R_joint3", "passive", "L_joint3")
        self.num_instances = source.mass_kg.shape[0]
        self.data = SimpleNamespace(body_mass=source.mass_kg.clone(),
            body_com_pos_b=source.com_pose_local_xyzw[..., :3].clone(),
            body_inertia=source.inertia_kg_m2.clone(), joint_armature=torch.zeros(self.num_instances, 3))
        self.ignore_inertia = ignore_inertia

    def set_masses_index(self, *, masses, env_ids, body_ids):
        self.data.body_mass.copy_(masses)

    def set_coms_index(self, *, coms, env_ids, body_ids):
        self.data.body_com_pos_b.copy_(coms)

    def set_inertias_index(self, *, inertias, env_ids, body_ids):
        if not self.ignore_inertia:
            self.data.body_inertia.copy_(inertias)

    def write_joint_armature_to_sim_index(self, *, armature, joint_ids, env_ids):
        self.data.joint_armature[:, joint_ids] = armature


def test_native_newton_setters_read_back_and_wheel_armature_share_one_draw(monkeypatch):
    module = ModuleType("transformer_rl.formal.lab_env.newton_properties")
    module.native_tensor = lambda value: value
    module.notify_newton_properties = lambda flag: None
    monkeypatch.setitem(sys.modules, module.__name__, module)

    class Base:
        def __init__(self, robot, binding, num_envs, mass, seed, config, **kwargs):
            assert config is None
            self._receipt = {}

    class Execution:
        def native_parameters(self):
            return {"L_joint3": {"armature": .2}, "R_joint3": {"armature": .3}}

        def set_wheel_inertia_scale(self, value):
            self.scale = value.clone()

    source, model, execution = mechanics(), randomized(), Execution()
    robot = FakeRobot(source)
    dyn_type = make_startup_dynamics_type(Base, model, execution)
    dynamics = dyn_type(robot, None, 128, 3., 123, None, device="cpu")
    scale = dynamics.body_inertia_scale[:, [0, 2]]
    torch.testing.assert_close(execution.scale, scale)
    torch.testing.assert_close(robot.data.joint_armature[:, [2, 0]], scale * torch.tensor([.2, .3]))
    assert dynamics._receipt["wheel_armature_same_body_draw"]
    assert dynamics._receipt["configuration_sha256"]
    assert not torch.equal(robot.data.body_inertia, source.inertia_kg_m2)
    with pytest.raises(RuntimeError, match="inertia readback"):
        dyn_type(FakeRobot(source, ignore_inertia=True), None, 128, 3., 123, None, device="cpu")


def test_disturbances_follow_native_semantics_and_spare_nominal_lanes():
    quiet, legacy = randomized(64), randomized(64)
    assert quiet.config.pushes is False and quiet.advance_wrench(100.).abs().sum() == 0
    rows, delta = quiet.push(100., torch.ones(64, dtype=torch.bool))
    assert rows.numel() == 0 and delta.shape == (0, 3)
    torch.testing.assert_close(quiet.generator.get_state(), legacy.generator.get_state())
    model = randomized(64, push_linear_m_s=.5, push_yaw_rad_s=1., external_force_n=15., external_torque_nm=1.5,
                       push_interval_s=(5., 10.), wrench_interval_s=(5., 10.))
    assert ((model.push_timer_s >= 5) & (model.push_timer_s <= 10)).all()
    assert model.advance_wrench(4.9).abs().sum() == 0      # nothing due before the first interval
    wrench = model.advance_wrench(5.2)
    disturbed = ~model.nominal
    assert wrench[model.nominal].abs().sum() == 0 and wrench[disturbed].abs().sum() > 0
    assert (wrench[:, :3].abs() <= 15).all() and (wrench[:, 3:].abs() <= 1.5).all()
    torch.testing.assert_close(model.advance_wrench(.1), wrench)   # held until redrawn
    eligible = torch.arange(64) % 2 == 0
    rows, delta = model.push(10.1, eligible)
    assert set(rows.tolist()) == set(eligible.nonzero().flatten().tolist())
    assert (delta[:, :2].abs() <= .5).all() and (delta[:, 2].abs() <= 1.).all()
    assert (delta[model.nominal[rows]] == 0).all() and (model.push_timer_s[rows] >= 5).all()
    ineligible = (~eligible).nonzero().flatten()
    assert (model.push_timer_s[ineligible] <= 0).all()   # waits for an eligible mode, as natively
    with pytest.raises(ValueError, match="ordered"):
        RequestedRandomizationConfig(push_interval_s=(10., 5.))
    with pytest.raises(ValueError, match="positive"):
        RequestedRandomizationConfig(motor_strength_fraction=1.)
    model = randomized(4)
    with pytest.raises(ValueError, match="boolean"):
        model.reset(torch.ones(4), 1)


def test_native_constructor_bindings_restore_on_success_and_failure(monkeypatch):
    import transformer_rl.formal.lab_env as package

    native = ModuleType("transformer_rl.formal.lab_env.env")
    dynamics = ModuleType("transformer_rl.formal.lab_env.dynamics")
    materials = ModuleType("transformer_rl.formal.lab_env.materials")
    original_material = type("BaseMaterial", (), {})
    materials.NewtonWheelMaterialReader = original_material
    properties = ModuleType("transformer_rl.formal.lab_env.newton_properties")
    properties.native_tensor = lambda value: value
    properties.notify_newton_properties = lambda flag: None
    original_episode = object()
    native.EpisodeRandomization, native.LabSensorModel = original_episode, LabSensorModel

    class BaseDynamics:
        def __init__(self, *args, **kwargs):
            self._receipt = {}

    dynamics.NewtonStartupDynamics = BaseDynamics
    for module in (native, dynamics, properties, materials):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(package, "env", native, raising=False)
    monkeypatch.setattr(package, "dynamics", dynamics, raising=False)
    monkeypatch.setattr(package, "materials", materials, raising=False)
    shared = randomized(4)

    class NativeBase:
        def __init__(self, fail=False):
            assert issubclass(materials.NewtonWheelMaterialReader, original_material)
            assert materials.NewtonWheelMaterialReader is not original_material
            self._is_newton = True
            self._startup_dynamics = dynamics.NewtonStartupDynamics(
                FakeRobot(mechanics(4)), None, 4, 3., 71, None, device="cpu")
            self._episode_randomization = native.EpisodeRandomization(4, None, seed=17, device="cpu")
            self._sensor_model = native.LabSensorModel(4, 3, SensorModelConfig())
            self._control = SimpleNamespace(controller=OutputFeedbackController((0.,) * 6, 4, pair_domain=None))
            if fail:
                raise RuntimeError("native constructor failed")

    wrapped = make_randomized_environment(NativeBase, shared, None)
    environment = wrapped()
    assert environment._requested_randomization is shared
    assert environment._episode_randomization is shared
    assert environment._sensor_model.noise_distribution == 1
    assert environment._control.controller._requested_feedback_installed
    assert native.EpisodeRandomization is original_episode
    assert native.LabSensorModel is LabSensorModel
    assert dynamics.NewtonStartupDynamics is BaseDynamics
    assert materials.NewtonWheelMaterialReader is original_material
    with pytest.raises(RuntimeError, match="native constructor failed"):
        wrapped(fail=True)
    assert native.EpisodeRandomization is original_episode
    assert native.LabSensorModel is LabSensorModel
    assert dynamics.NewtonStartupDynamics is BaseDynamics
    assert materials.NewtonWheelMaterialReader is original_material


class FakePhysxView:
    def __init__(self, robot):
        self.robot, self.count = robot, robot.num_instances

    def get_masses(self):
        return self.robot.mass.clone()

    def get_coms(self):
        return self.robot.com.clone()

    def get_inertias(self):
        return self.robot.inertia.clone()


class FakePhysxRobot:
    """PhysX public setters take full 7D COM poses and require full_data."""

    def __init__(self, source):
        self.body_names = ("L_link3", "base_link", "R_link3")
        self.joint_names = ("R_joint3", "passive", "L_joint3")
        self.num_instances = source.mass_kg.shape[0]
        self.mass, self.com = source.mass_kg.clone(), source.com_pose_local_xyzw.clone()
        self.inertia = source.inertia_kg_m2.clone()
        self.root_view = FakePhysxView(self)
        self.data = SimpleNamespace(joint_armature=torch.zeros(self.num_instances, 3))
        self.com_writes = 0
        self.sim_device = torch.device("meta")

    def set_masses_index(self, *, masses, env_ids, body_ids, full_data):
        assert full_data and masses.device == self.sim_device
        self.mass.copy_(masses)

    def set_coms_index(self, *, coms, env_ids, body_ids, full_data):
        assert full_data and coms.shape[-1] == 7
        self.com.copy_(coms)
        self.com_writes += 1

    def set_inertias_index(self, *, inertias, env_ids, body_ids, full_data):
        assert full_data
        self.inertia.copy_(inertias)

    def write_joint_armature_to_sim_index(self, *, armature, joint_ids, env_ids):
        self.data.joint_armature[:, joint_ids] = armature


def test_physx_writer_uses_full_com_poses_and_republishes_the_final_frame():
    from supermimic_rl.course.randomization import PhysxMechanicsWriter

    class Base:
        def __init__(self, robot, binding, num_envs, mass, seed, config, **kwargs):
            assert config is None
            self._receipt = {}

    class Execution:
        device = torch.device("cpu")

        def native_parameters(self):
            return {"L_joint3": {"armature": .2}, "R_joint3": {"armature": .3}}

        def set_wheel_inertia_scale(self, value):
            self.scale = value.clone()

    source, model, execution = mechanics(), randomized(), Execution()
    robot = FakePhysxRobot(source)
    dyn_type = make_startup_dynamics_type(Base, model, execution, PhysxMechanicsWriter())
    # Setters must receive simulation-device tensors even though getters return host tensors.
    with pytest.raises(AssertionError):
        dyn_type(robot, None, 128, 3., 123, None, device="cpu")
    robot = FakePhysxRobot(source)
    robot.sim_device = torch.device("cpu")
    dynamics = dyn_type(robot, None, 128, 3., 123, None, device="cpu")
    assert robot.com_writes == 2 and dynamics._receipt["property_backend"] == "physx"
    torch.testing.assert_close(robot.data.joint_armature[:, [2, 0]],
                               dynamics.body_inertia_scale[:, [0, 2]] * torch.tensor([.2, .3]))
    torch.testing.assert_close(robot.com[..., 3:], source.com_pose_local_xyzw[..., 3:])
