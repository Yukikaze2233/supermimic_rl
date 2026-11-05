"""Gimbal encoder capture on the real upstream CPU sensor clock; no simulator."""
from types import SimpleNamespace
import os
from pathlib import Path
import sys

import pytest
import torch

_UPSTREAM = Path(os.environ.get("SUPERMIMIC_UPSTREAM_ROOT",
    Path(__file__).resolve().parents[2] / "transformer_rl-worktrees/full-vehicle-gimbal"))
if (_UPSTREAM / "src").is_dir():
    sys.path.insert(0, str(_UPSTREAM / "src"))

from supermimic_rl.course.adapter import GimbalSensorHistory


@pytest.fixture
def gimbal_sensor():
    source = pytest.importorskip("transformer_rl.formal.lab_env.sensors")
    config_type = pytest.importorskip("transformer_rl.formal.control.sensors").SensorModelConfig
    config = config_type(physics_dt_s=.005, policy_dt_s=.02,
                         observation_delays_s=(0., .005, .01))
    sensor = source.LabSensorModel(2, 41, config,
                                  uplink_delay_s=torch.full((2,), .0025, dtype=torch.float64),
                                  maximum_uplink_s=.0025)
    data = SimpleNamespace(joint_pos=torch.zeros(2, 4), joint_vel=torch.zeros(2, 4))
    env = SimpleNamespace(num_envs=2, device="cpu", _sensor_model=sensor,
                          _as_torch=lambda value: value,
                          robot=SimpleNamespace(data=data),
                          _gimbal=SimpleNamespace(joint_indices=[2, 0],
                              _zero=torch.tensor([.3, -.2]), _sign=torch.tensor([-1., 1.])))
    return env, GimbalSensorHistory(env), source.MeasuredState


def measured(measured_type, tick):
    return measured_type(
        policy_position_rad=torch.zeros(2, 6), policy_velocity_rad_s=torch.zeros(2, 6),
        tree_position_rad=torch.zeros(2, 18), tree_velocity_rad_s=torch.zeros(2, 18),
        velocity_origin_body_m_s=torch.zeros(2, 3), angular_velocity_body_rad_s=torch.zeros(2, 3),
        gravity_body=torch.tensor([[0., 0., -1.]]).repeat(2, 1),
        height_above_ground_m=torch.full((2,), .3), wheel_ground_force_n=torch.ones(2, 2),
        wheel_friction=torch.ones(2, 2), wheel_support_above_environment_m=torch.zeros(2, 2),
        timestamp_s=torch.full((2,), tick * .005, dtype=torch.float64))


def set_encoders(env, tick):
    # The native index order is deliberately different from yaw/pitch order.
    data = env.robot.data
    data.joint_pos[:, 2] = .3 + torch.tensor([.1, -.2]) * tick
    data.joint_pos[:, 0] = -.2 + torch.tensor([.02, -.03]) * tick
    data.joint_vel[:, 2] = torch.tensor([1., -2.]) * tick
    data.joint_vel[:, 0] = torch.tensor([3., -4.]) * tick


def test_gimbal_sensor_instantaneous_has_native_indices_zero_and_controller_sign(gimbal_sensor):
    env, history, _ = gimbal_sensor
    set_encoders(env, 2)
    value = history.instantaneous()
    angles = torch.tensor([-.2, .4])
    expected = torch.stack((angles.sin(), angles.cos(), torch.tensor([.04, -.06]),
                            torch.tensor([-2., 4.]), torch.tensor([6., -8.])), -1)
    torch.testing.assert_close(value, expected)


def test_gimbal_sensor_ring_delay_wrap_and_single_row_reset(gimbal_sensor):
    env, history, measured_type = gimbal_sensor
    sensor = env._sensor_model
    all_rows = torch.ones(2, dtype=torch.bool)
    sensor.reset(all_rows, torch.zeros(2, dtype=torch.float64))
    set_encoders(env, 0)
    sensor.seed_after_reset(all_rows, measured(measured_type, 0))
    snapshots = [history.instantaneous().clone()]
    for tick in range(1, 9):
        set_encoders(env, tick)
        sensor.push(measured(measured_type, tick), tick)
        snapshots.append(history.instantaneous().clone())
    assert sensor.capacity == 4 and sensor.physical_tick == 8
    torch.testing.assert_close(history.ordered(), torch.stack(snapshots[-4:][::-1], 1))
    sensor.lags[:] = torch.tensor([[1, 2, 0, 0], [0, 1, 0, 0]])
    delayed = history.delayed_extra()
    # Shared uplink adds half a physical tick: q reads ages1.5/0.5 and
    # dq ages2.5/1.5. The oldest-to-newest policy history is unrelated.
    expected = torch.stack((
        torch.cat(((snapshots[6][0, 2:3] + snapshots[7][0, 2:3]) / 2,
                   (snapshots[5][0, 3:] + snapshots[6][0, 3:]) / 2)),
        torch.cat(((snapshots[7][1, 2:3] + snapshots[8][1, 2:3]) / 2,
                   (snapshots[6][1, 3:] + snapshots[7][1, 3:]) / 2)),
    ))
    torch.testing.assert_close(delayed, expected)
    continuing = history.ordered()[1].clone()
    sensor(measured(measured_type, 8))  # Cache the actual pre-reset endpoint.
    reset = torch.tensor([True, False])
    sensor.reset(reset, torch.full((2,), .04, dtype=torch.float64))
    env.robot.data.joint_pos[0, [2, 0]] = torch.tensor([1.3, .4])
    env.robot.data.joint_vel[0, [2, 0]] = torch.tensor([7., 8.])
    sensor.seed_after_reset(reset, measured(measured_type, 8))
    sensor.lags[0] = 2
    instant = history.instantaneous()
    torch.testing.assert_close(history.ordered()[0], instant[0].expand(4, 5))
    torch.testing.assert_close(history.ordered()[1], continuing)
    torch.testing.assert_close(history.delayed_extra()[0], instant[0, 2:])
    torch.testing.assert_close(history.delayed_extra()[1], delayed[1])
    assert sensor.physical_tick == 8 and sensor.count.tolist() == [1, 4]


@pytest.fixture
def periodic_adapter():
    from supermimic_rl.course.adapter import CourseAdapter
    from supermimic_rl.compact_state import CompactStateProjector

    n = 2
    names = tuple([f"joint_{i}" for i in range(20)])
    names = names[:3] + ("M_joint1", "M_joint2") + names[5:]
    bodies = ("base_link",) + tuple(f"body_{i}" for i in range(20))
    pose = torch.zeros(n, 21, 7)
    pose[..., 2] = .3
    pose[..., 6] = 1
    q = torch.arange(20).float()[None].repeat(n, 1) * .01
    q[:, 3] = torch.tensor([torch.pi - 1e-4, -torch.pi + 1e-4])
    data = SimpleNamespace(body_link_pose_w=pose, body_com_pose_w=pose.clone(),
        body_com_vel_w=torch.zeros(n, 21, 6), joint_pos=q,
        joint_vel=torch.zeros(n, 20))
    sample = SimpleNamespace(joint_position_rad=q.clone(),
        terrain_contact_force_world_n=torch.zeros(n, 21, 3),
        environment_origin_world_m=torch.zeros(n, 3),
        ground_at_root_world_z_m=torch.zeros(n),
        ground_at_wheel_world_z_m=torch.zeros(n, 2))
    # The source reader may have accumulated turns, while native phase has not.
    sample.joint_position_rad[:, 3] += 4 * torch.pi
    measured = SimpleNamespace(velocity_origin_body_m_s=torch.zeros(n, 3),
                               wheel_ground_force_n=torch.full((n, 2), 3.))
    angles = q[:, [3, 4]].clone()
    controller = SimpleNamespace(config=SimpleNamespace(joint_limits_rad=(None, (-.6, .7))),
        target_angles=angles.clone(), _last_joint=angles.clone(), _last_world=angles.clone(),
        _integral=torch.arange(8).reshape(n, 2, 2).float(),
        _previous_error=torch.tensor([[[3.13, .1], [5., 6.]], [[-3.13, .1], [7., 8.]]]))
    source = {
        "course.gimbal.target": angles.clone(),
        "course.gimbal.direction": angles[:, 0].clone(),
        "gimbal.controller.target_angles": controller.target_angles,
        "gimbal.controller._last_joint": controller._last_joint,
        "gimbal.controller._last_world": controller._last_world,
        "gimbal.controller._integral": controller._integral,
        "gimbal.controller._previous_error": controller._previous_error,
    }
    env = SimpleNamespace(robot=SimpleNamespace(data=data), _sample=lambda: sample,
        _observations=SimpleNamespace(assembler=SimpleNamespace(measure=lambda _: measured)),
        _clock=lambda: torch.zeros(n, dtype=torch.float64), _commands=lambda: None,
        _gimbal=SimpleNamespace(joint_indices=[3, 4], controller=controller),
        _binding=SimpleNamespace(policy_joint_indices=[0, 1, 6, 7, 8, 9]),
        _recovery_preparation=None, _recover_task=None)
    adapter = CourseAdapter.__new__(CourseAdapter)
    adapter.env, adapter.num_envs, adapter.device = env, n, torch.device("cpu")
    adapter.mechanics_backend = "newton"
    adapter.joint_names, adapter.body_names = names, bodies
    adapter.joint_types = tuple("prismatic" if i in (10, 19) else "revolute" for i in range(20))
    adapter._episode_id = torch.tensor([4, 7])
    adapter._control_epoch = torch.tensor([2, 3])
    adapter._static_context = {}
    adapter._projector = CompactStateProjector(bodies, names)
    adapter._actor_frames = lambda *a, **kw: (torch.zeros(n, 45), torch.ones(n, 45))
    adapter._gimbal_sensors = SimpleNamespace(instantaneous=lambda: torch.zeros(n, 5))
    adapter._capture_memory = lambda state, context, clock: state.update(
        {"target.control_source": torch.zeros(n, dtype=torch.long)})
    adapter._capture_course = lambda state, context, clock: state.update(
        {name: value.clone() for name, value in source.items()})
    adapter._field_metadata = lambda state, context: {
        "state": {name: {"units": "source-defined", "frame": "source-defined"} for name in state},
        "context": {name: {"units": "source-defined", "frame": "source-defined"} for name in context}}
    return adapter, source, sample


def test_course_snapshot_encodes_yaw_memory_continuously_without_touching_pid(periodic_adapter):
    adapter, source, sample = periodic_adapter
    before = {name: value.clone() for name, value in source.items()}
    result = adapter._capture_snapshot(terminal=True)
    for name in ("course.gimbal.target", "gimbal.controller.target_angles",
                 "gimbal.controller._last_joint", "gimbal.controller._last_world"):
        assert name not in result.state
        phase = result.state[name + ".yaw_sin_cos"]
        assert (phase[0] - phase[1]).abs().max() < 3e-4
        torch.testing.assert_close(phase.norm(dim=-1), torch.ones(2))
        torch.testing.assert_close(result.state[name + ".pitch_rad"], before[name][:, 1:2])
        reconstructed = torch.atan2(phase[:, 0], phase[:, 1])
        torch.testing.assert_close(reconstructed, before[name][:, 0])
    assert "course.gimbal.direction" not in result.state
    assert (result.state["course.gimbal.direction_sin_cos"][0]
            - result.state["course.gimbal.direction_sin_cos"][1]).abs().max() < 3e-4
    for name in ("gimbal.controller._integral", "gimbal.controller._previous_error"):
        torch.testing.assert_close(result.state[name], before[name])
    # Previous PID errors across the branch remain distinct real numbers.
    assert abs(float(result.state["gimbal.controller._previous_error"][0, 0, 0]
                     - result.state["gimbal.controller._previous_error"][1, 0, 0])) > 6
    assert "joint_position_continuous" not in result.state
    mixed = result.state["joint_position_gimbal_yaw_sine"]
    torch.testing.assert_close(mixed[:, 3], result.state["gimbal.yaw_sin_cos"][:, 0])
    other = [i for i in range(20) if i != 3]
    torch.testing.assert_close(mixed[:, other], sample.joint_position_rad[:, other])
    info = result.field_metadata["state"]["joint_position_gimbal_yaw_sine"]
    units = info["units"].removeprefix("native order: ").split(",")
    assert units[3] == "1" and units[10] == units[19] == "m"
    assert "M_joint1 is sin(native yaw)" in info["frame"]
    for name, value in before.items():
        torch.testing.assert_close(source[name], value)
    assert result.episode_id.tolist() == [4, 7]
    torch.testing.assert_close(result.observation, torch.zeros(2, 45))
    torch.testing.assert_close(result.actor_observation, torch.ones(2, 45))


def test_course_snapshot_rejects_bounded_yaw_before_circular_memory_loss(periodic_adapter):
    adapter, _, _ = periodic_adapter
    adapter.env._gimbal.controller.config.joint_limits_rad = ((-1., 1.), (-.6, .7))
    with pytest.raises(ValueError, match="unlimited yaw"):
        adapter._capture_snapshot(terminal=False)


def test_cold_reset_can_stagger_first_time_limits_reproducibly(monkeypatch):
    from supermimic_rl.adapters.isaac import IsaacAdapter
    from supermimic_rl.course.adapter import CourseAdapter

    monkeypatch.setattr(IsaacAdapter, "reset", lambda self: "snapshot")

    def adapter(desynchronize):
        result = object.__new__(CourseAdapter)
        result.num_envs, result._desynchronize = 512, desynchronize
        result._desync_generator = torch.Generator().manual_seed(7)
        result.env = SimpleNamespace(max_episode_length=1500,
                                     episode_length_buf=torch.zeros(512, dtype=torch.long))
        return result

    first, second, plain = adapter(True), adapter(True), adapter(False)
    assert first.reset() == "snapshot" and plain.reset() == "snapshot"
    offsets = first.env.episode_length_buf
    assert offsets.min() >= 0 and offsets.max() < 1500 and offsets.unique().numel() > 300
    second.reset()
    assert torch.equal(offsets, second.env.episode_length_buf)
    assert not plain.env.episode_length_buf.any()


def test_staggered_episodes_do_not_change_the_world_model_schema_source():
    from supermimic_rl.cli import _native_schema_source

    base = {"upstream_commit": "x", "curriculum": {"a": 1}}
    assert _native_schema_source({**base, "desynchronize_episodes": True}) == _native_schema_source(base)
