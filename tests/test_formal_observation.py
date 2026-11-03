"""Numerical causal queues, gimbal signs and exogenous horizon boundaries."""
import math
import os
from pathlib import Path
import sys

import pytest
import torch

_UPSTREAM = Path(os.environ.get("SUPERMIMIC_UPSTREAM_ROOT",
    Path(__file__).resolve().parents[2] / "transformer_rl-worktrees/full-vehicle-gimbal"))
if (_UPSTREAM / "src").is_dir():
    sys.path.insert(0, str(_UPSTREAM / "src"))

from supermimic_rl.contracts import Field, StateSchema
from supermimic_rl.compact_state import named_field
from supermimic_rl.formal_observation import (
    advance_conditions, imagination_horizon, observe, observe_clean,
)


def fixture():
    n, capacity = 2, 4
    values = torch.arange(n * capacity * 18).reshape(n, capacity, 18).float() * .001
    yaw = torch.tensor([[math.pi - .2, -math.pi + .2, .4, .6], [.1, .2, .3, .4]])
    gimbal = torch.stack((yaw.sin(), yaw.cos(), yaw / 10, yaw * 2, yaw * 3), -1)
    state = {
        "root_orientation_xyzw": torch.tensor([[0., 0., 0., 1.]]).repeat(n, 1),
        "root_angular_velocity_body_rad_s": torch.ones(n, 3),
        "joint_position_continuous": torch.full((n, 6), .5),
        "joint_velocity_native": torch.full((n, 6), 2.),
        "control._previous_action": torch.zeros(n, 6),
        "gimbal.yaw_sin_cos": torch.tensor([[.6, .8]]).repeat(n, 1),
        "gimbal.pitch_position_rad": torch.full((n, 1), .3),
        "gimbal.velocity_rad_s": torch.tensor([[.4, .5]]).repeat(n, 1),
        "sensor.values": values,
        "sensor.count": torch.tensor([3., 1.]),
        "sensor.gimbal_history": gimbal,
        "course.imagination_remaining_s": torch.tensor([[.1], [.02]]),
        "course.dynamic_goal": torch.zeros(n),
    }
    context = {
        "controller.nominal": torch.zeros(n, 6),
        "policy_joint_indices": torch.arange(6).float().repeat(n, 1),
        "gimbal.control_joint_signs": torch.tensor([[1., -1.]]).repeat(n, 1),
        "gimbal.joint_zero_rad": torch.zeros(n, 2),
        "sensor.lags": torch.tensor([[1., 2., 0., 1.]]).repeat(n, 1),
        "sensor.nominal": torch.tensor([0., 1.]),
        "sensor.noise_scale": torch.full((n,), .5),
        "sensor.noise_halfwidth": torch.full((n, 18), .01),
        "sensor.noise_distribution": torch.zeros(n),
        "physics_dt_s": torch.full((n, 1), .005),
        "sensor.uplink_delay_s": torch.full((n,), .0025),
    }
    context["sensor.noise_halfwidth"][:, 4:6] = 0
    schema = StateSchema(tuple(Field(k, tuple(v.shape[1:])) for k, v in state.items()),
                         tuple(Field(k, tuple(v.shape[1:])) for k, v in context.items()),
                         observation_dim=45, goal_dim=16)
    goal = torch.zeros(n, 16)
    goal[:, 4] = 1
    goal[:, 11:] = torch.tensor([0., 1., .2, .3, .4])
    return schema, schema.pack(state), schema.pack(context, context=True), goal, state, context


def test_delayed_groups_fractional_age_and_reset_do_not_read_current_physics():
    schema, state, context, goal, raw, _ = fixture()
    value = observe(state, goal, context, None, schema, noise=False)
    history = raw["sensor.values"]
    # q/gravity read age1.5, velocity clamps to oldest available age2,
    # gyro reads age0.5. Reset row has one real sample, irrespective of lag.
    torch.testing.assert_close(value[0, 10:14], history[0, 1:3, :4].mean(0))
    torch.testing.assert_close(value[0, 16:22], history[0, 2, 6:12] * .1)
    torch.testing.assert_close(value[0, 4:7], history[0, :2, 12:15].mean(0) * .5)
    torch.testing.assert_close(value[1, 10:14], history[1, 0, :4])
    torch.testing.assert_close(value[:, 35:40], goal[:, 11:16])
    phase = raw["sensor.gimbal_history"][0, 1:3, :2].mean(0)
    torch.testing.assert_close(value[0, 40:42], phase / phase.norm())
    torch.testing.assert_close(value[0, 42], raw["sensor.gimbal_history"][0, 1:3, 2].mean())
    torch.testing.assert_close(value[0, 43:], raw["sensor.gimbal_history"][0, 2, 3:])
    clean = observe_clean(state, goal, context, None, schema)
    torch.testing.assert_close(clean[:, 10:14], torch.full((2, 4), .5))
    torch.testing.assert_close(clean[:, 40:], torch.tensor([[.6, .8, -.3, .4, -.5]]).repeat(2, 1))
    assert not torch.equal(clean, value)


def test_source_uniform_noise_is_after_delay_and_nominal_rows_are_unchanged():
    schema, state, context, goal, _, _ = fixture()
    torch.manual_seed(123)
    raw_noise = (torch.rand(2, 18) * 2 - 1) * .01 * .5
    torch.manual_seed(123)
    noisy = observe(state, goal, context, None, schema)
    mean = observe(state, goal, context, None, schema, noise=False)
    torch.testing.assert_close(noisy[0, 4:7] - mean[0, 4:7], raw_noise[0, 12:15] * .5)
    torch.testing.assert_close(noisy[0, 10:14] - mean[0, 10:14], raw_noise[0, :4])
    torch.testing.assert_close(noisy[1], mean[1])
    torch.testing.assert_close(noisy[:, 35:], mean[:, 35:])


def test_requested_gaussian_sigma_and_gravity_normalization_are_explicit():
    schema, state, context, goal, _, _ = fixture()
    named_field(context, schema, "sensor.noise_distribution", context=True).fill_(1)
    torch.manual_seed(321)
    noise = torch.randn(2, 18) * .01 * .5
    torch.manual_seed(321)
    noisy = observe(state, goal, context, None, schema)
    mean = observe(state, goal, context, None, schema, noise=False)
    torch.testing.assert_close(noisy[0, 10:14] - mean[0, 10:14], noise[0, :4])
    torch.testing.assert_close(noisy[0, 4:7] - mean[0, 4:7], noise[0, 12:15] * .5)
    torch.testing.assert_close(noisy[:, 7:10].norm(dim=-1), torch.ones(2))
    torch.testing.assert_close(noisy[1], mean[1])


def test_clean_gimbal_uses_controller_zero_before_axis_sign():
    schema, state, context, goal, _, _ = fixture()
    named_field(context, schema, "gimbal.joint_zero_rad", context=True)[:] = torch.tensor([.2, -.1])
    named_field(context, schema, "gimbal.control_joint_signs", context=True)[:] = torch.tensor([-1., -1.])
    yaw = -(math.atan2(.6, .8) - .2)
    value = observe_clean(state, goal, context, None, schema)
    torch.testing.assert_close(value[:, 40:], torch.tensor(
        [[math.sin(yaw), math.cos(yaw), -.4, -.4, -.5]]).repeat(2, 1))


def test_observer_gradients_flow_through_selected_queue_and_issued_action():
    schema, state, context, goal, _, _ = fixture()
    state.requires_grad_()
    action = torch.zeros(2, 6, requires_grad=True)
    observe(state, goal, context, action, schema, noise=False).sum().backward()
    assert named_field(state.grad, schema, "sensor.values").abs().sum() > 0
    assert named_field(state.grad, schema, "sensor.gimbal_history").abs().sum() > 0
    assert named_field(state.grad, schema, "joint_position_continuous").eq(0).all()
    assert named_field(state.grad, schema, "root_angular_velocity_body_rad_s").eq(0).all()
    torch.testing.assert_close(action.grad, torch.ones_like(action))


def test_deadlines_are_real_start_bounds_and_never_rewrite_task_mode():
    schema, state, context, goal, _, _ = fixture()
    torch.testing.assert_close(imagination_horizon(state, goal, context, schema), torch.tensor([4, 0]))
    _, ordinary, _ = advance_conditions(state, goal, context, torch.zeros(2, 6), schema)
    assert ordinary[:, 10].eq(0).all()
    jump = goal.clone()
    jump[:, 4] = 0
    jump[:, 8] = 1
    torch.testing.assert_close(imagination_horizon(state, jump, context, schema), torch.ones(2, dtype=torch.long))
    following, command, conditions = advance_conditions(state, jump, context, torch.zeros(2, 6), schema)
    assert following is state and conditions is context
    torch.testing.assert_close(command[:, :10], jump[:, :10])
    torch.testing.assert_close(command[:, 11:], jump[:, 11:])
    torch.testing.assert_close(command[:, 10], torch.full((2,), .02))
    named_field(state, schema, "course.dynamic_goal")[0] = 1
    torch.testing.assert_close(imagination_horizon(state, goal, context, schema), torch.tensor([1, 0]))


def test_delayed_values_match_pinned_sensor_model_cpu():
    sensors = pytest.importorskip("transformer_rl.formal.control.sensors")
    schema, state, context, goal, raw, settings = fixture()
    config = sensors.SensorModelConfig(
        physics_dt_s=.005, policy_dt_s=.02, observation_delays_s=(0., .005, .01),
        leg_position_halfwidth_rad=.01, leg_velocity_halfwidth_rad_s=.01,
        wheel_velocity_halfwidth_rad_s=.01, gyro_halfwidth_rad_s=.01, gravity_halfwidth=.01)
    native = sensors.SensorModel(2, 123, config, uplink_delay_s=torch.full((2,), .0025, dtype=torch.float64),
                                  maximum_uplink_s=.0025)
    native.physical_tick = 4
    native.count.copy_(torch.tensor([3, 1]))
    native.lags.copy_(settings["sensor.lags"].long())
    native.nominal.copy_(torch.tensor([False, True]))
    native.noise_scale.fill_(.5)
    for age in range(4):
        native.values[(4 - age) % 4] = raw["sensor.values"][:, age]
        native.times[(4 - age) % 4] = (4 - age) * .005
    sensor = native._read_rows(torch.arange(2), native.generator)
    torch.manual_seed(123)
    value = observe(state, goal, context, None, schema)
    torch.testing.assert_close(value[:, 10:14], sensor.policy_position_rad[:, :4])
    torch.testing.assert_close(value[:, 16:22], sensor.policy_velocity_rad_s * .1)
    torch.testing.assert_close(value[:, 4:7], sensor.angular_velocity_body_rad_s * .5)
    torch.testing.assert_close(value[:, 7:10], sensor.gravity_body)


def test_requested_gaussian_observer_matches_native_sensor_wrapper_cpu():
    sensors = pytest.importorskip("transformer_rl.formal.control.sensors")
    base = pytest.importorskip("transformer_rl.formal.lab_env.sensors").LabSensorModel
    from supermimic_rl.course.randomization import CourseRandomization, make_sensor_model_type

    schema, state, context, goal, raw, settings = fixture()
    native_type = make_sensor_model_type(base, CourseRandomization(2))
    native = native_type(2, 123, sensors.SensorModelConfig(
        physics_dt_s=.005, policy_dt_s=.02, observation_delays_s=(0., .005, .01)))
    native.physical_tick = 4
    native.count.copy_(torch.tensor([3, 1]))
    native.lags.copy_(settings["sensor.lags"].long())
    native.nominal.copy_(torch.tensor([False, True]))
    native.noise_scale.fill_(.5)
    for age in range(native.capacity):
        native.values[(4 - age) % native.capacity] = raw["sensor.values"][:, age]
        native.times[(4 - age) % native.capacity] = (4 - age) * .005
    named_field(context, schema, "sensor.noise_distribution", context=True).fill_(1)
    named_field(context, schema, "sensor.noise_halfwidth", context=True).copy_(native._halfwidth)
    named_field(context, schema, "sensor.uplink_delay_s", context=True).zero_()
    sensor = native._read_rows(torch.arange(2), native.generator)
    torch.manual_seed(123)
    value = observe(state, goal, context, None, schema)
    torch.testing.assert_close(value[:, 10:14], sensor.policy_position_rad[:, :4])
    torch.testing.assert_close(value[:, 16:22], sensor.policy_velocity_rad_s * .1)
    torch.testing.assert_close(value[:, 4:7], sensor.angular_velocity_body_rad_s * .5)
    torch.testing.assert_close(value[:, 7:10], sensor.gravity_body)


def test_clean_observer_phase_joint_codec_preserves_non_gimbal_policy_indices():
    _, _, _, goal, state, context = fixture()
    # Independent yaw is interleaved among the native joints, never a policy axis.
    mapping = torch.tensor([0, 2, 4, 5, 7, 9])
    q, dq = torch.randn(2, 20), torch.randn(2, 20)
    state["joint_position_continuous"] = q
    state["joint_velocity_native"] = dq
    context["policy_joint_indices"] = mapping.float()[None].repeat(2, 1)
    original = StateSchema(tuple(Field(k, tuple(v.shape[1:])) for k, v in state.items()),
        tuple(Field(k, tuple(v.shape[1:])) for k, v in context.items()),
        observation_dim=45, goal_dim=16)
    expected = observe_clean(original.pack(state), goal, original.pack(context, context=True), None, original)
    changed = state.pop("joint_position_continuous").clone()
    changed[:, 3] = torch.tensor([torch.pi - 1e-4, -torch.pi + 1e-4]).sin()
    state["joint_position_gimbal_yaw_sine"] = changed
    updated = StateSchema(tuple(Field(k, tuple(v.shape[1:])) for k, v in state.items()),
        original.context_fields, observation_dim=45, goal_dim=16)
    actual = observe_clean(updated.pack(state), goal, updated.pack(context, context=True), None, updated)
    torch.testing.assert_close(actual, expected)
