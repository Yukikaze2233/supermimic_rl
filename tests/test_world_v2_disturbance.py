"""Disturbance replay matches the native clock, draw order and COM convention."""
from dataclasses import replace

import pytest
import torch

from supermimic_rl.adapters.disturbances import DisturbanceRecorder
from supermimic_rl.adapters.substeps import SubstepRecorder
from supermimic_rl.course.randomization import CourseRandomization, RequestedRandomizationConfig
from supermimic_rl.world_model_v2.disturbance import DisturbanceState, apply_velocity_push
from supermimic_rl.world_model_v2.physical import PhysicalState
from supermimic_rl.world_model_v2 import so3
from test_substeps import FakeNative


def owner(n=8, *, enabled=True):
    config = RequestedRandomizationConfig(
        external_force_n=15. if enabled else 0., external_torque_nm=1.5 if enabled else 0.,
        push_linear_m_s=.5 if enabled else 0., push_yaw_rad_s=1. if enabled else 0.)
    result = CourseRandomization(n, seed=7, config=config)
    result.reset(torch.ones(n, dtype=torch.bool), config.full_at_coordinate)
    return result


def captured(native):
    zero = torch.zeros_like(native.disturbance_scale)
    return DisturbanceState(
        native.push_timer_s.clone() if native.config.pushes else zero,
        native.wrench_timer_s.clone() if native.config.wrenches else zero,
        native.base_wrench.clone(), native.disturbance_scale.clone())


@pytest.mark.parametrize('enabled', [False, True])
def test_scheduler_matches_native_rng_including_deferred_and_nominal_rows(enabled):
    native = owner(enabled=enabled)
    if enabled:
        native.push_timer_s[:] = torch.tensor([.01, .02, .03, -2., .04, .015, .005, .5])
        native.wrench_timer_s[:] = torch.tensor([.002, .007, .012, .017, .5, .008, .003, .02])
    state = captured(native)
    generator = torch.Generator().set_state(native.generator.get_state())
    for step in range(12):
        eligible = torch.arange(8) % 3 != step % 3
        delta, wrenches, state = state.advance(native.config, eligible, generator)
        actual_delta = torch.zeros(8, 3)
        if enabled:
            rows, draws = native.push(.02, eligible)
            actual_delta[rows] = draws
        actual_wrench = torch.stack([native.advance_wrench(.005).clone() for _ in range(4)], 1)
        torch.testing.assert_close(delta, actual_delta, rtol=0, atol=0)
        torch.testing.assert_close(wrenches, actual_wrench, rtol=0, atol=0)
        torch.testing.assert_close(state.encode(), captured(native).encode(), rtol=0, atol=0)
        torch.testing.assert_close(generator.get_state(), native.generator.get_state(), rtol=0, atol=0)
        assert delta[native.nominal].count_nonzero() == 0
        assert wrenches[native.nominal].count_nonzero() == 0
    torch.testing.assert_close(DisturbanceState.decode(state.encode()).encode(), state.encode())


def test_recording_keeps_all_four_wrenches_and_does_not_change_random_draws():
    env = FakeNative()
    native, control = owner(2), owner(2)
    env._requested_randomization = env._episode_randomization = native
    for item in (native, control):
        item.push_timer_s[:] = .001
        item.wrench_timer_s[:] = torch.tensor([.007, .012])
    recorder = SubstepRecorder(env, disturbances=True).install()
    recorder.begin()
    eligible = torch.tensor([True, False])
    rows, delta = native.push(.02, eligible)
    expected_rows, expected_delta = control.push(.02, eligible)
    torch.testing.assert_close(rows, expected_rows)
    torch.testing.assert_close(delta, expected_delta, rtol=0, atol=0)
    expected_wrench = []
    for _ in range(4):
        native.advance_wrench(.005)
        expected_wrench.append(control.advance_wrench(.005).clone())
        env.step(ticks=1)
    values, _ = recorder.finish()
    expected_push = torch.zeros(2, 3)
    expected_push[rows] = delta
    torch.testing.assert_close(values['velocity_push_world'], expected_push[:, None].expand(-1, 4, -1))
    torch.testing.assert_close(values['external_wrench_body'], torch.stack(expected_wrench, 1), rtol=0, atol=0)
    torch.testing.assert_close(native.generator.get_state(), control.generator.get_state(), rtol=0, atol=0)
    assert not torch.equal(values['external_wrench_body'][:, 0], values['external_wrench_body'][:, -1])
    # A following interval without a push must not repeat the previous impulse.
    recorder.begin()
    env.step()
    next_values, _ = recorder.finish()
    assert next_values['velocity_push_world'].count_nonzero() == 0
    recorder.close()
    assert 'push' not in vars(native) and '_disturbance_recorder' not in vars(native)


def test_disabled_snapshot_is_finite_and_legacy_recording_is_unchanged():
    env = FakeNative()
    assert all('wrench' not in field.name and 'push' not in field.name for field in SubstepRecorder(env).fields)
    env._requested_randomization = env._episode_randomization = owner(2, enabled=False)
    recorder = DisturbanceRecorder(env)
    assert all(torch.isfinite(value).all() for value in recorder.snapshot().values())
    assert torch.isinf(recorder.owner.push_timer_s).all()
    assert torch.isinf(recorder.owner.wrench_timer_s).all()
    env._episode_randomization = object()
    with pytest.raises(ValueError, match='owner'):
        DisturbanceRecorder(env)


def test_velocity_push_preserves_com_velocity_convention_and_gradients():
    generator = torch.Generator().manual_seed(21)
    rand = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.float64)
    rotation_vector = rand(5, 3).requires_grad_()
    com_local = rand(5, 3).requires_grad_()
    delta = rand(5, 3).requires_grad_()
    state = PhysicalState(rand(5, 3), so3.exp(rotation_vector), rand(5, 3), rand(5, 3), rand(5, 6), rand(5, 8))
    result = apply_velocity_push(state, delta, com_local)
    offset = (state.rotation @ com_local[..., None]).squeeze(-1)
    before_com = state.velocity + torch.linalg.cross(state.angular_velocity, offset)
    after_com = result.velocity + torch.linalg.cross(result.angular_velocity, offset)
    torch.testing.assert_close(after_com - before_com, torch.cat((delta[:, :2], torch.zeros(5, 1)), -1))
    torch.testing.assert_close(result.angular_velocity - state.angular_velocity,
                               torch.cat((torch.zeros(5, 2), delta[:, 2:]), -1))
    for name in ('position', 'rotation', 'joint_position', 'joint_velocity'):
        assert getattr(result, name) is getattr(state, name)

    def pushed(rotvec, local, increment):
        return apply_velocity_push(replace(state, rotation=so3.exp(rotvec)), increment, local).velocity

    assert torch.autograd.gradcheck(pushed, (rotation_vector, com_local, delta))
    zero = apply_velocity_push(state, torch.zeros_like(delta), com_local)
    torch.testing.assert_close(zero.velocity, state.velocity, rtol=0, atol=0)
