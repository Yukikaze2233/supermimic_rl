"""Full sensor/request/reward/critic time chain inside a differentiable rollout.

The motion and task timeline below are fixtures. They test integration and
derivatives, not dynamics accuracy or the robot's ability to climb a step.
"""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from supermimic_rl.world_model_v2 import so3
from supermimic_rl.world_model_v2.imagination import InvalidImagination
from supermimic_rl.world_model_v2.learning import frozen_parameters
from supermimic_rl.world_model_v2.motion_task import MotionTask
from supermimic_rl.world_model_v2.stair_source import StairSourceState, TOF_FEATURE_DIM
from supermimic_rl.world_model_v2.stair_task import StairTaskState, STAIR_FEATURE_DIM
from support.world_v2.stair_imagination import fixture, move_to_scan_boundary


def test_stair_rollout_refreshes_goal_contact_permission_and_critic_state():
    learner, start, frames = fixture()
    original = {key: value.clone() for key, value in start.items()}
    with frozen_parameters(learner.model, learner.critic):
        rollout = learner.imagination.rollout(start, frames, 16)
        assert rollout.critic_inputs.shape == (17, 2, 369)
        assert not rollout.invalid.any()
        assert torch.isfinite(rollout.objective).all()
        # Base feature order: 61 motion/memory, then 16 goal; conditions at 222.
        assert (rollout.critic_inputs[1:, :, 64] > start['goal'][:, 3]).any()
        assert rollout.critic_inputs[0, :, 223].max() == 0
        assert rollout.critic_inputs[1:, :, 223].max() == 1
        gradient, = torch.autograd.grad(rollout.objective.mean(), learner.actor.decoder[-1].bias)
        assert torch.isfinite(gradient).all() and gradient.norm() > 0
        parameter = learner.actor.decoder[-1].bias
        baseline, direction = parameter.detach().clone(), gradient / gradient.norm()
        epsilon = .001
        with torch.no_grad():
            parameter.copy_(baseline + epsilon * direction)
            positive = learner.imagination.rollout(start, frames, 16)
            parameter.copy_(baseline - epsilon * direction)
            negative = learner.imagination.rollout(start, frames, 16)
            parameter.copy_(baseline)
        assert torch.equal(positive.critic_inputs[:, :, 222:224], negative.critic_inputs[:, :, 222:224])
        numeric = (positive.objective.mean() - negative.objective.mean()) / (2 * epsilon)
        analytic = (gradient * direction).sum()
        assert (numeric - analytic).abs() / analytic.abs().clamp_min(1e-8) < .002
    for key, value in start.items():
        assert torch.equal(value, original[key]), key


def test_same_source_realization_keeps_actor_microbatch_gradient_and_restore():
    learner, start, frames = fixture(sensor_noise=True)
    other, _, _ = fixture(sensor_noise=True)
    other.load_state_dict(deepcopy(learner.state_dict()))
    torch.manual_seed(19)
    full = learner.update_policy(start, frames, microbatch_size=2)
    torch.manual_seed(19)
    split = other.update_policy(start, frames, microbatch_size=1)
    assert full['actor/updated'] and split['actor/updated']
    for a, b in zip(learner.actor.decoder.parameters(), other.actor.decoder.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=2e-6)
    restored, _, _ = fixture()
    restored.load_state_dict(deepcopy(learner.state_dict()))
    assert restored.update_policy(start, frames)['actor/updated']
    assert restored.skipped == dict.fromkeys(restored.skipped, 0)


def test_one_invalid_trajectory_does_not_poison_the_valid_actor_gradient():
    learner, start, frames = fixture(sensor_noise=True)
    other, _, _ = fixture(sensor_noise=True)
    other.load_state_dict(deepcopy(learner.state_dict()))
    move_to_scan_boundary(learner, start, [0])
    with frozen_parameters(learner.model, learner.critic):
        with pytest.raises(InvalidImagination) as captured:
            learner.imagination.rollout(start, frames, 16)
        assert captured.value.invalid_rows.tolist() == [True, False]
        assert captured.value.reason == 'terrain_query'
        isolated, rejected, replayed, causes = learner._usable_rollout(start, frames)
        assert rejected == 1 and replayed == 0
        assert causes['terrain_query'] == 1
        gradient = torch.autograd.grad(isolated.objective.sum() / 2, learner.actor.decoder[-1].bias)[0]
        single = learner.imagination.rollout({key: value[1:] for key, value in start.items()}, frames[1:], 16)
        expected = torch.autograd.grad(single.objective.sum() / 2, learner.actor.decoder[-1].bias)[0]
        torch.testing.assert_close(gradient, expected, rtol=0, atol=1e-7)
        assert torch.isfinite(gradient).all() and gradient.norm() > 0
    torch.manual_seed(19)
    full = learner.update_policy(start, frames, microbatch_size=2)
    torch.manual_seed(19)
    split = other.update_policy(start, frames, microbatch_size=1)
    for record in (full, split):
        assert record['actor/updated']
        assert record['imagination/accepted_trajectories'] == 1
        assert record['imagination/invalid_fraction'] == .5
    assert full['imagination/replayed_trajectories'] == 0
    assert split['imagination/replayed_trajectories'] == 0
    for a, b in zip(learner.actor.decoder.parameters(), other.actor.decoder.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=2e-6)


def test_no_usable_trajectories_skips_actor_and_critic_without_an_empty_batch_crash():
    learner, start, frames = fixture(sensor_noise=True)
    move_to_scan_boundary(learner, start, [0, 1])
    parameters = [value.detach().clone() for value in learner.actor.decoder.parameters()]
    record = learner.update_policy(start, frames, microbatch_size=2)
    assert not record['actor/updated']
    assert record['imagination/invalid_fraction'] == 1.
    assert record['imagination/accepted_trajectories'] == 0
    assert learner.updates['actor'] == learner.updates['critic'] == 0
    assert learner.skipped['actor'] == 1
    assert not learner.optimizers['actor'].state and not learner.optimizers['critic'].state
    for a, b in zip(parameters, learner.actor.decoder.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_screening_continues_survivors_once_across_different_failure_steps(monkeypatch):
    from supermimic_rl.world_model_v2.records import decode_physical

    learner, start, frames = fixture(sensor_noise=True)
    rows = torch.tensor([0, 0, 0, 1])
    start, frames = {key: value[rows] for key, value in start.items()}, frames[rows]
    start['physical'][:3, 0] = torch.tensor([4.45, 4.15, 3.95])
    observed = learner.model.terrain.observe(decode_physical(start['physical']), start['terrain_tile'][:, 0])
    start['height'], start['scan'] = observed.height_above_ground, observed.scan_relative
    calls, predict = [], learner.model.step

    def count(state, *args, **kwargs):
        calls.append(len(state.position))
        return predict(state, *args, **kwargs)

    monkeypatch.setattr(learner.model, 'step', count)
    with pytest.raises(ValueError, match='no_grad'):
        learner.imagination.rollout(start, frames, 16, screen_invalid=True)
    with torch.no_grad():
        screened = learner.imagination.rollout(start, frames, 16, screen_invalid=True)
    assert len(calls) == 16 and calls[0] == 4 and calls[-1] == 1
    assert len(set(calls)) == 4
    assert screened.usable.tolist() == [False, False, False, True]
    assert screened.rejection_reasons['terrain_query'].tolist() == [True, True, True, False]
    calls.clear()
    with frozen_parameters(learner.model, learner.critic):
        isolated, rejected, replayed, causes = learner._usable_rollout(start, frames)
        assert len(calls) == 16 and calls[0] == 4 and calls[-1] == 1
        assert len(set(calls)) == 4
        assert isolated.source_rows.tolist() == [3]
        assert rejected == 3 and replayed == 0 and causes['terrain_query'] == 3
        single = learner.imagination.rollout({key: value[3:] for key, value in start.items()}, frames[3:], 16)
        torch.testing.assert_close(screened.objective[3:], single.objective, rtol=0, atol=1e-6)
        torch.testing.assert_close(isolated.objective, single.objective, rtol=0, atol=1e-6)
        from supermimic_rl.world_model_v2.learning import lambda_returns
        torch.testing.assert_close(lambda_returns(isolated, .99, .95),
                                   lambda_returns(single, .99, .95), rtol=0, atol=1e-6)
        torch.testing.assert_close(isolated.critic_inputs, single.critic_inputs, rtol=0, atol=1e-6)
        assert torch.equal(isolated.active, single.active)
        actual = torch.autograd.grad(isolated.objective.sum(), learner.actor.decoder[-1].bias)[0]
        expected = torch.autograd.grad(single.objective.sum(), learner.actor.decoder[-1].bias)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)


def test_screening_rejects_all_rows_without_advancing_the_request_source(monkeypatch):
    learner, start, frames = fixture()
    move_to_scan_boundary(learner, start, [0, 1])
    predict = learner.model.step

    def undefined(*args, **kwargs):
        result = predict(*args, **kwargs)
        return replace(result, relative_ticks=result.relative_ticks * float('nan'))

    def forbidden(*args, **kwargs):
        raise AssertionError('An invalid trajectory reached the request protocol')

    monkeypatch.setattr(learner.model, 'step', undefined)
    monkeypatch.setattr(learner.imagination.stair_source, 'advance', forbidden)
    with torch.no_grad():
        screened = learner.imagination.rollout(start, frames, 16, screen_invalid=True)
    assert not screened.usable.any() and screened.rejection_reasons['nonfinite_prediction'].all()
    assert not screened.active.any()


@pytest.mark.parametrize('reason', ['nonfinite_prediction', 'reward_reference'])
def test_screening_preserves_good_rows_when_another_motion_or_reference_fails(monkeypatch, reason):
    from supermimic_rl.world_model_v2.imagined_reward import LocomotionRewardContext
    from supermimic_rl.world_model_v2.records import decode_physical

    learner, start, frames = fixture(sensor_noise=True)
    start['physical'][:, 1] = torch.tensor([1., -1.])
    observed = learner.model.terrain.observe(decode_physical(start['physical']), start['terrain_tile'][:, 0])
    start['height'], start['scan'] = observed.height_above_ground, observed.scan_relative

    def bad_rows(state):
        return (state.position[:, 1] > 0) & (state.position[:, 0] > -.06)

    if reason == 'nonfinite_prediction':
        predict = learner.model.step

        def undefined(state, *args, **kwargs):
            result = predict(state, *args, **kwargs)
            physical = result.ticks.state
            physical = replace(physical, position=torch.where(bad_rows(state)[:, None, None],
                torch.full_like(physical.position, float('nan')), physical.position))
            return replace(result, state=physical.tick(-1), ticks=replace(result.ticks, state=physical))

        monkeypatch.setattr(learner.model, 'step', undefined)
    else:
        integrate = LocomotionRewardContext.integrate

        def undefined(context, state, *args, **kwargs):
            result = integrate(context, state, *args, **kwargs)
            return replace(result, invalid=result.invalid | bad_rows(state))

        monkeypatch.setattr(LocomotionRewardContext, 'integrate', undefined)
    with torch.no_grad():
        screened = learner.imagination.rollout(start, frames, 16, screen_invalid=True)
    assert screened.usable.tolist() == [False, True]
    assert screened.rejection_reasons[reason].tolist() == [True, False]
    with frozen_parameters(learner.model, learner.critic):
        isolated, rejected, _, causes = learner._usable_rollout(start, frames)
        assert rejected == 1 and causes[reason] == 1
        single = learner.imagination.rollout({key: value[1:] for key, value in start.items()}, frames[1:], 16)
        actual = torch.autograd.grad(isolated.objective.sum(), learner.actor.decoder[-1].bias)[0]
        expected = torch.autograd.grad(single.objective.sum(), learner.actor.decoder[-1].bias)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)


def test_source_features_remove_global_xy_yaw_clock_and_request_identity():
    learner, record, _ = fixture()
    from supermimic_rl.world_model_v2.records import decode_physical
    physical = decode_physical(record['physical'])
    stair, source = StairTaskState.decode(record), StairSourceState.decode(record)
    motion = MotionTask.decode(record['motion_task'])
    ids = stair.ids.clone(); ids[:, 1] = 5
    stair = replace(stair, ids=ids)
    ids = source.ids.clone(); ids[:, 1:4] = torch.tensor([2, 1, 5])
    source = replace(source, ids=ids, flags=torch.ones_like(source.flags))
    first = torch.cat((stair.critic_features(), source.critic_features(physical, stair, motion)), -1)
    assert first.shape == (2, STAIR_FEATURE_DIM + TOF_FEATURE_DIM) and torch.isfinite(first).all()
    angle, offset = .7, torch.tensor([120., -33.])
    rotation = so3.exp(torch.tensor([[0., 0., angle]])).expand(2, -1, -1)
    position = (rotation @ physical.position[..., None]).squeeze(-1)
    position[:, :2] += offset
    physical = replace(physical, position=position, rotation=rotation @ physical.rotation)
    values = source.values.clone()
    for begin in (2, 4, 6, 8, 10):
        xy = (rotation[:, :2, :2] @ values[:, begin:begin + 2, None]).squeeze(-1)
        values[:, begin:begin + 2] = xy + (offset if begin in (2, 6, 8) else 0.)
    moved_ids = source.ids.clone()
    moved_ids[:, [0, 3]] += 2**54
    source = replace(source, values=values, ids=moved_ids)
    ids = stair.ids.clone(); ids[:, :2] += 2**54
    stair = replace(stair, ids=ids, clock=stair.clock + 1_000_000.)
    second = torch.cat((stair.critic_features(), source.critic_features(physical, stair, motion)), -1)
    torch.testing.assert_close(first, second, rtol=0, atol=2e-5)


def test_expired_segment_stays_known_until_request_release_then_bootstraps(monkeypatch):
    from supermimic_rl.world_model_v2.learning import lambda_returns

    learner, start, frames = fixture()
    predicted_batch_sizes = []
    original_step = learner.model.step

    def predict_active(state, *args, **kwargs):
        predicted_batch_sizes.append(len(state.position))
        return original_step(state, *args, **kwargs)

    monkeypatch.setattr(learner.model, 'step', predict_active)
    task = MotionTask.decode(start['motion_task'])
    task.route[:, 4] = 1.  # Managed segments and gimbal share one resampling boundary.
    task.timing[:, :2] = torch.tensor([[.18, .18], [0., 0.]])
    start['motion_task'] = task.encode()
    start['tof_segment_end'] = torch.full((2, 1), .04)
    start['horizon'][:, 0] = 0.  # Legacy remaining would exclude both rows.
    start['stair_ids'][:, 1:3] = torch.tensor([7, 3])
    start['stair_clock'][:, 0] = start['stair_clock'][:, 2] = torch.tensor([.18, 0.], dtype=torch.float64)
    start['stair_clock'][:, 1] = 0.
    start['stair_values'][:, 0] = .20
    start['failure_conditions'][:, 1] = 1.
    rollout = learner.imagination.rollout(start, frames, 16)
    assert rollout.lengths.tolist() == [1, 10]
    assert rollout.active.sum(0).tolist() == [1, 10]
    assert rollout.boundary.sum(0).tolist() == [1, 1]
    assert predicted_batch_sizes == [2] + [1] * 9
    torch.testing.assert_close(rollout.critic_inputs[1:, 0],
                               rollout.critic_inputs[1, 0].expand_as(rollout.critic_inputs[1:, 0]))
    expected = rollout.rewards[0, 0] + .99 * rollout.continuation[0, 0] * rollout.values[1, 0]
    torch.testing.assert_close(lambda_returns(rollout, .99, .95)[0, 0], expected.detach())


def test_unblocked_unknown_command_boundary_is_not_invented():
    import pytest

    learner, start, frames = fixture()
    task = MotionTask.decode(start['motion_task'])
    task.route[:, 4] = 1.
    start['motion_task'] = task.encode()
    start['tof_segment_end'] = torch.full((2, 1), .02)
    with pytest.raises(ValueError, match='known complete control interval'):
        learner.imagination.rollout(start, frames, 16)


def test_terrain_removal_precedes_reward_and_future_motion_inputs(monkeypatch):
    from supermimic_rl.world_model_v2.imagined_reward import LocomotionRewardContext

    learner, start, frames = fixture(sensor_noise=True)
    move_to_scan_boundary(learner, start, [0])
    original = {name: value.clone() for name, value in start.items()}
    predict, integrate = learner.model.step, LocomotionRewardContext.integrate
    motion_sizes, reward_sizes = [], []

    def checked_motion(state, height, scan, *args, **kwargs):
        assert torch.isfinite(height).all() and torch.isfinite(scan).all()
        assert torch.isfinite(state.position).all()
        motion_sizes.append(len(height))
        return predict(state, height, scan, *args, **kwargs)

    def checked_reward(context, state, prediction, *args, **kwargs):
        assert torch.isfinite(prediction.tick_height_above_ground).all()
        assert torch.isfinite(prediction.terrain_scan).all()
        reward_sizes.append(len(state.position))
        return integrate(context, state, prediction, *args, **kwargs)

    monkeypatch.setattr(learner.model, 'step', checked_motion)
    monkeypatch.setattr(LocomotionRewardContext, 'integrate', checked_reward)
    with frozen_parameters(learner.model, learner.critic):
        rollout, rejected, replayed, _ = learner._usable_rollout(start, frames)
        assert rejected == 1 and replayed == 0
        assert len(motion_sizes) == len(reward_sizes) == 16
        assert motion_sizes[0] == 2 and motion_sizes[-1] == reward_sizes[-1] == 1
        assert motion_sizes.count(2) == reward_sizes.count(2) + 1
        assert motion_sizes == sorted(motion_sizes, reverse=True)
        assert reward_sizes == sorted(reward_sizes, reverse=True)
        gradient = torch.autograd.grad(rollout.objective.sum(), learner.actor.decoder[-1].bias)[0]
        assert torch.isfinite(gradient).all() and gradient.norm() > 0
    for name, value in start.items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


def test_nonfinite_after_terrain_removal_keeps_fallback_and_original_row_causes(monkeypatch):
    learner, start, frames = fixture(sensor_noise=True)
    rows = torch.tensor([0, 0, 1])
    start, frames = {name: value[rows] for name, value in start.items()}, frames[rows]
    start['physical'][:, 1] = torch.tensor([0., 1., -1.])
    move_to_scan_boundary(learner, start, [0])
    predict = learner.model.step

    def nonfinite_later(state, *args, **kwargs):
        prediction = predict(state, *args, **kwargs)
        bad = (state.position[:, 1] > .5) & (state.position[:, 0] > -.06)
        physical = replace(prediction.ticks.state,
            position=torch.where(bad[:, None, None], float('nan'), prediction.ticks.state.position))
        return replace(prediction, state=physical.tick(-1), ticks=replace(prediction.ticks, state=physical))

    monkeypatch.setattr(learner.model, 'step', nonfinite_later)
    with frozen_parameters(learner.model, learner.critic):
        rollout, rejected, replayed, causes = learner._usable_rollout(start, frames)
        assert rejected == 2 and replayed == 2
        assert causes == {'nonfinite_prediction': 1, 'terrain_query': 1,
                          'reward_reference': 0, 'nonfinite_return': 0}
        healthy = learner.imagination.rollout({name: value[2:] for name, value in start.items()}, frames[2:], 16)
        actual = torch.autograd.grad(rollout.objective.sum() / 3, learner.actor.decoder[-1].bias)[0]
        expected = torch.autograd.grad(healthy.objective.sum() / 3, learner.actor.decoder[-1].bias)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)
        assert torch.isfinite(actual).all() and actual.norm() > 0
