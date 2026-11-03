"""Optional native jump records flow through collection, BPTT and command arbitration.

The tiny learned motion is a differentiable fixture, not a jump ability claim.
Native request/physical producer equivalence is checked in test_world_v2_jump_task.
"""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.world_model_v2.collection import Collector
from supermimic_rl.world_model_v2.jump_task import JUMP_FEATURE_DIM, JumpTaskState
from supermimic_rl.world_model_v2.learning import frozen_parameters
from supermimic_rl.world_model_v2.motion_task import MotionTask
from supermimic_rl.world_model_v2.online import OnlineConfig, policy_start_candidates
from supermimic_rl.world_model_v2.records import RecordSpec, decode_physical
from supermimic_rl.world_model_v2.replay import CompactReplay
from supermimic_rl.world_model_v2.stair_source import StairSourceState
from supermimic_rl.world_model_v2.stair_task import StairTaskState

from support.world_v2.collection import Environment
from support.world_v2.jump_task import native_jumps
from support.world_v2.jump_imagination import fixture


def test_jump_h16_uses_deployed_estimator_native_goal_and_finite_actor_gradient():
    learner, start, frames = fixture()
    original = {name: value.clone() for name, value in start.items()}
    with frozen_parameters(learner.model, learner.critic):
        rollout = learner.imagination.rollout(start, frames, 16)
        assert rollout.critic_inputs.shape == (17, 2, 369 + JUMP_FEATURE_DIM)
        assert rollout.lengths.tolist() == [16, 16]
        assert not rollout.invalid.any()
        goal = rollout.critic_inputs[:, :, 61:77]
        assert (goal[1:, :, 8] == 1).all()  # Actual CommandMode.JUMP is 4.
        assert (goal[:, :, 7] == 0).all()   # Recovery remains excluded.
        torch.testing.assert_close(goal[1:, :, 9], torch.full((16, 2), 1.25))
        expected_clock = (torch.arange(16) * .02)[:, None].expand(-1, 2)
        torch.testing.assert_close(goal[1:, :, 10], expected_clock, rtol=0, atol=1e-7)
        gradient = torch.autograd.grad(rollout.objective.mean(), learner.actor.decoder[-1].bias)[0]
        assert torch.isfinite(gradient).all() and gradient.norm() > 0
        parameter = learner.actor.decoder[-1].bias
        baseline, direction = parameter.detach().clone(), gradient / gradient.norm()
        with torch.no_grad():
            parameter.copy_(baseline + .001 * direction)
            positive = learner.imagination.rollout(start, frames, 16).objective.mean()
            parameter.copy_(baseline - .001 * direction)
            negative = learner.imagination.rollout(start, frames, 16).objective.mean()
            parameter.copy_(baseline)
        numeric, analytic = (positive - negative) / .002, gradient.dot(direction)
        assert (numeric - analytic).abs() / analytic.abs() < .003
    for name, value in start.items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0, equal_nan=True)
    # A low-confidence deployed estimate rejects the automatic request even
    # though this same model predicts both wheels in physical contact.
    with torch.no_grad():
        learner.actor.estimator[-1].bias[-2:] = -10.
        refused = learner.imagination.rollout(start, frames, 3)
    assert (refused.critic_inputs[:, :, 69] == 0).all()


def test_jump_admission_precedes_stair_sensor_and_existing_stair_blocks_jump(monkeypatch):
    learner, start, frames = fixture()
    start['stair_ids'][1, 1] = 7
    start['stair_clock'][1, 1] = 0.
    start['stair_values'][1, 0] = .20
    idle_seen = []
    source = learner.imagination.stair_source
    original = source.step

    def observe(trigger, beam, *, idle, latency, draws):
        idle_seen.append(idle.clone())
        return original(trigger, beam, idle=idle, latency=latency, draws=draws)

    monkeypatch.setattr(source, 'step', observe)
    rollout = learner.imagination.rollout(start, frames, 1)
    goal = rollout.critic_inputs[-1, :, 61:77]
    assert goal[:, 8].tolist() == [1., 0.]
    assert goal[:, 9].tolist() == [1.25, 0.]
    assert idle_seen[0].tolist() == [False, False]


def test_integrated_reward_credits_physical_landing_once_and_failure_suppresses_it():
    from supermimic_rl.world_model_v2.records import decode_memory
    from supermimic_rl.world_model_v2.reward_assembly import RewardMemory

    learner, start, _ = fixture()
    state, memory = decode_physical(start['physical']), decode_memory(start['memory'])
    task = learner.imagination.jump_task
    jump = JumpTaskState.decode(start)
    jump.ids[:, 1] = jump.ids[:, 4] = 7
    jump.values[:, [1, 9]] = 0.  # Neither deployment nor physical attempt cancelled.
    jump.values[:, [2, 10, 11, 12, 13]] = 1.
    jump.values[:, 5] = jump.values[:, 15] = .25
    jump.values[:, 6] = .35
    jump.values[:, 7] = 0.
    jump.values[:, 16] = 1.99
    start['goal'][:, 4] = 0.
    start['goal'][:, 8] = 1.
    start['goal'][:, 9] = 1.25  # Larger than mode one-hot; never a mode selector.
    start['reward_route'][:, 0] = 4.
    # A stationary supported fixture after an already-recorded successful
    # flight completes the native 2s landing hold within this four-tick window.
    with torch.no_grad():
        learner.model.motion.net[-1].weight.zero_()
    action = torch.zeros(2, 6)
    prediction = learner.model.step(state, start['height'], start['scan'], memory,
        start['sim_context'], start['motor_strength'][:, 4:6], action,
        terrain_tile=start['terrain_tile'][:, 0])
    context = learner.imagination.reward.bind(start, state)
    reward_memory = RewardMemory.decode(torch.cat((start['reward_memory'], start['support_memory']), -1))
    first = context.integrate(state, prediction, action, reward_memory, memory, start['goal'], 0,
                              jump_state=jump, base_mode=torch.zeros(2, dtype=torch.long))
    assert first.success_credited.tolist() == [True, True]
    no_failure = torch.zeros(2, dtype=torch.bool)
    paid = first.finish(no_failure, first.success_credited, context.assembly.parameters)
    torch.testing.assert_close(paid - first.physical_reward, torch.full((2,), 1., dtype=paid.dtype))
    failed = first.finish(~no_failure, first.success_credited, context.assembly.parameters)
    without_success = first.finish(~no_failure, no_failure, context.assembly.parameters)
    torch.testing.assert_close(failed, without_success, rtol=0, atol=0)
    second = context.integrate(state, prediction, action, first.memory, memory, start['goal'], 1,
                               jump_state=first.jump_state, base_mode=torch.zeros(2, dtype=torch.long))
    assert not second.success_credited.any()


def test_jump_realization_preserves_microbatch_gradients_and_invalid_records_are_rejected():
    learner, start, frames = fixture()
    other, _, _ = fixture()
    other.load_state_dict(deepcopy(learner.state_dict()))
    torch.manual_seed(41)
    first = learner.update_policy(start, frames, microbatch_size=2)
    torch.manual_seed(41)
    second = other.update_policy(start, frames, microbatch_size=1)
    assert first['actor/updated'] and second['actor/updated']
    for actual, expected in zip(learner.actor.decoder.parameters(), other.actor.decoder.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=2e-6)
    invalid = {**start, 'jump_record_valid': torch.zeros(2, 1)}
    with pytest.raises(ValueError, match='unrecorded jump'):
        learner.imagination.rollout(invalid, frames, 16)


def test_policy_start_gate_requires_complete_jump_records_and_pauses_segment_resampling():
    learner, start, _ = fixture()
    record = {name: value for name, value in start.items()
              if name not in ('observation_noise', 'tof_latency', 'tof_uniform', 'jump_uniform')}
    record['frame'] = torch.zeros(2, 45)
    # Row one is a real recorded active jump. Row two is an unrecorded legacy row.
    record['jump_record_valid'][1] = 0.
    record['jump_ids'][:, 1] = 8
    record['reward_route'][:, 0] = 4
    record['goal'][:, 4] = 0
    record['goal'][:, 8] = 1
    task = MotionTask.decode(record['motion_task'])
    task.route[:, 4] = 1
    record['motion_task'] = task.encode()
    record['tof_segment_end'].fill_(0.)
    fields = {name: value.shape[1:] for name, value in record.items()}
    replay = CompactReplay(fields, 2, 2, 'cpu')
    boundary = {name: torch.zeros(2, dtype=torch.bool) for name in ('terminated', 'truncated')}
    boundary.update({name: torch.ones(2, dtype=torch.bool) for name in ('valid', 'policy_owned', 'end_policy_owned')})
    boundary.update({name: torch.zeros(2, dtype=torch.int64)
                     for name in ('start_episode_id', 'end_episode_id', 'start_control_epoch', 'end_control_epoch')})
    replay.add(record, boundary)
    config = OnlineConfig(record_tasks=True, record_stair_source=True, record_jumps=True, course_coordinate=0)
    assert policy_start_candidates(replay, config, [0, 1], .02).tolist() == [0]
    ordinary_config = replace(config, record_jumps=False)
    assert policy_start_candidates(replay, ordinary_config, [0, 1], .02).numel() == 0
    source, stair, jump = StairSourceState.decode(record), StairTaskState.decode(record), JumpTaskState.decode(record)
    assert source.boundary_known(task, stair, .02, jump).all()
    released = replace(jump, ids=jump.ids.clone())
    released.ids[:, 1] = -1
    assert not source.boundary_known(task, stair, .02, released).any()
    assert 'jump_ids' not in RecordSpec(16, 65).fields()
    restored = CompactReplay.from_state_dict(replay.state_dict())
    assert restored.data['jump_ids'].dtype == torch.int64
    assert restored.data['jump_clock'].dtype == torch.float64


def test_collector_records_pre_step_jump_state_without_changing_physical_contract(monkeypatch):
    _, start, _ = fixture()
    native = native_jumps(2)
    env = Environment()
    env.adapter = SimpleNamespace(env=native)
    monkeypatch.setattr('supermimic_rl.world_model_v2.collection.SnapshotLayout',
        lambda *a, **kw: SimpleNamespace(encode=lambda transition: {'frame': transition.start.actor_observation}))
    monkeypatch.setattr(MotionTask, 'capture', lambda native: start['motion_task'].clone())
    monkeypatch.setattr(StairTaskState, 'capture', lambda native: StairTaskState.decode(start))
    monkeypatch.setattr(StairSourceState, 'capture', lambda native: StairSourceState.decode(start))
    original = env.step

    def step(action):
        native._terrain_course.tasks.next_id.add_(1)
        native.time.add_(.02)
        return original(action)

    env.step = step
    collector = Collector(env, lambda history: (torch.zeros(2, 6), torch.ones(2, 5)), capacity_steps=3,
        record_terrain=True, record_tasks=True, record_stairs=True, record_stair_source=True, record_jumps=True)
    collector.collect(2)
    assert (collector.replay.data['jump_ids'][1, :, 3] - collector.replay.data['jump_ids'][0, :, 3] == 1).all()
    assert (collector.replay.data['jump_record_valid'][:2] == 1).all()
    assert collector.replay.data['frame'].shape[-1] == 45
    assert env.estimates[0].shape == (2, 8)


def test_terrain_isolation_retains_jump_history_and_microbatch_gradient():
    from support.world_v2.stair_imagination import move_to_scan_boundary
    from supermimic_rl.world_model_v2.learning import lambda_returns

    learner, start, frames = fixture()
    with torch.no_grad():
        learner.model.motion.net[-1].bias.view(4, -1)[:, 0] = torch.arange(1, 5) * .012
    move_to_scan_boundary(learner, start, [0])
    other, _, _ = fixture()
    other.load_state_dict(deepcopy(learner.state_dict()))
    with frozen_parameters(learner.model, learner.critic):
        rollout, rejected, replayed, _ = learner._usable_rollout(start, frames)
        assert rejected == 1 and replayed == 0 and rollout.source_rows.tolist() == [1]
        single = learner.imagination.rollout({name: value[1:] for name, value in start.items()}, frames[1:], 16)
        torch.testing.assert_close(rollout.critic_inputs, single.critic_inputs, rtol=0, atol=2e-6)
        torch.testing.assert_close(lambda_returns(rollout, .99, .95),
                                   lambda_returns(single, .99, .95), rtol=0, atol=2e-6)
        actual = torch.autograd.grad(rollout.objective.sum() / 2, learner.actor.decoder[-1].bias)[0]
        expected = torch.autograd.grad(single.objective.sum() / 2, learner.actor.decoder[-1].bias)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)
    torch.manual_seed(41)
    full = learner.update_policy(start, frames, microbatch_size=2)
    torch.manual_seed(41)
    split = other.update_policy(start, frames, microbatch_size=1)
    for record in (full, split):
        assert record['actor/updated']
        assert record['imagination/accepted_trajectories'] == 1
        assert record['imagination/replayed_trajectories'] == 0
    for actual, expected in zip(learner.actor.decoder.parameters(), other.actor.decoder.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=2e-6)
