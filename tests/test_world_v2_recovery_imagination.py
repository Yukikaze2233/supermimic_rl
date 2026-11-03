"""Recovery records advance the native lifecycle inside the full BPTT chain.

Tiny motion fixtures exercise task semantics and derivatives, not skill success.
"""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.world_model_v2.collection import Collector
from supermimic_rl.world_model_v2.learning import frozen_parameters, lambda_returns
from supermimic_rl.world_model_v2.motion_task import MotionTask
from supermimic_rl.world_model_v2.online import OnlineConfig, policy_start_candidates
from supermimic_rl.world_model_v2.records import RecordSpec, decode_physical, decode_memory
from supermimic_rl.world_model_v2.recovery_task import RECOVERY_FEATURE_DIM
from supermimic_rl.world_model_v2.replay import CompactReplay
from supermimic_rl.world_model_v2.reward_assembly import RewardMemory
from supermimic_rl.world_model_v2.stair_source import StairSourceState
from supermimic_rl.world_model_v2.stair_task import StairTaskState
from supermimic_rl.world_model_v2.jump_task import JumpTaskState

from support.world_v2.collection import Environment
from support.world_v2.recovery_task import native_recovery
from support.world_v2.recovery_imagination import fixture


def first_interval_reward(learner, start, frames):
    state, memory = decode_physical(start['physical']), decode_memory(start['memory'])
    action = learner.actor(frames)
    prediction = learner.model.step(state, start['height'], start['scan'], memory,
        start['sim_context'], start['motor_strength'][:, 4:6], action,
        terrain_tile=start['terrain_tile'][:, 0])
    context = learner.imagination.reward.bind(start, state)
    reward_memory = RewardMemory.decode(torch.cat((start['reward_memory'], start['support_memory']), -1))
    interval = context.integrate(state, prediction, action, reward_memory, memory, start['goal'], 0,
        jump_state=JumpTaskState.decode(start), base_mode=MotionTask.decode(start['motion_task']).route[:, 3].long())
    flags = torch.zeros(len(frames), dtype=torch.bool)
    return interval.finish(flags, flags, context.assembly.parameters)


def test_recovery_handoff_precedes_admission_and_does_not_change_old_reward(monkeypatch):
    from supermimic_rl.world_model_v2.reward_assembly import RewardAssembly
    from transformer_rl.formal.tasks.reward import RewardPhase

    phases = []
    original_integrate = RewardAssembly.integrate

    def observe_phase(assembly, state, samples, *args, **kwargs):
        phases.append(torch.stack([tick.reference.phase for tick in samples]))
        return original_integrate(assembly, state, samples, *args, **kwargs)

    monkeypatch.setattr(RewardAssembly, 'integrate', observe_phase)
    learner, start, frames = fixture(near_handoff=True)
    original = {name: value.clone() for name, value in start.items()}
    with frozen_parameters(learner.model, learner.critic):
        rollout = learner.imagination.rollout(start, frames, 16)
        assert rollout.critic_inputs.shape == (17, 2, 401 + RECOVERY_FEATURE_DIM)
        assert (phases[0] == int(RewardPhase.RECOVERY)).all()
        assert (phases[1] == int(RewardPhase.SUPPORT)).all()
        goal = rollout.critic_inputs[:, :, 61:77]
        assert (goal[0, :, 7] == 1).all()
        assert (goal[1:, :, 4] == 1).all() and (goal[1:, :, 7] == 0).all()
        planar = (start['recovery_uniform'][:, 0] * 2 - 1) * torch.tensor([.6, 1.3])
        torch.testing.assert_close(goal[1, :, [0, 2]], planar)
        torch.testing.assert_close(rollout.critic_inputs[1, :, 368], torch.full((2,), 1.68))
        recovery = rollout.critic_inputs[:, :, -RECOVERY_FEATURE_DIM:]
        assert (recovery[0, :, 0] == 1).all() and (recovery[1:, :, 0] == 0).all()
        assert (recovery[1:, :, 1] == 1).all()
        torch.testing.assert_close(rollout.rewards[0], first_interval_reward(learner, start, frames))
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
        assert (numeric - analytic).abs() / analytic.abs() < .005
    for name, value in start.items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


def test_recovery_timeout_bootstraps_without_failure_payment():
    learner, start, frames = fixture()
    start['recovery_clock'][:, 1] = 4.98
    with torch.no_grad():
        learner.actor.estimator[-1].bias[-2:] = -10.
    with frozen_parameters(learner.model, learner.critic):
        rollout = learner.imagination.rollout(start, frames, 16)
        assert rollout.lengths.tolist() == [1, 1]
        assert rollout.active.shape == (1, 2) and rollout.active.all()
        assert rollout.boundary.all() and (rollout.continuation == 1).all()
        assert (rollout.critic_inputs[-1, :, -7] == 0).all()
        torch.testing.assert_close(rollout.rewards[0], first_interval_reward(learner, start, frames))
        torch.testing.assert_close(lambda_returns(rollout, .99, .95)[0],
                                   rollout.rewards[0].detach() + .99 * rollout.values[1].detach())
        torch.testing.assert_close(rollout.objective, rollout.rewards[0] + .99 * rollout.values[1])


def test_true_support_cannot_bypass_low_confidence_estimate():
    learner, start, frames = fixture(near_handoff=True)
    with torch.no_grad():
        learner.actor.estimator[-1].bias[-2:] = -10.
        rollout = learner.imagination.rollout(start, frames, 16)
    assert (rollout.critic_inputs[:, :, 68] == 1).all()  # RECOVERY goal bit.
    assert (rollout.critic_inputs[-1, :, -7] == 1).all()  # Still active.
    assert (rollout.critic_inputs[-1, :, -6] == 1).all()  # Physics certified independently.


def test_recovery_handoff_randomness_is_prepared_once_for_microbatches():
    learner, start, frames = fixture(near_handoff=True)
    other, _, _ = fixture(near_handoff=True)
    other.load_state_dict(deepcopy(learner.state_dict()))
    torch.manual_seed(51)
    full = learner.update_policy(start, frames, microbatch_size=2)
    torch.manual_seed(51)
    split = other.update_policy(start, frames, microbatch_size=1)
    assert full['actor/updated'] and split['actor/updated']
    for actual, expected in zip(learner.actor.decoder.parameters(), other.actor.decoder.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=2e-6)
    invalid = {**start, 'recovery_record_valid': torch.zeros(2, 1)}
    with pytest.raises(ValueError, match='unrecorded recovery'):
        learner.imagination.rollout(invalid, frames, 16)


def test_recovery_records_gate_policy_starts_but_not_world_model_data():
    _, start, _ = fixture()
    excluded = {'observation_noise', 'tof_latency', 'tof_uniform', 'jump_uniform', 'recovery_uniform'}
    record = {name: value for name, value in start.items() if name not in excluded}
    record['frame'] = torch.zeros(2, 45)
    record['recovery_record_valid'][1] = 0.
    fields = {name: value.shape[1:] for name, value in record.items()}
    replay = CompactReplay(fields, 2, 2, 'cpu')
    boundary = {name: torch.zeros(2, dtype=torch.bool) for name in ('terminated', 'truncated')}
    boundary.update({name: torch.ones(2, dtype=torch.bool) for name in ('valid', 'policy_owned', 'end_policy_owned')})
    boundary.update({name: torch.zeros(2, dtype=torch.int64)
                     for name in ('start_episode_id', 'end_episode_id', 'start_control_epoch', 'end_control_epoch')})
    replay.add(record, boundary)
    config = OnlineConfig(record_tasks=True, record_stair_source=True, record_jumps=True,
                          record_recovery=True, course_coordinate=0)
    assert policy_start_candidates(replay, config, [0, 1], .02).tolist() == [0]
    assert replay.window_candidates(1).numel() == 2
    assert policy_start_candidates(replay, replace(config, record_recovery=False), [0, 1], .02).numel() == 0
    restored = CompactReplay.from_state_dict(replay.state_dict())
    assert restored.data['recovery_flags'].dtype == torch.bool
    assert restored.data['recovery_clock'].dtype == torch.float64
    assert 'recovery_flags' not in RecordSpec(16, 65).fields()
    with pytest.raises(ValueError, match='complete jump'):
        OnlineConfig(record_recovery=True)


def test_collector_captures_pre_step_recovery_clock_and_flags(monkeypatch):
    _, start, _ = fixture()
    native = native_recovery(2)
    env = Environment()
    env.adapter = SimpleNamespace(env=native)
    monkeypatch.setattr('supermimic_rl.world_model_v2.collection.SnapshotLayout',
        lambda *a, **kw: SimpleNamespace(encode=lambda transition: {'frame': transition.start.actor_observation}))
    for cls, value in ((MotionTask, start['motion_task']), (StairTaskState, StairTaskState.decode(start)),
                       (StairSourceState, StairSourceState.decode(start)), (JumpTaskState, JumpTaskState.decode(start))):
        monkeypatch.setattr(cls, 'capture', lambda native, value=value: value)
    original = env.step

    def step(action):
        native._terrain_course.recovery.policy_elapsed_s.add_(.02)
        native.time.add_(.02)
        return original(action)

    env.step = step
    collector = Collector(env, lambda history: (torch.zeros(2, 6), torch.ones(2, 5)), capacity_steps=3,
        record_terrain=True, record_tasks=True, record_stairs=True, record_stair_source=True,
        record_jumps=True, record_recovery=True)
    collector.collect(2)
    torch.testing.assert_close(collector.replay.data['recovery_clock'][:2, :, 1],
                               torch.tensor([[0., 0.], [.02, .02]], dtype=torch.float64))
    assert collector.replay.data['recovery_flags'][:2, :, 0].all()
    assert (collector.replay.data['recovery_record_valid'][:2] == 1).all()


def test_four_recovery_orientations_have_finite_bptt_without_changing_actor_or_world():
    from supermimic_rl.formal_observation import deployed_frame
    from supermimic_rl.world_model_v2 import so3
    from supermimic_rl.world_model_v2.records import (
        encode_physical, encode_queues, decode_readout, decode_transforms,
    )
    from supermimic_rl.world_model_v2.sensing import (
        SensorQueues, encoder_imu_sample, gimbal_sample, heading_sample,
    )

    learner, start, _ = fixture()
    rows = torch.tensor([0, 0, 1, 1])
    start = {name: value[rows] for name, value in start.items()}
    physical = decode_physical(start['physical'])
    rotation = torch.zeros(4, 3)
    rotation[:, 0] = torch.tensor([-torch.pi / 2, torch.pi / 2, 0., torch.pi])
    physical = replace(physical, rotation=so3.exp(rotation))
    start['physical'] = encode_physical(physical)
    observation = learner.model.terrain.observe(physical, start['terrain_tile'][:, 0])
    start['height'], start['scan'] = observation.height_above_ground, observation.scan_relative
    transforms = decode_transforms(start['transforms'])
    sensor = encoder_imu_sample(physical.joint_position[:, :4], physical.joint_velocity[:, :6],
        physical.angular_velocity_body, physical.gravity_body, transforms)
    gimbal = gimbal_sample(physical.joint_position[:, 4:], physical.joint_velocity[:, 6:], transforms)
    heading = heading_sample(so3.yaw(physical.rotation))
    queues = SensorQueues(sensor[:, None].expand(-1, 3, -1).clone(),
        gimbal[:, None].expand(-1, 3, -1).clone(), heading[:, None].expand(-1, 3, -1).clone(), torch.full((4,), 3.))
    start['queues'] = encode_queues(queues)
    frame = deployed_frame(queues.values, queues.count, queues.gimbal, start['action'], start['goal'],
                           decode_readout(start['readout']), noise=False)
    frames = frame[:, None].expand(-1, 5, -1)
    with frozen_parameters(learner.model, learner.critic):
        rollout = learner.imagination.rollout(start, frames, 16)
        assert rollout.usable.all()
        gradient = torch.autograd.grad(rollout.objective.mean(), learner.actor.decoder[-1].bias)[0]
        assert torch.isfinite(gradient).all() and gradient.norm() > 0
    assert learner.actor.frame_dim == 35 and learner.actor.action_bounds.numel() == 6
    assert learner.model.motion.input_norm.mean.numel() == 209
    assert learner.model.motion.output_norm.mean.numel() == 224


def test_physical_failure_is_not_recovery_timeout_or_handoff():
    learner, start, frames = fixture(near_handoff=True)
    start['recovery_clock'][:, 1] = 4.98
    start['physical'][:, 2] = -.1
    observed = learner.model.terrain.observe(decode_physical(start['physical']), start['terrain_tile'][:, 0])
    start['height'], start['scan'] = observed.height_above_ground, observed.scan_relative
    start['failure_memory'][:, 1] = learner.imagination.termination.limits.pose_failure_confirm_s
    rollout = learner.imagination.rollout(start, frames, 16)
    assert (rollout.continuation[0] == 0).all()
    assert rollout.active.sum(0).tolist() == [1, 1]
    assert (rollout.critic_inputs[1:, :, 68] == 1).all()
    assert (rollout.critic_inputs[1:, :, -7] == 0).all()
    torch.testing.assert_close(rollout.objective, rollout.rewards[0])


def test_terrain_isolation_keeps_recovery_handoff_and_healthy_gradient():
    from support.world_v2.stair_imagination import move_to_scan_boundary

    learner, start, frames = fixture(near_handoff=True)
    with torch.no_grad():
        learner.model.motion.net[-1].bias.view(4, -1)[:, 0] = torch.arange(1, 5) * .012
    move_to_scan_boundary(learner, start, [0])
    with frozen_parameters(learner.model, learner.critic):
        isolated, rejected, replayed, _ = learner._usable_rollout(start, frames)
        assert rejected == 1 and replayed == 0 and isolated.source_rows.tolist() == [1]
        single = learner.imagination.rollout({name: value[1:] for name, value in start.items()}, frames[1:], 16)
        torch.testing.assert_close(isolated.critic_inputs, single.critic_inputs, rtol=0, atol=2e-6)
        torch.testing.assert_close(lambda_returns(isolated, .99, .95),
                                   lambda_returns(single, .99, .95), rtol=0, atol=2e-6)
        actual = torch.autograd.grad(isolated.objective.sum() / 2, learner.actor.decoder[-1].bias)[0]
        expected = torch.autograd.grad(single.objective.sum() / 2, learner.actor.decoder[-1].bias)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)


def test_unmanaged_recovery_handoff_updates_family_without_inventing_random_draws():
    from transformer_rl.formal.tasks.curriculum import TaskFamily

    learner, start, frames = fixture(near_handoff=True, managed=False)
    assert 'recovery_uniform' not in start
    with frozen_parameters(learner.model, learner.critic):
        rollout = learner.imagination.rollout(start, frames, 2)
    families = rollout.critic_inputs[:, :, 287:293].argmax(-1)
    assert (families[0] == int(TaskFamily.RECOVERY)).all()
    assert (families[1:] == int(TaskFamily.HEIGHT)).all()
    assert (rollout.critic_inputs[1:, :, 65] == 1).all()  # NORMAL command.
    torch.testing.assert_close(rollout.critic_inputs[1:, :, 61:65],
                               start['goal'][:, :4].expand(2, -1, -1))
