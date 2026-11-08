"""Causal platform value context and rollout state; tiny motion is not ability evidence."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.course.jump_physics import PlatformLandingEvidence, PlatformLandingState
from supermimic_rl.course.platform_intent import PlatformIntentPhase, PlatformIntentState
from supermimic_rl.world_model_v2 import so3
from supermimic_rl.world_model_v2.imagination import Critic, Imagination
from supermimic_rl.world_model_v2.jump_task import JumpTaskState
from supermimic_rl.world_model_v2.learning import frozen_parameters
from supermimic_rl.world_model_v2.motion_task import MOTION_TASK_DIM, MotionTask
from supermimic_rl.world_model_v2.platform_features import PLATFORM_FEATURE_DIM, platform_critic_features
from supermimic_rl.world_model_v2.platform_records import PlatformRecordState
from supermimic_rl.world_model_v2.platform_task import PlatformTask
from supermimic_rl.world_model_v2.records import decode_memory, decode_physical
from support.world_v2.jump_imagination import fixture as jump_fixture
from support.world_v2.stair_imagination import move_to_scan_boundary


class SurfaceFixture:
    """Explicit finite target points; actual triangle geometry has separate tests."""
    def __init__(self, terrain):
        self.terrain = terrain

    def nearest(self, position, tile, target):
        point = position.new_tensor([.5, .1, .2]).expand_as(position)
        valid = tile == 1
        return SimpleNamespace(point_local=torch.where(valid[:, None], point, torch.nan), source_valid=valid)


class PredictedContactFixture:
    """Has no record/replay reference, only predicted physical inputs."""
    def __init__(self):
        self.calls = []

    def __call__(self, ticks, *, intent, jump_state, command_height, terrain_tile):
        assert terrain_tile.dtype == torch.long
        self.calls.append((intent.target_index.clone(), intent.phase.clone(), jump_state.physical_id.clone()))
        selected = ((intent.bound_request_id >= 0) & (intent.bound_request_id == jump_state.physical_id))
        selected = selected[:, None].expand(-1, 4)
        yes, no = torch.ones_like(selected), torch.zeros_like(selected)
        wheel = yes[..., None].expand(-1, -1, 2)
        return PlatformLandingEvidence(selected, yes, no, wheel, wheel, no, yes,
                                       ticks.state.position[..., 2])


def fixture():
    learner, start, frames = jump_fixture(gated=True)
    old, contact = learner.imagination, PredictedContactFixture()
    geometry = SurfaceFixture(learner.model.terrain)
    routes = dict(task_dim=MOTION_TASK_DIM, terrain=True, stairs=True, jumps=True,
                  jump_trigger=True, platforms=True)
    critic = Critic(Critic.feature_size(16, 65, **routes), (16,), **routes)
    task = PlatformTask(old.jump_task, old.stair_task, old.stair_source)
    imagined = Imagination(learner.model, learner.actor, critic, old.termination, reward=old.reward,
        stair_task=old.stair_task, stair_source=old.stair_source, jump_task=old.jump_task,
        platform_task=task, platform_contact=contact, platform_geometry=geometry)
    intent = PlatformIntentState.plan(target_index=torch.tensor([0, 1]),
        earliest_episode_s=torch.tensor([.12, .18], dtype=torch.float64),
        rise_m=torch.tensor([.10, .20]), support_height_m=torch.full((2,), .35))
    platform = PlatformRecordState(intent, PlatformLandingState.empty(intent.bound_request_id),
                                   torch.ones(2, 1, dtype=torch.bool))
    start.update(platform.record())
    return learner, imagined, start, frames, contact


def test_platform_features_are_22d_relative_causal_and_mask_inactive_gradients():
    learner, imagined, start, _, _ = fixture()
    state = decode_physical(start['physical'])
    state = replace(state, position=state.position.clone().requires_grad_())
    platform, motion, jump = (PlatformRecordState.decode(start), MotionTask.decode(start['motion_task']),
                              JumpTaskState.decode(start))
    # The inactive row's geometry is genuinely missing, not fabricated as zero.
    intent = replace(platform.intent, target_index=torch.tensor([0, -1]), phase=torch.tensor([1, 0]))
    tile = torch.tensor([1, 0])
    args = (intent, platform.landing, state, motion, jump, imagined.platform_geometry, tile,
            imagined.jump_task.config)
    features = platform_critic_features(*args)
    assert features.shape == (2, PLATFORM_FEATURE_DIM) == (2, 22)
    assert torch.isfinite(features).all() and not features[1].any()
    expected_offset = torch.tensor([.5, .1, .2]) - state.position[0]
    torch.testing.assert_close(features[0, 11:14], expected_offset, rtol=0, atol=0)
    gradient = torch.autograd.grad(features.sum(), state.position)[0]
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient[0], -torch.ones(3), rtol=0, atol=0)
    assert not gradient[1].any()
    # Absolute wall time and request IDs are absent; only the matching relation
    # and time until the same caller deadline matter.
    shifted_intent = replace(intent, earliest_episode_s=intent.earliest_episode_s + 10.,
                             bound_request_id=intent.bound_request_id + 1000)
    shifted_jump = replace(jump, ids=jump.ids.clone())
    shifted_jump.ids[:, 4] += 2000  # Both old/new relations remain unmatched.
    shifted_motion = replace(motion, timing=motion.timing.clone())
    shifted_motion.timing[:, 0] += 10.
    shifted = platform_critic_features(shifted_intent, platform.landing, state, shifted_motion,
        shifted_jump, imagined.platform_geometry, tile, imagined.jump_task.config)
    torch.testing.assert_close(shifted, features, rtol=0, atol=1e-7)
    turned = replace(state, rotation=so3.exp(torch.tensor([[0., 0., torch.pi / 2]]).expand(2, -1)))
    rotated = platform_critic_features(intent, platform.landing, turned, motion, jump,
        imagined.platform_geometry, tile, imagined.jump_task.config)
    torch.testing.assert_close(rotated[0, 11:14], torch.stack((expected_offset[1], -expected_offset[0], expected_offset[2])),
                               rtol=0, atol=2e-7)


def test_legacy_critic_shape_keys_and_strict_restore_remain_unchanged():
    routes = dict(task_dim=MOTION_TASK_DIM, terrain=True, stairs=True, jumps=True, jump_trigger=True)
    old_size = Critic.feature_size(16, 65, **routes)
    assert Critic.feature_size(16, 65, **routes, platforms=False) == old_size
    assert Critic.feature_size(16, 65, **routes, platforms=True) == old_size + 22
    torch.manual_seed(83)
    legacy = Critic(old_size, (16,), **routes)
    torch.manual_seed(83)
    explicit = Critic(old_size, (16,), **routes, platforms=False)
    assert legacy.state_dict().keys() == explicit.state_dict().keys()
    explicit.load_state_dict(legacy.state_dict(), strict=True)
    for name, value in legacy.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[name], rtol=0, atol=0)
    sample = torch.randn(2, old_size, requires_grad=True)
    torch.testing.assert_close(legacy(sample), explicit(sample), rtol=0, atol=0)
    with pytest.raises(ValueError, match='gated jump'):
        Critic(old_size, platforms=True)


@pytest.mark.parametrize('diagnostic_height', [-999., float('nan')])
def test_rollout_starts_from_recorded_context_advances_and_never_reads_future_labels(diagnostic_height):
    learner, imagined, start, frames, contact = fixture()
    original = {name: value.clone() for name, value in start.items()}
    platform = PlatformRecordState.decode(start)
    initial = platform_critic_features(platform.intent, platform.landing, decode_physical(start['physical']),
        MotionTask.decode(start['motion_task']), JumpTaskState.decode(start), imagined.platform_geometry,
        start['terrain_tile'][:, 0], imagined.jump_task.config)
    with frozen_parameters(learner.model, imagined.critic):
        result = imagined.rollout(start, frames, 12)
        assert result.critic_inputs.shape[-1] == imagined.critic.norm.mean.numel()
        torch.testing.assert_close(result.critic_inputs[0, :, -22:], initial, rtol=0, atol=0)
        assert (result.critic_inputs[1:, :, -22:-16].argmax(-1) == int(PlatformIntentPhase.ACCEPTED)).any()
        gradient = torch.autograd.grad(result.objective.sum(), learner.actor.decoder[-1].bias)[0]
        assert torch.isfinite(gradient).all() and gradient.norm() > 0
        assert contact.calls[0][1].tolist() == [1, 1]
        assert contact.calls[0][2].tolist() == [-1, -1]
        assert any((call[2] >= 0).any() for call in contact.calls[1:])
        # These are deliberately contradictory future labels from a record. They
        # must neither qualify imagined landing nor influence its scalar return.
        poison = {**start, 'platform_evidence_ticks': torch.ones(2, 4, 9, dtype=torch.bool),
            'platform_landing_world_height_ticks': torch.full((2, 4, 1), diagnostic_height),
            'platform_landing_state_ticks': torch.full((2, 4, 3), 999, dtype=torch.long),
            'contact_valid_ticks': torch.zeros(2, 4, 1, dtype=torch.bool)}
        repeated = imagined.rollout(poison, frames, 12)
        torch.testing.assert_close(repeated.objective, result.objective, rtol=0, atol=0)
        torch.testing.assert_close(repeated.critic_inputs, result.critic_inputs, rtol=0, atol=0)
        second_gradient = torch.autograd.grad(repeated.objective.sum(), learner.actor.decoder[-1].bias)[0]
        torch.testing.assert_close(second_gradient, gradient, rtol=0, atol=0)
    for name, value in original.items():
        torch.testing.assert_close(start[name], value, rtol=0, atol=0)
    invalid = {**start, 'platform_record_valid': torch.zeros(2, 1, dtype=torch.bool)}
    with pytest.raises(ValueError, match='known caller'):
        imagined.rollout(invalid, frames, 1)
    with pytest.raises(ValueError, match='silently ignored'):
        learner.imagination.critic_features(decode_physical(start['physical']), start['height'], start['scan'],
            decode_memory(start['memory']), start['goal'], start['sim_context'], start['failure_memory'],
            start['failure_conditions'], torch.cat((start['reward_memory'], start['support_memory']), -1),
            start['motion_task'], **learner.imagination.task_features(start))


def test_terrain_compaction_preserves_platform_rows_and_actor_gradient():
    learner, imagined, start, frames, contact = fixture()
    with torch.no_grad():
        learner.model.motion.net[-1].bias.view(4, -1)[:, 0] = torch.arange(1, 5) * .012
    move_to_scan_boundary(learner, start, [0])
    with frozen_parameters(learner.model, imagined.critic):
        result = imagined.rollout(start, frames, 8, isolate_terrain=True)
        assert result.source_rows.tolist() == [1]
        assert result.rejection_reasons['terrain_query'].tolist() == [True, False]
        assert contact.calls[-1][0].tolist() == [1]
        gradient = torch.autograd.grad(result.objective.sum(), learner.actor.decoder[-1].bias)[0]
        single = imagined.rollout({name: value[1:] for name, value in start.items()}, frames[1:], 8)
        reference = torch.autograd.grad(single.objective.sum(), learner.actor.decoder[-1].bias)[0]
        torch.testing.assert_close(result.critic_inputs[:, 0], single.critic_inputs[:, 0], rtol=0, atol=2e-6)
        torch.testing.assert_close(gradient, reference, rtol=0, atol=1e-7)
        assert torch.isfinite(gradient).all() and gradient.norm() > 0


def test_ended_row_holds_platform_context_while_other_row_continues():
    learner, imagined, start, frames, _ = fixture()
    start['platform_intent_clock'].fill_(10.)
    motion = MotionTask.decode(start['motion_task'])
    motion.route[0, 4] = 1.
    start['motion_task'] = motion.encode()
    start['tof_segment_end'][0] = .06
    result = imagined.rollout(start, frames, 8)
    assert 0 < result.lengths[0] < result.lengths[1] == 8
    end = int(result.lengths[0])
    torch.testing.assert_close(result.critic_inputs[end:, 0, -22:],
        result.critic_inputs[end:end + 1, 0, -22:].expand(len(result.critic_inputs) - end, -1), rtol=0, atol=0)
