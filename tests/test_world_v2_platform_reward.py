"""Explicit platform evidence reaches the sole shared physics/payment owner."""
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.course.jump_physics import (
    PlatformLandingEvidence, PlatformLandingPhase, PlatformLandingState, observe_jump_physics,
)
from supermimic_rl.world_model_v2.imagined_reward import LocomotionRewardContext
from supermimic_rl.world_model_v2.jump_task import JumpTask, JumpTaskConfig, JumpTaskState
from supermimic_rl.world_model_v2.model import Ticks
from supermimic_rl.world_model_v2.records import decode_memory
from support.world_v2.jump_task import native_jumps
from support.world_v2.reward_assembly import setup


def inputs():
    _, assembly, geometry, state, legs, memory = setup('legacy')
    state = replace(state, position=state.position.clone())
    state.position[:, 2] = .38
    motion = state.with_time_axis().map(lambda value: value.expand(-1, 4, *value.shape[2:]).clone())
    knee = torch.stack([geometry.linkage.evaluate(side.name,
        state.joint_position[:, side.secondary] - state.joint_position[:, side.primary])[0]
        ['passive_joint'][:, side.passive_knee] for side in geometry.sides], -1)
    ticks = Ticks(motion, torch.full((2, 4, 2), 20.), torch.zeros(2, 4, 21),
                  torch.zeros(2, 4, 3), knee[:, None].expand(-1, 4, -1), torch.zeros(2, 4, 2))
    native = native_jumps(2, origin_z=1.125)
    owner = native._terrain_course.tasks
    for name, value in (('physical_id', 3), ('active_id', 3), ('latched_height', .32),
                        ('active_rise', .06), ('ground_anchor', 1.125), ('peak_rise', .06),
                        ('landed_s', 1.99)):
        getattr(owner, name).fill_(value)
    for name in ('supported_start', 'flight_seen', 'ascent_seen'):
        getattr(owner, name).fill_(True)
    owner.protocol.active_id.fill_(3)
    owner.protocol.accepted_at_s.fill_(0.)
    context = LocomotionRewardContext(assembly, geometry, legs, torch.zeros(2, 6), torch.ones(2, 6),
        torch.zeros(2), torch.zeros(2, dtype=torch.bool),
        jump_task=JumpTask(JumpTaskConfig.capture(native)), base_com_offset=torch.zeros(2, 3))
    goal = torch.zeros(2, 16)
    goal[:, 3], goal[:, 8], goal[:, 11] = 1.6, 1., 1.
    platform = PlatformLandingState(torch.full((2,), 3, dtype=torch.long), torch.zeros(2, dtype=torch.long),
        torch.tensor([int(PlatformLandingPhase.TARGET_BOTH_WHEELS), int(PlatformLandingPhase.MISSED)]))
    yes, no = torch.ones(2, 4, dtype=torch.bool), torch.zeros(2, 4, dtype=torch.bool)
    evidence = PlatformLandingEvidence(yes, yes, no, yes[..., None].expand(-1, -1, 2),
        yes[..., None].expand(-1, -1, 2), no, yes, torch.full((2, 4), .38))
    kwargs = dict(start=state, prediction=SimpleNamespace(ticks=ticks), action=torch.full((2, 6), .05),
                  reward_memory=memory, controller_memory=decode_memory(torch.zeros(2, 16)), goal=goal, step=0,
                  jump_state=JumpTaskState.capture(native), base_mode=owner.base_mode)
    return context, kwargs, native, platform, evidence


def test_platform_interval_matches_native_world_coordinates_and_pays_only_once():
    context, kwargs, native, platform, evidence = inputs()
    original_platform = {f.name: getattr(platform, f.name).clone() for f in fields(platform)}
    original_evidence = {f.name: getattr(evidence, f.name).clone() for f in fields(evidence)}
    native_platform = platform
    owner, ticks = native._terrain_course.tasks, kwargs['prediction'].ticks
    world = replace(evidence, landing_world_height=evidence.landing_world_height + 1.125)
    for tick in range(4):
        native_platform = observe_jump_physics(owner, ticks.wheel_load[:, tick], torch.zeros(2),
            torch.zeros(2), ticks.state.position[:, tick, 2] + 1.125,
            angular_speed_rad_s=torch.zeros(2), platform_state=native_platform,
            platform_evidence=world.tick(tick))
    result = context.integrate(**kwargs, platform_state=platform, platform_evidence=evidence)
    assert result.success_credited.tolist() == [True, False]
    expected = JumpTaskState.capture(native)
    for name, value in expected.record().items():
        torch.testing.assert_close(result.jump_state.record()[name], value, rtol=0, atol=1e-7)
    for field in fields(platform):
        torch.testing.assert_close(getattr(result.platform_state, field.name),
                                   getattr(native_platform, field.name), rtol=0, atol=0)
        torch.testing.assert_close(getattr(platform, field.name), original_platform[field.name], rtol=0, atol=0)
    for field in fields(evidence):
        torch.testing.assert_close(getattr(evidence, field.name), original_evidence[field.name], rtol=0, atol=0)
    again = context.integrate(**{**kwargs, 'jump_state': result.jump_state},
                              platform_state=result.platform_state, platform_evidence=evidence)
    assert not again.success_credited.any()
    torch.testing.assert_close(again.jump_state.ids[:, 6], result.jump_state.ids[:, 6], rtol=0, atol=0)


def test_default_and_explicit_none_keep_reward_memory_and_gradients_identical():
    context, kwargs, _, _, _ = inputs()
    # A prediction-side attribute is not a producer interface and must never be
    # treated as a future recorded label. Only the explicit argument is consumed.
    kwargs['prediction'].platform_evidence = object()
    outputs, gradients = [], []
    for options in ({}, {'platform_state': None, 'platform_evidence': None}):
        action = kwargs['action'].clone().requires_grad_()
        result = context.integrate(**{**kwargs, 'action': action}, **options)
        outputs.append(result)
        gradients.append(torch.autograd.grad(result.physical_reward.sum(), action)[0])
        assert result.platform_state is None
    first, second = outputs
    assert first.components.keys() == second.components.keys()
    for name in first.components:
        torch.testing.assert_close(first.components[name], second.components[name], rtol=0, atol=0)
    for name in ('physical_reward', 'failure_scale', 'wheel_torque_history', 'invalid',
                 'termination_height', 'success_credited'):
        torch.testing.assert_close(getattr(first, name), getattr(second, name), rtol=0, atol=0)
    torch.testing.assert_close(first.memory.encode(), second.memory.encode(), rtol=0, atol=0)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)
    assert torch.isfinite(gradients[0]).all()


@pytest.mark.parametrize('supplied', ['history_only', 'evidence_only', 'without_jump_task'])
def test_platform_interval_requires_both_history_evidence_and_the_shared_task(supplied):
    context, kwargs, _, platform, evidence = inputs()
    options = dict(platform_state=platform, platform_evidence=evidence)
    if supplied == 'history_only':
        options.pop('platform_evidence')
    elif supplied == 'evidence_only':
        options.pop('platform_state')
    else:
        context = replace(context, jump_task=None)
    with pytest.raises(ValueError, match='Platform reward requires'):
        context.integrate(**kwargs, **options)
