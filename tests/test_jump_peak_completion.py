"""Requested peak progress must remove small-hop profit without changing other tasks."""
from dataclasses import replace

import pytest
import torch

pytest.importorskip('transformer_rl.formal.tasks.reward')
from transformer_rl.formal.tasks.jump import ReferenceStage
from transformer_rl.formal.tasks.reward import CommandMode

from supermimic_rl.reward_kernel import (
    JumpProgress, RewardExtensions, RewardHistory, RewardParameters,
    physical_reward_densities,
)
from test_reward_kernel import begin, sample, setup


TRACKING = ('forward', 'yaw', 'height', 'jump_vertical')


def _progress(*, stage='HOLD', started=(True, True), closed=(True, True),
              peak=(.085, .085), requested=(.2, .2)):
    return JumpProgress(
        torch.full((2,), int(getattr(ReferenceStage, stage))),
        torch.tensor(started), torch.tensor(closed), torch.zeros(2),
        torch.tensor(started),
        peak if isinstance(peak, torch.Tensor) else torch.tensor(peak),
        torch.tensor(requested),
    )


def _evaluate(progress, *, enabled=True, case='jump_landing', mode=None):
    composer, reference = setup(case)
    command = reference.admitted_body_command.clone()
    command[:, :3] = 0.
    reference = replace(reference, admitted_body_command=command)
    if mode is not None:
        reference = replace(reference, mode=torch.full((2,), int(mode)))
    begin(composer, reference, torch.zeros(2, 6))
    physics = replace(sample(composer, 1),
        velocity_origin_body_m_s=torch.zeros(2, 3),
        velocity_origin_world_m_s=torch.zeros(2, 3),
        angular_velocity_body_rad_s=torch.zeros(2, 3))
    reference = replace(reference, jump_height_world_z_m=physics.root_world_z_m,
                        jump_velocity_world_z_m_s=torch.zeros(2))
    history = RewardHistory(composer._window['dq'], torch.zeros(2), torch.zeros(2), torch.zeros(2))
    extension = RewardExtensions(jump_missed=1., jump_peak_completion=enabled)
    parameters = replace(RewardParameters.from_composer(composer), extensions=extension)
    return physical_reward_densities(physics, reference, history, parameters, jump_progress=progress)


def _assert_only_completion_changed(actual, baseline, completion):
    for key, value in baseline.positive.items():
        expected = value * completion if key in TRACKING else value
        torch.testing.assert_close(actual.positive[key], expected)
    for key, value in baseline.cost.items():
        expected = 1. - completion if key == 'jump_missed' else value
        torch.testing.assert_close(actual.cost[key], expected)
    assert not actual.invalid.any()


def test_small_hop_landing_gets_partial_tracking_and_remaining_missed_cost():
    progress = _progress(peak=(.0425, .085))
    baseline = _evaluate(progress, enabled=False)
    actual = _evaluate(progress)
    for key, expected in zip(TRACKING, (.8, 1., 1., 1.)):
        torch.testing.assert_close(baseline.positive[key], torch.full((2,), expected))
    _assert_only_completion_changed(actual, baseline, torch.tensor([.25, .5]))
    assert actual.gates['jump_missed'].all()


@pytest.mark.parametrize('requested,lower', [(.03, .015), (.04, .02), (.06, .03), (.2, .17), (.4, .37)])
def test_completion_uses_native_target_tolerance_for_small_and_large_requests(requested, lower):
    progress = _progress(peak=(lower * .25, lower * .75), requested=(requested, requested))
    _assert_only_completion_changed(_evaluate(progress), _evaluate(progress, enabled=False),
                                    torch.tensor([.25, .75]))


def test_supported_peak_without_real_flight_does_not_complete_hold():
    progress = _progress(started=(False, False), closed=(False, False), peak=(.04, .3))
    actual = _evaluate(progress)
    old_missed = _evaluate(progress, enabled=False)
    for key in TRACKING:
        assert not actual.positive[key].any()
    torch.testing.assert_close(actual.cost['jump_missed'], torch.ones(2))
    for key, value in old_missed.components.items():
        torch.testing.assert_close(actual.components[key], value, atol=0., rtol=0.)


def test_first_real_touchdown_gates_small_hop_before_reference_hold():
    progress = _progress(stage='BALLISTIC', closed=(True, False), peak=(.085, .085))
    actual = _evaluate(progress)
    baseline = _evaluate(progress, enabled=False)
    _assert_only_completion_changed(actual, baseline, torch.tensor([.5, 1.]))
    assert actual.gates['jump_missed'].tolist() == [True, False]


@pytest.mark.parametrize('stage', ['COMPRESSION', 'EXTENSION', 'BALLISTIC'])
def test_takeoff_shaping_remains_before_hold_and_before_first_touchdown(stage):
    progress = _progress(stage=stage, started=(False, True), closed=(False, False), peak=(0., .025))
    actual = _evaluate(progress, case='jump_takeoff')
    baseline = _evaluate(progress, enabled=False, case='jump_takeoff')
    for key, value in baseline.components.items():
        torch.testing.assert_close(actual.components[key], value, atol=0., rtol=0.)
    assert not actual.gates['jump_missed'].any()


def test_late_real_jump_can_grow_completion_during_hold():
    progress = _progress(closed=(False, False), peak=(.085, .17))
    actual = _evaluate(progress, case='jump_flight')
    baseline = _evaluate(progress, enabled=False, case='jump_flight')
    _assert_only_completion_changed(actual, baseline, torch.tensor([.5, 1.]))


def test_at_target_and_overshoot_restore_tracking_without_new_overshoot_cost():
    progress = _progress(peak=(.2, .4))
    actual = _evaluate(progress)
    baseline = _evaluate(progress, enabled=False)
    for key, value in baseline.components.items():
        torch.testing.assert_close(actual.components[key], value, atol=0., rtol=0.)
    assert not actual.gates['jump_missed'].any()


@pytest.mark.parametrize('mode', [CommandMode.NORMAL, CommandMode.SLOPE, CommandMode.STAIR, CommandMode.RECOVERY])
def test_other_modes_and_normal_protocol_tail_keep_their_original_reward(mode):
    case = 'recovery' if mode == CommandMode.RECOVERY else 'ordinary'
    # NORMAL can retain a physically incomplete request after estimated handoff.
    progress = _progress(peak=(.025, .04), requested=(.2, .2))
    actual = _evaluate(progress, case=case, mode=mode)
    baseline = _evaluate(progress, enabled=False, case=case, mode=mode)
    for key, value in baseline.components.items():
        torch.testing.assert_close(actual.components[key], value, atol=0., rtol=0.)
    assert not actual.gates['jump_missed'].any()


def test_inactive_rows_may_have_zero_requested_rise():
    progress = _progress(started=(False, False), closed=(False, False), peak=(0., 0.), requested=(0., 0.))
    actual = _evaluate(progress, case='ordinary')
    baseline = _evaluate(progress, enabled=False, case='ordinary')
    for key, value in baseline.components.items():
        torch.testing.assert_close(actual.components[key], value, atol=0., rtol=0.)
    assert not actual.invalid.any()


def test_disabled_default_keeps_old_progress_contract_and_exact_values():
    assert RewardExtensions().jump_peak_completion is False
    old_progress = replace(_progress(started=(False, True)), peak_rise=None, requested_rise=None)
    enriched_progress = replace(old_progress, peak_rise=torch.tensor([.04, .04]),
                                 requested_rise=torch.tensor([.2, .2]))
    baseline = _evaluate(old_progress, enabled=False)
    actual = _evaluate(enriched_progress, enabled=False)
    for key, value in baseline.components.items():
        torch.testing.assert_close(actual.components[key], value, atol=0., rtol=0.)


def test_peak_gradient_is_positive_until_completion_and_matches_finite_difference():
    peak = torch.tensor([.05, .1], dtype=torch.float64, requires_grad=True)

    def score(value):
        return _evaluate(_progress(peak=value)).total.sum()

    gradient, = torch.autograd.grad(score(peak), (peak,))
    assert (gradient > 0.).all() and torch.isfinite(gradient).all()
    direction = torch.tensor([.3, -.2], dtype=torch.float64)
    epsilon = 1e-5
    numeric = (score(peak + epsilon * direction) - score(peak - epsilon * direction)) / (2. * epsilon)
    torch.testing.assert_close((gradient * direction).sum(), numeric, atol=1e-7, rtol=1e-7)
    # Saturation adds no growing reward or new penalty beyond the completion threshold.
    overshoot = torch.tensor([.2, .4], dtype=torch.float64, requires_grad=True)
    saturated, = torch.autograd.grad(score(overshoot), (overshoot,))
    assert not saturated.any()


def test_no_peak_gradient_before_real_flight_or_before_late_window():
    peak = torch.tensor([.05, .1], dtype=torch.float64, requires_grad=True)
    no_flight = _progress(started=(False, False), closed=(False, False), peak=peak)
    early = _progress(stage='EXTENSION', closed=(False, False), peak=peak)
    for progress in (no_flight, early):
        gradient, = torch.autograd.grad(_evaluate(progress).total.sum(), (peak,))
        assert not gradient.any()
