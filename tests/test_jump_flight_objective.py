"""Geometric jump shaping through the shared native/imagination reward kernel."""
from dataclasses import replace

import pytest
import torch

pytest.importorskip('transformer_rl.formal.tasks.reward')
from transformer_rl.formal.tasks.jump import ReferenceStage

from supermimic_rl.reward_kernel import (
    JumpFlightGeometry, JumpProgress, RewardExtensions, RewardHistory,
    RewardParameters, physical_reward_densities,
)
from test_reward_kernel import begin, sample, setup


def geometry(clearance=None, length=None, valid=None):
    # Source 08bf's table-endpoint virtual lengths; these are test data, not
    # runtime defaults or a newly prescribed mechanical range.
    minimum = torch.tensor([[.162463215903925, .162463215766070]], dtype=torch.float64).repeat(2, 1)
    span = torch.tensor([[.208367168830814, .208367168939429]], dtype=torch.float64).repeat(2, 1)
    return JumpFlightGeometry(
        torch.tensor([[.003, .004], [.05, .06]], dtype=torch.float64) if clearance is None else clearance,
        minimum + .7 * span if length is None else length, minimum, span,
        torch.ones(2, dtype=torch.bool) if valid is None else valid,
    )


def evaluate(value, *, case='jump_flight', extensions=None, ascent=True, closed=False,
             stage=ReferenceStage.BALLISTIC, root_z=None):
    composer, reference = setup(case)
    begin(composer, reference, torch.zeros(2, 6))
    physics = sample(composer, 1)
    if root_z is not None:
        physics = replace(physics, root_world_z_m=root_z)
    parameters = replace(RewardParameters.from_composer(composer), extensions=(extensions or
                         RewardExtensions(jump_flight=4., jump_flight_clearance=True, jump_tuck=1.7)))
    history = RewardHistory(composer._window['dq'], torch.zeros(2), torch.zeros(2), torch.zeros(2))
    progress = JumpProgress(
        torch.full((2,), int(stage)), torch.full((2,), ascent), torch.full((2,), closed),
        flight_started=torch.full((2,), ascent), peak_rise=torch.full((2,), .1),
        requested_rise=torch.full((2,), .1), flight_geometry=value,
    )
    return physical_reward_densities(physics, reference, history, parameters, jump_progress=progress)


def test_flight_values_effective_clearance_instead_of_root_only_peak():
    base = geometry()
    result = evaluate(base)
    assert result.positive['jump_flight'][1] > result.positive['jump_flight'][0] > 0
    assert (result.positive['jump_flight'] < 4.).all()
    raised_root = evaluate(base, root_z=torch.tensor([.7, .8]))
    torch.testing.assert_close(result.positive['jump_flight'], raised_root.positive['jump_flight'])
    # One well-cleared wheel does not hide its partner's small clearance.
    one_high = evaluate(replace(base, wheel_clearance_m=torch.tensor(
        [[.003, .25], [.05, .25]], dtype=torch.float64)))
    torch.testing.assert_close(result.positive['jump_flight'], one_high.positive['jump_flight'])
    assert not result.invalid.any()


@pytest.mark.parametrize('case,ascent,closed,stage', [
    ('jump_takeoff', False, False, ReferenceStage.COMPRESSION),
    ('jump_takeoff', True, False, ReferenceStage.EXTENSION),
    ('jump_flight', False, False, ReferenceStage.BALLISTIC),
    ('jump_flight', True, True, ReferenceStage.HOLD),
    ('jump_landing', True, True, ReferenceStage.HOLD),
    ('ordinary', True, False, ReferenceStage.BALLISTIC),
    ('recovery', True, False, ReferenceStage.BALLISTIC),
])
def test_tuck_never_competes_with_compression_push_or_landing(case, ascent, closed, stage):
    base = geometry()
    length = base.leg_length_m.clone().requires_grad_()
    result = evaluate(replace(base, leg_length_m=length), case=case, ascent=ascent, closed=closed, stage=stage)
    score = result.positive['jump_flight'].sum() + result.positive['jump_tuck'].sum()
    assert score == 0
    torch.testing.assert_close(torch.autograd.grad(score, length)[0], torch.zeros_like(length))


def test_geometry_keeps_real_flight_rewards_after_the_reference_clock():
    normal = evaluate(geometry())
    late = evaluate(geometry(), stage=ReferenceStage.HOLD)
    for key in ('jump_flight', 'jump_tuck'):
        torch.testing.assert_close(normal.positive[key], late.positive[key])


def test_per_leg_tuck_is_source_normalized_and_not_a_fixed_foreign_length():
    base = geometry()
    full = evaluate(replace(base, leg_length_m=base.minimum_leg_length_m))
    torch.testing.assert_close(full.positive['jump_tuck'], torch.full((2,), 1.7, dtype=torch.float64))
    one_extended = base.minimum_leg_length_m.clone()
    one_extended[:, 1] += base.leg_length_range_m[:, 1]
    mixed = evaluate(replace(base, leg_length_m=one_extended))
    assert (mixed.positive['jump_tuck'] < full.positive['jump_tuck']).all()
    # A translated/scaled hypothetical source with the same fraction of leg
    # travel gets the same reward. Source lengths are not global metre knobs.
    shifted = replace(base, leg_length_m=2 * base.leg_length_m + .1,
                      minimum_leg_length_m=2 * base.minimum_leg_length_m + .1,
                      leg_length_range_m=2 * base.leg_length_range_m)
    torch.testing.assert_close(evaluate(base).positive['jump_tuck'], evaluate(shifted).positive['jump_tuck'])


def test_clearance_and_tuck_have_useful_finite_gradients_through_shared_kernel():
    base = geometry()
    coordinates = torch.cat((base.wheel_clearance_m, base.leg_length_m), -1).requires_grad_()

    def objective(value):
        result = evaluate(replace(base, wheel_clearance_m=value[:, :2], leg_length_m=value[:, 2:]))
        return (result.positive['jump_flight'] + result.positive['jump_tuck']).sum()

    gradient = torch.autograd.grad(objective(coordinates), coordinates)[0]
    assert torch.isfinite(gradient).all()
    assert (gradient[:, 0] > 0).all()  # raise the less-cleared wheel
    assert (gradient[:, 2:] < 0).all()  # shorten both virtual legs
    direction = torch.tensor([[.1, .2, -.3, -.2], [.2, .1, -.1, -.4]], dtype=torch.float64)
    epsilon = 1e-6
    numeric = (objective(coordinates + epsilon * direction)
               - objective(coordinates - epsilon * direction)) / (2 * epsilon)
    torch.testing.assert_close((gradient * direction).sum(), numeric, atol=1e-7, rtol=1e-6)


def test_unavailable_terrain_row_does_not_contaminate_other_gradient():
    base = geometry()
    clearance = base.wheel_clearance_m.clone()
    clearance[0] = float('nan')
    clearance.requires_grad_()
    result = evaluate(replace(base, wheel_clearance_m=clearance, valid=torch.tensor([False, True])))
    score = result.positive['jump_flight'] + result.positive['jump_tuck']
    assert score[0] == 0 and score[1] > 0
    assert result.gates['jump_flight_geometry_available'].tolist() == [False, True]
    assert not result.invalid.any()
    gradient = torch.autograd.grad(score.sum(), clearance)[0]
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient[0], torch.zeros_like(gradient[0]))


def test_bad_active_geometry_is_reported_by_existing_invalid_bit():
    base = geometry()
    bad_span = base.leg_length_range_m.clone()
    bad_span[0] = 0
    result = evaluate(replace(base, leg_length_range_m=bad_span))
    assert result.invalid.tolist() == [True, False]
    assert torch.isfinite(result.total).all()
    # The same unused geometry cannot invalidate an ordinary row.
    assert not evaluate(replace(base, leg_length_range_m=bad_span), case='ordinary').invalid.any()


def test_missing_producer_is_explicit_and_disabled_terms_keep_old_flight():
    with pytest.raises(ValueError, match='explicit flight geometry producer'):
        evaluate(None)
    default = evaluate(None, extensions=RewardExtensions(jump_flight=4.))
    torch.testing.assert_close(default.positive['jump_flight'], torch.full((2,), 4.))
    assert 'jump_tuck' not in default.positive
    assert 'jump_flight_geometry_available' not in default.gates
    assert evaluate(None, case='ordinary').positive['jump_tuck'].sum() == 0


def test_geometry_cat_and_substeps_preserve_gradients_and_progress_coverage():
    base = geometry()
    length = base.leg_length_m.clone().requires_grad_()
    base = replace(base, leg_length_m=length)
    joined = JumpFlightGeometry.cat([base] * 4)
    interval = joined.intervals(2)
    assert interval.tick(3).leg_length_m.shape == (2, 2)
    assert torch.autograd.grad(interval.tick(3).leg_length_m.sum(), length)[0].sum() == 4
    stage = torch.full((2,), int(ReferenceStage.BALLISTIC))
    progress = JumpProgress(stage, torch.ones(2, dtype=torch.bool), flight_geometry=base)
    combined = JumpProgress.cat([progress, progress])
    assert combined.flight_geometry.leg_length_m.shape == (4, 2)
    with pytest.raises(ValueError, match='coverage changes'):
        JumpProgress.cat([progress, replace(progress, flight_geometry=None)])


@pytest.mark.parametrize('kwargs', [dict(jump_tuck=-1.), dict(jump_tuck=float('nan')),
                                  dict(jump_flight_clearance=1), dict(jump_flight_clearance=True)])
def test_invalid_extension_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        RewardExtensions(**kwargs)
