"""Endpoint-normalized tuck changes shaping, not first-flight ownership or bounds."""
from dataclasses import replace

import pytest
import torch

pytest.importorskip('transformer_rl.formal.tasks.reward')
from transformer_rl.formal.tasks.jump import ReferenceStage

from supermimic_rl.reward_kernel import RewardExtensions
from test_jump_flight_objective import evaluate, geometry


OLD = RewardExtensions(jump_flight=4., jump_flight_clearance=True, jump_tuck=1.7)
NEW = replace(OLD, jump_tuck_endpoint_normalized=True)


@pytest.mark.parametrize('fraction,expected', [(0., 1.7), (.5, 1.02), (1., 0.), (1.2, 0.)])
def test_source_endpoints_keep_the_same_maximum_budget(fraction, expected):
    base = geometry()
    length = (base.minimum_leg_length_m + fraction * base.leg_length_range_m).requires_grad_()
    result = evaluate(replace(base, leg_length_m=length), extensions=NEW)
    score = result.positive['jump_tuck']
    # The unchanged native formula multiplies weight by a bool gate first;
    # that scalar product has default float precision before geometry promotion.
    gate_weight = NEW.jump_tuck * torch.ones_like(score, dtype=torch.bool)
    expected_score = gate_weight.to(score) * (expected / NEW.jump_tuck)
    torch.testing.assert_close(score, expected_score, atol=1e-14, rtol=1e-14)
    assert (score >= 0).all() and (score <= gate_weight.to(score)).all()
    assert torch.isfinite(torch.autograd.grad(score.sum(), length)[0]).all()


def test_midpoint_has_twice_the_original_gradient_and_matches_finite_difference():
    base = geometry()
    length = (base.minimum_leg_length_m + .5 * base.leg_length_range_m).requires_grad_()
    old = evaluate(replace(base, leg_length_m=length), extensions=OLD).positive['jump_tuck'].sum()
    new = evaluate(replace(base, leg_length_m=length), extensions=NEW).positive['jump_tuck'].sum()
    old_gradient = torch.autograd.grad(old, length)[0]
    gradient = torch.autograd.grad(new, length)[0]
    torch.testing.assert_close(gradient, 2 * old_gradient, rtol=1e-14, atol=1e-14)
    assert (gradient < 0).all()
    direction = torch.tensor([[.1, -.2], [-.3, .4]], dtype=length.dtype)
    def objective(value):
        return evaluate(replace(base, leg_length_m=value), extensions=NEW).positive['jump_tuck'].sum()
    epsilon = 1e-6
    numeric = (objective(length+epsilon*direction)-objective(length-epsilon*direction))/(2*epsilon)
    torch.testing.assert_close((gradient*direction).sum(), numeric, atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize('case,ascent,closed,stage', [
    ('jump_takeoff', False, False, ReferenceStage.COMPRESSION),
    ('jump_takeoff', True, False, ReferenceStage.EXTENSION),
    ('jump_flight', False, False, ReferenceStage.BALLISTIC),
    ('jump_flight', True, True, ReferenceStage.HOLD),
    ('jump_landing', True, True, ReferenceStage.HOLD),
    ('ordinary', True, False, ReferenceStage.BALLISTIC),
    ('recovery', True, False, ReferenceStage.BALLISTIC),
])
def test_no_tuck_score_or_gradient_outside_first_real_flight(case, ascent, closed, stage):
    base = geometry()
    length = base.leg_length_m.clone().requires_grad_()
    result = evaluate(replace(base, leg_length_m=length), extensions=NEW,
                      case=case, ascent=ascent, closed=closed, stage=stage)
    score = result.positive['jump_tuck'].sum()
    assert score == 0
    assert not torch.autograd.grad(score, length)[0].any()


def test_unavailable_geometry_does_not_create_score_or_nan_gradient():
    base = geometry()
    length = base.leg_length_m.clone()
    length[0] = float('nan')
    length.requires_grad_()
    result = evaluate(replace(base, leg_length_m=length, valid=torch.tensor([False, True])), extensions=NEW)
    score = result.positive['jump_tuck']
    assert score[0] == 0 and score[1] > 0
    gradient = torch.autograd.grad(score.sum(), length)[0]
    assert torch.isfinite(gradient).all() and not gradient[0].any()
    assert not result.invalid.any()


def test_default_is_bitwise_legacy_and_enabled_only_changes_tuck():
    base = geometry()
    default = evaluate(base, extensions=OLD)
    explicit_false = evaluate(base, extensions=replace(OLD, jump_tuck_endpoint_normalized=False))
    enabled = evaluate(base, extensions=NEW)
    legacy_formula = OLD.jump_tuck * torch.ones(2, dtype=torch.bool) * (1+((base.leg_length_m-base.minimum_leg_length_m).clamp_min(0)
                                         /base.leg_length_range_m).square()).reciprocal().mean(-1)
    assert torch.equal(default.positive['jump_tuck'], legacy_formula)
    for name in default.components:
        assert torch.equal(default.components[name], explicit_false.components[name])
        if name != 'positive.jump_tuck':
            assert torch.equal(default.components[name], enabled.components[name])
    for name in default.gates:
        assert torch.equal(default.gates[name], enabled.gates[name])
    assert torch.equal(default.invalid, enabled.invalid)
    assert OLD.receipt() == replace(OLD, jump_tuck_endpoint_normalized=False).receipt()
    assert 'jump_tuck_endpoint_normalized' not in OLD.receipt()
    assert OLD.tuck_objective_receipt() is None
    assert RewardExtensions(**NEW.receipt()) == NEW
    assert NEW.tuck_objective_receipt()['geometry_source'] == 'reward_kernel.jump_flight_geometry'


@pytest.mark.parametrize('kwargs', [
    {'jump_tuck_endpoint_normalized': True},
    {'jump_tuck': 0., 'jump_tuck_endpoint_normalized': True},
    {'jump_tuck': 1.7, 'jump_tuck_endpoint_normalized': 1},
])
def test_normalization_requires_a_positive_tuck_weight_and_boolean_opt_in(kwargs):
    with pytest.raises(ValueError, match='Endpoint-normalized tuck'):
        RewardExtensions(**kwargs)
