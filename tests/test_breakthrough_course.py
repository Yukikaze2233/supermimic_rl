"""Task quotas and real terrain placement for the PPO breakthrough continuation."""
from collections import Counter
from copy import deepcopy
import json

import pytest
import torch

from transformer_rl.formal.tasks.curriculum import TaskFamily
from transformer_rl.formal.tasks.geometry import TerrainKind
from supermimic_rl.course.curriculum import CourseCurriculumConfig
from supermimic_rl.course.jump_first import make_course_curriculum
from test_foundation_authored_rehearsal import ROOT, atlas, router, route, assert_draw_equal


COORDINATE = 1_525_481_472  # Actual PPO coordinate at update 23038.
WEIGHTS = {'FLAT': 1, 'SLOPE': 2, 'ROUGH': 5, 'STEP_UP': 8, 'STEP_DOWN': 3, 'MATERIAL': 1}


def configuration(version='v10-breakthrough'):
    return json.loads((ROOT / f'configs/gpufree/ppo-course-gated-{version}.json').read_text())


def course(value=None, *, n=1200, seed=7, coordinate=COORDINATE):
    result = make_course_curriculum(n, 'cpu', seed, configuration() if value is None else value)
    result.set_progress(coordinate, 23038)
    result.set_opened_gates(None)
    return result


def test_v10_changes_only_three_mixtures_and_named_foundation_quotas():
    before = configuration('v9-authored-half')
    after = configuration()
    expected = deepcopy(before)
    for index in (3, 4, 7):
        expected['base']['stage_weights'][index] = [.02, .15, .03, .35, .25, .20]
    expected['base']['foundation_terrain_kind_weights'] = WEIGHTS
    assert after == expected
    current = course(after)
    assert current.stage_name == 'terrain_foundation'
    assert current.command_context().domain.jump_m[1] == .4
    assert current.weights == pytest.approx((.02, .15, .03, .35, .25, .20))
    assert len(current.recovery_birth_samples()) == 22
    assert current.recovery_posture_pool() == course(before).recovery_posture_pool()


@pytest.mark.parametrize('weights', [None, [], {'FLAT': 1}, {**WEIGHTS, 'FLAT': 0},
                                   {**WEIGHTS, 'FLAT': 1.5}, {**WEIGHTS, 'FLAT': True},
                                   {**WEIGHTS, 'UNKNOWN': 1}])
def test_invalid_named_quotas_are_rejected(weights):
    with pytest.raises(ValueError, match='positive integer quota'):
        CourseCurriculumConfig(foundation_terrain_kind_weights=weights)


def test_default_identity_and_explicit_uniform_cycle_match_old_draws():
    before = configuration('v9-authored-half')
    parsed = CourseCurriculumConfig.from_dict(before['base'])
    assert 'foundation_terrain_kind_weights' not in parsed.to_dict()
    assert parsed.to_dict() == before['base']
    after = deepcopy(before)
    after['base']['foundation_terrain_kind_weights'] = {name: 1 for name in WEIGHTS}
    left, right = course(before), course(after)
    for _ in range(3):
        assert_draw_equal(left.sample(torch.arange(1200)), right.sample(torch.arange(1200)))
        assert torch.equal(left.sampler.generator.get_state(), right.sampler.generator.get_state())


def test_actual_weighted_terrain_quotas_and_alternating_authored_assets(atlas):
    value = course()
    placement = router(atlas, value)
    terrain, selected_assets = [], []
    for _ in range(20):
        ids = torch.arange(1200)
        draw = value.sample(ids)
        chosen = value.foundation_authored_rehearsal_rows(ids)
        route(placement, value, ids, draw)
        terrain.extend(draw.terrain_kind[draw.family == int(TaskFamily.TERRAIN)].tolist())
        selected_assets.extend(placement.tile[ids][chosen].tolist())
    # The 20-slot weighted cycle has eight STEP_UP slots, four kept generic
    # and four assigned to the two actual assets, alternating within the kind.
    complete = len(terrain) // 20
    assert Counter(terrain[:20 * complete]) == Counter({
        int(TerrainKind.FLAT): complete, int(TerrainKind.SLOPE): 2 * complete,
        int(TerrainKind.ROUGH): 5 * complete, int(TerrainKind.STEP_UP): 6 * complete,
        int(TerrainKind.STEP_DOWN): 3 * complete, int(TerrainKind.MATERIAL): complete,
        int(TerrainKind.TWO_PLATFORMS): 2 * complete})
    assert selected_assets == [101, 102] * (len(selected_assets) // 2) + ([101] if len(selected_assets) % 2 else [])
    assert abs(len(selected_assets) / 24000 - .07) < .001
    assert value.foundation_authored_rehearsal_receipt()['terrain_kind_weights'] == WEIGHTS
    assert abs(value.sample_counts['jump_motion'][0] - value.sample_counts['jump_motion'][1]) <= 1


def test_weighting_consumes_no_rng_and_preserves_other_family_draws():
    weighted = configuration()
    uniform = deepcopy(weighted)
    uniform['base'].pop('foundation_terrain_kind_weights')
    left, right = course(uniform), course(weighted)
    for _ in range(8):
        a, b = left.sample(torch.arange(1200)), right.sample(torch.arange(1200))
        assert torch.equal(a.family, b.family)
        assert_draw_equal(a, b, a.family != int(TaskFamily.TERRAIN))
        assert torch.equal(left.sampler.generator.get_state(), right.sampler.generator.get_state())
        for key in ('family', 'recovery_posture', 'jump_motion'):
            assert left.sample_counts[key] == right.sample_counts[key]


def test_weighted_restore_retains_partial_kind_cycle_and_real_routing(atlas):
    first = course(n=256)
    first_router = router(atlas, first)
    for ids in (torch.arange(31), torch.tensor([9]), torch.arange(193)):
        route(first_router, first, ids, first.sample(ids))
    second = course(n=256, seed=777)
    second.load_state_dict(deepcopy(first.state_dict()))
    second_router = router(atlas, second)
    second_router.load_state_dict(deepcopy(first_router.state_dict()))
    for ids in (torch.tensor([4]), torch.arange(211), torch.tensor([8, 70]), torch.arange(223)):
        a, b = first.sample(ids), second.sample(ids)
        assert_draw_equal(a, b)
        route(first_router, first, ids, a)
        route(second_router, second, ids, b)
        assert torch.equal(first_router.tile, second_router.tile)
        assert first.sample_counts == second.sample_counts
        assert torch.equal(first.sampler.generator.get_state(), second.sampler.generator.get_state())
        assert torch.equal(first_router.generator.get_state(), second_router.generator.get_state())


def test_full_terrain_stages_keep_existing_coverage_and_endpoint_rules():
    weighted = configuration()
    uniform = deepcopy(weighted)
    uniform['base'].pop('foundation_terrain_kind_weights')
    template = course()
    for stage in ('authored_steps', 'full_recovery', 'combined_dr'):
        index = template.stage_names.index(stage)
        coordinate = sum(template.stage_budgets[:index]) + template.stage_budgets[index] // 2
        a, b = course(uniform, coordinate=coordinate), course(weighted, coordinate=coordinate)
        assert_draw_equal(a.sample(torch.arange(1200)), b.sample(torch.arange(1200)))
        assert a.command_context().domain.jump_m[1] == .4
