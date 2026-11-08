"""Actual authored birth placement without advancing the mixed course clock."""
from copy import deepcopy
from dataclasses import fields, replace
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from transformer_rl.formal.lab_env.course_tasks import LabCourseTasks
from transformer_rl.formal.lab_env.formal_curriculum import LabTaskDrawSampler
from transformer_rl.formal.lab_env.terrain import LabTerrainCourse
from transformer_rl.formal.tasks.command_pool import CommandPoolConfig, SegmentKind
from transformer_rl.formal.tasks.curriculum import TaskFamily
from transformer_rl.formal.tasks.terrain_courses import TerrainCourseRouter
from transformer_rl.formal.tasks.trajectory import DriveEnvelope

from supermimic_rl.course.curriculum import CourseCurriculumConfig
from supermimic_rl.course.jump_first import JumpFirstCommandPool, make_course_curriculum
from supermimic_rl.course.routing import AuthoredEndpointRouter, install_authored_routing
from supermimic_rl.world_model_v2.migrations.course import verify_course_sampling


ROOT = Path(__file__).resolve().parents[1]
COORDINATE = 1_121_746_944  # Recorded PPO foundation coordinate, not a new schedule.
ATLAS_SHA = '8e34c6e3cb90ec48643f3d74d124c4eaaa829ff39f6e37bd11405023da08a594'


def configuration(enabled, every=None):
    value = json.loads((ROOT / 'configs/gpufree/ppo-course-gated-v7.json').read_text())
    if enabled:
        value['base']['foundation_authored_rehearsal'] = True
    if every is not None:
        value['base']['foundation_authored_rehearsal_every'] = every
    return value


def curriculum(enabled=True, n=4096, seed=7, coordinate=COORDINATE, every=None):
    result = make_course_curriculum(n, 'cpu', seed, configuration(enabled, every))
    result.set_progress(coordinate, 18924)
    return result


@pytest.fixture(scope='module')
def atlas():
    path = Path(os.environ.get('SUPERMIMIC_TEST_ATLAS', ROOT / 'assets/rmuc2027_course_atlas/manifest.json'))
    if not path.is_file():
        pytest.skip('The authenticated course atlas is required for actual-placement checks')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == ATLAS_SHA
    return path


def router(atlas, course, *, enabled=True, seed=31):
    return AuthoredEndpointRouter(TerrainCourseRouter(atlas, course.num_envs, seed=seed),
                                 rehearsal_curriculum=course if enabled else None)


def route(value, course, ids, draw):
    return value.reset(ids, draw.terrain_kind, stage=course.stage,
                       command_forward=draw.requested_command[:, 0],
                       requested={name: getattr(draw, name) for name in
                                  ('slope_rad', 'roughness_m', 'step_up_m', 'step_down_m')},
                       level_cap=course.terrain_level_cap(value.manifest['levels']),
                       step_up_cap_m=course.terrain_step_up_cap_m())


def assert_draw_equal(left, right, mask=None, except_fields=()):
    for field in fields(left):
        if field.name in except_fields:
            continue
        a, b = getattr(left, field.name), getattr(right, field.name)
        assert torch.equal(a if mask is None else a[mask], b if mask is None else b[mask]), field.name


def test_disabled_configuration_and_router_identity_remain_legacy(atlas):
    course = curriculum(False, n=32)
    state = course.state_dict()
    assert 'foundation_authored_rehearsal' not in state['config']
    assert state['config'] == configuration(False)['base']
    current = router(atlas, course, enabled=False)
    assert 'foundation_rehearsal' not in current.receipt()
    assert 'foundation_rehearsal' not in current.state_dict()
    restored = curriculum(False, n=32, seed=999)
    restored.load_state_dict(state)
    assert_draw_equal(course.sample(torch.arange(32)), restored.sample(torch.arange(32)))
    with pytest.raises(ValueError, match='boolean'):
        CourseCurriculumConfig(foundation_authored_rehearsal=1)


@pytest.mark.parametrize('every', [5, 2])
def test_named_assets_receive_the_derived_quota_without_changing_other_routes(atlas, every):
    old, new = curriculum(False, n=1200), curriculum(True, n=1200, every=every)
    before, after = router(atlas, old, enabled=False), router(atlas, new)
    ids = torch.arange(1200)
    selected_count = terrain_count = 0
    for _ in range(12):
        left, right = old.sample(ids), new.sample(ids)
        chosen = new.foundation_authored_rehearsal_rows(ids)
        assert (left.terrain_kind[chosen] == 3).all()
        assert (right.family[chosen] == int(TaskFamily.TERRAIN)).all()
        assert (right.step_up_m[chosen] == .2).all()
        assert_draw_equal(left, right, except_fields=('terrain_kind', 'step_up_m'))
        assert_draw_equal(left, right, ~chosen)
        route(before, old, ids, left)
        origins, spawn = route(after, new, ids, right)
        expected = torch.where(right.terrain_kind[chosen] == 3, 101, 102)
        assert torch.equal(after.tile[ids][chosen], expected)
        assert torch.equal(after.kinds[expected], right.terrain_kind[chosen])
        assert torch.equal(after.tile[ids][~chosen], before.tile[ids][~chosen])
        assert torch.equal(after.origins, after.origins_table[after.tile])
        assert torch.equal(origins, after.origins[ids])
        assert torch.equal(spawn, after.spawn_z[after.tile[ids]])
        assert torch.equal(after.level, before.level)
        assert torch.equal(after.generator.get_state(), before.generator.get_state())
        assert torch.equal(new.sampler.generator.get_state(), old.sampler.generator.get_state())
        assert new.command_context() == old.command_context()
        selected_count += int(chosen.sum())
        terrain_count += int((right.family == int(TaskFamily.TERRAIN)).sum())
    counts = after.actual_metrics()['foundation_rehearsal_reset_counts']
    assert sum(counts.values()) == selected_count
    assert abs(counts['road_step'] - counts['central_steps']) <= 1
    assert abs(selected_count - terrain_count / 6 / every) <= 1
    assert abs(selected_count / (12 * 1200) - .3 / 6 / every) < .0002
    assert after.receipt()['foundation_rehearsal']['slot_period'] == every
    assert new.terrain_level_cap(10) == old.terrain_level_cap(10) == 1
    assert new.terrain_step_up_cap_m() == old.terrain_step_up_cap_m() < .072
    assert new.sample_counts['family'] == old.sample_counts['family']
    assert new.sample_counts['recovery_posture'] == old.sample_counts['recovery_posture']


@pytest.mark.parametrize('coordinate', [800_000_000, 1_867_776_000, 2_024_686_819])
@pytest.mark.parametrize('every', [5, 2])
def test_other_stages_including_current_world_course_are_identical(atlas, coordinate, every):
    old, new = [curriculum(enabled, n=256, coordinate=coordinate, every=every) for enabled in (False, True)]
    left, right = old.sample(torch.arange(256)), new.sample(torch.arange(256))
    assert_draw_equal(left, right)
    assert not new.foundation_authored_rehearsal_rows(torch.arange(256)).any()
    before, after = router(atlas, old, enabled=False), router(atlas, new)
    route(before, old, torch.arange(256), left)
    route(after, new, torch.arange(256), right)
    assert torch.equal(before.tile, after.tile)
    assert torch.equal(before.generator.get_state(), after.generator.get_state())


@pytest.mark.parametrize('every', [5, 2])
def test_restore_retains_slot_phase_and_routing_rng_across_irregular_batches(atlas, every):
    original = curriculum(n=256, every=every)
    first = router(atlas, original)
    for ids in (torch.tensor([3]), torch.arange(83), torch.tensor([1, 97, 45]), torch.arange(191)):
        route(first, original, ids, original.sample(ids))
    state, routing = deepcopy(original.state_dict()), deepcopy(first.state_dict())
    restored = curriculum(n=256, seed=981, every=every)
    restored.load_state_dict(state)
    assert not restored.foundation_authored_rehearsal_rows(torch.arange(256)).any()
    second = router(atlas, restored, seed=777)
    second.load_state_dict(routing)
    for ids in (torch.arange(211), torch.tensor([9]), torch.arange(157), torch.tensor([250, 101, 0])):
        left, right = original.sample(ids), restored.sample(ids)
        assert_draw_equal(left, right)
        assert torch.equal(original.foundation_authored_rehearsal_rows(ids),
                           restored.foundation_authored_rehearsal_rows(ids))
        route(first, original, ids, left)
        route(second, restored, ids, right)
        assert torch.equal(first.tile, second.tile)
        assert first.actual_metrics() == second.actual_metrics()
        assert torch.equal(first.generator.get_state(), second.generator.get_state())


@pytest.mark.parametrize('every', [5, 2])
def test_single_row_sampling_keeps_balanced_authored_rotation(every):
    course = curriculum(n=1, every=every)
    selected = []
    for _ in range(1200):
        draw = course.sample([0])
        if course.foundation_authored_rehearsal_rows([0]).item():
            selected.append(int(draw.terrain_kind[0]))
    assert len(selected) >= 10
    assert selected == [3, 7] * (len(selected) // 2) + ([3] if len(selected) % 2 else [])


def test_router_does_not_infer_rehearsal_from_a_high_request(atlas):
    course = curriculum(n=32)
    value = router(atlas, course)
    # No certified sampler draw: a high request must not enable the new override.
    draw = curriculum(False, n=32).sample(torch.arange(32))
    draw = replace(draw, terrain_kind=torch.full_like(draw.terrain_kind, 3),
                   step_up_m=torch.full_like(draw.step_up_m, .2))
    route(value, course, torch.arange(32), draw)
    assert sum(value.actual_metrics()['foundation_rehearsal_reset_counts'].values()) == 0
    assert not value._last_forced.any()
    # A real selected row cannot be routed while still recording a low request.
    ids = torch.arange(32)
    for _ in range(4):
        draw = course.sample(ids)
        if course.foundation_authored_rehearsal_rows(ids).any():
            break
    assert course.foundation_authored_rehearsal_rows(ids).any()
    with pytest.raises(ValueError, match='actual kind and endpoint'):
        route(value, course, ids, replace(draw, step_up_m=torch.zeros_like(draw.step_up_m)))


def reset_command_host(tasks, route_state):
    """Light shell around the actual native reset_commands method, without physics."""
    n = tasks.num_envs
    env = SimpleNamespace(num_envs=n, _device=torch.device('cpu'),
                          _clock=lambda: torch.zeros(n, dtype=torch.float64))
    for name in ('_command', '_command_base'):
        setattr(env, name, torch.zeros(n, 4))
    for name in ('_command_mode', '_base_command_mode'):
        setattr(env, name, torch.zeros(n, dtype=torch.long))
    env._request_height = torch.zeros(n)
    no_op = SimpleNamespace(reset=lambda *args: None)
    host = SimpleNamespace(env=env, router=route_state, tasks=tasks,
                           _record_stair_end=lambda *args: None, stair_evidence=no_op,
                           support_reference=no_op, _support_sample=None, recovery=no_op,
                           stair_requests=no_op, tof_trigger=no_op, _pending_requests={},
                           _birth_family_counts=torch.zeros(6, dtype=torch.long))
    for name in ('stair_next_id', 'manual_stair_request_id', 'stair_cancel_id', 'tof_request_id',
                 '_last_stair_response'):
        setattr(host, name, torch.zeros(n, dtype=torch.long))
    for name in ('_stair_course_contact_allowed', 'body_contact_allowed', 'manual_stair_request'):
        setattr(host, name, torch.zeros(n, dtype=torch.bool))
    for name in ('manual_stair_height_m', 'tof_step_height_m', '_stair_request_height'):
        setattr(host, name, torch.zeros(n))
    return host


def test_native_commands_use_actual_authored_height_and_existing_speed_mixture(atlas):
    course = curriculum()
    draw = course.sample(torch.arange(course.num_envs))
    route_state = router(atlas, course)
    route(route_state, course, torch.arange(course.num_envs), draw)
    selected = course.foundation_authored_rehearsal_rows(torch.arange(course.num_envs))
    subset = replace(draw, **{f.name: getattr(draw, f.name)[selected] for f in fields(draw)})
    task = LabTaskDrawSampler(course).map_draw(subset)
    n = len(subset.family)
    sampled_router = SimpleNamespace(tile=route_state.tile[selected], kinds=route_state.kinds,
        step_table=route_state.step_table, origins=route_state.origins[selected])
    authored_course = curriculum(False, n=n, coordinate=1_867_776_000)
    config = CommandPoolConfig(high_step_low_speed_fraction=.5)  # Pinned runtime value.
    owners = []
    for context in (course.command_context, authored_course.command_context):
        tasks = LabCourseTasks(n, DriveEnvelope(.06, .34), seed=31, policy_dt_s=.02,
            command_pool=JumpFirstCommandPool(config), command_context=context, terrain_mode_labels=False)
        host = reset_command_host(tasks, sampled_router)
        LabTerrainCourse.reset_commands(host, torch.arange(n), task)
        owners.append(tasks)
    left, right = owners
    assert (left.terrain_step_height_m == .2).all()
    assert torch.equal(left.terrain_kind, subset.terrain_kind)
    assert set(left.terrain_kind.tolist()) == {3, 7}
    assert torch.equal(left.trajectory.end, right.trajectory.end)
    assert torch.equal(left.segment_kind, right.segment_kind)
    assert torch.equal(left.generator.get_state(), right.generator.get_state())
    normal = left.segment_kind == int(SegmentKind.NORMAL)
    speed = left.trajectory.end[normal, 0].abs()
    assert (speed < config.high_approach_forward_m_s[0]).any()
    assert (speed >= config.high_approach_forward_m_s[0]).any()
    assert speed.min() >= config.approach_forward_m_s[0]
    assert speed.max() <= config.high_approach_forward_m_s[1]
    assert (left.trajectory.end[:, 3] >= course.domain.height_m[0]).all()
    assert (left.trajectory.end[:, 3] <= config.approach_height_m[1]).all()
    stand = left.segment_kind == int(SegmentKind.STAND)
    assert not left.trajectory.end[stand, :3].count_nonzero()


def test_opt_in_requires_a_distinct_sampling_identity_and_router_contract(atlas):
    old, new = configuration(False), configuration(True)

    def source(value):
        raw = json.dumps(value).encode()
        return {'course_sha256': hashlib.sha256(raw).hexdigest(), 'curriculum': value,
                'course_schedule': {'schedule': value['schedule'],
                                    'early_jump_domain': value['base']['initial_domain']['jump_m']}}, raw

    before, old_bytes = source(old)
    after, new_bytes = source(new)
    with pytest.raises(ValueError, match='Only task mixtures'):
        verify_course_sampling(before, after, source_course=old_bytes, target_course=new_bytes)
    off, on = curriculum(False, n=32), curriculum(True, n=32)
    with pytest.raises(ValueError, match='identity/configuration'):
        on.load_state_dict(off.state_dict())
    plain, enabled = router(atlas, off, enabled=False), router(atlas, on)
    with pytest.raises(ValueError, match='contract differs'):
        enabled.load_state_dict(plain.state_dict())
    env = SimpleNamespace(_terrain_course=SimpleNamespace(router=TerrainCourseRouter(atlas, 32)))
    installed = install_authored_routing(env, rehearsal_curriculum=on)
    assert installed is install_authored_routing(env, rehearsal_curriculum=on)
    assert env.authored_routing_receipt['foundation_rehearsal']['slot_period'] == 5


def test_default_period_preserves_enabled_legacy_identity_and_draws():
    old, explicit = curriculum(n=256), curriculum(n=256, every=5)
    assert old.config.to_dict() == explicit.config.to_dict() == configuration(True)['base']
    assert 'foundation_authored_rehearsal_every' not in old.state_dict()['config']
    assert old.foundation_authored_rehearsal_receipt() == explicit.foundation_authored_rehearsal_receipt()
    for ids in (torch.arange(241), torch.tensor([3]), torch.arange(109)):
        assert_draw_equal(old.sample(ids), explicit.sample(ids))
    explicit.load_state_dict(old.state_dict())
    assert_draw_equal(old.sample(torch.arange(256)), explicit.sample(torch.arange(256)))


@pytest.mark.parametrize('value', [0, -1, True, 2.5, '2', None])
def test_rehearsal_period_requires_a_positive_integer(value):
    with pytest.raises(ValueError, match='positive integer'):
        CourseCurriculumConfig(foundation_authored_rehearsal_every=value)


def test_candidate_changes_only_rehearsal_frequency_and_preserves_other_skills():
    before = json.loads((ROOT / 'configs/gpufree/ppo-course-gated-v8-early-authored.json').read_text())
    after = json.loads((ROOT / 'configs/gpufree/ppo-course-gated-v9-authored-half.json').read_text())
    assert after == {**before, 'base': {**before['base'], 'foundation_authored_rehearsal_every': 2}}
    old, new = [make_course_curriculum(1200, 'cpu', 7, value) for value in (before, after)]
    for course in (old, new):
        course.set_progress(COORDINATE, 18924)
        course.set_opened_gates(None)  # Existing schedule progression.
        assert course.command_context().domain.jump_m[1] == .4
    selected = {5: [], 2: []}
    for count in (1200, 83, 1, 191, 1200, 900, 157, 3, 1200, 1200):
        ids = torch.arange(count)
        left, right = old.sample(ids), new.sample(ids)
        a, b = [course.foundation_authored_rehearsal_rows(ids) for course in (old, new)]
        assert_draw_equal(left, right, except_fields=('terrain_kind', 'step_up_m'))
        assert_draw_equal(left, right, ~(a | b))
        for every, draw, chosen in ((5, left, a), (2, right, b)):
            selected[every].extend(draw.terrain_kind[chosen].tolist())
        assert old.sample_counts['family'] == new.sample_counts['family']
        assert old.sample_counts['recovery_posture'] == new.sample_counts['recovery_posture']
        assert old.sample_counts['jump_motion'] == new.sample_counts['jump_motion']
        assert torch.equal(old.sampler.generator.get_state(), new.sampler.generator.get_state())
    for values in selected.values():
        assert values == [3, 7] * (len(values) // 2) + ([3] if len(values) % 2 else [])
    assert len(new.sample_counts['recovery_posture']) == 22
    assert all(count > 0 for count in new.sample_counts['recovery_posture'])
    stationary, moving = new.sample_counts['jump_motion']
    assert abs(stationary - moving) <= 1
    assert len(selected[2]) > 2 * len(selected[5])
    with pytest.raises(ValueError, match='identity/configuration'):
        new.load_state_dict(old.state_dict())
