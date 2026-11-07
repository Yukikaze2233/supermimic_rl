"""Actual-route plans and native partial-reset order, without Isaac or RNG draws."""
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from transformer_rl.formal.control.observation import CommandMode
from transformer_rl.formal.tasks.curriculum import TaskFamily

from supermimic_rl.course.platform_course import NativePlatformCourse, PlatformCoursePlan
from supermimic_rl.course.platform_intent import PlatformIntentPhase, PlatformIntentState
from supermimic_rl.course.platform_sampler import PlatformSamplerConfig

from test_native_platform_intent import setup, boundary, activate_stair
from test_platform_course import assembly, configuration


def rule(tile=101, *, families=('TERRAIN',), earliest=2.):
    return {'tile_id': tile, 'target_id': f'{tile}:profile:1', 'task_families': list(families),
            'earliest_episode_s': earliest, 'rise_m': .2, 'support_height_m': .32}


def sampler_configuration(tmp_path, rules=None):
    config = configuration(tmp_path)
    path = tmp_path / 'catalog.json'
    catalog = json.loads(path.read_text())
    for target in catalog['targets']:
        target['tile_id'] += 101
    for face in catalog['face_catalog'].values():
        face['tile_id'] += 101
    path.write_text(json.dumps(catalog))
    config['catalog_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    del config['rows']
    config['sampler'] = {'rules': [rule(), rule(102)] if rules is None else rules}
    return config


def read(config, count=3):
    return PlatformCoursePlan.read(config, num_envs=count, atlas_sha256='atlas-pin',
                                   record_substeps=True, jump_takeoff={})


def task(families, *, modes=None):
    count = len(families)
    return SimpleNamespace(
        mode=torch.zeros(count, dtype=torch.long) if modes is None else torch.tensor(modes),
        draw=SimpleNamespace(family=torch.tensor(families, dtype=torch.long),
            nominal=torch.zeros(count, dtype=torch.bool), trajectory_kind=torch.zeros(count, dtype=torch.long)),
        terrain_kind=torch.zeros(count, dtype=torch.long), request_height_m=torch.full((count,), .2),
        command=torch.tensor([[0., 0., 0., .32]]).expand(count, -1).clone())


def native_owner(tmp_path, *, count=3, rules=None):
    native, _, adapter, _ = setup((-1,) * count, install=False)
    owner = NativePlatformCourse.__new__(NativePlatformCourse)
    owner.native = native
    owner.plan = read(sampler_configuration(tmp_path, rules), count)
    owner._initial = owner.plan.state('cpu')
    owner._reset_counts = torch.zeros(count, dtype=torch.long)
    owner._birth_routes = torch.full((count, 2), -1, dtype=torch.long)
    owner.geometry = SimpleNamespace(tiles=torch.tensor([101, 102]))
    owner.task_adapter, owner.caller = adapter, None
    native._terrain_course.router = SimpleNamespace(tile=torch.zeros(count, dtype=torch.long))
    owner.install_caller()
    return native, owner


def reset(native, ids, tiles, families, *, modes=None):
    ids = torch.tensor(ids, dtype=torch.long)
    native._terrain_course.router.tile[ids] = torch.tensor(tiles)
    native._terrain_course.reset_commands(ids, task(families, modes=modes))


def test_actual_tile_and_explicit_family_only_without_random_draw_or_mutation(tmp_path):
    plan = read(sampler_configuration(tmp_path, [rule(), rule(102, families=('TERRAIN', 'MOTION'))]), 7)
    tile = torch.tensor([101, 102, 101, 102, 0, 101, 102])
    family = torch.tensor([3, 3, 1, 1, 3, 3, 5])
    mode = torch.tensor([0, 0, 0, 0, 0, int(CommandMode.RECOVERY), 0])
    before = [value.clone() for value in (tile, family, mode)]
    rng = torch.get_rng_state().clone()
    result = plan.sampler.select(tile, family, mode)
    assert result.target_index.tolist() == [0, 1, -1, 1, -1, -1, -1]
    assert torch.equal(torch.get_rng_state(), rng)
    for a, b in zip(before, (tile, family, mode)):
        assert torch.equal(a, b)
    # Partition and row order never affect a deterministic rule assignment.
    order = torch.tensor([6, 1, 3, 0, 4, 2, 5])
    other = plan.sampler.select(tile[order], family[order], mode[order])
    for name, value in result.record().items():
        assert torch.equal(value[order], other.record()[name])
    empty = plan.sampler.select(tile[:0], family[:0], mode[:0])
    assert len(empty.phase) == 0


@pytest.mark.parametrize('change', [
    {'tile_id': 100}, {'target_id': '102:profile:1'}, {'task_families': []},
    {'task_families': ['RECOVERY']}, {'task_families': ['terrain']},
    {'task_families': ['TERRAIN', 'TERRAIN']}, {'rise_m': 0}, {'support_height_m': None},
    {'earliest_episode_s': -1}, {'earliest_episode_s': True}, {'rise_m': float('nan')},
])
def test_rules_require_explicit_values_and_supported_nonconflicting_route(change):
    with pytest.raises(ValueError, match='Platform sampler'):
        PlatformSamplerConfig.read({'rules': [{**rule(), **change}]})


def test_overlap_missing_values_and_mixed_configuration_are_rejected(tmp_path):
    with pytest.raises(ValueError, match='overlap'):
        PlatformSamplerConfig.read({'rules': [rule(), rule(earliest=3.)]})
    incomplete = rule()
    del incomplete['support_height_m']
    with pytest.raises(ValueError):
        PlatformSamplerConfig.read({'rules': [incomplete]})
    config = sampler_configuration(tmp_path)
    with pytest.raises(ValueError, match='either explicit rows or sampler'):
        read({**config, 'rows': [None] * 3})
    path = tmp_path / 'catalog.json'
    catalog = json.loads(path.read_text())
    catalog['targets'][0]['tile_id'] = 77
    path.write_text(json.dumps(catalog))
    config['catalog_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='pinned target catalog'):
        read(config)


def test_native_reset_selects_actual_post_reset_route_without_changing_command_or_rng(tmp_path):
    native, owner = native_owner(tmp_path)
    plain, _, _, _ = setup((-1,) * 3)
    ids = torch.arange(3)
    draw = task([3, 3, 1])
    plain._terrain_course.reset_commands(ids, draw)
    reset(native, [0, 1, 2], [101, 102, 101], [3, 3, 1])
    tasks, original = native._terrain_course.tasks, plain._terrain_course.tasks
    assert torch.equal(tasks.generator.get_state(), original.generator.get_state())
    assert torch.equal(tasks.stream.current, original.stream.current)
    assert torch.equal(tasks.trajectory.start, original.trajectory.start)
    assert torch.equal(tasks.trajectory.end, original.trajectory.end)
    assert owner.caller.state.target_index.tolist() == [0, 1, -1]
    assert owner._birth_routes.tolist() == [[101, 3], [102, 3], [101, 1]]
    assert owner._reset_counts.tolist() == [1, 1, 1]
    assert owner.platform_rows.tolist() == [True, True, False]
    # Births are already physically placed; the producer never calls place_reset.
    assert tasks.elapsed.tolist() == [0., 0., 0.]


def test_partial_reset_does_not_replace_another_flight_or_physical_tail(tmp_path):
    native, owner = native_owner(tmp_path)
    reset(native, [0, 1, 2], [101, 102, 101], [3, 3, 3])
    state = owner.caller.state
    owner.caller.state = replace(state,
        phase=torch.tensor([3, 4, 5]), bound_request_id=torch.tensor([7, 8, 9]))
    tasks = native._terrain_course.tasks
    tasks.physical_id[:] = torch.tensor([7, 8, -1])
    tasks.protocol.max_seen_id[:] = torch.tensor([7, 8, 9])
    before = owner.caller.capture()
    reset(native, [2], [0], [3])
    assert owner.caller.state.target_index.tolist() == [0, 1, -1]
    assert owner.caller.state.bound_request_id.tolist() == [7, 8, -1]
    for name, value in before.items():
        assert torch.equal(value[:2], owner.caller.capture()[name][:2])
    assert tasks.physical_id[:2].tolist() == [7, 8]
    assert tasks.protocol.max_seen_id.tolist() == [7, 8, 9]
    reset(native, [2], [102], [3])
    assert owner.caller.state.target_index.tolist() == [0, 1, 1]
    assert owner.caller.state.phase[2] == PlatformIntentPhase.WAITING
    assert owner._reset_counts.tolist() == [1, 1, 3]


def test_sampled_plan_uses_existing_stair_cancel_idle_and_native_gate(tmp_path):
    native, owner = native_owner(tmp_path, rules=[rule(earliest=0.), rule(102, earliest=0.)])
    reset(native, [0, 1, 2], [101, 102, 0], [3, 3, 1])
    tasks = native._terrain_course.tasks
    tasks.elapsed.fill_(2.)
    tasks._last_supported.fill_(True)  # Completed physics evidence after birth.
    tasks._last_supported[1] = False
    activate_stair(native, torch.tensor([0]))
    boundary(native)
    assert owner.caller.state.phase.tolist() == [PlatformIntentPhase.PENDING] * 2 + [0]
    assert native._terrain_course.stair_requests.active_id[0] == -1
    assert tasks.physical_id.tolist() == [-1, -1, -1]
    boundary(native)
    assert tasks.physical_id[0] >= 0
    assert tasks.physical_id[1] == -1
    assert native.tof_ready[-1].tolist() == [False, False, True]
    assert owner.caller.state.target_index.tolist() == [0, 1, -1]
    assert owner._reset_counts.tolist() == [1, 1, 1]


def test_source_is_runtime_stable_and_cold_resume_never_restores_old_flight(tmp_path):
    native, owner = native_owner(tmp_path)
    receipt = owner.receipt()
    reset(native, [0, 1, 2], [101, 102, 0], [3, 3, 1])
    owner.caller.state = replace(owner.caller.state,
        phase=torch.tensor([3, 1, 0]), bound_request_id=torch.tensor([7, -1, -1]))
    assert receipt == owner.receipt()
    assert 'rows' not in receipt
    assert receipt['sampler']['random_draws'] == 0
    assert receipt['joint_imagination_supported'] is True
    assert receipt['joint_imagination_enabled'] is False
    saved = owner.state_dict()
    rng = native._terrain_course.tasks.generator.get_state().clone()
    owner.restore_after_physical_reset(saved)
    assert not owner.platform_rows.any()
    assert owner.caller.state.bound_request_id.tolist() == [-1, -1, -1]
    assert owner._reset_counts.tolist() == [1, 1, 1]
    assert torch.equal(rng, native._terrain_course.tasks.generator.get_state())
    reset(native, [0, 1, 2], [102, 0, 101], [3, 3, 3])
    assert owner.caller.state.target_index.tolist() == [1, -1, 0]
    assert owner._reset_counts.tolist() == [2, 2, 2]
    assert saved['caller']['platform_intent_ids'][0].tolist() == [0, 3, 7]
    damaged = {**saved, 'task': {**saved['task'], 'catalog_sha256': 'wrong'}}
    with pytest.raises(ValueError, match='saved source'):
        owner.restore_after_physical_reset(damaged)


def test_fixed_rows_keep_original_receipt_and_reset_mode(tmp_path):
    native, caller, _, _ = setup((0, -1))
    assert caller.reset_plan is None
    native._terrain_course.reset_commands(torch.tensor([0]), task([3]))
    assert caller.state.target_index.tolist() == [0, -1]
    plan = PlatformCoursePlan.read(configuration(tmp_path), num_envs=1,
                                  atlas_sha256='atlas-pin', record_substeps=True, jump_takeoff={})
    assert plan.sampler is None
    assert plan.state('cpu').target_index.tolist() == [0]


def test_fixed_owner_receipt_adds_no_sampler_or_support_claim(tmp_path, monkeypatch):
    _, owner = assembly(tmp_path, monkeypatch)
    receipt = owner.receipt()
    assert receipt['format'] == 'native-explicit-platform-course-v1'
    assert receipt['rows'] == [dict(owner.plan.rows[0])]
    assert receipt['plan_scope'] == (
        'explicit fixed per-row plans rearmed after native episode reset; no mixed-course sampler')
    assert 'sampler' not in receipt and 'joint_imagination_supported' not in receipt
