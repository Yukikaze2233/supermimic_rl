"""Reward-only continuation retains replay truth, learning state and source lineage."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path

import pytest
import torch

from supermimic_rl.cli import _native_schema_source
from supermimic_rl.course.gimbal_arbitration import arbitration_receipt
from supermimic_rl.reward_kernel import physical_reward_densities
from supermimic_rl.world_model_v2.migrations import extension_height, flight_reward, reward_continuity
from supermimic_rl.world_model_v2.migrations.source_lineage import source_lineage_from_saved
from test_world_v2_extension_height_upgrade import candidate as extension_fixture
from test_world_v2_recovery_bridge_upgrade import equal_nested
import test_world_v2_settling_upgrade as settling_tests


@pytest.fixture
def candidate(monkeypatch):
    original = settling_tests.old_reward
    monkeypatch.setattr(settling_tests, 'old_reward', lambda: {
        **original(), 'extensions': {**original()['extensions'], 'jump_flight': 4., 'jump_push': 2.}})
    saved, data, old_target, _, _ = extension_fixture.__wrapped__(monkeypatch)
    with monkeypatch.context() as history:
        history.setattr(reward_continuity, '_kernel_sha256', lambda: extension_height.TARGET_KERNEL_SHA256)
        saved, data, _ = extension_height.upgrade_jump_extension_height_floor(saved, data, old_target)
    storage = data['replay']['storage']
    storage.update(task_family=torch.zeros(4, 2, 1), motion_task=torch.zeros(4, 2, 40),
                   goal=torch.zeros(4, 2, 16), task_record_valid=torch.ones(4, 2, 1))
    geometry = {'format': 'synthetic-unit-geometry', 'source_asset': 'unit-test-only'}
    digest = hashlib.sha256(json.dumps(geometry, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    monkeypatch.setattr(flight_reward, 'GEOMETRY_RECEIPT_SHA256', digest)
    monkeypatch.setattr(flight_reward, 'TARGET_KERNEL_SHA256', 'a' * 64)
    monkeypatch.setattr(reward_continuity, '_kernel_sha256', lambda: 'a' * 64)
    target = json.loads(old_target.source)
    target['reward_kernel']['kernel_sha256'] = 'a' * 64
    target['reward_kernel']['extensions'].update(jump_flight_clearance=True, jump_tuck=1.7)
    target['reward_kernel']['jump_flight_geometry'] = geometry
    target['request_gimbal_exclusion'] = arbitration_receipt()
    target['course_schedule'].update(request_motion='birth_forward_held_v1',
        request_motion_resume='cold_physics_reset_repopulates_cache',
        spin_motion='zero_forward_spin_segments_v1')
    schema = replace(old_target, source=_native_schema_source(target))
    return saved, data, schema


def test_continuation_keeps_all_learning_data_and_rng(candidate):
    saved, data, target = candidate
    previous, truth = deepcopy(saved), deepcopy(data)
    rng = torch.get_rng_state().clone()
    updated, migrated, report = flight_reward.upgrade_flight_reward(saved, data, target)
    equal_nested(saved, previous)
    equal_nested(data, truth)
    assert torch.equal(rng, torch.get_rng_state())
    for name in saved.keys() - {'schema', 'schema_fingerprint', 'resume_migrations'}:
        assert updated[name] is saved[name], name
    assert migrated['replay'] is data['replay']
    assert report['additional_warmup_cycles'] == 0
    assert source_lineage_from_saved(updated, target)[-1] == report['jump_flight_geometry_reward']


@pytest.mark.parametrize('field', ['runtime', 'course', 'geometry', 'weight', 'kernel', 'sampler', 'arbitration'])
def test_only_reviewed_reward_and_geometry_changes(candidate, field):
    saved, _, target = candidate
    source, changed = json.loads(saved['schema']['source']), json.loads(target.source)
    if field in ('runtime', 'course'):
        changed[field + '_sha256'] = 'different'
    elif field == 'geometry':
        changed['reward_kernel']['jump_flight_geometry']['source_asset'] = 'another'
    elif field == 'weight':
        changed['reward_kernel']['extensions']['jump_push'] = 20.
    elif field == 'sampler':
        changed['course_schedule']['request_yaw_rate'] = 8.
    elif field == 'arbitration':
        changed['request_gimbal_exclusion']['ownership'] = 'all_tasks'
    else:
        changed['reward_kernel']['kernel_sha256'] = 'b' * 64
    with pytest.raises(ValueError):
        flight_reward.verify_flight_reward(source, changed)


def test_missing_optimizer_is_not_a_complete_resume(candidate):
    saved, data, target = candidate
    saved = deepcopy(saved)
    del saved['learner']['optimizers']['actor']
    with pytest.raises(ValueError):
        flight_reward.upgrade_flight_reward(saved, data, target)


def test_only_incompatible_old_actor_starts_are_excluded(candidate):
    saved, data, target = candidate
    storage = data['replay']['storage']
    storage['goal'][0, 0, 12] = 1.
    storage['motion_task'][1, 1, 35] = 1.
    storage['task_record_valid'][1, 1, 0] = 0.
    before = deepcopy(data)
    _, migrated, report = flight_reward.upgrade_flight_reward(saved, data, target)
    equal_nested(data, before)
    result = migrated['replay']['storage']
    assert result['task_record_valid'][0, 0, 0] == 0
    assert result['task_record_valid'][1, 1, 0] == 0
    assert int(result['task_record_valid'].sum()) == 6
    assert report['request_gimbal_conflicting_starts'] == 2
    assert report['request_gimbal_newly_excluded_starts'] == 1
    assert all(result[name] is value for name, value in storage.items() if name != 'task_record_valid')


def test_target_is_the_integrated_reward_implementation():
    actual = hashlib.sha256(Path(inspect.getfile(physical_reward_densities)).read_bytes()).hexdigest()
    assert flight_reward.TARGET_KERNEL_SHA256 == actual
    assert flight_reward.SOURCE_KERNEL_SHA256 == extension_height.TARGET_KERNEL_SHA256
    assert len(flight_reward.GEOMETRY_RECEIPT_SHA256) == 64
