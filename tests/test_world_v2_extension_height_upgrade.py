"""One fixed height-objective opt-in retains learners, truth and source lineage."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path

import pytest
import torch

from supermimic_rl.cli import _native_schema_source
from supermimic_rl.reward_kernel import physical_reward_densities
from supermimic_rl.world_model_v2.auxiliary_data import AuxiliaryWorldData
from supermimic_rl.world_model_v2.migrations import extension_height, reward_continuity, settling
from supermimic_rl.world_model_v2.migrations.source_lineage import (
    compatible_source_lineage, source_lineage_from_saved,
)
from supermimic_rl.world_model_v2.teacher import (
    export_ppo_teacher_payload, file_sha256, load_native_teacher,
)
from test_world_v2_auxiliary_data import dataset, windows
from test_world_v2_native_ppo_teacher import fixture as teacher_fixture
from test_world_v2_recovery_bridge_upgrade import equal_nested
from test_world_v2_settling_upgrade import candidate as settling_fixture


def target_source(source):
    target = deepcopy(source)
    target['reward_kernel']['kernel_sha256'] = extension_height.TARGET_KERNEL_SHA256
    target['reward_kernel']['extensions']['jump_extension_height_floor'] = True
    return target


@pytest.fixture
def candidate(monkeypatch):
    saved, data, settled_schema, earlier_proofs, sources = settling_fixture.__wrapped__()
    # Reconstruct the historical handoff only in this small synthetic fixture.
    with monkeypatch.context() as historical:
        historical.setattr(reward_continuity, '_kernel_sha256', lambda: settling.TARGET_KERNEL_SHA256)
        saved, data, report = settling.upgrade_settling_reward_continuity(saved, data, settled_schema)
    source = json.loads(settled_schema.source)
    target = replace(settled_schema, source=_native_schema_source(target_source(source)))
    return saved, data, target, (*earlier_proofs, report['settling_reward_continuity']), sources


def test_complete_pair_retains_every_learner_optimizer_label_and_rng(candidate):
    saved, data, target, prior_proofs, _ = candidate
    before, records = deepcopy(saved), deepcopy(data)
    cpu_rng = torch.get_rng_state().clone()
    versions = {name: value._version for name, value in data['replay']['storage'].items()}
    updated, replay, report = extension_height.upgrade_jump_extension_height_floor(saved, data, target)
    equal_nested(saved, before)
    equal_nested(data, records)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    for name in saved.keys() - {'schema', 'schema_fingerprint', 'resume_migrations'}:
        assert updated[name] is saved[name], name
    assert replay['replay'] is data['replay']
    assert versions == {name: value._version for name, value in replay['replay']['storage'].items()}
    assert updated['resume_migrations'][:-1] == saved['resume_migrations']
    assert report['legacy_reward_sequence_before'] == data['replay']['total_steps']
    assert report['additional_warmup_cycles'] == 0
    assert report['pending_integration']
    assert source_lineage_from_saved(updated, target) == (*prior_proofs, report['jump_extension_height_floor'])


@pytest.mark.parametrize('change', [
    'source_hash', 'target_hash', 'already_enabled', 'source_nonboolean', 'target_nonboolean',
    'missing_peak', 'missing_tail', 'changed_quiet', 'another_weight', 'runtime', 'course',
    'task_receipt', 'source_extra', 'target_extra',
])
def test_only_reviewed_kernel_and_boolean_opt_in_are_authorized(candidate, change):
    saved, _, target, _, _ = candidate
    source, after = json.loads(saved['schema']['source']), json.loads(target.source)
    before_kernel, after_kernel = source['reward_kernel'], after['reward_kernel']
    if change == 'source_hash':
        before_kernel['kernel_sha256'] = 'unknown'
    elif change == 'target_hash':
        after_kernel['kernel_sha256'] = 'unknown'
    elif change in ('already_enabled', 'source_nonboolean'):
        before_kernel['extensions']['jump_extension_height_floor'] = True if change == 'already_enabled' else 0
    elif change == 'target_nonboolean':
        after_kernel['extensions']['jump_extension_height_floor'] = 1
    elif change in ('missing_peak', 'missing_tail'):
        flag = {'missing_peak': 'jump_peak_completion', 'missing_tail': 'jump_physical_tail'}[change]
        before_kernel['extensions'][flag] = False
    elif change == 'changed_quiet':
        after_kernel['extensions']['stationary_quiet'] = 2.
    elif change == 'another_weight':
        after_kernel['extensions']['jump_missed'] = 2.
    elif change in ('runtime', 'course'):
        after[change + '_sha256'] = 'other'
    elif change == 'task_receipt':
        after['stair_label'] = {'other': True}
    elif change == 'source_extra':
        source['unknown_contract'] = True
    else:
        after['unknown_contract'] = True
    with pytest.raises(ValueError):
        extension_height.verify_jump_extension_height_floor(source, after)


def test_explicit_false_source_flag_has_the_same_narrow_opt_in(candidate):
    saved, _, target, _, _ = candidate
    source = json.loads(saved['schema']['source'])
    source['reward_kernel']['extensions']['jump_extension_height_floor'] = False
    proof = extension_height.verify_jump_extension_height_floor(source, json.loads(target.source))
    assert extension_height.verify_saved_proof(proof) == proof


@pytest.mark.parametrize('change', ['cycle', 'optimizer', 'layout', 'latch', 'record_dtype', 'recorded_tasks'])
def test_complete_pair_and_existing_request_memory_are_required(candidate, change):
    saved, data, target, _, _ = candidate
    if change == 'cycle':
        data['cycle'] += 1
    elif change == 'optimizer':
        saved['learner']['optimizers'].pop('critic')
    elif change == 'layout':
        target = replace(target, policy_dt_s=target.policy_dt_s * 2)
    elif change == 'latch':
        saved['jump_scenario']['flight_latch'] = False
    elif change == 'record_dtype':
        data['replay']['storage']['jump_ids'] = data['replay']['storage']['jump_ids'].float()
    else:
        saved['config']['record_jumps'] = False
    with pytest.raises(ValueError):
        extension_height.upgrade_jump_extension_height_floor(saved, data, target)


@pytest.mark.parametrize('change', ['wrong_order', 'removed', 'unknown', 'extra', 'disconnected'])
def test_source_ledger_must_remain_exact_and_ordered(candidate, change):
    saved, data, target, _, _ = candidate
    updated, _, _ = extension_height.upgrade_jump_extension_height_floor(saved, data, target)
    updated = deepcopy(updated)
    ledger = updated['resume_migrations']
    proof = ledger[-1]['jump_extension_height_floor']
    if change == 'wrong_order':
        ledger[-1], ledger[-2] = ledger[-2], ledger[-1]
    elif change == 'removed':
        ledger.pop()
    elif change == 'unknown':
        proof['format'] = 'unreviewed-reward-override'
    elif change == 'extra':
        proof['ignore_reward_differences'] = True
    else:
        proof['before']['asset_manifest_sha256'] = 'changed'
        proof['after']['asset_manifest_sha256'] = 'changed'
    with pytest.raises(ValueError):
        source_lineage_from_saved(updated, target)


def test_historical_proofs_survive_later_file_but_upgrade_does_not(candidate, monkeypatch):
    saved, data, target, prior, _ = candidate
    proof = extension_height.verify_jump_extension_height_floor(
        json.loads(saved['schema']['source']), json.loads(target.source))
    monkeypatch.setattr(reward_continuity, '_kernel_sha256', lambda: 'a future reward kernel')
    assert extension_height.verify_saved_proof(proof) == proof
    assert settling.verify_saved_proof(prior[-1]) == prior[-1]
    with pytest.raises(ValueError, match='Executing reward kernel'):
        extension_height.upgrade_jump_extension_height_floor(saved, data, target)


def test_original_teacher_artifact_keeps_weights_and_qualification(candidate, tmp_path):
    saved, data, target, _, sources = candidate
    updated, _, _ = extension_height.upgrade_jump_extension_height_floor(saved, data, target)
    proofs = source_lineage_from_saved(updated, target)
    _, checkpoint, evaluation, _ = teacher_fixture(tmp_path)
    recorded = deepcopy(sources[0])
    # Existing exports can carry the newer hint with the original course.
    recorded['stair_label'] = deepcopy(sources[2]['stair_label'])
    report = json.loads(evaluation.read_text())
    report['native_source'] = recorded
    evaluation.write_text(json.dumps(report))
    payload = export_ppo_teacher_payload(checkpoint, evaluation, ['foundation'])
    path = tmp_path / 'teacher.pt'
    torch.save(payload, path)
    digest = file_sha256(path)
    original = load_native_teacher(path, digest, device='cpu', native_provenance=recorded)
    current = load_native_teacher(path, digest, device='cpu', native_provenance=json.loads(target.source),
                                  source_lineage=proofs)
    assert current.source_metadata == original.source_metadata
    assert current.source_metadata['qualified_for'] == ['foundation']
    assert current.source_metadata['qualification']['report']['native_source'] == recorded
    frames = torch.randn(3, 5, 35)
    torch.testing.assert_close(current.action_and_estimate(frames), original.action_and_estimate(frames), rtol=0, atol=0)
    assert file_sha256(path) == digest


def test_older_auxiliary_source_split_and_next_draw_survive_both_reward_proofs(candidate, tmp_path):
    saved, data, target, _, sources = candidate
    updated, _, _ = extension_height.upgrade_jump_extension_height_floor(saved, data, target)
    proofs = source_lineage_from_saved(updated, target)
    old, replay, config = dataset(tmp_path)
    recorded = deepcopy(sources[0])
    recorded['reward_kernel']['kernel_sha256'] = 'historical-unused-reward'
    old = replace(old, source=_native_schema_source(recorded))
    live = replace(old, source=target.source)
    path = Path(config.path)
    payload = torch.load(path, weights_only=True)
    payload.update(schema=old.to_dict(), schema_fingerprint=old.fingerprint)
    torch.save(payload, path)
    config = replace(config, name='student750', sha256=file_sha256(path))
    baseline = AuxiliaryWorldData((config,), old, replay, 2, seed=19)
    current = AuxiliaryWorldData((config,), live, replay, 2, seed=999, source_lineage=proofs)
    baseline.mix(windows(replay))
    current.load_state_dict(baseline.state_dict())
    expected, actual = baseline.mix(windows(replay)), current.mix(windows(replay))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    equal_nested(current.state_dict(), baseline.state_dict())
    assert current.source_schemas == {config.name: old.fingerprint}
    assert current.sources[0][0] == config
    assert not set(config.train_env_ids) & set(config.validation_env_ids)
    for source in sources:
        assert compatible_source_lineage(source, json.loads(target.source), proofs) == source
    assert file_sha256(path) == config.sha256


def test_new_target_kernel_is_the_reviewed_production_file():
    assert hashlib.sha256(Path(inspect.getfile(physical_reward_densities)).read_bytes()).hexdigest() == (
        extension_height.TARGET_KERNEL_SHA256)
    assert extension_height.SOURCE_KERNEL_SHA256 == settling.TARGET_KERNEL_SHA256
