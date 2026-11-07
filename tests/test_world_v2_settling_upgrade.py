"""Two fixed reward opt-ins preserve full source lineage and old learning truth."""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import inspect
import json
from pathlib import Path

import pytest
import torch

from supermimic_rl.cli import _native_schema_source
from supermimic_rl.contracts import StateSchema
from supermimic_rl.reward_kernel import physical_reward_densities
from supermimic_rl.world_model_v2.auxiliary_data import AuxiliaryWorldData
from supermimic_rl.world_model_v2.jump_task import JumpTaskConfig
from supermimic_rl.world_model_v2.migrations import settling
from supermimic_rl.world_model_v2.migrations.source_lineage import (
    compatible_source_lineage, source_lineage_from_saved,
)
from supermimic_rl.world_model_v2.teacher import (
    export_ppo_teacher_payload, file_sha256, load_native_teacher,
)
from support.world_v2.jump_task import native_jumps
from test_world_v2_auxiliary_data import dataset, windows
from test_world_v2_course_sampling_upgrade import candidate as course_fixture
from test_world_v2_native_ppo_teacher import fixture as teacher_fixture
from test_world_v2_recovery_bridge_upgrade import candidate as bridge_fixture, equal_nested
from test_world_v2_source_lineage import chain


def old_reward():
    return {'implementation': 'shared_physical_reward_densities',
            'kernel_sha256': settling.SOURCE_KERNEL_SHA256, 'physics_dt_s': .005,
            'extensions': {'jump_peak_completion': True, 'jump_missed': 1.,
                           'jump_settle': 2., 'jump_landing_rate': 4., 'recovery_quiet': 1.}}


def target_source(source):
    target = deepcopy(source)
    target['reward_kernel']['kernel_sha256'] = settling.TARGET_KERNEL_SHA256
    target['reward_kernel']['extensions'].update(stationary_quiet=1., jump_physical_tail=True)
    return target


@pytest.fixture
def candidate():
    original = course_fixture.__wrapped__()
    saved, schema, proofs, sources = chain(original, extra_source={'reward_kernel': old_reward()})
    full, _, _, _ = bridge_fixture.__wrapped__()
    saved['learner'] = full['learner']
    saved.update({name: full[name] for name in ('cuda_rng', 'validation_rng', 'collection_rng', 'mixing_rng')})
    saved['jump_scenario'] = asdict(replace(JumpTaskConfig.capture(native_jumps()), flight_latch=True))
    saved['teacher_source'] = deepcopy(saved['teacher_source'])
    saved['teacher_source']['native_source'] = deepcopy(sources[0])
    saved['teacher_source']['qualification']['report']['native_source'] = deepcopy(sources[0])
    data = original[1]
    data.update(schema=schema.to_dict(), schema_fingerprint=schema.fingerprint)
    storage = data['replay']['storage']
    storage.update(jump_ids=torch.zeros(4, 2, 9, dtype=torch.int64),
                   jump_clock=torch.zeros(4, 2, 5, dtype=torch.float64),
                   jump_flight_latch=torch.zeros(4, 2, 3, dtype=torch.int64),
                   reward=torch.arange(8.).reshape(4, 2, 1),
                   stair_source_valid=torch.tensor([1., 0., 1., 1., 0., 0., 1., 0.]).reshape(4, 2, 1))
    target = replace(schema, source=_native_schema_source(target_source(sources[-1])))
    return saved, data, target, proofs, sources


def test_complete_pair_preserves_all_learning_and_recorded_labels(candidate):
    saved, data, target, proofs, sources = candidate
    before, records = deepcopy(saved), deepcopy(data)
    rng = torch.get_rng_state().clone()
    updated, replay, report = settling.upgrade_settling_reward_continuity(saved, data, target)
    equal_nested(saved, before)
    equal_nested(data, records)
    assert torch.equal(rng, torch.get_rng_state())
    for key in saved.keys() - {'schema', 'schema_fingerprint', 'resume_migrations'}:
        assert updated[key] is saved[key], key
    assert replay['replay'] is data['replay']
    assert report['legacy_reward_sequence_before'] == data['replay']['total_steps']
    assert report['additional_warmup_cycles'] == 0
    assert updated['teacher_source'] is saved['teacher_source']
    assert source_lineage_from_saved(updated, target) == (*proofs, report['settling_reward_continuity'])
    for source in sources:
        assert compatible_source_lineage(source, json.loads(target.source),
            source_lineage_from_saved(updated, target)) == source
    assert updated['jump_scenario'] is saved['jump_scenario']
    assert updated['recovery_scenario'] is saved['recovery_scenario']
    assert updated['stair_scenario'] is saved['stair_scenario']


def test_saved_proof_remains_valid_under_a_later_kernel_file(candidate, monkeypatch, tmp_path):
    saved, data, target, _, _ = candidate
    source = json.loads(saved['schema']['source'])
    proof = settling.verify_settling_reward_continuity(source, json.loads(target.source))
    later = tmp_path / 'later_reward.py'
    later.write_text('a later reviewed kernel\n')
    monkeypatch.setattr(settling.inspect, 'getfile', lambda value: str(later))
    assert settling.verify_saved_proof(proof) == proof
    with pytest.raises(ValueError, match='Executing reward kernel'):
        settling.upgrade_settling_reward_continuity(saved, data, target)


@pytest.mark.parametrize('change', [
    'old_hash', 'new_hash', 'missing_peak', 'quiet_already', 'tail_already',
    'quiet_weight', 'quiet_bool', 'tail_not_bool', 'other_weight', 'runtime', 'course', 'kernel_extra',
])
def test_reward_proof_is_not_an_arbitrary_override(candidate, change):
    saved, _, target, _, _ = candidate
    source, after = json.loads(saved['schema']['source']), json.loads(target.source)
    before, updated = source['reward_kernel'], after['reward_kernel']
    if change == 'old_hash':
        before['kernel_sha256'] = 'unreviewed'
    elif change == 'new_hash':
        updated['kernel_sha256'] = 'unreviewed'
    elif change == 'missing_peak':
        before['extensions']['jump_peak_completion'] = False
    elif change == 'quiet_already':
        before['extensions']['stationary_quiet'] = .1
    elif change == 'tail_already':
        before['extensions']['jump_physical_tail'] = True
    elif change == 'quiet_weight':
        updated['extensions']['stationary_quiet'] = 2.
    elif change == 'quiet_bool':
        updated['extensions']['stationary_quiet'] = True
    elif change == 'tail_not_bool':
        updated['extensions']['jump_physical_tail'] = 1
    elif change == 'other_weight':
        updated['extensions']['recovery_quiet'] = 2.
    elif change == 'runtime':
        after['runtime_sha256'] = 'another'
    elif change == 'course':
        after['curriculum']['base']['stage_weights'][0][0] += .01
    else:
        updated['unreviewed_reward'] = True
    with pytest.raises(ValueError):
        settling.verify_settling_reward_continuity(source, after)


@pytest.mark.parametrize('change', ['extra', 'disconnect', 'wrong_order', 'missing_proof'])
def test_full_source_ledger_rejects_unproven_reward_changes(candidate, change):
    saved, data, target, _, _ = candidate
    updated, _, _ = settling.upgrade_settling_reward_continuity(saved, data, target)
    updated = deepcopy(updated)
    proof = updated['resume_migrations'][-1]['settling_reward_continuity']
    if change == 'extra':
        proof['relax_other_sources'] = True
    elif change == 'disconnect':
        proof['before']['asset_manifest_sha256'] = 'different'
        proof['after']['asset_manifest_sha256'] = 'different'
    elif change == 'wrong_order':
        updated['resume_migrations'][-1], updated['resume_migrations'][-3] = (
            updated['resume_migrations'][-3], updated['resume_migrations'][-1])
    else:
        updated['resume_migrations'].pop()
    with pytest.raises(ValueError):
        source_lineage_from_saved(updated, target)


@pytest.mark.parametrize('change', ['cycle', 'optimizer', 'layout', 'latch', 'ids', 'recorded_tasks'])
def test_upgrade_requires_the_existing_complete_physical_request_contract(candidate, change):
    saved, data, target, _, _ = candidate
    if change == 'cycle':
        data['cycle'] += 1
    elif change == 'optimizer':
        saved['learner']['optimizers'].pop('critic')
    elif change == 'layout':
        target = replace(target, policy_dt_s=target.policy_dt_s * 2)
    elif change == 'latch':
        saved['jump_scenario']['flight_latch'] = False
    elif change == 'ids':
        data['replay']['storage']['jump_ids'] = data['replay']['storage']['jump_ids'].float()
    else:
        saved['config']['record_jumps'] = False
    with pytest.raises(ValueError):
        settling.upgrade_settling_reward_continuity(saved, data, target)


def test_teacher_stays_at_its_original_qualification_after_combined_reward(candidate, tmp_path):
    _, checkpoint, evaluation, original = teacher_fixture(tmp_path)
    extra = {k: v for k, v in original.items() if k not in ('runtime_sha256', 'course_sha256')}
    extra['reward_kernel'] = old_reward()
    _, _, proofs, sources = chain(course_fixture.__wrapped__(), extra_source=extra)
    recorded = {**sources[0], 'stair_label': sources[2]['stair_label']}
    target = target_source(sources[-1])
    proofs = (*proofs, settling.verify_settling_reward_continuity(sources[-1], target))
    report = json.loads(evaluation.read_text())
    report['native_source'] = recorded
    evaluation.write_text(json.dumps(report))
    payload = export_ppo_teacher_payload(checkpoint, evaluation, ['foundation'])
    path = tmp_path / 'teacher.pt'
    torch.save(payload, path)
    digest = file_sha256(path)
    baseline = load_native_teacher(path, digest, device='cpu', native_provenance=recorded)
    current = load_native_teacher(path, digest, device='cpu', native_provenance=target, source_lineage=proofs)
    assert current.source_metadata == baseline.source_metadata
    assert current.source_metadata['qualification']['report']['native_source'] == recorded
    assert current.source_metadata['qualified_for'] == ['foundation']
    frames = torch.randn(3, 5, 35)
    torch.testing.assert_close(current.action_and_estimate(frames), baseline.action_and_estimate(frames), rtol=0, atol=0)
    assert file_sha256(path) == digest


def test_auxiliary_reward_exception_runs_after_full_source_chain_and_preserves_next_draw(candidate, tmp_path):
    saved, _, target, proofs, sources = candidate
    proof = settling.verify_settling_reward_continuity(json.loads(saved['schema']['source']), json.loads(target.source))
    old, replay, config = dataset(tmp_path)
    recorded = deepcopy(sources[0])
    recorded['reward_kernel']['kernel_sha256'] = 'historical-unused-reward'
    old = replace(old, source=_native_schema_source(recorded))
    live = replace(old, source=target.source)
    path = Path(config.path)
    payload = torch.load(path, weights_only=True)
    payload.update(schema=old.to_dict(), schema_fingerprint=old.fingerprint)
    torch.save(payload, path)
    config = replace(config, sha256=file_sha256(path))
    baseline = AuxiliaryWorldData((config,), old, replay, 2, seed=19)
    current = AuxiliaryWorldData((config,), live, replay, 2, seed=999, source_lineage=(*proofs, proof))
    baseline.mix(windows(replay))
    current.load_state_dict(baseline.state_dict())
    expected, actual = baseline.mix(windows(replay)), current.mix(windows(replay))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    equal_nested(current.state_dict(), baseline.state_dict())
    assert current.source_schemas == {config.name: old.fingerprint}
    assert current.sources[0][0] == config
    assert file_sha256(path) == config.sha256


def test_target_kernel_constant_is_the_reviewed_production_file():
    assert hashlib.sha256(Path(inspect.getfile(physical_reward_densities)).read_bytes()).hexdigest() == (
        settling.TARGET_KERNEL_SHA256)
