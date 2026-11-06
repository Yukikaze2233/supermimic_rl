"""A request-lifetime handoff preserves learning and old physical evidence."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.contracts import StateSchema
from supermimic_rl.course.stair_label import install_tof_stair_label
from supermimic_rl.world_model_v2.auxiliary_data import AuxiliaryWorldData
from supermimic_rl.world_model_v2.migrations.stair_hint import (
    compatible_stair_hint_source, upgrade_stair_hint, verify_stair_hint,
)
from supermimic_rl.world_model_v2.migrations.course import IDENTITY_KEYS, verify_course_sampling
from supermimic_rl.world_model_v2.online import OnlineConfig, OnlineTrainer, policy_start_candidates
from supermimic_rl.world_model_v2.replay import CompactReplay
from supermimic_rl.world_model_v2.stair_source import StairScenario
from supermimic_rl.world_model_v2.teacher import file_sha256
from support.world_v2.stair_task import native_stairs
from test_world_v2_auxiliary_data import dataset as aux_dataset, windows
from test_world_v2_course_sampling_upgrade import candidate
from test_world_v2_reward_reference_upgrade import fixture as reward_fixture


def fixture(tmp_path, *, peak=False):
    saved, data, reference, options = reward_fixture(tmp_path, jump_peak=peak)
    native = native_stairs(2, labels=True, assist=True)
    native.cfg.stair_height_assist_spec = replace(native.cfg.stair_height_assist_spec, hold_s=2.)
    protocol = native._terrain_course.stair_requests
    protocol.config = replace(protocol.config, timeout_s=6.)
    before = deepcopy(saved['stair_scenario'])
    before['task']['height_assist']['hold_s'] = 2.
    before['task']['request']['timeout_s'] = 6.
    before['task']['use_mode_labels'] = True
    saved['stair_scenario'] = before
    after = deepcopy(before)
    after['task']['height_assist']['hold_s'] = 6.
    reference['stair_source_config'] = after
    runtime = json.dumps({'stair_height_assist': before['task']['height_assist']}).encode()
    source = json.loads(saved['schema']['source'])
    source.update(runtime_sha256=hashlib.sha256(runtime).hexdigest(),
                  stair_label=install_tof_stair_label(native))
    target = {**source, 'stair_label': install_tof_stair_label(native, height_hint_lifetime='request')}
    if peak:
        target['reward_kernel'] = json.loads(reference['schema']['source'])['reward_kernel']
    for item, provenance in ((saved, source), (reference, target)):
        schema = StateSchema.from_dict({**saved['schema'], 'source': json.dumps(provenance)})
        item.update(schema=schema.to_dict(), schema_fingerprint=schema.fingerprint)
    data.update(schema=saved['schema'], schema_fingerprint=saved['schema_fingerprint'])
    parameters = StairScenario.from_dict(before).task.signature(torch.zeros(1))
    data['replay']['storage']['stair_parameters'].copy_(parameters)
    data['replay']['storage']['stair_ids'][..., 3] = 57  # Historical remaining ticks must stay 57.
    data['replay']['storage']['stair_source_valid'].fill_(1.)
    for key, provenance in (('source_teacher_path', source), ('target_teacher_path', target)):
        payload = torch.load(options[key], weights_only=True)
        payload['source']['native_source'] = provenance
        payload['source']['qualification']['report']['native_source'] = provenance
        torch.save(payload, options[key])
        if key == 'source_teacher_path':
            saved['teacher_source'] = {**payload['source'], 'teacher_artifact': {
                'path': str(options[key]), 'sha256': file_sha256(options[key])}}
    return saved, data, reference, {**options, 'runtime_raw': runtime, 'jump_peak_completion': peak}


@pytest.mark.parametrize('peak', [False, True])
def test_handoff_keeps_all_learning_state_and_real_labels_and_uses_one_final_reference(tmp_path, peak):
    saved, data, reference, options = fixture(tmp_path, peak=peak)
    saved['auxiliary_world'] = {'sample_rng': torch.Generator().manual_seed(71).get_state(),
                              'sampled_windows': {'student750': 288780}}
    rng = torch.get_rng_state().clone()
    updated, migrated, report = upgrade_stair_hint(saved, data, reference, **options)
    assert torch.equal(rng, torch.get_rng_state())
    changed = {'schema', 'schema_fingerprint', 'teacher_source', 'stair_scenario', 'resume_migrations'}
    for key in saved.keys() - changed:
        assert updated[key] is saved[key], key
    original, result = data['replay']['storage'], migrated['replay']['storage']
    for key, value in original.items():
        if key != 'stair_source_valid':
            assert result[key] is value, key
    assert original['stair_source_valid'].eq(1).all() and result['stair_source_valid'].eq(0).all()
    assert result['stair_ids'][..., 3].eq(57).all()
    assert updated['stair_scenario']['task']['height_assist']['hold_s'] == 6.
    assert saved['stair_scenario']['task']['height_assist']['hold_s'] == 2.
    assert report['additional_warmup_cycles'] == 0
    assert report['jump_peak_completion'] == peak
    if peak:
        assert report['legacy_reward_sequence_before'] == data['replay']['total_steps']
    else:
        assert 'legacy_reward_sequence_before' not in report
    assert report['legacy_stair_starts_excluded'] == original['stair_source_valid'].numel()
    trainer = object.__new__(OnlineTrainer)
    trainer.env = SimpleNamespace(schema=StateSchema.from_dict(updated['schema']))
    trainer.config = OnlineConfig.from_dict(updated['config'])
    trainer.teacher_source = updated['teacher_source']
    for name in ('stair_scenario', 'jump_scenario', 'recovery_scenario'):
        setattr(trainer, name, updated[name])
    trainer._check_saved(updated)
    assert updated['cycle'] > trainer.config.policy_start_cycle


def test_new_collection_immediately_supplies_starts_without_changing_old_parameters(tmp_path):
    saved, data, reference, options = fixture(tmp_path)
    updated, data, _ = upgrade_stair_hint(saved, data, reference, **options)
    replay = CompactReplay.from_state_dict(data['replay'])
    config = OnlineConfig.from_dict(updated['config'])
    assert policy_start_candidates(replay, config, [0, 1], .02).numel() == 0
    boundary_names = ('terminated', 'truncated', 'valid', 'policy_owned', 'end_policy_owned',
                      'start_episode_id', 'end_episode_id', 'start_control_epoch', 'end_control_epoch')
    last = (replay.head - 1) % replay.capacity
    boundary = {name: replay.data[name][last].clone() for name in boundary_names}
    for name in ('valid', 'policy_owned', 'end_policy_owned'):
        boundary[name].fill_(True)
    for name in ('terminated', 'truncated'):
        boundary[name].fill_(False)
    record = {name: value[last].clone() for name, value in replay.data.items()
              if name not in (*boundary_names, 'segment_age')}
    for name in ('task_record_valid', 'stair_source_valid', 'jump_record_valid',
                 'recovery_record_valid', 'reward_context_valid'):
        record[name].fill_(1.)
    record['stair_parameters'].copy_(StairScenario.from_dict(updated['stair_scenario']).task.signature(
        record['stair_parameters']))
    record['motion_task'][:, 21] = 0.  # No unrecorded managed-pool boundary.
    record['reward_route'].zero_()
    record['task_family'].zero_()
    record['frame'].fill_(7.)
    replay.add(record, boundary)
    assert policy_start_candidates(replay, config, [0, 1], .02).numel() == 2
    for _ in range(23):
        replay.add(record, boundary)
    candidates = policy_start_candidates(replay, config, [0, 1], .02)
    assert candidates.numel() > 0
    trainer = object.__new__(OnlineTrainer)
    trainer.config, trainer.device = config, torch.device('cpu')
    trainer.collector, trainer.env = SimpleNamespace(replay=replay), SimpleNamespace(policy_dt_s=.02)
    trainer.learner = SimpleNamespace(actor=SimpleNamespace(history_length=5))
    trainer.train_ids, trainer.generator = [0, 1], torch.Generator().manual_seed(31)
    start, _ = trainer.sample_starts(128)
    assert start['frame'].eq(7.).all() and start['stair_source_valid'].eq(1.).all()
    assert start['stair_parameters'][:, 13].eq(6.).all()


@pytest.mark.parametrize('damage', ['runtime', 'asset', 'request_timeout', 'hold', 'raise', 'label',
                                   'jump', 'record', 'pair', 'teacher', 'reward'])
def test_only_authenticated_hint_change_is_allowed(tmp_path, damage):
    saved, data, reference, options = fixture(tmp_path)
    if damage == 'runtime':
        options['runtime_raw'] += b' '
    elif damage in ('asset', 'label', 'reward'):
        source = json.loads(reference['schema']['source'])
        if damage == 'asset':
            source['asset'] = 'other-robot'
        elif damage == 'label':
            source['stair_label']['height_hint']['runtime_hold_s'] = 1.
        else:
            source['reward_kernel']['extensions']['jump_push'] += 1.
        schema = StateSchema.from_dict({**reference['schema'], 'source': json.dumps(source)})
        reference.update(schema=schema.to_dict(), schema_fingerprint=schema.fingerprint)
    elif damage in ('request_timeout', 'hold', 'raise'):
        task = reference['stair_source_config']['task']
        if damage == 'request_timeout':
            task['request']['timeout_s'] += 1.
        else:
            task['height_assist']['hold_s' if damage == 'hold' else 'raise_by_m'] += 1.
    elif damage == 'jump':
        reference['jump_task_config']['reference']['compression_duration_s'] += .1
    elif damage == 'record':
        reference['replay']['storage']['stair_clock'] = torch.zeros(1, 2, 3)
    elif damage == 'pair':
        data['cycle'] += 1
    elif damage == 'teacher':
        payload = torch.load(options['target_teacher_path'], weights_only=True)
        payload['actor']['weights'].add_(1.)
        torch.save(payload, options['target_teacher_path'])
    with pytest.raises(ValueError):
        upgrade_stair_hint(saved, data, reference, **options)


def test_auxiliary_motion_windows_retain_source_split_and_resume_rng(tmp_path):
    saved, _, reference, options = fixture(tmp_path)
    source, target = [json.loads(item['schema']['source']) for item in (saved, reference)]
    proof = verify_stair_hint(source, target, saved['stair_scenario'], reference['stair_source_config'],
                              runtime_raw=options['runtime_raw'])
    schema, replay, config = aux_dataset(tmp_path)
    old, new = [replace(schema, source=json.dumps(item)) for item in (source, target)]
    path = Path(config.path)
    payload = torch.load(path, weights_only=True)
    payload.update(schema=old.to_dict(), schema_fingerprint=old.fingerprint)
    torch.save(payload, path)
    config = replace(config, sha256=file_sha256(path))
    with pytest.raises(ValueError, match='contract differs'):
        AuxiliaryWorldData((config,), new, replay, 2, seed=19)
    previous = AuxiliaryWorldData((config,), old, replay, 2, seed=19)
    previous.mix(windows(replay))
    restored = AuxiliaryWorldData((config,), new, replay, 2, seed=999, stair_hint_migrations=(proof,))
    restored.load_state_dict(previous.state_dict())
    expected, actual = previous.mix(windows(replay)), restored.mix(windows(replay))
    assert restored.source_schemas == {config.name: old.fingerprint}
    assert restored.sources[0][0] == config
    assert restored.sampled == previous.sampled
    assert torch.equal(restored.validation_generator.get_state(), previous.validation_generator.get_state())
    assert all(torch.equal(actual[name], value) for name, value in expected.items())
    assert (actual['action'][actual['action'] >= 0] < 1000).all()
    assert file_sha256(path) == config.sha256
    assert compatible_stair_hint_source(source, target, (proof,)) == source
    forged = deepcopy(proof)
    forged['target_scenario']['task']['fsm']['failure_tilt_rad'] += .1
    with pytest.raises(ValueError, match='Only binding'):
        AuxiliaryWorldData((config,), new, replay, 2, seed=19, stair_hint_migrations=(forged,))
    other = replace(new, source=json.dumps({**target, 'asset': 'different'}))
    with pytest.raises(ValueError, match='contract differs'):
        AuxiliaryWorldData((config,), other, replay, 2, seed=19, stair_hint_migrations=(proof,))


def test_hint_proof_composes_with_existing_course_and_terrain_lineage(tmp_path, candidate):
    saved, _, reference, options = fixture(tmp_path)
    source, target = [json.loads(item['schema']['source']) for item in (saved, reference)]
    hint = verify_stair_hint(source, target, saved['stair_scenario'], reference['stair_source_config'],
                             runtime_raw=options['runtime_raw'])
    course_saved, _, course_target, course_options = candidate
    before, after = json.loads(course_saved['schema']['source']), json.loads(course_target.source)
    course = verify_course_sampling(before, after, **course_options)
    source.update({key: before[key] for key in IDENTITY_KEYS})
    target.update({key: after[key] for key in IDENTITY_KEYS})
    source.update(terrain_sha256='older-atlas', runtime_sha256='older-runtime',
                  authored_routing={'atlas_sha256': 'older-atlas'})
    target['authored_routing'] = {'atlas_sha256': target['terrain_sha256']}
    terrain = {'source_atlas_sha256': 'older-atlas', 'target_atlas_sha256': target['terrain_sha256'],
               'runtime_pair': {'before': 'older-runtime', 'after': target['runtime_sha256']},
               'changed_tiles': [1]}
    schema = StateSchema.from_dict(saved['schema'])
    old, new = [replace(schema, source=json.dumps(value)) for value in (source, target)]
    same = AuxiliaryWorldData._same_dynamics_contract
    assert same(old, new, terrain_migrations=(terrain,), course_migrations=(course,),
                stair_hint_migrations=(hint,))
    assert not same(old, new, terrain_migrations=(terrain,), course_migrations=(course,))
    assert not same(old, new, terrain_migrations=(terrain,), stair_hint_migrations=(hint,))
    assert not same(old, new, course_migrations=(course,), stair_hint_migrations=(hint,))


def test_combined_peak_change_cannot_change_unrelated_reward_weights(tmp_path):
    saved, data, reference, options = fixture(tmp_path, peak=True)
    source = json.loads(reference['schema']['source'])
    source['reward_kernel']['extensions']['jump_push'] += 1.
    schema = StateSchema.from_dict({**reference['schema'], 'source': json.dumps(source)})
    reference.update(schema=schema.to_dict(), schema_fingerprint=schema.fingerprint)
    with pytest.raises(ValueError, match='completion switch'):
        upgrade_stair_hint(saved, data, reference, **options)
