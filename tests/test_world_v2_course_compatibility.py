"""Sampling receipts preserve teacher evidence and immutable auxiliary records."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import runpy

import pytest
import torch

from supermimic_rl.contracts import StateSchema
from supermimic_rl.world_model_v2.auxiliary_data import AuxiliaryWorldData
from supermimic_rl.world_model_v2.migrations.course import (
    IDENTITY_KEYS, upgrade_course_sampling, verify_course_sampling,
)
from supermimic_rl.world_model_v2.teacher import (
    export_ppo_teacher_payload, file_sha256, load_native_teacher,
)
from test_world_v2_auxiliary_data import dataset, windows
from test_world_v2_course_sampling_upgrade import candidate
from test_world_v2_native_ppo_teacher import fixture as teacher_fixture


ROOT = Path(__file__).resolve().parents[1]


def test_sampling_alias_keeps_old_auxiliary_truth_split_and_rng(tmp_path, candidate):
    saved, _, target, options = candidate
    old, replay, config = dataset(tmp_path)
    source, live = json.loads(saved['schema']['source']), json.loads(target.source)
    old, target = replace(old, source=json.dumps(source)), replace(old, source=json.dumps(live))
    proof = verify_course_sampling(source, live, **options)
    path = Path(config.path)
    payload = torch.load(path, weights_only=True)
    payload.update(schema=old.to_dict(), schema_fingerprint=old.fingerprint)
    payload['replay']['storage']['jump_values'] = torch.full((8, 2, 17), .4)
    torch.save(payload, path)
    config = replace(config, name='student750', sha256=file_sha256(path))
    with pytest.raises(ValueError, match='contract differs'):
        AuxiliaryWorldData((config,), target, replay, 2, seed=19)
    aux = AuxiliaryWorldData((config,), target, replay, 2, seed=19, course_migrations=(proof,))
    assert aux.source_schemas == {'student750': old.fingerprint}
    assert aux.sources[0][0] == config
    assert aux.sources[0][1].data['jump_values'].eq(.4).all()
    assert torch.equal(aux.sources[0][1].data['valid'], replay.data['valid'])
    state = aux.state_dict()
    restored = AuxiliaryWorldData((config,), target, replay, 2, seed=101, course_migrations=(proof,))
    restored.load_state_dict(state)
    expected, actual = aux.mix(windows(replay)), restored.mix(windows(replay))
    for name in expected:
        assert torch.equal(actual[name], expected[name])
    assert (actual['action'][actual['action'] >= 0] < 1000).all()  # held-out row remains excluded
    assert file_sha256(path) == config.sha256
    for field, value in (('runtime_sha256', 'unknown-runtime'), ('upstream_commit', 'unknown-code')):
        other = replace(target, source=json.dumps({**live, field: value}))
        with pytest.raises(ValueError, match='contract differs'):
            AuxiliaryWorldData((config,), other, replay, 2, seed=19, course_migrations=(proof,))


def test_course_terrain_and_disturbance_aliases_compose_without_hiding_other_differences(tmp_path, candidate):
    from test_world_v2_disturbance_upgrade import fixture as disturbance_fixture

    course_saved, _, course_target, course_options = candidate
    before, after = json.loads(course_saved['schema']['source']), json.loads(course_target.source)
    proof = verify_course_sampling(before, after, **course_options)
    saved, _, reference, options, *_ = disturbance_fixture(tmp_path)
    schemas = []
    for item, course in ((saved, before), (reference, after)):
        schema = StateSchema.from_dict(item['schema'])
        source = json.loads(schema.source)
        source.update({key: course[key] for key in IDENTITY_KEYS})
        source.update(terrain_sha256='old-atlas', authored_routing={'atlas_sha256': 'old-atlas'})
        schemas.append(replace(schema, source=json.dumps(source)))
    source = json.loads(schemas[1].source)
    terrain = {'source_atlas_sha256': 'old-atlas', 'target_atlas_sha256': 'new-atlas',
               'runtime_pair': {'before': source['runtime_sha256'], 'after': 'new-runtime'},
               'changed_tiles': [1]}
    source.update(terrain_sha256='new-atlas', runtime_sha256='new-runtime',
                  authored_routing={'atlas_sha256': 'new-atlas'})
    live = replace(schemas[1], source=json.dumps(source))
    disturbance = {'before': hashlib.sha256(options['source_runtime']).hexdigest(),
                   'after': hashlib.sha256(options['target_runtime']).hexdigest()}
    same = AuxiliaryWorldData._same_dynamics_contract
    assert same(schemas[0], live, (disturbance,), (terrain,), (proof,))
    assert not same(schemas[0], live, (disturbance,), (terrain,))
    assert not same(schemas[0], live, (), (terrain,), (proof,))
    assert not same(schemas[0], live, (disturbance,), (), (proof,))
    assert not same(schemas[0], replace(live, source=json.dumps({**source, 'asset': 'different'})),
                    (disturbance,), (terrain,), (proof,))


def test_teacher_sampling_alias_retains_artifact_qualification_and_actions(tmp_path, candidate):
    saved, _, target, options = candidate
    _, checkpoint, evaluation, source = teacher_fixture(tmp_path)
    before, after = json.loads(saved['schema']['source']), json.loads(target.source)
    native = [{**source, **{key: course[key] for key in IDENTITY_KEYS}} for course in (before, after)]
    proof = verify_course_sampling(*native, **options)
    report = json.loads(evaluation.read_text())
    report['native_source'] = native[0]
    evaluation.write_text(json.dumps(report))
    payload = export_ppo_teacher_payload(checkpoint, evaluation, ['foundation'])
    artifact = tmp_path / 'teacher.pt'
    torch.save(payload, artifact)
    digest = file_sha256(artifact)
    baseline = load_native_teacher(artifact, digest, device='cpu', native_provenance=native[0])
    with pytest.raises(ValueError, match='native collection source differ'):
        load_native_teacher(artifact, digest, device='cpu', native_provenance=native[1])
    teacher = load_native_teacher(artifact, digest, device='cpu', native_provenance=native[1],
                                  course_migrations=(proof,))
    assert teacher.source_metadata == baseline.source_metadata
    assert teacher.source_metadata['qualified_for'] == ['foundation']
    assert teacher.source_metadata['qualification']['report']['native_source'] == native[0]
    history = torch.randn(3, 5, 35)
    for actual, expected in zip(teacher.action_and_estimate(history), baseline.action_and_estimate(history)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert file_sha256(artifact) == digest
    for field, value in (('runtime_sha256', 'unregistered'), ('upstream_commit', 'other-source'),
                         ('action_bounds', [12.] * 4 + [9.] * 2)):
        with pytest.raises(ValueError, match='native collection source differ'):
            load_native_teacher(artifact, digest, device='cpu', native_provenance={**native[1], field: value},
                                course_migrations=(proof,))
    forged = deepcopy(proof)
    forged['target_course_utf8'] += ' '
    with pytest.raises(ValueError, match='source pin'):
        load_native_teacher(artifact, digest, device='cpu', native_provenance=native[1],
                            course_migrations=(forged,))


def test_entry_only_reads_authenticated_receipts_from_matching_complete_checkpoint(tmp_path, candidate):
    saved, data, target, options = candidate
    upgraded, _, report = upgrade_course_sampling(saved, data, target, **options)
    read = runpy.run_path(str(ROOT / 'scripts/train_world_v2.py'))['read_course_migrations']
    checkpoint = tmp_path / 'checkpoint.pt'
    torch.save(upgraded, checkpoint)
    assert read(None, target) == ()
    assert read(checkpoint, target) == (report['course_sampling'],)
    assert not (tmp_path / upgraded['replay_file']).exists()  # receipt read never opens replay
    mutations = [
        {**upgraded, 'format': 'arbitrary-script-receipt'},
        {**upgraded, 'learner': {}},
        {**upgraded, 'schema_fingerprint': 'not-the-schema'},
        {**upgraded, 'resume_migrations': [
            {**report, 'migration': 'unknown-script-dict'}]},
        {**upgraded, 'resume_migrations': [
            {**report, 'course_sampling': {**report['course_sampling'], 'target_course_utf8': '{}'}}]},
    ]
    for invalid in mutations:
        torch.save(invalid, checkpoint)
        with pytest.raises(ValueError):
            read(checkpoint, target)
    torch.save(upgraded, checkpoint)
    with pytest.raises(ValueError, match='complete native contract'):
        read(checkpoint, replace(target, observation_dim=45))
    live = json.loads(target.source)
    with pytest.raises(ValueError, match='complete native contract'):
        read(checkpoint, replace(target, source=json.dumps({**live, 'runtime_sha256': 'different'})))
    # A valid but disconnected course proof is not a license to alias this checkpoint.
    wrong = {**upgraded, 'schema': saved['schema'], 'schema_fingerprint': saved['schema_fingerprint']}
    torch.save(wrong, checkpoint)
    with pytest.raises(ValueError, match='does not reach'):
        read(checkpoint, StateSchema.from_dict(saved['schema']))
