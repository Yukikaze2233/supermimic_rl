"""Geometry migration preserves learning and excludes only changed source tiles."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest
import torch

from supermimic_rl.contracts import Field, StateSchema
from supermimic_rl.world_model_v2.auxiliary_data import AuxiliaryWorldData, WorldDatasetConfig
from supermimic_rl.world_model_v2.migrations.terrain import (
    compatible_terrain_source, exclude_changed_tiles, upgrade_terrain, verify_atlas_repair,
)
from supermimic_rl.world_model_v2.records import MODEL_CONTRACT
from supermimic_rl.world_model_v2.replay import CompactReplay
from supermimic_rl.world_model_v2.teacher import PPO_TEACHER_FORMAT, file_sha256
from test_world_v2_auxiliary_data import dataset as auxiliary_fixture, windows


def fixture(tmp_path):
    runtimes, sources = [], []
    for index in range(2):
        root = tmp_path / str(index)
        root.mkdir()
        arrays = dict(vertices_m=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                                          [2, 0, 0], [2, 1, 0], [2, 1, .15]], np.float32),
                      triangles=np.array([[0, 1, 2], [3, 4, 5] if index == 0 else [3, 5, 4]]),
                      face_tile=np.array([0, 1]), face_material=np.array([0, 0]))
        np.savez(root / 'geometry.npz', **arrays)
        file = root / 'geometry.npz'
        manifest = {'format': 'terrain_course_atlas_v1', 'tiles': [{'name': 'flat'}, {'name': 'step_up'}],
                    'files': {file.name: {'file': file.name, 'bytes': file.stat().st_size,
                                         'sha256': file_sha256(file)}}}
        path = root / 'manifest.json'
        path.write_text(json.dumps(manifest))
        pin = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': file_sha256(path)}
        runtime = json.dumps({'physics': 'unchanged', 'terrain_atlas_pin': pin}).encode()
        runtimes.append(runtime)
        sources.append({'terrain_sha256': pin['sha256'], 'runtime_sha256': hashlib.sha256(runtime).hexdigest(),
                        'authored_routing': {'atlas_sha256': pin['sha256'], 'other': 'unchanged'},
                        'reward_kernel': {'implementation': 'shared_physical_reward_densities'},
                        'jump_takeoff': None, 'stair_label': None, 'jump_assist': None})
    schemas = [StateSchema((Field('fixture', (1,)),), (), source=json.dumps(s)) for s in sources]
    replay = CompactReplay({'terrain_tile': (1,), 'physical': (2,)}, 2, 4, 'cpu')
    replay.size = replay.head = replay.total_steps = 4
    replay.sequence[:] = torch.arange(4)
    replay.data['valid'].fill_(True)
    replay.data['terrain_tile'][:, 1] = 1
    saved = {'format': 'supermimic-online-v2', 'model_contract': MODEL_CONTRACT, 'cycle': 100,
             'schema': schemas[0].to_dict(), 'schema_fingerprint': schemas[0].fingerprint,
             'learner': {'weight': torch.randn(3), 'adam': torch.randn(3)},
             'torch_rng': torch.get_rng_state(), 'auxiliary_world': {'counts': 29},
             'environment': {'course': {'router': {'base': {'atlas_sha256': sources[0]['terrain_sha256'],
                                                           'rng': torch.get_rng_state()}}},
                             'control_timing': {'clock': 99}},
             'resume_migrations': []}
    data = {'cycle': 100, 'schema_fingerprint': schemas[0].fingerprint, 'replay': replay.state_dict()}
    reference = {'format': 'supermimic-world-v2-dataset-v1', 'model_contract': MODEL_CONTRACT,
                 'schema': schemas[1].to_dict(), 'schema_fingerprint': schemas[1].fingerprint}
    paths = []
    for i, source in enumerate(sources):
        metadata = {'format': PPO_TEACHER_FORMAT, 'qualified_for': ['data-only'],
                    'initial_checkpoint': {'sha256': 'a' * 64, 'updates': 20, 'transitions': 100},
                    'history_length': 5, 'observation_dim': 35, 'action_bounds': [1.] * 6,
                    'action_semantics': 'clipped', 'history_semantics': 'native',
                    'estimate_semantics': 'velocity/contact', 'native_source': source,
                    'qualification': {'report': {'native_source': source, 'steps': 5,
                        'simulated_seconds': 1., 'checkpoint_sha256': 'a' * 64}}}
        path = tmp_path / f'teacher-{i}.pt'
        torch.save({'format': PPO_TEACHER_FORMAT, 'source': metadata,
                    'actor': {'weight': torch.tensor([1.])}}, path)
        paths.append(path)
        if i == 0:
            saved['teacher_source'] = {**metadata, 'teacher_artifact': {'path': str(path), 'sha256': file_sha256(path)}}
    options = dict(source_runtime=runtimes[0], target_runtime=runtimes[1],
                   source_teacher_path=paths[0], target_teacher_path=paths[1])
    return saved, data, reference, options


def test_complete_migration_preserves_weights_adam_rng_and_physical_labels(tmp_path):
    saved, data, reference, options = fixture(tmp_path)
    upgraded, replay, report = upgrade_terrain(saved, data, reference, **options)
    assert upgraded['learner'] is saved['learner']
    assert upgraded['torch_rng'] is saved['torch_rng']
    assert upgraded['auxiliary_world'] is saved['auxiliary_world']
    assert upgraded['environment']['control_timing'] is saved['environment']['control_timing']
    for name, value in data['replay']['storage'].items():
        if name != 'valid':
            assert replay['replay']['storage'][name] is value
    assert data['replay']['storage']['valid'].all()
    assert replay['replay']['storage']['valid'][:, 0].all()
    assert not replay['replay']['storage']['valid'][:, 1].any()
    assert report['changed_tile_intervals_excluded'] == 4
    assert report['terrain_repair']['changed_tiles'] == [1]
    assert report['terrain_repair']['changed_face_count'] == 1
    assert upgraded['schema_fingerprint'] == reference['schema_fingerprint']


def test_unknown_runtime_or_observation_change_cannot_hide_behind_terrain_repair(tmp_path):
    saved, data, reference, options = fixture(tmp_path)
    altered = json.loads(options['target_runtime'])
    altered['physics'] = 'different'
    with pytest.raises(ValueError, match='other runtime'):
        verify_atlas_repair(options['source_runtime'], json.dumps(altered).encode())
    reference['schema']['observation_dim'] = 45
    reference['schema_fingerprint'] = StateSchema.from_dict(reference['schema']).fingerprint
    with pytest.raises(ValueError, match='other native'):
        upgrade_terrain(saved, data, reference, **options)


def test_unregistered_atlas_pair_does_not_make_auxiliary_sources_compatible(tmp_path):
    saved, _, reference, options = fixture(tmp_path)
    old, new = (StateSchema.from_dict(item['schema']) for item in (saved, reference))
    pair = verify_atlas_repair(options['source_runtime'], options['target_runtime'])
    assert not AuxiliaryWorldData._same_dynamics_contract(old, new)
    assert AuxiliaryWorldData._same_dynamics_contract(old, new, terrain_migrations=(pair,))
    bad = deepcopy(pair)
    bad['runtime_pair']['after'] = 'unregistered'
    assert not AuxiliaryWorldData._same_dynamics_contract(old, new, terrain_migrations=(bad,))
    source = json.loads(new.source)
    source['authored_routing']['atlas_sha256'] = 'different'
    normalized, excluded = compatible_terrain_source(json.loads(old.source), source, (pair,))
    assert normalized == source and excluded == []


def test_auxiliary_changed_tiles_never_enter_training_or_heldout_windows(tmp_path):
    asset_root = tmp_path / 'assets'
    asset_root.mkdir()
    saved, _, reference, options = fixture(asset_root)
    old, new = (StateSchema.from_dict(item['schema']) for item in (saved, reference))
    pair = verify_atlas_repair(options['source_runtime'], options['target_runtime'])
    schema, replay, config = auxiliary_fixture(tmp_path)
    old = replace(schema, source=old.source)
    new = replace(schema, source=new.source)
    replay.data['terrain_tile'] = torch.zeros(8, 2, 1)
    replay.data['terrain_tile'][5] = 1
    path = tmp_path / 'qualified.pt'
    torch.save({'format': 'supermimic-world-v2-dataset-v1', 'model_contract': MODEL_CONTRACT,
                'schema': old.to_dict(), 'schema_fingerprint': old.fingerprint,
                'replay': replay.state_dict()}, path)
    digest = file_sha256(path)
    config = replace(config, path=str(path), sha256=digest)
    aux = AuxiliaryWorldData((config,), new, replay, 2, seed=1, terrain_migrations=(pair,))
    assert aux.terrain_exclusions['flight']['historical_intervals_excluded'] == 2
    assert file_sha256(path) == digest
    result = aux.mix(windows(replay, 2048))['action']
    assert not torch.isin(result, torch.tensor([104., 105.])).any()
    assert aux.sources[0][1].window_candidates(2, [1]).tolist() == [1, 3, 5, 7, 13]
    state = aux.state_dict()
    restored = AuxiliaryWorldData((config,), new, replay, 2, seed=29, terrain_migrations=(pair,))
    restored.load_state_dict(state)
    expected, actual = aux.mix(windows(replay)), restored.mix(windows(replay))
    for name, value in expected.items():
        assert torch.equal(value, actual[name])


def test_tile_eligibility_does_not_clip_labels_or_invent_integer_routes():
    storage = {'terrain_tile': torch.tensor([[[0.], [1.]]]), 'valid': torch.tensor([[True, True]]),
               'ticks': torch.tensor([[[1e9], [-1e9]]])}
    original = storage['ticks'].clone()
    valid, count = exclude_changed_tiles(storage, [1])
    assert valid.tolist() == [[True, False]] and count == 1
    assert torch.equal(storage['ticks'], original) and storage['valid'].all()
    storage['terrain_tile'][0, 0] = .5
    with pytest.raises(ValueError, match='integer routes'):
        exclude_changed_tiles(storage, [1])


def test_terrain_identity_mapping_composes_with_verified_zero_disturbance_history(tmp_path):
    from test_world_v2_disturbance_upgrade import fixture as disturbance_fixture

    saved, _, reference, options, *_ = disturbance_fixture(tmp_path)
    schemas = []
    for item in (saved, reference):
        schema = StateSchema.from_dict(item['schema'])
        source = json.loads(schema.source)
        source.update(terrain_sha256='old-atlas', authored_routing={'atlas_sha256': 'old-atlas'})
        schemas.append(replace(schema, source=json.dumps(source)))
    live = json.loads(schemas[1].source)
    terrain_pair = {'source_atlas_sha256': 'old-atlas', 'target_atlas_sha256': 'new-atlas',
                    'runtime_pair': {'before': live['runtime_sha256'], 'after': 'new-runtime'},
                    'changed_tiles': [1]}
    live.update(terrain_sha256='new-atlas', runtime_sha256='new-runtime',
                authored_routing={'atlas_sha256': 'new-atlas'})
    live = replace(schemas[1], source=json.dumps(live))
    disturbance_pair = {'before': hashlib.sha256(options['source_runtime']).hexdigest(),
                        'after': hashlib.sha256(options['target_runtime']).hexdigest()}
    assert not AuxiliaryWorldData._same_dynamics_contract(schemas[0], live, (disturbance_pair,))
    assert AuxiliaryWorldData._same_dynamics_contract(
        schemas[0], live, (disturbance_pair,), (terrain_pair,))
