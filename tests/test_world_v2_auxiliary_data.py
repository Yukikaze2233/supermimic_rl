"""Independent skill records retain their boundaries, splits and resume RNGs."""
from dataclasses import asdict, replace
import hashlib
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.contracts import Field, StateSchema
from supermimic_rl.world_model_v2.auxiliary_data import AuxiliaryWorldData, WorldDatasetConfig
from supermimic_rl.world_model_v2.fitting import WorldModelFit
from supermimic_rl.world_model_v2.online import OnlineConfig, OnlineTrainer
from supermimic_rl.world_model_v2.records import MODEL_CONTRACT, RecordSpec
from supermimic_rl.world_model_v2.replay import CompactReplay


def dataset(tmp_path):
    schema = StateSchema((Field('fixture', (1,)),), (), goal_dim=16, observation_dim=45)
    shapes = RecordSpec(16, 65).fields()
    replay = CompactReplay(shapes, 2, 8, 'cpu')
    boundary = {k: torch.zeros(2, dtype=torch.bool) for k in ('terminated', 'truncated')}
    boundary.update({k: torch.ones(2, dtype=torch.bool)
                     for k in ('valid', 'policy_owned', 'end_policy_owned')})
    boundary.update({k: torch.zeros(2, dtype=torch.long) for k in
                     ('start_episode_id', 'end_episode_id', 'start_control_epoch', 'end_control_epoch')})
    for step in range(8):
        record = {name: torch.zeros(2, *shape) for name, shape in shapes.items()}
        record['physical'][:, 6] = record['end_physical'][:, 6] = 1.
        record['action'][0] = 100 + step
        record['action'][1] = 1000 + step
        boundary['valid'][0] = step != 3
        replay.add(record, boundary)
    path = tmp_path / 'native.pt'
    torch.save({'format': 'supermimic-world-v2-dataset-v1', 'model_contract': MODEL_CONTRACT,
                'schema': schema.to_dict(), 'schema_fingerprint': schema.fingerprint,
                'replay': replay.state_dict()}, path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    config = WorldDatasetConfig('flight', str(path), sha, .4, (0,), (1,))
    return schema, replay, config


def windows(replay, batch=128, horizon=2):
    names = WorldModelFit.RECORD_FIELDS + tuple(
        k for k in WorldModelFit.OPTIONAL_RECORD_FIELDS if k in replay.data)
    return {k: torch.full((batch, horizon, *replay.data[k].shape[2:]),
                         -1, dtype=replay.data[k].dtype) for k in names}


def test_skill_windows_do_not_cross_invalid_rows_or_read_heldout_environments(tmp_path):
    schema, replay, config = dataset(tmp_path)
    aux = AuxiliaryWorldData((config,), schema, replay, 2, seed=19)
    original = aux.sources[0][1].data['action'].clone()
    result = aux.mix(windows(replay, 2048))['action']
    selected = result[:, 0, 0] >= 100
    assert 600 < int(selected.sum()) < 1000
    assert (result[selected] < 1000).all()
    starts = result[selected, 0, 0].unique().tolist()
    assert set(starts) == {100., 101., 104., 105., 106.}
    assert (result[selected, 1] - result[selected, 0] == 1).all()
    assert torch.equal(aux.sources[0][1].data['action'], original)
    assert aux.sampled['flight'] == int(selected.sum())


def test_resume_reproduces_the_next_auxiliary_draw_and_keeps_live_rng_separate(tmp_path):
    schema, replay, config = dataset(tmp_path)
    aux = AuxiliaryWorldData((config,), schema, replay, 2, seed=19)
    aux.mix(windows(replay))
    state = aux.state_dict()
    expected = aux.mix(windows(replay))
    restored = AuxiliaryWorldData((config,), schema, replay, 2, seed=900)
    restored.load_state_dict(state)
    actual = restored.mix(windows(replay))
    for name in expected:
        assert torch.equal(actual[name], expected[name])
    assert aux.sampled == restored.sampled
    assert torch.equal(aux.validation_generator.get_state(), restored.validation_generator.get_state())


def test_online_sampling_keeps_the_live_draw_and_only_replaces_selected_world_rows(tmp_path):
    schema, replay, config = dataset(tmp_path)
    trainer = object.__new__(OnlineTrainer)
    trainer.config = OnlineConfig(world_datasets=(config,))
    trainer.device = torch.device('cpu')
    trainer.collector = SimpleNamespace(replay=replay)
    trainer.train_ids = [0]
    trainer.generator = torch.Generator().manual_seed(57)
    expected_rng = torch.Generator().set_state(trainer.generator.get_state())
    trainer.auxiliary_world = AuxiliaryWorldData((config,), schema, replay, 2, seed=19)
    replay.sample_windows(32, 2, generator=expected_rng, env_ids=[0])
    trainer.sample_windows(32, 2)
    assert torch.equal(trainer.generator.get_state(), expected_rng.get_state())
    assert trainer.auxiliary_world.sampled['flight'] > 0


def test_only_auxiliary_validation_rows_reach_the_diagnostic(tmp_path, monkeypatch):
    schema, replay, config = dataset(tmp_path)
    aux = AuxiliaryWorldData((config,), schema, replay, 2, seed=19)
    calls = []

    def check(fit, data, ids, *args, **kwargs):
        calls.append(tuple(ids))
        raise ValueError('diagnostic unavailable')

    monkeypatch.setattr('supermimic_rl.world_model_v2.auxiliary_data.validate', check)
    learner = SimpleNamespace(fit=None, imagination=SimpleNamespace(contact_floor=None,
        termination=SimpleNamespace(limits=SimpleNamespace(forbidden_body_load_n=1.))))
    report = aux.validate(learner, OnlineConfig(), 'cpu')
    assert calls == [(1,)]
    assert report['flight']['prediction'] == {'error': 'diagnostic unavailable'}


def test_dataset_pins_and_layout_are_checked_before_sampling(tmp_path):
    schema, replay, config = dataset(tmp_path)
    with pytest.raises(ValueError, match='SHA256 differs'):
        AuxiliaryWorldData((replace(config, sha256='0' * 64),), schema, replay, 2, seed=19)
    with pytest.raises(ValueError, match='exceeds'):
        AuxiliaryWorldData((replace(config, validation_env_ids=(2,)),), schema, replay, 2, seed=19)
    other = StateSchema((Field('different', (1,)),), (), goal_dim=16, observation_dim=45)
    with pytest.raises(ValueError, match='contract differs'):
        AuxiliaryWorldData((config,), other, replay, 2, seed=19)
    replay.data['action'] = replay.data['action'].double()
    with pytest.raises(ValueError, match='field differs'):
        AuxiliaryWorldData((config,), schema, replay, 2, seed=19)


def test_config_roundtrip_and_explicit_resume_activation_preserve_training_recipe(tmp_path):
    schema, _, source = dataset(tmp_path)
    config = OnlineConfig(cycles=40, world_datasets=(source,))
    assert OnlineConfig.from_dict(asdict(config)) == config
    assert OnlineConfig.from_dict({}).world_datasets == ()
    with pytest.raises(ValueError, match='overlap'):
        replace(source, validation_env_ids=(0,))
    with pytest.raises(ValueError, match='distinct names'):
        OnlineConfig(world_datasets=(source, source))
    with pytest.raises(ValueError, match='fraction below one'):
        OnlineConfig(world_datasets=(replace(source, fraction=.8), replace(source, name='other')))
    trainer = object.__new__(OnlineTrainer)
    trainer.env = SimpleNamespace(schema=schema)
    trainer.config = config
    trainer.teacher_source = trainer.stair_scenario = trainer.jump_scenario = trainer.recovery_scenario = None
    saved = {'format': 'supermimic-online-v2', 'model_contract': MODEL_CONTRACT,
             'schema_fingerprint': schema.fingerprint, 'config': asdict(OnlineConfig()), 'cycle': 32}
    trainer._check_saved(saved)
    assert trainer.world_datasets_changed['before'] == ()
    assert trainer.world_datasets_changed['after'][0]['sha256'] == source.sha256
    trainer.config = replace(config, model_batch_size=1024)
    with pytest.raises(ValueError, match='training recipe'):
        trainer._check_saved(saved)
