"""Appending recovery data retains the existing independent flight-data split."""
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.world_model_v2.auxiliary_data import AuxiliaryWorldData
from supermimic_rl.world_model_v2.online import OnlineConfig, OnlineTrainer
from supermimic_rl.world_model_v2.records import MODEL_CONTRACT
from supermimic_rl.world_model_v2.teacher import file_sha256
from test_world_v2_auxiliary_data import dataset, windows


@pytest.fixture
def sources(tmp_path):
    old_root, new_root = tmp_path / 'flight', tmp_path / 'recovery'
    old_root.mkdir()
    new_root.mkdir()
    schema, live, old = dataset(old_root)
    _, _, new = dataset(new_root)
    path = Path(new.path)
    payload = torch.load(path, weights_only=True)
    actions = payload['replay']['storage']['action']
    actions[:, 0] += 1900  # Recovery training rows: 2000..2007.
    actions[:, 1] += 8000  # Recovery held-out rows: 9000..9007.
    torch.save(payload, path)
    old = replace(old, name='student750_flight', fraction=.06108096729431678)
    new = replace(new, name='student1000_recovery16', sha256=file_sha256(path),
                  fraction=(1 - old.fraction) * 47037 / (3347440 + 47037))
    return schema, live, old, new


def test_appending_source_restores_original_rng_and_counts_without_training_on_heldout_rows(sources):
    schema, live, old, new = sources
    previous = AuxiliaryWorldData((old,), schema, live, 2, seed=13)
    previous.mix(windows(live, 256))
    torch.rand(17, generator=previous.validation_generator)
    state = previous.state_dict()
    appended = AuxiliaryWorldData((old, new), schema, live, 2, seed=999)
    appended.load_state_dict(state)
    assert appended.sources[0][0] == old
    assert appended.sources[1][0] == new
    restored = appended.state_dict()
    for name in ('sample_rng', 'validation_rng'):
        assert torch.equal(restored[name], state[name])
    assert restored['sampled_windows'] == {old.name: state['sampled_windows'][old.name], new.name: 0}

    # Keeping the old source first preserves its first resumed draw exactly.
    # Later draws legitimately use a shared RNG advanced by the new source.
    baseline = previous.mix(windows(live, 4096))['action']
    actual = appended.mix(windows(live, 4096))['action']
    old_rows = (actual[:, 0, 0] >= 100) & (actual[:, 0, 0] < 1000)
    new_rows = (actual[:, 0, 0] >= 2000) & (actual[:, 0, 0] < 9000)
    assert old_rows.any() and new_rows.any()
    assert torch.equal(old_rows, baseline[:, 0, 0] >= 100)
    assert torch.equal(actual[old_rows], baseline[old_rows])
    assert ((actual < 1000) | ((actual >= 2000) & (actual < 9000))).all()
    assert appended.sampled[old.name] == state['sampled_windows'][old.name] + int(old_rows.sum())
    assert appended.sampled[new.name] == int(new_rows.sum())

    continuation = AuxiliaryWorldData((old, new), schema, live, 2, seed=101)
    continuation.load_state_dict(appended.state_dict())
    expected, actual = appended.mix(windows(live, 512)), continuation.mix(windows(live, 512))
    for name in expected:
        assert torch.equal(actual[name], expected[name])
    assert file_sha256(old.path) == old.sha256
    assert file_sha256(new.path) == new.sha256


def test_online_resume_accepts_second_dataset_but_still_checks_learning_recipe(sources):
    schema, _, old, new = sources
    before = OnlineConfig(cycles=40, world_datasets=(old,))
    requested = replace(before, world_datasets=(old, new))
    trainer = object.__new__(OnlineTrainer)
    trainer.env, trainer.config = SimpleNamespace(schema=schema), requested
    trainer.teacher_source = trainer.stair_scenario = trainer.jump_scenario = trainer.recovery_scenario = None
    saved = {'format': 'supermimic-online-v2', 'model_contract': MODEL_CONTRACT,
             'schema_fingerprint': schema.fingerprint, 'config': asdict(before), 'cycle': 32}
    trainer._check_saved(saved)
    change = trainer.world_datasets_changed
    assert change['before'][0] == change['after'][0] == asdict(old)
    assert change['after'][1] == asdict(new)
    assert saved['config'] == asdict(before)
    trainer.config = replace(requested, model_batch_size=before.model_batch_size + 1)
    with pytest.raises(ValueError, match='training recipe'):
        trainer._check_saved(saved)


def test_both_diagnostics_use_their_own_validation_rows_and_restored_validation_rng(sources, monkeypatch):
    schema, live, old, new = sources
    previous = AuxiliaryWorldData((old,), schema, live, 2, seed=17)
    torch.rand(13, generator=previous.validation_generator)
    state = previous.state_dict()
    appended = AuxiliaryWorldData((old, new), schema, live, 2, seed=37)
    appended.load_state_dict(state)
    calls = []

    def validation(fit, replay, ids, horizon, batch, generator, device, **kwargs):
        draws = replay.sample_windows(batch, horizon, generator=generator, env_ids=ids,
                                      field_names=('action',))['action']
        calls.append((tuple(ids), draws.clone()))
        return {'minimum_action': float(draws.min())}

    monkeypatch.setattr('supermimic_rl.world_model_v2.auxiliary_data.validate', validation)
    learner = SimpleNamespace(fit=None, imagination=SimpleNamespace(contact_floor=None,
        termination=SimpleNamespace(limits=SimpleNamespace(forbidden_body_load_n=1.))))
    config = OnlineConfig(validation_horizon=2, validation_batch_size=16)
    previous.validate(learner, config, 'cpu')
    report = appended.validate(learner, config, 'cpu')
    assert [row[0] for row in calls] == [(1,), (1,), (1,)]
    assert torch.equal(calls[0][1], calls[1][1])
    assert 1000 <= calls[1][1].min() <= calls[1][1].max() < 1008
    assert 9000 <= calls[2][1].min() <= calls[2][1].max() < 9008
    assert report[old.name]['validation_env_ids'] == list(old.validation_env_ids)
    assert report[new.name]['validation_env_ids'] == list(new.validation_env_ids)
    assert report[old.name]['sampled_training_windows'] == report[new.name]['sampled_training_windows'] == 0
