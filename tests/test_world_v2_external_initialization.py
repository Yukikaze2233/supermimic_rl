"""A new native dataset can reuse WM/Adam without reusing the old replay split."""
import hashlib
import json
import runpy

import pytest
import torch

from supermimic_rl.contracts import StateSchema
from supermimic_rl.world_model_v2.fitting import WorldModelFit
from supermimic_rl.world_model_v2.records import MODEL_CONTRACT
from support.world_v2.offline_continuation import ROOT, fixture


@pytest.fixture
def native_source(tmp_path):
    checkpoint, _, replay = fixture(tmp_path)
    saved = torch.load(checkpoint, weights_only=True)
    data = replay.state_dict()
    data['num_envs'] = 4
    data['storage'] = {name: value[:, [0, 0, 0, 1]].clone()
                       for name, value in data['storage'].items()}
    data['storage']['action'][..., 1] = .2
    dataset = tmp_path / 'new-native.pt'
    torch.save({'format': 'supermimic-world-v2-dataset-v1', 'model_contract': MODEL_CONTRACT,
                'schema': saved['schema'], 'schema_fingerprint': saved['schema_fingerprint'],
                'replay': data}, dataset)
    # A model-only source must not require the old 15 GB online replay.
    (tmp_path / 'replay-a.pt').rename(tmp_path / 'old-replay-unavailable.pt')
    return checkpoint, dataset


@pytest.fixture
def run_fit(monkeypatch):
    common = ['fit_world_v2', '--device', 'cpu', '--validation-horizon', '8',
              '--validation-batch', '2', '--identified-bundle', str(ROOT / 'configs/identified/20261031'),
              '--linkage', str(ROOT / 'configs/linkage/leg_linkage_e726d23c.json')]

    def run(extra):
        monkeypatch.setattr('sys.argv', common + extra)
        runpy.run_path(str(ROOT / 'scripts/fit_world_v2.py'), run_name='__main__')

    return run


def test_new_data_split_and_adam_resume_without_original_source(native_source, run_fit, tmp_path, monkeypatch):
    checkpoint, dataset = native_source
    observed = []
    original_observe = WorldModelFit.observe

    def observe(self, windows):
        observed.append(windows['action'].shape)
        assert not windows['action'][..., 0].any()  # No held-out row enters fitting.
        torch.testing.assert_close(windows['action'][..., 1], torch.full((4, 8), .2))
        return original_observe(self, windows)

    monkeypatch.setattr(WorldModelFit, 'observe', observe)
    first, resumed, full = [tmp_path / name for name in ('first.json', 'resumed.json', 'full.json')]
    init = [str(dataset), '--initial-online-checkpoint', str(checkpoint), '--validation-fraction', '.25']
    run_fit([*init, '--iterations', '1', '--output', str(first)])
    state = torch.load(first.with_suffix('.checkpoint.pt'), weights_only=True)
    assert state['configuration']['train_env_ids'] == [0, 1, 2]
    assert state['configuration']['validation_env_ids'] == [3]
    assert state['source']['kind'] == 'recorded_dataset'
    assert state['source']['replay_sha256'] == hashlib.sha256(dataset.read_bytes()).hexdigest()
    origin = state['source']['initialization']
    assert origin['kind'] == 'online_model_initialization' and 'replay' not in origin
    assert origin['checkpoint_sha256'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert all(value['step'] == 2 for value in state['optimizer']['state'].values())
    assert state['history'][0]['model_updates'] == 43
    assert not {'actor', 'critic', 'learner'}.intersection(state)

    hidden_checkpoint = checkpoint.with_name('source-unavailable.pt')
    checkpoint.rename(hidden_checkpoint)
    run_fit(['--resume', str(first.with_suffix('.checkpoint.pt')), '--iterations', '3', '--output', str(resumed)])
    hidden_checkpoint.rename(checkpoint)
    run_fit([*init, '--iterations', '3', '--output', str(full)])
    actual = torch.load(resumed.with_suffix('.checkpoint.pt'), weights_only=True)
    expected = torch.load(full.with_suffix('.checkpoint.pt'), weights_only=True)
    assert actual['source'] == expected['source'] == state['source']
    assert actual['history'][-1]['model_updates'] == 45
    for name, value in actual['model'].items():
        torch.testing.assert_close(value, expected['model'][name], rtol=0, atol=0)
    for key, values in actual['optimizer']['state'].items():
        for name, value in values.items():
            torch.testing.assert_close(value, expected['optimizer']['state'][key][name], rtol=0, atol=0)
    assert torch.equal(actual['generator'], expected['generator'])
    assert observed == [torch.Size((4, 8, 6))] * 6
    report = json.loads(resumed.read_text())
    assert 'not the original online replay' in report['scope']
    assert report['validation']['8_step']['sampled_windows'] == 2


def test_external_dataset_must_match_initial_model_schema(native_source, run_fit):
    checkpoint, dataset = native_source
    payload = torch.load(dataset, weights_only=True)
    payload['schema']['source'] = json.dumps({'different_mechanics': True})
    payload['schema_fingerprint'] = StateSchema.from_dict(payload['schema']).fingerprint
    torch.save(payload, dataset)
    with pytest.raises(ValueError, match='initial world-model schemas differ'):
        run_fit([str(dataset), '--initial-online-checkpoint', str(checkpoint), '--iterations', '1'])


def test_external_resume_detects_replaced_dataset(native_source, run_fit, tmp_path):
    checkpoint, dataset = native_source
    output = tmp_path / 'first.json'
    run_fit([str(dataset), '--initial-online-checkpoint', str(checkpoint),
             '--iterations', '1', '--output', str(output)])
    payload = torch.load(dataset, weights_only=True)
    payload['replay']['storage']['action'][0, 0, 0] += .1
    torch.save(payload, dataset)
    with pytest.raises(ValueError, match='dataset changed since initialization'):
        run_fit(['--resume', str(output.with_suffix('.checkpoint.pt')), '--iterations', '2'])
