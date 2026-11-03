"""Report restoration uses saved contact filters and pinned native clock limits."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


spec = importlib.util.spec_from_file_location(
    'report_world_v2_data', Path(__file__).parents[1] / 'scripts/report_world_v2_data.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_saved_imagination_floors_are_restored_without_refitting():
    saved = {'contact_floor': {'body': torch.arange(21).float(), 'base': torch.tensor(4.)}}
    floors = module.checkpoint_contact_floor(saved, 'cpu')
    torch.testing.assert_close(floors[0], saved['contact_floor']['body'])
    torch.testing.assert_close(floors[1], saved['contact_floor']['base'])
    assert module.checkpoint_contact_floor({}, 'cpu') is None


def test_contact_clock_requires_the_recorded_runtime_pin(tmp_path):
    path = tmp_path / 'runtime.json'
    runtime = {'termination_limits': {
        'minimum_normal_height_m': .1, 'minimum_recovery_height_m': .05,
        'failure_tilt_rad': 1., 'forbidden_body_load_n': 1.,
        'forbidden_contact_confirm_s': .04, 'pose_failure_confirm_s': .04,
        'forbidden_body_names': ['base_link', 'M_link1', 'M_link2']}}
    path.write_text(json.dumps(runtime))
    schema = SimpleNamespace(policy_dt_s=.02, substeps_per_step=4,
        source=json.dumps({'runtime_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}))
    termination = module.pinned_termination(schema, path)
    assert termination.dt == .005 and termination.limits.forbidden_contact_confirm_s == .04
    assert module.pinned_termination(schema, None) is None
    path.write_text(json.dumps(runtime) + '\n')
    with pytest.raises(ValueError, match='SHA256'):
        module.pinned_termination(schema, path)


@pytest.mark.parametrize('damage', [None, 'source', 'split'])
def test_online_replay_uses_offline_override_without_stale_contact_floor(tmp_path, monkeypatch, damage):
    from supermimic_rl.contracts import StateSchema
    from supermimic_rl.world_model_v2.replay import CompactReplay

    online_path, replay_path, world_path = [tmp_path / name for name in ('online.pt', 'replay.pt', 'world.pt')]
    online_path.write_bytes(b'original online pair identity')
    world_path.write_bytes(b'world checkpoint identity')
    schema = SimpleNamespace(fingerprint='same-native-schema')
    online = {'schema': {}, 'schema_fingerprint': schema.fingerprint, 'cycle': 32,
              'replay_file': replay_path.name,
              'learner': {'model': {'marker': 'old'},
                          'contact_floor': {'body': torch.ones(21), 'base': torch.tensor(4.)}},
              'config': {'model_hidden': [8], 'validation_fraction': .25}}
    world = {'format': 'supermimic-offline-world-v2', 'iteration': 200,
             'model_contract': {'version': module.MODEL_CONTRACT, 'schema_fingerprint': schema.fingerprint,
                                'hidden': [8]},
             'model': {'marker': 'new'},
             'source': {'kind': 'online_pair', 'checkpoint_sha256': hashlib.sha256(online_path.read_bytes()).hexdigest()},
             'configuration': {'validation_fraction': .25, 'train_env_ids': list(range(6)),
                               'validation_env_ids': [6, 7]}}
    if damage == 'source':
        world['source']['checkpoint_sha256'] = 'different-online-pair'
    if damage == 'split':
        world['configuration']['validation_env_ids'] = [5, 6]
    payloads = {online_path: online, replay_path: {'cycle': 32, 'schema_fingerprint': schema.fingerprint,
                                                'replay': {}}, world_path: world}
    monkeypatch.setattr(module.torch, 'load', lambda path, **kwargs: payloads[Path(path)])
    monkeypatch.setattr(StateSchema, 'from_dict', lambda value: schema)
    monkeypatch.setattr(CompactReplay, 'from_state_dict', lambda *args, **kwargs: SimpleNamespace(num_envs=8))
    monkeypatch.setattr(module, '_model', lambda schema, args, hidden, state, device: state)

    def report(replay, **kwargs):
        assert kwargs['fit'] == {'marker': 'new'}
        assert kwargs['contact_floor'] is None
        assert kwargs['env_ids'] == [6, 7]
        return {'populations': {'flat': {'coverage': {'transitions': 16}}}}

    monkeypatch.setattr(module, 'report', report)
    output = tmp_path / 'report.json'
    monkeypatch.setattr(sys, 'argv', ['report', '--online-checkpoint', str(online_path),
        '--world-checkpoint', str(world_path), '--linkage', str(tmp_path / 'linkage.json'),
        '--device', 'cpu', '--output', str(output)])
    if damage is not None:
        with pytest.raises(ValueError, match='split differs' if damage == 'split' else 'selected online pair'):
            module.main()
        assert not output.exists()
        return
    module.main()
    result = json.loads(output.read_text())
    assert result['checkpoint_cycle'] == 32
    assert result['prediction_model']['iteration'] == 200
    assert result['prediction_model']['sha256'] == hashlib.sha256(world_path.read_bytes()).hexdigest()
    assert 'different model weights' in result['prediction_model']['contact_floor']


@pytest.mark.parametrize('damage', ['schema', 'version'])
def test_offline_world_override_rejects_incompatible_contract(tmp_path, damage):
    schema = SimpleNamespace(fingerprint='native-schema')
    contract = {'version': module.MODEL_CONTRACT, 'schema_fingerprint': schema.fingerprint}
    contract['schema_fingerprint' if damage == 'schema' else 'version'] = 'different'
    path = tmp_path / 'world.pt'
    torch.save({'format': 'supermimic-offline-world-v2', 'model_contract': contract}, path)
    with pytest.raises(ValueError, match='contracts differ'):
        module.offline_world_source(path, schema)
