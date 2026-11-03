"""WM-only continuation retains source partitions and resumes its own optimizer/RNG."""
import json
import runpy

import pytest
import torch

from supermimic_rl.world_model_v2.fitting import WorldModelFit
from supermimic_rl.world_model_v2.validation import validate
from support.world_v2.offline_continuation import ROOT, fixture

SCRIPT = ROOT / 'scripts/fit_world_v2.py'


def test_online_world_only_h8_split_optimizer_and_exact_resumption(tmp_path, monkeypatch):
    checkpoint, _, _ = fixture(tmp_path)
    original = checkpoint.read_bytes()
    seen = []
    observe = WorldModelFit.observe

    def measured(self, windows):
        seen.append(windows['action'].shape)
        assert windows['action'].shape == (4, 8, 6)
        assert not windows['action'][..., 0].any()
        return observe(self, windows)

    monkeypatch.setattr(WorldModelFit, 'observe', measured)
    common = ['fit_world_v2', '--device', 'cpu', '--replay-device', 'cpu', '--validation-horizon', '8',
              '--validation-batch', '2', '--identified-bundle', str(ROOT / 'configs/identified/20261031'),
              '--linkage', str(ROOT / 'configs/linkage/leg_linkage_e726d23c.json')]

    def run(extra):
        monkeypatch.setattr('sys.argv', common + extra)
        runpy.run_path(str(SCRIPT), run_name='__main__')

    first, restored, full = (tmp_path / name for name in ('first.json', 'restored.json', 'full.json'))
    run(['--online-checkpoint', str(checkpoint), '--iterations', '1', '--output', str(first)])
    first_state = torch.load(first.with_suffix('.checkpoint.pt'), weights_only=True)
    assert set(first_state) >= {'model', 'optimizer', 'generator', 'source', 'schema'}
    assert 'learner' not in first_state and 'actor' not in first_state and 'critic' not in first_state
    assert first_state['configuration']['train_env_ids'] == [0]
    assert first_state['configuration']['validation_env_ids'] == [1]
    assert first_state['source']['initial_model_updates'] == 42
    assert all(value['step'] == 2 for value in first_state['optimizer']['state'].values())
    unavailable = tmp_path / 'online-unavailable.pt'
    checkpoint.rename(unavailable)
    run(['--resume', str(first.with_suffix('.checkpoint.pt')), '--iterations', '3', '--output', str(restored)])
    unavailable.rename(checkpoint)
    run(['--online-checkpoint', str(checkpoint), '--iterations', '3', '--output', str(full)])
    actual = torch.load(restored.with_suffix('.checkpoint.pt'), weights_only=True)
    expected = torch.load(full.with_suffix('.checkpoint.pt'), weights_only=True)
    for name in actual['model']:
        torch.testing.assert_close(actual['model'][name], expected['model'][name], rtol=0, atol=0)
    for key, values in actual['optimizer']['state'].items():
        for name, value in values.items():
            torch.testing.assert_close(value, expected['optimizer']['state'][key][name], rtol=0, atol=0)
    assert torch.equal(actual['generator'], expected['generator'])
    assert checkpoint.read_bytes() == original
    assert len(seen) == 6
    report = json.loads(restored.read_text())
    assert report['validation']['8_step']['sampled_windows'] == 2
    assert 'no Isaac, actor, critic or new PPO-data ingestion' in report['scope']


def test_online_world_continuation_refuses_a_different_holdout_split(tmp_path, monkeypatch):
    checkpoint, _, _ = fixture(tmp_path)
    monkeypatch.setattr('sys.argv', ['fit_world_v2', '--online-checkpoint', str(checkpoint),
                                   '--validation-fraction', '.25', '--device', 'cpu'])
    with pytest.raises(ValueError, match='held-out environment IDs'):
        runpy.run_path(str(SCRIPT), run_name='__main__')


def test_resume_allows_only_explicit_batch_change_and_preserves_adam_rng_and_source(tmp_path, monkeypatch):
    checkpoint, _, replay = fixture(tmp_path)
    common = ['fit_world_v2', '--device', 'cpu', '--replay-device', 'cpu', '--validation-horizon', '8',
              '--validation-batch', '2', '--identified-bundle', str(ROOT / 'configs/identified/20261031'),
              '--linkage', str(ROOT / 'configs/linkage/leg_linkage_e726d23c.json')]

    def run(extra):
        monkeypatch.setattr('sys.argv', common + extra)
        runpy.run_path(str(SCRIPT), run_name='__main__')

    first, changed, repeated = (tmp_path / name for name in ('batch4.json', 'batch8.json', 'batch8-next.json'))
    run(['--online-checkpoint', str(checkpoint), '--iterations', '1', '--output', str(first)])
    before = torch.load(first.with_suffix('.checkpoint.pt'), weights_only=True)
    expected_rng = torch.Generator().set_state(before['generator'])
    replay.sample_windows(8, 8, generator=expected_rng, env_ids=[0])
    run(['--resume', str(first.with_suffix('.checkpoint.pt')), '--batch', '8',
         '--iterations', '2', '--output', str(changed)])
    after = torch.load(changed.with_suffix('.checkpoint.pt'), weights_only=True)
    assert after['configuration'] == {**before['configuration'], 'batch': 8}
    assert after['source'] == before['source']
    assert after['model_contract'] == before['model_contract']
    assert torch.equal(after['generator'], expected_rng.get_state())
    assert all(value['step'] == 3 for value in after['optimizer']['state'].values())
    change, = after['resume_changes']
    assert (change['before'], change['after'], change['after_iteration'], change['effective_iteration']) == (4, 8, 1, 2)
    assert 'not original-training equivalence' in change['scope']
    assert after['history'][-1]['batch'] == 8
    assert json.loads(changed.read_text())['resume_changes'] == after['resume_changes']
    run(['--resume', str(changed.with_suffix('.checkpoint.pt')),
         '--iterations', '3', '--output', str(repeated)])
    next_state = torch.load(repeated.with_suffix('.checkpoint.pt'), weights_only=True)
    assert next_state['configuration']['batch'] == 8
    assert next_state['resume_changes'] == after['resume_changes']
    for option, value in (('--seed', '3'), ('--learning-rate', '.001'),
                          ('--validation-fraction', '.25'), ('--window', '4'), ('--hidden', '8')):
        with pytest.raises(ValueError, match='model/data/training contract differs'):
            run(['--resume', str(first.with_suffix('.checkpoint.pt')), '--batch', '8',
                 '--iterations', '2', option, value])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Cross-device replay requires CUDA')
def test_cpu_replay_validation_transfers_complete_windows_to_cuda(tmp_path):
    _, model, replay = fixture(tmp_path)
    model.to('cuda')
    fit = WorldModelFit(model)
    report = validate(fit, replay, [1], 8, 2, torch.Generator(device='cpu').manual_seed(4), 'cuda')
    assert replay.device.type == 'cpu'
    assert next(model.parameters()).device.type == 'cuda'
    assert report['8_step']['sampled_windows'] == 2
