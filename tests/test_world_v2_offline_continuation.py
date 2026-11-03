"""WM-only continuation retains source partitions and resumes its own optimizer/RNG."""
import json
import runpy

import pytest
import torch

from supermimic_rl.world_model_v2.fitting import WorldModelFit
from supermimic_rl.world_model_v2.validation import validate
from support.world_v2.offline_continuation import ROOT, fixture

SCRIPT = ROOT / 'scripts/fit_world_v2.py'


@pytest.mark.parametrize('wheel_contact_weight', [0., 1.])
def test_online_world_only_h8_split_optimizer_and_exact_resumption(tmp_path, monkeypatch, wheel_contact_weight):
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
    candidate = ['--wheel-contact-weight', str(wheel_contact_weight)]
    run(['--online-checkpoint', str(checkpoint), '--iterations', '1', '--output', str(first), *candidate])
    first_state = torch.load(first.with_suffix('.checkpoint.pt'), weights_only=True)
    assert set(first_state) >= {'model', 'optimizer', 'generator', 'source', 'schema'}
    assert 'learner' not in first_state and 'actor' not in first_state and 'critic' not in first_state
    assert first_state['configuration']['train_env_ids'] == [0]
    assert first_state['configuration']['validation_env_ids'] == [1]
    assert first_state['configuration']['wheel_contact_weight'] == wheel_contact_weight
    first_terms = first_state['history'][0]
    if wheel_contact_weight:
        assert first_terms['wheel_contact'] >= 0
        assert first_terms['loss'] == pytest.approx(
            first_terms['state'] + first_terms['ticks'] + wheel_contact_weight * first_terms['wheel_contact'])
    else:
        assert first_terms['wheel_contact'] is None
        assert first_terms['loss'] == pytest.approx(first_terms['state'] + first_terms['ticks'])
    assert first_state['source']['initial_model_updates'] == 42
    assert all(value['step'] == 2 for value in first_state['optimizer']['state'].values())
    unavailable = tmp_path / 'online-unavailable.pt'
    checkpoint.rename(unavailable)
    run(['--resume', str(first.with_suffix('.checkpoint.pt')), '--iterations', '3', '--output', str(restored)])
    unavailable.rename(checkpoint)
    run(['--online-checkpoint', str(checkpoint), '--iterations', '3', '--output', str(full), *candidate])
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
    assert actual['configuration']['wheel_contact_weight'] == wheel_contact_weight
    with pytest.raises(ValueError, match='model/data/training contract differs'):
        run(['--resume', str(first.with_suffix('.checkpoint.pt')), '--iterations', '2',
             '--wheel-contact-weight', str(1. - wheel_contact_weight)])


def test_online_world_continuation_refuses_a_different_holdout_split(tmp_path, monkeypatch):
    checkpoint, _, _ = fixture(tmp_path)
    monkeypatch.setattr('sys.argv', ['fit_world_v2', '--online-checkpoint', str(checkpoint),
                                   '--validation-fraction', '.25', '--device', 'cpu'])
    with pytest.raises(ValueError, match='held-out environment IDs'):
        runpy.run_path(str(SCRIPT), run_name='__main__')


def test_cpu_paired_replay_is_a_readonly_mmap_view(tmp_path, monkeypatch):
    checkpoint, _, _ = fixture(tmp_path)
    script = runpy.run_path(str(SCRIPT))
    source, schema, _, _ = script['online_source'](checkpoint)
    from supermimic_rl.world_model_v2.replay import CompactReplay
    original = CompactReplay.from_state_dict
    observed = []

    def capture(cls, payload, device, **kwargs):
        replay = original(payload, device, **kwargs)
        observed.append(kwargs)
        assert replay.data['ticks'].data_ptr() == payload['storage']['ticks'].data_ptr()
        return replay

    monkeypatch.setattr(CompactReplay, 'from_state_dict', classmethod(capture))
    replay = script['paired_replay'](source, schema, 'cpu')
    assert observed == [{'read_only': True}] and replay.read_only
    with pytest.raises(RuntimeError, match='read-only'):
        replay.add({}, {})


def test_event_fit_normalizes_uniform_prefix_scores_full_batch_and_resumes_exactly(tmp_path, monkeypatch):
    checkpoint, _, _ = fixture(tmp_path)
    seen_observe, seen_terms = [], []
    original_observe, original_terms = WorldModelFit.observe, WorldModelFit.terms

    def observe(self, windows):
        seen_observe.append(windows['action'].shape[0])
        assert windows['action'].shape == (3, 8, 6)
        assert not windows['action'][..., 0].any()
        return original_observe(self, windows)

    def terms(self, windows):
        seen_terms.append(windows['action'].shape[0])
        assert windows['action'].shape == (4, 8, 6)
        assert not windows['action'][..., 0].any()
        return original_terms(self, windows)

    monkeypatch.setattr(WorldModelFit, 'observe', observe)
    monkeypatch.setattr(WorldModelFit, 'terms', terms)
    common = ['fit_world_v2', '--device', 'cpu', '--validation-horizon', '8', '--validation-batch', '2',
              '--identified-bundle', str(ROOT / 'configs/identified/20261031'),
              '--linkage', str(ROOT / 'configs/linkage/leg_linkage_e726d23c.json')]

    def run(extra):
        monkeypatch.setattr('sys.argv', common + extra)
        runpy.run_path(str(SCRIPT), run_name='__main__')

    first, resumed, full = (tmp_path / name for name in ('event.json', 'event-next.json', 'event-full.json'))
    run(['--online-checkpoint', str(checkpoint), '--contact-event-fraction', '.25',
         '--iterations', '1', '--output', str(first)])
    run(['--resume', str(first.with_suffix('.checkpoint.pt')), '--iterations', '2', '--output', str(resumed)])
    run(['--online-checkpoint', str(checkpoint), '--contact-event-fraction', '.25',
         '--iterations', '2', '--output', str(full)])
    actual = torch.load(resumed.with_suffix('.checkpoint.pt'), weights_only=True)
    expected = torch.load(full.with_suffix('.checkpoint.pt'), weights_only=True)
    assert actual['configuration']['contact_event_fraction'] == .25
    assert actual['configuration']['normalization'] == 'update_on_uniform_prefix_only'
    assert seen_observe == [3] * 4 and seen_terms == [4] * 4
    assert actual['history'][-1]['uniform_windows'] == 3
    assert actual['history'][-1]['contact_event_windows'] == 1
    for name in actual['model']:
        torch.testing.assert_close(actual['model'][name], expected['model'][name], rtol=0, atol=0)
    assert torch.equal(actual['generator'], expected['generator'])
    with pytest.raises(ValueError, match='model/data/training contract differs'):
        run(['--resume', str(first.with_suffix('.checkpoint.pt')), '--contact-event-fraction', '0', '--iterations', '2'])


def test_legacy_uniform_checkpoint_without_event_setting_resumes_as_fraction_zero(tmp_path, monkeypatch):
    checkpoint, _, _ = fixture(tmp_path)
    common = ['fit_world_v2', '--device', 'cpu', '--validation-horizon', '8', '--validation-batch', '2',
              '--identified-bundle', str(ROOT / 'configs/identified/20261031'),
              '--linkage', str(ROOT / 'configs/linkage/leg_linkage_e726d23c.json')]

    def run(extra):
        monkeypatch.setattr('sys.argv', common + extra)
        runpy.run_path(str(SCRIPT), run_name='__main__')

    first, output = tmp_path / 'uniform.json', tmp_path / 'legacy-next.json'
    run(['--online-checkpoint', str(checkpoint), '--iterations', '1', '--output', str(first)])
    path = first.with_suffix('.checkpoint.pt')
    saved = torch.load(path, weights_only=True)
    del saved['configuration']['contact_event_fraction']
    del saved['configuration']['wheel_contact_weight']
    torch.save(saved, path)
    run(['--resume', str(path), '--iterations', '2', '--output', str(output)])
    configuration = json.loads(output.read_text())['configuration']
    assert configuration['contact_event_fraction'] == 0.
    assert configuration['wheel_contact_weight'] == 0.
    assert configuration['normalization'] == 'update_on_real_training_windows'


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
