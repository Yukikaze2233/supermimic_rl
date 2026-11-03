"""Rejoin fixed-replay WM fitting without replacing policy ancestry or held-out IDs."""
from copy import deepcopy
import hashlib
from pathlib import Path
import runpy

import pytest
import torch

from supermimic_rl.world_model_v2.migrations.actor import inject_fitted_world, restrict_actor_to_chassis
from support.world_v2.world_rejoin import pair


def test_injection_keeps_policy_ancestry_and_loads_fitted_model_normalization_adam(tmp_path):
    _, saved, dataset, world, model = pair(tmp_path)
    branch, _, actor_receipt = restrict_actor_to_chassis(saved, dataset, replay_device='cpu',
        sampling_seed=123, imagination_microbatch_size=4096)
    combined, receipt = inject_fitted_world(branch, dataset, world,
        source_checkpoint_sha256=world['source']['checkpoint_sha256'])
    for name in ('actor', 'critic'):
        assert combined['learner'][name] is branch['learner'][name]
    for name in ('actor', 'critic', 'estimator'):
        assert combined['learner']['optimizers'][name] is branch['learner']['optimizers'][name]
        assert combined['learner']['updates'][name] == saved['learner']['updates'][name]
        assert combined['learner']['skipped'][name] == saved['learner']['skipped'][name]
    assert combined['learner']['model'] is world['model']
    assert combined['learner']['optimizers']['model'] is world['optimizer']
    assert combined['learner']['updates']['model'] == 46
    assert combined['learner']['skipped']['model'] == 3
    assert saved['learner']['updates']['model'] == 42
    assert branch['learner']['updates']['model'] == 42
    assert actor_receipt['ancestor_updates']['model'] == 42
    assert combined['sample_rng'] is branch['sample_rng']
    assert not torch.equal(combined['sample_rng'], world['generator'])
    assert combined['config'] is branch['config']
    assert combined['cycle'] == saved['cycle']
    assert combined['resume_migrations'] == [actor_receipt, receipt]
    assert receipt['offline_configuration']['validation_env_ids'] == [1]
    assert receipt['offline_resume_changes'] == world['resume_changes']
    model.load_state_dict(combined['learner']['model'])
    assert model.motion.input_norm.count == 3
    restored = torch.optim.Adam(model.parameters())
    restored.load_state_dict(combined['learner']['optimizers']['model'])
    assert all(value['step'] == 5 for value in restored.state.values())
    restored.zero_grad(set_to_none=True)
    sum(parameter.square().mean() for parameter in model.parameters()).backward()
    restored.step()
    assert all(value['step'] == 6 for value in restored.state.values())


@pytest.mark.parametrize('mismatch', ['ancestor', 'counts', 'cycle', 'split', 'hidden', 'shape', 'adam', 'iterations'])
def test_injection_rejects_changed_ancestor_data_partition_or_model_contract(tmp_path, mismatch):
    _, saved, dataset, world, _ = pair(tmp_path)
    world = deepcopy(world)
    if mismatch == 'ancestor':
        world['source']['checkpoint_sha256'] = 'unrelated checkpoint'
    elif mismatch == 'counts':
        world['source']['initial_model_updates'] += 1
    elif mismatch == 'cycle':
        dataset = {**dataset, 'cycle': dataset['cycle'] + 1}
    elif mismatch == 'split':
        world['configuration']['train_env_ids'] = [0, 1]
    elif mismatch == 'hidden':
        world['model_contract']['hidden'] = [32]
    elif mismatch == 'shape':
        world['model']['motion.input_norm.mean'] = torch.zeros(1)
    elif mismatch == 'adam':
        world['optimizer']['param_groups'][0]['lr'] = .01
    else:
        world['skipped'] = world['iteration'] + 1
    with pytest.raises(ValueError):
        inject_fitted_world(saved, dataset, world,
            source_checkpoint_sha256=hashlib.sha256((tmp_path / 'online.pt').read_bytes()).hexdigest())


def test_branch_cli_injects_fitted_world_and_preserves_immutable_replay(tmp_path, monkeypatch):
    checkpoint, saved, _, world, _ = pair(tmp_path)
    source_bytes = checkpoint.read_bytes()
    world_path = tmp_path / 'fit.checkpoint.pt'
    torch.save(world, world_path)
    output = tmp_path / 'chassis-with-fitted-world'
    monkeypatch.setattr('sys.argv', ['upgrade_online_chassis_actor', '--checkpoint', str(checkpoint),
        '--output', str(output), '--world-fit-checkpoint', str(world_path), '--replay-device', 'cpu',
        '--sampling-seed', '77', '--imagination-microbatch-size', '4096'])
    root = Path(__file__).resolve().parents[1]
    runpy.run_path(str(root / 'scripts/upgrade_online_chassis_actor.py'), run_name='__main__')
    combined = torch.load(output / 'resume-a.pt', weights_only=True)
    assert checkpoint.read_bytes() == source_bytes
    assert (tmp_path / saved['replay_file']).stat().st_ino == (output / 'replay-a.pt').stat().st_ino
    assert combined['config']['actor_frame_dim'] == 35
    assert combined['config']['replay_device'] == 'cpu'
    assert combined['config']['imagination_microbatch_size'] == 4096
    assert combined['learner']['updates']['model'] == 46
    receipt = combined['resume_migrations'][-1]
    assert receipt['checkpoint_sha256'] == hashlib.sha256(world_path.read_bytes()).hexdigest()
    for name, value in world['model'].items():
        torch.testing.assert_close(combined['learner']['model'][name], value, rtol=0, atol=0)
    actor = torch.load(output / 'actor-000078.pt', weights_only=True)
    assert actor['actor_branch']['ancestor_updates'] == saved['learner']['updates']
    assert actor['actor_branch']['world_fit'] == receipt
    assert actor['updates']['model'] == 46
