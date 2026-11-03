"""Task expansion retains learned values/Adam state and excludes invented histories."""
from copy import deepcopy
from types import SimpleNamespace

import torch

from supermimic_rl.world_model_v2.imagination import Critic
from supermimic_rl.world_model_v2.motion_task import MOTION_TASK_DIM
from supermimic_rl.world_model_v2.migrations.motion import upgrade_motion_tasks
from supermimic_rl.world_model_v2.online import OnlineConfig, OnlineTrainer
from supermimic_rl.world_model_v2.records import PHYSICAL_DIM
from supermimic_rl.world_model_v2.replay import CompactReplay
from support.world_v2.motion_upgrade import fixture


def test_expansion_preserves_function_moments_counts_and_source_replay():
    saved, dataset, critic, optimizer, x, _, _ = fixture()
    original_weight = saved['learner']['critic']['net.0.weight'].clone()
    upgraded, data, report = upgrade_motion_tasks(saved, dataset)
    expanded = Critic(x.shape[-1] + MOTION_TASK_DIM, (16, 8), task_dim=MOTION_TASK_DIM)
    expanded.load_state_dict(upgraded['learner']['critic'])
    new_input = torch.cat((x, torch.randn(32, MOTION_TASK_DIM) * 50), -1)
    torch.testing.assert_close(expanded(new_input), critic(x), rtol=1e-6, atol=1e-7)
    assert torch.equal(saved['learner']['critic']['net.0.weight'], original_weight)
    assert 'motion_task' not in dataset['replay']['storage']
    assert upgraded['learner']['model'] is saved['learner']['model']
    assert upgraded['learner']['actor'] is saved['learner']['actor']
    assert upgraded['learner']['updates'] == saved['learner']['updates']
    assert torch.equal(upgraded['sample_rng'], saved['sample_rng'])
    restored_optimizer = torch.optim.Adam(expanded.parameters(), lr=2e-4)
    restored_optimizer.load_state_dict(upgraded['learner']['optimizers']['critic'])
    for key in ('exp_avg', 'exp_avg_sq'):
        old = optimizer.state[critic.net[0].weight][key]
        new = restored_optimizer.state[expanded.net[0].weight][key]
        assert torch.equal(new[:, :x.shape[-1]], old)
        assert (new[:, x.shape[-1]:] == 0).all()
    expanded(new_input).square().mean().backward()
    restored_optimizer.step()  # Moment expansion must support the next real update.
    assert torch.isfinite(expanded.net[0].weight).all()
    assert not data['replay']['storage']['task_record_valid'].any()
    assert report['old_physical_transitions_retained'] == 6 and upgraded['task_norm_pending']


def test_old_data_fits_world_model_but_only_fresh_tasks_seed_imagination():
    saved, dataset, _, _, _, record, boundary = fixture()
    upgraded, data, _ = upgrade_motion_tasks(saved, dataset)
    replay = CompactReplay.from_state_dict(data['replay'])
    assert replay.window_candidates(1).numel() == 6
    assert replay.start_candidates().numel() == 6  # Generic physical ownership remains unchanged.
    record = {**record, 'motion_task': torch.zeros(2, MOTION_TASK_DIM), 'task_record_valid': torch.ones(2, 1)}
    record['frame'].fill_(7.)
    replay.add(record, boundary)
    trainer = object.__new__(OnlineTrainer)
    trainer.device = torch.device('cpu')
    trainer.config = OnlineConfig(record_tasks=True)
    trainer.collector = SimpleNamespace(replay=replay)
    trainer.env = SimpleNamespace(policy_dt_s=.02)
    trainer.learner = SimpleNamespace(actor=SimpleNamespace(history_length=5))
    trainer.train_ids, trainer.generator = [0, 1], torch.Generator().manual_seed(3)
    start, _ = trainer.sample_starts(64)
    assert (start['task_record_valid'] == 1).all()
    assert (start['frame'] == 7).all()
    assert replay.window_candidates(1).numel() == 8


def test_new_task_norm_calibration_does_not_reinitialize_old_columns_or_counts():
    saved, dataset, _, _, x, _, _ = fixture()
    upgraded, _, _ = upgrade_motion_tasks(saved, dataset)
    critic = Critic(x.shape[-1] + MOTION_TASK_DIM, (16, 8), task_dim=MOTION_TASK_DIM)
    critic.load_state_dict(upgraded['learner']['critic'])
    before = deepcopy(critic.norm.state_dict())
    start = {'physical': torch.zeros(32, PHYSICAL_DIM), 'motion_task': torch.randn(32, MOTION_TASK_DIM)}
    start['physical'][:, 6] = 1.
    trainer = object.__new__(OnlineTrainer)
    trainer.config = OnlineConfig(record_tasks=True)
    trainer.learner = SimpleNamespace(critic=critic)
    trainer.sample_starts = lambda _: (start, None)
    trainer.task_norm_pending = True
    trainer.initialize_task_norm()
    assert not trainer.task_norm_pending
    for key in ('mean', 'var'):
        assert torch.equal(getattr(critic.norm, key)[:-MOTION_TASK_DIM], before[key][:-MOTION_TASK_DIM])
    assert torch.equal(critic.norm.count, before['count'])
    assert torch.isfinite(critic.norm.mean).all() and torch.isfinite(critic.norm.var).all()
