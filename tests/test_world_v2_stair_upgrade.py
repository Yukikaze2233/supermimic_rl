"""Terrain expansion preserves prior learning and admits only recorded request state."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.contracts import StateSchema
from supermimic_rl.world_model_v2.imagination import Critic
from supermimic_rl.world_model_v2.online import OnlineConfig, OnlineTrainer
from supermimic_rl.world_model_v2.replay import CompactReplay
from supermimic_rl.world_model_v2.migrations.stair import upgrade_stair_source
from supermimic_rl.world_model_v2.stair_source import StairScenario

from support.world_v2.stair_imagination import fixture as imagination_fixture
from support.world_v2.stair_upgrade import fixture


def test_stair_migration_preserves_function_moments_rng_and_actual_data():
    saved, data, reference, atlas, options, _ = fixture()
    before = Critic(287, (16, 8), task_dim=40)
    before.load_state_dict(saved['learner']['critic'])
    upgraded, replay, receipt = upgrade_stair_source(saved, data, reference, atlas, **options)
    after = Critic(369, (16, 8), task_dim=40, terrain=True, stairs=True)
    after.load_state_dict(upgraded['learner']['critic'])
    x = torch.randn(8, 287)
    torch.testing.assert_close(after(torch.cat((x, torch.randn(8, 82)), -1)), before(x), atol=1e-7, rtol=1e-6)
    first = saved['learner']['optimizers']['critic']['param_groups'][0]['params'][0]
    for name in ('exp_avg', 'exp_avg_sq'):
        old = saved['learner']['optimizers']['critic']['state'][first][name]
        new = upgraded['learner']['optimizers']['critic']['state'][first][name]
        assert torch.equal(old, new[:, :287]) and (new[:, 287:] == 0).all()
    assert torch.equal(saved['sample_rng'], upgraded['sample_rng'])
    assert saved['learner']['model'] is upgraded['learner']['model']
    assert saved['learner']['actor'] is upgraded['learner']['actor']
    for key, value in data['replay']['storage'].items():
        assert torch.equal(value, replay['replay']['storage'][key]), key
    assert 'tof_ids' not in data['replay']['storage']
    assert not replay['replay']['storage']['stair_source_valid'].any()
    assert upgraded['critic_norm_from'] == 287
    assert upgraded['config']['policy_start_cycle'] == 36
    assert saved['environment']['curriculum']['real_transitions'] == 0
    assert upgraded['environment']['curriculum']['real_transitions'] == 30
    assert receipt['height_hint_enabled']
    scenario = StairScenario.from_dict(upgraded['stair_scenario'])
    assert scenario.task.height_assist.enabled and scenario == StairScenario.from_dict(scenario.to_dict())


def test_stair_imagination_uses_fresh_state_including_expired_blocked_segments():
    saved, data, reference, atlas, options, boundary = fixture()
    upgraded, data, _ = upgrade_stair_source(saved, data, reference, atlas, **options)
    replay = CompactReplay.from_state_dict(data['replay'])
    record = {key: value[-1].clone() for key, value in data['replay']['storage'].items()
              if key not in ('segment_age', *boundary)}
    record['task_record_valid'].fill_(1.)
    record['stair_source_valid'].fill_(1.)
    record['frame'].fill_(7.)
    record['horizon'].zero_()
    record['motion_task'][:, 21] = 1.  # Managed command pool, segment already due.
    record['stair_ids'][:, 1] = torch.tensor([7, -1])
    record['tof_segment_end'].zero_()
    record['task_family'].fill_(3.)
    record['reward_route'][:, 1] = 3.
    replay.add(record, boundary)
    trainer = object.__new__(OnlineTrainer)
    trainer.device = torch.device('cpu')
    trainer.config = OnlineConfig.from_dict(upgraded['config'])
    trainer.collector = SimpleNamespace(replay=replay)
    trainer.env = SimpleNamespace(policy_dt_s=.02)
    trainer.learner = SimpleNamespace(actor=SimpleNamespace(history_length=5))
    trainer.train_ids, trainer.generator = [0, 1], torch.Generator().manual_seed(3)
    start, _ = trainer.sample_starts(32)
    assert (start['frame'] == 7.).all() and (start['stair_source_valid'] == 1.).all()
    assert (start['stair_ids'][:, 1] == 7).all()
    assert replay.window_candidates(1).numel() == 8


def test_added_stair_norm_initialization_preserves_all_287_old_columns():
    learner, start, frames = imagination_fixture()
    learner.critic.norm.count.fill_(123456.)
    before = deepcopy(learner.critic.norm.state_dict())
    trainer = object.__new__(OnlineTrainer)
    trainer.config = OnlineConfig(record_tasks=True, record_stair_source=True, course_coordinate=0)
    trainer.learner, trainer.critic_norm_from = learner, 287
    trainer.sample_starts = lambda _: (start, frames)
    trainer.initialize_added_critic_norm()
    for name in ('mean', 'var'):
        assert torch.equal(getattr(learner.critic.norm, name)[:287], before[name][:287])
    assert torch.equal(learner.critic.norm.count, before['count'])
    assert trainer.critic_norm_from is None


def test_migration_rejects_hidden_physics_changes_and_ambiguous_old_surface():
    saved, data, reference, atlas, options, _ = fixture()
    changed = json.loads(options['target_runtime_raw'])
    changed['physics_dt_s'] = .01
    options['target_runtime_raw'] = json.dumps(changed).encode()
    schema = reference['schema']
    source = json.loads(schema['source'])
    source['runtime_sha256'] = hashlib.sha256(options['target_runtime_raw']).hexdigest()
    target = StateSchema.from_dict({**schema, 'source': json.dumps(source)})
    reference.update(schema=target.to_dict(), schema_fingerprint=target.fingerprint)
    with pytest.raises(ValueError, match='physics, actuator'):
        upgrade_stair_source(saved, data, reference, atlas, **options)
    saved, data, reference, atlas, options, _ = fixture()
    data['replay']['storage']['reward_route'][0, 0, 1] = 3.
    with pytest.raises(ValueError, match='flat placement'):
        upgrade_stair_source(saved, data, reference, atlas, **options)
