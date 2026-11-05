"""Reward-only migration retains actual histories, optimizers and active learning."""
from copy import deepcopy
from dataclasses import asdict
import inspect
import json
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.contracts import StateSchema
from supermimic_rl.reward_kernel import RewardExtensions, physical_reward_densities
from supermimic_rl.world_model_v2.migrations.recovery import upgrade_recovery_tasks
from supermimic_rl.world_model_v2.migrations.reward_reference import (
    SOURCE_KERNEL_SHA256, TARGET_REFERENCE, upgrade_reward_reference,
)
from supermimic_rl.world_model_v2.online import OnlineConfig, OnlineTrainer
from supermimic_rl.world_model_v2.teacher import PPO_TEACHER_FORMAT, file_sha256
from support.world_v2.recovery_upgrade import fixture as recovery_fixture


def fixture(tmp_path):
    saved, dataset, reference, options, _ = recovery_fixture()
    saved, dataset, _ = upgrade_recovery_tasks(saved, dataset, reference, **options)
    saved['jump_scenario']['flight_latch'] = True
    saved['config']['policy_start_cycle'] = saved['cycle'] - 1
    saved['critic_norm_from'] = None
    storage = dataset['replay']['storage']
    storage['jump_flight_latch'] = torch.zeros(*storage['frame'].shape[:2], 3, dtype=torch.int64)
    storage['jump_flight_latch'][..., 0] = -1
    source = json.loads(saved['schema']['source'])
    for name in ('jump_takeoff', 'stair_label', 'jump_assist'):
        source.setdefault(name, None)
    source['reward_kernel'] = {
        'implementation': 'shared_physical_reward_densities', 'physics_dt_s': .005,
        'kernel_sha256': SOURCE_KERNEL_SHA256,
        'extensions': asdict(RewardExtensions(jump_push=2., jump_missed=1.)),
    }
    target = {**source, 'reward_kernel': {**source['reward_kernel'],
        'kernel_sha256': file_sha256(inspect.getfile(physical_reward_densities)),
        'jump_tracking_reference': TARGET_REFERENCE}}
    for item, provenance in ((saved, source), (reference, target)):
        schema = StateSchema.from_dict({**saved['schema'], 'source': json.dumps(provenance)})
        item.update(schema=schema.to_dict(), schema_fingerprint=schema.fingerprint)
    dataset['schema_fingerprint'] = saved['schema_fingerprint']
    reference.update(jump_task_config=deepcopy(saved['jump_scenario']),
                     replay={'storage': {name: value[:1].clone() for name, value in storage.items()}})
    metadata = {'format': PPO_TEACHER_FORMAT, 'qualified_for': ['flat-data-only'],
        'initial_checkpoint': {'sha256': 'a' * 64, 'path': '/weights/ppo.pt', 'updates': 8974, 'transitions': 1000},
        'history_length': 5, 'observation_dim': 35, 'action_bounds': [12.] * 4 + [9.] * 2,
        'action_semantics': 'clipped', 'history_semantics': 'native', 'estimate_semantics': 'velocity/contact'}
    paths = [tmp_path / 'source-teacher.pt', tmp_path / 'target-teacher.pt']
    for path, native in zip(paths, (source, target)):
        teacher_source = {**metadata, 'native_source': native, 'qualification': {'report': {
            'native_source': native, 'checkpoint_sha256': 'a' * 64, 'steps': 384, 'simulated_seconds': 7.68}}}
        torch.save({'format': PPO_TEACHER_FORMAT, 'source': teacher_source,
                    'actor': {'weights': torch.tensor([1., 2.])}}, path)
        if path == paths[0]:
            saved['teacher_source'] = {**teacher_source,
                'teacher_artifact': {'path': str(path), 'sha256': file_sha256(path)}}
    return saved, dataset, reference, dict(source_teacher_path=paths[0], target_teacher_path=paths[1])


def test_reward_fix_retains_learning_and_historical_measurements(tmp_path):
    saved, dataset, reference, options = fixture(tmp_path)
    rng = torch.get_rng_state().clone()
    upgraded, replay, receipt = upgrade_reward_reference(saved, dataset, reference, **options)
    assert torch.equal(torch.get_rng_state(), rng)
    for key in saved.keys() - {'schema', 'schema_fingerprint', 'teacher_source', 'resume_migrations'}:
        assert upgraded[key] is saved[key], key
    assert replay['replay'] is dataset['replay']
    assert upgraded['schema_fingerprint'] != saved['schema_fingerprint']
    assert receipt['legacy_reward_sequence_before'] == dataset['replay']['total_steps']
    assert receipt['additional_warmup_cycles'] == 0
    assert not dataset['replay']['storage']['jump_record_valid'].any()
    trainer = object.__new__(OnlineTrainer)
    trainer.env = SimpleNamespace(schema=StateSchema.from_dict(upgraded['schema']))
    trainer.config = OnlineConfig.from_dict(upgraded['config'])
    trainer.teacher_source = upgraded['teacher_source']
    for name in ('stair_scenario', 'jump_scenario', 'recovery_scenario'):
        setattr(trainer, name, upgraded[name])
    trainer._check_saved(upgraded)
    assert upgraded['cycle'] > trainer.config.policy_start_cycle


@pytest.mark.parametrize('damage', ['physical', 'weight', 'clock', 'records', 'pair', 'kernel', 'teacher'])
def test_migration_does_not_hide_an_incompatible_change(tmp_path, damage):
    saved, dataset, reference, options = fixture(tmp_path)
    if damage in ('physical', 'weight', 'kernel'):
        source = json.loads(reference['schema']['source'])
        if damage == 'physical':
            source['asset_manifest_sha256'] = 'different-robot'
        elif damage == 'weight':
            source['reward_kernel']['extensions']['jump_push'] += 1.
        else:
            source['reward_kernel']['kernel_sha256'] = 'unverified-code'
        schema = StateSchema.from_dict({**reference['schema'], 'source': json.dumps(source)})
        reference.update(schema=schema.to_dict(), schema_fingerprint=schema.fingerprint)
    elif damage == 'clock':
        reference['jump_task_config']['reference']['compression_duration_s'] += .1
    elif damage == 'records':
        reference['replay']['storage']['jump_flight_latch'] = torch.zeros(1, 2, 3)
    elif damage == 'pair':
        dataset['cycle'] += 1
    elif damage == 'teacher':
        path = options['target_teacher_path']
        payload = torch.load(path, weights_only=True)
        payload['actor']['weights'].add_(1.)
        torch.save(payload, path)
    with pytest.raises(ValueError):
        upgrade_reward_reference(saved, dataset, reference, **options)
