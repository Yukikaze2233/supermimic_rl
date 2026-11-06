"""Reward-only migration retains actual histories, optimizers and active learning."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.contracts import StateSchema
from supermimic_rl.reward_kernel import RewardExtensions, physical_reward_densities
from supermimic_rl.world_model_v2.migrations.recovery import upgrade_recovery_tasks
from supermimic_rl.world_model_v2.migrations.reward_reference import (
    ACCELERATION_SOURCE_KERNEL_SHA256, PEAK_SOURCE_KERNEL_SHA256,
    RECOVERY_SOURCE_KERNEL_SHA256, SOURCE_KERNEL_SHA256, TARGET_REFERENCE,
    upgrade_jump_peak_completion, upgrade_ordinary_acceleration, upgrade_recovery_quiet, upgrade_reward_reference,
)
from supermimic_rl.world_model_v2.online import OnlineConfig, OnlineTrainer
from supermimic_rl.world_model_v2.teacher import PPO_TEACHER_FORMAT, file_sha256
from support.world_v2.recovery_upgrade import fixture as recovery_fixture


def fixture(tmp_path, recovery_quiet=False, ordinary_acceleration=False, jump_peak=False):
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
    if recovery_quiet:
        source['reward_kernel']['kernel_sha256'] = RECOVERY_SOURCE_KERNEL_SHA256
        source['reward_kernel']['jump_tracking_reference'] = TARGET_REFERENCE
        source['reward_kernel']['extensions'].pop('recovery_quiet')
    if ordinary_acceleration:
        source['reward_kernel']['kernel_sha256'] = ACCELERATION_SOURCE_KERNEL_SHA256
        source['reward_kernel']['jump_tracking_reference'] = TARGET_REFERENCE
        source['reward_kernel']['extensions'].pop('ordinary_leg_acceleration')
    if jump_peak:
        source['reward_kernel']['kernel_sha256'] = PEAK_SOURCE_KERNEL_SHA256
        source['reward_kernel']['jump_tracking_reference'] = TARGET_REFERENCE
        source['reward_kernel']['extensions'].pop('jump_peak_completion', None)
    target = {**source, 'reward_kernel': {**source['reward_kernel'],
        'kernel_sha256': file_sha256(inspect.getfile(physical_reward_densities)),
        'jump_tracking_reference': TARGET_REFERENCE}}
    if recovery_quiet:
        target['reward_kernel']['extensions'] = {**source['reward_kernel']['extensions'], 'recovery_quiet': 1.}
    if ordinary_acceleration:
        target['reward_kernel']['extensions'] = {**source['reward_kernel']['extensions'], 'ordinary_leg_acceleration': 1e-7}
    if jump_peak:
        target['reward_kernel']['extensions'] = {**source['reward_kernel']['extensions'], 'jump_peak_completion': True}
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


@pytest.mark.parametrize('recovery_quiet,ordinary_acceleration', [(False, False), (True, False), (False, True)])
def test_reward_fix_retains_learning_and_historical_measurements(tmp_path, recovery_quiet, ordinary_acceleration):
    saved, dataset, reference, options = fixture(tmp_path, recovery_quiet, ordinary_acceleration)
    saved['auxiliary_world'] = {'sample_rng': torch.Generator().manual_seed(71).get_state(),
                              'sampled_windows': {'flight': 288780}}
    saved['course_progression'] = {'coordinate': 900000000, 'passed': [True, False]}
    rng = torch.get_rng_state().clone()
    migrate = (upgrade_ordinary_acceleration if ordinary_acceleration else
               upgrade_recovery_quiet if recovery_quiet else upgrade_reward_reference)
    upgraded, replay, receipt = migrate(saved, dataset, reference, **options)
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
@pytest.mark.parametrize('recovery_quiet,ordinary_acceleration', [(False, False), (True, False), (False, True)])
def test_migration_does_not_hide_an_incompatible_change(tmp_path, damage, recovery_quiet, ordinary_acceleration):
    saved, dataset, reference, options = fixture(tmp_path, recovery_quiet, ordinary_acceleration)
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
        migrate = (upgrade_ordinary_acceleration if ordinary_acceleration else
                   upgrade_recovery_quiet if recovery_quiet else upgrade_reward_reference)
        migrate(saved, dataset, reference, **options)


def test_recovery_migration_is_not_a_general_reward_override(tmp_path):
    saved, dataset, reference, options = fixture(tmp_path, recovery_quiet=True)
    source = json.loads(reference['schema']['source'])
    source['reward_kernel']['extensions']['recovery_quiet'] = 2.
    schema = StateSchema.from_dict({**reference['schema'], 'source': json.dumps(source)})
    reference.update(schema=schema.to_dict(), schema_fingerprint=schema.fingerprint)
    with pytest.raises(ValueError, match='unrelated weights'):
        upgrade_recovery_quiet(saved, dataset, reference, **options)


@pytest.mark.parametrize('change', ['wording', 'projection', 'features'])
def test_old_teacher_history_wording_requires_identical_actual_preprocessing(tmp_path, change):
    saved, dataset, reference, options = fixture(tmp_path, recovery_quiet=True)
    old = torch.load(options['source_teacher_path'], weights_only=True)
    new = torch.load(options['target_teacher_path'], weights_only=True)
    old['source']['history_semantics'] = (
        'native obs_buf policy [N,5,42], oldest first, first 35 channels; native reset filling')
    new['source']['history_semantics'] = (
        'native obs_buf policy [N,5,42], oldest first; versioned chassis_frame preprocessing; native reset filling')
    new['source']['chassis_command_contract'] = 'native_first35_v1'
    if change == 'projection':
        new['source']['chassis_command_contract'] = 'gimbal_heading_v1'
    if change == 'features':
        new['source']['actor_features'] = {'version': 'another-transform'}
    torch.save(old, options['source_teacher_path'])
    torch.save(new, options['target_teacher_path'])
    saved['teacher_source'] = {**old['source'], 'teacher_artifact': {
        'path': str(options['source_teacher_path']), 'sha256': file_sha256(options['source_teacher_path'])}}
    if change == 'wording':
        upgraded, _, _ = upgrade_recovery_quiet(saved, dataset, reference, **options)
        assert upgraded['learner'] is saved['learner']
    else:
        with pytest.raises(ValueError, match='same teacher weights'):
            upgrade_recovery_quiet(saved, dataset, reference, **options)


def test_peak_completion_preserves_learning_replay_and_existing_stair_contract(tmp_path):
    saved, dataset, reference, options = fixture(tmp_path, jump_peak=True)
    original_storage = deepcopy(dataset['replay']['storage'])
    rng = torch.get_rng_state().clone()
    upgraded, replay, report = upgrade_jump_peak_completion(saved, dataset, reference, **options)
    for name in saved.keys() - {'schema', 'schema_fingerprint', 'teacher_source', 'resume_migrations'}:
        assert upgraded[name] is saved[name]
    assert torch.equal(torch.get_rng_state(), rng)
    assert replay['replay'] is dataset['replay']
    for name, value in original_storage.items():
        torch.testing.assert_close(replay['replay']['storage'][name], value, rtol=0, atol=0, equal_nan=True)
    assert report['migration'] == 'jump-peak-completion-v1'
    assert report['additional_warmup_cycles'] == 0
    assert report['legacy_reward_sequence_before'] == dataset['replay']['total_steps']


@pytest.mark.parametrize('change', ['kernel', 'already_enabled', 'missing_penalty', 'other_weight', 'stair'])
def test_peak_completion_migration_does_not_hide_unrelated_task_or_weight_changes(tmp_path, change):
    saved, dataset, reference, options = fixture(tmp_path, jump_peak=True)
    if change == 'stair':
        reference['stair_source_config'] = {'unrelated': True}
    else:
        item = reference if change == 'other_weight' else saved
        source = json.loads(item['schema']['source'])
        kernel = source['reward_kernel']
        if change == 'kernel':
            kernel['kernel_sha256'] = 'unverified'
        elif change == 'already_enabled':
            kernel['extensions']['jump_peak_completion'] = True
        elif change == 'missing_penalty':
            kernel['extensions']['jump_missed'] = 0.
        else:
            kernel['extensions']['jump_push'] += 1.
        schema = StateSchema.from_dict({**item['schema'], 'source': json.dumps(source)})
        item.update(schema=schema.to_dict(), schema_fingerprint=schema.fingerprint)
        if item is saved:
            dataset['schema_fingerprint'] = schema.fingerprint
    with pytest.raises(ValueError):
        upgrade_jump_peak_completion(saved, dataset, reference, **options)


@pytest.mark.parametrize('lineage', ['valid', 'missing', 'forged'])
def test_reward_rebinding_composes_with_retained_teacher_course_lineage(tmp_path, lineage):
    from supermimic_rl.world_model_v2.migrations.course import IDENTITY_KEYS, verify_course_sampling

    saved, dataset, reference, options = fixture(tmp_path, recovery_quiet=True)
    root = Path(__file__).resolve().parents[1]
    raw = [(root / f'configs/gpufree/ppo-course-gated-v{version}.json').read_bytes() for version in (5, 7)]
    identities = []
    for content in raw:
        course = json.loads(content)
        identities.append({'course_sha256': hashlib.sha256(content).hexdigest(), 'curriculum': course,
            'course_schedule': {'schedule': course['schedule'],
                               'early_jump_domain': course['base']['initial_domain']['jump_m']}})
    original = {**json.loads(saved['schema']['source']), **identities[0]}
    current = {**original, **identities[1]}
    target = {**json.loads(reference['schema']['source']), **identities[1]}
    proof = verify_course_sampling(original, current, source_course=raw[0], target_course=raw[1])
    for item, source in ((saved, current), (reference, target)):
        schema = StateSchema.from_dict({**item['schema'], 'source': json.dumps(source)})
        item.update(schema=schema.to_dict(), schema_fingerprint=schema.fingerprint)
    dataset['schema_fingerprint'] = saved['schema_fingerprint']
    for path, source in ((options['source_teacher_path'], original), (options['target_teacher_path'], target)):
        payload = torch.load(path, weights_only=True)
        payload['source']['native_source'] = deepcopy(source)
        payload['source']['qualification']['report']['native_source'] = deepcopy(source)
        torch.save(payload, path)
        if path == options['source_teacher_path']:
            saved['teacher_source'] = {**payload['source'], 'teacher_artifact': {
                'path': str(path), 'sha256': file_sha256(path)}}
    if lineage == 'forged':
        proof['target_course_utf8'] += ' '
    saved['resume_migrations'] = [] if lineage == 'missing' else [{'course_sampling': proof}]
    if lineage == 'valid':
        upgraded, _, _ = upgrade_recovery_quiet(saved, dataset, reference, **options)
        assert upgraded['learner'] is saved['learner']
        assert upgraded['teacher_source']['native_source'] == target
        assert saved['teacher_source']['native_source'] == original
        assert all(original[name] == identities[0][name] for name in IDENTITY_KEYS)
    else:
        with pytest.raises(ValueError):
            upgrade_recovery_quiet(saved, dataset, reference, **options)
