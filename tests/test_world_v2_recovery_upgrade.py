"""Recovery migration adds task memory while preserving the trained jump pair."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.contracts import StateSchema
from supermimic_rl.course.jump_first import make_course_curriculum
from supermimic_rl.world_model_v2.imagination import Critic
from supermimic_rl.world_model_v2.migrations.jump import upgrade_jump_tasks
from supermimic_rl.world_model_v2.online import OnlineConfig, OnlineTrainer, policy_start_candidates
from supermimic_rl.world_model_v2.records import MODEL_CONTRACT
from supermimic_rl.world_model_v2.recovery_task import (
    RECOVERY_RECORD_DTYPES, RecoveryTaskConfig, RecoveryTaskState,
)
from supermimic_rl.world_model_v2.migrations.recovery import upgrade_recovery_tasks
from supermimic_rl.world_model_v2.replay import CompactReplay

from support.world_v2.jump_imagination import fixture as jump_imagination_fixture
from support.world_v2.jump_upgrade import fixture as jump_upgrade_fixture
from support.world_v2.recovery_task import native_recovery


def fixture():
    saved, dataset, reference, options, boundary = jump_upgrade_fixture()
    saved, dataset, _ = upgrade_jump_tasks(saved, dataset, reference, **options)
    saved['critic_norm_from'] = None
    saved['teacher_source'] = {'fixture': 'own-foundation', 'sha256': 'unchanged-teacher'}
    saved['mixing'] = {'calls': 1234, 'teacher_fraction': .15}
    for name, seed in (('collection_rng', 111), ('mixing_rng', 222), ('validation_rng', 333), ('torch_rng', 444)):
        saved[name] = torch.Generator().manual_seed(seed).get_state()
    native = native_recovery(2)
    reference_data = {name: value[None] for name, value in RecoveryTaskState.capture(native).record().items()}
    reference_data.update(recovery_record_valid=torch.ones(1, 2, 1),
                          valid=torch.ones(1, 2, dtype=torch.bool),
                          policy_owned=torch.ones(1, 2, dtype=torch.bool))
    reference = {
        'format': 'supermimic-world-v2-dataset-v1', 'model_contract': MODEL_CONTRACT,
        'schema': saved['schema'], 'schema_fingerprint': saved['schema_fingerprint'],
        'stair_source_config': saved['stair_scenario'], 'jump_task_config': saved['jump_scenario'],
        'recovery_task_config': asdict(RecoveryTaskConfig.capture(native)),
        'replay': {'storage': reference_data},
    }
    return saved, dataset, reference, {
        'course_raw': options['target_course_raw'], 'course_coordinate': 125, 'world_only_cycles': 3,
    }, boundary


def test_recovery_branch_preserves_critic_function_optimizer_and_all_prior_learning():
    saved, dataset, reference, options, _ = fixture()
    upgraded, replay, receipt = upgrade_recovery_tasks(saved, dataset, reference, **options)
    before = Critic(401, (16, 8), task_dim=40, terrain=True, stairs=True, jumps=True)
    after = Critic(408, (16, 8), task_dim=40, terrain=True, stairs=True, jumps=True, recovery=True)
    before.load_state_dict(saved['learner']['critic'])
    after.load_state_dict(upgraded['learner']['critic'])
    x = torch.randn(8, 401)
    torch.testing.assert_close(after(torch.cat((x, torch.randn(8, 7)), -1)), before(x), atol=1e-7, rtol=1e-6)
    first = saved['learner']['optimizers']['critic']['param_groups'][0]['params'][0]
    for name in ('exp_avg', 'exp_avg_sq'):
        old = saved['learner']['optimizers']['critic']['state'][first][name]
        new = upgraded['learner']['optimizers']['critic']['state'][first][name]
        assert torch.equal(old, new[:, :401]) and not new[:, 401:].any()
    assert not after.net[0].weight[:, 401:].any()
    assert not after.norm.mean[401:].any() and (after.norm.var[401:] == 1).all()
    for name in ('actor', 'model', 'updates'):
        assert upgraded['learner'][name] is saved['learner'][name]
    for name in ('model', 'actor', 'estimator'):
        assert upgraded['learner']['optimizers'][name] is saved['learner']['optimizers'][name]
    for name in ('sample_rng', 'collection_rng', 'mixing_rng', 'validation_rng', 'torch_rng',
                 'mixing', 'teacher_source', 'stair_scenario', 'jump_scenario', 'schema'):
        assert upgraded[name] is saved[name]
    assert upgraded['collection_steps'] == saved['collection_steps'] == 1234
    assert upgraded['critic_norm_from'] == 401
    assert upgraded['config']['actor_frame_dim'] == 35
    assert upgraded['config']['record_recovery'] and upgraded['config']['record_jumps']
    assert upgraded['config']['policy_start_cycle'] == 35
    assert upgraded['schema_fingerprint'] == saved['schema_fingerprint']
    for name, value in dataset['replay']['storage'].items():
        assert replay['replay']['storage'][name] is value
    assert not replay['replay']['storage']['recovery_record_valid'].any()
    assert 'recovery_flags' not in dataset['replay']['storage']
    for name, dtype in RECOVERY_RECORD_DTYPES.items():
        assert replay['replay']['storage'][name].dtype == dtype
    optimizer = torch.optim.Adam(after.parameters())
    optimizer.load_state_dict(upgraded['learner']['optimizers']['critic'])
    after(torch.randn(8, 408)).square().mean().backward()
    optimizer.step()
    assert torch.isfinite(after.net[0].weight).all()
    assert receipt['critic_input_before'] == 401 and receipt['critic_input_after'] == 408


def test_explicit_course_coordinate_preserves_sampler_credits_rng_and_exposure_ledger():
    saved, dataset, reference, options, _ = fixture()
    upgraded, _, receipt = upgrade_recovery_tasks(saved, dataset, reference, **options)
    old = saved['environment']['curriculum']
    new = upgraded['environment']['curriculum']
    assert old['real_transitions'] == 45 and new['real_transitions'] == 125
    assert new['sampler'] is old['sampler']
    assert new['sample_counts'] is old['sample_counts']
    assert new['actor_updates'] == old['actor_updates']
    restored = make_course_curriculum(2, 'cpu', 919, json.loads(options['course_raw']))
    restored.load_state_dict(new)
    assert restored.stage_names[restored.stage] == 'full_recovery'
    assert receipt['course']['target_stage'] == 'full_recovery'
    assert receipt['course']['sampler_credits_reset'] is False
    assert receipt['course']['coordinate_is_capability_promotion'] is False


def test_old_physical_windows_remain_but_only_new_recovery_records_seed_policy():
    saved, dataset, reference, options, boundary = fixture()
    upgraded, migrated, _ = upgrade_recovery_tasks(saved, dataset, reference, **options)
    old = CompactReplay.from_state_dict(dataset['replay'])
    new = CompactReplay.from_state_dict(migrated['replay'])
    assert torch.equal(old.window_candidates(1), new.window_candidates(1))
    config = OnlineConfig.from_dict(upgraded['config'])
    assert policy_start_candidates(new, config, [0, 1], .02).numel() == 0
    _, start, _ = jump_imagination_fixture()
    record = {name: value[-1].clone() for name, value in migrated['replay']['storage'].items()
              if name not in ('segment_age', *boundary)}
    record.update({name: value for name, value in start.items() if name in record})
    record.update(RecoveryTaskState.capture(native_recovery(2)).record())
    record['recovery_record_valid'].fill_(1.)
    record['frame'].fill_(11.)
    record['task_family'].fill_(5.)
    record['reward_route'][:, 0] = 3.
    record['goal'][:, 4:9] = 0.
    record['goal'][:, 7] = 1.
    record['motion_task'][:, 20] = 3.  # Base mode RECOVERY, after timing and route prefix.
    record['motion_task'][:, 21] = 1.  # Managed, expired but blocked during recovery.
    record['tof_segment_end'].zero_()
    new.add(record, boundary)
    candidates = policy_start_candidates(new, config, [0, 1], .02)
    assert candidates.numel() == 2
    selected, _ = new.sample_starts(8, 5, candidates=candidates)
    assert (selected['recovery_record_valid'] == 1).all()
    assert (selected['frame'] == 11.).all()
    assert new.window_candidates(1).numel() == old.window_candidates(1).numel() + 2


@pytest.mark.parametrize('change', [
    'physics', 'tensor_contract', 'course', 'source_pin', 'stair', 'jump',
    'missing_native_config', 'missing_records', 'invalid_records', 'unowned', 'clock_dtype',
    'existing_records', 'uncalibrated', 'actor45', 'not_jump', 'negative_warmup', 'negative_coordinate',
])
def test_recovery_expansion_rejects_source_changes_and_unrecorded_task_memory(change):
    saved, dataset, reference, options, _ = fixture()
    if change in ('physics', 'tensor_contract', 'course'):
        fields = dict(reference['schema'])
        if change == 'tensor_contract':
            fields['observation_dim'] = 35 if fields['observation_dim'] == 45 else 45
        else:
            source = json.loads(fields['source'])
            if change == 'physics':
                source['runtime_sha256'] = 'different-physics'
            else:
                source['course_sha256'] = 'different-course'
            fields['source'] = json.dumps(source)
        schema = StateSchema.from_dict(fields)
        reference = {**reference, 'schema': schema.to_dict(), 'schema_fingerprint': schema.fingerprint}
    elif change == 'source_pin':
        options['course_raw'] += b' '
    elif change in ('stair', 'jump'):
        name = 'stair_source_config' if change == 'stair' else 'jump_task_config'
        reference[name] = {**reference[name], 'unrelated': True}
    elif change == 'missing_native_config':
        reference.pop('recovery_task_config')
    elif change == 'missing_records':
        reference['replay']['storage'].pop('recovery_flags')
    elif change == 'invalid_records':
        reference['replay']['storage']['valid'].zero_()
    elif change == 'unowned':
        reference['replay']['storage']['policy_owned'].zero_()
    elif change == 'clock_dtype':
        reference['replay']['storage']['recovery_clock'] = reference['replay']['storage']['recovery_clock'].float()
    elif change == 'existing_records':
        dataset['replay']['storage']['recovery_flags'] = torch.zeros(3, 2, 2, dtype=torch.bool)
    elif change == 'uncalibrated':
        saved['critic_norm_from'] = 369
    elif change == 'actor45':
        saved['config']['actor_frame_dim'] = 45
    elif change == 'not_jump':
        saved['config']['record_jumps'] = False
    elif change == 'negative_warmup':
        options['world_only_cycles'] = -1
    else:
        options['course_coordinate'] = -1
    with pytest.raises(ValueError):
        upgrade_recovery_tasks(saved, dataset, reference, **options)


def test_script_keeps_original_pair_and_publishes_complete_new_pair(tmp_path, monkeypatch):
    saved, dataset, reference, options, _ = fixture()
    saved['replay_file'] = 'original-replay.pt'
    checkpoint, source_replay = tmp_path / 'original.pt', tmp_path / saved['replay_file']
    reference_path, course_path = tmp_path / 'reference.pt', tmp_path / 'course.json'
    torch.save(saved, checkpoint)
    torch.save(dataset, source_replay)
    torch.save(reference, reference_path)
    course_path.write_bytes(options['course_raw'])
    original = (checkpoint.read_bytes(), source_replay.read_bytes())
    output = tmp_path / 'recovery-branch'
    monkeypatch.setattr('sys.argv', ['upgrade_online_recovery_tasks', '--checkpoint', str(checkpoint),
        '--reference-dataset', str(reference_path), '--course', str(course_path),
        '--course-coordinate', '125', '--world-only-cycles', '3', '--output', str(output)])
    root = Path(__file__).resolve().parents[1]
    runpy.run_path(str(root / 'scripts/upgrade_online_recovery_tasks.py'), run_name='__main__')
    assert (checkpoint.read_bytes(), source_replay.read_bytes()) == original
    result = torch.load(output / 'resume-a.pt', weights_only=True)
    replay = torch.load(output / result['replay_file'], weights_only=True)
    receipt = json.loads((output / 'migration.json').read_text())
    assert result['resume_migrations'][-1] == receipt
    assert receipt['source_checkpoint_sha256'] == hashlib.sha256(original[0]).hexdigest()
    assert result['recovery_scenario'] == reference['recovery_task_config']
    assert replay['schema_fingerprint'] == result['schema_fingerprint'] == saved['schema_fingerprint']
    assert (output / 'actor-000032.pt').is_file()
    trainer = object.__new__(OnlineTrainer)
    trainer.env = SimpleNamespace(schema=SimpleNamespace(fingerprint=result['schema_fingerprint']))
    trainer.config = OnlineConfig.from_dict(result['config'])
    trainer.teacher_source = result['teacher_source']
    trainer.stair_scenario, trainer.jump_scenario = result['stair_scenario'], result['jump_scenario']
    trainer.recovery_scenario = result['recovery_scenario']
    trainer._check_saved(result)
    with pytest.raises(FileExistsError, match='new output'):
        runpy.run_path(str(root / 'scripts/upgrade_online_recovery_tasks.py'), run_name='__main__')


def test_only_seven_recovery_normalization_columns_are_calibrated():
    from support.world_v2.recovery_imagination import fixture as recovery_imagination_fixture

    learner, start, frames = recovery_imagination_fixture()
    learner.critic.norm.count.fill_(45678.)
    before = deepcopy(learner.critic.norm.state_dict())
    trainer = object.__new__(OnlineTrainer)
    trainer.learner, trainer.critic_norm_from = learner, 401
    trainer.config = OnlineConfig(record_tasks=True, record_stair_source=True, record_jumps=True,
                                  record_recovery=True, course_coordinate=0)
    trainer.sample_starts = lambda _: (start, frames)
    trainer.initialize_added_critic_norm()
    for name in ('mean', 'var'):
        assert torch.equal(getattr(learner.critic.norm, name)[:401], before[name][:401])
    assert torch.equal(learner.critic.norm.count, before['count'])
    assert torch.isfinite(learner.critic.norm.mean).all()
    assert trainer.critic_norm_from is None
