"""Jump branch migration preserves physics and learning without inventing requests."""
from copy import deepcopy
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
from supermimic_rl.world_model_v2.jump_task import JUMP_RECORD_DTYPES
from supermimic_rl.world_model_v2.migrations.jump import upgrade_jump_tasks
from supermimic_rl.world_model_v2.online import OnlineConfig, OnlineTrainer, policy_start_candidates
from supermimic_rl.world_model_v2.replay import CompactReplay

from support.world_v2.jump_imagination import fixture as jump_fixture
from support.world_v2.jump_upgrade import fixture


def test_jump_branch_preserves_values_adam_and_physics_with_explicit_course_reset():
    saved, dataset, reference, options, _ = fixture()
    upgraded, replay, receipt = upgrade_jump_tasks(saved, dataset, reference, **options)
    before = Critic(369, (16, 8), task_dim=40, terrain=True, stairs=True)
    after = Critic(401, (16, 8), task_dim=40, terrain=True, stairs=True, jumps=True)
    before.load_state_dict(saved['learner']['critic'])
    after.load_state_dict(upgraded['learner']['critic'])
    x = torch.randn(4, 369)
    torch.testing.assert_close(after(torch.cat((x, torch.randn(4, 32)), -1)), before(x), atol=1e-7, rtol=1e-6)
    first = saved['learner']['optimizers']['critic']['param_groups'][0]['params'][0]
    for name in ('exp_avg', 'exp_avg_sq'):
        old = saved['learner']['optimizers']['critic']['state'][first][name]
        new = upgraded['learner']['optimizers']['critic']['state'][first][name]
        assert torch.equal(old, new[:, :369]) and not new[:, 369:].any()
    assert not after.net[0].weight[:, 369:].any()
    assert not after.norm.mean[369:].any() and (after.norm.var[369:] == 1).all()
    for name in ('actor', 'model'):
        assert upgraded['learner'][name] is saved['learner'][name]
    for name in ('model', 'actor', 'estimator'):
        assert upgraded['learner']['optimizers'][name] is saved['learner']['optimizers'][name]
    assert upgraded['learner']['updates'] is saved['learner']['updates']
    assert upgraded['sample_rng'] is saved['sample_rng']
    assert upgraded['collection_steps'] == 1234
    assert upgraded['critic_norm_from'] == 369
    assert upgraded['config']['actor_frame_dim'] == 35 and upgraded['config']['record_jumps']
    assert upgraded['config']['policy_start_cycle'] == 34
    for name, value in dataset['replay']['storage'].items():
        assert replay['replay']['storage'][name] is value
    assert not replay['replay']['storage']['jump_record_valid'].any()
    assert 'jump_ids' not in dataset['replay']['storage']
    for name, dtype in JUMP_RECORD_DTYPES.items():
        assert replay['replay']['storage'][name].dtype == dtype
    old, new = saved['environment']['curriculum'], upgraded['environment']['curriculum']
    assert old['real_transitions'] == 30 and new['real_transitions'] == 45
    assert new['format'] == 'supermimic_jump_first_curriculum_v1'
    assert new['sample_counts'] == old['sample_counts']
    assert torch.equal(new['sampler']['rng'], old['sampler']['rng'])
    assert new['sampler']['stage_weights'] != old['sampler']['stage_weights']
    assert all(value == 0 for row in new['sampler']['family_credit'] for value in row)
    restored = make_course_curriculum(2, 'cpu', 99, json.loads(options['target_course_raw']))
    restored.load_state_dict(new)
    assert restored.stage_names[restored.stage] == 'active_jump'
    assert receipt['course']['sampler_credits_reset'] is True
    optimizer = torch.optim.Adam(after.parameters())
    optimizer.load_state_dict(upgraded['learner']['optimizers']['critic'])
    after(torch.randn(4, 401)).square().mean().backward()
    optimizer.step()
    assert torch.isfinite(after.net[0].weight).all()


def test_same_course_retains_sampler_credits_and_old_rows_still_fit_world():
    saved, dataset, reference, options, boundary = fixture(change_course=False)
    upgraded, replay, receipt = upgrade_jump_tasks(saved, dataset, reference, **options)
    assert upgraded['environment']['curriculum']['sampler'] is saved['environment']['curriculum']['sampler']
    assert receipt['course']['sampler_credits_reset'] is False
    old, new = CompactReplay.from_state_dict(dataset['replay']), CompactReplay.from_state_dict(replay['replay'])
    assert torch.equal(old.window_candidates(1), new.window_candidates(1))
    config = OnlineConfig.from_dict(upgraded['config'])
    assert policy_start_candidates(new, config, [0, 1], .02).numel() == 0
    _, actual, _ = jump_fixture()
    record = {name: value[-1].clone() for name, value in replay['replay']['storage'].items()
              if name not in ('segment_age', *boundary)}
    record.update({name: value for name, value in actual.items() if name in record})
    record['frame'].fill_(7.)
    new.add(record, boundary)
    candidates = policy_start_candidates(new, config, [0, 1], .02)
    assert candidates.numel() == 2
    start, _ = new.sample_starts(4, 5, candidates=candidates)
    assert (start['jump_record_valid'] == 1).all()
    assert (start['frame'] == 7.).all()


def test_jump_norm_initialization_changes_only_the_32_new_columns():
    learner, start, frames = jump_fixture()
    learner.critic.norm.count.fill_(9876.)
    before = deepcopy(learner.critic.norm.state_dict())
    trainer = object.__new__(OnlineTrainer)
    trainer.learner, trainer.critic_norm_from = learner, 369
    trainer.config = OnlineConfig(record_tasks=True, record_stair_source=True, record_jumps=True, course_coordinate=0)
    trainer.sample_starts = lambda _: (start, frames)
    trainer.initialize_added_critic_norm()
    for name in ('mean', 'var'):
        assert torch.equal(getattr(learner.critic.norm, name)[:369], before[name][:369])
    assert torch.equal(learner.critic.norm.count, before['count'])
    assert torch.isfinite(learner.critic.norm.mean).all() and trainer.critic_norm_from is None


@pytest.mark.parametrize('change', ['source_pin', 'physics', 'domain', 'missing_native', 'existing_records', 'uncalibrated'])
def test_jump_migration_rejects_unrelated_changes_and_invented_records(change):
    saved, dataset, reference, options, _ = fixture()
    if change == 'source_pin':
        options['source_course_raw'] += b' '
    elif change in ('physics', 'domain'):
        source = json.loads(reference['schema']['source'])
        if change == 'physics':
            source['runtime_sha256'] = 'different-physics'
        else:
            course = json.loads(options['target_course_raw'])
            course['base']['domain']['jump_m'][1] = .9
            options['target_course_raw'] = json.dumps(course).encode()
            source.update(curriculum=course, course_sha256=hashlib.sha256(options['target_course_raw']).hexdigest())
        schema = StateSchema.from_dict({**reference['schema'], 'source': json.dumps(source)})
        reference.update(schema=schema.to_dict(), schema_fingerprint=schema.fingerprint)
    elif change == 'missing_native':
        reference['replay']['storage']['jump_record_valid'].zero_()
    elif change == 'existing_records':
        dataset['replay']['storage']['jump_ids'] = torch.zeros(3, 2, 9)
    else:
        saved['critic_norm_from'] = 287
    with pytest.raises(ValueError):
        upgrade_jump_tasks(saved, dataset, reference, **options)


@pytest.mark.parametrize('with_world_fit', [False, True])
def test_jump_branch_script_keeps_original_pair_and_embeds_source_receipt(tmp_path, monkeypatch, with_world_fit):
    saved, dataset, reference, options, _ = fixture()
    saved['replay_file'] = 'original-replay.pt'
    checkpoint, source_replay = tmp_path / 'original.pt', tmp_path / saved['replay_file']
    reference_path = tmp_path / 'reference.pt'
    world = None
    if with_world_fit:
        from support.world_v2.world_rejoin import pair
        fit_directory = tmp_path / 'fitted'
        fit_directory.mkdir()
        _, ancestor, _, world, _ = pair(fit_directory)
        saved['learner']['model'] = ancestor['learner']['model']
        saved['learner']['optimizers']['model'] = ancestor['learner']['optimizers']['model']
        saved['learner']['skipped'] = dict(ancestor['learner']['skipped'])
        saved['learner']['updates']['estimator'] = ancestor['learner']['updates']['estimator']
        for entry in (saved, reference):
            schema = StateSchema.from_dict({**entry['schema'],
                'substep_fields': ancestor['schema']['substep_fields'], 'substeps_per_step': 4})
            entry.update(schema=schema.to_dict(), schema_fingerprint=schema.fingerprint)
        dataset['schema_fingerprint'] = saved['schema_fingerprint']
        for name in ('model_hidden', 'validation_fraction'):
            saved['config'][name] = ancestor['config'][name]
        saved['learner']['updates']['model'] = world['source']['initial_model_updates']
        saved['learner']['skipped']['model'] = world['source']['initial_model_skipped']
    torch.save(saved, checkpoint)
    torch.save(dataset, source_replay)
    torch.save(reference, reference_path)
    originals = (checkpoint.read_bytes(), source_replay.read_bytes())
    extra_args = []
    if world is not None:
        world['schema'] = saved['schema']
        world['model_contract']['schema_fingerprint'] = saved['schema_fingerprint']
        world['source'].update(checkpoint=str(checkpoint), replay=str(source_replay),
            checkpoint_sha256=hashlib.sha256(originals[0]).hexdigest(), cycle=saved['cycle'],
            schema_fingerprint=saved['schema_fingerprint'])
        world['configuration']['replay'] = str(source_replay)
        world_path = tmp_path / 'world-fit.pt'
        torch.save(world, world_path)
        extra_args = ['--world-fit-checkpoint', str(world_path)]
    courses = [tmp_path / name for name in ('source-course.json', 'target-course.json')]
    for path, name in zip(courses, ('source_course_raw', 'target_course_raw')):
        path.write_bytes(options[name])
    output = tmp_path / 'jump-branch'
    monkeypatch.setattr('sys.argv', ['upgrade_online_jump_tasks', '--checkpoint', str(checkpoint),
        '--reference-dataset', str(reference_path), '--source-course', str(courses[0]),
        '--target-course', str(courses[1]), '--course-coordinate', '45', '--world-only-cycles', '2',
        '--output', str(output), *extra_args])
    root = Path(__file__).resolve().parents[1]
    runpy.run_path(str(root / 'scripts/upgrade_online_jump_tasks.py'), run_name='__main__')
    assert (checkpoint.read_bytes(), source_replay.read_bytes()) == originals
    result = torch.load(output / 'resume-a.pt', weights_only=True)
    receipt = json.loads((output / 'migration.json').read_text())
    assert result['resume_migrations'][-1] == receipt
    assert receipt['source_checkpoint_sha256'] == hashlib.sha256(originals[0]).hexdigest()
    replay = torch.load(output / result['replay_file'], weights_only=True)
    assert replay['schema_fingerprint'] == result['schema_fingerprint']
    assert result['jump_scenario'] == reference['jump_task_config']
    assert (output / 'actor-000032.pt').is_file()
    if world is not None:
        assert result['learner']['updates']['model'] == saved['learner']['updates']['model'] + 4
        for name, value in world['model'].items():
            torch.testing.assert_close(result['learner']['model'][name], value, rtol=0, atol=0)
        for name in ('actor', 'estimator'):
            assert result['learner']['updates'][name] == saved['learner']['updates'][name]
        world_receipt = result['resume_migrations'][-2]
        assert world_receipt['checkpoint_sha256'] == hashlib.sha256(world_path.read_bytes()).hexdigest()
        assert torch.equal(result['sample_rng'], saved['sample_rng'])
    trainer = object.__new__(OnlineTrainer)
    trainer.env = SimpleNamespace(schema=SimpleNamespace(fingerprint=result['schema_fingerprint']))
    trainer.config = OnlineConfig.from_dict(result['config'])
    trainer.teacher_source = None
    trainer.stair_scenario, trainer.jump_scenario = result['stair_scenario'], result['jump_scenario']
    trainer.recovery_scenario = None
    trainer._check_saved(result)
