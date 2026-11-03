"""Task-objective handoff preserves dynamics without relabeling historical memory."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.contracts import StateSchema
from supermimic_rl.course.jump_first import make_course_curriculum
from supermimic_rl.course.jump_takeoff import install_jump_takeoff
from supermimic_rl.reward_kernel import NATIVE_SOURCE_SHA256, RewardExtensions, physical_reward_densities
from supermimic_rl.world_model_v2.jump_task import JumpTaskConfig, JumpTaskState
from supermimic_rl.world_model_v2.migrations.recovery import upgrade_recovery_tasks
from supermimic_rl.world_model_v2.migrations.task_semantics import TASK_VALID_FLAGS, upgrade_task_semantics
from supermimic_rl.world_model_v2.online import OnlineConfig, OnlineTrainer, policy_start_candidates
from supermimic_rl.world_model_v2.replay import CompactReplay
from supermimic_rl.world_model_v2.teacher import PPO_TEACHER_FORMAT, file_sha256
from support.world_v2.jump_imagination import fixture as jump_imagination_fixture
from support.world_v2.jump_task import native_jumps
from support.world_v2.recovery_imagination import fixture as recovery_imagination_fixture
from support.world_v2.recovery_task import native_recovery
from supermimic_rl.world_model_v2.recovery_task import RecoveryTaskState
from test_world_v2_recovery_upgrade import fixture as recovery_upgrade_fixture


ROOT = Path(__file__).resolve().parents[1]


def fixture(tmp_path):
    saved, dataset, reference, options, boundary = recovery_upgrade_fixture()
    saved, dataset, _ = upgrade_recovery_tasks(saved, dataset, reference, **options)
    saved['critic_norm_from'] = None
    saved['config'].update(critic_hidden=[16, 8], course_coordinate=196608000)
    saved['guard'] = {'best_reward': 999., 'bad_cycles': 2}
    saved['stair_scenario']['task']['use_mode_labels'] = False
    old_runtime_raw = (ROOT / 'configs/gpufree/runtime-physx64-16-fullturn-knee40-107.json').read_bytes()
    new_runtime = json.loads(old_runtime_raw)
    new_runtime['terrain_policy']['use_mode_labels'] = True
    new_runtime_raw = json.dumps(new_runtime).encode()
    old_course_raw, new_course_raw = [(ROOT / f'configs/gpufree/ppo-course-gated-v{n}.json').read_bytes()
                                      for n in (3, 5)]
    curriculum = make_course_curriculum(2, 'cpu', 93, json.loads(old_course_raw))
    curriculum.set_progress(196608000, 768)
    curriculum.sample(torch.arange(2))
    saved['environment'].update(curriculum=curriculum.state_dict(), actual_transitions=1234567)
    reward = {'implementation': 'shared_physical_reward_densities', 'physics_dt_s': .005,
              'native_source_sha256': NATIVE_SOURCE_SHA256,
              'kernel_sha256': file_sha256(inspect.getfile(physical_reward_densities)),
              'extensions': asdict(RewardExtensions(jump_flight=4., jump_lift=2., success_event=5.))}
    source = {'asset': 'unchanged', 'runtime_sha256': hashlib.sha256(old_runtime_raw).hexdigest(),
              'course_sha256': hashlib.sha256(old_course_raw).hexdigest(),
              'curriculum': json.loads(old_course_raw), 'reward_kernel': {**reward, 'kernel_sha256': 'legacy'}}
    native = native_jumps(2)
    gate = install_jump_takeoff(native, {'compression_depth_max_m': .04,
        'compression_duration_s': .2, 'nominal_extension_duration_s': .1})
    target = {**source, 'runtime_sha256': hashlib.sha256(new_runtime_raw).hexdigest(),
              'course_sha256': hashlib.sha256(new_course_raw).hexdigest(), 'curriculum': json.loads(new_course_raw),
              'jump_takeoff': gate, 'jump_assist': None,
              'stair_label': {'stair_label': 'STAIR set by the ToF stair state machine only; terrain-kind labels disabled'},
              'reward_kernel': {**reward, 'extensions': {**reward['extensions'], 'jump_push': 2., 'jump_missed': 1.}}}
    for item, provenance in ((saved, source), (reference, target)):
        schema = StateSchema.from_dict({**saved['schema'], 'source': json.dumps(provenance)})
        item.update(schema=schema.to_dict(), schema_fingerprint=schema.fingerprint)
    dataset['schema_fingerprint'] = saved['schema_fingerprint']
    reference.update(stair_source_config=deepcopy(saved['stair_scenario']),
        jump_task_config=asdict(JumpTaskConfig.capture(native)), recovery_task_config=saved['recovery_scenario'])
    reference['stair_source_config']['task']['use_mode_labels'] = True
    for name in TASK_VALID_FLAGS:
        dataset['replay']['storage'][name].fill_(1.)
    recorded = {name: value[:1].clone() for name, value in dataset['replay']['storage'].items()}
    recorded.update({name: value[None] for name, value in JumpTaskState.capture(native).record().items()})
    reference['replay'] = {'storage': recorded}
    metadata = {'format': PPO_TEACHER_FORMAT, 'qualified_for': ['flat-data-only'],
        'initial_checkpoint': {'sha256': 'a' * 64, 'path': '/weights/ppo.pt', 'updates': 8974, 'transitions': 1000},
        'history_length': 5, 'observation_dim': 35, 'action_bounds': [12.] * 4 + [9.] * 2,
        'action_semantics': 'clipped', 'history_semantics': 'native', 'estimate_semantics': 'velocity/contact'}
    paths = [tmp_path / 'old-teacher.pt', tmp_path / 'new-teacher.pt']
    for path, native_source in zip(paths, (source, target)):
        teacher_metadata = {**metadata, 'native_source': native_source, 'qualification': {'report': {
            'native_source': native_source, 'checkpoint_sha256': 'a' * 64, 'steps': 500, 'simulated_seconds': 10.}}}
        payload = {'format': PPO_TEACHER_FORMAT, 'source': teacher_metadata, 'actor': {'weight': torch.tensor([1., 2.])},
                   'actor_hidden': [16, 8], 'estimator_hidden': [8], 'mean_squash': 'none'}
        torch.save(payload, path)
        if path == paths[0]:
            saved['teacher_source'] = {**teacher_metadata,
                'teacher_artifact': {'path': str(path), 'sha256': file_sha256(path)}}
    options = {'source_runtime_raw': old_runtime_raw, 'target_runtime_raw': new_runtime_raw,
               'source_course_raw': old_course_raw, 'target_course_raw': new_course_raw,
               'source_teacher_path': paths[0], 'target_teacher_path': paths[1],
               'course_coordinate': 491520000, 'critic_warmup_cycles': 3}
    return saved, dataset, reference, options, boundary


def test_handoff_keeps_physics_optimizers_counters_but_resets_reward_learning(tmp_path):
    saved, dataset, reference, options, boundary = fixture(tmp_path)
    rng = torch.get_rng_state().clone()
    upgraded, migrated, report = upgrade_task_semantics(saved, dataset, reference, **options)
    assert torch.equal(torch.get_rng_state(), rng)
    for name in ('model', 'actor', 'updates'):
        assert upgraded['learner'][name] is saved['learner'][name]
    for name in ('actor', 'model', 'estimator'):
        assert upgraded['learner']['optimizers'][name] is saved['learner']['optimizers'][name]
    for name in ('collection_steps', 'sample_rng', 'collection_rng', 'mixing', 'torch_rng'):
        assert upgraded[name] is saved[name]
    assert upgraded['guard'] is None and saved['guard']['best_reward'] == 999.
    assert upgraded['critic_norm_from'] == 0 and upgraded['config']['critic_warmup']
    assert upgraded['config']['policy_start_cycle'] == saved['cycle'] + 3
    assert upgraded['learner']['optimizers']['critic']['state'] == {}
    critic = upgraded['learner']['critic']
    assert critic['net.0.weight'].shape == (16, 411)
    assert not critic['norm.mean'].any() and (critic['norm.var'] == 1).all()
    for name, value in dataset['replay']['storage'].items():
        if name in TASK_VALID_FLAGS:
            assert (value == 1).all() and not migrated['replay']['storage'][name].any()
        else:
            assert migrated['replay']['storage'][name] is value
    assert report['legacy_gate_histories_inferred'] == 0 and report['guard_reset']
    assert report['teacher_qualification'] == ['flat-data-only']
    assert report['course']['sampler_credits_reset'] and not report['course']['coordinate_is_capability_promotion']
    course = upgraded['environment']['curriculum']
    assert course['sample_counts'] == saved['environment']['curriculum']['sample_counts']
    assert torch.equal(course['sampler']['rng'], saved['environment']['curriculum']['sampler']['rng'])
    assert upgraded['environment']['actual_transitions'] == 1234567
    restored = make_course_curriculum(2, 'cpu', 0, json.loads(options['target_course_raw']))
    restored.load_state_dict(course)
    assert restored.stage_names[restored.stage] == 'active_jump'
    old_ring = CompactReplay.from_state_dict(dataset['replay'])
    new_ring = CompactReplay.from_state_dict(migrated['replay'])
    assert torch.equal(old_ring.window_candidates(1), new_ring.window_candidates(1))
    config = OnlineConfig.from_dict(upgraded['config'])
    assert policy_start_candidates(new_ring, config, [0, 1], .02).numel() == 0
    _, actual, _ = jump_imagination_fixture(gated=True)
    actual.update(RecoveryTaskState.capture(native_recovery(2)).record())
    record = {name: value[-1].clone() for name, value in migrated['replay']['storage'].items()
              if name not in ('segment_age', *boundary)}
    record.update({name: value for name, value in actual.items() if name in record})
    for name in TASK_VALID_FLAGS:
        record[name].fill_(1.)
    new_ring.add(record, boundary)
    assert policy_start_candidates(new_ring, config, [0, 1], .02).numel() == 2
    trainer = object.__new__(OnlineTrainer)
    trainer.env = SimpleNamespace(schema=StateSchema.from_dict(upgraded['schema']))
    trainer.config, trainer.teacher_source = config, upgraded['teacher_source']
    for name in ('stair_scenario', 'jump_scenario', 'recovery_scenario'):
        setattr(trainer, name, upgraded[name])
    trainer._check_saved(upgraded)


def test_reward_change_can_calibrate_all_fresh_critic_columns():
    learner, start, frames = recovery_imagination_fixture()
    learner.critic.norm.mean.fill_(99.)
    trainer = object.__new__(OnlineTrainer)
    trainer.learner, trainer.critic_norm_from = learner, 0
    trainer.config = OnlineConfig()
    trainer.sample_starts = lambda _: (start, frames)
    trainer.initialize_added_critic_norm()
    assert trainer.critic_norm_from is None
    assert torch.isfinite(learner.critic.norm.mean).all()
    assert not (learner.critic.norm.mean == 99.).any()


def test_real_c15_curriculum_missing_new_optional_defaults_is_explicitly_normalized(tmp_path):
    saved, dataset, reference, options, _ = fixture(tmp_path)
    old_config = saved['environment']['curriculum']['config']
    defaults = {name: old_config.pop(name) for name in
                ('jump_height_stage', 'jump_height_gates', 'recovery_posture_stages')}
    assert defaults == {'jump_height_stage': 'combined_dr', 'jump_height_gates': [],
                        'recovery_posture_stages': {}}
    upgraded, _, receipt = upgrade_task_semantics(saved, dataset, reference, **options)
    assert receipt['course']['source_checkpoint_defaults_materialized'] == defaults
    assert not set(defaults) & old_config.keys()
    assert upgraded['environment']['curriculum']['config']['jump_height_stage'] == 'active_jump'
    assert upgraded['environment']['actual_transitions'] == saved['environment']['actual_transitions']


@pytest.mark.parametrize('damage', ['unknown', 'missing_required', 'wrong_default'])
def test_legacy_curriculum_normalization_still_checks_the_pinned_source(tmp_path, damage):
    saved, dataset, reference, options, _ = fixture(tmp_path)
    old = saved['environment']['curriculum']['config']
    for name in ('jump_height_stage', 'jump_height_gates', 'recovery_posture_stages'):
        old.pop(name)
    if damage == 'unknown':
        old['unknown_course_parameter'] = 1
    elif damage == 'missing_required':
        old.pop('mixture_blend_fraction')
    else:
        old['jump_height_stage'] = 'active_jump'
    with pytest.raises(ValueError):
        upgrade_task_semantics(saved, dataset, reference, **options)


@pytest.mark.parametrize('damage', [
    'delay', 'wrench', 'runtime_sha', 'asset', 'assist', 'course_domain', 'course_sha', 'reward_native',
    'reward_code', 'recovery', 'stair_sensor', 'gate_unknown', 'gate_dtype', 'partial_data',
    'teacher_weights', 'teacher_evaluation', 'teacher_old_pin', 'zero_warmup',
])
def test_handoff_refuses_unmodeled_physics_or_unsupported_task_rebinding(tmp_path, damage):
    saved, dataset, reference, options, _ = fixture(tmp_path)
    source = json.loads(reference['schema']['source'])
    if damage in ('delay', 'wrench'):
        runtime = json.loads(options['target_runtime_raw'])
        if damage == 'delay':
            runtime['unmodeled_sensor_change'] = True
        else:
            runtime['requested_randomization']['external_force_n'] = 15.
        options['target_runtime_raw'] = json.dumps(runtime).encode()
        source['runtime_sha256'] = hashlib.sha256(options['target_runtime_raw']).hexdigest()
    elif damage == 'runtime_sha':
        options['target_runtime_raw'] += b' '
    elif damage == 'asset':
        source['asset'] = 'other-robot'
    elif damage == 'assist':
        source['jump_assist'] = {'force': 1.}
    elif damage == 'course_domain':
        course = json.loads(options['target_course_raw'])
        course['base']['domain']['height_m'][0] = .20
        options['target_course_raw'] = json.dumps(course).encode()
        source.update(curriculum=course, course_sha256=hashlib.sha256(options['target_course_raw']).hexdigest())
    elif damage == 'course_sha':
        options['target_course_raw'] += b' '
    elif damage == 'reward_native':
        source['reward_kernel']['native_source_sha256'] = 'other'
    elif damage == 'reward_code':
        source['reward_kernel']['kernel_sha256'] = 'other'
    elif damage == 'recovery':
        reference['recovery_task_config'] = {**reference['recovery_task_config'], 'changed': True}
    elif damage == 'stair_sensor':
        reference['stair_source_config']['task']['estimate_max_age_s'] = 1.
    elif damage == 'gate_unknown':
        reference['replay']['storage']['jump_trigger_flags'].zero_()
    elif damage == 'gate_dtype':
        reference['replay']['storage']['jump_trigger_clock'] = reference['replay']['storage']['jump_trigger_clock'].float()
    elif damage == 'partial_data':
        reference['replay']['storage'].pop('ticks')
    elif damage.startswith('teacher_'):
        payload = torch.load(options['target_teacher_path'], weights_only=True)
        if damage == 'teacher_weights':
            payload['actor']['weight'] += 1.
        elif damage == 'teacher_evaluation':
            payload['source']['qualification']['report']['checkpoint_sha256'] = 'other'
        else:
            saved['teacher_source']['teacher_artifact']['sha256'] = 'other'
        torch.save(payload, options['target_teacher_path'])
    else:
        options['critic_warmup_cycles'] = 0
    schema = StateSchema.from_dict({**reference['schema'], 'source': json.dumps(source)})
    reference.update(schema=schema.to_dict(), schema_fingerprint=schema.fingerprint)
    with pytest.raises(ValueError):
        upgrade_task_semantics(saved, dataset, reference, **options)


def test_cli_writes_new_pair_without_touching_source_and_records_provenance(tmp_path, monkeypatch):
    saved, dataset, reference, options, _ = fixture(tmp_path)
    saved['replay_file'] = 'old-replay.pt'
    paths = {name: tmp_path / (name + '.pt') for name in ('checkpoint', 'reference-dataset')}
    torch.save(saved, paths['checkpoint'])
    torch.save(dataset, tmp_path / saved['replay_file'])
    torch.save(reference, paths['reference-dataset'])
    for name in ('source-runtime', 'target-runtime', 'source-course', 'target-course'):
        paths[name] = tmp_path / (name + '.json')
        paths[name].write_bytes(options[name.replace('-', '_') + '_raw'])
    paths.update({'source-teacher': options['source_teacher_path'], 'target-teacher': options['target_teacher_path']})
    before = {path: path.read_bytes() for path in (*paths.values(), tmp_path / saved['replay_file'])}
    output = tmp_path / 'handoff'
    arguments = ['upgrade_online_task_semantics', '--output', str(output), '--course-coordinate', '491520000',
                 '--critic-warmup-cycles', '3']
    for name, path in paths.items():
        arguments.extend(['--' + name, str(path)])
    monkeypatch.setattr('sys.argv', arguments)
    runpy.run_path(str(ROOT / 'scripts/upgrade_online_task_semantics.py'), run_name='__main__')
    assert all(path.read_bytes() == content for path, content in before.items())
    migrated = torch.load(output / 'resume-a.pt', weights_only=True)
    report = json.loads((output / 'migration.json').read_text())
    assert migrated['resume_migrations'][-1] == report
    assert migrated['guard'] is None and migrated['critic_norm_from'] == 0
    assert report['reference_dataset']['sha256'] == file_sha256(paths['reference-dataset'])
    assert (output / migrated['replay_file']).is_file()
    with pytest.raises(FileExistsError):
        runpy.run_path(str(ROOT / 'scripts/upgrade_online_task_semantics.py'), run_name='__main__')
