"""External-input migration preserves learning and distinguishes unknown clocks."""
from dataclasses import asdict, replace
import hashlib
import json

import pytest
import torch

from supermimic_rl.contracts import Field, StateSchema
from supermimic_rl.course.randomization import CourseRandomization, RequestedRandomizationConfig
from supermimic_rl.world_model_v2.auxiliary_data import AuxiliaryWorldData, WorldDatasetConfig
from supermimic_rl.world_model_v2.disturbance import DISTURBANCE_RECORD_FIELDS, DISTURBANCE_STATE_FIELDS
from supermimic_rl.world_model_v2.imagination import Critic
from supermimic_rl.world_model_v2.migrations.disturbance import upgrade_disturbances, zero_disturbance_contract
from supermimic_rl.world_model_v2.model import WorldModelV2
from supermimic_rl.world_model_v2.online import OnlineConfig, policy_start_candidates
from supermimic_rl.world_model_v2.records import MODEL_CONTRACT, decode_memory, decode_physical
from supermimic_rl.world_model_v2.replay import CompactReplay
from supermimic_rl.world_model_v2.teacher import PPO_TEACHER_FORMAT, file_sha256
from support.world_v2.motion_upgrade import fixture as critic_fixture
from support.world_v2.offline_continuation import fixture as motion_fixture


def fixture(tmp_path):
    checkpoint, model, _ = motion_fixture(tmp_path)
    saved = torch.load(checkpoint, weights_only=True)
    dataset = torch.load(tmp_path / 'replay-a.pt', weights_only=True)
    critic_saved, _, critic, _, x, _, _ = critic_fixture()
    for key in ('critic', 'actor'):
        saved['learner'][key] = critic_saved['learner'][key]
    saved['learner']['optimizers']['critic'] = critic_saved['learner']['optimizers']['critic']
    saved['auxiliary_world'] = {'sample_rng': torch.Generator().manual_seed(19).get_state(),
        'validation_rng': torch.Generator().manual_seed(20).get_state(), 'sampled_windows': {'flight': 998}}
    saved['course_progression'] = {'coordinate': 800000000, 'passed': [True, False]}
    configs = [RequestedRandomizationConfig(), RequestedRandomizationConfig(
        external_force_n=15., external_torque_nm=1.5, push_linear_m_s=.5, push_yaw_rad_s=1.)]
    randomization = CourseRandomization(2, seed=41, config=configs[0])
    randomization.reset(torch.tensor([True, True]), 5000)
    saved['environment'] = {'course': {'coordinate': 800000000}, 'control_timing': {
        'episode_randomization': randomization.state_dict(), 'communication': {'untouched': torch.randn(3)}}}
    runtimes = [json.dumps({'physics': 'unchanged', 'requested_randomization': asdict(c)}).encode()
                for c in configs]
    old = StateSchema.from_dict(saved['schema'])
    sources, schemas = [], []
    for runtime, config in zip(runtimes, configs):
        source = {'asset': 'unchanged', 'runtime_sha256': hashlib.sha256(runtime).hexdigest(),
            'randomization': {**config.provenance(), 'applied': True},
            'reward_kernel': {'implementation': 'shared_physical_reward_densities',
                              'kernel_sha256': 'same', 'extensions': {}, 'physics_dt_s': .005},
            'jump_takeoff': None, 'stair_label': None, 'jump_assist': None}
        sources.append(json.loads(json.dumps(source)))
        schemas.append(replace(old, source=json.dumps(source)))
    new = replace(schemas[1], state_fields=old.state_fields + tuple(
        Field(name, shape) for name, shape in zip(DISTURBANCE_STATE_FIELDS, ((1,), (1,), (6,), (1,)))),
        substep_fields=old.substep_fields + (Field('velocity_push_world', (3,)), Field('external_wrench_body', (6,))))
    saved.update(schema=schemas[0].to_dict(), schema_fingerprint=schemas[0].fingerprint)
    dataset['schema_fingerprint'] = schemas[0].fingerprint
    storage = dataset['replay']['storage']
    storage['reward_context_valid'].fill_(1.)
    storage['failure_conditions_valid'].fill_(1.)
    storage['horizon'][..., 0] = 1.
    recorded = {name: value[:1].clone() for name, value in storage.items()}
    recorded.update({name: torch.zeros(1, 2, *shape) for name, shape in DISTURBANCE_RECORD_FIELDS.items()})
    recorded['disturbance_record_valid'].fill_(1.)
    reference = {'format': 'supermimic-world-v2-dataset-v1', 'model_contract': MODEL_CONTRACT,
        'schema': new.to_dict(), 'schema_fingerprint': new.fingerprint, 'replay': {'storage': recorded}}
    metadata = {'format': PPO_TEACHER_FORMAT, 'qualified_for': ['flat-data-only'],
        'initial_checkpoint': {'sha256': 'a' * 64, 'updates': 8974, 'transitions': 1000},
        'history_length': 5, 'observation_dim': 35, 'action_bounds': [12.] * 4 + [9.] * 2,
        'action_semantics': 'clipped', 'history_semantics': 'native', 'estimate_semantics': 'velocity/contact'}
    paths = [tmp_path / 'old-teacher.pt', tmp_path / 'new-teacher.pt']
    for path, source in zip(paths, sources):
        native = {**metadata, 'native_source': source, 'qualification': {'report': {
            'native_source': source, 'checkpoint_sha256': 'a' * 64, 'steps': 600, 'simulated_seconds': 12.}}}
        torch.save({'format': PPO_TEACHER_FORMAT, 'source': native, 'actor': {'weight': torch.tensor([1.])}}, path)
        if path == paths[0]:
            saved['teacher_source'] = {**native, 'teacher_artifact': {'path': str(path), 'sha256': file_sha256(path)}}
    options = dict(source_runtime=runtimes[0], target_runtime=runtimes[1],
                   source_teacher_path=paths[0], target_teacher_path=paths[1])
    return saved, dataset, reference, options, model, critic, x


def test_expansion_preserves_predictions_adam_statistics_and_training_state(tmp_path):
    saved, data, reference, options, old_model, old_critic, x = fixture(tmp_path)
    rng = torch.get_rng_state().clone()
    upgraded, replay, report = upgrade_disturbances(saved, data, reference, **options)
    assert torch.equal(torch.get_rng_state(), rng)
    for key in saved.keys() - {'schema', 'schema_fingerprint', 'learner', 'teacher_source', 'resume_migrations', 'environment'}:
        assert upgraded[key] is saved[key], key
    old_timing = saved['environment']['control_timing']
    new_timing = upgraded['environment']['control_timing']
    assert new_timing['communication'] is old_timing['communication']
    assert upgraded['environment']['course'] is saved['environment']['course']
    for key in old_timing['episode_randomization'].keys() - {'configuration'}:
        assert new_timing['episode_randomization'][key] is old_timing['episode_randomization'][key]
    restored = CourseRandomization(2, seed=41,
        config=RequestedRandomizationConfig(**new_timing['episode_randomization']['configuration']))
    restored.restore_after_physical_reset(new_timing['episode_randomization'])
    assert torch.equal(restored.generator.get_state(), old_timing['episode_randomization']['rng'])
    assert restored.reset_rows_consumed == old_timing['episode_randomization']['reset_rows_consumed']
    for key in ('updates', 'skipped', 'actor'):
        assert upgraded['learner'][key] is saved['learner'][key]
    assert report['additional_warmup_cycles'] == 0
    for name, value in data['replay']['storage'].items():
        assert replay['replay']['storage'][name] is value
    model = WorldModelV2(old_model.inputs, old_model.wheels, hidden=(16,), disturbances=True)
    model.load_state_dict(upgraded['learner']['model'])
    critic = Critic(x.shape[-1] + 9, (16, 8), disturbances=True)
    critic.load_state_dict(upgraded['learner']['critic'])
    torch.testing.assert_close(critic(torch.cat((x, torch.randn(len(x), 9)), -1)), old_critic(x))
    for group, old, new, extra in (('model', old_model, model, 24), ('critic', old_critic, critic, 9)):
        norm = new.motion.input_norm if group == 'model' else new.norm
        original = old.motion.input_norm if group == 'model' else old.norm
        torch.testing.assert_close(norm.mean[:-extra], original.mean, rtol=0, atol=0)
        torch.testing.assert_close(norm.var[:-extra], original.var, rtol=0, atol=0)
        assert norm.count == original.count
        optimizer = torch.optim.Adam(new.parameters())
        optimizer.load_state_dict(upgraded['learner']['optimizers'][group])
        old_opt = saved['learner']['optimizers'][group]
        first = old_opt['param_groups'][0]['params'][0]
        state = optimizer.state[next(new.parameters())]
        for name in ('exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(state[name][:, :-extra], old_opt['state'][first][name], rtol=0, atol=0)
            assert not state[name][:, -extra:].any()
        assert state['step'] == old_opt['state'][first]['step']
        # The padded moments must support a real next optimizer step.
        sum(p.square().mean() for p in new.parameters()).backward()
        optimizer.step()
        assert state['step'] == old_opt['state'][first]['step'] + 1
    # Restore migrated weights after the optimizer-continuation check.
    model.load_state_dict(upgraded['learner']['model'])
    ring = CompactReplay.from_state_dict(replay['replay'], 'cpu')
    start = {name: value[:, 0] for name, value in ring.sample_windows(2, 1).items()}
    args = (decode_physical(start['physical']), start['height'], start['scan'], decode_memory(start['memory']),
            start['sim_context'], start['motor_strength'][:, 4:6], start['action'])
    torch.testing.assert_close(model.step(*args, velocity_push=start['velocity_push'],
        external_wrench=start['external_wrench']).relative_ticks, old_model.step(*args).relative_ticks,
        rtol=1e-5, atol=1e-7)


def test_unknown_clocks_exclude_only_old_actor_starts_not_dynamics(tmp_path):
    saved, data, reference, options, *_ = fixture(tmp_path)
    _, replay, _ = upgrade_disturbances(saved, data, reference, **options)
    ring = CompactReplay.from_state_dict(replay['replay'], 'cpu')
    config = OnlineConfig()
    assert len(ring.window_candidates(8, [0, 1])) > 0
    assert len(policy_start_candidates(ring, config, [0, 1], .02)) == 0
    record = {name: value[0] for name, value in reference['replay']['storage'].items() if value.ndim >= 3}
    boundary = {name: value[0] for name, value in reference['replay']['storage'].items() if value.ndim == 2}
    ring.add(record, boundary)
    assert len(policy_start_candidates(ring, config, [0, 1], .02)) == 2


@pytest.mark.parametrize('damage', ['runtime_pin', 'runtime_physics', 'source_physics', 'old_disturbed',
                                  'unknown_reference', 'teacher', 'pair', 'record_shape'])
def test_migration_does_not_hide_changed_physics_or_unrecorded_forces(tmp_path, damage):
    saved, data, reference, options, *_ = fixture(tmp_path)
    if damage == 'runtime_pin':
        options['source_runtime'] += b' '
    elif damage == 'runtime_physics':
        value = json.loads(options['target_runtime'])
        value['physics'] = 'changed'
        options['target_runtime'] = json.dumps(value).encode()
        source = json.loads(reference['schema']['source'])
        source['runtime_sha256'] = hashlib.sha256(options['target_runtime']).hexdigest()
        reference['schema']['source'] = json.dumps(source)
    elif damage in ('source_physics', 'old_disturbed'):
        item = reference if damage == 'source_physics' else saved
        source = json.loads(item['schema']['source'])
        if damage == 'source_physics':
            source['asset'] = 'other'
        else:
            source['randomization']['configuration']['external_force_n'] = 1.
        item['schema']['source'] = json.dumps(source)
    elif damage == 'unknown_reference':
        reference['replay']['storage']['disturbance_record_valid'].zero_()
    elif damage == 'teacher':
        payload = torch.load(options['target_teacher_path'], weights_only=True)
        payload['actor']['weight'] += 1
        torch.save(payload, options['target_teacher_path'])
    elif damage == 'pair':
        data['cycle'] += 1
    else:
        reference['replay']['storage']['velocity_push'] = torch.zeros(1, 2, 4)
    for item in (saved, reference):
        item['schema_fingerprint'] = StateSchema.from_dict(item['schema']).fingerprint
    data['schema_fingerprint'] = saved['schema_fingerprint']
    with pytest.raises(ValueError):
        upgrade_disturbances(saved, data, reference, **options)


def test_auxiliary_zero_conditions_keep_original_file_split_and_rng(tmp_path):
    saved, data, reference, options, *_ = fixture(tmp_path)
    upgraded, replay, report = upgrade_disturbances(saved, data, reference, **options)
    path = tmp_path / 'old-skill.pt'
    torch.save({**data, 'format': 'supermimic-world-v2-dataset-v1', 'model_contract': MODEL_CONTRACT,
                'schema': saved['schema']}, path)
    config = WorldDatasetConfig('flight', str(path), file_sha256(path), .5, (0,), (1,))
    schema = StateSchema.from_dict(upgraded['schema'])
    live = CompactReplay.from_state_dict(replay['replay'], 'cpu')
    with pytest.raises(ValueError, match='contract differs'):
        AuxiliaryWorldData((config,), schema, live, 2, seed=3)
    aux = AuxiliaryWorldData((config,), schema, live, 2, seed=3,
        disturbance_runtime_pairs=(report['disturbance_runtime_pair'],))
    assert aux.derived_conditions == {'flight': ['velocity_push', 'external_wrench']}
    assert aux.source_schemas == {'flight': saved['schema_fingerprint']}
    aux.load_state_dict(saved['auxiliary_world'])
    assert torch.equal(aux.generator.get_state(), saved['auxiliary_world']['sample_rng'])
    assert torch.equal(aux.validation_generator.get_state(), saved['auxiliary_world']['validation_rng'])
    sample = {name: torch.zeros(128, 2, *live.data[name].shape[2:], dtype=live.data[name].dtype)
              for name in aux.fields}
    mixed = aux.mix(sample)
    assert aux.sampled['flight'] > 998 and not mixed['action'][..., 0].any()  # Held row has action 17.
    assert not mixed['velocity_push'].any() and not mixed['external_wrench'].any()
    assert file_sha256(path) == config.sha256
    # Even an audited hash pair cannot reinterpret already perturbed history.
    before, after = (StateSchema.from_dict(item['schema']).to_dict() for item in (saved, upgraded))
    for item in (before, after):
        item['source'] = json.loads(item['source'])
    before['source']['randomization']['configuration']['push_linear_m_s'] = .5
    assert not zero_disturbance_contract(before, after, (report['disturbance_runtime_pair'],))
