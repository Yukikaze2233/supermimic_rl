"""Changing the imagined horizon resumes the learner, not a new experiment state."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.contracts import StateSchema
from supermimic_rl.world_model_v2.guard import NativeRegressionGuard
from supermimic_rl.world_model_v2.imagination import Critic
from supermimic_rl.world_model_v2.learning import Learner, LearningConfig
from supermimic_rl.world_model_v2.online import CHECKPOINT_FORMAT, OnlineConfig, OnlineTrainer
from supermimic_rl.world_model_v2.policy import EstimatorActor
from supermimic_rl.world_model_v2.records import MODEL_CONTRACT
from support.world_v2.offline_continuation import fixture as world_fixture


@pytest.mark.parametrize('horizon', [0, -1, 16., True, None])
def test_horizon_requires_a_positive_number_of_policy_intervals(horizon):
    with pytest.raises(ValueError, match='horizon'):
        LearningConfig(imagination_horizon=horizon)


def check_fixture(before=16, after=32):
    previous = OnlineConfig(cycles=100, actor_frame_dim=35,
        initial_actor='existing.pt', initial_actor_sha256='0' * 64, learning=LearningConfig(
        imagination_horizon=before, bootstrap_gradient=False))
    trainer = object.__new__(OnlineTrainer)
    trainer.env = SimpleNamespace(schema=SimpleNamespace(fingerprint='same-source'))
    trainer.teacher_source = {'checkpoint': 'teacher', 'qualification': 'data-only'}
    trainer.stair_scenario = trainer.jump_scenario = trainer.recovery_scenario = None
    trainer.config = replace(previous, learning=replace(previous.learning, imagination_horizon=after))
    saved = {'format': CHECKPOINT_FORMAT, 'model_contract': MODEL_CONTRACT,
             'schema_fingerprint': 'same-source', 'config': asdict(previous), 'cycle': 10,
             'teacher_source': deepcopy(trainer.teacher_source), 'stair_scenario': None,
             'jump_scenario': None, 'recovery_scenario': None}
    return trainer, saved


@pytest.mark.parametrize('before,after', [(16, 32), (32, 16), (16, 16)])
def test_horizon_change_is_explicit_without_mutating_saved_configuration(before, after):
    trainer, saved = check_fixture(before, after)
    if before == 16:
        saved['config']['learning'].pop('imagination_horizon')
    snapshot = deepcopy(saved)
    rng = torch.get_rng_state().clone()
    trainer._check_saved(saved)
    assert trainer.imagination_horizon_changed == (None if before == after else {'before': before, 'after': after})
    assert saved == snapshot
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    assert LearningConfig().imagination_horizon == 16
    assert OnlineConfig.from_dict(asdict(trainer.config)) == trainer.config


@pytest.mark.parametrize('field,value', [
    ('bootstrap_gradient', True), ('anchor_coefficient', .1), ('gamma', .8),
    ('actor_learning_rate', .01), ('critic_learning_rate', .01), ('model_learning_rate', .01),
    ('estimator_learning_rate', .01), ('critic_epochs', 3),
])
def test_horizon_permission_does_not_allow_other_learning_changes(field, value):
    trainer, saved = check_fixture()
    learning = replace(trainer.config.learning, **{field: value})
    trainer.config = replace(trainer.config, learning=learning)
    with pytest.raises(ValueError, match='training recipe'):
        trainer._check_saved(saved)


@pytest.mark.parametrize('field,value', [('model_hidden', (32,)), ('actor_hidden', (32,)),
                                        ('fit_horizon', 16), ('imagination_batch_size', 2048),
                                        ('action_noise_std', .2), ('seed', 11)])
def test_horizon_permission_keeps_network_fit_and_sampling_configuration(field, value):
    trainer, saved = check_fixture()
    trainer.config = replace(trainer.config, **{field: value})
    with pytest.raises(ValueError, match='training recipe'):
        trainer._check_saved(saved)


@pytest.mark.parametrize('field,value', [
    ('schema_fingerprint', 'another-runtime-or-reward'),
    ('teacher_source', {'checkpoint': 'other'}), ('recovery_scenario', {'changed': True}),
])
def test_horizon_permission_does_not_alias_source_or_task_contracts(field, value):
    trainer, saved = check_fixture()
    saved[field] = value
    with pytest.raises(ValueError):
        trainer._check_saved(saved)


def assert_retained(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            assert_retained(actual[name], value)
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for first, second in zip(actual, expected):
            assert_retained(first, second)
    else:
        assert actual == expected


@pytest.mark.parametrize('before_horizon,after_horizon,before_microbatch,after_microbatch', [
    (16, 32, 512, 512), (16, 32, 4096, 2048), (32, 32, 4096, 2048),
])
def test_complete_online_restore_preserves_adam_replay_rng_and_records_recipe_once(
        tmp_path, monkeypatch, before_horizon, after_horizon, before_microbatch, after_microbatch):
    # Reuse genuine compact data and a small genuine dynamics model. Only the
    # native snapshot adapter and asset-backed constructor are replaced: the
    # online restore, learner, four Adams, replay and RNG restoration are real.
    checkpoint, base_model, replay = world_fixture(tmp_path)
    schema = StateSchema.from_dict(torch.load(checkpoint, weights_only=True)['schema'])
    previous = OnlineConfig(cycles=100, replay_steps=10, replay_device='cpu', model_hidden=(16,),
        actor_frame_dim=35, actor_hidden=(16,), estimator_hidden=(8,), critic_hidden=(16,),
        imagination_batch_size=4096, imagination_microbatch_size=before_microbatch, actor_updates=24,
        model_batch_size=4, validation_fraction=.5, guard_drop_fraction=.1, guard_mode='report',
        initial_actor='existing-initialization.pt', initial_actor_sha256='a' * 64,
        learning=LearningConfig(imagination_horizon=before_horizon, bootstrap_gradient=False,
                                actor_learning_rate=5e-5, anchor_coefficient=.1,
                                model_fit_precision='bfloat16'))

    def build(schema, runtime, config, **kwargs):
        actor = EstimatorActor(frame_dim=35, estimator_hidden=(8,), actor_hidden=(16,))
        critic = Critic(Critic.feature_size(schema.goal_dim, 65), (16,))
        return Learner(deepcopy(base_model), actor, critic, SimpleNamespace(), config, reward=None)

    learner = build(schema, {}, previous.learning)
    for index, name in enumerate(learner.optimizers):
        parameters = [p for group in learner.optimizers[name].param_groups for p in group['params']]
        assert learner._step(name, sum(parameter.square().sum() for parameter in parameters))
        learner.skipped[name] = index
    learner.set_anchor(learner.actor, torch.full((6,), .3))
    learner.imagination.contact_floor = (torch.tensor([1., 2.]), torch.tensor(3.))
    # Buffer preservation matters as much as weights during a full resume.
    learner.critic.norm.update(torch.full((2, learner.critic.norm.mean.numel()), .2))
    guard = NativeRegressionGuard(.1, 2, 5e-5, mode='report')
    guard.best_reward, guard.bad_cycles = .093, 4
    guard.recent.append(.091)
    source = {'checkpoint': 'same-teacher', 'qualification': 'data-only'}
    environment_state = {'curriculum': {'real_transitions': 123456, 'sample_counts': [4, 7]},
                         'source_pin': 'unchanged', 'native_rng': torch.Generator().manual_seed(41).get_state()}
    saved = {'format': CHECKPOINT_FORMAT, 'model_contract': MODEL_CONTRACT,
        'schema': schema.to_dict(), 'schema_fingerprint': schema.fingerprint,
        'config': asdict(previous), 'cycle': 78, 'replay_file': 'replay-a.pt',
        'learner': deepcopy(learner.state_dict()), 'environment': environment_state,
        'collection_steps': 1872, 'seed_source': {'data': 'original'}, 'teacher_source': source,
        'actor_initialization': {'source': 'retained'}, 'resume_migrations': [{'older': 'retained'}],
        'stair_scenario': None, 'jump_scenario': None, 'recovery_scenario': None,
        'sample_rng': torch.Generator().manual_seed(21).get_state(), 'sample_rng_device': 'cpu',
        'validation_rng': torch.Generator().manual_seed(22).get_state(), 'validation_rng_device': 'cpu',
        'collection_rng': torch.Generator().manual_seed(23).get_state(),
        'mixing': {'format': 'teacher-population-v1', 'priority': torch.tensor([1, 0]),
                   'rng': torch.Generator().manual_seed(24).get_state()},
        'torch_rng': torch.Generator().manual_seed(25).get_state(), 'cuda_rng': [],
        'guard': guard.state_dict(), 'task_norm_pending': False, 'critic_norm_from': None}
    torch.save(saved, checkpoint)
    dataset_before = deepcopy(replay.state_dict())

    class Environment:
        num_envs, device = 2, torch.device('cpu')

        def __init__(self):
            self.schema = schema
            self.current = SimpleNamespace(actor_observation=torch.zeros(2, 45), episode_id=torch.zeros(2).long())

        def load_training_state_dict(self, value):
            self.restored = deepcopy(value)

        def training_state_dict(self):
            return deepcopy(self.restored)

    monkeypatch.setattr('supermimic_rl.world_model_v2.online.make_learner', build)
    monkeypatch.setattr('supermimic_rl.world_model_v2.collection.SnapshotLayout',
                        lambda schema: SimpleNamespace(record_disturbances=False))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    requested = replace(previous, imagination_microbatch_size=after_microbatch,
                        learning=replace(previous.learning, imagination_horizon=after_horizon))
    output = tmp_path / 'resumed'
    trainer = OnlineTrainer(Environment(), lambda: None, requested, output,
        runtime={}, linkage_path=None, asset_directory=None, identified_bundle=None,
        resume=checkpoint, teacher_source=source)
    torch.testing.assert_close(trainer.learner.state_dict(), saved['learner'], atol=0, rtol=0)
    assert_retained(trainer.collector.replay.state_dict(), dataset_before)
    assert_retained(trainer.env.restored, environment_state)
    assert_retained(trainer.mixing.state_dict(), saved['mixing'])
    assert trainer.learner.config.imagination_horizon == after_horizon
    assert trainer.config.imagination_microbatch_size == after_microbatch
    assert trainer.config.imagination_batch_size == 4096
    assert trainer.config.actor_updates == 24
    assert trainer.learner.imagination.bootstrap_gradient is False
    assert trainer.learner.config.anchor_coefficient == .1
    assert trainer.cycle == 78 and trainer.collector.steps == 1872
    assert trainer.seed_source == saved['seed_source']
    assert trainer.teacher_source == source
    assert trainer.guard.restorations == trainer.guard_cold_cycles_remaining == 0
    assert_retained(trainer.guard.state_dict(), saved['guard'])
    for key, generator in [('sample', trainer.generator), ('validation', trainer.validation_generator),
                           ('collection', trainer.collector.generator)]:
        torch.testing.assert_close(generator.get_state(), saved[key + '_rng'], atol=0, rtol=0)
    torch.testing.assert_close(torch.get_rng_state(), saved['torch_rng'], atol=0, rtol=0)
    receipts = trainer.resume_migrations[len(saved['resume_migrations']):]
    expected_changes = {}
    if before_horizon != after_horizon:
        expected_changes['imagination_horizon'] = {'before': before_horizon, 'after': after_horizon}
    if before_microbatch != after_microbatch:
        expected_changes['imagination_microbatch_size'] = {
            'before': before_microbatch, 'after': after_microbatch}
    assert trainer.resume_migrations[:len(saved['resume_migrations'])] == saved['resume_migrations']
    assert len(receipts) == len(expected_changes)
    for field, change in expected_changes.items():
        receipt = next(receipt for receipt in receipts if field in receipt)
        assert receipt[field] == change
        assert receipt['resumed_cycle'] == 78 and receipt['additional_warmup_cycles'] == 0
        assert receipt['learner_replay_rng_preserved'] is True
        if field == 'imagination_microbatch_size':
            assert receipt['imagination_batch_size'] == 4096
            assert receipt['actor_updates'] == 24
    assert json.loads((output / 'resume-migrations.json').read_text()) == trainer.resume_migrations

    # The next complete pair carries the requested recipe, original weights,
    # original data and receipt. Reloading the same H32 must not add a second
    # migration or warmup, nor consume a different next sampling draw.
    trainer.save(with_replay=True)
    pointer = json.loads((output / 'resume.json').read_text())
    target = torch.load(output / pointer['checkpoint'], weights_only=True)
    assert target['config']['learning']['imagination_horizon'] == after_horizon
    assert target['config']['imagination_microbatch_size'] == after_microbatch
    torch.testing.assert_close(target['learner'], saved['learner'], atol=0, rtol=0)
    assert_retained(torch.load(output / pointer['replay_file'], weights_only=True)['replay'], dataset_before)
    restored = OnlineTrainer(Environment(), lambda: None, requested, tmp_path / 'again',
        runtime={}, linkage_path=None, asset_directory=None, identified_bundle=None,
        resume=output / pointer['checkpoint'], teacher_source=source)
    assert restored.resume_migrations == trainer.resume_migrations
    assert restored.imagination_horizon_changed is None
    assert restored.imagination_microbatch_changed is None
    torch.testing.assert_close(restored.learner.state_dict(), trainer.learner.state_dict(), atol=0, rtol=0)
    expected = trainer.collector.replay.sample_windows(2, 2, env_ids=trainer.train_ids,
                                                     generator=trainer.generator)
    actual = restored.collector.replay.sample_windows(2, 2, env_ids=restored.train_ids,
                                                     generator=restored.generator)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(restored.generator.get_state(), trainer.generator.get_state(), atol=0, rtol=0)
    assert_retained(torch.load(checkpoint, weights_only=True), saved)
