"""Optional CUDA fitting precision never changes policy or checkpoint semantics."""
from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.world_model_v2.dynamics import MotionModel
from supermimic_rl.world_model_v2.fitting import WorldModelFit
from supermimic_rl.world_model_v2.learning import Learner, LearningConfig
from supermimic_rl.world_model_v2.online import CHECKPOINT_FORMAT, OnlineConfig, OnlineTrainer
from supermimic_rl.world_model_v2.records import MODEL_CONTRACT
from support.world_v2.offline_continuation import fixture as world_fixture


@pytest.mark.parametrize('precision', ['float16', 'bf16', '', None, 1])
def test_fit_precision_rejects_unknown_configurations(precision):
    with pytest.raises(ValueError, match='precision'):
        LearningConfig(model_fit_precision=precision)
    with pytest.raises(ValueError, match='precision'):
        with MotionModel(3, (4,)).fitting_precision(precision):
            pytest.fail('Invalid precision entered the fitting scope')


def test_legacy_configuration_defaults_to_float32_and_roundtrips():
    old = OnlineConfig.from_dict({'learning': {'model_learning_rate': 3e-4}})
    assert old.learning.model_fit_precision == 'float32'
    changed = replace(old, learning=replace(old.learning, model_fit_precision='bfloat16'))
    assert OnlineConfig.from_dict(asdict(changed)) == changed
    assert changed.learning.model_learning_rate == old.learning.model_learning_rate


def test_precision_scope_restores_on_nesting_and_exception_without_checkpoint_keys():
    motion = MotionModel(3, (4,))
    keys = tuple(motion.state_dict())
    with pytest.raises(RuntimeError, match='failed fit'):
        with motion.fitting_precision('bfloat16'):
            assert motion._fit_precision == 'bfloat16'
            with motion.fitting_precision('float32'):
                assert motion._fit_precision == 'float32'
            assert motion._fit_precision == 'bfloat16'
            raise RuntimeError('failed fit')
    assert motion._fit_precision == 'float32'
    assert tuple(motion.state_dict()) == keys
    assert all(value.dtype == torch.float32 for value in motion.state_dict().values())


def test_cpu_bfloat16_request_keeps_reference_outputs_gradients_and_normalization(monkeypatch):
    motion = MotionModel(7, (8,))
    with torch.no_grad():
        motion.net[-1].weight.uniform_(-.1, .1)
        motion.output_norm.mean.uniform_(-1., 1.)
        motion.output_norm.var.uniform_(.1, 2.)
    inputs = torch.randn(5, 7, requires_grad=True)
    snapshot = deepcopy(motion.state_dict())
    expected = motion(inputs)[1]
    expected_grad = torch.autograd.grad(expected.square().sum(), (inputs, *motion.parameters()))

    def forbidden(*args, **kwargs):
        pytest.fail('CPU must not enter an autocast path')

    monkeypatch.setattr(torch, 'autocast', forbidden)
    with motion.fitting_precision('bfloat16'):
        actual = motion(inputs)[1]
    actual_grad = torch.autograd.grad(actual.square().sum(), (inputs, *motion.parameters()))
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for actual_value, expected_value in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual_value, expected_value, atol=0, rtol=0)
    for name, value in snapshot.items():
        torch.testing.assert_close(motion.state_dict()[name], value, atol=0, rtol=0)


@pytest.mark.parametrize('fails', [False, True])
def test_fit_scope_excludes_normalizers_optimizer_and_subsequent_forward(tmp_path, monkeypatch, fails):
    _, model, replay = world_fixture(tmp_path)
    learner = object.__new__(Learner)
    learner.model, learner.fit = model, WorldModelFit(model)
    learner.config = LearningConfig(model_fit_precision='bfloat16')
    learner.optimizers = {'model': torch.optim.Adam(model.parameters(), lr=2e-4)}
    learner.updates, learner.skipped = {'model': 0}, {'model': 0}
    windows = replay.sample_windows(2, 2, env_ids=[0], generator=torch.Generator().manual_seed(4))
    phases = []
    observe, terms, step = learner.fit.observe, learner.fit.terms, learner._step

    def observed(*args, **kwargs):
        phases.append(('observe', model.motion._fit_precision))
        return observe(*args, **kwargs)

    def fitted(*args, **kwargs):
        phases.append(('terms', model.motion._fit_precision))
        if fails:
            raise RuntimeError('failed fit')
        return terms(*args, **kwargs)

    def stepped(*args, **kwargs):
        phases.append(('optimizer', model.motion._fit_precision))
        return step(*args, **kwargs)

    monkeypatch.setattr(learner.fit, 'observe', observed)
    monkeypatch.setattr(learner.fit, 'terms', fitted)
    monkeypatch.setattr(learner, '_step', stepped)
    if fails:
        with pytest.raises(RuntimeError, match='failed fit'):
            learner.fit_model(windows)
        assert learner.updates['model'] == 0
        assert not learner.optimizers['model'].state
    else:
        assert learner.fit_model(windows)['model/updated']
        assert learner.updates['model'] == 1
        for state in learner.optimizers['model'].state.values():
            assert all(value.dtype == torch.float32 for value in state.values() if isinstance(value, torch.Tensor))
    assert phases == [('observe', 'float32'), ('terms', 'bfloat16')] + (
        [] if fails else [('optimizer', 'float32')])
    assert model.motion._fit_precision == 'float32'
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    seen = []
    handle = model.motion.net.register_forward_pre_hook(lambda module, args: seen.append(model.motion._fit_precision))
    model.motion(torch.zeros(2, model.motion.input_norm.mean.numel()))
    handle.remove()
    assert seen == ['float32']


def test_cpu_precision_change_restores_full_learner_and_adam_before_identical_continuation(tmp_path):
    _, model, replay = world_fixture(tmp_path)
    previous = object.__new__(Learner)
    previous.model, previous.fit = model, WorldModelFit(model)
    previous.config = LearningConfig()
    previous.actor, previous.critic = torch.nn.Linear(3, 2), torch.nn.Linear(3, 1)
    previous.optimizers = {'model': torch.optim.Adam(model.parameters(), lr=2e-4)}
    previous.updates, previous.skipped = {'model': 0}, {'model': 0}
    previous.anchor = None
    previous.imagination = SimpleNamespace(contact_floor=None)
    windows = replay.sample_windows(2, 2, env_ids=[0], generator=torch.Generator().manual_seed(4))
    assert previous.fit_model(windows)['model/updated']
    saved = deepcopy(previous.state_dict())
    resumed = deepcopy(previous)
    resumed.config = replace(previous.config, model_fit_precision='bfloat16')
    with torch.no_grad():
        for parameter in resumed.model.parameters():
            parameter.zero_()
    resumed.optimizers['model'].state.clear()
    resumed.load_state_dict(saved)
    torch.testing.assert_close(resumed.state_dict(), saved, atol=0, rtol=0)
    assert resumed.model.motion._fit_precision == 'float32'
    assert resumed.config.model_fit_precision == 'bfloat16'
    rng = torch.get_rng_state().clone()
    assert previous.fit_model(windows) == resumed.fit_model(windows)
    torch.testing.assert_close(resumed.state_dict(), previous.state_dict(), atol=0, rtol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)


@pytest.mark.parametrize('before,after', [('float32', 'bfloat16'), ('bfloat16', 'float32'), ('float32', 'float32')])
def test_resume_only_allows_precision_change_and_preserves_saved_learner_rng(before, after):
    old = OnlineConfig(cycles=100, learning=LearningConfig(model_fit_precision=before))
    stored = asdict(old)
    if before == 'float32':
        stored['learning'].pop('model_fit_precision')  # Pre-field checkpoints use FP32.
    saved = {'format': CHECKPOINT_FORMAT, 'model_contract': MODEL_CONTRACT, 'schema_fingerprint': 'native',
        'config': stored, 'cycle': 10, 'teacher_source': None, 'stair_scenario': None,
        'jump_scenario': None, 'recovery_scenario': None,
        'learner': {'weights': torch.tensor([1., 2.]), 'adam': torch.tensor([.1, .2])},
        'sample_rng': torch.Generator().manual_seed(8).get_state()}
    original = deepcopy(saved)
    trainer = object.__new__(OnlineTrainer)
    trainer.env = SimpleNamespace(schema=SimpleNamespace(fingerprint='native'))
    trainer.teacher_source = trainer.stair_scenario = trainer.jump_scenario = trainer.recovery_scenario = None
    trainer.config = replace(old, learning=replace(old.learning, model_fit_precision=after))
    rng = torch.get_rng_state().clone()
    trainer._check_saved(saved)
    assert trainer.model_fit_precision_changed == (None if before == after else {'before': before, 'after': after})
    assert saved['config'] == original['config']
    for name in saved['learner']:
        torch.testing.assert_close(saved['learner'][name], original['learner'][name], atol=0, rtol=0)
    torch.testing.assert_close(saved['sample_rng'], original['sample_rng'], atol=0, rtol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    for field, value in (('model_learning_rate', .01), ('imagination_horizon', 32), ('critic_learning_rate', .01)):
        trainer.config = replace(old, learning=replace(old.learning, model_fit_precision=after, **{field: value}))
        with pytest.raises(ValueError, match='training recipe'):
            trainer._check_saved(saved)
