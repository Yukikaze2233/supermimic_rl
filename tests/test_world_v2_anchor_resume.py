"""Resume changes only the scalar weight of the original saved teacher anchor."""
from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.world_model_v2.learning import Learner, LearningConfig
from supermimic_rl.world_model_v2.online import CHECKPOINT_FORMAT, OnlineConfig, OnlineTrainer
from supermimic_rl.world_model_v2.records import MODEL_CONTRACT


def check_fixture(before=.1, after=.0025, *, anchor=True):
    previous = OnlineConfig(cycles=100, actor_frame_dim=35,
        initial_actor='same-initial-teacher.pt', initial_actor_sha256='a' * 64,
        learning=LearningConfig(anchor_coefficient=before, bootstrap_gradient=False))
    trainer = object.__new__(OnlineTrainer)
    trainer.env = SimpleNamespace(schema=SimpleNamespace(fingerprint='same-source'))
    trainer.teacher_source = {'checkpoint': 'same-teacher', 'qualification': 'data-only'}
    trainer.stair_scenario = trainer.jump_scenario = trainer.recovery_scenario = None
    trainer.config = replace(previous, learning=replace(previous.learning, anchor_coefficient=after))
    saved = {'format': CHECKPOINT_FORMAT, 'model_contract': MODEL_CONTRACT,
        'schema_fingerprint': 'same-source', 'config': asdict(previous), 'cycle': 10,
        'teacher_source': deepcopy(trainer.teacher_source),
        'stair_scenario': None, 'jump_scenario': None, 'recovery_scenario': None,
        'learner': {'anchor': {'actor': 'preserved-loader-fixture', 'std': [1.] * 6}} if anchor else {}}
    return trainer, saved


@pytest.mark.parametrize('coefficient', [-.1, float('nan'), float('inf'), -float('inf')])
def test_anchor_preserves_existing_finite_nonnegative_validation(coefficient):
    with pytest.raises(ValueError, match='finite and nonnegative'):
        LearningConfig(anchor_coefficient=coefficient)


@pytest.mark.parametrize('before,after', [(.1, .0025), (.1, 0.), (0., .0025), (.1, .2), (.1, .1)])
def test_explicit_anchor_recipe_change_keeps_saved_identity_and_rng(before, after):
    trainer, saved = check_fixture(before, after)
    before_saved = deepcopy(saved)
    rng = torch.get_rng_state().clone()
    trainer._check_saved(saved)
    assert trainer.anchor_coefficient_changed == (
        None if before == after else {'before': before, 'after': after})
    assert trainer.imagination_horizon_changed is None
    assert trainer.imagination_microbatch_changed is None
    assert saved == before_saved
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize('before', [0., .1])
def test_nonzero_change_cannot_manufacture_a_missing_anchor_from_initial_actor(before):
    trainer, saved = check_fixture(before, .0025, anchor=False)
    with pytest.raises(ValueError, match='requires a saved teacher anchor'):
        trainer._check_saved(saved)


def test_legacy_default_zero_needs_no_anchor_and_cannot_enable_one_on_resume():
    trainer, saved = check_fixture(0., 0., anchor=False)
    saved['config']['learning'].pop('anchor_coefficient')
    trainer._check_saved(saved)
    assert trainer.anchor_coefficient_changed is None
    trainer.config = replace(trainer.config, learning=replace(trainer.config.learning, anchor_coefficient=.0025))
    with pytest.raises(ValueError, match='requires a saved teacher anchor'):
        trainer._check_saved(saved)


@pytest.mark.parametrize('field,value', [
    ('bootstrap_gradient', True), ('gamma', .8), ('gradient_clip', .5),
    ('actor_learning_rate', 1e-5), ('estimator_learning_rate', 1e-5),
    ('model_learning_rate', 1e-5), ('critic_learning_rate', 1e-5),
])
def test_anchor_change_does_not_authorize_other_learning_recipe_changes(field, value):
    trainer, saved = check_fixture()
    trainer.config = replace(trainer.config, learning=replace(trainer.config.learning, **{field: value}))
    with pytest.raises(ValueError, match='training recipe'):
        trainer._check_saved(saved)


@pytest.mark.parametrize('field,value', [
    ('initial_actor', 'different-teacher.pt'), ('initial_actor_sha256', 'b' * 64),
    ('model_hidden', (16,)), ('teacher_fraction', .25), ('seed', 11),
])
def test_anchor_change_preserves_teacher_model_and_sampling_recipe(field, value):
    trainer, saved = check_fixture()
    trainer.config = replace(trainer.config, **{field: value})
    with pytest.raises(ValueError, match='training recipe'):
        trainer._check_saved(saved)


@pytest.mark.parametrize('coefficient', [.0025, 0.])
def test_restored_learner_uses_requested_weight_in_actual_policy_backward(coefficient, monkeypatch):
    from support.world_v2.jump_imagination import fixture

    original, start, frames = fixture()
    original.set_anchor(original.actor, torch.full((6,), .3))
    with torch.no_grad():
        original.actor.decoder[-1].bias.add_(.05)
    saved = deepcopy(original.state_dict())
    config = replace(original.config, imagination_horizon=4, anchor_coefficient=coefficient)
    restored = Learner(original.model, original.actor, original.critic, original.imagination.termination,
        config, reward=original.imagination.reward, stair_task=original.imagination.stair_task,
        stair_source=original.imagination.stair_source, jump_task=original.imagination.jump_task)
    restored.load_state_dict(saved)
    torch.testing.assert_close(restored.state_dict(), saved, atol=0, rtol=0)
    derivative = []
    compute_deviation = restored._anchor_deviation

    def observe_scalar_gradient(history):
        deviation = compute_deviation(history)
        deviation.register_hook(lambda gradient: derivative.append(float(gradient)))
        return deviation

    monkeypatch.setattr(restored, '_anchor_deviation', observe_scalar_gradient)
    outcome = restored.update_policy(start, frames, microbatch_size=2)
    assert outcome['actor/updated']
    assert derivative == ([] if coefficient == 0 else [pytest.approx(coefficient)])
    torch.testing.assert_close(restored.state_dict()['anchor'], saved['anchor'], atol=0, rtol=0)
