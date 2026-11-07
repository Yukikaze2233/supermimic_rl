"""Cosine learning-rate decay over actor updates (SuperMimic Table A.5)."""
from dataclasses import asdict, replace
import json
import math

import pytest

from supermimic_rl.world_model_v2.learning import LearningConfig, LearningRateDecay


def test_cosine_factor_reaches_the_final_fraction_and_then_stays_constant():
    decay = LearningRateDecay(updates=100, final_fraction=.03)
    assert decay.factor(0) == pytest.approx(1.)
    assert decay.factor(50) == pytest.approx(.03 + .97 * .5)
    assert decay.factor(100) == pytest.approx(.03)
    assert decay.factor(250) == pytest.approx(.03)
    values = [decay.factor(step) for step in range(0, 101, 5)]
    assert all(later <= earlier for earlier, later in zip(values, values[1:]))
    assert decay.factor(25) == pytest.approx(.03 + .97 * .5 * (1 + math.cos(math.pi / 4)))


@pytest.mark.parametrize('kwargs', [dict(updates=0), dict(updates=1.5), dict(updates=10, final_fraction=0.),
                                    dict(updates=10, final_fraction=1.5)])
def test_invalid_decay_is_rejected(kwargs):
    with pytest.raises(ValueError):
        LearningRateDecay(**kwargs)


def test_config_round_trips_through_json_and_defaults_to_constant_rates():
    assert LearningConfig().learning_rate_decay is None
    config = LearningConfig(learning_rate_decay={'updates': 40000})
    assert config.learning_rate_decay == LearningRateDecay(40000, .03)
    restored = LearningConfig(**json.loads(json.dumps(asdict(config))))
    assert restored == config
    # A saved config without the field keeps the legacy (constant) recipe.
    legacy = asdict(LearningConfig())
    legacy.pop('learning_rate_decay')
    assert asdict(LearningConfig(**legacy)) == asdict(LearningConfig())
    with pytest.raises(ValueError):
        LearningConfig(learning_rate_decay=3)


def _learner(config):
    from supermimic_rl.world_model_v2.learning import Learner
    from support.world_v2.jump_imagination import fixture

    learner, start, frames = fixture()
    scheduled = Learner(learner.model, learner.actor, learner.critic, learner.imagination.termination,
                        replace(learner.config, **config), reward=learner.imagination.reward,
                        stair_task=learner.imagination.stair_task, stair_source=learner.imagination.stair_source,
                        jump_task=learner.imagination.jump_task)
    return scheduled, start, frames


def test_every_optimizer_steps_at_the_decayed_rate_of_the_actor_clock():
    learner, start, frames = _learner({'learning_rate_decay': LearningRateDecay(updates=4, final_fraction=.1)})
    learner.updates['actor'] = 2
    expected = .1 + .9 * .5
    result = learner.update_policy(start, frames, microbatch_size=2)
    assert result['actor/updated']
    # The actor stepped at the factor of its pre-step clock; the report uses the new clock.
    assert learner.optimizers['actor'].param_groups[0]['lr'] == pytest.approx(learner.config.actor_learning_rate
                                                                              * expected)
    assert result['learning_rate/factor'] == pytest.approx(LearningRateDecay(4, .1).factor(3))
    assert learner.updates['critic'] > 0
    assert learner.optimizers['critic'].param_groups[0]['lr'] == pytest.approx(
        learner.config.critic_learning_rate * LearningRateDecay(4, .1).factor(3))
    # Model fitting before policy learning would use the same actor clock.
    assert learner.optimizers['model'].param_groups[0]['lr'] == learner.config.model_learning_rate


def test_disabled_decay_leaves_externally_managed_rates_untouched():
    learner, start, frames = _learner({})
    for group in learner.optimizers['actor'].param_groups:
        group['lr'] = 1e-7   # e.g. a guard halving or a resume reset
    learner.update_policy(start, frames, microbatch_size=2)
    assert learner.optimizers['actor'].param_groups[0]['lr'] == 1e-7
    assert learner.learning_rate_factor() == 1.


def test_decay_cannot_share_the_actor_rate_with_a_restoring_guard():
    from supermimic_rl.world_model_v2.online import OnlineConfig

    decay = {'learning_rate_decay': {'updates': 1000}}
    with pytest.raises(ValueError, match='restoring guard'):
        OnlineConfig.from_dict({'learning': decay, 'guard_drop_fraction': .05, 'guard_mode': 'restore'})
    OnlineConfig.from_dict({'learning': decay, 'guard_drop_fraction': .05, 'guard_mode': 'report'})
