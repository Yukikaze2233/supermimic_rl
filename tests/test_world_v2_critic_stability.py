"""Bounded critic inputs keep imagined lambda-return bootstrapping from extrapolating."""
import pytest
import torch

from supermimic_rl.world_model_v2.imagination import Critic


def test_input_clip_bounds_normalized_features_only_when_configured():
    critic = Critic(3, (4,), input_clip=5.)
    features = torch.tensor([[0., 300., -300.]])
    torch.testing.assert_close(critic(features), critic.net(torch.tensor([[0., 5., -5.]])).squeeze(-1))
    unbounded = Critic(3, (4,))
    unbounded.load_state_dict(critic.state_dict())
    assert not torch.allclose(unbounded(features), critic(features))
    with pytest.raises(ValueError):
        Critic(3, (4,), input_clip=0.)


def test_bootstrap_value_can_be_excluded_from_the_actor_gradient():
    from dataclasses import replace

    from supermimic_rl.world_model_v2.learning import Learner, frozen_parameters
    from support.world_v2.jump_imagination import fixture

    learner, start, frames = fixture()
    gradients = {}
    for enabled in (True, False):
        constant = Learner(learner.model, learner.actor, learner.critic, learner.imagination.termination,
                           replace(learner.config, bootstrap_gradient=enabled), reward=learner.imagination.reward,
                           stair_task=learner.imagination.stair_task, stair_source=learner.imagination.stair_source,
                           jump_task=learner.imagination.jump_task)
        with frozen_parameters(learner.model, learner.critic):
            rollout = constant.imagination.rollout(start, frames, 4)
        # The actor objective still depends on the actor through the rewards.
        assert rollout.objective.requires_grad
        gradients[enabled] = rollout.values[-1].requires_grad
        if enabled:
            values = rollout.values[-1].detach()
        else:
            torch.testing.assert_close(rollout.values[-1], values)
    assert gradients == {True: True, False: False}
