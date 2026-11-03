"""Critic-only warmup fits the critic on imagined returns without moving the actor."""
from copy import deepcopy

import torch

from support.world_v2.jump_imagination import fixture


def test_critic_warmup_keeps_the_actor_and_trains_the_critic():
    learner, start, frames = fixture()
    actor = deepcopy(learner.actor.state_dict())
    critic = deepcopy(learner.critic.state_dict())
    result = learner.update_policy(start, frames, microbatch_size=2, train_actor=False)
    assert result['actor/updated'] is False and learner.updates['actor'] == 0 and learner.skipped['actor'] == 0
    for name, value in learner.actor.state_dict().items():
        if not name.startswith('estimator.'):
            torch.testing.assert_close(value, actor[name], rtol=0, atol=0)
    assert learner.updates['critic'] > 0
    assert any(not torch.equal(value, critic[name]) for name, value in learner.critic.state_dict().items()
               if not name.startswith('norm.'))
    assert all(parameter.grad is None for parameter in learner.actor.decoder.parameters())
