"""Analytic target environment and a small PPO configuration for trainer tests."""

import torch
from supermimic_rl.ppo.environment import Observation, Step
from supermimic_rl.ppo.trainer import PPOConfig


class TargetEnvironment:
    """Reward = -(action - target)^2 where target is frame[..., 0]; 20-step episodes, rare failures."""

    def __init__(self, num_envs=64, seed=0):
        self.num_envs, self.device = num_envs, torch.device("cpu")
        self.generator = torch.Generator().manual_seed(seed)
        self.progress, self.estimates = [], 0
        self.steps = torch.zeros(num_envs)

    def _frames(self):
        frame = torch.randn(self.num_envs, 35, generator=self.generator) * .1
        frame[:, 0] = self.target
        return frame

    def _observation(self, valid):
        self.history = torch.cat((self.history[:, 1:], self._frames()[:, None]), 1)
        label = torch.cat((self.target[:, None].expand(-1, 3), torch.ones(self.num_envs, 2)), -1)
        return Observation(self.history.clone(), torch.cat((self.history[:, -1], self.steps[:, None] / 20), -1),
                           label, valid)

    def reset(self):
        self.target = torch.rand(self.num_envs, generator=self.generator) * 2 - 1
        self.history = self._frames()[:, None].expand(-1, 5, -1).clone()
        self.steps.zero_()
        return self._observation(torch.zeros(self.num_envs, dtype=torch.bool))

    def set_progress(self, real_transitions, actor_updates):
        self.progress.append((real_transitions, actor_updates))

    def set_estimate(self, history, estimate):
        assert estimate.shape == (self.num_envs, 5)
        self.estimates += 1

    def step(self, action):
        assert (action.abs() <= torch.tensor([3., 3., 3., 3., 9., 9.]) + 1e-6).all()
        reward = -(action[:, 0] - self.target).square()
        self.steps += 1
        truncated = self.steps >= 20
        terminated = (torch.rand(self.num_envs, generator=self.generator) < .01) & ~truncated
        terminal = torch.cat((self.history[:, -1], self.steps[:, None] / 20), -1)
        done = terminated | truncated
        self.steps[done] = 0
        self.target = torch.where(done, torch.rand(self.num_envs, generator=self.generator) * 2 - 1, self.target)
        observation = self._observation(~done)
        return Step(observation, reward, terminated, truncated, terminal,
                    {"forward_error_m_s": action[:, 0] - self.target})

    def termination_reasons(self):
        return {"episodes_ended_by_reason": {}}


def config(**overrides):
    values = dict(steps_per_update=20, updates=40, epochs=4, minibatches=4, learning_rate=3e-4,
                  actor_hidden=(64, 64), estimator_hidden=(32,), critic_hidden=(64,), initial_std=.5,
                  checkpoint_interval=20)
    values.update(overrides)
    return PPOConfig(**values)
