"""Analytic target environment and a small PPO configuration for trainer tests."""

from types import SimpleNamespace

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


class NativeCommandBoundary:
    num_envs = 2
    device = "cpu"

    def __init__(self):
        self._command = torch.tensor([[.4, 0., .2, .30], [-.3, 0., -.5, .32]])
        self._terminal_critic_182 = torch.full((2, 182), 7.)
        self.obs_buf = {"policy": torch.zeros(2, 5, 42), "critic": torch.zeros(2, 182)}
        self._last_state = SimpleNamespace(
            velocity_origin_body_m_s=torch.tensor([[.45, 0., 0.], [-.25, 0., 0.]]),
            angular_velocity_body_rad_s=torch.tensor([[0., 0., .1], [0., 0., -.4]]),
            gravity_body=torch.tensor([[0., 0., -1.], [0., 0., -1.]]),
            height_above_ground_m=torch.tensor([.29, .24]),
            wheel_ground_force_n=torch.full((2, 2), 100.),
            contact_measurement_valid=torch.ones(2, dtype=torch.bool))
        # Deliberately differs from a gate reconstructed at the endpoint:
        # the producer averages all four physical substeps.
        self._formal_result = SimpleNamespace(
            gate_fractions={"velocity_height_gate": torch.tensor([.9, .3]),
                            "ordinary": torch.ones(2)},
            component_densities={"positive.forward": torch.tensor([1.5, 2.]),
                                 "cost.height": torch.tensor([-.5, -2.])},
            event_rewards={"failure": torch.tensor([0., -4.]),
                           "physical_skill_success": torch.zeros(2)},
            physical_step_reward=torch.tensor([.02, -4.], dtype=torch.float64),
            learning_reward=torch.tensor([.02, -4.]))

    def step(self, action):
        # Row 0 changes command; row 1 terminates and receives a reset command.
        # Both mutations happen after the reported physical endpoint.
        self._command.copy_(torch.tensor([[-2., 0., 3., .4], [2., 0., 4., .28]]))
        self.obs_buf["policy"][:, -1, :4] = self._command
        self.obs_buf["critic"].fill_(8.)
        return (self.obs_buf, torch.tensor([.02, -4.]), torch.tensor([False, True]),
                torch.zeros(2, dtype=torch.bool),
                {"interval_observation_valid": torch.ones(2, dtype=torch.bool),
                 "numerical_invalid": torch.zeros(2, dtype=torch.bool)})
