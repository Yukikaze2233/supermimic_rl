"""Numerical integration fixture, deliberately NOT a wheel-leg simulator."""
from __future__ import annotations

import torch

from ..contracts import ACTION_BOUNDS, Field, Snapshot, StateSchema, Transition, encoded_goal


class SyntheticEnvironment:
    """Stable six-axis dynamics for exercising the entire learning/data pipeline."""

    def __init__(self, num_envs: int = 16, *, device="cpu", episode_steps: int = 80, seed: int = 7):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.episode_steps = episode_steps
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.schema = StateSchema((Field("position", (6,), "rad", "fixture"),
                                   Field("velocity", (6,), "rad/s", "fixture"),
                                   Field("previous_issued_action", (6,), "normalized", "policy")),
                                  (Field("response", (1,), "dimensionless", "fixture"),),
                                  source="synthetic-fixture-v1-not-robot-physics")
        self.position = torch.zeros(num_envs, 6, device=self.device)
        self.velocity = torch.zeros_like(self.position)
        self.previous_action = torch.zeros_like(self.position)
        self.episode_id = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self.steps = torch.zeros_like(self.episode_id)
        self.commands = torch.zeros(num_envs, 4, device=self.device)
        self.response = torch.full((num_envs, 1), 0.12, device=self.device)
        self.current = None

    def _snapshot(self):
        observation = torch.zeros(self.num_envs, 35, device=self.device)
        observation[:, :4] = self.commands
        observation[:, 4:7] = self.velocity[:, :3] * 0.03
        observation[:, 9] = -1
        observation[:, 10:14] = self.position[:, :4]
        observation[:, 16:22] = self.velocity * 0.1
        observation[:, 22:28] = self.previous_action
        observation[:, 28] = 1
        observation[:, 34] = (self.steps * 0.02).clamp_max(5)
        return Snapshot(torch.cat((self.position, self.velocity, self.previous_action), -1).clone(),
                        observation.clone(), encoded_goal(observation), self.response.clone(),
                        self.episode_id.clone(), self.episode_id.clone(),
                        torch.ones_like(self.steps, dtype=torch.bool), observation.clone())

    def _reset_rows(self, mask):
        self.position[mask] = 0
        self.velocity[mask] = 0
        self.previous_action[mask] = 0
        self.steps[mask] = 0
        count = int(mask.sum())
        self.commands[mask, 0] = torch.rand(count, generator=self.generator, device=self.device) * 1.6 - 0.8
        self.commands[mask, 3] = 0.3 * 5

    def reset(self):
        self.episode_id += 1
        self._reset_rows(torch.ones_like(self.steps, dtype=torch.bool))
        self.current = self._snapshot()
        return self.current

    @torch.no_grad()
    def step(self, action):
        if self.current is None:
            raise RuntimeError("Reset before stepping")
        start = self.current.clone()
        bounds = action.new_tensor(ACTION_BOUNDS)
        issued = action.clamp(-bounds, bounds)
        target_velocity = issued / bounds * 2
        self.velocity.lerp_(target_velocity, self.response)
        self.position += self.velocity * 0.02
        self.previous_action.copy_(issued)
        self.steps += 1
        reward = 0.02 * (1.0 - (self.velocity[:, 4:].mean(-1) - self.commands[:, 0]).square()
                         - 0.01 * (issued / bounds).square().sum(-1))
        terminated = self.velocity.abs().amax(-1) > 10
        truncated = self.steps >= self.episode_steps
        end = self._snapshot()
        transition = Transition(start, issued.clone(), end, reward, terminated, truncated,
                                torch.ones_like(terminated))
        reset = terminated | truncated
        self.episode_id[reset] += 1
        if reset.any():
            self._reset_rows(reset)
        self.current = self._snapshot()
        return transition

    def close(self):
        pass
