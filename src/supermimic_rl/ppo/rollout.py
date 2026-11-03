"""Fixed-length PPO rollout storage and GAE returns.

Truncated rows end the episode for GAE but are bootstrapped with the value of
their pre-reset endpoint (added to the reward), so time limits and spatial
course exits are not treated as failures. Terminated rows receive no bootstrap.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor


@dataclass
class Rollout:
    history: Tensor        # [T, N, 5, 35]
    critic: Tensor         # [T, N, C]
    action: Tensor         # [T, N, 6] sampled (pre-clip) actions
    log_prob: Tensor       # [T, N]
    mean: Tensor           # [T, N, 6]
    std: Tensor            # [T, N, 6]
    value: Tensor          # [T, N]
    reward: Tensor         # [T, N] including truncation bootstrap
    done: Tensor           # [T, N]
    label: Tensor          # [T, N, 5]
    label_valid: Tensor    # [T, N]
    valid: Tensor          # [T, N] False censors an invalid transition, not a physical failure.

    @classmethod
    def empty(cls, steps: int, num_envs: int, critic_dim: int, device) -> Rollout:
        def zeros(*shape, dtype=torch.float32):
            return torch.zeros(steps, num_envs, *shape, dtype=dtype, device=device)

        return cls(zeros(5, 35), zeros(critic_dim), zeros(6), zeros(), zeros(6), zeros(6), zeros(), zeros(),
                   zeros(dtype=torch.bool), zeros(5), zeros(dtype=torch.bool),
                   torch.ones(steps, num_envs, dtype=torch.bool, device=device))

    def flatten(self) -> dict[str, Tensor]:
        return {f.name: getattr(self, f.name).flatten(0, 1) for f in fields(self)}


def gae(reward: Tensor, value: Tensor, done: Tensor, last_value: Tensor, gamma: float, lam: float,
        *, valid: Tensor | None = None):
    """Cut invalid intervals at their last trustworthy decision state.

    An invalid interval has zero advantage and its old value as the return.
    The preceding valid interval therefore bootstraps V(s_t), without learning
    a fabricated zero-value terminal state or propagating the post-reset tail.
    """
    advantages = torch.zeros_like(reward)
    running = torch.zeros_like(last_value)
    next_value = last_value
    for step in reversed(range(reward.shape[0])):
        alive = (~done[step]).to(reward.dtype)
        delta = reward[step] + gamma * next_value * alive - value[step]
        running = delta + gamma * lam * alive * running
        if valid is not None:
            running = torch.where(valid[step], running, 0.)
        advantages[step] = running
        next_value = value[step]
    return advantages + value, advantages
