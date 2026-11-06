"""Fudan-style estimator + history MLP actor, and a privileged critic.

Deployment interface (RMCS 5x35): the actor reads the five most recent 35D
deployed frames, oldest first. A history encoder estimates body velocity and
wheel contact; its output enters the actor detached, so it is trained only by
its supervised loss (Fudan ``ActorCriticSequence``). Actions are the existing
six targets: four leg position targets and two wheel velocity targets, clipped
to the runtime action bounds when executed.

The Gaussian mean is ``tanh(network) * action_bounds``, the same output form as
the SuperMimic student. With an unbounded mean the PPO2000 run drove both hip
means to about twice the clip bound; every sample then executed the same
boundary action and exploration collapsed. ``mean_squash="none"`` only loads
such legacy checkpoints.

``action_bounds`` must equal the native runtime's ``task.action_bounds``. The
legacy +-3 leg bound limits each hip target to +-43 deg around nominal; the
full-turn bound +-4*pi lets the common leg angle reach any orientation, as the
coaxial hip allows. The knee is limited by the asset's mechanical stop.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..contracts import ACTION_BOUNDS
from ..chassis_observation import COMMAND_CONTRACTS, LEGACY_COMMANDS
from ..actor_features import ActorFeatures

HISTORY_LENGTH = 5
FRAME_DIM = 35
ESTIMATE_DIM = 5   # body velocity xyz [m/s], left/right wheel contact logits
MEAN_SQUASHES = ("tanh", "none")


def mlp(inputs: int, hidden: tuple[int, ...], outputs: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for width in hidden:
        layers += [nn.Linear(inputs, width), nn.ELU()]
        inputs = width
    layers.append(nn.Linear(inputs, outputs))
    return nn.Sequential(*layers)


class HistoryEstimator(nn.Module):
    def __init__(self, hidden: tuple[int, ...] = (128, 64)):
        super().__init__()
        self.network = mlp(HISTORY_LENGTH * FRAME_DIM, hidden, ESTIMATE_DIM)

    def forward(self, history: Tensor) -> Tensor:
        return self.network(history.flatten(-2))


class TeacherActor(nn.Module):
    """Deterministic mean ``forward``; PPO samples around it with a learned diagonal std."""

    def __init__(self, hidden: tuple[int, ...] = (512, 256, 128), estimator_hidden: tuple[int, ...] = (128, 64),
                 initial_std: float = 1., *, mean_squash: str = "tanh",
                 action_bounds: tuple[float, ...] = ACTION_BOUNDS,
                 chassis_command_contract: str = LEGACY_COMMANDS, actor_features: dict | None = None):
        super().__init__()
        if mean_squash not in MEAN_SQUASHES:
            raise ValueError(f"mean_squash must be one of {MEAN_SQUASHES}")
        self.mean_squash = mean_squash
        if chassis_command_contract not in COMMAND_CONTRACTS:
            raise ValueError("Unknown chassis command contract")
        self.chassis_command_contract = chassis_command_contract
        self.features = ActorFeatures(actor_features)
        self.estimator = HistoryEstimator(estimator_hidden)
        self.network = mlp(HISTORY_LENGTH * FRAME_DIM + ESTIMATE_DIM, hidden, 6)
        nn.init.uniform_(self.network[-1].weight, -1e-2, 1e-2)
        nn.init.zeros_(self.network[-1].bias)
        self.log_std = nn.Parameter(torch.full((6,), float(torch.log(torch.tensor(initial_std)))))
        self.register_buffer("action_bounds", torch.tensor(action_bounds, dtype=torch.float32))

    def forward(self, history: Tensor, estimate: Tensor | None = None) -> Tensor:
        """Deterministic action mean; ``estimate`` reuses an already computed estimator output."""
        if history.shape[-2:] != (HISTORY_LENGTH, FRAME_DIM):
            raise ValueError("Teacher actor requires chronological [batch, 5, 35] deployed frames")
        if estimate is None:
            estimate = self.estimator(history)
        raw = self.network(torch.cat((self.features(history).flatten(-2), estimate.detach()), -1))
        return torch.tanh(raw) * self.action_bounds if self.mean_squash == "tanh" else raw

    def distribution(self, history: Tensor) -> torch.distributions.Normal:
        mean = self.forward(history)
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))

    def clip(self, action: Tensor) -> Tensor:
        return torch.maximum(torch.minimum(action, self.action_bounds), -self.action_bounds)


class RunningNorm(nn.Module):
    """Running input statistics for the critic; never touches the deployed actor."""

    def __init__(self, dim: int, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.eps = eps

    @torch.no_grad()
    def update(self, value: Tensor) -> None:
        value = value.reshape(-1, value.shape[-1])
        n = value.shape[0]
        mean, var = value.mean(0), value.var(0, unbiased=False)
        total = self.count + n
        delta = mean - self.mean
        self.mean += delta * n / total
        self.var.copy_((self.var * self.count + var * n + delta.square() * self.count * n / total) / total)
        self.count.copy_(total)

    def forward(self, value: Tensor) -> Tensor:
        return (value - self.mean) / torch.sqrt(self.var + self.eps ** 2)


class Critic(nn.Module):
    def __init__(self, input_dim: int, hidden: tuple[int, ...] = (512, 256, 128)):
        super().__init__()
        self.norm = RunningNorm(input_dim)
        self.network = mlp(input_dim, hidden, 1)

    def forward(self, observation: Tensor) -> Tensor:
        return self.network(self.norm(observation)).squeeze(-1)
