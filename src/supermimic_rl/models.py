"""Differentiable surrogate dynamics, bounded actor and privileged critic."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .contracts import ACTION_BOUNDS, StateSchema


def multilayer(input_dim: int, hidden: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for width in hidden:
        layers.extend((nn.Linear(input_dim, width), nn.ELU()))
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class FixedNormalizer(nn.Module):
    """Fit once on the collection warmup; never move the target during BPTT."""

    def __init__(self, size: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("scale", torch.ones(size))

    @torch.no_grad()
    def fit(self, values: Tensor, *, minimum_scale: float = 0.05) -> None:
        if values.shape[-1] == 0:
            return
        flat = values.reshape(-1, values.shape[-1])
        if not torch.isfinite(flat).all():
            raise ValueError("Cannot normalize nonfinite data")
        if flat.shape[-1]:
            self.mean.copy_(flat.mean(0))
            self.scale.copy_(flat.std(0, correction=0).clamp_min(minimum_scale))

    def forward(self, value: Tensor) -> Tensor:
        return (value - self.mean) / self.scale

    def decode(self, value: Tensor) -> Tensor:
        return value * self.scale + self.mean


@dataclass
class Prediction:
    state: Tensor
    reward: Tensor
    termination_logits: Tensor

    @property
    def continuation(self) -> Tensor:
        return torch.sigmoid(-self.termination_logits)


class WorldModel(nn.Module):
    """Residual full-state surrogate with learned observation/reward/failure heads.

    The simulator's 200 Hz accumulated reward is supervised directly. A 50 Hz
    endpoint reward formula is not assumed equivalent to that source reward.
    Command slots are analytic so imagination cannot change the user's goal.
    """

    def __init__(self, schema: StateSchema, hidden: tuple[int, ...] = (512, 512, 512)):
        super().__init__()
        self.schema = schema
        self.state_normalizer = FixedNormalizer(schema.state_dim)
        self.context_normalizer = FixedNormalizer(schema.context_dim)
        self.observation_normalizer = FixedNormalizer(schema.observation_dim)
        self.reward_normalizer = FixedNormalizer(1)
        self.register_buffer("action_bounds", torch.tensor(ACTION_BOUNDS))
        self._orientation_slice = None
        self._gimbal_phase_slice = None
        offset = 0
        for field in schema.state_fields:
            if field.name == "root_orientation_xyzw":
                if field.size != 4:
                    raise ValueError("compact root orientation must have four xyzw components")
                self._orientation_slice = slice(offset, offset + field.size)
            if field.name == "gimbal.yaw_sin_cos":
                if field.size != 2:
                    raise ValueError("gimbal yaw phase must contain sin and cos")
                self._gimbal_phase_slice = slice(offset, offset + field.size)
            offset += field.size
        self.dynamics = multilayer(schema.state_dim + schema.context_dim + 6,
                                   hidden, schema.state_dim)
        self.observation_head = (multilayer(schema.state_dim + schema.context_dim, hidden[:2], schema.observation_dim)
                                 if self._orientation_slice is None and schema.observation_dim == 35 else None)
        self.outcome_head = multilayer(2 * schema.state_dim + schema.context_dim + 6 + schema.goal_dim,
                                      hidden[:2], 2)
        # An initially small residual avoids unbounded open-loop initialization.
        nn.init.zeros_(self.dynamics[-1].weight)
        nn.init.zeros_(self.dynamics[-1].bias)

    def observe(self, state: Tensor, goal: Tensor, context: Tensor,
                previous_action: Tensor | None = None) -> Tensor:
        if self.schema.observation_dim == 45:
            from .formal_observation import observe
            return observe(state, goal, context, previous_action, self.schema)
        if self._orientation_slice is not None:
            from .compact_state import analytic_actor_observation
            return analytic_actor_observation(state, context, goal, previous_action, self.schema)
        features = torch.cat((self.state_normalizer(state),
                              self.context_normalizer(context)), dim=-1)
        observed = self.observation_normalizer.decode(self.observation_head(features))
        previous = observed[..., 22:28] if previous_action is None else previous_action
        # Wheel position is deliberately absent from the deployment interface.
        frame = torch.cat((goal[..., :4], observed[..., 4:14],
                           torch.zeros_like(observed[..., 14:16]), observed[..., 16:22],
                           previous, goal[..., 4:11]), dim=-1)
        # Match the native ObservationAssembler.actor output contract after
        # decoding and assembling both learned and analytic observation slots.
        return frame.clamp(-100, 100)

    def observe_clean(self, state: Tensor, goal: Tensor, context: Tensor,
                      previous_action: Tensor | None = None) -> Tensor:
        """Clean supervision and deployed observations have separate contracts."""
        if self.schema.observation_dim == 45:
            from .formal_observation import observe_clean
            return observe_clean(state, goal, context, previous_action, self.schema)
        return self.observe(state, goal, context, previous_action)

    def advance_conditions(self, state: Tensor, goal: Tensor, context: Tensor,
                           previous_action: Tensor, *, advance_clock: bool = True):
        if self.schema.observation_dim == 45:
            from .formal_observation import advance_conditions
            return advance_conditions(state, goal, context, previous_action, self.schema)
        if advance_clock:
            goal = torch.cat((goal[..., :10],
                              (goal[..., 10:11] + self.schema.policy_dt_s).clamp_max(5),
                              goal[..., 11:]), -1)
        return state, goal, context

    def imagination_horizon(self, state: Tensor, goal: Tensor, context: Tensor, maximum: int):
        if self.schema.observation_dim == 45:
            from .formal_observation import imagination_horizon
            return imagination_horizon(state, goal, context, self.schema).clamp(0, maximum)
        return torch.full(state.shape[:-1], maximum, dtype=torch.long, device=state.device)

    def step(self, state: Tensor, action: Tensor, goal: Tensor, context: Tensor) -> Prediction:
        normalized = self.state_normalizer(state)
        conditions = self.context_normalizer(context)
        command = action / self.action_bounds
        next_normalized = normalized + self.dynamics(torch.cat(
            (normalized, conditions, command), dim=-1))
        next_state = self.state_normalizer.decode(next_normalized)
        for index in (self._orientation_slice, self._gimbal_phase_slice):
            if index is None:
                continue
            direction = next_state[..., index]
            direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            next_state = torch.cat((next_state[..., :index.start], direction,
                                    next_state[..., index.stop:]), dim=-1)
        if self._orientation_slice is not None or self._gimbal_phase_slice is not None:
            next_normalized = self.state_normalizer(next_state)
        outcomes = self.outcome_head(torch.cat(
            (normalized, next_normalized, conditions, command, goal), dim=-1))
        return Prediction(next_state,
                          self.reward_normalizer.decode(outcomes[..., :1]).squeeze(-1),
                          outcomes[..., 1])


class Actor(nn.Module):
    """Chronological history MLP with six issued targets."""

    def __init__(self, history_length: int = 5, hidden: tuple[int, ...] = (256, 128, 64),
                 observation_dim: int = 35):
        super().__init__()
        if type(history_length) is not int or history_length < 1:
            raise ValueError("history_length must be positive")
        self.history_length = history_length
        self.observation_dim = observation_dim
        self.network = multilayer(observation_dim * history_length, hidden, 6)
        self.register_buffer("action_bounds", torch.tensor(ACTION_BOUNDS))
        nn.init.uniform_(self.network[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, history: Tensor) -> Tensor:
        if history.shape[-2:] != (self.history_length, self.observation_dim):
            raise ValueError("Actor requires chronological [batch, history, observation_dim]")
        return torch.tanh(self.network(history.flatten(-2))) * self.action_bounds


class HistoryEstimator(nn.Module):
    """Deployment-only inputs; real velocity/contact labels train this separately."""

    def __init__(self, history_length: int = 5, observation_dim: int = 45):
        super().__init__()
        self.history_length, self.observation_dim = history_length, observation_dim
        self.network = nn.Sequential(
            nn.Linear(history_length * observation_dim, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 32), nn.GELU(),
            nn.Linear(32, 8))

    def forward(self, history: Tensor) -> Tensor:
        if history.shape[-2:] != (self.history_length, self.observation_dim):
            raise ValueError("Estimator requires a chronological deployment history")
        raw = self.network(history.flatten(-2))
        return torch.cat((raw[..., :3], raw[..., 3:6].clamp(-4, 2), raw[..., 6:]), -1)


class Critic(nn.Module):
    """A value estimate conditioned on true/model state, task and sim context."""

    def __init__(self, schema: StateSchema, hidden: tuple[int, ...] = (256, 128, 64)):
        super().__init__()
        self.network = multilayer(schema.state_dim + schema.context_dim + schema.goal_dim, hidden, 1)

    def forward(self, normalized_state: Tensor, goal: Tensor, normalized_context: Tensor) -> Tensor:
        return self.network(torch.cat((normalized_state, goal, normalized_context), -1)).squeeze(-1)
