"""Autoregressive model fitting and short-horizon analytic policy gradients."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .contracts import StateSchema, WindowBatch
from .models import Actor, Critic, HistoryEstimator, WorldModel


@dataclass(frozen=True)
class LearningConfig:
    model_hidden: tuple[int, ...] = (512, 512, 512)
    actor_hidden: tuple[int, ...] = (256, 128, 64)
    critic_hidden: tuple[int, ...] = (256, 128, 64)
    history_length: int = 5
    fit_horizon: int = 8
    imagination_horizon: int = 32
    learning_rate: float = 0.0002
    gamma: float = 0.99
    gae_lambda: float = 0.95
    critic_epochs: int = 5
    critic_minibatches: int = 4
    gradient_clip: float = 1.0
    actor_element_clip: float = 0.1
    advance_command_clock: bool = True
    observation_noise_std: tuple[float, ...] = (0.0,) * 35
    actor_architecture: str = "history_mlp"
    actor_policy_config: dict | None = None

    def __post_init__(self):
        for name in ("history_length", "fit_horizon", "imagination_horizon",
                     "critic_epochs", "critic_minibatches"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("model_hidden", "actor_hidden", "critic_hidden"):
            if not getattr(self, name) or any(type(n) is not int or n < 1 for n in getattr(self, name)):
                raise ValueError(f"Invalid {name}")
        if not (0 < self.gamma <= 1 and 0 <= self.gae_lambda <= 1):
            raise ValueError("Invalid discount/GAE lambda")
        for name in ("learning_rate", "gradient_clip", "actor_element_clip"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"Invalid {name}")
        if len(self.observation_noise_std) not in (35, 45) or any(
                not math.isfinite(x) or x < 0 for x in self.observation_noise_std):
            raise ValueError("Observation noise must contain 35 or 45 nonnegative standard deviations")
        if any(self.observation_noise_std[i] for i in (*range(4), *range(22, 35))):
            raise ValueError("Noise cannot alter commands or issued-action history")
        if len(self.observation_noise_std) == 45 and any(self.observation_noise_std[35:40]):
            raise ValueError("Noise cannot alter gimbal task commands")
        if type(self.advance_command_clock) is not bool:
            raise ValueError("advance_command_clock must be boolean")
        if self.actor_architecture not in ("history_mlp", "transferred_transformer"):
            raise ValueError("Unknown actor_architecture")
        if self.actor_architecture == "history_mlp":
            if self.actor_policy_config is not None:
                raise ValueError("history_mlp does not accept actor_policy_config")
        else:
            if not isinstance(self.actor_policy_config, dict):
                raise ValueError("transferred_transformer requires actor_policy_config")
            if (type(self.actor_policy_config.get("history_length")) is not int
                    or self.actor_policy_config["history_length"] != self.history_length):
                raise ValueError("Actor policy configuration history_length differs")
            # Configuration parsing precedes installation of the pinned optional
            # source by the native factory. Validate its full schema in the actor.
            try:
                policy = json.loads(json.dumps(self.actor_policy_config, allow_nan=False))
            except (TypeError, ValueError) as error:
                raise ValueError("actor_policy_config must contain finite JSON values") from error
            object.__setattr__(self, "actor_policy_config", policy)

    @classmethod
    def from_dict(cls, data: dict) -> LearningConfig:
        data = dict(data)
        for name in ("model_hidden", "actor_hidden", "critic_hidden", "observation_noise_std"):
            if name in data:
                data[name] = tuple(data[name])
        return cls(**data)


@contextmanager
def frozen_parameters(*modules: nn.Module):
    """Freeze weights without detaching state/action Jacobians."""
    parameters = [(parameter, parameter.requires_grad)
                  for module in modules for parameter in module.parameters()]
    try:
        for parameter, _ in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, required in parameters:
            parameter.requires_grad_(required)


def lambda_returns(rewards: Tensor, continuation: Tensor, values: Tensor,
                   gamma: float, gae_lambda: float, horizon_boundary: Tensor | None = None) -> Tensor:
    """Time-major imagined lambda returns; values includes the H+1 endpoint."""
    if values.shape != (rewards.shape[0] + 1, *rewards.shape[1:]):
        raise ValueError("Expected one more value endpoint than rewards")
    if continuation.shape != rewards.shape:
        raise ValueError("Continuation shape mismatch")
    if horizon_boundary is not None and (horizon_boundary.shape != rewards.shape
                                         or horizon_boundary.dtype != torch.bool):
        raise ValueError("Imagination boundaries must be a boolean reward-shaped mask")
    result = torch.empty_like(rewards)
    tail = values[-1]
    for index in reversed(range(rewards.shape[0])):
        future = (tail if horizon_boundary is None else
                  torch.where(horizon_boundary[index], values[index + 1], tail))
        tail = rewards[index] + gamma * continuation[index] * (
            (1 - gae_lambda) * values[index + 1] + gae_lambda * future)
        result[index] = tail
    return result


class Learner:
    def __init__(self, schema: StateSchema, config: LearningConfig = LearningConfig(), *, device="cpu"):
        self.schema, self.config = schema, config
        self.device = torch.device(device)
        noise = config.observation_noise_std
        if len(noise) != schema.observation_dim:
            if schema.observation_dim == 45 and len(noise) == 35 and not any(noise):
                noise = (0.,) * 45
            else:
                raise ValueError("Observation noise width differs from the schema")
        self._observation_noise_std = noise
        self.world = WorldModel(schema, config.model_hidden).to(self.device)
        if config.actor_architecture == "history_mlp":
            self.actor = Actor(config.history_length, config.actor_hidden,
                               schema.observation_dim).to(self.device)
            self._actor_source_metadata = None
        else:
            if schema.observation_dim != 35:
                raise ValueError("Transferred Transformer requires the original 35D interface")
            from .transferred_actor import TransferredActor
            self.actor = TransferredActor(config.actor_policy_config, config.history_length).to(self.device)
            self._actor_source_metadata = self.actor.source_metadata
        self.critic = Critic(schema, config.critic_hidden).to(self.device)
        self.world_optimizer = torch.optim.Adam(self.world.parameters(), lr=config.learning_rate)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate,
                                                betas=(0.9, 0.95))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        self.estimator = (HistoryEstimator(config.history_length, schema.observation_dim).to(self.device)
                          if schema.observation_dim == 45 else None)
        self.estimator_optimizer = (torch.optim.Adam(self.estimator.parameters(), lr=config.learning_rate)
                                    if self.estimator is not None else None)
        self.estimator_updates = 0
        self.normalizers_fitted = False
        self.world_updates = 0
        self.actor_updates = 0
        self.critic_updates = 0

    @property
    def actor_source_metadata(self) -> dict | None:
        """Preserve initial transfer evidence independently of later actor updates."""
        return deepcopy(self._actor_source_metadata)

    def load_initial_actor(self, path, expected_sha256: str) -> None:
        """Initialize only actor weights before policy optimization or collection."""
        if self.config.actor_architecture != "transferred_transformer":
            raise ValueError("Initial actor transfer requires transferred_transformer")
        if (self.actor_updates or self.actor_optimizer.state
                or self._actor_source_metadata["initial_checkpoint"] is not None):
            raise ValueError("Initial actor transfer cannot replace an initialized or optimized actor")
        self._actor_source_metadata = self.actor.load_initial_checkpoint(path, expected_sha256)

    @torch.no_grad()
    def fit_normalizers(self, windows: WindowBatch) -> None:
        """Freeze warmup statistics with a physical scale for native contacts.

        Sparse contact channels can be identically zero during warmup. Their
        scale is at least one robot weight: mean(sum(body_mass_kg)) * 9.81 N
        across sampled environments/endpoints. This preserves larger empirical
        standard deviations and leaves actual forces and other fields intact.
        The named native fields must declare compatible body shapes and SI units;
        missing or invalid masses are errors, never replaced by a guessed mass.
        """
        if self.world_updates or self.actor_updates:
            raise RuntimeError("Normalization is frozen after learning starts")
        windows = windows.to(self.device)
        contact_field = next((field for field in self.schema.state_fields
                              if field.name == "terrain_contact_force_world_n"), None)
        contact_scale = None
        if contact_field is not None:
            if (len(contact_field.shape) != 2 or contact_field.shape[-1] != 3
                    or contact_field.units != "N"):
                raise ValueError("terrain_contact_force_world_n must have shape (bodies, 3) in N")
            mass_field = next((field for field in self.schema.context_fields
                               if field.name == "body_mass_kg"), None)
            if (mass_field is None or mass_field.shape != (contact_field.shape[0],)
                    or mass_field.units != "kg"):
                raise ValueError("Native contact normalization requires body_mass_kg "
                                 "in kg with one mass per contact body")
            if windows.contexts.shape[-1] != self.schema.context_dim:
                raise ValueError("body_mass_kg context width does not match the schema")
            mass_offset = sum(field.size for field in self.schema.context_fields[
                :self.schema.context_fields.index(mass_field)])
            masses = windows.contexts[..., mass_offset:mass_offset + mass_field.size]
            if not masses.numel() or not torch.isfinite(masses).all() or (masses <= 0).any():
                raise ValueError("body_mass_kg must contain finite positive body masses")
            contact_scale = masses.sum(dim=-1).mean() * 9.81
            if not torch.isfinite(contact_scale) or contact_scale <= 0:
                raise ValueError("body_mass_kg must produce a finite positive robot weight")
        self.world.state_normalizer.fit(windows.states)
        if contact_scale is not None:
            contact_offset = sum(field.size for field in self.schema.state_fields[
                :self.schema.state_fields.index(contact_field)])
            self.world.state_normalizer.scale[
                contact_offset:contact_offset + contact_field.size].clamp_min_(contact_scale)
        self.world.context_normalizer.fit(windows.contexts)
        self.world.observation_normalizer.fit(windows.observations)
        self.world.reward_normalizer.fit(windows.rewards.unsqueeze(-1), minimum_scale=0.1)
        self.normalizers_fitted = True

    def _ready(self):
        if not self.normalizers_fitted:
            raise RuntimeError("Fit and freeze normalization using collection warmup first")

    def model_loss(self, windows: WindowBatch) -> tuple[Tensor, dict[str, Tensor]]:
        self._ready()
        state = windows.states[:, 0]
        state_losses, observation_losses, reward_losses, done_losses = [], [], [], []
        for step in range(windows.actions.shape[1]):
            valid = windows.valid[:, step]
            if not valid.any():
                raise ValueError("An autoregressive fit window must contain valid transitions")
            predicted = self.world.step(state, windows.actions[:, step], windows.goals[:, step],
                                        windows.contexts[:, step])
            state_losses.append(F.mse_loss(self.world.state_normalizer(predicted.state)[valid],
                self.world.state_normalizer(windows.states[:, step + 1])[valid]))
            observed = self.world.observe_clean(predicted.state, windows.goals[:, step + 1],
                                          windows.contexts[:, step + 1], windows.actions[:, step])
            observation_losses.append(F.mse_loss(self.world.observation_normalizer(observed)[valid],
                self.world.observation_normalizer(windows.observations[:, step + 1])[valid]))
            reward_losses.append(F.mse_loss(self.world.reward_normalizer(predicted.reward[:, None])[valid],
                self.world.reward_normalizer(windows.rewards[:, step, None])[valid]))
            # Time limits truncate a sample; they are not physical failure labels.
            done_losses.append(F.binary_cross_entropy_with_logits(predicted.termination_logits[valid],
                windows.terminated[:, step][valid].float()))
            state = predicted.state
        losses = {"state": torch.stack(state_losses).mean(),
                  "observation": torch.stack(observation_losses).mean(),
                  "reward": torch.stack(reward_losses).mean(),
                  "termination": torch.stack(done_losses).mean()}
        return sum(losses.values()), losses

    def fit_world(self, windows: WindowBatch) -> dict[str, float]:
        self.world_optimizer.zero_grad(set_to_none=True)
        loss, components = self.model_loss(windows.to(self.device))
        self._backward_step(loss, self.world, self.world_optimizer)
        self.world_updates += 1
        metrics = {"model/loss": float(loss.detach()),
                   **{f"model/{name}": float(value.detach()) for name, value in components.items()}}
        if self.estimator is not None:
            metrics.update(self.fit_estimator(windows))
        return metrics

    def fit_estimator(self, windows: WindowBatch) -> dict[str, float]:
        """Fit deployment histories against aligned real endpoint labels only."""
        if self.estimator is None:
            return {}
        windows = windows.to(self.device)
        frames, targets, valid = (windows.actor_observations, windows.estimator_targets,
                                  windows.estimator_valid)
        if frames is None:
            raise ValueError("Estimator training requires real deployment observations")
        empty = {"estimator/loss": 0., "estimator/velocity_nll": 0.,
                 "estimator/contact_bce": 0., "estimator/valid_labels": 0.}
        if targets is None or valid is None:
            return empty
        histories, labels = [], []
        length = self.config.history_length
        for endpoint in range(length - 1, frames.shape[1]):
            start = endpoint - length + 1
            accepted = valid[:, endpoint].clone()
            if length > 1:
                accepted &= windows.valid[:, start:endpoint].all(-1)
                # A terminal endpoint itself is a legitimate label, but its
                # history may never include an earlier completed episode.
                if endpoint - start > 1:
                    accepted &= ~(windows.terminated[:, start:endpoint - 1]
                                  | windows.truncated[:, start:endpoint - 1]).any(-1)
            if accepted.any():
                histories.append(frames[accepted, start:endpoint + 1])
                labels.append(targets[accepted, endpoint])
        if not histories:
            return empty
        history, target = torch.cat(histories).detach(), torch.cat(labels).detach()
        raw = self.estimator(history)
        velocity = (.5 * ((raw[:, :3] - target[:, :3]) * (-raw[:, 3:6]).exp()).square()
                    + raw[:, 3:6]).mean()
        contact = F.binary_cross_entropy_with_logits(raw[:, 6:], target[:, 3:])
        loss = velocity + contact
        self.estimator_optimizer.zero_grad(set_to_none=True)
        self._backward_step(loss, self.estimator, self.estimator_optimizer)
        self.estimator_updates += 1
        return {"estimator/loss": float(loss.detach()),
                "estimator/velocity_nll": float(velocity.detach()),
                "estimator/contact_bce": float(contact.detach()),
                "estimator/valid_labels": float(target.shape[0])}

    @torch.no_grad()
    def validate_world(self, windows: WindowBatch) -> dict[str, float]:
        windows = windows.to(self.device)
        _, components = self.model_loss(windows)
        metrics = {f"validation/{name}": float(value) for name, value in components.items()}
        state = windows.states[:, 0]
        for step in range(windows.actions.shape[1]):
            state = self.world.step(state, windows.actions[:, step], windows.goals[:, step],
                                    windows.contexts[:, step]).state
            if step + 1 in {1, 8, 32, windows.actions.shape[1]}:
                offset = 0
                for field in self.schema.state_fields:
                    delta = (state[:, offset:offset + field.size]
                             - windows.states[:, step + 1, offset:offset + field.size])
                    metrics[f"validation/{step + 1}_step/{field.name}_rmse"] = float(delta.square().mean().sqrt())
                    offset += field.size
        return metrics

    def _value(self, state: Tensor, goal: Tensor, context: Tensor) -> Tensor:
        return self.critic(self.world.state_normalizer(state), goal,
                           self.world.context_normalizer(context))

    def imagine(self, starts: WindowBatch) -> tuple[Tensor, dict[str, Tensor]]:
        """Flat commands stay fixed; their encoded clock advances analytically."""
        self._ready()
        starts = starts.to(self.device)
        used = max(1, self.config.history_length - 1)
        if not starts.policy_owned[:, :used].all() or not starts.valid[:, :used].all():
            raise ValueError("Policy imagination must start in a valid policy-owned interval")
        if self.config.history_length > 1 and (starts.terminated[:, :used] | starts.truncated[:, :used]).any():
            raise ValueError("Actor history cannot cross an episode boundary")
        state, goal, context = starts.states[:, 0].detach(), starts.goals[:, 0].detach(), starts.contexts[:, 0].detach()
        frames = starts.actor_observations
        if frames is None:
            if self.schema.observation_dim == 45:
                raise ValueError("Formal imagination requires actual deployment history")
            frames = starts.observations
        if frames.shape[1] < self.config.history_length:
            raise ValueError("Imagination starts require real chronological actor history")
        # The caller supplies a history window and imagination begins at its last
        # state, not at its first state with artificially repeated observations.
        endpoint = self.config.history_length - 1
        state = starts.states[:, endpoint].detach()
        goal, context = starts.goals[:, endpoint].detach(), starts.contexts[:, endpoint].detach()
        history = frames[:, :self.config.history_length].detach()
        noise_std = state.new_tensor(self._observation_noise_std)
        if starts.actor_observations is None:
            history = history + torch.randn_like(history) * noise_std
        survival = torch.ones(state.shape[0], device=self.device)
        objective = torch.zeros_like(survival)
        initial_goal = goal
        initial_context = context
        states, rewards, continues, values, weights, goals, contexts = [], [], [], [], [], [], []
        boundaries = []
        lengths = self.world.imagination_horizon(state, goal, context, self.config.imagination_horizon).detach()
        horizon = max(1, int(lengths.max()))
        for step in range(horizon):
            active = step < lengths
            states.append(state)
            goals.append(goal)
            contexts.append(context)
            values.append(self._value(state, goal, context))
            weights.append(survival * active)
            action = self.actor(history)
            predicted = self.world.step(state, action, goal, context)
            reward = torch.where(active, predicted.reward, 0.)
            continuation = torch.where(active, predicted.continuation, 1.)
            objective = objective + self.config.gamma ** step * survival * reward
            rewards.append(reward)
            continues.append(continuation)
            boundaries.append(step + 1 == lengths)
            survival = survival * continuation
            next_state, next_goal, next_context = self.world.advance_conditions(
                predicted.state, goal, context, action,
                advance_clock=self.config.advance_command_clock)
            state = torch.where(active[:, None], next_state, state)
            goal = torch.where(active[:, None], next_goal, goal)
            context = torch.where(active[:, None], next_context, context)
            observation = self.world.observe(state, goal, context, previous_action=action)
            observation = observation + torch.randn_like(observation) * noise_std
            history = torch.cat((history[:, 1:], observation[:, None]), dim=1)
        values.append(self._value(state, goal, context))
        objective = objective + self.config.gamma ** lengths * survival * values[-1]
        data = {"states": torch.stack(states), "goal": initial_goal, "goals": torch.stack(goals),
                "context": initial_context, "contexts": torch.stack(contexts),
                "rewards": torch.stack(rewards), "continuation": torch.stack(continues),
                "values": torch.stack(values), "weights": torch.stack(weights),
                "horizon_boundary": torch.stack(boundaries), "horizon_lengths": lengths,
                "final_survival": survival}
        return -objective.mean(), data

    def improve_policy(self, starts: WindowBatch) -> dict[str, float]:
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.world_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        with frozen_parameters(self.world, self.critic):
            loss, imagined = self.imagine(starts)
            if not imagined["weights"].any():
                return {"actor/loss": float(loss.detach()), "critic/loss": 0.,
                        "imagination/final_survival": float(imagined["final_survival"].mean()),
                        "actor/update_skipped": 1.}
            self._backward_step(loss, self.actor, self.actor_optimizer, element_clip=True)
        self.actor_updates += 1
        imagined = {name: value.detach() for name, value in imagined.items()}
        targets = lambda_returns(imagined["rewards"], imagined["continuation"], imagined["values"],
                                  self.config.gamma, self.config.gae_lambda,
                                  imagined["horizon_boundary"]).flatten()
        states = imagined["states"].flatten(0, 1)
        horizon = imagined["states"].shape[0]
        goals = imagined["goals"].flatten(0, 1)
        contexts = imagined["contexts"].flatten(0, 1)
        weights = imagined["weights"].flatten()
        active = weights > 0
        states, goals, contexts, targets, weights = (value[active] for value in
                                                    (states, goals, contexts, targets, weights))
        critic_losses = []
        for _ in range(self.config.critic_epochs):
            indices = torch.randperm(states.shape[0], device=self.device)
            for batch in indices.tensor_split(min(self.config.critic_minibatches, len(indices))):
                prediction = self._value(states[batch], goals[batch], contexts[batch])
                critic_loss = ((prediction - targets[batch]).square() * weights[batch]).sum() / weights[batch].sum().clamp_min(1e-6)
                self.critic_optimizer.zero_grad(set_to_none=True)
                self._backward_step(critic_loss, self.critic, self.critic_optimizer)
                critic_losses.append(float(critic_loss.detach()))
                self.critic_updates += 1
        return {"actor/loss": float(loss.detach()), "critic/loss": sum(critic_losses) / len(critic_losses),
                "imagination/final_survival": float(imagined["final_survival"].mean())}

    def _backward_step(self, loss, module, optimizer, *, element_clip=False):
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss; optimizer was not stepped")
        loss.backward()
        nn.utils.clip_grad_norm_(module.parameters(), self.config.gradient_clip, error_if_nonfinite=True)
        if element_clip:
            for parameter in module.parameters():
                if parameter.grad is not None:
                    parameter.grad.clamp_(-self.config.actor_element_clip, self.config.actor_element_clip)
        optimizer.step()

    def state_dict(self) -> dict:
        result = {"schema": self.schema.to_dict(), "schema_hash": self.schema.fingerprint,
                "config": asdict(self.config), "world": self.world.state_dict(),
                "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "actor_source_metadata": self.actor_source_metadata,
                "world_optimizer": self.world_optimizer.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "normalizers_fitted": self.normalizers_fitted,
                "counts": [self.world_updates, self.actor_updates, self.critic_updates]}
        if self.estimator is not None:
            result.update(estimator=self.estimator.state_dict(),
                          estimator_optimizer=self.estimator_optimizer.state_dict(),
                          estimator_updates=self.estimator_updates)
        return result

    def load_state_dict(self, state: dict) -> None:
        # v1 MLP checkpoints predate the optional transfer configuration fields.
        if (state["schema_hash"] != self.schema.fingerprint
                or LearningConfig.from_dict(state["config"]) != self.config):
            raise ValueError("Checkpoint schema or learning configuration mismatch")
        if StateSchema.from_dict(state["schema"]).fingerprint != self.schema.fingerprint:
            raise ValueError("Checkpoint schema contents do not match their hash")
        source = state.get("actor_source_metadata")
        current = self._actor_source_metadata
        if current is None:
            if source is not None:
                raise ValueError("MLP checkpoint cannot contain transferred actor source metadata")
        else:
            if not isinstance(source, dict) or source.keys() != current.keys():
                raise ValueError("Transferred actor checkpoint requires source metadata")
            # File locations may differ across hosts; executable source hashes
            # and actor semantics must remain identical. No teacher file is read.
            for name in current.keys() - {"source_files", "initial_checkpoint"}:
                if json.dumps(source[name], sort_keys=True) != json.dumps(current[name], sort_keys=True):
                    raise ValueError("Transferred actor source metadata differs: " + name)
            files = source["source_files"]
            if (not isinstance(files, dict) or files.keys() != current["source_files"].keys()
                    or any(not isinstance(files[name], dict)
                           or files[name].get("sha256") != item["sha256"]
                           for name, item in current["source_files"].items())):
                raise ValueError("Transferred actor source file hashes differ")
            if source["initial_checkpoint"] is not None and not isinstance(source["initial_checkpoint"], dict):
                raise ValueError("Transferred actor initial checkpoint metadata is invalid")
        for name in ("world", "actor", "critic", "world_optimizer", "actor_optimizer", "critic_optimizer"):
            getattr(self, name).load_state_dict(state[name])
        if self.estimator is not None:
            self.estimator.load_state_dict(state["estimator"])
            self.estimator_optimizer.load_state_dict(state["estimator_optimizer"])
            self.estimator_updates = state["estimator_updates"]
        self._actor_source_metadata = deepcopy(source)
        self.normalizers_fitted = bool(state["normalizers_fitted"])
        self.world_updates, self.actor_updates, self.critic_updates = state["counts"]
