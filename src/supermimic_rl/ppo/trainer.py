"""PPO with the upstream course recipe, plus supervised training of the history estimator.

Recipe: 24 steps per update, 5 epochs x 4 minibatches, clipped surrogate and
clipped value loss (value coefficient 4), entropy 0.005, Adam with an adaptive
learning rate driven by the mean KL (target 0.01), global gradient clip 1.0.
The estimator is a separate supervised problem on real labels; its output is
detached inside the actor, so the PPO optimizer never updates it.

A run may start from a transferred actor and imitate a frozen teacher
(kickstarting on the learner's own states): during ``imitation_updates`` the
actor follows only the teacher while the critic warms up; the imitation weight
then decays linearly to zero over ``imitation_anneal_updates`` under PPO.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import time

import torch
from torch import nn
from torch.nn import functional as F

from ..contracts import ACTION_BOUNDS
from .course_gate import CourseGates, GateTimeout, StageGate
from .policy import MEAN_SQUASHES, Critic, TeacherActor
from .rollout import Rollout, gae

CHECKPOINT_FORMAT = "supermimic-ppo-teacher-v1"
ACTOR_INIT_FORMAT = "supermimic-ppo-actor-init-v1"


@dataclass(frozen=True)
class PPOConfig:
    steps_per_update: int = 24
    updates: int = 10000
    epochs: int = 5
    minibatches: int = 4
    gamma: float = .99
    lam: float = .95
    clip: float = .2
    value_clip: float = .2
    value_coefficient: float = 4.
    entropy_coefficient: float = .005
    learning_rate: float = 1e-4
    desired_kl: float = .01
    learning_rate_bounds: tuple[float, float] = (1e-5, 1e-2)
    max_grad_norm: float = 1.
    estimator_learning_rate: float = 1e-3
    initial_std: float = 1.
    actor_hidden: tuple[int, ...] = (512, 256, 128)
    estimator_hidden: tuple[int, ...] = (128, 64)
    critic_hidden: tuple[int, ...] = (512, 256, 128)
    checkpoint_interval: int = 100
    seed: int = 7
    curriculum_transition_offset: int = 0
    curriculum_transition_cap: int | None = None
    mean_squash: str = "tanh"
    # Fresh runs may start from an actor-initialization artifact; its SHA256 is required.
    initial_actor: str | None = None
    initial_actor_sha256: str | None = None
    # Critic-only updates while a transferred actor's critic is still untrained.
    actor_freeze_updates: int = 0
    # Must equal the native runtime task.action_bounds (legs then wheels).
    action_bounds: tuple[float, ...] = ACTION_BOUNDS
    # Frozen teacher (actor-initialization artifact) imitated on the learner's states.
    imitation_actor: str | None = None
    imitation_actor_sha256: str | None = None
    imitation_updates: int = 0
    imitation_anneal_updates: int = 0
    imitation_coefficient: float = 1.
    # Stage gates of the native course (see course_gate.StageGate); empty keeps exposure-only stages.
    course_gates: tuple[dict, ...] = ()
    # "restart_hold" restarts the waiting gate's timeout on resume, e.g. after a
    # course change; coordinate and opened gates are kept. "continue" keeps it.
    course_gate_resume: str = "continue"
    # Hand control to the native evaluation orchestrator at each newly opened gate.
    stop_after_gate: bool = False
    # Repeat a stage after a fix: on resume from a checkpoint written without
    # this rewind, the coordinate returns to the named stage's first transition.
    course_rewind_stage: str | None = None
    # A new tag repeats the rewind once even if the checkpoint already rewound to the same stage.
    course_rewind_tag: str | None = None
    # Exploration on entering a new skill stage: {stage: six action-std floors},
    # applied once (log_std = max(log_std, log(floor))) when the course enters the stage.
    stage_action_std: dict = field(default_factory=dict)

    def __post_init__(self):
        if (type(self.curriculum_transition_offset) is not int
                or self.curriculum_transition_offset < 0):
            raise ValueError("curriculum_transition_offset must be a nonnegative integer")
        if (self.curriculum_transition_cap is not None
                and (type(self.curriculum_transition_cap) is not int or self.curriculum_transition_cap < 0)):
            raise ValueError("curriculum_transition_cap must be a nonnegative integer or null")
        if self.mean_squash not in MEAN_SQUASHES:
            raise ValueError(f"mean_squash must be one of {MEAN_SQUASHES}")
        if (self.initial_actor is None) != (self.initial_actor_sha256 is None):
            raise ValueError("initial_actor and initial_actor_sha256 must be given together")
        if type(self.actor_freeze_updates) is not int or self.actor_freeze_updates < 0:
            raise ValueError("actor_freeze_updates must be a nonnegative integer")
        if (len(self.action_bounds) != 6
                or not all(math.isfinite(value) and value > 0 for value in self.action_bounds)):
            raise ValueError("action_bounds must be six finite positive values")
        if (self.imitation_actor is None) != (self.imitation_actor_sha256 is None):
            raise ValueError("imitation_actor and imitation_actor_sha256 must be given together")
        for name in ("imitation_updates", "imitation_anneal_updates"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(name + " must be a nonnegative integer")
        if self.imitation_actor is None and (self.imitation_updates or self.imitation_anneal_updates):
            raise ValueError("imitation schedule requires an imitation_actor")
        if not math.isfinite(self.imitation_coefficient) or self.imitation_coefficient < 0:
            raise ValueError("imitation_coefficient must be finite and nonnegative")
        object.__setattr__(self, "course_gates", tuple(dict(gate) for gate in self.course_gates))
        for gate in self.course_gates:
            StageGate(**gate)
        if self.course_gate_resume not in ("continue", "restart_hold"):
            raise ValueError("course_gate_resume must be continue or restart_hold")
        if type(self.stop_after_gate) is not bool or self.stop_after_gate and not self.course_gates:
            raise ValueError("stop_after_gate requires a boolean and a gated course")
        if self.course_rewind_stage is not None and not self.course_gates:
            raise ValueError("course_rewind_stage requires a gated course")
        if self.stage_action_std and not self.course_gates:
            raise ValueError("stage_action_std requires a gated course")
        for stage, floors in self.stage_action_std.items():
            if (type(stage) is not str or len(floors) != 6
                    or not all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in floors)):
                raise ValueError("stage_action_std maps stage names to six positive action stds")
        object.__setattr__(self, "stage_action_std", {k: tuple(float(v) for v in f)
                                                      for k, f in self.stage_action_std.items()})
        if self.course_gates and self.curriculum_transition_offset:
            raise ValueError("A gated course starts at coordinate zero; offsets apply to exposure-only courses")

    @classmethod
    def from_dict(cls, value: dict) -> PPOConfig:
        value = dict(value)
        for name in ("learning_rate_bounds", "actor_hidden", "estimator_hidden", "critic_hidden", "action_bounds"):
            if name in value:
                value[name] = tuple(value[name])
        # Checkpoints written before the bounded mean existed used a raw mean.
        value.setdefault("mean_squash", "none")
        return cls(**value)


def atomic_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class EpisodeStatistics:
    """Completed episodes and valid endpoint errors, without hiding clipped tails."""

    # Radiant 0bf3456 Cfg/mdp.py clips these errors before its reported MAE.
    # Our errors and sampling protocol remain native; these are shared bounds,
    # not a claim to reproduce its first-episode, trajectory-weighted evaluation.
    CLIPPED_MAE_LIMITS = {'forward_error_m_s': 1., 'yaw_rate_error_rad_s': .8, 'height_error_m': .15}

    def __init__(self, num_envs: int, device):
        self.returns = torch.zeros(num_envs, device=device)
        self.lengths = torch.zeros(num_envs, device=device)
        self.reset_window()

    def reset_window(self):
        self.finished_returns, self.finished_lengths = [], []
        self.squares, self.absolute, self.clipped_absolute = {}, {}, {}
        self.count, self.observed, self.terminated, self.truncated = 0, 0, 0, 0
        self.invalid, self.numerical_invalid = 0, 0
        self.nonfinite = 0
        self.diagnostic_names, self.diagnostics = None, None

    def record(self, step):
        valid = torch.ones_like(step.terminated) if step.valid is None else step.valid
        finite = torch.isfinite(step.reward)
        for value in step.tracking.values():
            finite = finite & torch.isfinite(value)
        numerical = (torch.zeros_like(valid) if step.numerical_invalid is None else step.numerical_invalid)
        valid = valid & ~numerical & finite
        self.returns += torch.where(valid, step.reward, 0.)
        self.lengths += 1
        done = step.terminated | step.truncated | ~valid
        completed = done & valid
        if completed.any():
            self.finished_returns.append(self.returns[completed].clone())
            self.finished_lengths.append(self.lengths[completed].clone())
        self.returns[done] = 0
        self.lengths[done] = 0
        self.terminated += int((step.terminated & valid).sum())
        self.truncated += int((step.truncated & valid).sum())
        self.invalid += int((~valid).sum())
        self.numerical_invalid += int(numerical.sum())
        self.nonfinite += int((~finite).sum())
        for name, value in step.tracking.items():
            error = value[valid].double()
            self.squares[name] = self.squares.get(name, 0.) + float(error.square().sum())
            self.absolute[name] = self.absolute.get(name, 0.) + float(error.abs().sum())
            if name in self.CLIPPED_MAE_LIMITS:
                clipped = error.abs().clamp_max(self.CLIPPED_MAE_LIMITS[name])
                self.clipped_absolute[name] = self.clipped_absolute.get(name, 0.) + float(clipped.sum())
        self.count += int(valid.sum())
        self.observed += valid.numel()
        diagnostics = getattr(step, "diagnostics", None)
        if diagnostics:
            names = tuple(diagnostics)
            values = torch.stack(tuple(diagnostics.values()), -1).double()
            measured = valid[:, None] & torch.isfinite(values)
            aggregate = torch.stack((
                torch.where(measured, values, 0.).sum(0), measured.sum(0),
                torch.where(measured, values, float("inf")).amin(0),
                torch.where(measured, values, -float("inf")).amax(0)))
            if self.diagnostics is None:
                self.diagnostic_names, self.diagnostics = names, aggregate
            else:
                if names != self.diagnostic_names:
                    raise ValueError("Tracking diagnostic fields changed inside one statistics window")
                self.diagnostics[:2] += aggregate[:2]
                self.diagnostics[2] = torch.minimum(self.diagnostics[2], aggregate[2])
                self.diagnostics[3] = torch.maximum(self.diagnostics[3], aggregate[3])

    def summary(self) -> dict:
        result = {"terminations": self.terminated, "truncations": self.truncated,
                  "invalid_transitions": self.invalid, "numerical_invalid_transitions": self.numerical_invalid,
                  "nonfinite_measurement_transitions": self.nonfinite, "observed_transitions": self.observed,
                  "valid_tracking_samples": self.count,
                  **{f"rms/{name}": math.sqrt(total / self.count) if self.count else None
                     for name, total in self.squares.items()},
                  **{f"mae/{name}": total / self.count if self.count else None
                     for name, total in self.absolute.items()},
                  **{f"clipped_mae/{name}": total / self.count if self.count else None
                     for name, total in self.clipped_absolute.items()},
                  "clipped_mae_limits": self.CLIPPED_MAE_LIMITS.copy(),
                  "tracking_aggregation": "sample-weighted valid endpoints, including terminal samples and subsequent reset episodes",
                  "clipped_mae_reference": "Radiant clip bounds only; native coordinates and sampling differ from its trajectory-weighted first-episode evaluation"}
        if self.finished_returns:
            returns, lengths = torch.cat(self.finished_returns), torch.cat(self.finished_lengths)
            result.update({"episode/return_mean": float(returns.mean()), "episode/length_mean": float(lengths.mean()),
                           "episode/count": int(returns.numel())})
        if self.diagnostics is not None:
            result["tracking_diagnostics"] = {
                name: {"samples": int(count), "mean": total / count if count else None,
                       "min": low if count else None, "max": high if count else None}
                for name, (total, count, low, high) in zip(
                    self.diagnostic_names, self.diagnostics.T.cpu().tolist())}
            result["tracking_diagnostic_timing"] = (
                "physical endpoint before reset versus command held during that interval; "
                "velocity_height_gate is the native four-substep mean")
            if any(name.startswith("reward_per_s/") for name in self.diagnostic_names):
                result["tracking_diagnostic_reward_units"] = {
                    "reward_per_s": "native signed weighted density, four-substep integral / policy_dt",
                    "reward_event": "once-per-interval reward amount, not multiplied or divided by dt",
                    "reward_gate": "native physical-substep mean gate occupancy or multiplier",
                    "reward_step": "native completed-interval reward amount"}
        return result


class PPOTrainer:
    def __init__(self, environment, config: PPOConfig, output: Path, *, resume: Path | None = None):
        self.environment, self.config, self.output = environment, config, Path(output)
        self.device = environment.device
        torch.manual_seed(config.seed)
        self.observation = environment.reset()
        critic_dim = self.observation.critic.shape[-1]
        native_bounds = getattr(environment, "action_bounds", None)
        if native_bounds is not None and not torch.allclose(torch.tensor(native_bounds),
                                                            torch.tensor(config.action_bounds)):
            raise ValueError(f"PPO action_bounds {config.action_bounds} differ from the native {native_bounds}")
        self.actor = TeacherActor(config.actor_hidden, config.estimator_hidden, config.initial_std,
                                  mean_squash=config.mean_squash, action_bounds=config.action_bounds).to(self.device)
        self.initialization = None
        if config.initial_actor is not None and resume is None:
            self.initialization = self._load_initial_actor(Path(config.initial_actor), config.initial_actor_sha256)
        self.imitation_teacher = None
        if config.imitation_actor is not None:
            self.imitation_teacher, _ = self._read_actor_artifact(Path(config.imitation_actor),
                                                                  config.imitation_actor_sha256)
            self.imitation_teacher.requires_grad_(False).eval()
        self.critic = Critic(critic_dim, config.critic_hidden).to(self.device)
        policy_parameters = [p for name, p in self.actor.named_parameters() if not name.startswith("estimator.")]
        self.optimizer = torch.optim.Adam(policy_parameters + list(self.critic.parameters()), lr=config.learning_rate)
        self.estimator_optimizer = torch.optim.Adam(self.actor.estimator.parameters(),
                                                    lr=config.estimator_learning_rate)
        self.learning_rate = config.learning_rate
        self.updates = self.transitions = 0
        self.stage_action_std_applied: list[str] = []
        self.course = (CourseGates(config.course_gates, environment.course_stage_starts(),
                                   config.curriculum_transition_cap) if config.course_gates else None)
        # Native skill counters are cumulative since the cold physical setup.
        self._course_counters = (environment.course_progress()[0]
                                 if hasattr(environment, "course_progress") else None)
        self._normalization_samples = None
        self.statistics = EpisodeStatistics(environment.num_envs, self.device)
        if resume is not None:
            self.load(resume)
            # The first reset above only discovers observation dimensions.
            # Resume starts fresh episodes at the restored/rewound course,
            # otherwise every gate audit would silently restart flat births.
            environment.set_progress(self.curriculum_transitions, self.updates)
            self.observation = environment.reset()
            self._course_counters = (environment.course_progress()[0]
                                     if hasattr(environment, "course_progress") else None)
        self.output.mkdir(parents=True, exist_ok=True)

    def _read_actor_artifact(self, path: Path, expected_sha256: str) -> tuple[TeacherActor, dict]:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected_sha256:
            raise ValueError(f"Actor artifact SHA256 differs from the configured value: {path}")
        payload = torch.load(io.BytesIO(raw), map_location=self.device, weights_only=True)
        if payload.get("format") != ACTOR_INIT_FORMAT:
            raise ValueError("Not an actor-initialization artifact: " + str(path))
        actor = artifact_actor(payload).to(self.device)
        provenance = {"path": str(path.resolve()), "sha256": digest, "source": payload["source"],
                      "equivalence": payload["equivalence"], "action_bounds": actor.action_bounds.tolist()}
        return actor, provenance

    def _load_initial_actor(self, path: Path, expected_sha256: str) -> dict:
        """Actor and estimator weights from a transfer artifact; std restarts at ``initial_std``."""
        source, provenance = self._read_actor_artifact(path, expected_sha256)
        config = self.config
        hidden = tuple(layer.out_features for layer in source.network if isinstance(layer, nn.Linear))[:-1]
        estimator = tuple(layer.out_features for layer in source.estimator.network
                          if isinstance(layer, nn.Linear))[:-1]
        if (hidden != config.actor_hidden or estimator != config.estimator_hidden
                or source.mean_squash != config.mean_squash
                or not torch.equal(source.action_bounds, self.actor.action_bounds)):
            raise ValueError("Initial actor artifact differs from the configured actor interface")
        state = source.state_dict()
        # Exploration noise is a PPO setting, not part of the transferred policy.
        state["log_std"] = self.actor.log_std.detach().clone()
        self.actor.load_state_dict(state, strict=True)
        return provenance | {"initial_std": config.initial_std}

    def _actor_frozen(self) -> bool:
        return self.updates < self.config.actor_freeze_updates

    def _imitation_weight(self) -> float:
        """Full weight while imitating only, then a linear decay to zero under PPO."""
        config = self.config
        if self.imitation_teacher is None:
            return 0.
        if self.updates < config.imitation_updates:
            return config.imitation_coefficient
        decayed = self.updates - config.imitation_updates
        if decayed >= config.imitation_anneal_updates:
            return 0.
        return config.imitation_coefficient * (1 - (decayed + 1) / (config.imitation_anneal_updates + 1))

    def _surrogate_active(self) -> bool:
        return not self._actor_frozen() and self.updates >= self.config.imitation_updates

    # -- collection -------------------------------------------------------------------
    @torch.no_grad()
    def collect(self) -> Rollout:
        config, env = self.config, self.environment
        # GAE and clipped values must share one normalization throughout a rollout.
        # Use complete preceding rollouts, not reset-only samples with zero variance.
        if self._normalization_samples is not None and self._normalization_samples.numel():
            self.critic.norm.update(self._normalization_samples)
        rollout = Rollout.empty(config.steps_per_update, env.num_envs, self.observation.critic.shape[-1], self.device)
        for step in range(config.steps_per_update):
            observation = self.observation
            env.set_progress(self.curriculum_transitions, self.updates)
            env.set_estimate(observation.history, self.actor.estimator(observation.history))
            distribution = self.actor.distribution(observation.history)
            action = distribution.sample()
            value = self.critic(observation.critic)
            result = env.step(self.actor.clip(action))
            valid = torch.ones_like(result.terminated) if result.valid is None else result.valid
            timeout = result.truncated & ~result.terminated & valid
            # Invalid physical endpoints never enter the critic, even through
            # a nominal zero bootstrap multiplier (0 * inf is still NaN).
            terminal_critic = torch.where(valid[:, None], result.terminal_critic, observation.critic)
            reward = result.reward + config.gamma * timeout * self.critic(terminal_critic)
            for name, tensor in (("history", observation.history), ("critic", observation.critic),
                                 ("action", action), ("log_prob", distribution.log_prob(action).sum(-1)),
                                 ("mean", distribution.mean), ("std", distribution.stddev), ("value", value),
                                 ("reward", reward), ("done", result.terminated | result.truncated | ~valid),
                                 ("label", observation.label), ("label_valid", observation.label_valid & valid),
                                 ("valid", valid)):
                getattr(rollout, name)[step] = tensor
            self.statistics.record(result)
            self.observation = result.observation
            self.transitions += env.num_envs
        env.set_progress(self.curriculum_transitions, self.updates)
        self._normalization_samples = rollout.critic[rollout.valid]
        return rollout

    @property
    def curriculum_transitions(self) -> int:
        if self.course is not None:
            return self.course.coordinate
        # A foundation run may hold its course at zero while accumulating
        # real data. Continuing the course must not reinterpret that exposure
        # as already completed later stages or reset the learning counters.
        coordinate = max(0, self.transitions - self.config.curriculum_transition_offset)
        cap = self.config.curriculum_transition_cap
        return coordinate if cap is None else min(coordinate, cap)

    # -- optimization -----------------------------------------------------------------
    def update(self, rollout: Rollout) -> dict:
        config = self.config
        with torch.no_grad():
            last_value = self.critic(self.observation.critic)
        returns, advantages = gae(rollout.reward, rollout.value, rollout.done, last_value, config.gamma, config.lam,
                                 valid=rollout.valid)
        data = rollout.flatten()
        data["returns"], data["advantages"] = returns.flatten(), advantages.flatten()
        # Compact before normalization and every loss, including entropy, KL
        # and estimator supervision; masking a final loss would retain NaNs.
        accepted = data.pop("valid")
        data = {name: value[accepted] for name, value in data.items()}
        count = data["returns"].numel()
        metrics = {"surrogate": 0., "value": 0., "entropy": 0., "kl": 0., "estimator": 0., "imitation": 0.}
        if not count:
            self.updates += 1
            return metrics | {"learning_rate": self.learning_rate, "optimization_samples": 0,
                              "optimization_skipped": True}
        advantages = data["advantages"]
        data["advantages"] = (advantages - advantages.mean()) / (advantages.std(unbiased=count > 1) + 1e-8)
        frozen, imitation_weight = self._actor_frozen(), self._imitation_weight()
        surrogate_active = self._surrogate_active()
        steps = 0
        for _ in range(config.epochs):
            permutation = torch.randperm(count, device=self.device)
            for batch in permutation.chunk(config.minibatches):
                metrics_step = self._minibatch({name: value[batch] for name, value in data.items()})
                for name, value in metrics_step.items():
                    metrics[name] += value
                steps += 1
        self.updates += 1
        return {name: value / steps for name, value in metrics.items()} | {
            "learning_rate": self.learning_rate, "optimization_samples": count, "optimization_skipped": False,
            "actor_frozen": frozen, "surrogate_active": surrogate_active, "imitation_weight": imitation_weight,
            **self._action_bound_metrics(data["mean"])}

    def _action_bound_metrics(self, mean: torch.Tensor) -> dict:
        """Fraction of rollout means within 1% of the execution bound, per action axis.

        A saturated mean executes the same boundary action for almost every
        sample, which is the exploration failure seen in PPO2000.
        """
        bounds = self.actor.action_bounds
        near = (mean.abs() >= .99 * bounds).float().mean(0)
        beyond = (mean.abs() > bounds).float().mean(0)
        return {"action/mean_near_bound_fraction": near.tolist(),
                "action/mean_beyond_bound_fraction": beyond.tolist(),
                "action/std": self.actor.log_std.detach().exp().tolist()}

    def _minibatch(self, batch: dict) -> dict:
        config = self.config
        frozen = self._actor_frozen()
        surrogate_active = self._surrogate_active()
        imitation_weight = self._imitation_weight()
        distribution = self.actor.distribution(batch["history"])
        log_prob = distribution.log_prob(batch["action"]).sum(-1)
        with torch.no_grad():
            mean, std = distribution.mean, distribution.stddev
            kl = (torch.log(std / batch["std"]) + (batch["std"].square() + (batch["mean"] - mean).square())
                  / (2 * std.square()) - .5).sum(-1).mean()
            # KL only measures PPO steps. A frozen or imitating actor would
            # otherwise drive the shared critic rate to its bound.
            if surrogate_active:
                if kl > 2 * config.desired_kl:
                    self.learning_rate = max(config.learning_rate_bounds[0], self.learning_rate / 1.5)
                elif 0 < kl < .5 * config.desired_kl:
                    self.learning_rate = min(config.learning_rate_bounds[1], self.learning_rate * 1.5)
            for group in self.optimizer.param_groups:
                group["lr"] = self.learning_rate
        ratio = torch.exp(log_prob - batch["log_prob"])
        advantage = batch["advantages"]
        surrogate = -torch.minimum(ratio * advantage, ratio.clamp(1 - config.clip, 1 + config.clip) * advantage).mean()
        value = self.critic(batch["critic"])
        clipped = batch["value"] + (value - batch["value"]).clamp(-config.value_clip, config.value_clip)
        value_loss = torch.maximum((value - batch["returns"]).square(), (clipped - batch["returns"]).square()).mean()
        entropy = distribution.entropy().sum(-1).mean()
        loss = config.value_coefficient * value_loss
        if surrogate_active:
            loss = loss + surrogate - config.entropy_coefficient * entropy
        imitation = torch.zeros((), device=self.device)
        if imitation_weight > 0 and not frozen:
            with torch.no_grad():
                target = self.imitation_teacher(batch["history"])
            imitation = (distribution.mean - target).square().sum(-1).mean()
            loss = loss + imitation_weight * imitation
        # Frozen actor parameters keep grad=None, which Adam skips entirely.
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_([p for group in self.optimizer.param_groups for p in group["params"]],
                                 config.max_grad_norm)
        self.optimizer.step()
        estimator = 0. if frozen else self._estimator_step(batch)
        return {"surrogate": surrogate.item(), "value": value_loss.item(), "entropy": entropy.item(),
                "kl": kl.item(), "estimator": estimator, "imitation": imitation.item()}

    def _estimator_step(self, batch: dict) -> float:
        valid = batch["label_valid"]
        if not valid.any():
            return 0.
        estimate = self.actor.estimator(batch["history"][valid])
        label = batch["label"][valid]
        loss = F.mse_loss(estimate[:, :3], label[:, :3]) + F.binary_cross_entropy_with_logits(
            estimate[:, 3:], label[:, 3:])
        self.estimator_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.estimator.parameters(), self.config.max_grad_norm)
        self.estimator_optimizer.step()
        return loss.item()

    # -- loop and persistence ---------------------------------------------------------
    def train(self) -> dict:
        config = self.config
        log = (self.output / "metrics.jsonl").open("a")
        # A later collection changes normalization and RNG; an interrupted
        # update can change both optimizers. Keep the completed boundary on
        # CPU, independent of those live tensors, without writing every update.
        completed_state = None
        try:
            if self.updates:
                completed_state = self._checkpoint_snapshot()
            while self.updates < config.updates:
                std_raised = self._raise_stage_action_std()
                self.statistics.reset_window()
                started = time.perf_counter()
                rollout = self.collect()
                collected = time.perf_counter()
                metrics = self.update(rollout)
                updated = time.perf_counter()
                completed_state = self._checkpoint_snapshot()
                record = {"update": self.updates, "transitions": self.transitions,
                          "curriculum_transitions": self.curriculum_transitions,
                          "curriculum_transition_offset": config.curriculum_transition_offset,
                          "curriculum_transition_cap": config.curriculum_transition_cap,
                          "reward_per_step": (float(rollout.reward[rollout.valid].mean())
                                              if rollout.valid.any() else None),
                          "action_std": float(self.actor.log_std.detach().exp().mean()),
                          "timing/collect_s": collected - started,
                          "timing/update_s": updated - collected,
                          "timing/checkpoint_snapshot_s": time.perf_counter() - updated,
                          **metrics, **self.statistics.summary(), **self._course_progress(),
                          **({} if std_raised is None else {"action_std_floor_stage": std_raised})}
                timeout = None
                if self.course is not None:
                    try:
                        record["course"] = self.course.advance(
                            record, config.steps_per_update * self.environment.num_envs)
                    except GateTimeout as error:
                        timeout = str(error)
                        record["course"] = {"timeout": timeout}
                log.write(json.dumps(record) + "\n")
                log.flush()
                gate_passed = config.stop_after_gate and record.get("course", {}).get("gate_passed", False)
                if (timeout is not None or gate_passed or self.updates % config.checkpoint_interval == 0
                        or self.updates == config.updates):
                    self.save(self.output / "checkpoint.pt")
                    self.save(self.output / "checkpoints" / f"update-{self.updates:06d}.pt", light=True)
                if gate_passed:
                    # The resumable boundary must remain immutable while its
                    # actor is evaluated in a separate native simulator process.
                    gate_checkpoint = self.output / "gate-checkpoints" / f"update-{self.updates:06d}.pt"
                    self.save(gate_checkpoint)
                    return {"status": "gate_passed", "gate": record["course"]["gate"],
                            "updates": self.updates, "transitions": self.transitions,
                            "checkpoint": str(gate_checkpoint.resolve()),
                            "actor_checkpoint": str((self.output / "checkpoints"
                                                      / f"update-{self.updates:06d}.pt").resolve())}
                if timeout is not None:
                    # A closed gate is a diagnosis point, not a reason to keep consuming data.
                    return {"status": "gate_timeout", "message": timeout,
                            "updates": self.updates, "transitions": self.transitions}
        except BaseException as error:
            if completed_state is not None:
                try:
                    payload = torch.load(io.BytesIO(completed_state), map_location="cpu", weights_only=True)
                    payload["recovery"] = {
                        "reason": "training_exception",
                        "completed_update": payload["updates"],
                        "discarded_transitions": self.transitions - payload["transitions"],
                        "exception_type": type(error).__name__,
                        "exception_message": str(error),
                    }
                    atomic_save(payload, self.output / "failure-checkpoint.pt")
                except Exception as save_error:
                    # Checkpoint I/O must never replace the training traceback.
                    error.add_note(f"Could not save completed PPO update: {save_error!r}")
            raise
        finally:
            log.close()
        return {"status": "complete", "updates": self.updates, "transitions": self.transitions}

    def _raise_stage_action_std(self) -> str | None:
        """Raise action-std floors once when the course coordinate is inside a configured stage."""
        if self.course is None or not self.config.stage_action_std:
            return None
        stage = self.course.stage()
        floors = self.config.stage_action_std.get(stage)
        if floors is None or stage in self.stage_action_std_applied:
            return None
        with torch.no_grad():
            floor = torch.log(torch.tensor(floors, device=self.actor.log_std.device))
            self.actor.log_std.copy_(torch.maximum(self.actor.log_std, floor))
        self.stage_action_std_applied.append(stage)
        return stage

    def _course_progress(self) -> dict:
        """Per-update native skill counter increments and current course gauges."""
        if self._course_counters is None:
            return {}
        counters, gauges = self.environment.course_progress()
        deltas = {name: value - self._course_counters.get(name, 0) for name, value in counters.items()}
        self._course_counters = counters
        return deltas | gauges

    def _checkpoint_snapshot(self) -> bytes:
        buffer = io.BytesIO()
        torch.save(self.state_dict(), buffer)
        return buffer.getvalue()

    def state_dict(self, *, light: bool = False) -> dict:
        payload = {"format": CHECKPOINT_FORMAT, "config": asdict(self.config), "actor": self.actor.state_dict(),
                   "critic": self.critic.state_dict(), "updates": self.updates, "transitions": self.transitions,
                   "initialization": self.initialization,
                   "course_gates": None if self.course is None else self.course.state_dict(),
                   "stage_action_std_applied": list(self.stage_action_std_applied)}
        if not light:
            payload.update(optimizer=self.optimizer.state_dict(),
                           estimator_optimizer=self.estimator_optimizer.state_dict(),
                           learning_rate=self.learning_rate, torch_rng=torch.get_rng_state(),
                           cuda_rng=(torch.cuda.get_rng_state_all() if self.device.type == "cuda" else []),
                           resume_semantics={"environment_continuation": False,
                                             "episodes": "fresh",
                                             "partial_rollout": "discarded",
                                             "pending_normalization_samples": "not_restored",
                                             "rng": "torch_cpu_and_cuda_only"})
        return payload

    def save(self, path: Path, *, light: bool = False) -> None:
        atomic_save(self.state_dict(light=light), path)

    def load(self, path: Path) -> None:
        """Resume learning state; the simulator restarts cold with fresh episodes."""
        payload = torch.load(path, map_location=self.device, weights_only=True)
        if payload.get("format") != CHECKPOINT_FORMAT or "optimizer" not in payload:
            raise ValueError("Resume requires a full PPO teacher checkpoint")
        saved = PPOConfig.from_dict(payload["config"])
        for name in ("steps_per_update", "actor_hidden", "estimator_hidden", "critic_hidden", "mean_squash",
                     "action_bounds"):
            if getattr(saved, name) != getattr(self.config, name):
                raise ValueError("Resume requires unchanged network shapes and rollout length: " + name)
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.estimator_optimizer.load_state_dict(payload["estimator_optimizer"])
        self.learning_rate, self.updates, self.transitions = (payload["learning_rate"], payload["updates"],
                                                              payload["transitions"])
        self.initialization = payload.get("initialization")
        self.stage_action_std_applied = list(payload.get("stage_action_std_applied", []))
        if self.course is not None and payload.get("course_gates") is not None:
            self.course.load_state_dict(payload["course_gates"],
                                        restart_hold=self.config.course_gate_resume == "restart_hold")
            stage, tag = self.config.course_rewind_stage, self.config.course_rewind_tag
            saved = payload["config"]
            if stage is not None and (saved.get("course_rewind_stage"), saved.get("course_rewind_tag")) != (stage, tag):
                receipt = {**self.course.rewind(stage), "resumed_update": payload["updates"],
                           "checkpoint": str(path)}
                self.stage_action_std_applied = [name for name in self.stage_action_std_applied
                    if self.course.stage_starts.get(name, -1) < self.course.coordinate]
                self.output.mkdir(parents=True, exist_ok=True)
                (self.output / "course-rewind.json").write_text(json.dumps(receipt, indent=2) + "\n")
        torch.set_rng_state(payload["torch_rng"].cpu())
        if self.device.type == "cuda" and payload["cuda_rng"]:
            torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])


def load_actor(path: Path, device="cpu") -> TeacherActor:
    """Deterministic deployment actor (estimator included) from a teacher checkpoint.

    An actor-initialization artifact is also accepted, so a transferred actor
    is evaluated through exactly the same native path as its PPO successors.
    """
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") == ACTOR_INIT_FORMAT:
        return artifact_actor(payload).to(device).eval()
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Not a PPO teacher checkpoint or actor-initialization artifact")
    config = PPOConfig.from_dict(payload["config"])
    actor = TeacherActor(config.actor_hidden, config.estimator_hidden, config.initial_std,
                         mean_squash=config.mean_squash, action_bounds=config.action_bounds).to(device)
    actor.load_state_dict(payload["actor"])
    return actor.eval()


def artifact_actor(payload: dict) -> TeacherActor:
    """Deterministic actor stored in an actor-initialization artifact."""
    # Artifacts written before configurable bounds used the legacy contract.
    bounds = tuple(payload.get("action_bounds", ACTION_BOUNDS))
    actor = TeacherActor(tuple(payload["actor_hidden"]), tuple(payload["estimator_hidden"]),
                         mean_squash=payload["mean_squash"], action_bounds=bounds)
    actor.load_state_dict(payload["actor"])
    return actor
