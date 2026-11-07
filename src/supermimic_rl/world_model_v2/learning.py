"""V2 model fitting, BPTT policy updates and imagined lambda-return critic fitting.

Collection is deliberately outside the learner: a driver can run multiple
model/policy updates per 24-step collection without changing these semantics.
Only real replay labels train the velocity/contact estimator.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .dynamics import MODEL_FIT_PRECISIONS
from .fitting import WorldModelFit
from .imagination import Critic, Imagined, Imagination, InvalidImagination
from .model import WorldModelV2
from .policy import EstimatorActor
from .records import decode_memory, decode_physical
from .termination import PoseContactTermination


@dataclass(frozen=True)
class LearningRateDecay:
    """Cosine decay of every learner rate over actor updates, then constant.

    SuperMimic (Table A.5) decays to 0.03x over 250k of 320k iterations, with
    one policy update per iteration; the actor-update count is that clock here.
    Model updates before policy learning starts therefore keep the full rate.
    """
    updates: int
    final_fraction: float = .03

    def __post_init__(self):
        if type(self.updates) is not int or self.updates < 1:
            raise ValueError('Learning-rate decay needs a positive integer number of actor updates')
        if not 0 < self.final_fraction <= 1:
            raise ValueError('Learning-rate decay final fraction must be in (0, 1]')

    def factor(self, actor_updates: int) -> float:
        progress = min(actor_updates / self.updates, 1.)
        return self.final_fraction + (1 - self.final_fraction) * .5 * (1 + math.cos(math.pi * progress))


@dataclass(frozen=True)
class LearningConfig:
    model_learning_rate: float = 2e-4
    actor_learning_rate: float = 2e-4
    critic_learning_rate: float = 2e-4
    estimator_learning_rate: float = 1e-3
    gamma: float = .99
    gae_lambda: float = .95
    gradient_clip: float = 1.
    imagination_horizon: int = 16
    critic_epochs: int = 5
    critic_minibatches: int = 4
    # Clamp normalized critic inputs to +-clip; None leaves them unbounded (legacy).
    critic_input_clip: float | None = None
    # Differentiate the actor objective's final bootstrap value through the
    # imagined state (legacy). False keeps it as a value, not a gradient source.
    bootstrap_gradient: bool = True
    # Trust region around the initialization teacher: weight of the KL between
    # equal-std Gaussians, 0.5 * ((student - teacher) / teacher_std)^2, on real
    # start histories. Imagination may improve the student only near the
    # teacher, where the world model has data. 0 disables it.
    anchor_coefficient: float = 0.
    # CUDA motion-MLP fitting only; policy imagination and CPU remain FP32.
    model_fit_precision: str = 'float32'
    # None keeps every rate constant (legacy).
    learning_rate_decay: LearningRateDecay | None = None

    def __post_init__(self):
        if isinstance(self.learning_rate_decay, dict):
            object.__setattr__(self, 'learning_rate_decay', LearningRateDecay(**self.learning_rate_decay))
        if self.learning_rate_decay is not None and not isinstance(self.learning_rate_decay, LearningRateDecay):
            raise ValueError('Learning-rate decay must be a LearningRateDecay or its dictionary')
        if type(self.imagination_horizon) is not int or self.imagination_horizon < 1:
            raise ValueError('Imagination horizon must be a positive integer')
        if not math.isfinite(self.anchor_coefficient) or self.anchor_coefficient < 0:
            raise ValueError('Anchor coefficient must be finite and nonnegative')
        if self.model_fit_precision not in MODEL_FIT_PRECISIONS:
            raise ValueError('Model fit precision must be float32 or bfloat16')


@torch.no_grad()
def lambda_returns(rollout: Imagined, gamma: float, lam: float) -> Tensor:
    """Bootstrap at each row's known boundary; true failure has zero continuation."""
    values = rollout.values.detach()
    future = values[-1]
    targets = []
    for step in range(len(rollout.rewards) - 1, -1, -1):
        bootstrap = torch.where(rollout.boundary[step], values[step + 1],
                                (1 - lam) * values[step + 1] + lam * future)
        target = rollout.rewards[step].detach() + gamma * rollout.continuation[step].detach() * bootstrap
        targets.append(target)
        future = torch.where(rollout.active[step], target, future)
    return torch.stack(targets[::-1])


@contextmanager
def frozen_parameters(*modules):
    """Freeze weights, preserving derivatives with respect to state/action inputs."""
    parameters = [(p, p.requires_grad) for module in modules for p in module.parameters()]
    try:
        for p, _ in parameters:
            p.requires_grad_(False)
        yield
    finally:
        for p, required in parameters:
            p.requires_grad_(required)


class Learner:
    def __init__(self, model: WorldModelV2, actor: EstimatorActor, critic: Critic,
                 termination: PoseContactTermination, config: LearningConfig, *, reward,
                 stair_task=None, stair_source=None, jump_task=None, recovery_task=None, disturbance_config=None,
                 platform_task=None, platform_contact=None, platform_geometry=None):
        self.model, self.actor, self.critic, self.config = model, actor, critic, config
        self.fit = WorldModelFit(model)
        self.imagination = Imagination(model, actor, critic, termination, reward=reward, gamma=config.gamma,
                                       stair_task=stair_task, stair_source=stair_source, jump_task=jump_task,
                                       recovery_task=recovery_task, bootstrap_gradient=config.bootstrap_gradient,
                                       disturbance_config=disturbance_config,
                                       platform_task=platform_task, platform_contact=platform_contact,
                                       platform_geometry=platform_geometry)
        self.platform_contact = platform_contact
        if platform_contact is not None:
            from .platform_contacts import PlatformContactModel

            if not isinstance(platform_contact, PlatformContactModel):
                raise ValueError('Platform learning requires a persistable contact model')
        self.optimizers = {
            "model": torch.optim.Adam(model.parameters(), lr=config.model_learning_rate),
            "actor": torch.optim.Adam(actor.decoder.parameters(), lr=config.actor_learning_rate),
            "critic": torch.optim.Adam(critic.parameters(), lr=config.critic_learning_rate),
            "estimator": torch.optim.Adam(actor.estimator.parameters(), lr=config.estimator_learning_rate),
        }
        if platform_contact is not None:
            self.optimizers['platform_contact'] = torch.optim.Adam(
                platform_contact.parameters(), lr=config.model_learning_rate)
        # Every optimizer's configured rate, including optional heads, is its schedule base.
        self.base_learning_rates = {name: optimizer.param_groups[0]['lr']
                                    for name, optimizer in self.optimizers.items()}
        self.updates = {name: 0 for name in self.optimizers}
        self.skipped = {name: 0 for name in self.optimizers}
        self.anchor = None   # (frozen teacher actor, teacher action std [6])

    def set_anchor(self, actor: EstimatorActor, std: Tensor) -> None:
        """Freeze a copy of ``actor`` (normally the PPO initialization) as the trust-region center."""
        device = next(self.actor.parameters()).device
        if std.shape != (6,) or not bool((std > 0).all()) or not bool(torch.isfinite(std).all()):
            raise ValueError('Anchor needs six positive finite action standard deviations')
        self.anchor = (copy.deepcopy(actor).requires_grad_(False).eval().to(device), std.to(device).float().clone())

    def _anchor_deviation(self, frames: Tensor) -> Tensor:
        teacher, std = self.anchor
        with torch.no_grad():
            target = teacher(frames)
        return .5 * ((self.actor(frames) - target) / std).square().mean()

    def learning_rate_factor(self) -> float:
        decay = self.config.learning_rate_decay
        return 1. if decay is None else decay.factor(self.updates['actor'])

    def _schedule(self, name: str) -> None:
        # Rates follow the schedule only when it is enabled; otherwise any
        # externally managed rate (resume reset, guard) is left untouched.
        if self.config.learning_rate_decay is None:
            return
        rate = self.base_learning_rates[name] * self.learning_rate_factor()
        for group in self.optimizers[name].param_groups:
            group['lr'] = rate

    def _step(self, name: str, loss: Tensor) -> bool:
        optimizer = self.optimizers[name]
        optimizer.zero_grad(set_to_none=True)
        finite = bool(torch.isfinite(loss).all())
        if finite:
            loss.backward()
            parameters = [p for group in optimizer.param_groups for p in group["params"]]
            norm = nn.utils.clip_grad_norm_(parameters, self.config.gradient_clip)
            finite = bool(torch.isfinite(norm))
        if finite:
            self._schedule(name)
            optimizer.step()
            self.updates[name] += 1
        else:
            # Do not update Adam moments with invalid gradients or keep them
            # around for the next batch. The driver logs each skip counter.
            optimizer.zero_grad(set_to_none=True)
            self.skipped[name] += 1
        return finite

    def fit_model(self, windows: dict[str, Tensor]) -> dict:
        # Statistics keep seeing the current real course distribution. They
        # are never updated with imagined states or during a backward graph.
        self.fit.observe(windows)
        with self.model.motion.fitting_precision(self.config.model_fit_precision):
            terms = self.fit.terms(windows)
        updated = self._step("model", terms.state + terms.ticks)
        return {"model/updated": updated, "model/state": float(terms.state.detach()),
                "model/ticks": float(terms.ticks.detach()), "model/skipped": self.skipped["model"],
                "model/terrain_truncated_fraction": float(terms.terrain_truncated_fraction)}

    def fit_estimator(self, start: dict[str, Tensor], frames: Tensor) -> dict:
        valid = start["estimator_valid"][..., 0] > .5
        if not valid.any():
            return {"estimator/updated": False, "estimator/labels": 0,
                    "estimator/loss": float('nan'), "estimator/skipped": self.skipped['estimator']}
        target = start["estimator_target"][valid]
        loss = self.actor.estimator_loss(frames[valid], target[..., :3], target[..., 3:5])
        return {"estimator/updated": self._step("estimator", loss), "estimator/labels": int(valid.sum()),
                "estimator/loss": float(loss.detach()), "estimator/skipped": self.skipped["estimator"]}

    def fit_platform_contact(self, record: dict[str, Tensor], *, command_height: Tensor) -> dict:
        """Fit attribution from actual four-tick evidence without updating motion.

        The caller provides the admitted height for this recorded interval.
        Unknown historical evidence supplies no labels, including no negatives.
        Normalizers see only real, labeled training examples. Actor imagination
        never invokes this method or supplies its own predictions as truth.
        """
        from .jump_task import JumpTaskState
        from .model import Ticks
        from .platform_records import PlatformRecordState

        model = self.platform_contact
        if model is None:
            raise ValueError('Platform contact fitting requires an installed model')
        ticks = Ticks.decode(record['ticks'], decode_physical(record['physical']))
        arguments = dict(intent=PlatformRecordState.decode(record).intent,
                         jump_state=JumpTaskState.decode(record),
                         terrain_tile=record['terrain_tile'][:, 0].long(), command_height=command_height)
        targets = model.targets(record, command_height=command_height)
        with torch.no_grad():
            prediction = model.predict(ticks, **arguments)
            valid = prediction.available & targets.valid
            labels = int(valid.sum())
            if not labels:
                return {'platform_contact/updated': False, 'platform_contact/labels': 0,
                        'platform_contact/skipped': self.skipped['platform_contact']}
            model.observe(prediction, targets)
        # Observe changes both feature and height scales; evaluate the loss
        # with the newly observed scales, just as the motion fitter does.
        terms = model.losses_from_features(prediction.features, targets, available=prediction.available)
        updated = self._step('platform_contact', sum(terms.values()))
        return {'platform_contact/updated': updated, 'platform_contact/labels': labels,
                'platform_contact/height_labels': int((valid & targets.height_valid).sum()),
                'platform_contact/skipped': self.skipped['platform_contact'],
                **{'platform_contact/' + name: float(value.detach()) for name, value in terms.items()}}

    def _usable_rollout(self, start, frames):
        """Retain the healthy graph on an atlas miss; replay other invalid graphs.

        Finite motion whose terrain is undefined is removed before analytic
        processing. Nonfinite predictions/returns or unknown reward references
        still use no-grad screening and a fresh healthy graph, because masking
        their loss cannot prevent a zero-times-NaN derivative in backward.
        """
        rejected = replayed = 0
        causes = dict.fromkeys(('nonfinite_prediction', 'terrain_query', 'reward_reference', 'nonfinite_return'), 0)
        while len(frames):
            rollout = None
            try:
                rollout = self.imagination.rollout(start, frames, self.config.imagination_horizon,
                                                    isolate_terrain=True)
                for name, rows in rollout.rejection_reasons.items():
                    causes[name] += int(rows.sum())
                retained = rollout.usable
                causes['nonfinite_return'] += int((~retained).sum())
                usable = torch.zeros(len(frames), dtype=torch.bool, device=frames.device)
                usable[rollout.source_rows] = retained
                if retained.all():
                    rejected += int((~usable).sum())
                    return (rollout if len(rollout.objective) else None), rejected, replayed, causes
            except InvalidImagination as error:
                usable = ~error.invalid_rows
                for name, rows in error.rejection_reasons.items():
                    causes[name] += int(rows.sum())
            rejected += int((~usable).sum())
            start = {name: value[usable] for name, value in start.items()}
            frames = frames[usable]
            del rollout
            if len(frames):
                replayed += len(frames)
                with torch.no_grad():
                    screened = self.imagination.rollout(start, frames, self.config.imagination_horizon,
                                                        screen_invalid=True)
                    usable = screened.usable
                    for name, rows in screened.rejection_reasons.items():
                        causes[name] += int(rows.sum())
                    causes['nonfinite_return'] += int((~usable & ~screened.invalid).sum())
                    rejected += int((~usable).sum())
                    start = {name: value[usable] for name, value in start.items()}
                    frames = frames[usable]
                    del screened
                replayed += len(frames)
        return None, rejected, replayed, causes

    def update_policy(self, start: dict[str, Tensor], frames: Tensor, *, microbatch_size: int | None = None,
                      train_actor: bool = True) -> dict:
        """One actor step through imagination, then critic fitting on its lambda returns.

        ``train_actor=False`` fits only the critic on the frozen actor's
        imagined returns. A transferred actor needs this before its first step:
        the H-step objective bootstraps from the critic, and an untrained
        critic otherwise steers the copied policy away from its competence.
        """
        config = self.config
        start = self.imagination.prepare(start, config.imagination_horizon)
        with torch.no_grad():
            features = self.imagination.critic_features(
                decode_physical(start["physical"]), start["height"], start["scan"],
                decode_memory(start["memory"]), start["goal"], start["sim_context"],
                start["failure_memory"], start["failure_conditions"],
                torch.cat((start['reward_memory'], start['support_memory']), -1), start.get('motion_task'),
                **self.imagination.task_features(start))
            self.critic.norm.update(features)
        batch_size = len(frames)
        microbatch_size = batch_size if microbatch_size is None else microbatch_size
        if batch_size < 1 or microbatch_size < 1:
            raise ValueError('Policy batches and microbatches must be positive')
        optimizer = self.optimizers['actor']
        optimizer.zero_grad(set_to_none=True)
        actor_loss, finite, targets, features = 0., True, [], []
        length_sum = survival_sum = invalid_sum = 0.
        accepted = replayed = 0
        causes = dict.fromkeys(('nonfinite_prediction', 'terrain_query', 'reward_reference', 'nonfinite_return'), 0)
        gradient = torch.enable_grad() if train_actor else torch.no_grad()
        contact = () if self.platform_contact is None else (self.platform_contact,)
        with frozen_parameters(self.model, self.critic, *contact), gradient:
            for first in range(0, batch_size, microbatch_size):
                part = slice(first, first + microbatch_size)
                rollout, rejected, retried, rejected_causes = self._usable_rollout(
                    {key: value[part] for key, value in start.items()}, frames[part])
                invalid_sum += rejected
                replayed += retried
                for name, count in rejected_causes.items():
                    causes[name] += count
                if rollout is None:
                    continue
                accepted += len(rollout.objective)
                # Keep the original requested batch denominator. Unknown rows
                # get no reward, failure label or bootstrap target invented.
                loss = -rollout.objective.sum() / batch_size
                finite = finite and bool(torch.isfinite(loss))
                if finite and train_actor:
                    loss.backward()
                actor_loss += float(loss.detach())
                valid = rollout.active & ~rollout.invalid[None]
                targets.append(lambda_returns(rollout, config.gamma, config.gae_lambda)[valid])
                features.append(rollout.critic_inputs[:-1][valid])
                length_sum += float(rollout.active.sum())
                survival_sum += float(rollout.final_survival.sum())
                # Retaining the last rollout would keep its full H-step graph
                # alive during critic fitting and increase peak GPU memory.
                del rollout, loss
        finite = finite and accepted > 0
        anchor = float('nan')
        if finite and train_actor and config.anchor_coefficient:
            if self.anchor is None:
                raise ValueError('anchor_coefficient requires set_anchor with the initialization teacher')
            deviation = self._anchor_deviation(frames)
            finite = bool(torch.isfinite(deviation))
            if finite:
                (config.anchor_coefficient * deviation).backward()
                anchor = float(deviation.detach())
        if finite and train_actor:
            norm = nn.utils.clip_grad_norm_(self.actor.decoder.parameters(), config.gradient_clip)
            finite = bool(torch.isfinite(norm))
        if finite and train_actor:
            self._schedule('actor')
            optimizer.step()
            self.updates['actor'] += 1
        elif train_actor:
            optimizer.zero_grad(set_to_none=True)
            self.skipped['actor'] += 1
        targets = torch.cat(targets) if targets else frames.new_empty(0)
        features = torch.cat(features) if features else frames.new_empty(0, self.critic.norm.mean.numel())
        total_loss, batches = 0., 0
        for _ in range(config.critic_epochs):
            permutation = torch.randperm(len(targets), device=targets.device)
            for batch in permutation.chunk(config.critic_minibatches):
                if not len(batch):
                    continue
                loss = (self.critic(features[batch]) - targets[batch]).square().mean()
                if self._step("critic", loss):
                    total_loss += float(loss.detach())
                    batches += 1
        return {"actor/updated": finite and train_actor, "actor/loss": actor_loss if accepted else float('nan'),
                "learning_rate/factor": self.learning_rate_factor(),
                "actor/anchor_kl": anchor,
                "actor/skipped": self.skipped["actor"], "critic/loss": total_loss / max(batches, 1),
                "critic/skipped": self.skipped["critic"],
                "imagination/mean_length": length_sum / batch_size,
                "imagination/survival": survival_sum / batch_size,
                "imagination/invalid_fraction": invalid_sum / batch_size,
                "imagination/accepted_trajectories": accepted,
                "imagination/replayed_trajectories": replayed,
                **{f'imagination/rejected_{name}_fraction': count / batch_size for name, count in causes.items()},
                **self.fit_estimator(start, frames)}

    @torch.no_grad()
    def calibrate_appended_critic_features(self, start: dict[str, Tensor], first: int) -> None:
        """Set normalizer statistics of critic columns ``first:`` from eligible recorded starts.

        A migration appends critic inputs with neutral statistics; the online
        trainer calibrates them once real records exist, keeping older columns.
        ``first=0`` explicitly recalibrates a fresh critic after a reward change.
        """
        norm, imagination = self.critic.norm, self.imagination
        if type(first) is not int or not 0 <= first < len(norm.mean):
            raise ValueError('Explicit critic feature calibration range required')
        features = imagination.critic_features(decode_physical(start['physical']), start['height'], start['scan'],
            decode_memory(start['memory']), start['goal'], start['sim_context'], start['failure_memory'],
            start['failure_conditions'], torch.cat((start['reward_memory'], start['support_memory']), -1),
            start['motion_task'], **imagination.task_features(start))[:, first:]
        if not torch.isfinite(features).all():
            raise ValueError('Recorded critic calibration features are nonfinite')
        norm.mean[first:].copy_(features.mean(0))
        norm.var[first:].copy_(features.var(0, unbiased=False))

    def state_dict(self) -> dict:
        floor = self.imagination.contact_floor
        return {"model": self.model.state_dict(), "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(), "updates": self.updates.copy(), "skipped": self.skipped.copy(),
                "optimizers": {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()},
                **({} if self.platform_contact is None else
                   {'platform_contact': self.platform_contact.state_dict()}),
                **({} if floor is None else {"contact_floor": {"body": floor[0], "base": floor[1]}}),
                **({} if self.anchor is None else
                   {"anchor": {"actor": self.anchor[0].state_dict(), "std": self.anchor[1],
                               "actor_features": self.anchor[0].features.configuration}})}

    def load_state_dict(self, state: dict) -> None:
        if (self.platform_contact is None) != ('platform_contact' not in state):
            raise ValueError('Checkpoint and learner platform contact contracts differ')
        for name in ("model", "actor", "critic"):
            getattr(self, name).load_state_dict(state[name])
        if self.platform_contact is not None:
            self.platform_contact.load_state_dict(state['platform_contact'])
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state["optimizers"][name])
        self.updates, self.skipped = dict(state["updates"]), dict(state["skipped"])
        anchor = state.get("anchor")
        if anchor is not None:
            from ..actor_features import ActorFeatures

            teacher = copy.deepcopy(self.actor)
            # An existing reference policy retains its own command semantics
            # when the trainable actor activates new decoder features.
            teacher.features = ActorFeatures(anchor.get("actor_features"))
            teacher.load_state_dict(anchor["actor"])
            self.set_anchor(teacher, anchor["std"])
        floor = state.get("contact_floor")
        device = next(self.model.parameters()).device
        self.imagination.contact_floor = (None if floor is None else
                                          (floor["body"].to(device), floor["base"].to(device)))
