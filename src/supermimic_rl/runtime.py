"""Collection/model fitting/imagination orchestration and cold-restart checkpoints."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time

import torch

from .contracts import Snapshot, StateSchema, Transition
from .learning import Learner, LearningConfig
from .replay import ReplayBuffer


@dataclass(frozen=True)
class RunConfig:
    seed: int = 7
    updates: int = 100
    warmup_steps: int = 128
    collection_steps: int = 3
    replay_steps: int = 256
    model_batches: int = 10
    model_batch_size: int = 256
    imagination_batch_size: int = 256
    validation_batch_size: int = 128
    validation_fraction: float = 0.125
    action_noise_std: float = 0.1
    checkpoint_interval: int = 10
    initial_world_batches: int = 0
    world_pretrain_iterations: int = 0
    # None shares the online collection cadence; fitting shares model_batches.
    world_pretrain_collection_steps: int | None = None
    evaluation_interval: int = 0
    evaluation_steps: int = 300
    tensorboard: bool = False

    def __post_init__(self):
        for name in ("updates", "warmup_steps", "collection_steps", "replay_steps", "model_batches",
                     "model_batch_size", "imagination_batch_size", "validation_batch_size",
                     "checkpoint_interval", "evaluation_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("initial_world_batches", "world_pretrain_iterations", "evaluation_interval"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if (self.world_pretrain_collection_steps is not None
                and (type(self.world_pretrain_collection_steps) is not int
                     or self.world_pretrain_collection_steps < 1)):
            raise ValueError("world_pretrain_collection_steps must be positive or None")
        if self.replay_steps < self.warmup_steps:
            raise ValueError("Replay must retain the complete initial warmup")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("Use an explicit nonempty environment holdout")
        if not math.isfinite(self.action_noise_std) or self.action_noise_std < 0:
            raise ValueError("Invalid action noise")
        if type(self.seed) is not int or self.seed < 0 or type(self.tensorboard) is not bool:
            raise ValueError("Invalid seed or tensorboard setting")


def atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(payload: dict, path: Path) -> None:
    """Readers see a complete progress record or evaluation request."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class Collector:
    def __init__(self, environment, learner: Learner, replay: ReplayBuffer, *,
                 seed: int, action_noise_std: float, generation: int = 0):
        self.environment, self.learner, self.replay = environment, learner, replay
        self.action_noise_std = action_noise_std
        self.generation = generation
        self.generator = torch.Generator(device=learner.device).manual_seed(seed)
        current = environment.current
        if current is None:
            raise ValueError("Initialize the environment and its schema before collection")
        if current.state.device != learner.device:
            raise ValueError("Environment and learner must share the training device")
        self.history = self._frame(current)[:, None].expand(-1, learner.config.history_length, -1).clone()
        self.steps = 0
        self.failed_in_environment_step = False

    @staticmethod
    def _frame(snapshot: Snapshot):
        return snapshot.observation if snapshot.actor_observation is None else snapshot.actor_observation

    def update_training_progress(self):
        hook = getattr(self.environment, "set_training_progress", None)
        if callable(hook):
            hook(real_transitions=self.replay.total_added, actor_updates=self.learner.actor_updates)

    def _session_snapshot(self, snapshot):
        result = snapshot.clone()
        # Cold restarts create fresh simulator episodes. Never connect their
        # first transition to an old episode with the same local counter.
        offset = self.generation * (1 << 40)
        for name in ("episode_id", "control_epoch"):
            value = getattr(result, name)
            if (value < 0).any() or (value >= (1 << 40)).any():
                raise ValueError("Native episode/epoch counter exceeds the session namespace")
            setattr(result, name, value + offset)
        return result

    @torch.no_grad()
    def collect(self, steps: int):
        self.failed_in_environment_step = False
        reward_sum = 0.0
        terminations = truncations = 0
        for _ in range(steps):
            self.update_training_progress()
            current = self.environment.current
            estimate_hook = getattr(self.environment, "set_policy_estimate", None)
            if self.learner.estimator is not None and callable(estimate_hook):
                estimate_hook(self._frame(current), self.learner.estimator(self.history))
            action = self.learner.actor(self.history)
            noise = torch.randn(action.shape, device=action.device, generator=self.generator) * self.action_noise_std
            issued = (action + noise).clamp(-self.learner.actor.action_bounds, self.learner.actor.action_bounds)
            try:
                transition = self.environment.step(issued)
            except BaseException:
                # No replay write has started for this interval. A failed
                # replay.add itself cannot provide this same guarantee.
                self.failed_in_environment_step = True
                raise
            self.replay.add(Transition(self._session_snapshot(transition.start), transition.action,
                                      self._session_snapshot(transition.end), transition.reward,
                                      transition.terminated, transition.truncated, transition.valid,
                                      transition.substeps))
            next_snapshot = self.environment.current
            frame = self._frame(next_snapshot)
            self.history = torch.cat((self.history[:, 1:], frame[:, None]), 1)
            reset = ((current.episode_id != next_snapshot.episode_id)
                     | (current.control_epoch != next_snapshot.control_epoch))
            self.history[reset] = frame[reset, None].expand(-1, self.learner.config.history_length, -1)
            reward_sum += float(transition.reward.sum())
            terminations += int(transition.terminated.sum())
            truncations += int(transition.truncated.sum())
            self.steps += 1
        self.update_training_progress()
        return {"collection/reward_mean": reward_sum / (steps * self.environment.num_envs),
                "collection/terminations": terminations, "collection/truncations": truncations}


class TrainingRun:
    """The real simulator supplies data repeatedly; this is not offline-only RL."""

    def __init__(self, environment, output: Path, config: RunConfig = RunConfig(),
                 learning: LearningConfig = LearningConfig(), *, resume: Path | None = None,
                 initial_world: Path | None = None, initial_actor: Path | None = None,
                 initial_actor_sha256: str | None = None):
        self.validate_initialization(learning, resume=resume, initial_world=initial_world,
                                     initial_actor=initial_actor,
                                     initial_actor_sha256=initial_actor_sha256)
        self.environment, self.output, self.config = environment, Path(output), config
        if resume is None and self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError("Use a new output directory or an explicit checkpoint resume")
        self.output.mkdir(parents=True, exist_ok=True)
        if environment.current is None:
            environment.reset()
        if environment.num_envs < 2:
            raise ValueError("At least two environments are needed for a separate validation pool")
        required = max(learning.fit_horizon, learning.history_length, learning.imagination_horizon)
        if config.replay_steps < required or config.warmup_steps < required:
            raise ValueError("Warmup and replay must contain the fit/history/validation horizons")
        torch.manual_seed(config.seed)
        self.learner = Learner(environment.schema, learning, device=environment.device)
        self.replay = ReplayBuffer(environment.schema, environment.num_envs,
                                   config.replay_steps * environment.num_envs)
        self.generator = torch.Generator().manual_seed(config.seed + 1)
        heldout = max(1, min(environment.num_envs - 1, round(environment.num_envs * config.validation_fraction)))
        split = environment.num_envs - heldout
        self.training_envs = list(range(split))
        self.validation_envs = list(range(split, environment.num_envs))
        self.completed_updates = 0
        self.generation = 0
        self.initial_world_batches_completed = 0
        self.initial_world_pretraining_complete = config.initial_world_batches == 0
        self.world_pretrain_iterations_completed = 0
        self.world_pretrain_collection_steps_completed = 0
        self.world_pretraining_complete = config.world_pretrain_iterations == 0
        self._run_started = None
        self._latest_metrics = None
        self._latest_evaluation_request = None
        self._restored_collection_rng = None
        self._restored_collection_steps = 0
        self._failure_recovery = None
        if resume is not None:
            self._restore(Path(resume))
        if initial_world is not None:
            payload = torch.load(initial_world, map_location=self.learner.device, weights_only=True)
            if payload.get("format") != "supermimic-rl-world-fit-v1":
                raise ValueError("initial_world must be a fit-world checkpoint")
            self.learner.load_state_dict(payload["learner"])
            if self.learner.actor_updates:
                raise ValueError("A world-fit initialization cannot contain policy updates")
        if initial_actor is not None:
            self.learner.load_initial_actor(initial_actor, initial_actor_sha256)
        self.collector = Collector(environment, self.learner, self.replay, seed=config.seed + 2,
                                   action_noise_std=config.action_noise_std, generation=self.generation)
        if self._restored_collection_rng is not None:
            self.collector.generator.set_state(self._restored_collection_rng.cpu())
        self.collector.steps = self._restored_collection_steps
        self.collector.update_training_progress()
        self.writer = None
        if config.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(self.output / "tensorboard"))
        adapter = getattr(environment, "adapter", None)
        provenance = getattr(adapter, "provenance", {})
        self.manifest = {"schema": environment.schema.to_dict(), "schema_hash": environment.schema.fingerprint,
                         "run": asdict(config), "learning": asdict(learning),
                         "actor_source_metadata": self.learner.actor_source_metadata,
                         "training_envs": self.training_envs, "validation_envs": self.validation_envs,
                         "provenance": provenance, "native_physics_verified_by_this_run": False,
                         "evaluation_semantics": "numbered checkpoints for a separate worker; no in-process evaluation or environment reset",
                         "resume_semantics": "learning/replay restored; simulator cold reset with a fresh episode namespace"}
        (self.output / f"manifest-{self.generation:03d}.json").write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2) + "\n")
        self._write_progress("ready", emit=True)

    @staticmethod
    def validate_initialization(learning: LearningConfig, *, resume=None, initial_world=None,
                                initial_actor=None, initial_actor_sha256=None) -> None:
        """Reject contradictory initialization before launching native physics."""
        if resume is not None and (initial_world is not None or initial_actor is not None):
            raise ValueError("Choose either a full resume or initial world/actor weights")
        if (initial_actor is None) != (initial_actor_sha256 is None):
            raise ValueError("Initial actor checkpoint path and SHA256 must be provided together")
        if initial_actor is not None:
            if learning.actor_architecture != "transferred_transformer":
                raise ValueError("Initial actor transfer requires transferred_transformer")
            if (not isinstance(initial_actor_sha256, str)
                    or re.fullmatch(r"[0-9a-f]{64}", initial_actor_sha256) is None):
                raise ValueError("Initial actor checkpoint requires an explicit lowercase SHA256")
            if not Path(initial_actor).is_file():
                raise FileNotFoundError(initial_actor)

    def _restore(self, path):
        # Replay remains on CPU. Module/optimizer loading moves only learner
        # tensors to their devices instead of staging the entire replay on GPU.
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != "supermimic-rl-checkpoint-v1":
            raise ValueError("Unknown training checkpoint")
        # Older v1 checkpoints omit newly introduced fields. Operational
        # checkpoint/evaluation scheduling may change without changing learning.
        previous_config = asdict(RunConfig(**payload["run_config"]))
        current_config = asdict(self.config)
        for name in ("updates", "checkpoint_interval", "evaluation_interval", "evaluation_steps"):
            previous_config.pop(name)
            current_config.pop(name)
        if previous_config != current_config:
            raise ValueError("Resume requires unchanged collection and learning settings")
        if self.config.updates < payload["completed_updates"]:
            raise ValueError("The requested update target precedes the saved progress")
        self.learner.load_state_dict(payload["learner"])
        self.replay.load_state_dict(payload["replay"])
        self.generator.set_state(payload["replay_rng"].cpu())
        torch.set_rng_state(payload["torch_rng"].cpu())
        if self.learner.device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])
        self.completed_updates = payload["completed_updates"]
        self.generation = payload["generation"] + 1
        self._restored_collection_rng = payload.get("collection_rng")
        self._restored_collection_steps = payload.get(
            "collection_steps", self.replay.total_added // self.environment.num_envs)
        pretraining = payload.get("initial_world_pretraining", {
            "completed_batches": 0, "complete": self.config.initial_world_batches == 0,
        })
        completed, complete = pretraining["completed_batches"], pretraining["complete"]
        if (type(completed) is not int or not 0 <= completed <= self.config.initial_world_batches
                or type(complete) is not bool
                or complete and completed != self.config.initial_world_batches
                or self.completed_updates > 0 and not complete):
            raise ValueError("Inconsistent initial world pretraining checkpoint progress")
        self.initial_world_batches_completed = completed
        self.initial_world_pretraining_complete = complete
        pretraining = payload.get("world_pretraining", {
            "completed_iterations": 0, "collection_steps_completed": 0,
            "complete": self.config.world_pretrain_iterations == 0,
        })
        iterations = pretraining["completed_iterations"]
        collected = pretraining["collection_steps_completed"]
        complete = pretraining["complete"]
        target = self.config.world_pretrain_iterations
        steps = self.world_pretrain_collection_steps
        if (type(iterations) is not int or not 0 <= iterations <= target
                or type(collected) is not int
                or not iterations * steps <= collected <= min(iterations + 1, target) * steps
                or collected > self._restored_collection_steps
                or type(complete) is not bool
                or complete and iterations != target
                or (iterations or collected) and not self.initial_world_pretraining_complete
                or (self.completed_updates or self.learner.actor_updates) and not complete):
            raise ValueError("Inconsistent collecting world pretraining checkpoint progress")
        self.world_pretrain_iterations_completed = iterations
        self.world_pretrain_collection_steps_completed = collected
        self.world_pretraining_complete = complete
        environment_state = payload.get("environment_training_state")
        if environment_state is not None:
            hook = getattr(self.environment, "load_training_state_dict", None)
            if not callable(hook):
                raise ValueError("Environment cannot restore its saved training state")
            hook(environment_state)

    def save(self, path: Path | None = None):
        self.collector.update_training_progress()
        payload = {"format": "supermimic-rl-checkpoint-v1", "run_config": asdict(self.config),
                   "learner": self.learner.state_dict(), "replay": self.replay.state_dict(),
                   "completed_updates": self.completed_updates, "generation": self.generation,
                   "initial_world_pretraining": {
                       "completed_batches": self.initial_world_batches_completed,
                       "complete": self.initial_world_pretraining_complete},
                   "world_pretraining": {
                       "completed_iterations": self.world_pretrain_iterations_completed,
                       "collection_steps_completed": self.world_pretrain_collection_steps_completed,
                       "complete": self.world_pretraining_complete},
                   "collection_rng": self.collector.generator.get_state(),
                   "collection_steps": self.collector.steps,
                   "replay_rng": self.generator.get_state(), "torch_rng": torch.get_rng_state(),
                   "cuda_rng": torch.cuda.get_rng_state_all() if self.learner.device.type == "cuda" else []}
        hook = getattr(self.environment, "training_state_dict", None)
        if callable(hook):
            payload["environment_training_state"] = hook()
        atomic_save(payload, path or self.output / "checkpoint.pt")

    def _write_progress(self, stage, *, metrics=None, error=None, emit=False):
        if metrics is not None:
            self._latest_metrics = metrics
        progress = {
            "format": "supermimic-rl-progress-v1", "stage": stage,
            "generation": self.generation, "latest_update": self.completed_updates,
            "target_updates": self.config.updates,
            "warmup_steps_completed": min(
                self.config.warmup_steps, self.replay.total_added // self.environment.num_envs),
            "warmup_steps_target": self.config.warmup_steps,
            "initial_world_batches_completed": self.initial_world_batches_completed,
            "initial_world_batches_target": self.config.initial_world_batches,
            "initial_world_pretraining_complete": self.initial_world_pretraining_complete,
            "world_pretrain_iterations_completed": self.world_pretrain_iterations_completed,
            "world_pretrain_iterations_target": self.config.world_pretrain_iterations,
            "world_pretrain_collection_steps_completed": self.world_pretrain_collection_steps_completed,
            "world_pretraining_complete": self.world_pretraining_complete,
            "world_optimizer_steps": self.learner.world_updates,
            "actor_optimizer_steps": self.learner.actor_updates,
            "critic_optimizer_steps": self.learner.critic_updates,
            "transitions": self.replay.total_added, "stored_transitions": len(self.replay),
            "elapsed_session_s": (0.0 if self._run_started is None
                                  else time.perf_counter() - self._run_started),
            "evaluation_interval": self.config.evaluation_interval,
            "evaluation_steps": self.config.evaluation_steps,
            "evaluation_mode": "separate-worker checkpoint requests; not evaluated in training",
            "latest_evaluation_request": self._latest_evaluation_request,
            "latest_metrics": self._latest_metrics,
            "recovery": self._failure_recovery,
        }
        if error is not None:
            progress["error"] = {"type": type(error).__name__, "message": str(error)}
        atomic_write_json(progress, self.output / "progress.json")
        if emit:
            print(
                f"[supermimic] {stage}: update={self.completed_updates}/{self.config.updates} "
                f"warmup={progress['warmup_steps_completed']}/{self.config.warmup_steps} "
                f"initial_world={self.initial_world_batches_completed}/{self.config.initial_world_batches} "
                f"world_pretrain={self.world_pretrain_iterations_completed}/{self.config.world_pretrain_iterations} "
                f"world={self.learner.world_updates} actor={self.learner.actor_updates} "
                f"transitions={self.replay.total_added}", file=sys.stderr, flush=True,
            )

    def _pretrain_world(self):
        if self.initial_world_pretraining_complete:
            return
        self._write_progress("pretraining_world", emit=True)
        with (self.output / "pretraining.jsonl").open("a") as handle:
            for index in range(self.initial_world_batches_completed, self.config.initial_world_batches):
                metrics = self.learner.fit_world(self._sample(self.learner.config.fit_horizon))
                if not all(math.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError("Nonfinite initial world pretraining metrics")
                self.initial_world_batches_completed = index + 1
                handle.write(json.dumps({"batch": index + 1, **metrics}, allow_nan=False) + "\n")
                if index == 0 or (index + 1) % 25 == 0:
                    handle.flush()
                    self._write_progress("pretraining_world", metrics=metrics, emit=True)
        validation = self.learner.validate_world(
            self._sample(self.learner.config.fit_horizon, validation=True))
        if not all(math.isfinite(value) for value in validation.values()):
            raise FloatingPointError("Nonfinite initial world validation metrics")
        self.initial_world_pretraining_complete = True
        # Persist completion before any actor update, including interrupted runs.
        self.save()
        self._write_progress("initial_world_complete", metrics=validation, emit=True)

    @property
    def world_pretrain_collection_steps(self):
        """Vectorized physical steps collected per world-only iteration."""
        return self.config.world_pretrain_collection_steps or self.config.collection_steps

    def _pretrain_world_with_collection(self):
        """Fit on fresh trajectories without updating actor, critic, or normalization."""
        if self.world_pretraining_complete:
            return
        target = self.config.world_pretrain_iterations
        metrics = None
        with (self.output / "world-pretraining.jsonl").open("a") as handle:
            for index in range(self.world_pretrain_iterations_completed, target):
                remaining = ((index + 1) * self.world_pretrain_collection_steps
                             - self.world_pretrain_collection_steps_completed)
                self._write_progress("world_pretraining_collecting")
                self._synchronize()
                started = time.perf_counter()
                metrics = self._collect(remaining, world_pretraining=True) if remaining else {}
                self._synchronize()
                collected = time.perf_counter()
                self._write_progress("world_pretraining_fitting")
                batches = [self.learner.fit_world(self._sample(self.learner.config.fit_horizon))
                           for _ in range(self.config.model_batches)]
                metrics.update({key: sum(batch[key] for batch in batches) / len(batches)
                                for key in batches[0]})
                metrics.update(self.learner.validate_world(
                    self._sample(self.learner.config.fit_horizon, validation=True)))
                self._synchronize()
                metrics.update({
                    "iteration": index + 1, "generation": self.generation,
                    "transitions": self.replay.total_added,
                    "world_optimizer_steps": self.learner.world_updates,
                    "actor_optimizer_steps": self.learner.actor_updates,
                    "critic_optimizer_steps": self.learner.critic_updates,
                    "collection_steps_completed": self.world_pretrain_collection_steps_completed,
                    # After a cold resume, collection metrics describe only the
                    # newly collected suffix; the saved prefix remains in replay.
                    "collection_steps_this_session": remaining,
                    "timing/collection_s": collected - started,
                    "timing/learning_validation_s": time.perf_counter() - collected,
                })
                if not all(math.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError("Nonfinite collecting world pretraining metrics")
                self.world_pretrain_iterations_completed = index + 1
                handle.write(json.dumps(metrics, sort_keys=True, allow_nan=False) + "\n")
                checkpoint_due = (index + 1) % self.config.checkpoint_interval == 0
                if checkpoint_due or index + 1 == target:
                    handle.flush()
                    if index + 1 < target:
                        self.save()
                self._write_progress("world_pretraining", metrics=metrics,
                                     emit=(index == 0 or checkpoint_due or index + 1 == target))
        self.world_pretraining_complete = True
        # A complete world-only checkpoint is reviewable before the actor changes.
        self.save()
        self._write_progress("world_pretraining_complete", metrics=metrics, emit=True)

    def _request_evaluation(self):
        name = f"update-{self.completed_updates:06d}"
        checkpoint = self.output / "checkpoints" / f"{name}.pt"
        self.save(checkpoint)
        request = {
            "format": "supermimic-rl-evaluation-request-v1",
            "status": "pending_external_evaluation", "update": self.completed_updates,
            "checkpoint": str(checkpoint.resolve()), "steps": self.config.evaluation_steps,
            "schema_hash": self.environment.schema.fingerprint,
            "suggested_output": str((self.output / "evaluations" / name).resolve()),
            "execution": "separate process; training does not reset or collect evaluation data",
        }
        factory_config = self.output.parent / "factory-config.json"
        if factory_config.is_file():
            request["factory_config"] = str(factory_config.resolve())
        path = self.output / "evaluations" / f"{name}.json"
        atomic_write_json(request, path)
        self._latest_evaluation_request = str(path.resolve())

    def _sample(self, horizon, *, validation=False):
        return self.replay.sample(self.config.validation_batch_size if validation else self.config.model_batch_size,
                                  horizon, device=self.learner.device, generator=self.generator,
                                  env_ids=self.validation_envs if validation else self.training_envs)

    def _synchronize(self):
        if self.learner.device.type == "cuda":
            torch.cuda.synchronize(self.learner.device)

    def _collect(self, steps, *, world_pretraining=False):
        """Save only simulator failures while the learner is between updates."""
        before = self.collector.steps
        try:
            metrics = self.collector.collect(steps)
        except BaseException as error:
            if world_pretraining:
                self.world_pretrain_collection_steps_completed += self.collector.steps - before
            if self.collector.failed_in_environment_step:
                checkpoint = self.output / "failure-checkpoint.pt"
                self._failure_recovery = {
                    "status": "save_failed", "checkpoint": None,
                    "attempted_checkpoint": str(checkpoint.resolve()),
                    "completed_updates": self.completed_updates,
                    "transitions": self.replay.total_added,
                    "semantics": (
                        "Completed learner updates and complete replay transitions only; "
                        "the failed simulator interval is excluded. Resume cold-resets the "
                        "simulator with a new episode generation, retains completed warmup "
                        "and world pretraining (including its collected prefix), "
                        "and starts a fresh online collection "
                        "phase while keeping its already collected prefix."),
                }
                try:
                    self.save(checkpoint)
                except BaseException as save_error:
                    self._failure_recovery["save_error"] = {
                        "type": type(save_error).__name__, "message": str(save_error),
                    }
                    error.add_note(f"Failure checkpoint could not be saved: {save_error}")
                else:
                    self._failure_recovery.update(
                        status="recoverable", checkpoint=str(checkpoint.resolve()))
            raise
        if world_pretraining:
            self.world_pretrain_collection_steps_completed += self.collector.steps - before
        return metrics

    def run(self):
        run_started = time.perf_counter()
        self._run_started = run_started
        try:
            remaining = max(0, self.config.warmup_steps
                            - self.replay.total_added // self.environment.num_envs)
            if remaining:
                self._write_progress("warmup", emit=True)
                while remaining:
                    chunk = min(remaining, 32)
                    self._collect(chunk)
                    remaining -= chunk
                    self._write_progress("warmup", emit=True)
            if not self.learner.normalizers_fitted:
                self._write_progress("fitting_normalizers")
                self.learner.fit_normalizers(self._sample(self.learner.config.fit_horizon))
            self._pretrain_world()
            self._pretrain_world_with_collection()
            for update in range(self.completed_updates, self.config.updates):
                self._write_progress("collecting")
                self._synchronize()
                started = time.perf_counter()
                metrics = self._collect(self.config.collection_steps)
                self._synchronize()
                collected = time.perf_counter()
                self._write_progress("fitting_world")
                world_metrics = []
                for _ in range(self.config.model_batches):
                    world_metrics.append(self.learner.fit_world(self._sample(self.learner.config.fit_horizon)))
                for key in world_metrics[0]:
                    metrics[key] = sum(record[key] for record in world_metrics) / len(world_metrics)
                available = self.replay.num_start_candidates(
                    self.learner.config.history_length, env_ids=self.training_envs)
                if available:
                    starts = self.replay.sample_starts(self.config.imagination_batch_size,
                        history_length=self.learner.config.history_length, device=self.learner.device,
                        generator=self.generator, env_ids=self.training_envs)
                    self._write_progress("improving_policy")
                    metrics.update(self.learner.improve_policy(starts))
                else:
                    # Preparation still supplies real dynamics data; it cannot
                    # supply a policy-owned imagination start until handoff.
                    metrics.update({"actor/update_skipped": 1., "imagination/start_candidates": 0.})
                self._synchronize()
                optimized = time.perf_counter()
                self._write_progress("validating")
                horizon = self.learner.config.imagination_horizon
                if self.replay.num_candidates(horizon, env_ids=self.validation_envs):
                    metrics.update(self.learner.validate_world(self._sample(horizon, validation=True)))
                else:
                    metrics["validation/long_window_available"] = 0
                    metrics.update(self.learner.validate_world(self._sample(self.learner.config.fit_horizon,
                                                                           validation=True)))
                self.completed_updates = update + 1
                metrics.update({"update": self.completed_updates,
                                "transitions": self.replay.total_added,
                                "timing/collection_s": collected - started,
                                "timing/learning_s": optimized - collected,
                                "timing/update_before_io_s": time.perf_counter() - started})
                if not all(math.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError("Nonfinite metrics; refusing to report a successful update")
                with (self.output / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(metrics, sort_keys=True) + "\n")
                if self.writer:
                    for key, value in metrics.items():
                        self.writer.add_scalar(key, value, self.completed_updates)
                if self.completed_updates % self.config.checkpoint_interval == 0:
                    self.save()
                if (self.config.evaluation_interval
                        and self.completed_updates % self.config.evaluation_interval == 0):
                    self._request_evaluation()
                self._write_progress("training", metrics=metrics, emit=(
                    self.completed_updates == 1
                    or self.completed_updates % self.config.checkpoint_interval == 0))
            self.save()
            self.replay.save(self.output / "replay.pt")
            if (self.config.evaluation_interval
                    and self.completed_updates % self.config.evaluation_interval != 0):
                self._request_evaluation()
            result = {"updates": self.completed_updates, "transitions": self.replay.total_added,
                      "world_optimizer_steps": self.learner.world_updates,
                      "actor_optimizer_steps": self.learner.actor_updates,
                      "critic_optimizer_steps": self.learner.critic_updates,
                      "initial_world_batches_completed": self.initial_world_batches_completed,
                      "world_pretrain_iterations_completed": self.world_pretrain_iterations_completed,
                      "world_pretrain_collection_steps_completed": self.world_pretrain_collection_steps_completed,
                      "latest_evaluation_request": self._latest_evaluation_request,
                      "training_wall_time_s": time.perf_counter() - run_started,
                      "timing_scope": "warmup, collection, learning, validation and checkpoint I/O; excludes simulator construction",
                      "source": self.environment.schema.source,
                      "checkpoint": str((self.output / "checkpoint.pt").resolve())}
            (self.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
            self._write_progress("complete", emit=True)
            return result
        except BaseException as error:
            try:
                self._write_progress("failed", error=error, emit=True)
            except Exception:
                pass
            raise
        finally:
            if self.writer:
                self.writer.close()

    def collect_dataset(self, steps: int) -> dict:
        """Capture data with frozen initialized/restored actor; perform no learning."""
        if type(steps) is not int or steps < 1:
            raise ValueError("Collection steps must be positive")
        if steps > self.config.replay_steps:
            raise ValueError("Increase replay_steps to retain every requested dataset step")
        self._run_started = time.perf_counter()
        try:
            self._write_progress("collecting_dataset", emit=True)
            collection_started = time.perf_counter()
            metrics = self._collect(steps)
            elapsed = time.perf_counter() - collection_started
            metrics["timing/collection_s"] = elapsed
            metrics["throughput/transitions_per_s"] = steps * self.environment.num_envs / elapsed
            self.replay.save(self.output / "replay.pt")
            self.save()
            self._write_progress("dataset_complete", metrics=metrics, emit=True)
            return {**metrics, "transitions": self.replay.total_added,
                    "actor_updates": self.learner.actor_updates,
                    "dataset": str((self.output / "replay.pt").resolve())}
        except BaseException as error:
            try:
                self._write_progress("failed", error=error, emit=True)
            except Exception:
                pass
            raise
        finally:
            if self.writer:
                self.writer.close()


def load_replay(path: Path) -> ReplayBuffer:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    schema = StateSchema.from_dict(payload["schema"])
    buffer = ReplayBuffer(schema, payload["num_envs"], payload["capacity"])
    buffer.load_state_dict(payload)
    return buffer
