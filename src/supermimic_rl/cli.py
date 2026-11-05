"""Explicit offline, synthetic-fixture and native training entry points."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import sys
import traceback

import torch

from .adapters.packed import PackedEnvironment
from .adapters.synthetic import SyntheticEnvironment
from .contracts import ACTION_BOUNDS
from .evaluation import evaluate
from .learning import Learner, LearningConfig
from .runtime import RunConfig, TrainingRun, atomic_save, load_replay


@contextmanager
def native_logs(directory: Path):
    """Native PhysX writes to file descriptors, not just Python's streams."""
    directory.mkdir(parents=True, exist_ok=True)
    original = [os.dup(1), os.dup(2)]
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        with (directory / "native.stdout.log").open("ab", buffering=0) as stdout, \
                (directory / "native.stderr.log").open("ab", buffering=0) as stderr:
            os.dup2(stdout.fileno(), 1)
            os.dup2(stderr.fileno(), 2)
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        for source, destination in zip(original, (1, 2)):
            os.dup2(source, destination)
            os.close(source)


def _configuration(path):
    payload = json.loads(Path(path).read_text())
    if set(payload) != {"run", "learning"}:
        raise ValueError("Configuration must contain exactly run and learning")
    return RunConfig(**payload["run"]), LearningConfig.from_dict(payload["learning"])


def _native_schema_source(provenance):
    """Separate authenticated configuration from full-vehicle SDK readback.

    The complete receipt remains execution evidence, but its floating-point
    masses, COM quaternion signs and tensor-byte hashes are not an interface
    identity. Source pins and the explicit gimbal coordinate/control contract
    remain identity inputs. Legacy factories retain their exact source string.
    """
    # These receipts describe row rejection, not dynamics of retained data.
    # Keep them in the full collection receipt without changing the physical
    # schema of earlier valid replay. Unknown recovery policies remain pinned.
    isolation = provenance.get("numerical_isolation", {})
    if (isolation.get("contract") in (
            "physx_finite_coordinate_increment_row_isolation_v1",
            "physx_finite_coordinate_sampling_row_isolation_v2")
            and isolation.get("native_state_modified") is False):
        provenance = {key: value for key, value in provenance.items() if key != "numerical_isolation"}
    if provenance.get("course_schedule", False) is None:
        provenance = {key: value for key, value in provenance.items() if key != "course_schedule"}
    # Staggered first time limits schedule resets; they do not change the dynamics of recorded data.
    provenance = {key: value for key, value in provenance.items() if key != "desynchronize_episodes"}
    # The legacy bound is already implied by every earlier runtime pin; only a
    # changed action contract becomes part of the identity.
    if provenance.get("action_bounds") == list(ACTION_BOUNDS):
        provenance = {key: value for key, value in provenance.items() if key != "action_bounds"}
    # PhysX keeps its native coordinate reader and reports no receipt here.
    if provenance.get("continuous_coordinates") is not None:
        # Floating-point guard evidence does not change the encoder coordinate
        # ABI. Keep period, correction mode, source and clock in the identity.
        coordinates = dict(provenance["continuous_coordinates"])
        coordinates.pop("half_turn_roundoff_rad", None)
        provenance = {**provenance, "continuous_coordinates": coordinates}
    if "execution" in provenance:
        execution = dict(provenance["execution"])
        # The distribution remains pinned in randomization. Its sampled extrema
        # vary with evaluation batch size and do not define the model interface.
        execution.pop("wheel_inertia_scale_range", None)
        provenance = {**provenance, "execution": execution}
    if provenance.get("physics_backend") == "kamino":
        backend = provenance.get("backend")
        if not isinstance(backend, dict) or not isinstance(backend.get("identity"), dict):
            raise ValueError("Kamino provenance requires its stable backend identity")
        provenance = {**provenance, "backend": {"identity": backend["identity"]}}
    if "full_vehicle" in provenance:
        receipt = provenance["full_vehicle"]
        fields = (
            "asset_manifest_sha256", "core_body_names", "core_joint_names",
            "auxiliary_body_names", "auxiliary_joint_names", "nominal_masses",
            "requested_startup_configuration", "gimbal_control",
        )
        if not isinstance(receipt, dict) or any(name not in receipt for name in fields):
            raise ValueError("Full-vehicle provenance requires its authenticated source/control contract")
        provenance = {
            **provenance,
            "full_vehicle": {"schema_identity_version": 1,
                             **{name: receipt[name] for name in fields}},
        }
    return json.dumps(provenance, sort_keys=True, separators=(",", ":"))


def _parser():
    parser = argparse.ArgumentParser(description="Wheel-leg differentiable world-model learning")
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke", help="Run a CPU integration fixture; not robot physics")
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--updates", type=int, default=3)
    smoke.add_argument("--seed", type=int, default=7)
    smoke.add_argument("--threads", type=int, default=1)
    for command, help_text in (("train", "Launch the pinned native simulator and online learning"),
                               ("collect", "Capture native transitions with a frozen actor; no learning")):
        train = commands.add_parser(command, help=help_text)
        train.add_argument("--config", type=Path, required=True)
        train.add_argument("--factory-config", type=Path, required=True)
        train.add_argument("--factory", default="supermimic_rl.adapters.factory:create_isaac")
        train.add_argument("--output", type=Path, required=True)
        train.add_argument("--resume", type=Path)
        train.add_argument("--initial-world", type=Path)
        train.add_argument("--initial-actor", type=Path,
                           help="Initialize a transferred Transformer from an upstream actor checkpoint")
        train.add_argument("--initial-actor-sha256",
                           help="Required SHA256 pin for --initial-actor; omit both when resuming")
        if command == "collect":
            train.add_argument("--steps", type=int, required=True)
    offline = commands.add_parser("fit-world", help="Fit/validate dynamics from a captured replay; no simulator")
    offline.add_argument("--replay", type=Path, required=True)
    offline.add_argument("--config", type=Path, required=True)
    offline.add_argument("--output", type=Path, required=True)
    offline.add_argument("--device", default="cpu")
    inspect = commands.add_parser("inspect-replay", help="Inspect schema and contiguous window availability")
    inspect.add_argument("path", type=Path)
    inspect.add_argument("--horizon", type=int, default=8)
    evaluation = commands.add_parser("evaluate", help="Evaluate a frozen checkpoint in its native environment")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--factory-config", type=Path, required=True)
    evaluation.add_argument("--factory", default="supermimic_rl.adapters.factory:create_isaac")
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--steps", type=int, required=True)
    return parser


def _smoke(args):
    if args.threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    environment = SyntheticEnvironment(8, seed=args.seed, episode_steps=64)
    config = RunConfig(seed=args.seed, updates=args.updates, warmup_steps=32, replay_steps=96,
                       collection_steps=4, model_batches=3, model_batch_size=32,
                       imagination_batch_size=16, validation_batch_size=16, checkpoint_interval=1)
    learning = LearningConfig(model_hidden=(32, 32), actor_hidden=(32,), critic_hidden=(32,),
                               history_length=5, fit_horizon=4, imagination_horizon=8,
                               critic_epochs=1, critic_minibatches=2)
    try:
        return TrainingRun(environment, args.output, config, learning).run()
    finally:
        environment.close()


def _train(args):
    config, learning = _configuration(args.config)
    TrainingRun.validate_initialization(
        learning, resume=args.resume, initial_world=args.initial_world,
        initial_actor=args.initial_actor, initial_actor_sha256=args.initial_actor_sha256)
    factory_config = json.loads(args.factory_config.read_text())
    module, name = args.factory.split(":", 1)
    factory = getattr(importlib.import_module(module), name)
    output = args.output.resolve()
    if args.resume is None and output.exists() and any(output.iterdir()):
        raise FileExistsError("Native runs require a new output directory")
    logs = output / "native"
    logs.mkdir(parents=True, exist_ok=True)
    configured_logs = factory_config.get("native_log_directory")
    if configured_logs is not None and Path(configured_logs).resolve() != logs:
        raise ValueError("native_log_directory must be the selected output/native directory")
    factory_config["native_log_directory"] = str(logs)
    (output / "factory-config.json").write_text(json.dumps(factory_config, indent=2) + "\n")
    with native_logs(logs):
        adapter = factory(factory_config)
        try:
            provenance = getattr(adapter, "provenance", {})
            if not provenance:
                raise ValueError("A native factory must provide source/asset/control provenance")
            (output / "native-provenance.json").write_text(
                json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
            identity = _native_schema_source(provenance)
            environment = PackedEnvironment(adapter, source=identity, policy_dt_s=adapter.policy_dt_s)
            run = TrainingRun(environment, output / "training", config, learning,
                              resume=args.resume, initial_world=args.initial_world,
                              initial_actor=args.initial_actor,
                              initial_actor_sha256=args.initial_actor_sha256)
            return run.collect_dataset(args.steps) if args.command == "collect" else run.run()
        except BaseException as error:
            traceback.print_exc()
            (output / "failure.json").write_text(json.dumps({
                "type": type(error).__name__, "message": str(error),
                "traceback": traceback.format_exc(),
            }, ensure_ascii=False, indent=2) + "\n")
            raise
        finally:
            adapter.close()


def _fit_world(args):
    config, learning = _configuration(args.config)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Offline output directory must be new")
    args.output.mkdir(parents=True, exist_ok=True)
    replay = load_replay(args.replay)
    if replay.num_envs < 2:
        raise ValueError("Offline validation requires separate source environments")
    heldout = max(1, min(replay.num_envs - 1, round(replay.num_envs * config.validation_fraction)))
    train_envs = list(range(replay.num_envs - heldout))
    validation_envs = list(range(replay.num_envs - heldout, replay.num_envs))
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed + 1)
    learner = Learner(replay.schema, learning, device=args.device)
    def sample(horizon, *, validation=False):
        return replay.sample(config.model_batch_size, horizon, device=args.device, generator=generator,
                             env_ids=validation_envs if validation else train_envs)
    learner.fit_normalizers(sample(learning.fit_horizon))
    for update in range(config.updates):
        metrics = learner.fit_world(sample(learning.fit_horizon))
        horizon = learning.imagination_horizon
        if replay.num_candidates(horizon, env_ids=validation_envs) == 0:
            horizon = learning.fit_horizon
        metrics.update(learner.validate_world(sample(horizon, validation=True)))
        metrics["update"] = update + 1
        with (args.output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
    atomic_save({"format": "supermimic-rl-world-fit-v1", "learner": learner.state_dict(),
                 "training_envs": train_envs, "validation_envs": validation_envs,
                 "config": {"run": asdict(config), "learning": asdict(learning)}},
                args.output / "world-model.pt")
    return {"updates": learner.world_updates, "actor_updates": learner.actor_updates,
            "source": replay.schema.source, "checkpoint": str((args.output / "world-model.pt").resolve())}


def _evaluate(args):
    if args.steps < 1:
        raise ValueError("Evaluation steps must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Evaluation requires a new output directory")
    factory_config = json.loads(args.factory_config.read_text())
    module, name = args.factory.split(":", 1)
    factory = getattr(importlib.import_module(module), name)
    logs = output / "native"
    logs.mkdir(parents=True, exist_ok=True)
    # Log paths are operational, not part of the checkpoint's source identity.
    factory_config["native_log_directory"] = str(logs)
    (output / "factory-config.json").write_text(json.dumps(factory_config, indent=2) + "\n")
    with native_logs(logs):
        adapter = None
        try:
            adapter = factory(factory_config)
            provenance = getattr(adapter, "provenance", {})
            if not provenance:
                raise ValueError("Native evaluation requires source/asset/control provenance")
            (output / "native-provenance.json").write_text(
                json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
            identity = _native_schema_source(provenance)
            environment = PackedEnvironment(adapter, source=identity, policy_dt_s=adapter.policy_dt_s)
            return evaluate(environment, args.checkpoint, output / "evaluation", steps=args.steps)
        except BaseException as error:
            traceback.print_exc()
            (output / "failure.json").write_text(json.dumps({
                "type": type(error).__name__, "message": str(error),
                "traceback": traceback.format_exc(),
            }, ensure_ascii=False, indent=2) + "\n")
            raise
        finally:
            if adapter is not None:
                adapter.close()


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        if args.command == "smoke":
            result = _smoke(args)
        elif args.command in ("train", "collect"):
            result = _train(args)
        elif args.command == "fit-world":
            result = _fit_world(args)
        elif args.command == "evaluate":
            result = _evaluate(args)
        else:
            replay = load_replay(args.path)
            result = {"schema": replay.schema.to_dict(), "schema_hash": replay.schema.fingerprint,
                      "stored_transitions": len(replay), "total_transitions": replay.total_added,
                      "eligible_windows": replay.num_candidates(args.horizon),
                      "eligible_actor_starts": replay.num_start_candidates(5)}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, RuntimeError, FileExistsError, FileNotFoundError, FloatingPointError) as error:
        print(f"supermimic-rl: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
