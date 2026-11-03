#!/usr/bin/env python3
"""PPO teacher/baseline entry: train (fresh or resume), evaluate and export ONNX.

train and evaluate build the native course through the pinned factory, so they
must run under the Isaac launcher (scripts/run_ppo_gpufree.sh). export runs on
plain CPU PyTorch.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import sys
import traceback

import torch


def _native_environment(factory_config_path: Path, output: Path, factory: str):
    from supermimic_rl.ppo.environment import CourseEnvironment

    configuration = json.loads(factory_config_path.read_text())
    logs = (output / "native").resolve()
    logs.mkdir(parents=True, exist_ok=True)
    configuration["native_log_directory"] = str(logs)
    (output / "factory-config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    module, name = factory.split(":", 1)
    return CourseEnvironment(getattr(importlib.import_module(module), name)(configuration))


def _new_output(path: Path, *, resume: bool) -> Path:
    path = path.resolve()
    if not resume and path.exists() and any(path.iterdir()):
        raise FileExistsError("Use a new output directory or --resume")
    path.mkdir(parents=True, exist_ok=True)
    return path


def train(args):
    from supermimic_rl.cli import native_logs
    from supermimic_rl.ppo.trainer import PPOConfig, PPOTrainer

    output = _new_output(args.output, resume=args.resume is not None)
    config = PPOConfig.from_dict(json.loads(args.config.read_text()))
    (output / "ppo-config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    with native_logs(output / "native"):
        environment = _native_environment(args.factory_config, output, args.factory)
        try:
            trainer = PPOTrainer(environment, config, output / "training", resume=args.resume)
            result = trainer.train()
            # Isaac shutdown may terminate the interpreter; persist results first.
            (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
        except BaseException:
            # Isaac close may exit the process before Python reports the error.
            # Persist the original exception, then retain normal propagation.
            (output / "failure-traceback.txt").write_text(traceback.format_exc())
            raise
        finally:
            environment.close()


def evaluate(args):
    from supermimic_rl.cli import native_logs, _native_schema_source
    from supermimic_rl.ppo.evaluation import evaluate as run
    from supermimic_rl.ppo.trainer import load_actor
    from supermimic_rl.world_model_v2.teacher import file_sha256

    output = _new_output(args.output, resume=False)
    with native_logs(output / "native"):
        environment = _native_environment(args.factory_config, output, args.factory)
        try:
            checkpoint_sha256 = file_sha256(args.checkpoint)
            decision = None
            source = None
            if getattr(args, "transferred_sha256", None):
                from supermimic_rl.transferred_actor import TransferredActor

                payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
                actor = TransferredActor(payload["recipe"]["policy"])
                source = actor.load_initial_checkpoint(args.checkpoint, args.transferred_sha256)
                actor.to(environment.device).eval().requires_grad_(False)
                decision = actor.action_and_estimate
            else:
                actor = load_actor(args.checkpoint, environment.device)

            def progress(record):
                with (output / "progress.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")

            report = run(environment, actor, args.steps, decision=decision, on_progress=progress)
            if file_sha256(args.checkpoint) != checkpoint_sha256:
                raise ValueError('PPO checkpoint changed during native evaluation; evaluate an immutable checkpoint')
            report["checkpoint"] = str(args.checkpoint.resolve())
            report['checkpoint_sha256'] = checkpoint_sha256
            report['native_source'] = json.loads(_native_schema_source(environment.adapter.provenance))
            native = getattr(environment, "env", None)
            composer = getattr(getattr(native, "_formal_reward", None), "composer", None)
            audit = getattr(composer, "shared_kernel_audit", None)
            if audit is not None:
                report["reward_kernel_audit"] = audit.summary()
            if source is not None:
                report["transferred_source"] = source
            (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2), flush=True)
        except BaseException:
            # Isaac close may exit the process before Python reports the error.
            # Persist the original exception, then retain normal propagation.
            (output / "failure-traceback.txt").write_text(traceback.format_exc())
            raise
        finally:
            environment.close()


def export(args):
    from supermimic_rl.ppo.evaluation import export_onnx
    from supermimic_rl.ppo.trainer import load_actor

    print(export_onnx(load_actor(args.checkpoint), args.output))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--factory-config", type=Path, required=True)
        command.add_argument("--factory", default="supermimic_rl.course.factory:create_course")
        command.add_argument("--output", type=Path, required=True)
    commands.choices["train"].add_argument("--config", type=Path, required=True)
    commands.choices["train"].add_argument("--resume", type=Path)
    commands.choices["evaluate"].add_argument("--checkpoint", type=Path, required=True)
    commands.choices["evaluate"].add_argument("--steps", type=int, default=1000)
    commands.choices["evaluate"].add_argument(
        "--transferred-sha256", help="Evaluate a pinned native Transformer checkpoint using its own estimator")
    export_command = commands.add_parser("export")
    export_command.add_argument("--checkpoint", type=Path, required=True)
    export_command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    {"train": train, "evaluate": evaluate, "export": export}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
