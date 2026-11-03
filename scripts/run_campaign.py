"""Run one native learning trajectory with sequential checkpoint evaluations."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def run(args):
    milestones = args.milestones
    native_restart_limit = getattr(args, "native_restarts", 0)
    initial_actor = getattr(args, "initial_actor", None)
    initial_actor_sha256 = getattr(args, "initial_actor_sha256", None)
    if (initial_actor is None) != (initial_actor_sha256 is None):
        raise ValueError("Initial actor checkpoint path and SHA256 must be provided together")
    if initial_actor is not None:
        if args.resume is not None:
            raise ValueError("Choose either a full resume or an initial actor checkpoint")
        if (not isinstance(initial_actor_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", initial_actor_sha256) is None):
            raise ValueError("Initial actor checkpoint requires an explicit lowercase SHA256")
        initial_actor = Path(initial_actor).resolve(strict=True)
        if not initial_actor.is_file():
            raise FileNotFoundError(initial_actor)
    if (not milestones or milestones != sorted(set(milestones))
            or milestones[0] < 1 or args.evaluation_steps < 1
            or type(native_restart_limit) is not int or native_restart_limit < 0):
        raise ValueError("Milestones must be positive and increasing")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new campaign output directory")
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    factory = json.loads(args.factory_config.read_text())
    launcher = args.launcher.resolve(strict=True)
    factory_path = output / "factory.json"
    write_json(factory_path, factory)
    write_json(output / "configuration.json", config)
    checkpoint = None if args.resume is None else args.resume.resolve(strict=True)
    status = {
        "format": "supermimic-campaign-v1", "pid": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "milestones": milestones, "evaluation_steps": args.evaluation_steps,
        "num_envs": factory["num_envs"], "phase": "starting", "completed_stages": [],
        "native_restart_limit": native_restart_limit, "native_restarts": [],
        "initial_actor": (None if initial_actor is None else {
            "path": str(initial_actor), "sha256": initial_actor_sha256}),
    }

    def publish():
        status["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "status.json", status)

    def execute(command, directory, phase):
        status.update(phase=phase, active_directory=str(directory), child_pid=None)
        publish()
        with directory.with_suffix(".launcher.log").open("w") as log:
            process = subprocess.Popen(["bash", str(launcher), *command],
                                       stdout=log, stderr=subprocess.STDOUT)
            status["child_pid"] = process.pid
            publish()
            code = process.wait()
        status["child_pid"] = None
        failure = directory / "failure.json"
        if failure.exists():
            detail = json.loads(failure.read_text())
            raise RuntimeError(f"{phase} failed: {detail.get('type')}: {detail.get('message')}")
        if code:
            raise RuntimeError(f"{phase} exited with code {code}; see launcher/native logs")

    def recover_native_collection(directory):
        """Cold-rebuild only after a diagnosed native joint failure and safe save.

        The incomplete vector interval never enters replay. Model/optimizer
        failures and failed evaluations are not eligible for automatic recovery.
        """
        if len(status["native_restarts"]) >= native_restart_limit:
            return None
        failure_path = directory / "failure.json"
        progress_path = directory / "training/progress.json"
        if not failure_path.is_file() or not progress_path.is_file():
            return None
        failure = json.loads(failure_path.read_text())
        progress = json.loads(progress_path.read_text())
        recovery = progress.get("recovery") or {}
        messages = {
            "continuous joint state is nonfinite or angular sampling is ambiguous",
            "continuous joint position increment is ambiguous",
        }
        candidate = directory / "training/failure-checkpoint.pt"
        if (failure.get("type") != "RuntimeError" or failure.get("message") not in messages
                or progress.get("stage") != "failed"
                or recovery.get("status") != "recoverable"
                or recovery.get("checkpoint") != str(candidate.resolve())
                or not candidate.is_file()):
            return None
        status["native_restarts"].append({
            "at": datetime.now(timezone.utc).isoformat(),
            "directory": str(directory), "failure": str(failure_path),
            "completed_updates": recovery["completed_updates"],
            "transitions": recovery["transitions"], "checkpoint": str(candidate.resolve()),
            "semantics": "incomplete vector interval discarded; learner/replay retained; simulator cold reset",
        })
        status["phase"] = "restarting_native"
        publish()
        return candidate.resolve()

    try:
        for target in milestones:
            status["target_update"] = target
            stage_config = {**config, "run": {**config["run"], "updates": target}}
            config_path = output / f"config-{target:06d}.json"
            write_json(config_path, stage_config)
            attempt = 0
            while True:
                suffix = "" if attempt == 0 else f"-retry-{attempt:02d}"
                train = output / f"train-{target:06d}{suffix}"
                command = ["train", "--config", str(config_path), "--factory-config", str(factory_path),
                           "--output", str(train)]
                if checkpoint is not None:
                    command += ["--resume", str(checkpoint)]
                elif initial_actor is not None:
                    command += ["--initial-actor", str(initial_actor),
                                "--initial-actor-sha256", initial_actor_sha256]
                try:
                    execute(command, train, "training")
                except RuntimeError:
                    recovered = recover_native_collection(train)
                    if recovered is None:
                        raise
                    checkpoint = recovered
                    attempt += 1
                    continue
                break
            # Kit may exit 0 during cleanup; completion must have been written
            # by the learner before closing the native application.
            summary = json.loads((train / "training/summary.json").read_text())
            if summary["updates"] != target:
                raise RuntimeError("Training did not finish the requested milestone")
            checkpoint = (train / "training/checkpoint.pt").resolve(strict=True)
            evaluation = output / f"evaluate-{target:06d}"
            execute(["evaluate", "--checkpoint", str(checkpoint),
                     "--factory-config", str(factory_path), "--output", str(evaluation),
                     "--steps", str(args.evaluation_steps)], evaluation, "evaluating")
            report = json.loads((evaluation / "evaluation/evaluation.json").read_text())
            if report["checkpoint_updates"] != target:
                raise RuntimeError("Evaluation did not use the requested checkpoint")
            status["completed_stages"].append({
                "update": target, "checkpoint": str(checkpoint),
                "transitions": summary["transitions"],
                "training_wall_time_s": summary["training_wall_time_s"],
                "evaluation": str(evaluation / "evaluation/evaluation.json"),
                "mean_step_reward": report["mean_step_reward"],
                "terminated_episodes": report["terminated_episodes"],
                "failure_free_environments": report["failure_free_environments"],
                "tracking": report["tracking"],
            })
            publish()
        status.update(phase="complete", active_directory=None)
        publish()
        return status
    except BaseException as error:
        status.update(phase="failed", error=f"{type(error).__name__}: {error}")
        publish()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--factory-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--milestones", type=int, nargs="+", default=[250, 1000, 3000])
    parser.add_argument("--evaluation-steps", type=int, default=1000)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initial-actor", type=Path,
                        help="Pinned upstream actor for the first stage only; not used for resume")
    parser.add_argument("--initial-actor-sha256",
                        help="Required SHA256 pin for --initial-actor")
    parser.add_argument("--native-restarts", type=int, default=0,
                        help="Cold restarts after safely checkpointed native joint failures")
    parser.add_argument("--launcher", type=Path,
                        default=Path(__file__).with_name("run_native_kaiser.sh"))
    args = parser.parse_args(argv)
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
