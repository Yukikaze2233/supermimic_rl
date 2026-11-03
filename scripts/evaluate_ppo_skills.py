#!/usr/bin/env python3
"""Per-skill native evaluation of a PPO actor on the fixed course task suite.

The factory config must contain an ``evaluation`` section (fixed scenarios,
e.g. ``{"evaluation_stage": "combined_dr", "seed": 7}`` for all 27). The PPO
actor reads the native ``obs_buf`` history exactly as in PPO training, not the
collector's handoff-refilled history; actions are its deterministic mean.
Run under the Isaac launcher (SUPERMIMIC_SCRIPT=scripts/evaluate_ppo_skills.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import torch

NATIVE_FRAME_DIM = 42


class NativeHistoryActor:
    """Adapter for ``evaluate_native``: ignore the collector history, read the native one."""

    def __init__(self, actor, environment):
        self.actor, self.environment, self._estimate = actor, environment, None

    def eval(self):
        self.actor.eval()
        return self

    def _history(self):
        from supermimic_rl.ppo.policy import FRAME_DIM, HISTORY_LENGTH

        native = self.environment.adapter.env
        frames = native.obs_buf["policy"].reshape(self.environment.num_envs, HISTORY_LENGTH, NATIVE_FRAME_DIM)
        return frames[..., :FRAME_DIM]

    def __call__(self, history):
        native = self._history()
        self._estimate = self.actor.estimator(native)
        return self.actor.clip(self.actor(native, self._estimate))

    def estimate(self, history):
        return self._estimate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    factory = json.loads(args.factory_config.read_text())
    if "evaluation" not in factory:
        parser.error("Per-skill evaluation needs the factory's fixed-scenario evaluation section")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new evaluation output directory")
    output.mkdir(parents=True, exist_ok=True)
    factory["record_substeps"] = True
    factory["native_log_directory"] = str(output / "native")
    (output / "factory-config.json").write_text(json.dumps(factory, indent=2) + "\n")

    from supermimic_rl.course.factory import create_course
    from supermimic_rl.adapters.packed import PackedEnvironment
    from supermimic_rl.cli import _native_schema_source, native_logs
    from supermimic_rl.ppo.trainer import load_actor
    from supermimic_rl.world_model_v2.evaluation import evaluate_native
    from supermimic_rl.world_model_v2.teacher import file_sha256

    with native_logs(output / "native"):
        adapter = create_course(factory)
        environment = PackedEnvironment(adapter, source=_native_schema_source(adapter.provenance),
                                        policy_dt_s=adapter.policy_dt_s)
        try:
            environment.reset()
            digest = file_sha256(args.checkpoint)
            actor = load_actor(args.checkpoint, environment.device)
            if not torch.allclose(actor.action_bounds.cpu(), adapter.env._action_bounds.cpu()):
                raise ValueError("Checkpoint action bounds differ from the native runtime")

            def progress(record):
                with (output / "progress.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")

            with torch.no_grad():
                report = evaluate_native(environment, NativeHistoryActor(actor, environment), args.steps,
                                         on_progress=progress)
            if file_sha256(args.checkpoint) != digest:
                raise ValueError("Checkpoint changed during evaluation")
            report.update(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest,
                          policy_interface="PPO deterministic mean on native obs_buf 5x35 history",
                          native_source=json.loads(_native_schema_source(adapter.provenance)))
            (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({key: report[key] for key in ("steps", "environments", "failures_per_env_minute")}))
        except BaseException:
            (output / "failure-traceback.txt").write_text(traceback.format_exc())
            raise
        finally:
            environment.close()


if __name__ == "__main__":
    main()
