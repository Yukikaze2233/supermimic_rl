#!/usr/bin/env python3
"""Per-skill native evaluation of a PPO actor on the fixed course task suite.

The factory config must contain an ``evaluation`` section (fixed scenarios,
e.g. ``{"evaluation_stage": "combined_dr", "seed": 7}`` for the full bank). The PPO
actor reads the native ``obs_buf`` history exactly as in PPO training, not the
collector's handoff-refilled history; actions default to its deterministic mean.
Optional PPO Gaussian sampling uses the checkpoint std and an independent RNG
for evaluation only; it does not change the actor or training distribution.
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

    def __init__(self, actor, environment, *, action_sampling="deterministic", sampling_seed=0):
        if action_sampling not in ("deterministic", "ppo-gaussian"):
            raise ValueError("Unknown evaluation action sampling mode")
        if type(sampling_seed) is not int or not 0 <= sampling_seed < 2**63:
            raise ValueError("Evaluation sampling seed must be an integer in [0, 2**63)")
        self.actor, self.environment, self._estimate = actor, environment, None
        if actor.features.configuration is not None:
            actor.features.validate_reference(environment.adapter.env._terrain_course.tasks.reference.config)
        self.action_sampling, self.sampling_seed = action_sampling, sampling_seed
        self._action_rng = (torch.Generator(device=actor.log_std.device).manual_seed(sampling_seed)
                            if action_sampling == "ppo-gaussian" else None)

    def eval(self):
        self.actor.eval()
        return self

    def _history(self):
        from supermimic_rl.ppo.policy import HISTORY_LENGTH
        from supermimic_rl.chassis_observation import chassis_frame

        native = self.environment.adapter.env
        frames = native.obs_buf["policy"].reshape(self.environment.num_envs, HISTORY_LENGTH, NATIVE_FRAME_DIM)
        return chassis_frame(frames, self.actor.chassis_command_contract)

    def __call__(self, history):
        native = self._history()
        self._estimate = self.actor.estimator(native)
        mean = self.actor(native, self._estimate)
        action = mean
        if self._action_rng is not None:
            # PPO samples Normal(mean, exp(log_std)) before the same hard clip.
            # A dedicated generator keeps action draws out of environment RNGs.
            action = torch.normal(mean, self.actor.log_std.exp().expand_as(mean), generator=self._action_rng)
        return self.actor.clip(action)

    def estimate(self, history):
        return self._estimate

    def sampling_report(self):
        return {"mode": self.action_sampling,
                "actor_features": self.actor.features.configuration,
                "seed": self.sampling_seed if self._action_rng is not None else None,
                "checkpoint_std": self.actor.log_std.detach().exp().cpu().tolist(),
                "std_semantics": "checkpoint Gaussian action units before clipping; unused in deterministic mode",
                "rng_scope": "independent action generator; environment generators unchanged"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--record-dataset", action="store_true",
                        help="Save this evaluation's complete compact V2 replay for world-model training")
    parser.add_argument("--action-sampling", choices=("deterministic", "ppo-gaussian"), default="deterministic")
    parser.add_argument("--sampling-seed", type=int, default=0,
                        help="Independent action RNG seed for --action-sampling ppo-gaussian")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.sampling_seed < 2**63:
        parser.error("--sampling-seed must be in [0, 2**63)")
    factory = json.loads(args.factory_config.read_text())
    if "evaluation" not in factory:
        parser.error("Per-skill evaluation needs the factory's fixed-scenario evaluation section")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new evaluation output directory")
    output.mkdir(parents=True, exist_ok=True)
    if args.record_dataset:
        factory["record_substeps"] = True
    else:
        # The native skill observers read physical ticks directly. An unused
        # world-model substep recorder adds packing work to every physics step.
        factory.update(record_substeps=False, substep_audit=False, record_disturbances=False)
    factory["native_log_directory"] = str(output / "native")
    (output / "factory-config.json").write_text(json.dumps(factory, indent=2) + "\n")

    from supermimic_rl.course.factory import create_course
    from supermimic_rl.adapters.packed import PackedEnvironment
    from supermimic_rl.cli import _native_schema_source, native_logs
    from supermimic_rl.ppo.trainer import load_actor
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

            policy = NativeHistoryActor(actor, environment, action_sampling=args.action_sampling,
                                        sampling_seed=args.sampling_seed)
            with torch.no_grad():
                if args.record_dataset:
                    # Recording is still subject to the world-model contract.
                    # PPO-only physics extensions are not silently encoded as
                    # a compatible V2 replay by the non-recording evaluator.
                    from supermimic_rl.world_model_v2.evaluation import evaluate_native

                    report = evaluate_native(environment, policy, args.steps, on_progress=progress,
                        record_dataset=output / "replay.pt",
                        recording_source={"kind": "own_ppo_native_evaluation", "checkpoint": str(args.checkpoint.resolve()),
                            "checkpoint_sha256": digest, "action_sampling": policy.sampling_report()})
                else:
                    from supermimic_rl.ppo.native_skill_evaluation import evaluate_native_policy

                    report = evaluate_native_policy(environment, policy, args.steps, on_progress=progress)
            if file_sha256(args.checkpoint) != digest:
                raise ValueError("Checkpoint changed during evaluation")
            report.update(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest,
                          policy_interface=("PPO deterministic mean on native obs_buf 5x35 history"
                                            if args.action_sampling == "deterministic" else
                                            "PPO checkpoint Gaussian sample then hard clip on native obs_buf 5x35 history"),
                          action_sampling=policy.sampling_report(),
                          native_source=json.loads(_native_schema_source(adapter.provenance)))
            if args.action_sampling == "ppo-gaussian":
                report["scope"] = ("native physics PPO with checkpoint Gaussian exploration only; "
                                   "no teacher, world model or gradient updates")
            (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({key: report[key] for key in ("steps", "environments", "failures_per_env_minute")}))
        except BaseException:
            (output / "failure-traceback.txt").write_text(traceback.format_exc())
            raise
        finally:
            environment.close()


if __name__ == "__main__":
    main()
