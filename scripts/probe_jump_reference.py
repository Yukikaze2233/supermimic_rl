#!/usr/bin/env python3
"""Compare short/long jump references with a bounded, gated native PPO probe.

Run one profile per fresh output directory through ``scripts/run_gpufree.sh``.
Both profiles resume the same full checkpoint, use 1024 x 24 collection and
retain the input factory's execution chain, task gates and assist selection.
Only the active-jump exposure budget is compressed; success gates still hold
the coordinate. Unsampled height bins are reported as unknown, never failures.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import traceback


PROFILES = {
    # Pinned native JumpReferenceConfig defaults, also with the R7 trigger gate.
    "short": {"compression_depth_max_m": .04, "compression_duration_s": .20,
              "nominal_extension_duration_s": .10},
    "long": {"compression_depth_max_m": .10, "compression_duration_s": .30,
             "nominal_extension_duration_s": .20},
}
NUM_ENVS, STEPS_PER_UPDATE = 1024, 24


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare_probe(factory_path: Path, config_path: Path, output: Path, profile: str,
                  stage_budget_updates: int) -> tuple[dict, dict]:
    """Pin derived probe inputs while preserving all non-probe settings."""
    if stage_budget_updates < 1:
        raise ValueError("The probe stage budget must be positive")
    factory = json.loads(factory_path.read_text())
    course_path = Path(factory["course_json"])
    if digest(course_path) != factory["course_sha256"]:
        raise ValueError("The source course changed before probe preparation")
    course = json.loads(course_path.read_text())
    if course.get("schedule") != "jump_first_v1":
        raise ValueError("The reference probe requires the jump-first course")
    budgets = course["base"]["stage_transition_budgets"]
    if len(budgets) != 8:
        raise ValueError("The probe requires the native eight-stage budget layout")
    # The base config is in canonical stage order; jump_first reorders index 5
    # into position 2. Changing this budget leaves earlier stage starts intact.
    budgets[5] = stage_budget_updates * NUM_ENVS * STEPS_PER_UPDATE
    derived_course = output / "probe-course.json"
    write_json(derived_course, course)
    factory.update(num_envs=NUM_ENVS, course_json=str(derived_course.resolve()),
                   course_sha256=digest(derived_course), native_log_directory=str(output / "native"))
    factory["reward_extensions"] = {**factory.get("reward_extensions", {}), "jump_missed": 1.}
    if "jump_assist" not in factory:
        raise ValueError("The reference probe requires an explicit training assist configuration")
    factory["jump_assist"] = {**factory["jump_assist"], "decay": "force"}
    factory["jump_takeoff"] = {**factory.get("jump_takeoff", {}), **PROFILES[profile]}
    config = json.loads(config_path.read_text())
    if config.get("steps_per_update", STEPS_PER_UPDATE) != STEPS_PER_UPDATE:
        raise ValueError("The probe must retain 24-step rollouts")
    config.update(course_rewind_stage="active_jump", course_rewind_tag="r7-reference-probe",
                  course_gate_resume="restart_hold", stop_after_gate=False)
    # Resuming supplies actor/critic/optimizer states. The old imitation schedule
    # has ended in R6; do not require its unrelated transfer artifact to exist.
    config.update(initial_actor=None, initial_actor_sha256=None,
                  imitation_actor=None, imitation_actor_sha256=None,
                  imitation_updates=0, imitation_anneal_updates=0)
    write_json(output / "factory-config.json", factory)
    write_json(output / "ppo-config.json", config)
    return factory, config


def summarize_probe(metrics_path: Path) -> dict:
    """Aggregate physical request outcomes; PPO metrics contain per-update deltas."""
    from supermimic_rl.course.jump_assist import RISE_BINS

    records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    prefix = "course/jump_assist/"
    bins = {}
    for name in RISE_BINS:
        entry = {}
        for cohort in ("", "unassisted_"):
            ended = sum(row.get(prefix + cohort + "ended_rise_" + name, 0) for row in records)
            success = sum(row.get(prefix + cohort + "success_rise_" + name, 0) for row in records)
            entry[cohort.rstrip("_") or "all"] = {
                "ended": ended, "physical_success": success,
                "success_rate": success / ended if ended else None,
            }
        bins[name] = entry
    return {"completed_probe_updates": len(records),
            "simulated_seconds_per_environment": len(records) * STEPS_PER_UPDATE * .02,
            "transitions": len(records) * NUM_ENVS * STEPS_PER_UPDATE,
            "height_bins_m": bins,
            "final_course": records[-1].get("course") if records else None,
            "final_assist_level": records[-1].get(prefix + "level") if records else None,
            "scope": "stochastic training outcomes, not the deterministic assist-off 27-scene evaluation"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--updates", type=int, default=400,
                        help="Additional PPO updates; 400 gives 192 simulated seconds per environment")
    parser.add_argument("--stage-budget-updates", type=int, default=400,
                        help="Probe-only active-jump budget; performance gates still hold progression")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.updates < 1 or args.stage_budget_updates < 1:
        parser.error("Update budgets must be positive")
    if digest(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("Probe checkpoint does not match its requested SHA256")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    factory, raw_config = prepare_probe(args.factory_config, args.config, output,
                                       args.profile, args.stage_budget_updates)

    from supermimic_rl.cli import native_logs
    from supermimic_rl.course.factory import create_course
    from supermimic_rl.ppo.environment import CourseEnvironment
    from supermimic_rl.ppo.trainer import PPOConfig, PPOTrainer

    with native_logs(output / "native"):
        environment = CourseEnvironment(create_course(factory))
        try:
            trainer = PPOTrainer(environment, PPOConfig.from_dict(raw_config), output / "training",
                                 resume=args.checkpoint)
            if digest(args.checkpoint) != args.checkpoint_sha256:
                raise ValueError("Probe checkpoint changed while loading")
            start = environment.course_stage_starts()["active_jump"]
            if trainer.course is None or trainer.course.coordinate != start:
                raise ValueError("Probe resume did not rewind to the active-jump stage start")
            trainer.config = replace(trainer.config, updates=trainer.updates + args.updates,
                                     curriculum_transition_cap=(
                                         start + args.stage_budget_updates * NUM_ENVS * STEPS_PER_UPDATE - 1))
            trainer.course.cap = trainer.config.curriculum_transition_cap
            # PPOTrainer now resets at the restored course coordinate and
            # reapplies stage std floors only when a new rewind tag takes effect.
            write_json(output / "ppo-config.json", asdict(trainer.config))
            write_json(output / "probe.json", {
                "profile": args.profile, "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": args.checkpoint_sha256, "resumed_update": trainer.updates,
                "coordinate": start, "stage_budget_updates": args.stage_budget_updates,
                "num_envs": NUM_ENVS, "steps_per_update": STEPS_PER_UPDATE,
                "requested_probe_updates": args.updates,
                "reference": factory["jump_takeoff"],
                "reward_extensions": factory["reward_extensions"], "assist": factory["jump_assist"],
            })
            result = trainer.train()
            report = {**result, "probe": summarize_probe(output / "training/metrics.jsonl")}
            write_json(output / "summary.json", report)
            print(json.dumps(report), flush=True)
        except BaseException:
            (output / "failure-traceback.txt").write_text(traceback.format_exc())
            raise
        finally:
            environment.close()


if __name__ == "__main__":
    main()
