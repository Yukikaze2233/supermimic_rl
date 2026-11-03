#!/usr/bin/env python3
"""Alternate gated PPO segments and immutable assist-off native skill audits.

This supervisor uses ordinary Python; each child uses the Isaac launcher.
Evaluation/process failures stop the run. ``--continue-run`` explicitly resumes
a pending audit after an interrupted process; it never bypasses a failed skill
audit. A checkpoint with zero physical success in a required case needs a new
diagnosis/training decision, not an automatic retry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import traceback


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def validate_factories(training: dict, evaluation: dict) -> None:
    if training.get("num_envs") != 4096 or evaluation.get("num_envs") != 64:
        raise ValueError("The formal course requires 4096 training and 64 evaluation environments")
    selection = evaluation.get("evaluation", {})
    if (selection.get("evaluation_stage") != "combined_dr" or selection.get("scenarios")
            or selection.get("family") is not None or "jump_assist" in evaluation):
        raise ValueError("Evaluation must use the complete assist-off combined_dr suite")
    for key in ("expected_commit", "asset_manifest_sha256", "identified_bundle", "course_sha256",
                "jump_takeoff", "reward_extensions", "tof_stair_label"):
        if training.get(key) != evaluation.get(key):
            raise ValueError("Training/evaluation contracts differ: " + key)
    runtimes = []
    for factory in (training, evaluation):
        path = Path(factory["runtime_json"])
        if sha256(path) != factory["runtime_sha256"]:
            raise ValueError("The pinned runtime changed before launch")
        runtime = json.loads(path.read_text())
        # The fixed suite deliberately excludes training pushes/wrenches. All
        # execution, sensing, geometry and other DR settings must still match.
        for key in ("external_force_n", "external_torque_nm", "push_linear_m_s", "push_yaw_rad_s",
                    "push_interval_s", "wrench_interval_s"):
            runtime.get("requested_randomization", {}).pop(key, None)
        runtimes.append(runtime)
    if runtimes[0] != runtimes[1]:
        raise ValueError("Training/evaluation runtime differs beyond the fixed-suite push exclusions")


def assess_evaluation(report: dict, checkpoint_sha256: str, gate: str, contract: dict) -> dict:
    """Check exposure and the explicitly named, already required physical events."""
    if (report.get("checkpoint_sha256") != checkpoint_sha256 or report.get("steps") != 1000
            or report.get("environments") != 64):
        raise ValueError("Evaluation report does not match the immutable 64 x 1000 audit")
    suite = report.get("task_suite", {})
    buckets = suite.get("buckets", {})
    if (len(buckets) != 27 or suite.get("plan", {}).get("unassigned_scenarios")
            or any(bucket.get("samples", 0) <= 0 for bucket in buckets.values())):
        raise ValueError("Evaluation did not cover all 27 scenarios")
    requirements = contract["requirements_by_gate"]
    if gate != "complete" and gate not in requirements:
        raise ValueError("No evaluation contract for passed gate: " + gate)
    requested = []
    for name, cases in requirements.items():
        requested.extend(cases)
        if name == gate:
            break
    missing, evidence = [], []
    for item in requested:
        value = buckets.get(item["scenario"], {})
        for field in item["counter"].split("."):
            value = value.get(field) if isinstance(value, dict) else None
        entry = {**item, "count": value}
        evidence.append(entry)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            missing.append(entry)
    invalid = report.get("numerical_invalid_transitions", 0)
    return {"status": "diagnostic_required" if missing or invalid else "recorded",
            "gate": gate, "required_physical_evidence": evidence,
            "missing_physical_success": missing, "numerical_invalid_transitions": invalid,
            "coverage_note": contract.get("coverage_notes", {}).get(gate),
            "interpretation": "positive event evidence is not a success-rate qualification; "
                              "performance thresholds remain the explicit training gates"}


def launch(root: Path, script: str, arguments: list[str], log: Path) -> None:
    environment = os.environ.copy()
    environment.update(SUPERMIMIC_ROOT=str(root), SUPERMIMIC_SCRIPT=str(root / "scripts" / script))
    with log.open("w") as stream:
        subprocess.run(["bash", str(root / "scripts/run_gpufree.sh"), *arguments], env=environment,
                       cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True)


def run(args) -> dict:
    output, root = args.output.resolve(), args.root.resolve()
    inputs = {name: Path(getattr(args, name)).resolve()
              for name in ("factory_config", "evaluation_factory_config", "config", "evaluation_contract")}
    hashes = {name: sha256(path) for name, path in inputs.items()}
    training, evaluation = (json.loads(inputs[name].read_text())
                            for name in ("factory_config", "evaluation_factory_config"))
    validate_factories(training, evaluation)
    config = json.loads(inputs["config"].read_text())
    contract = json.loads(inputs["evaluation_contract"].read_text())
    if not config.get("stop_after_gate") or config.get("steps_per_update") != 24:
        raise ValueError("The formal PPO config must stop after each gate and collect 24 steps")
    gate_names = [gate["stage"] + (f"@{gate['at_fraction']:g}" if gate.get("at_fraction") else "")
                  for gate in config["course_gates"]]
    if list(contract["requirements_by_gate"]) != gate_names:
        raise ValueError("Evaluation requirements must explicitly cover the configured gates in course order")
    state_path = output / "orchestration.json"
    if args.continue_run:
        state = json.loads(state_path.read_text())
        if state["input_sha256"] != hashes:
            raise ValueError("Explicit continuation requires unchanged pinned orchestration inputs")
        if state["status"] in ("budget_complete", "gate_timeout", "diagnostic_required"):
            return state
    else:
        output.mkdir(parents=True, exist_ok=False)
        state = {"status": "running", "input_sha256": hashes, "checkpoint": str(args.resume.resolve()),
                 "segment": 0, "audits": [], "pending": None, "last_update": -1}
        write_json(state_path, state)
    try:
        # At most one segment per gate plus the final budget segment. This is
        # progression through distinct gates, not retrying a failed process.
        for _ in range(len(config["course_gates"]) + 1):
            pending = state["pending"]
            if pending is None:
                segment = output / f"segment-{state['segment']:03d}"
                if segment.exists() and not (segment / "summary.json").is_file():
                    raise RuntimeError("Interrupted training needs an explicit checkpoint decision; segment exists")
                if not segment.exists():
                    launch(root, "train_ppo.py", ["train", "--factory-config", str(inputs["factory_config"]),
                           "--config", str(inputs["config"]), "--resume", state["checkpoint"],
                           "--output", str(segment)], output / f"train-{state['segment']:03d}.log")
                result = json.loads((segment / "summary.json").read_text())
                if result["status"] == "gate_timeout":
                    state.update(status="gate_timeout", training=result,
                                 checkpoint=str((segment / "training/checkpoint.pt").resolve()))
                    write_json(state_path, state)
                    return state
                if result["status"] not in ("gate_passed", "complete") or result["updates"] <= state["last_update"]:
                    raise ValueError("PPO returned an unexpected or non-advancing boundary")
                checkpoint = (Path(result["checkpoint"]) if result["status"] == "gate_passed"
                              else segment / "training/checkpoint.pt")
                pending = {"gate": result.get("gate", "complete"), "checkpoint": str(checkpoint.resolve()),
                           "checkpoint_sha256": sha256(checkpoint), "updates": result["updates"], "attempt": 0}
                state.update(pending=pending, last_update=result["updates"])
                write_json(state_path, state)
            checkpoint = Path(pending["checkpoint"])
            if sha256(checkpoint) != pending["checkpoint_sha256"]:
                raise ValueError("The pending evaluation checkpoint changed")
            audit_output = output / f"audit-{pending['updates']:06d}-{pending['attempt']:02d}"
            pending["attempt"] += 1
            state["status"] = "evaluating"
            write_json(state_path, state)
            launch(root, "evaluate_ppo_skills.py", ["--factory-config", str(inputs["evaluation_factory_config"]),
                   "--checkpoint", str(checkpoint), "--steps", "1000", "--output", str(audit_output)],
                   output / (audit_output.name + ".log"))
            report_path = audit_output / "evaluation.json"
            audit = assess_evaluation(json.loads(report_path.read_text()), pending["checkpoint_sha256"],
                                      pending["gate"], contract)
            audit.update(report=str(report_path), checkpoint=str(checkpoint),
                         checkpoint_sha256=pending["checkpoint_sha256"], updates=pending["updates"])
            state["audits"].append(audit)
            if audit["status"] == "diagnostic_required":
                state["status"] = "diagnostic_required"
                write_json(state_path, state)
                return state
            state.update(checkpoint=str(checkpoint), pending=None,
                         status="budget_complete" if pending["gate"] == "complete" else "running",
                         segment=state["segment"] + 1)
            write_json(state_path, state)
            if state["status"] == "budget_complete":
                return state
        raise RuntimeError("More gate stops than configured; inspect the course state before continuing")
    except BaseException:
        state["status"] = "process_failed"
        write_json(state_path, state)
        (output / "failure-traceback.txt").write_text(traceback.format_exc())
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("factory-config", "evaluation-factory-config", "config", "evaluation-contract", "resume", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--continue-run", action="store_true")
    result = run(parser.parse_args(argv))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
