"""A passed training gate is audited before any subsequent collection."""
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.ppo.trainer import PPOTrainer
from support.ppo import TargetEnvironment, config


spec = importlib.util.spec_from_file_location(
    "run_ppo_gated_course", Path(__file__).parents[1] / "scripts/run_ppo_gated_course.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class BirthEnvironment(TargetEnvironment):
    def __init__(self):
        super().__init__(num_envs=4)
        self.birth_coordinates = []

    def course_stage_starts(self):
        return {"flat": 0, "jump": 100, "terrain": 300}

    def reset(self):
        self.birth_coordinates.append(self.progress[-1][0] if self.progress else 0)
        return super().reset()


def gate_config(**overrides):
    values = dict(updates=6, steps_per_update=10, epochs=1, minibatches=1,
                  checkpoint_interval=100, stop_after_gate=True,
                  course_gates=[{"stage": "jump", "window": 1, "timeout_updates": 100,
                                 "criteria": [{"metric": "rms/forward_error_m_s", "max": 1e9}]}])
    values.update(overrides)
    return config(**values)


def test_gate_stop_saves_full_and_actor_before_next_collection_and_resumes_once(tmp_path):
    environment = BirthEnvironment()
    trainer = PPOTrainer(environment, gate_config(), tmp_path / "first")
    result = trainer.train()
    assert result["status"] == "gate_passed" and result["gate"] == "jump"
    assert trainer.updates == 4 and len(environment.progress) == 44
    assert max(coordinate for coordinate, _ in environment.progress) == 99
    full = torch.load(result["checkpoint"], weights_only=True)
    light = torch.load(result["actor_checkpoint"], weights_only=True)
    assert "optimizer" in full and "optimizer" not in light
    assert full["course_gates"]["opened"] == ["jump"]
    assert full["updates"] == light["updates"] == 4
    resumed_environment = BirthEnvironment()
    resumed = PPOTrainer(resumed_environment, gate_config(), tmp_path / "second",
                         resume=Path(result["checkpoint"]))
    assert resumed_environment.birth_coordinates == [0, full["course_gates"]["coordinate"]]
    assert resumed.train()["status"] == "complete"
    assert resumed.updates == 6
    assert torch.load(result["checkpoint"], weights_only=True)["updates"] == 4


def test_new_rewind_restarts_stage_exploration_but_gate_resume_does_not(tmp_path):
    seed = PPOTrainer(BirthEnvironment(), gate_config(), tmp_path / "seed")
    seed.train()
    seed.stage_action_std_applied = ["jump", "terrain"]
    seed.save(tmp_path / "seed.pt")
    settings = gate_config(course_rewind_stage="jump", course_rewind_tag="reward-fix",
                           stage_action_std={"jump": [.8] * 6})
    environment = BirthEnvironment()
    rewound = PPOTrainer(environment, settings, tmp_path / "rewound", resume=tmp_path / "seed.pt")
    assert environment.birth_coordinates[-1] == 100
    assert rewound.stage_action_std_applied == []
    assert rewound._raise_stage_action_std() == "jump"
    torch.testing.assert_close(rewound.actor.log_std.exp(), torch.full((6,), .8))
    rewound.save(tmp_path / "rewound.pt")
    continued = PPOTrainer(BirthEnvironment(), settings, tmp_path / "continued", resume=tmp_path / "rewound.pt")
    assert continued.stage_action_std_applied == ["jump"]
    assert continued._raise_stage_action_std() is None


@pytest.fixture
def run_inputs(tmp_path):
    runtime = tmp_path / "runtime.json"
    runner.write_json(runtime, {"task": {"action_bounds": [12.566] * 4 + [9, 9]},
                                "requested_randomization": {"push_linear_m_s": 0}})
    factory = {"num_envs": 4096, "runtime_json": str(runtime), "runtime_sha256": runner.sha256(runtime),
               "jump_takeoff": {}, "reward_extensions": {"jump_missed": 1}}
    training, evaluation = tmp_path / "training.json", tmp_path / "evaluation.json"
    runner.write_json(training, factory)
    runner.write_json(evaluation, {**factory, "num_envs": 64,
                                    "evaluation": {"evaluation_stage": "combined_dr"}})
    configuration, contract = tmp_path / "ppo.json", tmp_path / "contract.json"
    runner.write_json(configuration, {"stop_after_gate": True, "steps_per_update": 24,
                                       "course_gates": [{"stage": "jump"}]})
    runner.write_json(contract, {"requirements_by_gate": {"jump": [
        {"scenario": "required_case", "counter": "jump_physical_success_events"}]}})
    checkpoint = tmp_path / "seed.pt"
    checkpoint.write_bytes(b"seed")
    return SimpleNamespace(factory_config=training, evaluation_factory_config=evaluation, config=configuration,
                           evaluation_contract=contract, resume=checkpoint, output=tmp_path / "run",
                           root=tmp_path, continue_run=False)


def evaluation_report(checkpoint, success=1):
    buckets = {f"future_{index}": {"samples": 2000} for index in range(26)}
    buckets["required_case"] = {"samples": 2000, "jump_physical_success_events": success}
    return {"checkpoint_sha256": runner.sha256(checkpoint), "steps": 1000, "environments": 64,
            "task_suite": {"buckets": buckets, "plan": {"unassigned_scenarios": []}}}


def fake_launcher(calls, *, success=1, fail_first_evaluation=False):
    def launch(root, script, arguments, log):
        calls.append((script, arguments))
        output = Path(arguments[arguments.index("--output") + 1])
        if script == "train_ppo.py":
            index = sum(name == script for name, _ in calls)
            checkpoint = output / "training/checkpoint.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(f"policy-{index}".encode())
            runner.write_json(output / "summary.json", {
                "status": "gate_passed" if index == 1 else "complete", "updates": index * 10,
                **({"gate": "jump", "checkpoint": str(checkpoint)} if index == 1 else {})})
        else:
            if fail_first_evaluation and sum(name == script for name, _ in calls) == 1:
                raise subprocess.CalledProcessError(1, script)
            checkpoint = Path(arguments[arguments.index("--checkpoint") + 1])
            output.mkdir(parents=True)
            runner.write_json(output / "evaluation.json", evaluation_report(checkpoint, success))
    return launch


def test_orchestrator_evaluates_before_next_training_and_records_final_audit(run_inputs, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "launch", fake_launcher(calls))
    state = runner.run(run_inputs)
    assert [name for name, _ in calls] == ["train_ppo.py", "evaluate_ppo_skills.py"] * 2
    assert state["status"] == "budget_complete" and len(state["audits"]) == 2
    assert state["audits"][0]["status"] == "recorded"
    resume = calls[2][1][calls[2][1].index("--resume") + 1]
    assert resume == state["audits"][0]["checkpoint"]


def test_missing_current_skill_evidence_stops_without_next_collection(run_inputs, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "launch", fake_launcher(calls, success=0))
    state = runner.run(run_inputs)
    assert state["status"] == "diagnostic_required"
    assert len(calls) == 2 and state["audits"][0]["missing_physical_success"][0]["count"] == 0
    run_inputs.continue_run = True
    assert runner.run(run_inputs)["status"] == "diagnostic_required"
    assert len(calls) == 2


def test_process_failure_retains_pending_audit_and_explicit_continue_does_not_retrain(run_inputs, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "launch", fake_launcher(calls, fail_first_evaluation=True))
    with pytest.raises(subprocess.CalledProcessError):
        runner.run(run_inputs)
    pending = json.loads((run_inputs.output / "orchestration.json").read_text())["pending"]
    assert pending["gate"] == "jump"
    run_inputs.continue_run = True
    state = runner.run(run_inputs)
    assert state["status"] == "budget_complete"
    assert [name for name, _ in calls[:3]] == ["train_ppo.py", "evaluate_ppo_skills.py", "evaluate_ppo_skills.py"]
    assert sum(name == "train_ppo.py" for name, _ in calls) == 2
