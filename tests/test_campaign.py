"""Exercise process sequencing and completion evidence without starting Isaac."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def campaign(tmp_path):
    path = Path(__file__).parents[1] / "scripts/run_campaign.py"
    spec = importlib.util.spec_from_file_location("native_campaign", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config, factory, launcher = (tmp_path / name for name in ("config.json", "factory.json", "launch.sh"))
    config.write_text(json.dumps({"run": {"updates": 4}, "learning": {}}))
    factory.write_text(json.dumps({"num_envs": 2}))
    launcher.write_text("#!/bin/sh\n")
    args = SimpleNamespace(config=config, factory_config=factory, launcher=launcher,
                           output=tmp_path / "run", milestones=[2, 4], evaluation_steps=10, resume=None)
    return module, args


@pytest.mark.parametrize("first_native_failure", [False, True])
@pytest.mark.parametrize("teacher_initialization", [False, True])
def test_campaign_trains_evaluates_and_resumes_one_trajectory(
        campaign, monkeypatch, first_native_failure, teacher_initialization):
    module, args = campaign
    args.native_restarts = 1
    if teacher_initialization:
        args.initial_actor = args.config.parent / "teacher.pt"
        args.initial_actor.write_bytes(b"upstream actor fixture")
        args.initial_actor_sha256 = "a" * 64
    commands = []

    class Process:
        pid = 123

        def __init__(self, command, **kwargs):
            commands.append(command)
            output = Path(command[command.index("--output") + 1])
            if first_native_failure and len(commands) == 1:
                folder = output / "training"
                folder.mkdir(parents=True)
                checkpoint = folder / "failure-checkpoint.pt"
                checkpoint.write_bytes(b"completed learner and replay")
                (output / "failure.json").write_text(json.dumps({
                    "type": "RuntimeError", "message": "continuous joint position increment is ambiguous",
                }))
                (folder / "progress.json").write_text(json.dumps({
                    "stage": "failed", "recovery": {
                        "status": "recoverable", "checkpoint": str(checkpoint.resolve()),
                        "completed_updates": 1, "transitions": 10,
                    },
                }))
                return
            if command[2] == "train":
                target = json.loads(Path(command[command.index("--config") + 1]).read_text())["run"]["updates"]
                folder = output / "training"
                folder.mkdir(parents=True)
                (folder / "checkpoint.pt").write_bytes(b"fixture checkpoint")
                (folder / "summary.json").write_text(json.dumps({
                    "updates": target, "transitions": target * 10, "training_wall_time_s": 1,
                }))
            else:
                target = int(output.name.split("-")[-1])
                folder = output / "evaluation"
                folder.mkdir(parents=True)
                (folder / "evaluation.json").write_text(json.dumps({
                    "checkpoint_updates": target, "mean_step_reward": 0,
                    "terminated_episodes": 0, "failure_free_environments": 2, "tracking": {},
                }))

        def wait(self):
            return 0

    monkeypatch.setattr(module.subprocess, "Popen", Process)
    result = module.run(args)
    expected = ["train", "evaluate", "train", "evaluate"]
    if first_native_failure:
        expected.insert(0, "train")
    assert [command[2] for command in commands] == expected
    assert "--resume" not in commands[0]
    if teacher_initialization:
        assert commands[0][commands[0].index("--initial-actor") + 1] == str(args.initial_actor.resolve())
        assert commands[0][commands[0].index("--initial-actor-sha256") + 1] == args.initial_actor_sha256
        assert result["initial_actor"] == {"path": str(args.initial_actor.resolve()), "sha256": "a" * 64}
    else:
        assert "--initial-actor" not in commands[0]
        assert result["initial_actor"] is None
    for command in commands[1:]:
        assert "--initial-actor" not in command
        assert "--initial-actor-sha256" not in command
    assert commands[-2][commands[-2].index("--resume") + 1] == commands[-3][commands[-3].index("--checkpoint") + 1]
    assert len(result["native_restarts"]) == int(first_native_failure)
    assert result["phase"] == "complete"
    assert [s["update"] for s in result["completed_stages"]] == [2, 4]
    assert json.loads((args.output / "config-000004.json").read_text())["run"]["updates"] == 4


def test_exit_zero_without_summary_is_not_campaign_success(campaign, monkeypatch):
    module, args = campaign
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: SimpleNamespace(pid=123, wait=lambda: 0))
    with pytest.raises(FileNotFoundError):
        module.run(args)
    status = json.loads((args.output / "status.json").read_text())
    assert status["phase"] == "failed"
    assert status["completed_stages"] == []


@pytest.mark.parametrize("message,recoverable,limit,expected_calls", [
    ("continuous joint state is nonfinite or angular sampling is ambiguous", True, 2, 3),
    ("continuous joint position increment is ambiguous", True, 1, 2),
    ("continuous joint position increment is ambiguous", False, 2, 1),
    ("Nonfinite loss", True, 2, 1),
])
def test_only_saved_native_failures_restart_with_a_bounded_budget(
        campaign, monkeypatch, message, recoverable, limit, expected_calls):
    module, args = campaign
    args.native_restarts = limit
    commands = []

    class Process:
        pid = 456

        def __init__(self, command, **kwargs):
            commands.append(command)
            output = Path(command[command.index("--output") + 1])
            folder = output / "training"
            folder.mkdir(parents=True)
            checkpoint = folder / "failure-checkpoint.pt"
            if recoverable:
                checkpoint.write_bytes(b"complete learner and replay")
            (output / "failure.json").write_text(json.dumps({
                "type": "RuntimeError", "message": message,
            }))
            (folder / "progress.json").write_text(json.dumps({
                "stage": "failed", "recovery": {
                    "status": "recoverable" if recoverable else "save_failed",
                    "checkpoint": str(checkpoint.resolve()) if recoverable else None,
                    "completed_updates": 1, "transitions": 42,
                },
            }))

        def wait(self):
            # Isaac Kit may exit zero even after Python raised.
            return 0

    monkeypatch.setattr(module.subprocess, "Popen", Process)
    with pytest.raises(RuntimeError, match=message):
        module.run(args)
    assert len(commands) == expected_calls
    for previous, resumed in zip(commands, commands[1:]):
        previous_output = Path(previous[previous.index("--output") + 1])
        assert resumed[resumed.index("--resume") + 1] == str(
            previous_output / "training/failure-checkpoint.pt")
        assert resumed[resumed.index("--output") + 1] != str(previous_output)
    status = json.loads((args.output / "status.json").read_text())
    assert status["phase"] == "failed"
    assert status["completed_stages"] == []
    assert len(status["native_restarts"]) == expected_calls - 1


@pytest.mark.parametrize("options,match", [
    ({"initial_actor": "teacher.pt"}, "provided together"),
    ({"initial_actor_sha256": "a" * 64}, "provided together"),
    ({"initial_actor": "teacher.pt", "initial_actor_sha256": "bad"}, "lowercase SHA256"),
    ({"resume": "resume.pt", "initial_actor": "teacher.pt", "initial_actor_sha256": "a" * 64},
     "full resume"),
])
def test_campaign_rejects_invalid_actor_initialization_before_launch(campaign, monkeypatch, options, match):
    module, args = campaign
    for name, value in options.items():
        setattr(args, name, value)

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid teacher initialization cannot launch a process")

    monkeypatch.setattr(module.subprocess, "Popen", forbidden)
    with pytest.raises(ValueError, match=match):
        module.run(args)
    assert not args.output.exists()


def test_campaign_cli_accepts_initial_actor_pin(campaign, monkeypatch, capsys):
    module, _ = campaign
    received = []
    monkeypatch.setattr(module, "run", lambda args: received.append(args) or {"fixture": True})
    module.main(["--config", "learning.json", "--factory-config", "factory.json", "--output", "run",
                 "--initial-actor", "teacher.pt", "--initial-actor-sha256", "a" * 64])
    assert received[0].initial_actor == Path("teacher.pt")
    assert received[0].initial_actor_sha256 == "a" * 64
    assert received[0].resume is None
    assert json.loads(capsys.readouterr().out) == {"fixture": True}
