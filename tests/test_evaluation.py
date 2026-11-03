"""Policy evaluation and named-coordinate checks without starting Isaac."""
from contextlib import nullcontext
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl import cli
from supermimic_rl.adapters.packed import PackedEnvironment
from supermimic_rl.adapters.synthetic import SyntheticEnvironment
from supermimic_rl.contracts import Field, Snapshot, StateSchema, Transition
from supermimic_rl.evaluation import evaluate, native_tracking_sample, terminal_failure_masks
from supermimic_rl.learning import LearningConfig
from supermimic_rl.models import Actor
from supermimic_rl.runtime import RunConfig, TrainingRun


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def checkpoint(tmp_path):
    environment = SyntheticEnvironment(2, seed=7, episode_steps=100)
    config = RunConfig(
        updates=1, warmup_steps=4, collection_steps=2, replay_steps=16,
        model_batches=1, model_batch_size=4, imagination_batch_size=4,
        validation_batch_size=4, checkpoint_interval=1,
    )
    learning = LearningConfig(
        model_hidden=(8,), actor_hidden=(8,), critic_hidden=(8,),
        history_length=3, fit_horizon=2, imagination_horizon=2,
        critic_epochs=1, critic_minibatches=1,
    )
    try:
        run = TrainingRun(environment, tmp_path / "training", config, learning)
        run.run()
        return run.output / "checkpoint.pt"
    finally:
        environment.close()


class RecordedEnvironment(SyntheticEnvironment):
    def __init__(self, *, failures=False, episode_steps=2):
        super().__init__(2, seed=11, episode_steps=episode_steps)
        self.failures = failures
        self.actions, self.rewards = [], []

    def step(self, action):
        self.actions.append(action.detach().clone())
        result = super().step(action)
        if self.failures and len(self.actions) % 2 == 0:
            result.terminated[0] = True
            self.episode_id[0] += 1
            self._reset_rows(torch.tensor([True, False]))
            self.current = self._snapshot()
        self.rewards.append(result.reward.detach().clone())
        return result


def test_evaluation_is_deterministic_and_reports_synthetic_episode_accounting(
    checkpoint, tmp_path,
):
    before = checkpoint.read_bytes()
    first, second = RecordedEnvironment(), RecordedEnvironment()
    reports = [evaluate(environment, checkpoint, tmp_path / name, steps=5)
               for environment, name in ((first, "first"), (second, "second"))]
    assert reports[0] == reports[1]
    assert all(torch.equal(left, right) for left, right in zip(first.actions, second.actions))
    assert checkpoint.read_bytes() == before
    report = reports[0]
    assert report["deterministic_actor"] is True
    assert report["action_noise_std"] == 0
    assert report["actor_updates"] == 1
    assert report["tracking"]["available"] is False
    assert "failure_reason_masks" not in report
    assert "no robot physical metrics" in report["tracking"]["reason"]
    assert report["terminated_episodes"] == 0
    assert report["truncated_episodes"] == report["completed_episodes"] == 4
    assert report["failure_free_environments"] == 2
    assert report["total_reward"] == pytest.approx(torch.stack(first.rewards).double().sum())
    for entry in report["per_environment"]:
        assert entry["completed_episodes"] == 2
        assert entry["current_episode_steps"] == 1
        assert entry["longest_episode_steps"] == 2
        assert entry["episodes_started_including_final_autoreset"] == 3
        assert entry["steps_until_first_failure_or_horizon"] == 5
        assert entry["failed_during_evaluation"] is False
    saved = json.loads((tmp_path / "first" / "evaluation.json").read_text())
    assert saved == report
    records = [json.loads(line) for line in
               (tmp_path / "first" / "steps.jsonl").read_text().splitlines()]
    assert [record["step"] for record in records] == [1, 2, 3, 4, 5]
    assert [record["truncated"] for record in records] == [0, 2, 0, 2, 0]


def test_actor_is_frozen_and_history_restarts_after_auto_reset(
    checkpoint, tmp_path, monkeypatch,
):
    original = Actor.forward
    histories = []

    def observed_actor(actor, history):
        assert not actor.training
        assert not torch.is_grad_enabled()
        assert all(not parameter.requires_grad for parameter in actor.parameters())
        histories.append(history.detach().clone())
        return original(actor, history)

    monkeypatch.setattr(Actor, "forward", observed_actor)
    evaluate(RecordedEnvironment(), checkpoint, tmp_path / "evaluation", steps=5)
    assert len(histories) == 5
    for index in (0, 2, 4):
        torch.testing.assert_close(histories[index], histories[index][:, :1].expand(-1, 3, -1))
    assert not torch.equal(histories[1][:, 0], histories[1][:, -1])
    assert not torch.equal(histories[0][:, 0, :4], histories[2][:, 0, :4])


def test_failure_counts_and_first_failure_time_are_per_environment(checkpoint, tmp_path):
    environment = RecordedEnvironment(failures=True, episode_steps=100)
    report = evaluate(environment, checkpoint, tmp_path / "evaluation", steps=5)
    assert report["terminated_episodes"] == 2
    assert report["truncated_episodes"] == 0
    assert report["failure_free_environments"] == 1
    first, second = report["per_environment"]
    assert first["steps_until_first_failure_or_horizon"] == 2
    assert first["terminated_episodes"] == 2
    assert first["longest_episode_steps"] == 2
    assert second["steps_until_first_failure_or_horizon"] == 5
    assert second["terminated_episodes"] == 0
    assert second["longest_episode_steps"] == 5


def test_evaluation_hooks_see_terminal_endpoint_and_refresh_reset_history(checkpoint, tmp_path):
    class HookEnvironment(RecordedEnvironment):
        def evaluation_begin(self, schema):
            assert schema == self.schema
            self.observed = []
            self.reset()

        def evaluation_step(self, transition, tracking):
            assert tracking is None
            self.observed.append((transition.end.episode_id.clone(), transition.truncated.clone()))

        def evaluation_summary(self):
            return {"intervals": len(self.observed),
                    "pre_reset_terminated": int(sum(done.sum() for _, done in self.observed))}

    env = HookEnvironment()
    report = evaluate(env, checkpoint, tmp_path / "hooks", steps=3)
    assert report["course"] == {"intervals": 3, "pre_reset_terminated": 2}
    first, terminal, fresh = env.observed
    torch.testing.assert_close(first[0], terminal[0])
    assert (fresh[0] > terminal[0]).all()


def test_fixed_suite_allows_a_new_batch_without_loading_training_environment_state(checkpoint, tmp_path):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["environment_training_state"] = {"num_envs": 2, "training_only": True}
    saved = tmp_path / "environment-checkpoint.pt"
    torch.save(payload, saved)

    class FixedSuite(SyntheticEnvironment):
        def load_training_state_dict(self, state):
            raise AssertionError("Fixed suite must not load old native rows or task RNG")

        def evaluation_begin(self, schema):
            assert schema == self.schema
            self.reset()

    environment = FixedSuite(5, episode_steps=10)
    report = evaluate(environment, saved, tmp_path / "changed-batch", steps=2)
    assert report["num_envs"] == 5
    assert report["actor_updates"] == 1
    assert "training environment state not restored" in report["environment_initialization"]
    # Batch flexibility cannot conceal a changed physical/model contract.
    changed = FixedSuite(5)
    changed.schema = replace(changed.schema, source="changed-physical-contract")
    with pytest.raises(ValueError, match="schema"):
        evaluate(changed, saved, tmp_path / "changed-contract", steps=2)


def test_evaluation_restores_event_hooks_when_native_step_fails(checkpoint, tmp_path):
    class FailingSuite(SyntheticEnvironment):
        restored = False

        def evaluation_begin(self, schema):
            self.reset()

        def evaluation_end(self):
            self.restored = True

        def step(self, action):
            raise RuntimeError("native failure fixture")

    environment = FailingSuite(2)
    with pytest.raises(RuntimeError, match="native failure fixture"):
        evaluate(environment, checkpoint, tmp_path / "failed-evaluation", steps=2)
    assert environment.restored


def test_native_tracking_uses_heading_projection_and_world_vertical_yaw():
    up = torch.tensor([[0.6, 0., 0.8]]).expand(2, -1)
    forward = torch.tensor([[0.8, 0., -0.6]]).expand(2, -1)
    measured = SimpleNamespace(
        velocity_origin_body_m_s=2 * forward + 3 * up,
        angular_velocity_body_rad_s=0.5 * up + 7 * forward,
        gravity_body=-up, height_above_ground_m=torch.full((2,), 0.31),
    )
    native = SimpleNamespace(
        _observations=SimpleNamespace(assembler=SimpleNamespace(measure=lambda sample: measured)),
        _sample=lambda: object(),
        _commands=lambda: SimpleNamespace(
            admitted_body_command=torch.tensor([[2., 0., 0.5, 0.3]]).expand(2, -1)),
    )
    environment = PackedEnvironment(SimpleNamespace(env=native), source="coordinate-test")
    environment.num_envs = 2
    actual, targets = native_tracking_sample(environment)
    torch.testing.assert_close(actual, torch.tensor([[2., 0.5, 0.31]]).expand(2, -1))
    torch.testing.assert_close(targets, torch.tensor([[2., 0.5, 0.3]]).expand(2, -1))
    assert not torch.allclose(actual[:, 0], measured.velocity_origin_body_m_s[:, 0])
    assert not torch.allclose(actual[:, 1], measured.angular_velocity_body_rad_s[:, 2])
    assert native_tracking_sample(SyntheticEnvironment(2)) is None


def test_failure_masks_use_named_terminal_state_before_reset():
    schema = StateSchema((Field("prefix", (2,)), Field("failure.failure_reasons", (1,)),
                          Field("suffix", (1,))), ())

    def snapshot(masks):
        state = torch.zeros(3, 4)
        state[:, 2] = torch.tensor(masks, dtype=torch.float32)
        return Snapshot(state, torch.zeros(3, 35), torch.zeros(3, 11), torch.empty(3, 0),
                        torch.zeros(3, dtype=torch.int64), torch.zeros(3, dtype=torch.int64),
                        torch.ones(3, dtype=torch.bool))

    transition = Transition(
        snapshot([0, 0, 0]), torch.zeros(3, 6), snapshot([3, 5, 7]), torch.zeros(3),
        torch.tensor([True, True, False]), torch.tensor([False, False, True]),
        torch.ones(3, dtype=torch.bool),
    )
    assert terminal_failure_masks(schema, transition) == {"3": 1, "5": 1}
    transition.terminated[:] = False
    assert terminal_failure_masks(schema, transition) == {}


def test_gimbal_tracking_reuses_sample_and_matches_native_task_heading():
    calls = {"sample": 0, "gimbal": 0}
    measured = SimpleNamespace(
        velocity_origin_body_m_s=torch.tensor([[1., 0., 0.]]),
        angular_velocity_body_rad_s=torch.zeros(1, 3),
        gravity_body=torch.tensor([[0., 0., -1.]]), height_above_ground_m=torch.tensor([.32]))

    def sample():
        calls["sample"] += 1
        return SimpleNamespace(root_link_orientation_xyzw=torch.tensor([[0., 0., 2 ** -.5, 2 ** -.5]]))

    def gimbal():
        calls["gimbal"] += 1
        return SimpleNamespace(
            native_position=torch.tensor([[-torch.pi / 2, .2]]),
            native_velocity=torch.tensor([[.3, .4]]), joint_angles=torch.tensor([[-torch.pi / 2, -.2]]),
            relative_yaw=torch.tensor([0.]), world_angles=torch.tensor([[torch.pi / 2, .1]]),
            task_heading=torch.zeros(1))

    native = SimpleNamespace(
        _observations=SimpleNamespace(assembler=SimpleNamespace(measure=lambda value: measured)),
        _sample=sample, _commands=lambda: SimpleNamespace(admitted_body_command=torch.tensor([[1., 0., 0., .32]])),
        gimbal_measurement=gimbal, _gimbal=SimpleNamespace(_sign=torch.tensor([1., -1.])),
        _gimbal_course=SimpleNamespace(
            target=torch.tensor([[0., -.3]]), rate=torch.tensor([-1.]), chassis_rate=torch.tensor([7.5]),
            speed=torch.ones(1), direction=torch.tensor([torch.pi / 2]), special=torch.tensor([True])))
    environment = PackedEnvironment(SimpleNamespace(env=native), source="gimbal-coordinates")
    environment.num_envs = 1
    result = native_tracking_sample(environment)
    assert calls == {"sample": 1, "gimbal": 1}
    torch.testing.assert_close(result.gimbal["translation_actual_m_s"], torch.tensor([[0., 1.]]), atol=2e-7, rtol=0)
    torch.testing.assert_close(result.gimbal["translation_actual_m_s"], result.gimbal["translation_target_m_s"], atol=2e-7, rtol=0)
    torch.testing.assert_close(result.gimbal["controller_velocity_rad_s"], torch.tensor([[.3, -.4]]))
    # Using view-world yaw here would give [1,0], a different native target.
    assert result.gimbal["view_world_angles_rad"][0, 0] != result.gimbal["task_heading_rad"][0, 0]


def test_evaluation_rejects_changed_schema_and_existing_output(checkpoint, tmp_path):
    environment = RecordedEnvironment()
    environment.schema = replace(environment.schema, source="different-source")
    with pytest.raises(ValueError, match="schema"):
        evaluate(environment, checkpoint, tmp_path / "different", steps=2)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("previous evaluation")
    with pytest.raises(FileExistsError, match="new output"):
        evaluate(RecordedEnvironment(), checkpoint, output, steps=2)
    assert marker.read_text() == "previous evaluation"
    assert set(output.iterdir()) == {marker}
    with pytest.raises(ValueError, match="positive"):
        evaluate(RecordedEnvironment(), checkpoint, tmp_path / "zero", steps=0)


def test_evaluate_cli_parses_checkpoint_without_training_config(monkeypatch, capsys):
    received = []

    def entry(args):
        received.append(args)
        return {"evaluated": args.steps}

    monkeypatch.setattr(cli, "_evaluate", entry)
    assert cli.main([
        "evaluate", "--checkpoint", "checkpoint.pt", "--factory-config", "factory.json",
        "--output", "evaluation", "--steps", "100",
    ]) == 0
    assert received[0].factory == "supermimic_rl.adapters.factory:create_isaac"
    assert str(received[0].checkpoint) == "checkpoint.pt"
    assert not hasattr(received[0], "config")
    assert json.loads(capsys.readouterr().out) == {"evaluated": 100}


def test_native_evaluation_persists_failure_before_adapter_cleanup(
    tmp_path, monkeypatch, capsys,
):
    output = tmp_path / "native-evaluation"
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"validated-by-mocked-evaluator")
    configuration = tmp_path / "factory.json"
    configuration.write_text(json.dumps({"native_log_directory": "/previous/training/native"}))
    state = {"closed": False}

    def close():
        failure = json.loads((output / "failure.json").read_text())
        assert failure["type"] == "RuntimeError"
        assert failure["message"] == "fixture native evaluation failure"
        assert "RuntimeError" in failure["traceback"]
        state["closed"] = True

    def factory(config):
        assert config["native_log_directory"] == str(output / "native")
        return SimpleNamespace(provenance={"pinned": "same-source"},
                               policy_dt_s=0.02, close=close)

    def fail(environment, selected_checkpoint, selected_output, *, steps):
        assert environment.source == '{"pinned":"same-source"}'
        assert selected_checkpoint == checkpoint
        assert selected_output == output / "evaluation"
        assert steps == 10
        raise RuntimeError("fixture native evaluation failure")

    monkeypatch.setattr(cli.importlib, "import_module",
                        lambda name: SimpleNamespace(create=factory))
    monkeypatch.setattr(cli, "native_logs", lambda directory: nullcontext())
    monkeypatch.setattr(cli, "evaluate", fail)
    assert cli.main([
        "evaluate", "--checkpoint", str(checkpoint), "--factory-config", str(configuration),
        "--factory", "fixture:create", "--output", str(output), "--steps", "10",
    ]) == 2
    assert state["closed"]
    assert "fixture native evaluation failure" in capsys.readouterr().err
