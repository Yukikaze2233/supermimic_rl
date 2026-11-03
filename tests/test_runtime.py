"""End-to-end numerical-fixture checks; these do not validate robot physics."""
from dataclasses import asdict, replace
import json
import math

import pytest
import torch

from supermimic_rl import cli
from supermimic_rl import runtime
from supermimic_rl.adapters.synthetic import SyntheticEnvironment
from supermimic_rl.contracts import StateSchema
from supermimic_rl.learning import LearningConfig
from supermimic_rl.runtime import RunConfig, TrainingRun, load_replay


class FormalFixtureEnvironment(SyntheticEnvironment):
    """Real/noisy endpoint split and a non-policy preparation interval."""

    def __init__(self):
        super().__init__(4, seed=19, episode_steps=100)
        self.schema = replace(self.schema, observation_dim=45, goal_dim=14,
                              source="formal45-control-fixture")
        self.estimates = []
        self.progress = []
        self.restored = None
        self.course_token = 17

    def _snapshot(self):
        frame = super()._snapshot()
        frame.observation = torch.cat((frame.observation, torch.zeros(4, 10)), -1)
        frame.actor_observation = frame.observation.clone()
        frame.actor_observation[:, 4] += .25
        frame.goal = torch.cat((frame.goal, torch.zeros(4, 3)), -1)
        frame.policy_owned[0] = self.steps[0] >= 2
        frame.control_epoch = self.episode_id * 2 + (self.steps >= 2).long()
        frame.estimator_target = torch.cat((self.velocity[:, :3], torch.ones(4, 2)), -1)
        frame.estimator_valid = torch.ones(4, dtype=torch.bool)
        return frame

    def step(self, action):
        executed = action.clone()
        executed[~self.current.policy_owned] = 0
        return super().step(executed)

    def set_policy_estimate(self, frame, raw_state):
        torch.testing.assert_close(frame, self.current.actor_observation)
        self.estimates.append((frame.clone(), raw_state.clone()))

    def set_training_progress(self, real_transitions, actor_updates):
        self.progress.append((real_transitions, actor_updates))

    def training_state_dict(self):
        return {"token": self.course_token, "progress": self.progress[-1]}

    def load_training_state_dict(self, state):
        self.restored = state
        self.course_token = state["token"]


def test_formal_collection_estimator_ownership_and_course_state_resume(tmp_path, monkeypatch):
    from supermimic_rl.models import WorldModel

    def observe(model, state, goal, context, previous_action=None):
        frame = state.new_zeros(state.shape[0], 45)
        frame[:, :4] = goal[:, :4]
        frame[:, 4:7] = state[:, 6:9] * .03
        frame[:, 9] = -1
        frame[:, 10:14] = state[:, :4]
        frame[:, 16:22] = state[:, 6:12] * .1
        frame[:, 22:28] = previous_action
        frame[:, 28:35] = goal[:, 4:11]
        return frame

    def deployed(*args, **kwargs):
        frame = observe(*args, **kwargs)
        frame[:, 4] += .25
        return frame

    monkeypatch.setattr(WorldModel, "observe_clean", observe)
    monkeypatch.setattr(WorldModel, "observe", deployed)
    monkeypatch.setattr(WorldModel, "advance_conditions",
                        lambda self, state, goal, context, action, **kwargs: (state, goal, context))
    monkeypatch.setattr(WorldModel, "imagination_horizon",
                        lambda self, state, goal, context, maximum: torch.full((len(state),), maximum))
    config = RunConfig(updates=1, warmup_steps=12, collection_steps=2, replay_steps=32,
                       world_pretrain_iterations=1, world_pretrain_collection_steps=2,
                       model_batches=1, model_batch_size=4, imagination_batch_size=4,
                       validation_batch_size=4, checkpoint_interval=1)
    learning = LearningConfig(model_hidden=(8,), critic_hidden=(8,), history_length=5,
                              fit_horizon=4, imagination_horizon=2,
                              critic_epochs=1, critic_minibatches=1)
    environment = FormalFixtureEnvironment()
    run = TrainingRun(environment, tmp_path / "formal", config, learning)
    actor_inputs = []
    hook = run.learner.actor.register_forward_pre_hook(
        lambda module, args: actor_inputs.append(args[0].detach().clone()))
    result = run.run()
    hook.remove()
    assert run.learner.actor.network[0].in_features == 225
    assert run.learner.estimator_updates == 2
    assert len(environment.estimates) == 16
    assert actor_inputs[0][:, :, 4].eq(.25).all()
    assert environment.progress[-1] == (64, 1)
    payload = torch.load(result["checkpoint"], weights_only=True)
    data = payload["replay"]["storage"]
    assert not data["policy_owned"][:2, 0].any()
    assert data["actions"][:2, 0].eq(0).all()
    torch.testing.assert_close(data["start_actor_observation"][..., 4]
                               - data["start_observation"][..., 4], torch.full((16, 4), .25))
    assert payload["environment_training_state"] == {"token": 17, "progress": (64, 1)}
    following = FormalFixtureEnvironment()
    resumed = TrainingRun(following, run.output, replace(config, updates=2), learning,
                          resume=run.output / "checkpoint.pt")
    assert following.restored == payload["environment_training_state"]
    assert following.progress[-1] == (64, 1)
    assert resumed.learner.estimator_updates == 2
    final = resumed.run()
    assert final["transitions"] == 72
    assert resumed.learner.estimator_updates == 3
    assert following.progress[-1] == (72, 2)


def test_preparation_only_data_fits_world_without_fabricating_actor_starts(tmp_path, configurations, monkeypatch):
    config, learning = configurations
    environment = SyntheticEnvironment(4, episode_steps=100)
    snapshot = environment._snapshot

    def preparation():
        result = snapshot()
        result.policy_owned.zero_()
        return result

    monkeypatch.setattr(environment, "_snapshot", preparation)
    run = TrainingRun(environment, tmp_path / "preparation", replace(config, updates=1), learning)
    result = run.run()
    assert result["world_optimizer_steps"] == 2
    assert result["actor_optimizer_steps"] == 0
    assert run.replay.num_start_candidates(learning.history_length) == 0
    assert run.replay.num_candidates(learning.fit_horizon) > 0


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def configurations():
    run = RunConfig(
        seed=19, updates=2, warmup_steps=8, collection_steps=2,
        replay_steps=32, model_batches=2, model_batch_size=8,
        imagination_batch_size=8, validation_batch_size=8,
        validation_fraction=0.25, checkpoint_interval=1,
    )
    learning = LearningConfig(
        model_hidden=(16, 16), actor_hidden=(16,), critic_hidden=(8,),
        history_length=5, fit_horizon=2, imagination_horizon=4,
        critic_epochs=1, critic_minibatches=2, learning_rate=1e-3,
    )
    return run, learning


@pytest.fixture
def training_run(tmp_path, configurations):
    config, learning = configurations
    environment = SyntheticEnvironment(4, seed=config.seed, episode_steps=100)
    # Distinct context values make environment leakage observable in sampled data.
    environment.response[:, 0] = torch.tensor([0.05, 0.10, 0.15, 0.20])
    run = TrainingRun(environment, tmp_path / "online", config, learning)
    yield run
    environment.close()


@pytest.fixture
def completed_run(training_run):
    initial = {
        module: {name: value.detach().clone() for name, value in
                 getattr(training_run.learner, module).named_parameters()}
        for module in ("world", "actor", "critic")
    }
    result = training_run.run()
    return training_run, initial, result


def test_online_fixture_updates_all_networks_and_writes_complete_metrics(completed_run):
    run, initial, result = completed_run
    for module, before in initial.items():
        parameters = dict(getattr(run.learner, module).named_parameters())
        assert all(torch.isfinite(value).all() for value in parameters.values())
        assert any(not torch.equal(before[name], value) for name, value in parameters.items())
    assert run.learner.world_updates == 4
    assert run.learner.actor_updates == 2
    assert run.learner.critic_updates == 4
    assert run.learner.normalizers_fitted
    assert result["updates"] == 2
    assert result["transitions"] == 48
    assert result["source"] == "synthetic-fixture-v1-not-robot-physics"

    records = [json.loads(line) for line in
               (run.output / "metrics.jsonl").read_text().splitlines()]
    expected = {
        "collection/reward_mean", "collection/terminations", "collection/truncations",
        "model/loss", "actor/loss", "critic/loss", "imagination/final_survival",
        "update", "transitions", "timing/collection_s", "timing/learning_s",
        "timing/update_before_io_s",
    }
    expected |= {f"{prefix}/{component}" for prefix in ("model", "validation")
                 for component in ("state", "observation", "reward", "termination")}
    expected |= {f"validation/{horizon}_step/{field.name}_rmse"
                 for horizon in (1, run.learner.config.imagination_horizon)
                 for field in run.environment.schema.state_fields}
    assert [record["update"] for record in records] == [1, 2]
    assert [record["transitions"] for record in records] == [40, 48]
    for record in records:
        assert set(record) == expected
        assert all(math.isfinite(value) for value in record.values())
        assert 0 <= record["imagination/final_survival"] <= 1
        assert record["timing/collection_s"] >= 0
        assert record["timing/learning_s"] >= 0
        assert record["timing/update_before_io_s"] >= (
            record["timing/collection_s"] + record["timing/learning_s"]
        )

    assert result["training_wall_time_s"] >= sum(
        record["timing/update_before_io_s"] for record in records
    )

    manifest = json.loads((run.output / "manifest-000.json").read_text())
    assert StateSchema.from_dict(manifest["schema"]).fingerprint == manifest["schema_hash"]
    assert manifest["schema_hash"] == run.environment.schema.fingerprint
    assert manifest["training_envs"] == [0, 1, 2]
    assert manifest["validation_envs"] == [3]
    assert manifest["native_physics_verified_by_this_run"] is False
    checkpoint = torch.load(run.output / "checkpoint.pt", weights_only=True)
    assert checkpoint["format"] == "supermimic-rl-checkpoint-v1"
    assert checkpoint["completed_updates"] == 2
    assert checkpoint["learner"]["counts"] == [4, 2, 4]
    assert checkpoint["learner"]["schema_hash"] == manifest["schema_hash"]
    progress = json.loads((run.output / "progress.json").read_text())
    assert progress["stage"] == "complete"
    assert progress["latest_update"] == progress["target_updates"] == 2
    assert progress["world_optimizer_steps"] == 4
    assert progress["actor_optimizer_steps"] == 2
    assert progress["transitions"] == 48
    assert progress["warmup_steps_completed"] == 8
    assert progress["initial_world_pretraining_complete"] is True
    assert progress["latest_evaluation_request"] is None
    assert progress["latest_metrics"]["update"] == 2
    assert not (run.output / "evaluations").exists()


def test_online_training_and_validation_use_disjoint_source_environments(
    training_run, monkeypatch,
):
    observed = {"fit_normalizers": 0, "fit_world": 0,
                "improve_policy": 0, "validate_world": 0}

    for name in observed:
        original = getattr(training_run.learner, name)

        def checked(batch, *, operation=name, original=original):
            contexts = batch.contexts[..., 0]
            if operation == "validate_world":
                torch.testing.assert_close(contexts, torch.full_like(contexts, 0.20))
            else:
                assert torch.all(contexts < 0.20)
            observed[operation] += 1
            return original(batch)

        monkeypatch.setattr(training_run.learner, name, checked)

    training_run.run()
    assert observed == {"fit_normalizers": 1, "fit_world": 4,
                        "improve_policy": 2, "validate_world": 2}
    assert set(training_run.training_envs).isdisjoint(training_run.validation_envs)


def test_saved_dataset_supports_inspection_and_offline_world_fit(
    completed_run, tmp_path, capsys,
):
    run, initial, _ = completed_run
    replay_path = run.output / "replay.pt"
    replay = load_replay(replay_path)
    assert replay.schema.fingerprint == run.environment.schema.fingerprint
    assert replay.total_added == run.replay.total_added
    assert cli.main(["inspect-replay", str(replay_path), "--horizon", "4"]) == 0
    inspection = json.loads(capsys.readouterr().out)
    assert inspection["schema_hash"] == replay.schema.fingerprint
    assert inspection["total_transitions"] == 48
    assert inspection["stored_transitions"] == 48
    assert inspection["eligible_windows"] == replay.num_candidates(4)
    assert inspection["eligible_actor_starts"] == replay.num_start_candidates(5)

    configuration = tmp_path / "offline.json"
    configuration.write_text(json.dumps({
        "run": asdict(run.config), "learning": asdict(run.learner.config),
    }))
    output = tmp_path / "offline"
    assert cli.main([
        "fit-world", "--replay", str(replay_path), "--config", str(configuration),
        "--output", str(output), "--device", "cpu",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["updates"] == 2
    assert result["actor_updates"] == 0
    assert result["source"] == replay.schema.source
    checkpoint = torch.load(output / "world-model.pt", weights_only=True)
    assert checkpoint["format"] == "supermimic-rl-world-fit-v1"
    assert checkpoint["learner"]["schema_hash"] == replay.schema.fingerprint
    assert checkpoint["learner"]["counts"] == [2, 0, 0]
    assert checkpoint["training_envs"] == [0, 1, 2]
    assert checkpoint["validation_envs"] == [3]
    for name, parameter in initial["actor"].items():
        torch.testing.assert_close(checkpoint["learner"]["actor"][name], parameter)
    records = [json.loads(line) for line in
               (output / "metrics.jsonl").read_text().splitlines()]
    assert [record["update"] for record in records] == [1, 2]
    assert all(math.isfinite(value) for record in records for value in record.values())


def test_cold_resume_restores_learning_without_joining_old_episode_history(
    training_run, tmp_path, monkeypatch,
):
    original_collect = training_run.collector.collect
    collection_calls = 0

    def interrupted_collection(steps):
        nonlocal collection_calls
        collection_calls += 1
        if collection_calls == 3:
            raise RuntimeError("fixture interruption after checkpoint one")
        return original_collect(steps)

    monkeypatch.setattr(training_run.collector, "collect", interrupted_collection)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        training_run.run()
    path = training_run.output / "checkpoint.pt"
    prior = torch.load(path, weights_only=True)
    assert prior["completed_updates"] == 1
    old_steps = prior["replay"]["size_steps"]
    assert old_steps == 10

    environment = SyntheticEnvironment(4, seed=training_run.config.seed, episode_steps=100)
    try:
        resumed = TrainingRun(
            environment, tmp_path / "resumed", training_run.config,
            training_run.learner.config, resume=path,
        )
        assert resumed.completed_updates == 1
        assert resumed.generation == 1
        assert resumed.learner.world_updates == 2
        assert torch.equal(resumed.collector.generator.get_state(), prior["collection_rng"])
        for name, value in resumed.learner.actor.state_dict().items():
            torch.testing.assert_close(value, prior["learner"]["actor"][name])
        result = resumed.run()
        assert result["updates"] == 2
        assert result["transitions"] == 48
        assert resumed.learner.world_updates == 4
        assert resumed.learner.actor_updates == 2
        assert resumed.learner.critic_updates == 4
        assert resumed.collector.steps == prior["collection_steps"] + training_run.config.collection_steps

        stored = resumed.replay.state_dict()["storage"]
        offset = 1 << 40
        for name in ("episode_id", "control_epoch"):
            old = stored[f"start_{name}"][:old_steps]
            new = stored[f"start_{name}"][old_steps:]
            assert torch.all(old < offset)
            assert torch.all(new >= offset)
            assert set(old.flatten().tolist()).isdisjoint(new.flatten().tolist())
            assert torch.equal(stored[f"start_{name}"], stored[f"end_{name}"])
        new_steps = training_run.config.collection_steps
        assert resumed.replay.num_candidates(2) == 4 * ((old_steps - 1) + (new_steps - 1))
        assert resumed.replay.num_start_candidates(5) == 4 * (old_steps - 3)
        assert (resumed.output / "manifest-001.json").is_file()
        assert not (resumed.output / "manifest-000.json").exists()
        records = (resumed.output / "metrics.jsonl").read_text().splitlines()
        assert [json.loads(line)["update"] for line in records] == [2]
    finally:
        environment.close()


def test_simulator_failure_keeps_partial_collection_and_resumes_completed_learning(
    tmp_path, configurations, monkeypatch,
):
    config, learning = configurations
    config = replace(config, initial_world_batches=3, checkpoint_interval=99)
    environment = SyntheticEnvironment(4, seed=config.seed, episode_steps=100)
    run = TrainingRun(environment, tmp_path / "partial-collection", config, learning)
    original_step = environment.step
    calls = 0
    fault = RuntimeError("native continuous-joint state became nonfinite")

    def step(action):
        nonlocal calls
        calls += 1
        transition = original_step(action)
        # Physics can advance before a native validation error. Only the first
        # interval of the second online collection has reached replay here.
        if calls == config.warmup_steps + 2 * config.collection_steps:
            raise fault
        return transition

    monkeypatch.setattr(environment, "step", step)
    try:
        with pytest.raises(RuntimeError) as caught:
            run.run()
        assert caught.value is fault
        checkpoint = run.output / "failure-checkpoint.pt"
        payload = torch.load(checkpoint, weights_only=True)
        assert payload["completed_updates"] == 1
        assert payload["learner"]["counts"] == [5, 1, 2]
        assert payload["replay"]["total_steps"] == payload["collection_steps"] == 11
        assert payload["initial_world_pretraining"] == {"completed_batches": 3, "complete": True}
        # The regular checkpoint precedes online learning; the failure artifact
        # must include the more recent completed update and collection prefix.
        regular = torch.load(run.output / "checkpoint.pt", weights_only=True)
        assert regular["completed_updates"] == 0
        progress = json.loads((run.output / "progress.json").read_text())
        assert progress["stage"] == "failed"
        assert progress["error"]["message"] == str(fault)
        assert progress["recovery"]["status"] == "recoverable"
        assert progress["recovery"]["checkpoint"] == str(checkpoint.resolve())
        assert progress["recovery"]["completed_updates"] == 1
        assert progress["recovery"]["transitions"] == 44
        assert "cold-resets" in progress["recovery"]["semantics"]
    finally:
        environment.close()

    fresh = SyntheticEnvironment(4, seed=config.seed, episode_steps=100)
    try:
        resumed = TrainingRun(
            fresh, tmp_path / "partial-resumed", replace(config, checkpoint_interval=1),
            learning, resume=checkpoint,
        )
        assert resumed.generation == 1
        assert resumed.learner.normalizers_fitted
        assert resumed.initial_world_pretraining_complete
        assert resumed.collector.steps == 11
        assert torch.equal(resumed.collector.generator.get_state(), payload["collection_rng"])
        collected = []
        original_collect = resumed.collector.collect

        def collect(steps):
            collected.append(steps)
            return original_collect(steps)

        def repeated_normalizers(batch):
            pytest.fail("Completed normalization must not repeat on resume")

        monkeypatch.setattr(resumed.collector, "collect", collect)
        monkeypatch.setattr(resumed.learner, "fit_normalizers", repeated_normalizers)
        resumed.run()
        assert collected == [config.collection_steps]
        assert resumed.learner.world_updates == 7
        assert resumed.learner.actor_updates == 2
        assert resumed.learner.critic_updates == 4
        assert resumed.collector.steps == 13
        assert resumed.replay.total_added == 52
        assert not (resumed.output / "pretraining.jsonl").exists()
        stored = resumed.replay.state_dict()["storage"]
        for name in ("episode_id", "control_epoch"):
            assert stored[f"start_{name}"][:11].lt(1 << 40).all()
            assert stored[f"start_{name}"][11:].ge(1 << 40).all()
        resumed_progress = json.loads((resumed.output / "progress.json").read_text())
        assert resumed_progress["recovery"] is None
        assert resumed_progress["stage"] == "complete"
    finally:
        fresh.close()


def test_warmup_simulator_failure_resumes_only_missing_steps(tmp_path, configurations, monkeypatch):
    config, learning = configurations
    environment = SyntheticEnvironment(4, seed=config.seed, episode_steps=100)
    run = TrainingRun(environment, tmp_path / "warmup-failure", config, learning)
    original_step = environment.step
    calls = 0

    def step(action):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("fixture physics failure during warmup")
        return original_step(action)

    monkeypatch.setattr(environment, "step", step)
    try:
        with pytest.raises(RuntimeError, match="physics failure"):
            run.run()
        checkpoint = run.output / "failure-checkpoint.pt"
        payload = torch.load(checkpoint, weights_only=True)
        assert payload["replay"]["total_steps"] == 3
        assert payload["learner"]["counts"] == [0, 0, 0]
        assert payload["completed_updates"] == 0
    finally:
        environment.close()
    fresh = SyntheticEnvironment(4, seed=config.seed, episode_steps=100)
    try:
        resumed = TrainingRun(fresh, tmp_path / "warmup-resumed", config, learning, resume=checkpoint)
        collected = []
        original_collect = resumed.collector.collect

        def collect(steps):
            collected.append(steps)
            return original_collect(steps)

        monkeypatch.setattr(resumed.collector, "collect", collect)
        resumed.run()
        assert collected == [config.warmup_steps - 3, config.collection_steps, config.collection_steps]
        assert resumed.replay.total_added == 48
        assert resumed.collector.steps == 12
    finally:
        fresh.close()


@pytest.mark.parametrize("operation", ["fit_world", "improve_policy"])
def test_partial_optimizer_failure_never_publishes_recoverable_checkpoint(
    training_run, monkeypatch, operation,
):
    original = getattr(training_run.learner, operation)

    def failed_update(batch):
        original(batch)
        raise RuntimeError("fixture error after optimizer mutation")

    monkeypatch.setattr(training_run.learner, operation, failed_update)
    with pytest.raises(RuntimeError, match="optimizer mutation"):
        training_run.run()
    assert training_run.completed_updates == 0
    assert training_run.learner.world_updates > 0
    assert not (training_run.output / "failure-checkpoint.pt").exists()
    assert json.loads((training_run.output / "progress.json").read_text())["recovery"] is None


def test_failure_checkpoint_write_error_preserves_original_simulator_exception(training_run, monkeypatch):
    original_step = training_run.environment.step
    calls = 0
    fault = RuntimeError("original native simulation failure")

    def step(action):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise fault
        return original_step(action)

    def save(path=None):
        raise OSError("checkpoint disk is full")

    monkeypatch.setattr(training_run.environment, "step", step)
    monkeypatch.setattr(training_run, "save", save)
    with pytest.raises(RuntimeError) as caught:
        training_run.run()
    assert caught.value is fault
    assert any("disk is full" in note for note in fault.__notes__)
    progress = json.loads((training_run.output / "progress.json").read_text())
    assert progress["error"] == {"type": "RuntimeError", "message": str(fault)}
    assert progress["recovery"]["status"] == "save_failed"
    assert progress["recovery"]["checkpoint"] is None
    assert progress["recovery"]["save_error"] == {"type": "OSError", "message": "checkpoint disk is full"}
    assert not (training_run.output / "failure-checkpoint.pt").exists()


def test_resume_rejects_changed_source_schema(completed_run, tmp_path):
    run, _, _ = completed_run
    environment = SyntheticEnvironment(4, seed=run.config.seed)
    environment.schema = replace(environment.schema, source="different-dynamics-source")
    try:
        with pytest.raises(ValueError, match="schema"):
            TrainingRun(
                environment, tmp_path / "wrong-schema", run.config, run.learner.config,
                resume=run.output / "checkpoint.pt",
            )
    finally:
        environment.close()


def test_resume_extends_update_target_without_restarting_learning(completed_run, tmp_path):
    run, _, _ = completed_run
    environment = SyntheticEnvironment(4, seed=run.config.seed, episode_steps=100)
    try:
        resumed = TrainingRun(
            environment, tmp_path / "extended", replace(run.config, updates=3),
            run.learner.config, resume=run.output / "checkpoint.pt",
        )
        result = resumed.run()
        assert result["updates"] == 3
        assert resumed.learner.actor_updates == 3
        assert resumed.learner.world_updates == 6
        assert result["transitions"] == 56
        rows = (resumed.output / "metrics.jsonl").read_text().splitlines()
        assert [json.loads(row)["update"] for row in rows] == [3]
    finally:
        environment.close()


def test_dataset_rejects_schema_contents_changed_without_new_hash(
    completed_run, tmp_path, capsys,
):
    run, _, _ = completed_run
    payload = run.replay.state_dict()
    payload["schema"]["state_fields"][0]["units"] = "mismatched-units"
    path = tmp_path / "mixed-schema.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="fingerprint"):
        load_replay(path)
    assert cli.main(["inspect-replay", str(path), "--horizon", "2"]) == 2
    output = capsys.readouterr()
    assert "fingerprint" in output.err
    assert not output.out


def test_training_refuses_to_overwrite_nonempty_output(tmp_path, configurations):
    config, learning = configurations
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("existing experiment")
    environment = SyntheticEnvironment(4, seed=config.seed)
    try:
        with pytest.raises(FileExistsError, match="new output"):
            TrainingRun(environment, output, config, learning)
        assert marker.read_text() == "existing experiment"
        assert set(output.iterdir()) == {marker}
        assert environment.current is None
    finally:
        environment.close()


@pytest.mark.parametrize("command", ["smoke", "fit-world"])
def test_cli_refuses_to_overwrite_nonempty_output(
    command, tmp_path, configurations, capsys,
):
    config, learning = configurations
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("existing experiment")
    if command == "smoke":
        arguments = ["smoke", "--output", str(output), "--updates", "2"]
    else:
        configuration = tmp_path / "configuration.json"
        configuration.write_text(json.dumps({
            "run": asdict(config), "learning": asdict(learning),
        }))
        arguments = [
            "fit-world", "--replay", str(tmp_path / "not-loaded.pt"),
            "--config", str(configuration), "--output", str(output),
        ]
    assert cli.main(arguments) == 2
    assert "new" in capsys.readouterr().err
    assert marker.read_text() == "existing experiment"
    assert set(output.iterdir()) == {marker}


def test_initial_world_pretraining_is_saved_before_actor_and_not_repeated_on_resume(
    tmp_path, configurations, monkeypatch,
):
    config, learning = configurations
    config = replace(config, initial_world_batches=3)
    environment = SyntheticEnvironment(4, seed=config.seed, episode_steps=100)
    run = TrainingRun(environment, tmp_path / "pretraining", config, learning)
    actor_before = {name: value.clone() for name, value in run.learner.actor.state_dict().items()}
    critic_before = {name: value.clone() for name, value in run.learner.critic.state_dict().items()}
    original_collect = run.collector.collect
    calls = 0

    def interrupt_after_pretraining(steps):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("interrupt before first online actor update")
        return original_collect(steps)

    monkeypatch.setattr(run.collector, "collect", interrupt_after_pretraining)
    try:
        with pytest.raises(RuntimeError, match="first online actor"):
            run.run()
        payload = torch.load(run.output / "checkpoint.pt", weights_only=True)
        assert payload["completed_updates"] == 0
        assert payload["learner"]["counts"] == [3, 0, 0]
        assert payload["initial_world_pretraining"] == {
            "completed_batches": 3, "complete": True,
        }
        for name, value in payload["learner"]["actor"].items():
            torch.testing.assert_close(value, actor_before[name])
        for name, value in payload["learner"]["critic"].items():
            torch.testing.assert_close(value, critic_before[name])
        records = [json.loads(line) for line in
                   (run.output / "pretraining.jsonl").read_text().splitlines()]
        assert [record["batch"] for record in records] == [1, 2, 3]
        failed = json.loads((run.output / "progress.json").read_text())
        assert failed["stage"] == "failed"
        assert failed["initial_world_batches_completed"] == 3
        assert failed["error"]["type"] == "RuntimeError"
    finally:
        environment.close()

    fresh = SyntheticEnvironment(4, seed=config.seed, episode_steps=100)
    try:
        resumed = TrainingRun(fresh, tmp_path / "resumed-prefit", config, learning,
                              resume=run.output / "checkpoint.pt")
        assert resumed.initial_world_pretraining_complete
        resumed.run()
        assert resumed.learner.world_updates == 3 + config.updates * config.model_batches
        assert resumed.learner.actor_updates == config.updates
        assert not (resumed.output / "pretraining.jsonl").exists()
        progress = json.loads((resumed.output / "progress.json").read_text())
        assert progress["stage"] == "complete"
        assert progress["initial_world_batches_completed"] == 3
        assert progress["world_optimizer_steps"] == 7
    finally:
        fresh.close()


def test_periodic_and_final_checkpoints_request_external_evaluation_without_resetting(
    tmp_path, configurations, monkeypatch,
):
    config, learning = configurations
    config = replace(config, updates=3, evaluation_interval=2, evaluation_steps=37)
    environment = SyntheticEnvironment(4, seed=config.seed, episode_steps=100)
    original_reset = environment.reset
    resets = 0

    def reset():
        nonlocal resets
        resets += 1
        return original_reset()

    monkeypatch.setattr(environment, "reset", reset)
    run = TrainingRun(environment, tmp_path / "staged", config, learning)
    try:
        result = run.run()
        assert resets == 1
        assert run.replay.total_added == 56
        assert run.collector.steps == 14
        requests = sorted((run.output / "evaluations").glob("*.json"))
        assert [path.name for path in requests] == ["update-000002.json", "update-000003.json"]
        for path, expected_update in zip(requests, (2, 3)):
            request = json.loads(path.read_text())
            assert request["status"] == "pending_external_evaluation"
            assert request["steps"] == 37
            assert request["update"] == expected_update
            assert request["schema_hash"] == environment.schema.fingerprint
            payload = torch.load(request["checkpoint"], weights_only=True)
            assert payload["completed_updates"] == expected_update
            assert payload["learner"]["counts"][1] == expected_update
            assert not (run.output / "evaluations" / f"update-{expected_update:06d}").exists()
        assert result["latest_evaluation_request"] == str(requests[-1].resolve())
        assert json.loads((run.output / "progress.json").read_text())["stage"] == "complete"
    finally:
        environment.close()


def test_legacy_checkpoint_defaults_load_on_cpu_and_allow_new_evaluation_schedule(
    completed_run, tmp_path, monkeypatch,
):
    run, _, _ = completed_run
    payload = torch.load(run.output / "checkpoint.pt", weights_only=True)
    for name in ("initial_world_batches", "world_pretrain_iterations",
                 "world_pretrain_collection_steps", "evaluation_interval", "evaluation_steps"):
        payload["run_config"].pop(name)
    for name in ("initial_world_pretraining", "world_pretraining", "collection_rng", "collection_steps"):
        payload.pop(name)
    for name in ("actor_architecture", "actor_policy_config"):
        payload["learner"]["config"].pop(name)
    payload["learner"].pop("actor_source_metadata")
    checkpoint = tmp_path / "legacy.pt"
    torch.save(payload, checkpoint)
    original_load = torch.load
    load_locations = []

    def load(*args, **kwargs):
        load_locations.append(kwargs.get("map_location"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(runtime.torch, "load", load)
    environment = SyntheticEnvironment(4, seed=run.config.seed, episode_steps=100)
    try:
        resumed = TrainingRun(
            environment, tmp_path / "legacy-resume",
            replace(run.config, updates=3, evaluation_interval=1, evaluation_steps=19),
            run.learner.config, resume=checkpoint,
        )
        assert load_locations == ["cpu"]
        assert resumed.initial_world_pretraining_complete
        assert resumed.initial_world_batches_completed == 0
        assert resumed.world_pretraining_complete
        assert resumed.world_pretrain_iterations_completed == 0
        assert resumed.world_pretrain_collection_steps_completed == 0
        assert resumed.collector.steps == 12
        resumed.run()
        assert resumed.learner.world_updates == 6
        assert resumed.learner.actor_updates == 3
        request = json.loads((resumed.output / "evaluations" / "update-000003.json").read_text())
        assert request["steps"] == 19
        assert resumed.replay.total_added == 56
    finally:
        environment.close()


def test_warmup_progress_reports_completed_chunks(tmp_path, configurations, monkeypatch):
    config, learning = configurations
    config = replace(config, updates=1, warmup_steps=40, replay_steps=64)
    environment = SyntheticEnvironment(4, seed=config.seed, episode_steps=100)
    run = TrainingRun(environment, tmp_path / "warmup", config, learning)
    original_collect = run.collector.collect
    observed = []

    def collect(steps):
        progress = json.loads((run.output / "progress.json").read_text())
        observed.append((steps, progress["stage"], progress["warmup_steps_completed"]))
        return original_collect(steps)

    monkeypatch.setattr(run.collector, "collect", collect)
    try:
        run.run()
        assert observed == [(32, "warmup", 0), (8, "warmup", 32), (2, "collecting", 40)]
    finally:
        environment.close()


@pytest.mark.parametrize("field,value", [
    ("initial_world_batches", -1), ("initial_world_batches", True),
    ("world_pretrain_iterations", -1), ("world_pretrain_iterations", True),
    ("world_pretrain_collection_steps", 0), ("world_pretrain_collection_steps", False),
    ("world_pretrain_collection_steps", 1.5),
    ("evaluation_interval", -1), ("evaluation_interval", 2.5),
    ("evaluation_steps", 0), ("evaluation_steps", False),
])
def test_pretraining_and_evaluation_configuration_rejects_invalid_counts(field, value):
    with pytest.raises(ValueError, match=field):
        RunConfig(**{field: value})


@pytest.mark.parametrize("collection_steps", [None, 3])
def test_collecting_pretraining_keeps_actor_fixed_and_saves_before_online_learning(
    tmp_path, configurations, monkeypatch, collection_steps,
):
    config, learning = configurations
    config = replace(config, initial_world_batches=3, world_pretrain_iterations=3,
                     world_pretrain_collection_steps=collection_steps)
    environment = SyntheticEnvironment(4, seed=config.seed, episode_steps=100)
    run = TrainingRun(environment, tmp_path / "collecting-prefit", config, learning)
    before = {name: {key: value.clone() for key, value in getattr(run.learner, name).state_dict().items()}
              for name in ("actor", "critic")}
    world_before = {key: value.clone() for key, value in run.learner.world.named_parameters()}
    original_collect = run.collector.collect
    original_step = environment.step
    issued_noise = []

    def collect(steps):
        if run.world_pretraining_complete:
            raise RuntimeError("stop before online collection")
        return original_collect(steps)

    def step(action):
        if run.learner.normalizers_fitted:
            clean = run.learner.actor(run.collector.history).detach()
            issued_noise.append(float((action - clean).abs().max()))
        return original_step(action)

    monkeypatch.setattr(run.collector, "collect", collect)
    monkeypatch.setattr(environment, "step", step)
    try:
        with pytest.raises(RuntimeError, match="before online"):
            run.run()
        steps = collection_steps or config.collection_steps
        payload = torch.load(run.output / "checkpoint.pt", weights_only=True)
        assert payload["completed_updates"] == 0
        assert payload["learner"]["counts"] == [3 + 3 * config.model_batches, 0, 0]
        assert payload["world_pretraining"] == {
            "completed_iterations": 3, "collection_steps_completed": 3 * steps, "complete": True,
        }
        assert payload["collection_steps"] == config.warmup_steps + 3 * steps
        assert payload["replay"]["total_steps"] == payload["collection_steps"]
        for name in ("actor", "critic"):
            for key, value in payload["learner"][name].items():
                assert torch.equal(value, before[name][key])
        assert any(not torch.equal(value, world_before[key])
                   for key, value in run.learner.world.named_parameters())
        assert len(issued_noise) == 3 * steps and min(issued_noise) > 0
        records = [json.loads(line) for line in
                   (run.output / "world-pretraining.jsonl").read_text().splitlines()]
        assert [row["iteration"] for row in records] == [1, 2, 3]
        assert [row["transitions"] for row in records] == [
            (config.warmup_steps + index * steps) * 4 for index in (1, 2, 3)]
        assert [row["world_optimizer_steps"] for row in records] == [5, 7, 9]
        assert all(row["actor_optimizer_steps"] == row["critic_optimizer_steps"] == 0
                   for row in records)
        assert all("validation/state" in row for row in records)
        progress = json.loads((run.output / "progress.json").read_text())
        assert progress["world_pretraining_complete"]
        assert progress["world_pretrain_iterations_completed"] == 3
        assert progress["world_pretrain_collection_steps_completed"] == 3 * steps
        assert not (run.output / "metrics.jsonl").exists()
    finally:
        environment.close()


@pytest.mark.parametrize("partial_native_failure", [False, True])
def test_collecting_pretraining_resume_preserves_completed_iterations_and_collection_prefix(
    tmp_path, configurations, monkeypatch, partial_native_failure,
):
    config, learning = configurations
    config = replace(config, initial_world_batches=3, world_pretrain_iterations=3,
                     world_pretrain_collection_steps=3)
    environment = SyntheticEnvironment(4, seed=config.seed, episode_steps=100)
    run = TrainingRun(environment, tmp_path / "interrupted-prefit", config, learning)
    original_collect, original_step = run.collector.collect, environment.step

    def collect(steps):
        if run.world_pretrain_iterations_completed == 1 and not partial_native_failure:
            raise RuntimeError("interrupt between pretraining iterations")
        return original_collect(steps)

    def step(action):
        transition = original_step(action)
        if (partial_native_failure and run.world_pretrain_iterations_completed == 1
                and run.collector.steps == config.warmup_steps + 4):
            raise RuntimeError("native error after first retained interval")
        return transition

    monkeypatch.setattr(run.collector, "collect", collect)
    monkeypatch.setattr(environment, "step", step)
    try:
        with pytest.raises(RuntimeError):
            run.run()
        checkpoint = run.output / ("failure-checkpoint.pt" if partial_native_failure else "checkpoint.pt")
        payload = torch.load(checkpoint, weights_only=True)
        prefix = 1 if partial_native_failure else 0
        assert payload["learner"]["counts"] == [5, 0, 0]
        assert payload["world_pretraining"] == {
            "completed_iterations": 1, "collection_steps_completed": 3 + prefix, "complete": False,
        }
        assert payload["collection_steps"] == config.warmup_steps + 3 + prefix
        assert payload["replay"]["total_steps"] == payload["collection_steps"]
    finally:
        environment.close()

    fresh = SyntheticEnvironment(4, seed=config.seed, episode_steps=100)
    try:
        resumed = TrainingRun(fresh, tmp_path / "resumed-collecting-prefit", config, learning,
                              resume=checkpoint)
        assert resumed.generation == 1
        assert resumed.world_pretrain_iterations_completed == 1
        assert torch.equal(resumed.collector.generator.get_state(), payload["collection_rng"])
        normalizer_before = resumed.learner.world.state_normalizer.state_dict()
        normalizer_before = {key: value.clone() for key, value in normalizer_before.items()}
        calls = []
        original = resumed.collector.collect

        def resumed_collect(steps):
            calls.append(steps)
            return original(steps)

        monkeypatch.setattr(resumed.collector, "collect", resumed_collect)
        monkeypatch.setattr(resumed.learner, "fit_normalizers",
                            lambda batch: pytest.fail("Normalization must stay fixed on resume"))
        result = resumed.run()
        assert calls == [3 - prefix, 3, config.collection_steps, config.collection_steps]
        assert resumed.learner.world_updates == 3 + (3 + config.updates) * config.model_batches
        assert resumed.learner.actor_updates == config.updates
        assert resumed.learner.critic_updates == 4
        assert resumed.collector.steps == config.warmup_steps + 9 + config.updates * config.collection_steps
        assert result["transitions"] == resumed.collector.steps * 4
        assert result["world_pretrain_iterations_completed"] == 3
        assert result["world_pretrain_collection_steps_completed"] == 9
        for key, value in resumed.learner.world.state_normalizer.state_dict().items():
            assert torch.equal(value, normalizer_before[key])
        records = [json.loads(line) for line in
                   (resumed.output / "world-pretraining.jsonl").read_text().splitlines()]
        assert [row["iteration"] for row in records] == [2, 3]
        assert [row["collection_steps_this_session"] for row in records] == [3 - prefix, 3]
        assert [row["transitions"] for row in records] == [56, 68]
        stored = resumed.replay.state_dict()["storage"]
        retained = payload["collection_steps"]
        assert stored["start_episode_id"][:retained].lt(1 << 40).all()
        assert stored["start_episode_id"][retained:].ge(1 << 40).all()
        assert not (resumed.output / "pretraining.jsonl").exists()
    finally:
        fresh.close()


def test_completed_collecting_pretraining_does_not_repeat_on_extended_resume(
    tmp_path, configurations,
):
    config, learning = configurations
    config = replace(config, updates=1, world_pretrain_iterations=2,
                     world_pretrain_collection_steps=3)
    environment = SyntheticEnvironment(4, seed=config.seed, episode_steps=100)
    run = TrainingRun(environment, tmp_path / "complete-prefit", config, learning)
    try:
        run.run()
    finally:
        environment.close()
    fresh = SyntheticEnvironment(4, seed=config.seed, episode_steps=100)
    try:
        resumed = TrainingRun(fresh, tmp_path / "extended-prefit", replace(config, updates=2),
                              learning, resume=run.output / "checkpoint.pt")
        result = resumed.run()
        assert result["transitions"] == (config.warmup_steps + 6 + 4) * 4
        assert resumed.learner.world_updates == 8
        assert resumed.learner.actor_updates == 2
        assert not (resumed.output / "world-pretraining.jsonl").exists()
    finally:
        fresh.close()


def test_cli_configuration_accepts_collecting_pretraining_fields(tmp_path, configurations):
    config, learning = configurations
    config = replace(config, world_pretrain_iterations=7, world_pretrain_collection_steps=5)
    path = tmp_path / "continuous-world.json"
    path.write_text(json.dumps({"run": asdict(config), "learning": asdict(learning)}))
    loaded, loaded_learning = cli._configuration(path)
    assert loaded == config
    assert loaded_learning == learning


def test_collecting_pretraining_optimizer_failure_keeps_last_complete_checkpoint(
    tmp_path, configurations, monkeypatch,
):
    config, learning = configurations
    config = replace(config, world_pretrain_iterations=3, world_pretrain_collection_steps=3)
    environment = SyntheticEnvironment(4, seed=config.seed, episode_steps=100)
    run = TrainingRun(environment, tmp_path / "failed-prefit-optimizer", config, learning)
    original_fit = run.learner.fit_world

    def failed_fit(batch):
        result = original_fit(batch)
        if run.world_pretrain_iterations_completed == 1:
            raise RuntimeError("failure after world optimizer mutation")
        return result

    monkeypatch.setattr(run.learner, "fit_world", failed_fit)
    try:
        with pytest.raises(RuntimeError, match="optimizer mutation"):
            run.run()
        saved = torch.load(run.output / "checkpoint.pt", weights_only=True)
        assert saved["learner"]["counts"] == [2, 0, 0]
        assert saved["world_pretraining"] == {
            "completed_iterations": 1, "collection_steps_completed": 3, "complete": False,
        }
        assert run.learner.world_updates == 3
        assert not (run.output / "failure-checkpoint.pt").exists()
        assert json.loads((run.output / "progress.json").read_text())["recovery"] is None
    finally:
        environment.close()


@pytest.mark.parametrize("progress", [
    {"completed_iterations": 0, "collection_steps_completed": 0, "complete": True},
    {"completed_iterations": 1, "collection_steps_completed": 0, "complete": False},
    {"completed_iterations": 0, "collection_steps_completed": 4, "complete": False},
    {"completed_iterations": True, "collection_steps_completed": 3, "complete": False},
])
def test_resume_rejects_inconsistent_collecting_pretraining_progress(
    completed_run, tmp_path, progress,
):
    run, _, _ = completed_run
    config = replace(run.config, world_pretrain_iterations=2, world_pretrain_collection_steps=3)
    payload = torch.load(run.output / "checkpoint.pt", weights_only=True)
    payload["run_config"] = asdict(config)
    payload["world_pretraining"] = progress
    checkpoint = tmp_path / "invalid-prefit.pt"
    torch.save(payload, checkpoint)
    environment = SyntheticEnvironment(4, seed=config.seed, episode_steps=100)
    try:
        with pytest.raises(ValueError, match="collecting world pretraining"):
            TrainingRun(environment, tmp_path / "invalid-resume", config,
                        run.learner.config, resume=checkpoint)
    finally:
        environment.close()
