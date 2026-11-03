"""Teacher initialization and SM resume using the real optional upstream actor."""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl import cli
from supermimic_rl.adapters.synthetic import SyntheticEnvironment
from supermimic_rl.contracts import ACTION_BOUNDS
from supermimic_rl.evaluation import evaluate
from supermimic_rl.learning import Learner, LearningConfig
from supermimic_rl.runtime import RunConfig, TrainingRun
from supermimic_rl.transferred_actor import TransferredActor


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def transfer(tmp_path):
    upstream = pytest.importorskip(
        "transformer_rl.formal.policy",
        reason="Transfer integration requires the pinned transformer_rl source on PYTHONPATH",
    )
    policy = upstream.StructuredPolicyConfig(
        body_width=8, body_heads=2, body_ffn=16, temporal_width=16,
        temporal_heads=2, temporal_layers=1, temporal_ffn=32,
        state_hidden_dim=8, leg_hidden_dims=(16, 8), wheel_hidden_dims=(16, 8),
        prediction_hidden_dim=8, critic_hidden_dims=(16, 8, 4),
        mirror_canonical=True, mean_init_scale=0.1,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(135)
        reference = upstream.StructuredPolicy(policy).eval()
    payload = {
        "format": "private_structured_learning_snapshot",
        "recipe": {"policy": asdict(policy), "collection": {"action_bounds": list(ACTION_BOUNDS)},
                   "learner_source_sha256": "a" * 64, "asset_manifest_sha256": "b" * 64,
                   "task_contract_sha256": "c" * 64},
        "normalization": policy.normalization_contract,
        "model": {"actor.policy." + name: value.clone()
                  for name, value in reference.state_dict().items()},
        "actor_adam": {"must_not_be_imported": True},
        "critic_adam": {"must_not_be_imported": True},
        "trainer": {"update_index": 18026},
    }
    payload["model"].update({"actor.log_std": torch.full((6,), 123.),
                             "critic.unrelated": torch.full((7,), float("nan"))})
    path = tmp_path / "teacher.pt"
    torch.save(payload, path)
    learning = LearningConfig(
        model_hidden=(16,), actor_hidden=(16,), critic_hidden=(8,),
        fit_horizon=2, imagination_horizon=3, critic_epochs=1,
        critic_minibatches=1, learning_rate=1e-3,
        actor_architecture="transferred_transformer", actor_policy_config=asdict(policy),
    )
    run = RunConfig(
        seed=19, updates=1, warmup_steps=8, replay_steps=32, collection_steps=2,
        model_batches=1, model_batch_size=4, imagination_batch_size=4,
        validation_batch_size=4, checkpoint_interval=1, action_noise_std=0,
    )
    return SimpleNamespace(path=path, sha=hashlib.sha256(path.read_bytes()).hexdigest(),
                           reference=reference, learning=learning, run=run)


def test_transfer_configuration_roundtrips_json_without_importing_upstream():
    # Full upstream validation belongs after source installation, at construction.
    policy = {"history_length": 5, "leg_hidden_dims": (16, 8)}
    config = LearningConfig(actor_architecture="transferred_transformer", actor_policy_config=policy)
    loaded = LearningConfig.from_dict(json.loads(json.dumps(asdict(config))))
    assert loaded == config
    assert config.actor_policy_config["leg_hidden_dims"] == [16, 8]
    policy["history_length"] = 9
    assert config.actor_policy_config["history_length"] == 5


@pytest.mark.parametrize("options,match", [
    ({"actor_architecture": "unknown"}, "actor_architecture"),
    ({"actor_policy_config": {"history_length": 5}}, "history_mlp"),
    ({"actor_architecture": "transferred_transformer"}, "requires"),
    ({"actor_architecture": "transferred_transformer",
      "actor_policy_config": {"history_length": 4}}, "history_length"),
    ({"actor_architecture": "transferred_transformer",
      "actor_policy_config": {"history_length": 5, "mean_init_scale": float("nan")}}, "finite JSON"),
])
def test_transfer_configuration_rejects_inconsistent_settings(options, match):
    with pytest.raises(ValueError, match=match):
        LearningConfig(**options)


def test_initial_teacher_is_used_for_first_collection_without_ppo_state(
        transfer, tmp_path, monkeypatch):
    env = SyntheticEnvironment(4, seed=transfer.run.seed)
    run = TrainingRun(env, tmp_path / "transfer", transfer.run, transfer.learning,
                      initial_actor=transfer.path, initial_actor_sha256=transfer.sha)
    baseline = TrainingRun(SyntheticEnvironment(4, seed=transfer.run.seed), tmp_path / "fresh",
                           transfer.run, transfer.learning)
    assert isinstance(run.learner.actor, TransferredActor)
    for name in ("world", "critic"):
        for key, value in getattr(baseline.learner, name).state_dict().items():
            torch.testing.assert_close(getattr(run.learner, name).state_dict()[key], value, rtol=0, atol=0)
    assert (run.learner.world_updates, run.learner.actor_updates, run.learner.critic_updates) == (0, 0, 0)
    assert not run.learner.actor_optimizer.state
    assert not run.learner.critic_optimizer.state
    assert not run.learner.normalizers_fitted
    bounds = torch.tensor(ACTION_BOUNDS)
    expected = transfer.reference(run.collector.history).clamp(-bounds, bounds)
    observed = []
    original_step = env.step

    def step(action):
        observed.append(action.clone())
        return original_step(action)

    monkeypatch.setattr(env, "step", step)
    run.collector.collect(1)
    torch.testing.assert_close(observed[0], expected, rtol=0, atol=0)
    metadata = run.learner.actor_source_metadata
    assert metadata["initial_checkpoint"]["sha256"] == transfer.sha
    manifest = json.loads((run.output / "manifest-000.json").read_text())
    assert manifest["actor_source_metadata"] == json.loads(json.dumps(metadata))
    metadata["initial_checkpoint"]["sha256"] = "0" * 64
    assert run.learner.actor_source_metadata["initial_checkpoint"]["sha256"] == transfer.sha
    with pytest.raises(ValueError, match="initialized or optimized"):
        run.learner.load_initial_actor(transfer.path, transfer.sha)


def test_transferred_actor_trains_and_resumes_without_teacher_file(transfer, tmp_path, monkeypatch):
    run = TrainingRun(SyntheticEnvironment(4, seed=transfer.run.seed), tmp_path / "train",
                      transfer.run, transfer.learning,
                      initial_actor=transfer.path, initial_actor_sha256=transfer.sha)
    initial = {key: value.clone() for key, value in run.learner.actor.named_parameters()}
    run.run()
    assert run.learner.actor_updates == run.learner.critic_updates == 1
    assert any(not torch.equal(initial[key], value) for key, value in run.learner.actor.named_parameters())
    assert all(torch.isfinite(value).all() for value in run.learner.actor.parameters())
    assert run.learner.actor_optimizer.state
    source = run.learner.actor_source_metadata
    checkpoint = run.output / "checkpoint.pt"
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["learner"]["actor_source_metadata"] == source
    assert not any("critic" in key or "prediction_head" in key or "log_std" in key
                   for key in payload["learner"]["actor"])
    transfer.path.unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("A SuperMimic resume must not reload the upstream teacher")

    monkeypatch.setattr(TransferredActor, "load_initial_checkpoint", forbidden)
    config = LearningConfig.from_dict(json.loads(json.dumps(payload["learner"]["config"])))
    resumed = TrainingRun(SyntheticEnvironment(4, seed=transfer.run.seed), tmp_path / "resume",
                          replace(transfer.run, updates=2), config, resume=checkpoint)
    for key, value in run.learner.actor.state_dict().items():
        torch.testing.assert_close(resumed.learner.actor.state_dict()[key], value, rtol=0, atol=0)
    for old, new in zip(run.learner.actor_optimizer.state.values(),
                        resumed.learner.actor_optimizer.state.values()):
        for key in old:
            torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
    assert resumed.learner.actor_source_metadata == source
    resumed.run()
    assert resumed.learner.actor_updates == resumed.learner.critic_updates == 2
    assert resumed.learner.actor_source_metadata == source
    assert json.loads((resumed.output / "manifest-001.json").read_text())["actor_source_metadata"] == (
        json.loads(json.dumps(source)))
    report = evaluate(SyntheticEnvironment(4, seed=transfer.run.seed), resumed.output / "checkpoint.pt",
                      tmp_path / "evaluation", steps=2)
    assert report["actor_updates"] == 2
    assert report["deterministic_actor"] is True


@pytest.mark.parametrize("field,value", [("source_sha256", "0" * 64),
                                         ("normalization", "running_statistics")])
def test_resume_rejects_changed_actor_source_before_mutating_weights(transfer, field, value):
    learner = Learner(SyntheticEnvironment(2).schema, transfer.learning)
    learner.load_initial_actor(transfer.path, transfer.sha)
    saved = deepcopy(learner.state_dict())
    saved["actor_source_metadata"][field] = value
    restored = Learner(learner.schema, transfer.learning)
    before = {key: value.clone() for key, value in restored.actor.state_dict().items()}
    with pytest.raises(ValueError, match="source metadata differs"):
        restored.load_state_dict(saved)
    assert all(torch.equal(before[key], value) for key, value in restored.actor.state_dict().items())
    saved.pop("actor_source_metadata")
    with pytest.raises(ValueError, match="requires source metadata"):
        restored.load_state_dict(saved)


def test_world_fit_and_actor_transfer_can_initialize_together(transfer, tmp_path):
    fitted = TrainingRun(SyntheticEnvironment(4), tmp_path / "fit", transfer.run, transfer.learning)
    fitted.collector.collect(8)
    batch = fitted.replay.sample(4, 2)
    fitted.learner.fit_normalizers(batch)
    fitted.learner.fit_world(batch)
    world = tmp_path / "world-model.pt"
    torch.save({"format": "supermimic-rl-world-fit-v1", "learner": fitted.learner.state_dict()}, world)
    initialized = TrainingRun(SyntheticEnvironment(4), tmp_path / "initialized", transfer.run,
                              transfer.learning, initial_world=world,
                              initial_actor=transfer.path, initial_actor_sha256=transfer.sha)
    for key, value in fitted.learner.world.state_dict().items():
        torch.testing.assert_close(initialized.learner.world.state_dict()[key], value, rtol=0, atol=0)
    assert initialized.learner.world_updates == 1
    assert initialized.learner.actor_updates == initialized.learner.critic_updates == 0
    assert initialized.learner.normalizers_fitted
    assert not initialized.learner.actor_optimizer.state
    assert initialized.learner.actor_source_metadata["initial_checkpoint"]["sha256"] == transfer.sha


@pytest.mark.parametrize("options,match", [
    ({"initial_actor": "teacher.pt"}, "provided together"),
    ({"initial_actor_sha256": "a" * 64}, "provided together"),
    ({"resume": "resume.pt", "initial_actor": "teacher.pt", "initial_actor_sha256": "a" * 64},
     "full resume"),
    ({"initial_actor": "teacher.pt", "initial_actor_sha256": "a" * 64}, "transferred_transformer"),
])
def test_invalid_initialization_rejected_before_environment_or_output(tmp_path, options, match):
    environment = SyntheticEnvironment(4)
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match=match):
        TrainingRun(environment, output, **options)
    assert environment.current is None
    assert not output.exists()


@pytest.mark.parametrize("command", ["train", "collect"])
def test_cli_forwards_initial_actor_pins(transfer, tmp_path, monkeypatch, command):
    config = tmp_path / "learning.json"
    config.write_text(json.dumps({"run": asdict(transfer.run), "learning": asdict(transfer.learning)}))
    factory_config = tmp_path / "factory.json"
    factory_config.write_text("{}")
    received = []
    adapter = SimpleNamespace(provenance={"fixture": True}, policy_dt_s=0.02, close=lambda: None)
    monkeypatch.setattr(cli.importlib, "import_module", lambda name: SimpleNamespace(create=lambda _: adapter))
    monkeypatch.setattr(cli, "native_logs", lambda _: nullcontext())
    monkeypatch.setattr(cli, "PackedEnvironment", lambda *args, **kwargs: object())

    class CapturedRun:
        validate_initialization = staticmethod(TrainingRun.validate_initialization)

        def __init__(self, environment, output, config, learning, **kwargs):
            received.append(kwargs)

        def run(self):
            return {"fixture": True}

        def collect_dataset(self, steps):
            assert steps == 9
            return {"fixture": True}

    monkeypatch.setattr(cli, "TrainingRun", CapturedRun)
    args = [command, "--config", str(config), "--factory-config", str(factory_config),
            "--factory", "fixture:create", "--output", str(tmp_path / "output"),
            "--initial-actor", str(transfer.path), "--initial-actor-sha256", transfer.sha]
    if command == "collect":
        args += ["--steps", "9"]
    assert cli.main(args) == 0
    assert received == [{"resume": None, "initial_world": None,
                         "initial_actor": transfer.path, "initial_actor_sha256": transfer.sha}]


def test_cli_rejects_resume_and_teacher_before_importing_factory(tmp_path, monkeypatch, capsys):
    config = tmp_path / "learning.json"
    config.write_text(json.dumps({"run": {}, "learning": {}}))

    def forbidden(name):
        raise AssertionError("Invalid initialization must be rejected before native imports")

    monkeypatch.setattr(cli.importlib, "import_module", forbidden)
    assert cli.main([
        "train", "--config", str(config), "--factory-config", "not-read.json",
        "--output", str(tmp_path / "output"), "--resume", "resume.pt",
        "--initial-actor", "teacher.pt", "--initial-actor-sha256", "a" * 64,
    ]) == 2
    assert "full resume" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()
