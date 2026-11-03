"""Staged collection/offline fit/online initialization and observation clocks."""
from dataclasses import asdict
import json

import pytest
import torch

from supermimic_rl.adapters.synthetic import SyntheticEnvironment
from supermimic_rl.cli import main
from supermimic_rl.learning import LearningConfig
from supermimic_rl.models import FixedNormalizer
from supermimic_rl.runtime import RunConfig, TrainingRun


def test_frozen_collection_offline_fit_and_online_initialization(tmp_path):
    torch.set_num_threads(1)
    config = RunConfig(updates=2, warmup_steps=8, replay_steps=16, collection_steps=2,
                       model_batches=1, model_batch_size=4, imagination_batch_size=4,
                       validation_batch_size=4, checkpoint_interval=1)
    learning = LearningConfig(model_hidden=(8,), actor_hidden=(8,), critic_hidden=(8,),
                               history_length=2, fit_horizon=2, imagination_horizon=2,
                               critic_epochs=1, critic_minibatches=1)
    captured = TrainingRun(SyntheticEnvironment(4), tmp_path / "capture", config, learning)
    before = {name: value.clone() for name, value in captured.learner.actor.state_dict().items()}
    result = captured.collect_dataset(12)
    assert result["transitions"] == 48
    assert captured.learner.actor_updates == captured.learner.world_updates == 0
    assert all(torch.equal(before[name], value) for name, value in captured.learner.actor.state_dict().items())
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"run": asdict(config), "learning": asdict(learning)}))
    fit = tmp_path / "fit"
    assert main(["fit-world", "--config", str(config_path), "--replay", result["dataset"],
                 "--output", str(fit)]) == 0
    payload = torch.load(fit / "world-model.pt", weights_only=True)
    resumed = TrainingRun(SyntheticEnvironment(4), tmp_path / "online", config, learning,
                           initial_world=fit / "world-model.pt")
    assert all(torch.equal(value, resumed.learner.world.state_dict()[name])
               for name, value in payload["learner"]["world"].items())
    result = resumed.run()
    assert result["updates"] == 2
    assert result["transitions"] == 4 * (config.warmup_steps + 2 * config.collection_steps)
    assert resumed.learner.world_updates == 4
    assert resumed.learner.actor_updates == 2


def test_empty_context_normalizer_remains_differentiable():
    normalizer = FixedNormalizer(0)
    data = torch.empty(4, 3, 0)
    normalizer.fit(data)
    assert normalizer(data).shape == data.shape


def test_imagined_command_clock_advances_and_saturates(tmp_path):
    config = RunConfig(updates=1, warmup_steps=8, replay_steps=16,
                       model_batch_size=4, imagination_batch_size=4)
    learning = LearningConfig(model_hidden=(8,), actor_hidden=(8,), critic_hidden=(8,),
                               history_length=1, fit_horizon=2, imagination_horizon=3,
                               critic_epochs=1, critic_minibatches=1)
    run = TrainingRun(SyntheticEnvironment(4), tmp_path / "clock", config, learning)
    run.collector.collect(8)
    samples = run.replay.sample(4, 2)
    run.learner.fit_normalizers(samples)
    samples.goals[:, :, 10] = 4.99
    _, imagined = run.learner.imagine(samples)
    assert torch.allclose(imagined["goals"][:, :, 10],
                          torch.tensor([4.99, 5.0, 5.0])[:, None].expand(-1, 4))


def test_collect_refuses_silent_dataset_overwrite_by_ring(tmp_path):
    run = TrainingRun(SyntheticEnvironment(4), tmp_path / "bounded",
                       RunConfig(warmup_steps=32, replay_steps=32), LearningConfig())
    with pytest.raises(ValueError, match="retain"):
        run.collect_dataset(33)
