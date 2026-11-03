"""PPO teacher on a synthetic environment: learning, estimator isolation, GAE, resume and ONNX."""
import json

import pytest
import torch

from supermimic_rl.ppo.evaluation import evaluate, export_onnx
from supermimic_rl.ppo.policy import TeacherActor
from supermimic_rl.ppo.rollout import gae
from supermimic_rl.ppo.trainer import PPOConfig, PPOTrainer, load_actor
from support.ppo import TargetEnvironment, config


def test_estimator_output_is_detached_from_the_policy_gradient():
    actor = TeacherActor((16,), (8,))
    actor(torch.randn(4, 5, 35)).sum().backward()
    assert all(p.grad is None for p in actor.estimator.parameters())
    assert actor.network[0].weight.grad is not None


def test_gae_cuts_traces_at_done_steps():
    reward = torch.ones(3, 1)
    value = torch.zeros(3, 1)
    done = torch.tensor([[False], [True], [False]])
    returns, _ = gae(reward, value, done, torch.full((1,), 10.), gamma=1., lam=1.)
    torch.testing.assert_close(returns[:, 0], torch.tensor([2., 1., 11.]))


def test_external_teacher_uses_its_estimate_and_reports_partial_progress():
    environment = TargetEnvironment(num_envs=4)
    progress = []
    estimates = []
    original = environment.set_estimate

    def set_estimate(history, estimate):
        estimates.append(estimate.clone())
        original(history, estimate)

    environment.set_estimate = set_estimate

    def decision(history):
        action = torch.zeros(history.shape[0], 6)
        action[:, 0] = history[:, -1, 0]
        return action, torch.full((history.shape[0], 5), 7.)

    report = evaluate(environment, None, 5, decision=decision, on_progress=progress.append,
                      progress_interval=2)
    assert report["steps"] == 5
    assert [row["completed_steps"] for row in progress] == [2, 4, 5]
    assert len(estimates) == 5 and all(torch.all(value == 7) for value in estimates)


def test_teacher_holds_course_without_losing_real_exposure(tmp_path):
    environment = TargetEnvironment(num_envs=4)
    trainer = PPOTrainer(environment, config(updates=2, curriculum_transition_cap=0), tmp_path)
    trainer.train()
    assert trainer.transitions == 160
    assert all(transitions == 0 for transitions, _ in environment.progress)
    assert trainer.state_dict()["transitions"] == 160
    with pytest.raises(ValueError, match="curriculum_transition_cap"):
        config(curriculum_transition_cap=-1)


@pytest.mark.parametrize("value", [-1, 1.5, True, None])
def test_course_offset_requires_nonnegative_integer(value):
    with pytest.raises(ValueError, match="curriculum_transition_offset"):
        config(curriculum_transition_offset=value)


def test_course_offset_applies_before_cap_and_preserves_real_counter(tmp_path):
    environment = TargetEnvironment(num_envs=4)
    trainer = PPOTrainer(environment, config(updates=2, curriculum_transition_offset=40,
                                            curriculum_transition_cap=60), tmp_path)
    trainer.train()
    assert trainer.transitions == 160
    assert environment.progress[0] == (0, 0)
    assert environment.progress[-1] == (60, 1)
    assert all(a <= b for (a, _), (b, _) in zip(environment.progress, environment.progress[1:]))
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [record["curriculum_transitions"] for record in records] == [40, 60]
    assert [record["transitions"] for record in records] == [80, 160]
    assert all(record["curriculum_transition_offset"] == 40
               and record["curriculum_transition_cap"] == 60 for record in records)


def test_foundation_resume_restarts_course_without_resetting_learning_state(tmp_path):
    original = PPOTrainer(TargetEnvironment(num_envs=4),
                          config(updates=2, curriculum_transition_cap=0), tmp_path / "foundation")
    original.train()
    saved = original.state_dict()
    # The deployed foundation checkpoint predates the new offset field.
    del saved["config"]["curriculum_transition_offset"]
    checkpoint = tmp_path / "old-foundation.pt"
    torch.save(saved, checkpoint)
    assert PPOConfig.from_dict(saved["config"]).curriculum_transition_offset == 0
    environment = TargetEnvironment(num_envs=4, seed=9)
    resumed = PPOTrainer(environment, config(updates=3, curriculum_transition_offset=saved["transitions"]),
                         tmp_path / "full-course", resume=checkpoint)
    assert resumed.curriculum_transitions == 0
    actual = resumed.state_dict()
    for name in ("actor", "critic", "optimizer", "estimator_optimizer", "learning_rate",
                 "updates", "transitions", "torch_rng", "cuda_rng"):
        _assert_learning_state_equal(actual[name], saved[name])
    resumed.train()
    assert environment.progress[0] == (0, 2)
    assert environment.progress[-1] == (80, 2)
    assert (resumed.updates, resumed.transitions, resumed.curriculum_transitions) == (3, 240, 80)
    continued = torch.load(tmp_path / "full-course/checkpoint.pt", weights_only=True)
    assert continued["config"]["curriculum_transition_offset"] == 160
    same_course = PPOTrainer(TargetEnvironment(num_envs=4),
                            PPOConfig.from_dict(continued["config"]), tmp_path / "again",
                            resume=tmp_path / "full-course/checkpoint.pt")
    assert same_course.curriculum_transitions == 80


def test_full_course_continuation_budget_covers_all_stages():
    import math
    from pathlib import Path

    root = Path(__file__).parents[1]
    continuation = PPOConfig.from_dict(json.loads(
        (root / "configs/gpufree/ppo-formal35-full-course-resume2000.json").read_text()))
    course = json.loads((root / "configs/gpufree/ppo-course-jump-first.json").read_text())
    needed = sum(course["base"]["stage_transition_budgets"])
    per_update = 4096 * continuation.steps_per_update
    assert continuation.curriculum_transition_offset == 2000 * per_update
    assert continuation.updates == 2000 + math.ceil(needed / per_update)
    assert continuation.curriculum_transition_cap is None


def test_critic_normalization_is_frozen_inside_rollout(tmp_path):
    environment = TargetEnvironment(num_envs=4)
    trainer = PPOTrainer(environment, config(updates=1), tmp_path)
    original = trainer.critic.norm.update
    calls = []

    def record(values):
        calls.append(values.clone())
        original(values)

    trainer.critic.norm.update = record
    trainer.collect()
    assert len(calls) == 0
    trainer.collect()
    assert len(calls) == 1
    assert trainer.critic.norm.count == environment.num_envs * trainer.config.steps_per_update


@pytest.mark.parametrize("operation", ["train", "evaluate"])
def test_native_reports_survive_interpreter_exit_on_close(tmp_path, monkeypatch, operation):
    import importlib.util
    from contextlib import nullcontext
    from pathlib import Path
    from types import SimpleNamespace
    from supermimic_rl import cli

    spec = importlib.util.spec_from_file_location(
        "train_ppo_entry", Path(__file__).parents[1] / "scripts/train_ppo.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)

    class ExitingEnvironment(TargetEnvironment):
        adapter = SimpleNamespace(provenance={"test_environment": "target"})

        def close(self):
            raise SystemExit(0)

    monkeypatch.setattr(cli, "native_logs", lambda path: nullcontext())
    monkeypatch.setattr(entry, "_native_environment", lambda *args: ExitingEnvironment(num_envs=4))
    checkpoint = tmp_path / "seed.pt"
    seed = PPOTrainer(TargetEnvironment(num_envs=4), config(updates=1), tmp_path / "seed")
    seed.save(checkpoint)
    configuration = tmp_path / "config.json"
    from dataclasses import asdict
    configuration.write_text(json.dumps(asdict(config(updates=1))))
    output = tmp_path / operation
    args = SimpleNamespace(output=output, resume=None, config=configuration, factory_config=tmp_path / "fake",
                           factory="unused", checkpoint=checkpoint, steps=5)
    with pytest.raises(SystemExit):
        getattr(entry, operation)(args)
    report = json.loads((output / ("summary.json" if operation == "train" else "evaluation.json")).read_text())
    assert report["updates" if operation == "train" else "steps"] == (1 if operation == "train" else 5)


def test_ppo_improves_tracking_and_trains_the_estimator(tmp_path):
    torch.manual_seed(0)
    environment = TargetEnvironment()
    trainer = PPOTrainer(environment, config(), tmp_path / "run")
    trainer.train()
    records = [json.loads(line) for line in (tmp_path / "run/metrics.jsonl").read_text().splitlines()]
    # Negative tracking reward: at least halve the initial error.
    assert records[-1]["reward_per_step"] > .5 * records[0]["reward_per_step"]
    assert records[-1]["estimator"] < .1 * records[0]["estimator"]
    # Progress reports completed actor updates; the last collection precedes update 40.
    assert environment.progress[-1] == (40 * 20 * 64, 39)
    assert environment.estimates == 40 * 20
    actor = load_actor(tmp_path / "run/checkpoints/update-000040.pt")
    report = evaluate(TargetEnvironment(seed=3), actor, 20)
    assert report["rms/forward_error_m_s"] < .3 and report["steps"] == 20


def test_resume_restores_learning_state_and_continues(tmp_path):
    torch.manual_seed(1)
    first = PPOTrainer(TargetEnvironment(), config(updates=20), tmp_path / "run")
    first.train()
    saved = first.state_dict()
    resumed = PPOTrainer(TargetEnvironment(seed=1), config(updates=25), tmp_path / "run",
                         resume=tmp_path / "run/checkpoint.pt")
    assert (resumed.updates, resumed.transitions, resumed.learning_rate) == (20, saved["transitions"],
                                                                          saved["learning_rate"])
    for name, value in saved["actor"].items():
        torch.testing.assert_close(resumed.actor.state_dict()[name], value)
    resumed.train()
    assert resumed.updates == 25
    with pytest.raises(ValueError, match="network shapes"):
        PPOTrainer(TargetEnvironment(), config(actor_hidden=(32,)), tmp_path / "other",
                   resume=tmp_path / "run/checkpoint.pt")
    with pytest.raises(ValueError, match="full PPO"):
        PPOTrainer(TargetEnvironment(), config(), tmp_path / "light",
                   resume=tmp_path / "run/checkpoints/update-000020.pt")


def test_onnx_export_matches_the_clipped_deterministic_actor(tmp_path):
    onnxruntime = pytest.importorskip("onnxruntime")
    actor = TeacherActor((32, 16), (8,))
    path = export_onnx(actor, tmp_path / "teacher.onnx")
    history = torch.randn(1, 5, 35) * 3
    session = onnxruntime.InferenceSession(str(path))
    (output,) = session.run(None, {"history": history.numpy()})
    torch.testing.assert_close(torch.from_numpy(output), actor.clip(actor(history)).detach(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("operation", ["train", "evaluate"])
@pytest.mark.parametrize("exit_on_close", [False, True])
def test_native_failure_is_persisted_before_close(tmp_path, monkeypatch, operation, exit_on_close):
    import importlib.util
    from contextlib import nullcontext
    from pathlib import Path
    from types import SimpleNamespace
    from supermimic_rl import cli

    spec = importlib.util.spec_from_file_location(
        "train_ppo_failure_entry", Path(__file__).parents[1] / "scripts/train_ppo.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    output = tmp_path / operation
    closed = []

    class FailedEnvironment(TargetEnvironment):
        def step(self, action):
            raise RuntimeError("native PPO collection sentinel")

        def close(self):
            # The error must already be durable when Isaac attempts shutdown.
            text = (output / "failure-traceback.txt").read_text()
            assert "RuntimeError: native PPO collection sentinel" in text
            closed.append(True)
            if exit_on_close:
                raise SystemExit(0)

    monkeypatch.setattr(cli, "native_logs", lambda path: nullcontext())
    monkeypatch.setattr(entry, "_native_environment", lambda *args: FailedEnvironment(num_envs=4))
    checkpoint = tmp_path / "seed.pt"
    PPOTrainer(TargetEnvironment(num_envs=4), config(updates=1), tmp_path / "seed").save(checkpoint)
    from dataclasses import asdict
    configuration = tmp_path / "config.json"
    configuration.write_text(json.dumps(asdict(config(updates=1))))
    args = SimpleNamespace(output=output, resume=None, config=configuration, factory_config=tmp_path / "fake",
                           factory="unused", checkpoint=checkpoint, steps=5)
    if exit_on_close:
        with pytest.raises(SystemExit):
            getattr(entry, operation)(args)
    else:
        with pytest.raises(RuntimeError, match="native PPO collection sentinel"):
            getattr(entry, operation)(args)
    assert closed == [True]
    assert not (output / "summary.json").exists()
    assert not (output / "evaluation.json").exists()


def test_invalid_transition_bootstraps_the_preceding_action_at_last_trusted_state():
    reward = torch.tensor([[1., 2.], [float('nan'), 3.], [4., float('nan')], [5., 6.]])
    value = torch.tensor([[10., 20.], [12., 22.], [30., 40.], [32., 42.]])
    valid = torch.tensor([[True, True], [False, True], [True, False], [True, True]])
    returns, advantages = gae(reward, value, ~valid, torch.tensor([50., 60.]), .9, .8, valid=valid)
    expected = torch.tensor([[1.8, 14.04], [0., 17.], [15.76, 0.], [18., 18.]])
    torch.testing.assert_close(advantages, expected, atol=3e-6, rtol=1e-6)
    torch.testing.assert_close(returns[~valid], value[~valid], atol=0, rtol=0)
    assert returns[0, 0] == pytest.approx(1 + .9 * 12)
    # A different post-reset tail cannot change either trajectory before its
    # censored interval; there is no zero-terminal or tail leakage.
    changed = reward.clone()
    changed[2:, 0] = 1000.
    changed[3, 1] = -1000.
    _, other = gae(changed, value, ~valid, torch.tensor([-100., 900.]), .9, .8, valid=valid)
    torch.testing.assert_close(other[0], advantages[0], atol=0, rtol=0)
    torch.testing.assert_close(other[1, 1], advantages[1, 1], atol=0, rtol=0)


def test_invalid_rows_never_enter_any_ppo_loss_or_advantage_normalization(tmp_path):
    from copy import deepcopy

    first = PPOTrainer(TargetEnvironment(num_envs=8), config(updates=1), tmp_path / 'clean')
    rollout = first.collect()
    rollout.valid[3:5, 2] = False
    second = PPOTrainer(TargetEnvironment(num_envs=8), config(updates=1), tmp_path / 'poison')
    second.actor.load_state_dict(first.actor.state_dict())
    second.critic.load_state_dict(first.critic.state_dict())
    second.observation = first.observation
    poisoned = deepcopy(rollout)
    for name in ('history', 'critic', 'action', 'log_prob', 'mean', 'std', 'reward', 'label'):
        getattr(poisoned, name)[~poisoned.valid] = float('nan')
    # Value is the trusted pre-action baseline, not part of the failed endpoint.
    torch.manual_seed(113)
    expected = first.update(rollout)
    torch.manual_seed(113)
    actual = second.update(poisoned)
    assert actual == expected
    assert actual['optimization_samples'] == 158
    assert actual['optimization_skipped'] is False
    for left, right in ((first.actor, second.actor), (first.critic, second.critic)):
        for name, parameter in left.state_dict().items():
            torch.testing.assert_close(parameter, right.state_dict()[name], atol=0, rtol=0)


def test_all_invalid_rollout_has_no_optimizer_or_estimator_step(tmp_path):
    trainer = PPOTrainer(TargetEnvironment(num_envs=4), config(updates=1), tmp_path)
    rollout = trainer.collect()
    rollout.valid.zero_()
    rollout.reward.fill_(float('nan'))
    before = {name: tensor.clone() for name, tensor in trainer.actor.state_dict().items()}
    result = trainer.update(rollout)
    assert result['optimization_samples'] == 0
    assert result['optimization_skipped'] is True
    assert not trainer.optimizer.state and not trainer.estimator_optimizer.state
    for name, parameter in trainer.actor.state_dict().items():
        torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)


def test_collection_excludes_invalid_endpoints_and_separates_numerical_failures(tmp_path):
    from dataclasses import replace

    class InvalidEnvironment(TargetEnvironment):
        def step(self, action):
            result = super().step(action)
            valid = torch.ones(self.num_envs, dtype=torch.bool)
            valid[0] = False
            # Row zero supplies no trustworthy endpoint, reward or tracking.
            result.terminal_critic[0] = float('nan')
            result.reward[0] = float('nan')
            for value in result.tracking.values():
                value[0] = float('nan')
            terminated = torch.zeros_like(valid)
            terminated[0] = True
            return replace(result, terminated=terminated, truncated=torch.zeros_like(valid),
                           valid=valid, numerical_invalid=~valid)

    trainer = PPOTrainer(InvalidEnvironment(num_envs=4), config(updates=1, steps_per_update=4), tmp_path)
    samples = []
    hook = trainer.critic.register_forward_pre_hook(lambda module, args: samples.append(args[0].clone()))
    rollout = trainer.collect()
    hook.remove()
    assert all(torch.isfinite(values).all() for values in samples)
    assert not rollout.valid[:, 0].any()
    assert not rollout.label_valid[:, 0].any()
    assert rollout.done[:, 0].all()
    assert trainer._normalization_samples.shape[0] == 12
    summary = trainer.statistics.summary()
    assert summary['terminations'] == 0
    assert summary['invalid_transitions'] == 4
    assert summary['numerical_invalid_transitions'] == 4
    assert summary['valid_tracking_samples'] == 12
    assert 'episode/count' not in summary
    assert torch.isfinite(torch.tensor(summary['rms/forward_error_m_s']))
    result = trainer.update(rollout)
    assert result['optimization_samples'] == 12 and not result['optimization_skipped']


def _assert_learning_state_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for name in expected:
            _assert_learning_state_equal(actual[name], expected[name])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_learning_state_equal(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize("failure_phase", ["collection", "optimization"])
def test_failure_checkpoint_keeps_only_the_last_completed_update(tmp_path, failure_phase):
    from copy import deepcopy

    failure = RuntimeError("interrupted PPO " + failure_phase)

    class InterruptedEnvironment(TargetEnvironment):
        calls = 0

        def step(self, action):
            self.calls += 1
            result = super().step(action)
            if failure_phase == "collection" and self.calls == 6:
                torch.rand(3)
                raise failure
            return result

    configuration = config(updates=3, steps_per_update=4, epochs=1, minibatches=2, checkpoint_interval=100)
    trainer = PPOTrainer(InterruptedEnvironment(num_envs=4), configuration, tmp_path / "failed")
    expected = {}
    update, minibatch = trainer.update, trainer._minibatch

    def record_completed(rollout):
        metrics = update(rollout)
        expected.update(deepcopy(trainer.state_dict()))
        return metrics

    def interrupted_minibatch(batch):
        metrics = minibatch(batch)
        if failure_phase == "optimization" and trainer.updates == 1:
            torch.rand(3)
            raise failure
        return metrics

    trainer.update, trainer._minibatch = record_completed, interrupted_minibatch
    with pytest.raises(RuntimeError) as caught:
        trainer.train()
    assert caught.value is failure
    path = tmp_path / "failed/failure-checkpoint.pt"
    payload = torch.load(path, weights_only=True)
    recovery = payload.pop("recovery")
    _assert_learning_state_equal(payload, expected)
    assert (payload["updates"], payload["transitions"]) == (1, 16)
    assert recovery["completed_update"] == 1
    assert recovery["discarded_transitions"] == (4 if failure_phase == "collection" else 16)
    assert recovery["exception_message"] == str(failure)
    assert not (tmp_path / "failed/checkpoint.pt").exists()
    assert not torch.equal(torch.get_rng_state(), payload["torch_rng"])
    assert trainer.critic.norm.count > payload["critic"]["norm.count"]
    if failure_phase == "optimization":
        assert any(not torch.equal(value, payload["actor"][name])
                   for name, value in trainer.actor.state_dict().items())
        assert next(iter(trainer.optimizer.state.values()))["step"] > next(
            iter(payload["optimizer"]["state"].values()))["step"]

    fresh_environment = TargetEnvironment(num_envs=4, seed=19)
    resumed = PPOTrainer(fresh_environment, configuration, tmp_path / "resumed", resume=path)
    _assert_learning_state_equal(resumed.state_dict(), payload)
    assert torch.count_nonzero(fresh_environment.steps) == 0
    assert resumed._normalization_samples is None
    assert payload["resume_semantics"]["environment_continuation"] is False
    assert resumed.train()["updates"] == 3


def test_first_update_failure_does_not_claim_a_completed_checkpoint(tmp_path):
    trainer = PPOTrainer(TargetEnvironment(num_envs=4), config(updates=1), tmp_path)

    def fail():
        raise RuntimeError("no complete update")

    trainer.collect = fail
    with pytest.raises(RuntimeError, match="no complete update"):
        trainer.train()
    assert not (tmp_path / "failure-checkpoint.pt").exists()


def test_failure_checkpoint_write_error_preserves_the_training_exception(tmp_path, monkeypatch):
    from supermimic_rl.ppo import trainer as trainer_module

    trainer = PPOTrainer(TargetEnvironment(num_envs=4), config(updates=3, steps_per_update=4), tmp_path)
    collect = trainer.collect
    failure = RuntimeError("original collection failure")

    def fail_after_update():
        if trainer.updates:
            raise failure
        return collect()

    def failed_save(*args):
        raise OSError("checkpoint disk unavailable")

    trainer.collect = fail_after_update
    monkeypatch.setattr(trainer_module, "atomic_save", failed_save)
    with pytest.raises(RuntimeError) as caught:
        trainer.train()
    assert caught.value is failure
    assert any("checkpoint disk unavailable" in note for note in failure.__notes__)
    assert not (tmp_path / "failure-checkpoint.pt").exists()
