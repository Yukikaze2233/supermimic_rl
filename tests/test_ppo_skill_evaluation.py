"""PPO skill evaluation reads the native history the actor was trained on."""
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.ppo.policy import TeacherActor

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_ppo_skills.py"


def test_actor_uses_native_obs_history_not_the_collector_history():
    adapter_class = runpy.run_path(str(SCRIPT))["NativeHistoryActor"]
    torch.manual_seed(0)
    actor = TeacherActor((16,), (8,))
    native = torch.randn(3, 5 * 42)
    environment = SimpleNamespace(num_envs=3, adapter=SimpleNamespace(env=SimpleNamespace(obs_buf={"policy": native})))
    wrapped = adapter_class(actor, environment).eval()
    collector_history = torch.zeros(3, 5, 45)
    expected_history = native.reshape(3, 5, 42)[..., :35]
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        action = wrapped(collector_history)
        torch.testing.assert_close(action, actor.clip(actor(expected_history)))
        torch.testing.assert_close(wrapped.estimate(collector_history), actor.estimator(expected_history))
    assert torch.equal(torch.get_rng_state(), rng)
    assert wrapped.sampling_report()["mode"] == "deterministic"
    assert wrapped.sampling_report()["seed"] is None


def test_stochastic_evaluation_matches_training_normal_then_clip_without_consuming_global_rng():
    adapter_class = runpy.run_path(str(SCRIPT))["NativeHistoryActor"]
    torch.manual_seed(11)
    actor = TeacherActor((16,), (8,), action_bounds=(.25,) * 6)
    with torch.no_grad():
        actor.log_std.copy_(torch.tensor([.3, .4, .5, .6, .7, .8]).log())
    native = torch.randn(12, 5 * 42)
    environment = SimpleNamespace(num_envs=12,
        adapter=SimpleNamespace(env=SimpleNamespace(obs_buf={"policy": native})))
    collector_history = torch.full((12, 5, 45), 100.)
    native_history = native.reshape(12, 5, 42)[..., :35]
    seed = 59
    # The actual PPO distribution supplies the expected samples. fork_rng
    # isolates that reference calculation from the global-state assertion.
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(seed)
        raw = [actor.distribution(native_history).sample() for _ in range(2)]
        expected = [actor.clip(value) for value in raw]
    assert (raw[0].abs() > actor.action_bounds).any()
    before = {name: value.clone() for name, value in actor.state_dict().items()}
    rng = torch.get_rng_state().clone()
    wrapped = adapter_class(actor, environment, action_sampling="ppo-gaussian", sampling_seed=seed).eval()
    with torch.no_grad():
        actual = [wrapped(collector_history) for _ in range(2)]
        estimate = wrapped.estimate(collector_history)
    assert torch.equal(torch.get_rng_state(), rng)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    assert not torch.equal(actual[0], actual[1])
    torch.testing.assert_close(estimate, actor.estimator(native_history))
    for name, value in actor.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    report = wrapped.sampling_report()
    assert report["mode"] == "ppo-gaussian" and report["seed"] == seed
    torch.testing.assert_close(torch.tensor(report["checkpoint_std"]), actor.log_std.exp())


@pytest.mark.parametrize("options", [
    {"action_sampling": "unsupported"}, {"sampling_seed": -1},
    {"sampling_seed": 2**63}, {"sampling_seed": True},
])
def test_evaluation_rejects_ambiguous_sampling_configuration(options):
    adapter_class = runpy.run_path(str(SCRIPT))["NativeHistoryActor"]
    with pytest.raises(ValueError):
        adapter_class(None, None, **options)
