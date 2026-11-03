"""PPO skill evaluation reads the native history the actor was trained on."""
from pathlib import Path
import runpy
from types import SimpleNamespace

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
    with torch.no_grad():
        action = wrapped(collector_history)
        torch.testing.assert_close(action, actor.clip(actor(expected_history)))
        torch.testing.assert_close(wrapped.estimate(collector_history), actor.estimator(expected_history))
