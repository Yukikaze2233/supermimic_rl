"""Student evaluation uses the same valid-sample accounting as PPO."""
from types import SimpleNamespace

import torch

from supermimic_rl.world_model_v2 import evaluation


def test_native_evaluation_excludes_invalid_physics_from_tracking_and_failures(monkeypatch):
    owner = SimpleNamespace(env=SimpleNamespace(_numerical_invalid=torch.tensor([False, True])))
    environment = SimpleNamespace(
        num_envs=2, device='cpu', policy_dt_s=.02, adapter=owner,
        current=SimpleNamespace(state=torch.zeros(2, 1)),
        schema=SimpleNamespace(state_fields=[SimpleNamespace(name='failure.failure_reasons', size=1)]))
    tracking = SimpleNamespace(actual=torch.tensor([[1., 2., 3.], [float('nan'), 100., 100.]]),
                               target=torch.zeros(2, 3))
    monkeypatch.setattr(evaluation, 'native_tracking_sample', lambda env: tracking)

    class Actor:
        def eval(self):
            return self

        def __call__(self, history):
            return torch.zeros(2, 6)

        def estimate(self, history):
            return torch.zeros(2, 5)

    class Collector:
        def __init__(self, env, decision, **kwargs):
            self.decision = decision
            self.layout = SimpleNamespace(physical=lambda state: SimpleNamespace(
                gravity_body=torch.tensor([[0., 0., -1.], [0., 0., -1.]])))

        def collect(self, steps, **kwargs):
            self.decision(torch.zeros(2, 5, 35))
            kwargs['on_transition'](SimpleNamespace(
                reward=torch.tensor([1., float('nan')]), terminated=torch.tensor([True, True]),
                truncated=torch.zeros(2, dtype=torch.bool), valid=torch.tensor([True, False]),
                end=SimpleNamespace(state=torch.tensor([[1.], [float('nan')]]))))

    monkeypatch.setattr(evaluation, 'Collector', Collector)
    report = evaluation.evaluate_native(environment, Actor(), 1)
    assert report['terminations'] == report['invalid_transitions'] == 1
    assert report['numerical_invalid_transitions'] == report['valid_tracking_samples'] == 1
    assert report['termination_masks'] == {'1': 1}
    assert report['valid_simulated_seconds'] == .02
    assert report['failures_per_valid_env_minute'] == 3000.
    assert report['rms/forward_error_m_s'] == 1.
    assert report['rms/yaw_rate_error_rad_s'] == 2.
    assert report['rms/height_error_m'] == 3.
