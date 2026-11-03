"""Variable rollout boundaries and nonfinite optimizer recovery."""
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.world_model_v2.learning import Learner, LearningConfig, lambda_returns


def test_lambda_return_respects_each_boundary_and_actual_failure():
    rollout = SimpleNamespace(
        values=torch.tensor([[1., 2.], [2., 3.], [3., 4.], [4., 5.]]),
        rewards=torch.tensor([[.5, 1.], [1000., 2.], [1000., 1000.]]),
        continuation=torch.tensor([[1., 1.], [1., 0.], [1., 1.]]),
        active=torch.tensor([[True, True], [False, True], [False, False]]),
        boundary=torch.tensor([[True, False], [False, False], [False, True]]))
    targets = lambda_returns(rollout, .9, .8)
    torch.testing.assert_close(targets[rollout.active], torch.tensor([2.3, 2.98, 2.]))


@pytest.mark.parametrize("problem", ["nonfinite_loss", "nonfinite_gradient"])
def test_bad_update_does_not_poison_parameters_or_adam_and_next_step_recovers(problem):
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    learner = Learner.__new__(Learner)
    learner.config = LearningConfig()
    optimizer = torch.optim.Adam([parameter], lr=1e-3)
    learner.optimizers = {"actor": optimizer}
    learner.updates, learner.skipped = {"actor": 0}, {"actor": 0}
    assert learner._step("actor", parameter.square().sum())
    before = parameter.detach().clone()
    moments = {name: value.clone() for name, value in optimizer.state[parameter].items()}
    if problem == "nonfinite_gradient":
        handle = parameter.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
        loss = parameter.square().sum()
    else:
        loss = parameter.sum() * float("nan")
    assert not learner._step("actor", loss)
    if problem == "nonfinite_gradient":
        handle.remove()
    torch.testing.assert_close(parameter, before)
    assert parameter.grad is None
    for name, value in moments.items():
        torch.testing.assert_close(optimizer.state[parameter][name], value)
    assert learner._step("actor", parameter.square().sum())
    assert learner.updates == {"actor": 2} and learner.skipped == {"actor": 1}
    assert torch.isfinite(parameter).all() and not torch.equal(parameter, before)
