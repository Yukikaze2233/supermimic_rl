"""Native regression guard: keep the best actor, restore it after a sustained drop, halve the rate."""
import pytest
import torch

from supermimic_rl.world_model_v2.guard import NativeRegressionGuard


def actor_and_optimizer():
    actor = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    return actor, optimizer


def step(actor, optimizer, value):
    optimizer.zero_grad()
    (actor.weight.sum() * value).backward()
    optimizer.step()


def test_sustained_regression_restores_the_best_actor_and_halves_the_rate():
    actor, optimizer = actor_and_optimizer()
    guard = NativeRegressionGuard(.1, 2, 1e-3)
    best = guard.snapshot(actor, optimizer)
    assert guard.observe(.09, best, actor, optimizer)['best_student_reward'] == .09
    step(actor, optimizer, 1.)
    drift = guard.snapshot(actor, optimizer)
    assert guard.observe(.088, drift, actor, optimizer)['bad_cycles'] == 0     # within 10%
    step(actor, optimizer, 1.)
    first = guard.observe(.05, guard.snapshot(actor, optimizer), actor, optimizer)
    assert first['bad_cycles'] == 1 and not first['restored']
    step(actor, optimizer, 1.)
    second = guard.observe(.04, guard.snapshot(actor, optimizer), actor, optimizer)
    assert second['restored'] and second['restorations'] == 1 and second['actor_learning_rate'] == 5e-4
    torch.testing.assert_close(actor.weight.detach(), best['actor']['weight'])
    assert optimizer.param_groups[0]['lr'] == 5e-4
    restored = NativeRegressionGuard(.1, 2, 1e-3)
    restored.load_state_dict(guard.state_dict())
    assert restored.learning_rate() == 5e-4 and restored.best_reward == .09


def test_guard_ignores_missing_student_rows_and_validates_settings():
    actor, optimizer = actor_and_optimizer()
    guard = NativeRegressionGuard(.1, 1, 1e-3)
    assert guard.observe(None, guard.snapshot(actor, optimizer), actor, optimizer)['best_student_reward'] is None
    with pytest.raises(ValueError):
        NativeRegressionGuard(0., 1, 1e-3)
