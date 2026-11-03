"""Native regression guard: keep the best actor, restore it after a sustained drop, halve the rate."""
import copy

import pytest
import torch

from supermimic_rl.world_model_v2.guard import NativeRegressionGuard
from supermimic_rl.world_model_v2.policy import EstimatorActor


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


def estimated_actor_and_optimizers():
    actor = EstimatorActor(history_length=1, frame_dim=35, estimator_hidden=(4,), actor_hidden=(4,))
    return (actor, torch.optim.Adam(actor.decoder.parameters(), lr=1e-3),
            torch.optim.Adam(actor.estimator.parameters(), lr=2e-3))


def update_estimated_actor(actor, optimizer, estimator_optimizer, value):
    frames = torch.arange(70, dtype=torch.float32).reshape(2, 1, 35) / 70
    optimizer.zero_grad()
    (actor(frames) - value).square().mean().backward()
    optimizer.step()
    estimator_optimizer.zero_grad()
    actor.estimator_loss(frames, torch.full((2, 3), value), torch.zeros(2, 2)).backward()
    estimator_optimizer.step()


def assert_optimizer_state(optimizer, expected):
    actual = optimizer.state_dict()
    assert actual['param_groups'] == expected['param_groups']
    assert actual['state'].keys() == expected['state'].keys()
    for key, values in expected['state'].items():
        assert actual['state'][key].keys() == values.keys()
        for name, value in values.items():
            torch.testing.assert_close(actual['state'][key][name], value, rtol=0, atol=0)


def test_joint_actor_estimator_restore_matches_both_adams_and_keeps_best_snapshot_immutable():
    actor, optimizer, estimator_optimizer = estimated_actor_and_optimizers()
    for target in (1., -.5):
        update_estimated_actor(actor, optimizer, estimator_optimizer, target)
    guard = NativeRegressionGuard(.1, 1, 1e-3)
    best = guard.snapshot(actor, optimizer, estimator_optimizer=estimator_optimizer)
    frozen_best = copy.deepcopy(best)
    guard.observe(1., best, actor, optimizer, estimator_optimizer=estimator_optimizer)
    for target in (-2., 3.):
        update_estimated_actor(actor, optimizer, estimator_optimizer, target)
    # Live updates after snapshot do not mutate either saved optimizer.
    torch.testing.assert_close(best, frozen_best, rtol=0, atol=0)
    result = guard.observe(.1, guard.snapshot(actor, optimizer, estimator_optimizer=estimator_optimizer),
                           actor, optimizer, estimator_optimizer=estimator_optimizer)
    assert result['restored'] and not result['estimator_optimizer_legacy_reset']
    torch.testing.assert_close(actor.state_dict(), frozen_best['actor'], rtol=0, atol=0)
    expected_decoder = copy.deepcopy(frozen_best['optimizer'])
    expected_decoder['param_groups'][0]['lr'] *= .5
    assert_optimizer_state(optimizer, expected_decoder)
    assert_optimizer_state(estimator_optimizer, frozen_best['estimator_optimizer'])

    # The next real optimizer step is exactly the same as resuming the best
    # complete pair with only the decoder learning rate reduced.
    reference, reference_optimizer, reference_estimator_optimizer = estimated_actor_and_optimizers()
    reference.load_state_dict(frozen_best['actor'])
    reference_optimizer.load_state_dict(copy.deepcopy(expected_decoder))
    reference_estimator_optimizer.load_state_dict(copy.deepcopy(frozen_best['estimator_optimizer']))
    update_estimated_actor(actor, optimizer, estimator_optimizer, 1.5)
    update_estimated_actor(reference, reference_optimizer, reference_estimator_optimizer, 1.5)
    torch.testing.assert_close(actor.state_dict(), reference.state_dict(), rtol=0, atol=0)
    assert_optimizer_state(optimizer, reference_optimizer.state_dict())
    assert_optimizer_state(estimator_optimizer, reference_estimator_optimizer.state_dict())
    # load_state_dict can alias Adam moments: a post-restore update must not
    # corrupt the saved pair that will be used by a later restoration.
    torch.testing.assert_close(guard.best, frozen_best, rtol=0, atol=0)
    resumed = NativeRegressionGuard(.1, 1, 1e-3)
    resumed.load_state_dict(copy.deepcopy(guard.state_dict()))
    again = resumed.observe(.1, resumed.snapshot(actor, optimizer, estimator_optimizer=estimator_optimizer),
                            actor, optimizer, estimator_optimizer=estimator_optimizer)
    assert again['restorations'] == 2 and not again['estimator_optimizer_legacy_reset']
    torch.testing.assert_close(actor.state_dict(), frozen_best['actor'], rtol=0, atol=0)
    expected_decoder['param_groups'][0]['lr'] *= .5
    assert_optimizer_state(optimizer, expected_decoder)
    assert_optimizer_state(estimator_optimizer, frozen_best['estimator_optimizer'])


def test_legacy_best_resets_estimator_adam_only_on_real_restore_and_preserves_its_groups():
    actor, optimizer, estimator_optimizer = estimated_actor_and_optimizers()
    for target in (1., -1.):
        update_estimated_actor(actor, optimizer, estimator_optimizer, target)
    legacy = NativeRegressionGuard(.1, 2, 1e-3)
    best = legacy.snapshot(actor, optimizer)  # Historical snapshot has no estimator Adam.
    legacy.observe(1., best, actor, optimizer)
    update_estimated_actor(actor, optimizer, estimator_optimizer, 2.)
    estimator_optimizer.param_groups[0]['lr'] = 7e-4
    current = copy.deepcopy(estimator_optimizer.state_dict())
    guard = NativeRegressionGuard(.1, 2, 1e-3)
    guard.load_state_dict(copy.deepcopy(legacy.state_dict()))
    assert_optimizer_state(estimator_optimizer, current)
    for reward in (None, .95, .5):
        result = guard.observe(reward, guard.snapshot(actor, optimizer, estimator_optimizer=estimator_optimizer),
                               actor, optimizer, estimator_optimizer=estimator_optimizer)
        assert not result['restored'] and not result['estimator_optimizer_legacy_reset']
        assert_optimizer_state(estimator_optimizer, current)
    result = guard.observe(.4, guard.snapshot(actor, optimizer, estimator_optimizer=estimator_optimizer),
                           actor, optimizer, estimator_optimizer=estimator_optimizer)
    assert result['restored'] and result['estimator_optimizer_legacy_reset']
    torch.testing.assert_close(actor.state_dict(), best['actor'], rtol=0, atol=0)
    assert not estimator_optimizer.state
    assert estimator_optimizer.state_dict()['param_groups'] == current['param_groups']
    update_estimated_actor(actor, optimizer, estimator_optimizer, .2)
    assert all(int(value['step']) == 1 for value in estimator_optimizer.state.values())
