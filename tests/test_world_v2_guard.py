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
    assert not result['actor_optimizer_legacy_reset']
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
    assert not again['actor_optimizer_legacy_reset']
    torch.testing.assert_close(actor.state_dict(), frozen_best['actor'], rtol=0, atol=0)
    expected_decoder['param_groups'][0]['lr'] *= .5
    assert_optimizer_state(optimizer, expected_decoder)
    assert_optimizer_state(estimator_optimizer, frozen_best['estimator_optimizer'])


@pytest.mark.parametrize('previous_legacy_restore', [False, True])
def test_legacy_joint_best_resets_both_adams_only_on_real_restore_and_preserves_groups(previous_legacy_restore):
    actor, optimizer, estimator_optimizer = estimated_actor_and_optimizers()
    for target in (1., -1.):
        update_estimated_actor(actor, optimizer, estimator_optimizer, target)
    legacy = NativeRegressionGuard(.1, 2, 1e-3)
    best = legacy.snapshot(actor, optimizer)  # Historical snapshot has no estimator Adam.
    legacy.observe(1., best, actor, optimizer)
    if previous_legacy_restore:
        # Reproduce the old code path: load without deepcopy then train, so
        # best still contains old weights but its saved decoder moments drift.
        optimizer.load_state_dict(best['optimizer'])
    update_estimated_actor(actor, optimizer, estimator_optimizer, 2.)
    if previous_legacy_restore:
        saved = best['optimizer']['state'][0]['exp_avg']
        current_moment = optimizer.state_dict()['state'][0]['exp_avg']
        assert saved.untyped_storage().data_ptr() == current_moment.untyped_storage().data_ptr()
        assert all(int(state['step']) == 3 for state in best['optimizer']['state'].values())
    optimizer.param_groups[0]['betas'] = (.8, .99)
    estimator_optimizer.param_groups[0]['lr'] = 7e-4
    current_actor = copy.deepcopy(optimizer.state_dict())
    current = copy.deepcopy(estimator_optimizer.state_dict())
    guard = NativeRegressionGuard(.1, 2, 1e-3)
    guard.load_state_dict(copy.deepcopy(legacy.state_dict()))
    assert_optimizer_state(optimizer, current_actor)
    assert_optimizer_state(estimator_optimizer, current)
    for reward in (None, .95, .5):
        result = guard.observe(reward, guard.snapshot(actor, optimizer, estimator_optimizer=estimator_optimizer),
                               actor, optimizer, estimator_optimizer=estimator_optimizer)
        assert not result['restored'] and not result['estimator_optimizer_legacy_reset']
        assert not result['actor_optimizer_legacy_reset']
        assert_optimizer_state(optimizer, current_actor)
        assert_optimizer_state(estimator_optimizer, current)
    result = guard.observe(.4, guard.snapshot(actor, optimizer, estimator_optimizer=estimator_optimizer),
                           actor, optimizer, estimator_optimizer=estimator_optimizer)
    assert result['restored'] and result['estimator_optimizer_legacy_reset']
    assert result['actor_optimizer_legacy_reset']
    torch.testing.assert_close(actor.state_dict(), best['actor'], rtol=0, atol=0)
    assert not optimizer.state and not estimator_optimizer.state
    current_actor['param_groups'][0]['lr'] *= .5
    assert optimizer.state_dict()['param_groups'] == current_actor['param_groups']
    assert estimator_optimizer.state_dict()['param_groups'] == current['param_groups']
    update_estimated_actor(actor, optimizer, estimator_optimizer, .2)
    assert all(int(value['step']) == 1 for value in optimizer.state.values())
    assert all(int(value['step']) == 1 for value in estimator_optimizer.state.values())


def test_window_median_ignores_short_population_dips_but_catches_sustained_drops():
    actor, optimizer = actor_and_optimizer()
    guard = NativeRegressionGuard(.05, 2, 1e-3, window=5)
    observe = lambda reward: guard.observe(reward, guard.snapshot(actor, optimizer), actor, optimizer)
    for reward in (.093, .093, .093, .093):
        assert observe(reward)['judged_student_reward'] is None
    assert observe(.093)['best_student_reward'] == .093
    # A synchronized rebirth costs two collections (.06 and .03 in a05).
    statuses = [observe(reward) for reward in (.06, .03, .093, .093, .093)]
    assert not any(status['restored'] or status['bad_cycles'] for status in statuses)
    statuses = [observe(.06) for _ in range(4)]
    assert [status['bad_cycles'] for status in statuses] == [0, 0, 1, 0]
    assert statuses[2]['judged_student_reward'] == .06 and statuses[3]['restored']


def test_report_mode_judges_without_restoring_or_changing_the_rate():
    actor, optimizer = actor_and_optimizer()
    guard = NativeRegressionGuard(.1, 1, 1e-3, mode='report')
    guard.observe(.09, None, actor, optimizer)
    step(actor, optimizer, 1.)
    weight = actor.weight.detach().clone()
    status = guard.observe(.01, None, actor, optimizer)
    assert status['bad_cycles'] == 1 and not status['restored'] and status['mode'] == 'report'
    torch.testing.assert_close(actor.weight.detach(), weight)
    assert optimizer.param_groups[0]['lr'] == 1e-3 and guard.learning_rate() == 1e-3
    with pytest.raises(ValueError):
        NativeRegressionGuard(.1, 1, 1e-3, mode='skip')


def test_changed_window_or_mode_restarts_the_judgment_but_keeps_the_restoration_record():
    actor, optimizer = actor_and_optimizer()
    old = NativeRegressionGuard(.1, 1, 1e-3)
    old.observe(.09, old.snapshot(actor, optimizer), actor, optimizer)
    old.observe(.01, old.snapshot(actor, optimizer), actor, optimizer)
    assert old.restorations == 1
    state = old.state_dict()
    legacy = {key: state[key] for key in ('best_reward', 'best', 'bad_cycles', 'restorations')}
    same = NativeRegressionGuard(.1, 1, 1e-3)
    assert same.load_state_dict(legacy) and same.best_reward == .09 and same.learning_rate() == 5e-4
    report = NativeRegressionGuard(.1, 1, 1e-3, window=3, mode='report')
    assert not report.load_state_dict(legacy)
    assert report.best_reward is None and report.restorations == 1 and report.learning_rate() == 1e-3


def test_restore_restarts_the_window_and_the_rate_has_a_floor_and_recovers():
    actor, optimizer = actor_and_optimizer()
    guard = NativeRegressionGuard(.05, 1, 1e-3, window=3, min_rate_fraction=.25, recover_after=2)
    observe = lambda reward: guard.observe(reward, guard.snapshot(actor, optimizer), actor, optimizer)
    for reward in (.09, .09, .09):
        observe(reward)
    statuses = [observe(.01) for _ in range(2)]
    assert statuses[-1]['restored'] and len(guard.recent) == 0
    # Stale bad rewards are gone: the restored actor's first full window is judged afresh.
    assert not any(observe(.09)['restored'] for _ in range(3))
    for _ in range(3):                         # repeated real drops floor the rate at 1/4
        for reward in (.01, .01, .01):
            observe(reward)
    assert guard.halvings >= 2 and guard.learning_rate() == pytest.approx(2.5e-4)
    halvings = guard.halvings
    for reward in (.09,) * 7:
        observe(reward)
    assert guard.halvings < halvings


def test_guard_settings_are_validated_once_and_halvings_stop_at_the_floor():
    from supermimic_rl.world_model_v2.guard import validate_guard_settings

    with pytest.raises(ValueError, match='odd'):
        NativeRegressionGuard(.1, 1, 1e-3, window=4)
    validate_guard_settings(drop_fraction=0., patience=2, window=1, mode='restore', min_rate_fraction=0.,
                            recover_after=0)                      # a disabled guard is a valid setting
    actor, optimizer = actor_and_optimizer()
    guard = NativeRegressionGuard(.05, 1, 1e-3, min_rate_fraction=.25, recover_after=1)
    observe = lambda reward: guard.observe(reward, guard.snapshot(actor, optimizer), actor, optimizer)
    observe(.09)
    for _ in range(5):
        observe(.01)
    assert guard.restorations == 5 and guard.halvings == 2 and guard.learning_rate() == pytest.approx(2.5e-4)
    observe(.09)
    assert guard.halvings == 1
