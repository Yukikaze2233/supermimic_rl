"""Recovery diagnostics observe real native clocks and preserve their decisions."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.course.recovery_evaluation import RecoveryEvaluation


@pytest.fixture
def native_types():
    lifecycle = pytest.importorskip('transformer_rl.formal.lab_env.recovery_lifecycle')
    from transformer_rl.formal.tasks.protocol import CompletedEstimate
    return lifecycle.RecoveryLifecycle, lifecycle.RecoveryLifecycleConfig, CompletedEstimate


def setup(native_types, n=1, *, qualification=False):
    lifecycle, config, _ = native_types
    owner = lifecycle(n, config=config(policy_dt_s=.02, stable_s=.04, timeout_s=.08))
    owner.reset(torch.arange(n), torch.ones(n, dtype=torch.bool))
    env = SimpleNamespace(_terrain_course=SimpleNamespace(recovery=owner),
                          _episode_id=torch.ones(n, dtype=torch.long))
    options = ({'qualification_commands': torch.tensor([[0., 0., 0., .32]]).expand(n, -1).clone(),
                'qualification_clock': torch.zeros(n, dtype=torch.float64)} if qualification else {})
    evidence = RecoveryEvaluation(env, ('recovery_fixture',) * n, **options)
    return env, owner, evidence


def physics(n=1, **overrides):
    result = dict(height_m=torch.full((n,), .32), command_height_m=torch.full((n,), .32),
        tilt_rad=torch.zeros(n), angular_speed_rad_s=torch.zeros(n), velocity_z_m_s=torch.zeros(n),
        wheel_load_n=torch.full((n, 2), 20.), base_load_n=torch.zeros(n),
        policy_owned=torch.ones(n, dtype=torch.bool))
    result.update(overrides)
    return result


def estimate(native_types, now, **overrides):
    _, _, completed = native_types
    n = len(now)
    result = completed(torch.zeros(n), torch.ones(n, 2), torch.zeros(n), torch.zeros(n),
                       now - .01, torch.ones(n, dtype=torch.bool))
    return replace(result, **overrides)


def transition(evidence, start, *, valid=None, end=None):
    start = torch.as_tensor(start, dtype=torch.long)
    return evidence.observe_transition(SimpleNamespace(start=SimpleNamespace(episode_id=start),
        end=SimpleNamespace(episode_id=start if end is None else torch.tensor(end))),
        torch.ones(len(start), dtype=torch.bool) if valid is None else torch.tensor(valid))


def test_height_only_miss_keeps_native_handoff_and_distinct_physical_verification(native_types):
    env, owner, evidence = setup(native_types, 2)
    _, reference, _ = setup(native_types, 2)
    values = physics(2, height_m=torch.tensor([.23, .32]))
    rng = torch.random.get_rng_state().clone()
    evidence.install()
    try:
        for index in range(2):
            for _ in range(4):
                owner.observe_physics(**values)
                reference.observe_physics(**values)
            now = torch.full((2,), .02 * (index + 1), dtype=torch.float64)
            args = dict(policy_owned=torch.ones(2, dtype=torch.bool), failed=torch.zeros(2, dtype=torch.bool))
            actual = owner.finish_interval(now, estimate(native_types, now), **args)
            expected = reference.finish_interval(now, estimate(native_types, now), **args)
            torch.testing.assert_close(actual.handoff, expected.handoff)
            transition(evidence, [1, 1])
        for name, value in vars(reference).items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(getattr(owner, name), value)
        torch.testing.assert_close(torch.random.get_rng_state(), rng)
        report = evidence.summary()
        first, second = report['episodes']
        assert first['outcome'] == second['outcome'] == 'handoff'
        assert first['last_physical_conditions']['height'] is False
        assert first['only_failed_condition_ticks']['height'] == 8
        assert first['max_physical_stable_s'] == 0
        assert first['max_estimated_stable_s'] == pytest.approx(.04)
        assert not first['native_verified_physical_success']
        assert second['native_verified_physical_success']
        assert first['validated_boundary'] and second['validated_boundary']
        assert all(first['last_estimate_conditions'].values())
        group = report['by_scenario']['recovery_fixture']
        assert group['births'] == 2 and group['validated_verified_episodes'] == 1
        assert evidence.sample()['recovery_after_handoff'].all()
        json.dumps(report, allow_nan=False)
    finally:
        evidence.close()
    for name in ('observe_physics', 'finish_interval', 'reset'):
        assert name not in owner.__dict__


@pytest.mark.parametrize('reset_before_boundary', [False, True])
@pytest.mark.parametrize('prior_invalid', [False, True])
def test_physical_certificate_is_counted_before_handoff_and_survives_reset(
        native_types, reset_before_boundary, prior_invalid):
    env, owner, evidence = setup(native_types)
    evidence.install()
    try:
        for index in range(2):
            for _ in range(4):
                owner.observe_physics(**physics())
            owner.finish_interval(torch.tensor([.02 * (index + 1)], dtype=torch.float64), None,
                policy_owned=torch.tensor([True]), failed=torch.tensor([False]))
            if index == 0:
                assert transition(evidence, [1], valid=[not prior_invalid]) == []
        # The estimator has not handed control off, but native physics already
        # certified the recovery exactly once at finish_interval.
        assert owner.active.item() and owner._verified.item()
        assert owner.metrics()['counts']['verified_physical_successes'] == 1
        if reset_before_boundary:
            owner.reset(torch.tensor([0]), torch.tensor([True]))
            env._episode_id += 1
            assert not owner._verified.item()
        events = transition(evidence, [1])
        assert len(events) == 1
        assert events[0]['episode_id'] == 1 and events[0]['request_id'] is None
        assert events[0]['validated_event'] is not prior_invalid
        assert transition(evidence, env._episode_id) == []
        if not reset_before_boundary:
            for index in (3, 4):
                for _ in range(4):
                    owner.observe_physics(**physics())
                now = torch.tensor([.02 * index], dtype=torch.float64)
                owner.finish_interval(now, estimate(native_types, now),
                    policy_owned=torch.tensor([True]), failed=torch.tensor([False]))
                assert transition(evidence, [1]) == []
            assert not owner.active.item()
            assert evidence.summary()['episodes'][0]['outcome'] == 'handoff'
            assert owner.metrics()['counts']['verified_physical_successes'] == 1
    finally:
        evidence.close()


def test_timeout_reports_physical_and_estimator_obstructions_and_tick_maximum(native_types):
    _, owner, evidence = setup(native_types)
    evidence.install()
    try:
        for index in range(4):
            # A transient at the last substep must not hide earlier continuous stability.
            for tick in range(4):
                owner.observe_physics(**physics(angular_speed_rad_s=torch.tensor([.8 if tick == 3 else 0.])))
            now = torch.tensor([.02 * (index + 1)], dtype=torch.float64)
            owner.finish_interval(now, estimate(native_types, now, angular_speed_rad_s=torch.tensor([.8])),
                policy_owned=torch.tensor([True]), failed=torch.tensor([False]))
            transition(evidence, [1])
        record = evidence.summary()['episodes'][0]
        assert record['outcome'] == 'timeout' and record['validated_boundary']
        assert record['only_failed_condition_ticks']['angular_speed'] == 4
        assert record['max_physical_stable_s'] == pytest.approx(.015)
        assert record['physical_stable_at_end_s'] == 0
        assert record['max_estimated_stable_s'] == 0
        assert record['failed_estimate_condition_intervals']['angular_speed'] == 4
        assert record['last_estimate']['angular_speed_rad_s'] == pytest.approx(.8)
        assert record['last_physical_values']['angular_speed_rad_s'] == pytest.approx(.8)
    finally:
        evidence.close()


def test_auto_reset_keeps_old_boundary_identity_and_invalid_raw_outcome(native_types):
    env, owner, evidence = setup(native_types)
    evidence.install()
    try:
        for index in range(2):
            for _ in range(4):
                owner.observe_physics(**physics())
            now = torch.tensor([.02 * (index + 1)], dtype=torch.float64)
            owner.finish_interval(now, estimate(native_types, now),
                policy_owned=torch.tensor([True]), failed=torch.tensor([False]))
            if index == 0:
                transition(evidence, [1], valid=[False])
        # _reset_idx calls terrain.reset_commands/recovery.reset BEFORE publishing the new ID.
        owner.reset(torch.tensor([0]), torch.tensor([True]))
        assert evidence.pending_birth == {0}
        assert (0, 2) not in evidence.records
        env._episode_id += 1
        transition(evidence, [1])
        records = evidence.summary()['episodes']
        old, new = records
        assert old['outcome'] == 'handoff' and old['boundary_valid']
        assert old['had_invalid'] and not old['validated_boundary']
        assert old['observed_intervals'] == 2 and old['invalid_intervals'] == 1
        assert new['episode_id'] == 2 and new['observed_intervals'] == 0
        assert new['outcome'] == 'active_at_evaluation_end'
        assert new['birth_source'] == 'native_reset'
        assert not evidence.sample()['recovery_after_handoff'].any()
        assert evidence.summary()['by_scenario']['recovery_fixture']['validated_outcomes'] == {}
    finally:
        evidence.close()


def test_reset_without_finish_retains_interrupted_birth_and_rejects_cross_episode_boundary(native_types):
    env, owner, evidence = setup(native_types)
    evidence.install()
    try:
        owner.observe_physics(**physics())
        owner.reset(torch.tensor([0]), torch.tensor([False]))
        env._episode_id += 1
        transition(evidence, [1], end=[2])
        record = evidence.summary()['episodes'][0]
        assert record['outcome'] == 'reset_before_completion'
        assert record['boundary_valid'] is False and record['had_invalid']
        assert record['physical_ticks'] == 1
    finally:
        evidence.close()


@pytest.mark.parametrize('first_read', ['sample', 'observe', 'finish', 'summary'])
def test_deferred_birth_binds_on_first_post_reset_read(native_types, first_read):
    env, owner, evidence = setup(native_types)
    evidence.install()
    try:
        owner.reset(torch.tensor([0]), torch.tensor([True]))
        assert evidence.pending_birth == {0}
        assert len(evidence.records) == 1
        env._episode_id += 1
        if first_read == 'observe':
            owner.observe_physics(**physics())
        elif first_read == 'finish':
            owner.finish_interval(torch.tensor([.02], dtype=torch.float64), None,
                policy_owned=torch.tensor([True]), failed=torch.tensor([False]))
        else:
            getattr(evidence, first_read)()
        assert not evidence.pending_birth
        record = evidence.summary()['episodes'][1]
        assert record['episode_id'] == 2 and record['birth_source'] == 'native_reset'
        assert record['observed_intervals'] == 0
        assert record['physical_ticks'] == int(first_read == 'observe')
    finally:
        evidence.close()


def test_nonfinite_and_nonowned_ticks_are_explicit_and_never_predicate_evidence(native_types):
    _, owner, evidence = setup(native_types)
    previous = owner.observe_physics
    owner.observe_physics = previous  # A pre-existing instance override must survive close.
    evidence.install()
    try:
        owner.observe_physics(**physics(policy_owned=torch.tensor([False])))
        owner.observe_physics(**physics(height_m=torch.tensor([float('nan')])))
        owner.finish_interval(torch.tensor([.02], dtype=torch.float64), None,
            policy_owned=torch.tensor([True]), failed=torch.tensor([True]))
        transition(evidence, [1])
        record = evidence.summary()['episodes'][0]
        assert record['outcome'] == 'failed' and record['had_invalid']
        assert record['non_policy_owned_physical_ticks'] == record['nonfinite_physical_ticks'] == 1
        assert record['valid_owned_physical_ticks'] == 0
        assert sum(record['failed_condition_ticks'].values()) == 0
        assert record['last_physical_values']['height_error_m'] is None
        assert not record['validated_boundary']
        json.dumps(evidence.summary(), allow_nan=False)
    finally:
        evidence.close()
    assert owner.observe_physics is previous
    evidence.close()  # Exception cleanup is idempotent.


def test_each_physical_predicate_uses_the_bound_native_configuration(native_types):
    _, owner, evidence = setup(native_types, 6)
    owner.config = replace(owner.config, height_tolerance_m=.02, tilt_rad=.1,
        angular_speed_rad_s=.25, vertical_speed_m_s=.08, wheel_load_n=8., base_load_n=1.)
    values = physics(6)
    values['height_m'][0] = .35
    values['tilt_rad'][1] = .11
    values['angular_speed_rad_s'][2] = .26
    values['velocity_z_m_s'][3] = -.09
    values['wheel_load_n'][4, 0] = 7.
    values['base_load_n'][5] = 1.1
    evidence.install()
    try:
        owner.observe_physics(**values)
        owner.finish_interval(torch.full((6,), .02, dtype=torch.float64), None,
            policy_owned=torch.ones(6, dtype=torch.bool), failed=torch.ones(6, dtype=torch.bool))
        transition(evidence, [1] * 6)
        report = evidence.summary()
        assert report['configuration']['wheel_load_n'] == 8.
        for index, record in enumerate(report['episodes']):
            failed = record['only_failed_condition_ticks']
            assert failed[evidence.CONDITIONS[index]] == 1 and sum(failed.values()) == 1
            assert record['last_physical_conditions'][evidence.CONDITIONS[index]] is False
            assert record['max_physical_stable_s'] == 0
    finally:
        evidence.close()


def test_qualification_continues_after_native_handoff_and_keeps_request_clock_after_success(native_types):
    _, owner, evidence = setup(native_types, qualification=True)
    _, baseline, _ = setup(native_types)
    evidence.install()
    try:
        for index in range(5):
            values = physics(height_m=torch.tensor([.23 if index == 0 else .32]))
            for _ in range(4):
                owner.observe_physics(**values)
                baseline.observe_physics(**values)
            now = torch.tensor([.02 * (index + 1)], dtype=torch.float64)
            args = dict(policy_owned=torch.tensor([True]), failed=torch.tensor([False]))
            owner.finish_interval(now, estimate(native_types, now), **args)
            baseline.finish_interval(now, estimate(native_types, now), **args)
            if index == 1:
                assert not owner.active.any() and not owner._verified.any()
                assert evidence.summary()['qualification']['episodes'][0]['outcome'] == 'incomplete_at_evaluation_end'
            if index == 2:
                q = evidence.summary()['qualification']['episodes'][0]
                assert q['outcome'] == 'success' and not q['validated_boundary']
            transition(evidence, [1])
        q = evidence.summary()['qualification']
        record = q['episodes'][0]
        assert record['outcome'] == 'success' and record['validated_boundary']
        assert record['first_qualifying_tick_at_s'] == pytest.approx(.06)
        assert record['max_physical_stable_s'] == pytest.approx(.04)
        assert record['budget_elapsed_s'] == pytest.approx(.06)
        assert evidence.qualification['clock'].item() == pytest.approx(.08)
        assert q['by_scenario']['recovery_fixture']['success_rate'] == 1
        assert evidence.summary()['episodes'][0]['outcome'] == 'handoff'
        assert not evidence.summary()['episodes'][0]['native_verified_physical_success']
        for name, value in vars(baseline).items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(getattr(owner, name), value)
    finally:
        evidence.close()


@pytest.mark.parametrize('late', ['failure', 'nonfinite', 'invalid_transition'])
def test_qualification_success_waits_for_complete_valid_interval(native_types, late):
    env, owner, evidence = setup(native_types, qualification=True)
    owner.config = replace(owner.config, stable_s=.025)
    env._terminated = torch.tensor([False])
    evidence.install()
    try:
        for index in range(2):
            for tick in range(4):
                values = physics()
                if index == 1 and tick == 3:
                    env._terminated.fill_(late == 'failure')
                    if late == 'nonfinite':
                        values['height_m'].fill_(float('nan'))
                owner.observe_physics(**values)
            now = torch.tensor([.02 * (index + 1)], dtype=torch.float64)
            owner.finish_interval(now, None, policy_owned=torch.tensor([True]), failed=env._terminated)
            transition(evidence, [1], valid=[not (index == 1 and late == 'invalid_transition')])
        record = evidence.summary()['qualification']['episodes'][0]
        assert record['first_qualifying_tick_at_s'] == pytest.approx(.025)
        if late == 'failure':
            assert record['outcome'] == 'failure' and record['validated_boundary']
        else:
            assert record['outcome'] == 'success' and not record['validated_boundary']
            assert record['had_invalid']
        assert evidence.summary()['qualification']['by_scenario']['recovery_fixture']['validated_outcomes'].get('success', 0) == 0
    finally:
        evidence.close()


def test_later_tracking_failure_does_not_rewrite_completed_qualification(native_types):
    _, owner, evidence = setup(native_types, qualification=True)
    evidence.install()
    try:
        for index in range(3):
            for _ in range(4):
                owner.observe_physics(**physics())
            now = torch.tensor([.02 * (index + 1)], dtype=torch.float64)
            owner.finish_interval(now, estimate(native_types, now),
                policy_owned=torch.tensor([True]), failed=torch.tensor([index == 2]))
            transition(evidence, [1])
            if index == 1:
                completed = evidence.summary()['qualification']['episodes'][0]
        assert evidence.summary()['qualification']['episodes'][0] == completed
        assert completed['outcome'] == 'success' and completed['validated_boundary']
    finally:
        evidence.close()


def test_qualification_reset_binds_after_native_episode_publication_without_inheriting_stability(native_types):
    env, owner, evidence = setup(native_types, qualification=True)
    evidence.install()
    try:
        for _ in range(4):
            owner.observe_physics(**physics())
        owner.reset(torch.tensor([0]), torch.tensor([True]))
        assert evidence.qualification['clock'].item() == 0
        env._episode_id += 1
        transition(evidence, [1])
        for _ in range(4):
            owner.observe_physics(**physics())
        owner.finish_interval(torch.tensor([.04], dtype=torch.float64), None,
            policy_owned=torch.tensor([True]), failed=torch.tensor([False]))
        transition(evidence, [2])
        first, second = evidence.summary()['qualification']['episodes']
        assert first['outcome'] == 'incomplete_reset' and first['initial_birth']
        assert first['episode_id'] == 1 and first['budget_elapsed_s'] == pytest.approx(.02)
        assert second['episode_id'] == 2 and not second['initial_birth']
        assert second['outcome'] == 'incomplete_at_evaluation_end'
        assert second['max_physical_stable_s'] == pytest.approx(.02)
        assert evidence.summary()['qualification']['by_scenario']['recovery_fixture']['completed_valid'] == 0
    finally:
        evidence.close()


def test_qualification_budget_counts_owned_ticks_and_uses_original_height(native_types):
    _, owner, evidence = setup(native_types, 2, qualification=True)
    evidence.install()
    try:
        for index in range(6):
            for _ in range(4):
                owner.observe_physics(**physics(2, height_m=torch.tensor([.2, .32]),
                    command_height_m=torch.full((2,), .4),
                    policy_owned=torch.full((2,), index >= 2, dtype=torch.bool)))
            now = torch.full((2,), .02 * (index + 1), dtype=torch.float64)
            owner.finish_interval(now, None, policy_owned=torch.full((2,), index >= 2, dtype=torch.bool),
                failed=torch.zeros(2, dtype=torch.bool))
            transition(evidence, [1, 1])
            if index == 1:
                assert not evidence.qualification['clock'].any()
        failed, success = evidence.summary()['qualification']['episodes']
        assert failed['outcome'] == 'timeout' and failed['validated_boundary']
        assert failed['budget_elapsed_s'] == pytest.approx(.08)
        assert success['outcome'] == 'success' and success['validated_boundary']
        assert success['target_height_m'] == pytest.approx(.32)
        assert not owner._verified.any()  # Native still used its actual .4 m target.
    finally:
        evidence.close()
