"""Prescribed physics through native jump decisions, with passive request diagnostics."""
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.course.jump_evaluation import JumpEvaluation
from supermimic_rl.world_model_v2.jump_task import JumpTaskState
from transformer_rl.formal.lab_env.course_tasks import RequestCommand
from transformer_rl.formal.tasks.protocol import CompletedEstimate

from support.world_v2.jump_task import native_jumps


def environment(n=1):
    env = native_jumps(n)
    env._episode_id = torch.zeros(n, dtype=torch.long)
    env._interval_policy_owned = torch.ones(n, dtype=torch.bool)
    return env


def advance(env, *, request=-1, contact=.9, now=None, cancel=-1):
    n = len(env.time)
    env.time = env.time + .02 if now is None else torch.full_like(env.time, now)
    estimate = CompletedEstimate(torch.zeros(n), torch.full((n, 2), contact), torch.zeros(n),
                                 torch.zeros(n), env.time - .02, torch.ones(n, dtype=torch.bool))
    packet = RequestCommand(torch.full((n,), request, dtype=torch.long), torch.full((n,), .06),
                            torch.full((n,), cancel, dtype=torch.long))
    return env._terrain_course.tasks.step(env.time, estimate, torch.zeros(n), command=packet)


def tick(env, *, height=.32, vz=0., load=20., angular=0., tilt=0., dt=.005):
    n = len(env.time)
    values = [torch.as_tensor(value).float().expand(n) for value in (height, vz, angular, tilt)]
    env._terrain_course.tasks.observe_physics(torch.full((n, 2), load), values[1], values[3],
        values[0], angular_speed_rad_s=values[2], dt=dt)


def boundary(evidence, *, valid=True, episode=0):
    n = len(evidence.row_names)
    endpoints = SimpleNamespace(episode_id=torch.full((n,), episode, dtype=torch.long))
    return evidence.observe_transition(SimpleNamespace(start=endpoints, end=endpoints),
                                       torch.full((n,), valid, dtype=torch.bool))


def test_physical_substeps_distinguish_micro_unload_from_peak_and_noise_stability():
    env = environment(3)
    evidence = JumpEvaluation(env, ['micro', 'quiet', 'noisy'])
    evidence.install()
    advance(env, request=7)
    # All requests unload; only the latter two achieve requested peak/ascent.
    for _ in range(2):
        tick(env, height=torch.tensor([.321, .38, .38]), vz=torch.tensor([0., .4, .4]), load=0.)
    for _ in range(401):
        tick(env, angular=torch.tensor([0., 0., .6]))
    boundary(evidence)
    report = evidence.summary()
    micro, quiet, noisy = report['requests']
    assert all(row['flight_seen'] for row in (micro, quiet, noisy))
    assert all(row['max_continuous_both_wheels_unloaded_s'] == pytest.approx(.01)
               for row in (micro, quiet, noisy))
    assert all(row['max_minimum_wheel_clearance_m'] is None for row in (micro, quiet, noisy))
    assert not micro['ascent_seen'] and not micro['peak_in_target_window']
    assert micro['peak_rise_m'] == pytest.approx(.001, abs=1e-7)
    assert micro['physical_end_reason'] == 'active_at_evaluation_end'
    assert quiet['physical_end_reason'] == 'verified_success'
    assert quiet['validated_boundary'] and quiet['native_verified_physical_success']
    assert quiet['max_landed_s'] >= 2.
    assert noisy['peak_in_target_window'] and noisy['ascent_seen']
    assert noisy['max_landed_s'] == 0.
    assert noisy['landing_failed_condition_s']['angular_speed'] == pytest.approx(2.005)
    assert noisy['only_failed_condition_s']['angular_speed'] == pytest.approx(2.005)
    assert report['criteria']['required_stable_landing_s'] == 2.
    assert env._terrain_course.tasks.physical_outcome_counts.tolist() == [1, 0, 0]
    assert all(row['assisted'] is False for row in report['requests'])
    assert evidence.summary() == report  # Reports neither close nor update a live request.
    evidence.close()


def test_protocol_estimated_completion_does_not_claim_physical_success_or_close_evidence():
    env = environment()
    tasks = env._terrain_course.tasks
    evidence = JumpEvaluation(env, ['micro'])
    evidence.install()
    advance(env, request=13)
    for _ in range(2):
        tick(env, height=.321, load=0.)
        advance(env, contact=.1)
    for _ in range(103):
        tick(env)
        advance(env)
        boundary(evidence)
    record = evidence.summary()['requests'][0]
    assert record['protocol_end_reason'] == 'estimated_complete'
    assert record['physical_end_reason'] == 'active_at_evaluation_end'
    assert not record['native_verified_physical_success']
    assert tasks.physical_id.item() == 13 and tasks.protocol.active_id.item() == -1
    # Native physical evidence lives until its own six-second deadline.
    for _ in range(200):
        tick(env)
        advance(env)
        boundary(evidence)
    record = evidence.summary()['requests'][0]
    assert record['protocol_end_reason'] == 'estimated_complete'
    assert record['physical_end_reason'] == 'physical_timeout'
    assert record['validated_boundary']
    evidence.close()


def test_invalid_interval_remains_visible_after_success_and_assist_reset_by_existing_wrapper():
    env = environment()
    tasks = env._terrain_course.tasks
    # Model an already installed assist wrapper: it clears attribution as soon
    # as native success clears physical_id. The observer must read it first.
    assist = SimpleNamespace(request=torch.tensor([-1]), impulse=torch.tensor([0.]))
    env.supermimic_jump_assist = assist
    original = tasks.observe_physics

    def assist_observer(*args, **kwargs):
        result = original(*args, **kwargs)
        if tasks.physical_id.item() < 0:
            assist.request.fill_(-1)
            assist.impulse.zero_()
        return result

    tasks.observe_physics = assist_observer
    evidence = JumpEvaluation(env, ['jump'])
    evidence.install()
    advance(env, request=2)
    assist.request.fill_(2)
    assist.impulse.fill_(1.25)
    for _ in range(2):
        tick(env, height=.38, vz=.4, load=0.)
    boundary(evidence, valid=False)
    for _ in range(401):
        tick(env)
    boundary(evidence)
    row = evidence.summary()['requests'][0]
    assert row['native_verified_physical_success']
    assert row['had_invalid'] and not row['validated_boundary']
    assert row['actual_assist_impulse_n_s'] == pytest.approx(1.25) and row['assisted']
    assert assist.impulse.item() == 0
    evidence.close()
    assert tasks.observe_physics is assist_observer
    assert 'step' not in tasks.__dict__ and 'step' not in tasks.protocol.__dict__


@pytest.mark.parametrize('assist', [SimpleNamespace(), SimpleNamespace(
    request=torch.tensor([99]), impulse=torch.tensor([7.]))])
def test_existing_unreadable_or_mismatched_assist_metadata_is_unknown(assist):
    env = environment()
    env.supermimic_jump_assist = assist
    evidence = JumpEvaluation(env, ['jump'])
    evidence.install()
    advance(env, request=2)
    tick(env)
    row = evidence.summary()['requests'][0]
    assert row['actual_assist_impulse_n_s'] is None and row['assisted'] is None
    evidence.close()


def test_native_state_and_rng_are_unchanged_and_existing_takeoff_wrapper_is_restored():
    from supermimic_rl.course.jump_takeoff import install_jump_takeoff

    wrapped, plain = environment(), environment()
    for env in (wrapped, plain):
        install_jump_takeoff(env, {'min_episode_s': .1, 'cooldown_s': .1})
    prior_step = wrapped._terrain_course.tasks.step
    evidence = JumpEvaluation(wrapped, ['jump'])
    evidence.install()
    for env in (wrapped, plain):
        advance(env, request=2)
        for _ in range(2):
            tick(env, height=.38, vz=.4, load=0.)
        tick(env, angular=.6)
        advance(env, cancel=2)
    boundary(evidence)
    a, b = JumpTaskState.capture(wrapped).record(), JumpTaskState.capture(plain).record()
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    assert torch.equal(wrapped._terrain_course.tasks.generator.get_state(),
                       plain._terrain_course.tasks.generator.get_state())
    assert evidence.summary()['requests'][0]['protocol_end_reason'] == 'cancelled'
    evidence.close()
    assert wrapped._terrain_course.tasks.step is prior_step


def test_reset_preserves_pre_reset_summary_and_next_episode_does_not_inherit_evidence():
    from supermimic_rl.course.curriculum import SuperMimicCurriculum
    from transformer_rl.formal.lab_env.formal_curriculum import LabTaskDrawSampler

    env = environment()
    tasks = env._terrain_course.tasks
    evidence = JumpEvaluation(env, ['jump'])
    evidence.install()
    advance(env, request=4)
    for _ in range(2):
        tick(env, height=.38, vz=.4, load=0.)
    ids = torch.arange(1)
    draw = LabTaskDrawSampler(SuperMimicCurriculum(1))(ids)
    tasks.reset(ids, draw, env.time, terrain_kind=draw.terrain_kind)
    env._episode_id += 1
    boundary(evidence, episode=0)
    old = evidence.summary()['requests'][0]
    assert old['physical_end_reason'] == 'physical_reset'
    assert old['protocol_end_reason'] == 'reset'
    assert old['peak_rise_m'] == pytest.approx(.06)
    assert old['validated_boundary'] and old['episode_id'] == 0
    tasks._last_supported.fill_(True)
    tasks._last_world_height.fill_(.32)
    # Native reset preserves monotonically increasing IDs; a new episode uses
    # its actual next ID and cannot inherit the previous request's peak.
    advance(env, request=5, now=float(env.time[0]))
    report = evidence.summary()
    assert len(report['requests']) == 2
    new = report['requests'][1]
    assert new['episode_id'] == 1 and new['request_id'] == 5
    assert new['peak_rise_m'] == 0 and not new['flight_seen']
    assert new['max_continuous_both_wheels_unloaded_s'] == 0
    assert new['total_both_wheels_unloaded_s'] == 0
    assert new['wheel_clearance_observed_ticks'] == 0
    assert new['max_minimum_wheel_clearance_while_unloaded_m'] is None
    assert new['physical_end_reason'] == 'active_at_evaluation_end'
    evidence.close()


def test_final_substep_success_requires_observed_valid_boundary():
    env = environment()
    evidence = JumpEvaluation(env, ['jump'])
    evidence.install()
    advance(env, request=0)
    for _ in range(2):
        tick(env, height=.38, vz=.4, load=0.)
    boundary(evidence)
    for _ in range(401):
        tick(env)
    row = evidence.summary()['requests'][0]
    assert row['native_verified_physical_success'] and not row['validated_boundary']
    assert evidence.summary()['physical_success_events'][0]['validated_event'] is None
    events = boundary(evidence)
    assert len(events) == 1 and events[0]['validated_event']
    assert evidence.summary()['requests'][0]['validated_boundary']
    evidence.close()


@pytest.mark.parametrize('boundary_valid', [True, False])
@pytest.mark.parametrize('next_state', ['cleared', 'replacement', 'reset'])
def test_native_success_event_survives_clearing_replacement_and_reset(next_state, boundary_valid):
    from supermimic_rl.course.curriculum import SuperMimicCurriculum
    from transformer_rl.formal.lab_env.formal_curriculum import LabTaskDrawSampler

    env = environment()
    tasks = env._terrain_course.tasks
    evidence = JumpEvaluation(env, ['jump'])
    evidence.install()
    try:
        advance(env, request=4)
        for _ in range(2):
            tick(env, height=.38, vz=.4, load=0.)
        for _ in range(401):
            tick(env)
        assert tasks.physical_outcome_counts[0] == 1
        if next_state == 'reset':
            ids = torch.arange(1)
            draw = LabTaskDrawSampler(SuperMimicCurriculum(1))(ids)
            tasks.reset(ids, draw, env.time, terrain_kind=draw.terrain_kind)
            env._episode_id += 1
        elif next_state == 'replacement':
            advance(env, cancel=4)
            advance(env, request=5)
            assert tasks.physical_id.item() == 5
        else:
            advance(env)
        assert tasks.success_id.item() == -1
        # Even when native auto-reset already published a newborn ID, the
        # completed transition still contains the old pre-reset endpoint.
        events = boundary(evidence, valid=boundary_valid, episode=0)
        assert len(events) == 1
        event = events[0]
        assert (event['episode_id'], event['request_id']) == (0, 4)
        assert event['interval_matches_episode']
        assert event['validated_event'] is boundary_valid
        assert boundary(evidence, episode=int(env._episode_id[0])) == []
        records = evidence.summary()['requests']
        assert records[0]['native_verified_physical_success']
        if next_state == 'replacement':
            assert not records[1]['native_verified_physical_success']
    finally:
        evidence.close()


def test_unloading_excludes_birth_and_breaks_on_unowned_invalid_and_supported_ticks():
    env = environment()
    course = env._terrain_course
    course.wheel_ground = torch.tensor([[.10, .20]])
    course._support_sample = SimpleNamespace(
        wheel_bottom_world_z_m=torch.tensor([[.18, .202]]), valid=torch.tensor([True]))
    evidence = JumpEvaluation(env, ['jump'])
    evidence.install()
    for _ in range(20):
        tick(env, load=0.)
    # Establish supported native acceptance after the unrequested spawn drop.
    tick(env)
    advance(env, request=0)
    for _ in range(3):
        tick(env, load=0.)
    env._interval_policy_owned.fill_(False)
    tick(env, load=0.)
    env._interval_policy_owned.fill_(True)
    for _ in range(2):
        tick(env, load=0.)
    tick(env)
    tick(env, load=0.)
    tick(env, load=float('nan'))
    for _ in range(2):
        tick(env, load=0.)
    # Invalid geometry is unavailable, not a fabricated clearance maximum.
    course._support_sample.valid.fill_(False)
    course._support_sample.wheel_bottom_world_z_m.fill_(100.)
    tick(env, load=0.)
    row = evidence.summary()['requests'][0]
    assert row['max_continuous_both_wheels_unloaded_s'] == pytest.approx(.015)
    assert row['total_both_wheels_unloaded_s'] == pytest.approx(.045)
    assert row['max_minimum_wheel_clearance_m'] == pytest.approx(.002, abs=1e-7)
    assert row['max_minimum_wheel_clearance_while_unloaded_m'] == pytest.approx(.002, abs=1e-7)
    assert row['wheel_clearance_observed_ticks'] == 9
    assert row['nonfinite_physical_ticks'] == 1
    assert row['non_policy_owned_physical_ticks'] == 1
    assert not row['native_verified_physical_success']
    evidence.close()
