"""Joint exposure is physical time, not command marginals or birth counts."""
from copy import deepcopy
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('transformer_rl.formal.tasks.reward')
from supermimic_rl.course.reward import install_shared_reward
from supermimic_rl.ppo.environment import CourseEnvironment
from supermimic_rl.ppo.trainer import EpisodeStatistics
from support.ppo import NativeCommandBoundary
from support.reward_kernel import begin, finish, sample, setup


PREFIX = 'joint_speed_turn/'
QUADRANTS = tuple(f'forward_{forward}_yaw_{yaw}' for forward in ('positive', 'negative')
                  for yaw in ('positive', 'negative'))


def make_composer(case='ordinary', bounds=(-1., 1.), policy_dt_s=.02):
    composer, reference = setup(case)
    composer.tuning = replace(composer.tuning, policy_dt_s=policy_dt_s)
    composer.substeps = round(policy_dt_s / composer.physics_dt_s)
    install_shared_reward(composer, joint_speed_turn_initial_forward_m_s=bounds)
    return composer, reference


def command_reference(reference, forward=(2.7, -2.7), yaw=(1., -1.)):
    command = reference.admitted_body_command.clone()
    command[:, 0] = torch.tensor(forward)
    command[:, 2] = torch.tensor(yaw)
    return replace(reference, admitted_body_command=command)


def run_interval(composer, reference, *, owned=None, terminal=False):
    begin(composer, reference, torch.zeros(2, 6))
    for tick in range(1, composer.substeps + 1):
        # Different errors and gate values across ticks expose endpoint-only
        # counting and mean-of-RMS errors. Physical clock remains native.
        forward_error = (2., 0., .5, 1.)[tick - 1]
        height_error = (0., .075, .1, 0.)[tick - 1]
        physics = sample(composer, tick)
        velocity = torch.zeros(2, 3)
        velocity[:, 0] = reference.admitted_body_command[:, 0] + forward_error
        omega = torch.zeros_like(velocity)
        omega[:, 2] = reference.admitted_body_command[:, 2] + .25
        physics = replace(physics,
            policy_owned=(torch.ones(2, dtype=torch.bool) if owned is None else owned[tick - 1]),
            velocity_origin_body_m_s=velocity,
            gravity_body=torch.tensor([[0., 0., -1.]]).expand(2, -1),
            angular_velocity_body_rad_s=omega,
            height_above_ground_m=reference.admitted_body_command[:, 3] + height_error)
        composer.observe(physics, reference)
    return finish(composer, terminal)


def ppo_step(result):
    # NativeCommandBoundary mutates targets and resets a terminal row after
    # physics. The telemetry must keep the completed interval's attribution.
    native = NativeCommandBoundary()
    native._formal_result = result
    native._formal_reward = SimpleNamespace(composer=SimpleNamespace(
        supermimic_joint_speed_turn_exposure=None))
    environment = CourseEnvironment(SimpleNamespace(env=native, _health=SimpleNamespace(
        check=lambda: None, sample_policy=lambda: None)))
    return environment.step(torch.zeros(2, 6))


def test_quadrants_use_initial_domain_and_pre_reset_admitted_commands():
    composer, reference = make_composer()
    expected_keys = {PREFIX + quadrant + '/' + field for quadrant in QUADRANTS
                     for field in ('forward_m_s', 'yaw_rad_s')}
    for forward, yaw, active in (
        ((2.7, -2.7), (1., -1.), (QUADRANTS[0], QUADRANTS[3])),
        ((2.7, -2.7), (-1., 1.), (QUADRANTS[1], QUADRANTS[2])),
        ((1., -1.), (1., -1.), ()),  # Initial-domain boundary is not high speed.
        ((2.7, -2.7), (0., 0.), ()),
    ):
        result = run_interval(composer, command_reference(reference, forward, yaw))
        assert expected_keys <= result.scoped_error_counts.keys()
        step = ppo_step(result)
        for quadrant in QUADRANTS:
            key = PREFIX + quadrant + '/forward_m_s'
            assert float(result.scoped_error_counts[key].sum()) == (1. if quadrant in active else 0.)
            torch.testing.assert_close(step.diagnostics['tracking_scope/count/' + key],
                                       result.scoped_error_counts[key], rtol=0, atol=0)
    # Threshold follows an asymmetric course domain rather than hardcoded 1 m/s.
    alternate, reference = make_composer(bounds=(-.7, 1.4))
    result = run_interval(alternate, command_reference(reference, (1.2, -.8)))
    assert result.scoped_error_counts[PREFIX + QUADRANTS[0] + '/forward_m_s'].sum() == 0
    assert result.scoped_error_counts[PREFIX + QUADRANTS[3] + '/forward_m_s'].sum() == 1


@pytest.mark.parametrize('case,expected', [('ordinary', [1., 1.]), ('terrain', [1., 1.]),
                                         ('gimbal', [0., 1.]), ('jump_flight', [0., 0.]),
                                         ('recovery', [0., 0.])])
def test_scope_tracks_ordinary_policy_time_and_excludes_placeholder_gimbal(case, expected):
    composer, reference = make_composer(case)
    result = run_interval(composer, command_reference(reference))
    torch.testing.assert_close(result.gate_fractions[PREFIX + 'ordinary_policy_owned'],
                               torch.tensor(expected, dtype=torch.float64))
    assert float(sum(result.scoped_error_counts[PREFIX + quadrant + '/forward_m_s'].sum()
                     for quadrant in QUADRANTS)) == sum(expected)


def test_four_ticks_seconds_conditional_fractions_valid_and_terminal_ownership():
    composer, reference = make_composer()
    # Row 1 is policy-owned only during the last two physical ticks.
    owned = torch.tensor([[True, False], [True, False], [True, True], [True, True]])
    result = run_interval(composer, command_reference(reference), owned=owned)
    step = ppo_step(result)
    stats = EpisodeStatistics(2, 'cpu', joint_speed_turn_exposure=composer.supermimic_joint_speed_turn_exposure)
    stats.record(step)
    summary = stats.summary()['joint_speed_turn_exposure']
    assert summary['valid_interval_seconds'] == .04
    assert summary['ordinary_policy_owned_seconds'] == pytest.approx(.03)
    assert summary['joint_active_seconds'] == pytest.approx(.03)
    positive = summary['by_command_sign'][QUADRANTS[0]]
    assert positive['active_seconds'] == .02
    assert positive['fraction_of_ordinary_policy_time'] == pytest.approx(2. / 3.)
    assert positive['forward_mean_error_m_s'] == pytest.approx(.875)
    assert positive['forward_rms_error_m_s'] == pytest.approx((5.25 / 4.) ** .5)
    assert positive['yaw_rms_error_rad_s'] == pytest.approx(.25)
    assert positive['far_forward_error_fraction'] == .5
    assert positive['velocity_height_gate_mean'] == pytest.approx(.625)
    negative = summary['by_command_sign'][QUADRANTS[3]]
    assert negative['active_seconds'] == .01
    assert negative['forward_mean_error_m_s'] == .75
    assert negative['velocity_height_gate_mean'] == pytest.approx(.5)
    assert stats.summary()['terminations'] == 1  # Terminal interval is retained.
    assert summary['by_command_sign'][QUADRANTS[1]]['far_forward_error_fraction'] is None

    # Ordinary invalid and numeric invalid intervals supply neither numerator
    # nor exposure. A valid row ending at reset remains credited above.
    stats.reset_window()
    step.valid[1] = False
    stats.record(step)
    summary = stats.summary()['joint_speed_turn_exposure']
    assert summary['valid_interval_seconds'] == .02
    assert summary['joint_active_seconds'] == .02
    assert summary['by_command_sign'][QUADRANTS[0]]['fraction_of_ordinary_policy_time'] == 1.
    assert summary['by_command_sign'][QUADRANTS[3]]['active_seconds'] == 0.
    step.numerical_invalid[0] = True
    stats.reset_window()
    stats.record(step)
    summary = stats.summary()['joint_speed_turn_exposure']
    assert summary['valid_interval_seconds'] == 0.
    assert summary['joint_active_seconds'] == 0.
    assert all(row['velocity_height_gate_mean'] is None for row in summary['by_command_sign'].values())
    # Nonfinite endpoint tracking has the same exclusion semantics as the
    # explicit invalid flag, even when its substep diagnostics are finite.
    step.valid.fill_(True)
    step.numerical_invalid.zero_()
    step.tracking['forward_error_m_s'].fill_(float('nan'))
    stats.reset_window()
    stats.record(step)
    assert stats.summary()['joint_speed_turn_exposure']['joint_active_seconds'] == 0.


def test_native_policy_clock_not_assumed_to_be_twenty_ms():
    composer, reference = make_composer(policy_dt_s=.01)
    result = run_interval(composer, command_reference(reference))
    stats = EpisodeStatistics(2, 'cpu', joint_speed_turn_exposure=composer.supermimic_joint_speed_turn_exposure)
    stats.record(ppo_step(result))
    assert stats.summary()['joint_speed_turn_exposure']['joint_active_seconds'] == .02


def test_measurement_preserves_reward_actions_rng_and_default_receipt():
    enabled, reference = make_composer()
    disabled = deepcopy(enabled)
    receipt = install_shared_reward(disabled)
    assert 'joint_speed_turn_exposure' not in receipt
    reference = command_reference(reference)
    rng = torch.get_rng_state().clone()
    actual, expected = run_interval(enabled, reference), run_interval(disabled, reference)
    assert torch.equal(torch.get_rng_state(), rng)
    for field in fields(expected):
        left, right = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(right, dict):
            for key, value in right.items():
                torch.testing.assert_close(left[key], value, atol=0, rtol=0)
        else:
            torch.testing.assert_close(left, right, atol=0, rtol=0)
    for name in ('_previous', '_previous2', '_last_dq', '_action_count'):
        torch.testing.assert_close(getattr(enabled, name), getattr(disabled, name), atol=0, rtol=0)
    # Reset starts a fresh physical window; no previous joint exposure leaks.
    enabled.reset(torch.ones(2, dtype=torch.bool), enabled._time.clone(),
                  enabled._episode + 1, enabled._control.clone())
    result = run_interval(enabled, command_reference(reference, (0., 0.), (0., 0.)))
    assert all(value.sum() == 0 for key, value in result.scoped_error_counts.items()
               if key.startswith(PREFIX))


def test_invalid_pre_reset_identity_is_not_relabelled_as_exposure():
    composer, reference = make_composer()
    reference = command_reference(reference)
    begin(composer, reference, torch.zeros(2, 6))
    physics = replace(sample(composer, 1), episode_id=composer._episode + 1)
    with pytest.raises(ValueError, match='PRE-reset'):
        composer.observe(physics, reference)
    assert not composer._window['scoped_counts']


def test_receipt_reaches_ppo_without_reconstructing_a_default_curriculum():
    composer, _ = make_composer(bounds=(-1., 1.))
    native = NativeCommandBoundary()
    native._formal_reward = SimpleNamespace(composer=composer)
    environment = CourseEnvironment(SimpleNamespace(env=native, _health=None))
    assert environment.joint_speed_turn_exposure['initial_forward_m_s'] == [-1., 1.]
    assert environment.joint_speed_turn_exposure['policy_dt_s'] == .02
