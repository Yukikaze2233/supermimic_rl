"""PPO reporting must not pair a pre-reset endpoint with the next command."""
import math
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.ppo.environment import CourseEnvironment
from supermimic_rl.ppo.trainer import EpisodeStatistics
from support.ppo import NativeCommandBoundary


def test_tracking_uses_interval_command_across_command_change_and_reset():
    native = NativeCommandBoundary()
    health = SimpleNamespace(check=lambda: None, sample_policy=lambda: None)
    environment = CourseEnvironment(SimpleNamespace(env=native, _health=health))
    result = environment.step(torch.zeros(2, 6))
    torch.testing.assert_close(result.tracking["forward_error_m_s"], torch.tensor([.05, .05]))
    torch.testing.assert_close(result.tracking["yaw_rate_error_rad_s"], torch.tensor([-.1, .1]))
    torch.testing.assert_close(result.tracking["height_error_m"], torch.tensor([-.01, -.08]))
    torch.testing.assert_close(result.diagnostics["command_height_m"], torch.tensor([.30, .32]))
    torch.testing.assert_close(result.diagnostics["actual_height_m"], torch.tensor([.29, .24]))
    torch.testing.assert_close(result.diagnostics["velocity_height_gate"], torch.tensor([.9, .3]))
    torch.testing.assert_close(result.diagnostics["reward_per_s/positive.forward"], torch.tensor([1.5, 2.]))
    torch.testing.assert_close(result.diagnostics["reward_per_s/cost.height"], torch.tensor([-.5, -2.]))
    torch.testing.assert_close(result.diagnostics["reward_event/failure"], torch.tensor([0., -4.]))
    torch.testing.assert_close(result.diagnostics["reward_gate/ordinary"], torch.ones(2))
    # Densities integrate once; the event is added without an extra dt factor.
    reconstructed = .02 * sum(value for name, value in result.diagnostics.items()
                              if name.startswith("reward_per_s/"))
    reconstructed += sum(value for name, value in result.diagnostics.items()
                         if name.startswith("reward_event/"))
    torch.testing.assert_close(reconstructed, result.reward)
    # Reporting does not change the next actor observation, reward or bootstrap.
    torch.testing.assert_close(result.observation.history[:, -1, :4], native._command)
    torch.testing.assert_close(result.observation.critic, native.obs_buf["critic"])
    torch.testing.assert_close(result.terminal_critic, torch.full((2, 182), 7.))
    torch.testing.assert_close(result.reward, torch.tensor([.02, -4.]))
    assert result.observation.label_valid.tolist() == [True, False]
    native._command.zero_()
    torch.testing.assert_close(result.diagnostics["command_height_m"], torch.tensor([.30, .32]))


def test_raw_diagnostics_exclude_invalid_rows_but_keep_terminal_sample():
    native = NativeCommandBoundary()
    environment = CourseEnvironment(SimpleNamespace(env=native, _health=SimpleNamespace(
        check=lambda: None, sample_policy=lambda: None)))
    result = environment.step(torch.zeros(2, 6))
    statistics = EpisodeStatistics(2, "cpu")
    statistics.record(result)
    first = statistics.summary()["tracking_diagnostics"]
    assert first["actual_height_m"]["samples"] == 2
    assert abs(first["command_height_m"]["mean"] - .31) < 1e-6
    assert abs(first["actual_height_m"]["mean"] - .265) < 1e-6
    assert abs(first["velocity_height_gate"]["mean"] - .6) < 1e-6
    assert first["reward_per_s/positive.forward"]["mean"] == 1.75
    assert first["reward_per_s/cost.height"]["mean"] == -1.25
    assert first["reward_event/failure"]["mean"] == -2.
    result.valid[1] = False
    result.diagnostics["actual_height_m"][1] = float("nan")
    statistics.record(result)
    summary = statistics.summary()
    assert summary["terminations"] == 1
    assert summary["invalid_transitions"] == 1
    measured = summary["tracking_diagnostics"]["actual_height_m"]
    assert measured["samples"] == 3
    assert abs(measured["mean"] - (.29 + .24 + .29) / 3) < 1e-6
    assert abs(measured["min"] - .24) < 1e-6
    assert abs(measured["max"] - .29) < 1e-6
    assert summary["tracking_diagnostics"]["reward_event/failure"]["samples"] == 3
    assert summary["tracking_diagnostics"]["reward_event/failure"]["mean"] == -4 / 3
    assert "not multiplied or divided by dt" in summary["tracking_diagnostic_reward_units"]["reward_event"]
    statistics.reset_window()
    result.valid.zero_()
    statistics.record(result)
    empty = statistics.summary()["tracking_diagnostics"]["actual_height_m"]
    assert empty == {"samples": 0, "mean": None, "min": None, "max": None}


def scoped_step(error, square, exposure):
    native = NativeCommandBoundary()
    name = "ordinary_horizontal_tracking/forward_m_s"
    native._formal_result.scoped_error_sums = {name: torch.tensor(error, dtype=torch.float64)}
    native._formal_result.scoped_error_square_sums = {name: torch.tensor(square, dtype=torch.float64)}
    native._formal_result.scoped_error_counts = {name: torch.tensor(exposure, dtype=torch.float64)}
    environment = CourseEnvironment(SimpleNamespace(env=native, _health=SimpleNamespace(
        check=lambda: None, sample_policy=lambda: None)))
    return environment.step(torch.zeros(2, 6))


def test_scoped_tracking_weights_substep_exposure_and_keeps_terminal_interval():
    # Three ticks at +2 m/s and one tick at -4 m/s. Averaging the
    # two interval RMS values would incorrectly produce 3 m/s.
    step = scoped_step([1.5, -1.], [3., 4.], [.75, .25])
    statistics = EpisodeStatistics(2, "cpu")
    statistics.record(step)
    scope = statistics.summary()["tracking_scopes"]["ordinary_horizontal_tracking/forward_m_s"]
    assert scope == {"equivalent_policy_intervals": 1., "mean_error": .5,
                     "rms_error": math.sqrt(7.)}
    assert statistics.summary()["terminations"] == 1
    # Endpoint commands and their legacy diagnostics remain a separate metric.
    assert statistics.summary()["rms/forward_error_m_s"] == pytest.approx(.05)
    step.valid[1] = False
    statistics.record(step)
    scope = statistics.summary()["tracking_scopes"]["ordinary_horizontal_tracking/forward_m_s"]
    assert scope["equivalent_policy_intervals"] == 1.75
    assert scope["mean_error"] == pytest.approx(2. / 1.75)
    assert scope["rms_error"] == pytest.approx(math.sqrt(10. / 1.75))


def test_scoped_tracking_excludes_inactive_rows_without_reporting_zero_error():
    step = scoped_step([0., 0.], [0., 0.], [0., 0.])
    statistics = EpisodeStatistics(2, "cpu")
    statistics.record(step)
    scope = statistics.summary()["tracking_scopes"]["ordinary_horizontal_tracking/forward_m_s"]
    assert scope == {"equivalent_policy_intervals": 0., "mean_error": None, "rms_error": None}
    statistics.reset_window()
    assert "tracking_scopes" not in statistics.summary()


def test_scoped_tracking_nonfinite_moment_excludes_the_same_denominator():
    step = scoped_step([1.5, float("nan")], [3., 4.], [.75, .25])
    statistics = EpisodeStatistics(2, "cpu")
    statistics.record(step)
    scope = statistics.summary()["tracking_scopes"]["ordinary_horizontal_tracking/forward_m_s"]
    assert scope == {"equivalent_policy_intervals": .75, "mean_error": 2., "rms_error": 2.}
    # Invalid diagnostics must not discard an otherwise valid PPO transition.
    assert statistics.summary()["valid_tracking_samples"] == 2
