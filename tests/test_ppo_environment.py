"""PPO reporting must not pair a pre-reset endpoint with the next command."""
from types import SimpleNamespace

import torch

from supermimic_rl.ppo.environment import CourseEnvironment
from supermimic_rl.ppo.trainer import EpisodeStatistics


class NativeCommandBoundary:
    num_envs = 2
    device = "cpu"

    def __init__(self):
        self._command = torch.tensor([[.4, 0., .2, .30], [-.3, 0., -.5, .32]])
        self._terminal_critic_182 = torch.full((2, 182), 7.)
        self.obs_buf = {"policy": torch.zeros(2, 5, 42), "critic": torch.zeros(2, 182)}
        self._last_state = SimpleNamespace(
            velocity_origin_body_m_s=torch.tensor([[.45, 0., 0.], [-.25, 0., 0.]]),
            angular_velocity_body_rad_s=torch.tensor([[0., 0., .1], [0., 0., -.4]]),
            gravity_body=torch.tensor([[0., 0., -1.], [0., 0., -1.]]),
            height_above_ground_m=torch.tensor([.29, .24]),
            wheel_ground_force_n=torch.full((2, 2), 100.),
            contact_measurement_valid=torch.ones(2, dtype=torch.bool))
        # Deliberately differs from a gate reconstructed at the endpoint:
        # the producer averages all four physical substeps.
        self._formal_result = SimpleNamespace(
            gate_fractions={"velocity_height_gate": torch.tensor([.9, .3]),
                            "ordinary": torch.ones(2)},
            component_densities={"positive.forward": torch.tensor([1.5, 2.]),
                                 "cost.height": torch.tensor([-.5, -2.])},
            event_rewards={"failure": torch.tensor([0., -4.]),
                           "physical_skill_success": torch.zeros(2)},
            physical_step_reward=torch.tensor([.02, -4.], dtype=torch.float64),
            learning_reward=torch.tensor([.02, -4.]))

    def step(self, action):
        # Row 0 changes command; row 1 terminates and receives a reset command.
        # Both mutations happen after the reported physical endpoint.
        self._command.copy_(torch.tensor([[-2., 0., 3., .4], [2., 0., 4., .28]]))
        self.obs_buf["policy"][:, -1, :4] = self._command
        self.obs_buf["critic"].fill_(8.)
        return (self.obs_buf, torch.tensor([.02, -4.]), torch.tensor([False, True]),
                torch.zeros(2, dtype=torch.bool),
                {"interval_observation_valid": torch.ones(2, dtype=torch.bool),
                 "numerical_invalid": torch.zeros(2, dtype=torch.bool)})


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
