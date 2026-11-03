"""Training-only jump assist: decaying selection, push-window force and separate outcome counts."""
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.course.jump_assist import BALLISTIC, EXTENSION, JumpAssistConfig, install_jump_assist

HOLD = 3


class Native:
    num_envs, _base_body_index = 3, 0

    def __init__(self):
        self.tasks = SimpleNamespace(
            physical_id=torch.full((3,), -1), success_id=torch.full((3,), -1),
            flight_seen=torch.zeros(3, dtype=torch.bool), stage=torch.full((3,), EXTENSION),
            reference_vz=torch.ones(3), launch=torch.ones(3))
        self.tasks.jump_reference = lambda: SimpleNamespace(
            clock_stage=self.tasks.stage, velocity_world_z_m_s=self.tasks.reference_vz,
            launch_velocity_m_s=self.tasks.launch)
        self.outcome = None   # scripted native decision applied by observe_physics
        self.tasks.observe_physics = lambda: self.outcome and self.outcome()
        self._terrain_course = SimpleNamespace(tasks=self.tasks)
        self.wrenches, self.native_calls = [], 0
        self.robot = SimpleNamespace(
            data=SimpleNamespace(root_com_vel_w=torch.zeros(3, 6)),
            instantaneous_wrench_composer=SimpleNamespace(
                add_forces_and_torques_index=lambda **kwargs: self.wrenches.append(kwargs)))
        self._formal_reward = SimpleNamespace(composer=SimpleNamespace(_mass_kg=torch.full((3,), 20.)))

    @staticmethod
    def _as_torch(value):
        return value

    def _apply_action(self):
        self.native_calls += 1


def curriculum(transitions):
    return SimpleNamespace(stage_names=("flat", "active_jump", "dynamic"), stage_budgets=(100, 200, 100),
                           real_transitions=transitions)


def test_probability_fades_inside_the_assist_stage_only():
    native = Native()
    assist = install_jump_assist(native, curriculum(0), {"probability": .8, "fade_stage_fraction": .5})
    for transitions, expected in ((0, 0.), (100, .8), (150, .4), (200, 0.), (299, 0.), (300, 0.)):
        assist.curriculum.real_transitions = transitions
        assert assist.probability() == pytest.approx(expected)
    with pytest.raises(ValueError):
        JumpAssistConfig(probability=1.5)


def test_selected_requests_get_capped_shortfall_force_until_flight_and_outcomes_are_split():
    native = Native()
    assist = install_jump_assist(native, curriculum(100), {"probability": 1., "velocity_gain_per_s": 10.,
                                                           "max_weight_fraction": .5})
    native._apply_action()
    # Inside the assist stage the wrench is always added (no host sync); it is zero without a request.
    assert native.native_calls == 1 and native.wrenches[-1]['forces'].abs().sum() == 0
    native.tasks.physical_id[:] = torch.tensor([4, 5, 6])
    native.robot.data.root_com_vel_w[:, 2] = torch.tensor([0., .8, 0.])
    native.tasks.stage[:] = torch.tensor([EXTENSION, BALLISTIC, HOLD])
    native._apply_action()
    force = native.wrenches[-1]["forces"][:, 0, 2]
    # Row 0: 20 kg * 10/s * 1 m/s = 200 N, capped at 0.5 * 20 * 9.81; row 1: 20 * 10 * 0.2;
    # row 2's reference is past its push window.
    torch.testing.assert_close(force, torch.tensor([98.1, 40., 0.]))
    assert native.wrenches[-1]["is_global"] and native.wrenches[-1]["body_ids"] == [0]
    native.tasks.flight_seen[0] = True
    native._apply_action()
    assert native.wrenches[-1]["forces"][0, 0, 2] == 0
    assist.curriculum.real_transitions = 299   # fade complete: new requests are unassisted

    def decide():
        native.tasks.success_id[0] = 4
        native.tasks.physical_id[:] = torch.tensor([-1, -1, 7])

    native.outcome = decide
    native.tasks.observe_physics()
    native.tasks.success_id.fill_(-1)        # the native policy step clears it before the next substep
    native._apply_action()
    counters, gauges = assist.metrics()
    assert counters["course/jump_assist/assisted_requests"] == 3
    assert counters["course/jump_assist/assisted_ended"] == 3 and counters["course/jump_assist/assisted_success"] == 1
    native.tasks.physical_id[2] = -1
    native._apply_action()
    counters, _ = assist.metrics()
    assert counters["course/jump_assist/unassisted_ended"] == 1 and counters["course/jump_assist/unassisted_success"] == 0
    assert gauges["course/jump_assist/probability"] == 0.


def test_outside_the_assist_stage_no_wrench_is_added_but_outcomes_are_counted():
    native = Native()
    assist = install_jump_assist(native, curriculum(350), {"probability": 1.})
    native.tasks.physical_id[:] = torch.tensor([4, -1, -1])
    native._apply_action()
    assert not native.wrenches
    native.outcome = lambda: (native.tasks.success_id.__setitem__(0, 4), native.tasks.physical_id.__setitem__(0, -1))
    native.tasks.observe_physics()
    counters, _ = assist.metrics()
    assert counters["course/jump_assist/unassisted_success"] == 1
    assert counters["course/jump_assist/unassisted_ended"] == 1 and counters["course/jump_assist/assisted_requests"] == 0
