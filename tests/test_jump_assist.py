"""Training-only jump assist: decaying selection, push-window force and separate outcome counts."""
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.course.jump_assist import BALLISTIC, EXTENSION, JumpAssistConfig, install_jump_assist

HOLD = 3


class Native:
    num_envs, _base_body_index, physics_dt = 3, 0, .005

    def __init__(self):
        self.tasks = SimpleNamespace(
            physical_id=torch.full((3,), -1), success_id=torch.full((3,), -1),
            flight_seen=torch.zeros(3, dtype=torch.bool), ascent_seen=torch.zeros(3, dtype=torch.bool),
            air_ticks=torch.zeros(3, dtype=torch.long), stage=torch.full((3,), EXTENSION),
            reference_vz=torch.ones(3), launch=torch.ones(3), active_rise=torch.tensor([.05, .25, .35]),
            peak_rise=torch.zeros(3), supported_start=torch.ones(3, dtype=torch.bool), landed_s=torch.zeros(3),
            ground_anchor=torch.zeros(3), latched_height=torch.full((3,), .32))
        self.tasks.jump_reference = lambda: SimpleNamespace(
            clock_stage=self.tasks.stage, velocity_world_z_m_s=self.tasks.reference_vz,
            launch_velocity_m_s=self.tasks.launch)
        self.outcome = None   # scripted native decision applied by observe_physics
        self.tasks.observe_physics = lambda *args, **kwargs: self.outcome and self.outcome()
        self._terrain_course = SimpleNamespace(tasks=self.tasks)
        self.wrenches, self.native_calls = [], 0
        self.robot = SimpleNamespace(
            data=SimpleNamespace(root_com_vel_w=torch.zeros(3, 6)),
            instantaneous_wrench_composer=SimpleNamespace(
                add_forces_and_torques_index=lambda **kwargs: self.wrenches.append(kwargs)))
        self._formal_reward = SimpleNamespace(composer=SimpleNamespace(_mass_kg=torch.full((3,), 20.)))

    def observe(self, force=20., velocity_z=0., tilt=0., height=.32, angular_speed=0.):
        """One native observe_physics substep with uniform physical measurements."""
        full = lambda value: torch.full((3,), float(value))
        self.tasks.observe_physics(torch.full((3, 2), float(force)), full(velocity_z), full(tilt), full(height),
                                   angular_speed_rad_s=full(angular_speed))

    @staticmethod
    def _as_torch(value):
        return value

    def _apply_action(self):
        self.native_calls += 1


def curriculum(transitions):
    return SimpleNamespace(stage_names=("flat", "active_jump", "dynamic"), stage_budgets=(100, 200, 100),
                           real_transitions=transitions)


def test_level_fades_inside_the_assist_stage_only():
    native = Native()
    assist = install_jump_assist(native, curriculum(0), {"probability": .8, "fade_stage_fraction": .5})
    for transitions, expected in ((0, 0.), (100, .8), (150, .4), (200, 0.), (299, 0.), (300, 0.)):
        assist.curriculum.real_transitions = transitions
        assert assist.level() == pytest.approx(expected)
    with pytest.raises(ValueError):
        JumpAssistConfig(probability=1.5)


def test_selected_requests_get_capped_shortfall_force_until_the_first_touchdown_and_outcomes_are_split():
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
    # Row 0 (extension): 20 kg * 10/s * (1 m/s launch - 0) = 200 N, capped at 0.5 * 20 * 9.81;
    # row 1 (rising ballistic): 20 * 10 * (1 m/s reference - .8); row 2's reference is past its apex.
    torch.testing.assert_close(force, torch.tensor([98.1, 40., 0.]))
    assert native.wrenches[-1]["is_global"] and native.wrenches[-1]["body_ids"] == [0]
    # Native flight_seen from wheel unloading without rising keeps the assist on.
    native.tasks.flight_seen[0] = True
    native.tasks.air_ticks[0] = 2
    native.observe()         # the shared first-flight latch observes each tick
    native._apply_action()
    assert native.wrenches[-1]["forces"][0, 0, 2] > 0
    native.tasks.ascent_seen[0] = True     # real liftoff: the reference still rises, keep closing the gap
    native.observe()
    native._apply_action()
    assert native.wrenches[-1]["forces"][0, 0, 2] > 0
    assert native.tasks.supermimic_flight_latch.lifted[0]
    native.tasks.air_ticks[0] = 0          # the first touchdown after real flight ends the assist
    native.observe()
    native._apply_action()
    assert native.wrenches[-1]["forces"][0, 0, 2] == 0
    assist.curriculum.real_transitions = 299   # fade complete: new requests are unassisted

    def decide():
        native.tasks.success_id[0] = 4
        native.tasks.physical_id[:] = torch.tensor([-1, -1, 7])

    native.outcome = decide
    native.observe()
    native.tasks.success_id.fill_(-1)        # the native policy step clears it before the next substep
    native._apply_action()
    counters, gauges = assist.metrics()
    assert counters["course/jump_assist/assisted_requests"] == 3
    # Request 6 was selected but never received force (its push window had passed): unassisted.
    assert counters["course/jump_assist/assisted_ended"] == 2 and counters["course/jump_assist/assisted_success"] == 1
    assert counters["course/jump_assist/unassisted_ended"] == 1
    native.tasks.physical_id[2] = -1
    native._apply_action()
    counters, _ = assist.metrics()
    assert counters["course/jump_assist/unassisted_ended"] == 2 and counters["course/jump_assist/unassisted_success"] == 0
    assert gauges["course/jump_assist/level"] == 0.


def test_outside_the_assist_stage_no_wrench_is_added_but_outcomes_are_counted():
    native = Native()
    assist = install_jump_assist(native, curriculum(350), {"probability": 1.})
    native.tasks.physical_id[:] = torch.tensor([4, -1, -1])
    native._apply_action()
    assert not native.wrenches
    native.outcome = lambda: (native.tasks.success_id.__setitem__(0, 4), native.tasks.physical_id.__setitem__(0, -1))
    native.observe()
    counters, _ = assist.metrics()
    assert counters["course/jump_assist/unassisted_success"] == 1
    assert counters["course/jump_assist/unassisted_ended"] == 1 and counters["course/jump_assist/assisted_requests"] == 0


def test_force_decay_assists_every_request_with_a_fading_scale_and_bins_unassisted_outcomes():
    native = Native()
    assist = install_jump_assist(native, curriculum(150), {"probability": 1., "decay": "force",
                                                           "fade_stage_fraction": .5, "max_weight_fraction": 10.})
    native.tasks.physical_id[:] = torch.tensor([4, 5, 6])
    native._apply_action()
    # Level 0.5 at the stage midpoint: every request selected, force halved (20 kg * 10/s * 1 m/s * 0.5).
    torch.testing.assert_close(native.wrenches[-1]["forces"][:, 0, 2], torch.full((3,), 100.))
    counters, gauges = assist.metrics()
    assert counters["course/jump_assist/assisted_requests"] == 3 and gauges["course/jump_assist/level"] == .5
    assist.curriculum.real_transitions = 250   # faded out: new requests are unassisted
    native.tasks.physical_id[:] = torch.tensor([7, 8, 9])
    native._apply_action()

    def decide():
        native.tasks.success_id[:] = torch.tensor([7, -1, 9])
        native.tasks.physical_id[:] = -1

    native.outcome = decide
    native.observe()
    counters, _ = assist.metrics()
    assert counters["course/jump_assist/unassisted_ended"] == 3 and counters["course/jump_assist/unassisted_success"] == 2
    assert counters["course/jump_assist/unassisted_success_rise_0.00-0.10"] == 1
    assert counters["course/jump_assist/unassisted_ended_rise_0.20-0.30"] == 1
    assert counters["course/jump_assist/unassisted_success_rise_0.20-0.30"] == 0
    assert counters["course/jump_assist/unassisted_success_rise_0.30-0.40"] == 1
    # All-request bins include the earlier assisted requests 4-6 (rises 0.05/0.25/0.35, none succeeded).
    assert counters["course/jump_assist/ended_rise_0.20-0.30"] == 2
    assert counters["course/jump_assist/success_rise_0.00-0.10"] == 1


def test_force_decay_keeps_a_fixed_unassisted_request_cohort_at_a_held_gate():
    native = Native()
    assist = install_jump_assist(native, curriculum(150), {
        'probability': 1., 'decay': 'force', 'fade_stage_fraction': .5,
        'max_weight_fraction': 10., 'assist_request_probability': .5, 'seed': 1})
    native.tasks.physical_id[:] = torch.tensor([4, 5, 6])
    native._apply_action()
    # Seed 1 chooses [unassisted, assisted, assisted]. The selected rows still
    # receive the same fading force amplitude, not a weaker force on all rows.
    assert assist.selected.tolist() == [False, True, True]
    torch.testing.assert_close(native.wrenches[-1]['forces'][:, 0, 2], torch.tensor([0., 100., 100.]))
    for _ in range(3):
        native._apply_action()
        assert assist.selected.tolist() == [False, True, True]
    assert assist.impulse[0] == 0 and (assist.impulse[1:] > 0).all()

    def decide():
        native.tasks.success_id[:] = torch.tensor([4, 5, 6])
        native.tasks.physical_id[:] = -1

    native.outcome = decide
    native.observe()
    counters, _ = assist.metrics()
    assert counters['course/jump_assist/unassisted_success'] == 1
    assert counters['course/jump_assist/unassisted_ended'] == 1
    assert counters['course/jump_assist/assisted_success'] == 2
    assert counters['course/jump_assist/unassisted_success_rise_0.00-0.10'] == 1
    assert assist.level() == .5
    with pytest.raises(ValueError):
        JumpAssistConfig(probability=1., assist_request_probability=1.1)


def test_ended_requests_are_classified_against_the_native_peak_window():
    native = Native()
    assist = install_jump_assist(native, curriculum(100), {"probability": 0.})
    native.tasks.physical_id[:] = torch.tensor([4, 5, 6])
    native._apply_action()
    # Requested rises .05/.25/.35 m: windows .025-.075, .22-.28, .32-.38.
    native.tasks.peak_rise[:] = torch.tensor([.02, .25, .40])
    native.tasks.physical_id[:] = -1
    native._apply_action()
    counters, _ = assist.metrics()
    assert counters["course/jump_assist/peak_low_rise_0.00-0.10"] == 1
    assert counters["course/jump_assist/peak_ok_rise_0.20-0.30"] == 1
    assert counters["course/jump_assist/unassisted_peak_high_rise_0.30-0.40"] == 1


def test_settling_ticks_after_a_correct_jump_are_split_by_native_condition():
    native = Native()
    assist = install_jump_assist(native, curriculum(100), {"probability": 0.})
    tasks = native.tasks
    tasks.physical_id[:] = torch.tensor([4, 5, -1])
    tasks.flight_seen[:] = True
    tasks.ascent_seen[:] = True
    tasks.peak_rise[:] = torch.tensor([.05, .02, .35])       # row 1 below its .25 m window
    native.observe(velocity_z=.3)                             # grounded but still moving vertically
    native.observe()
    tasks.landed_s[0] = 1.2
    native.observe(angular_speed=.8)
    counters, _ = assist.metrics()
    assert counters["course/jump_assist/settle_ticks"] == 3
    assert counters["course/jump_assist/settle_vertical_speed"] == 2
    assert counters["course/jump_assist/settle_angular_speed"] == 2
    assert counters["course/jump_assist/settle_all"] == 1
    native._apply_action()
    tasks.physical_id[:] = -1
    native._apply_action()
    counters, _ = assist.metrics()
    assert counters["course/jump_assist/streak_1.0s"] == 1 and counters["course/jump_assist/streak_1.5s"] == 0
