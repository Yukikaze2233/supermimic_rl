"""Performance-gated course coordinate: holding, opening, ratios, timeout and trainer wiring."""
import json

import pytest

from supermimic_rl.ppo.course_gate import CourseGates, GateCriterion, GateTimeout, StageGate
from supermimic_rl.ppo.trainer import PPOTrainer

from support.ppo import TargetEnvironment, config

STARTS = {"flat": 0, "jump": 100, "terrain": 300}


def gates(**overrides):
    jump = {"stage": "jump", "window": 2, "timeout_updates": 5,
            "criteria": [{"metric": "rms/height_error_m", "max": .03}]}
    jump.update(overrides)
    return [jump]


def test_coordinate_waits_before_a_gated_stage_until_the_window_holds():
    course = CourseGates(gates(), STARTS)
    assert course.advance({"rms/height_error_m": .1}, 60)["coordinate"] == 60
    assert course.advance({"rms/height_error_m": .1}, 60)["coordinate"] == 99
    held = course.advance({"rms/height_error_m": .01}, 60)
    assert held["gate_passed"] is False and held["coordinate"] == 99   # window still has 0.1
    opened = course.advance({"rms/height_error_m": .01}, 60)
    assert opened["gate_passed"] is True and opened["coordinate"] == 159 and opened["waiting_for"] is None


def test_ratio_criteria_use_summed_counters_and_a_minimum_count():
    criterion = GateCriterion(numerator=("ok",), denominator=("ok", "bad"), min=.5, min_count=10)
    assert criterion.value([{"ok": 2, "bad": 2}]) is None
    assert criterion.value([{"ok": 4, "bad": 2}, {"ok": 4, "bad": 2}]) == pytest.approx(2 / 3)
    nested = GateCriterion(metric="tracking_diagnostics.velocity_height_gate.mean", min=.9)
    assert nested.value([{"tracking_diagnostics": {"velocity_height_gate": {"mean": .95}}}]) == .95


def test_height_bin_decimal_points_are_literal_metric_keys():
    criterion = GateCriterion(
        numerator=("course/jump_assist/unassisted_success_rise_0.00-0.10",),
        denominator=("course/jump_assist/unassisted_ended_rise_0.00-0.10",), min=.4, min_count=200)
    record = {criterion.numerator[0]: 100, criterion.denominator[0]: 200}
    assert criterion.value([record]) == .5
    assert criterion.holds(criterion.value([record]))


def test_closed_gate_times_out_and_state_round_trips():
    course = CourseGates(gates(timeout_updates=2), STARTS, cap=250)
    course.advance({"rms/height_error_m": .1}, 200)
    state = course.state_dict()
    restored = CourseGates(gates(timeout_updates=2), STARTS, cap=250)
    restored.load_state_dict(state)
    assert restored.coordinate == 99
    restored.advance({"rms/height_error_m": .1}, 200)
    restored.advance({"rms/height_error_m": .1}, 200)
    with pytest.raises(GateTimeout, match="jump"):
        restored.advance({"rms/height_error_m": .1}, 200)


def test_gate_configuration_is_validated():
    with pytest.raises(ValueError, match="not in the course"):
        CourseGates([{"stage": "missing", "criteria": [{"metric": "x", "max": 1}]}], STARTS)
    with pytest.raises(ValueError, match="min or max"):
        StageGate("jump", ({"metric": "x"},))
    with pytest.raises(ValueError, match="offset"):
        config(course_gates=gates(), curriculum_transition_offset=10)


class GatedEnvironment(TargetEnvironment):
    def __init__(self, num_envs=4):
        super().__init__(num_envs)
        self.successes = 0

    def course_stage_starts(self):
        return {"flat": 0, "jump": 100, "terrain": 300}

    def course_progress(self):
        self.successes += 3
        return {"course/jump/verified_success": self.successes}, {"course/terrain/level_fraction": 0.}


def test_trainer_records_counter_increments_and_stops_on_gate_timeout(tmp_path):
    course = [{"stage": "jump", "window": 1, "timeout_updates": 100,
               "criteria": [{"metric": "rms/forward_error_m_s", "max": 1e9}]},
              {"stage": "terrain", "window": 1, "timeout_updates": 1,
               "criteria": [{"metric": "rms/forward_error_m_s", "max": -1.}]}]
    environment = GatedEnvironment()
    trainer = PPOTrainer(environment, config(updates=40, steps_per_update=10, course_gates=course), tmp_path)
    result = trainer.train()
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert all(record["course/jump/verified_success"] == 3 for record in records)
    assert result["status"] == "gate_timeout" and "terrain" in result["message"]
    assert trainer.curriculum_transitions == 299 and result["updates"] < 40
    assert [record["curriculum_transitions"] for record in records[:3]] == [0, 40, 80]
    assert records[2]["course"]["coordinate"] == 99 and records[3]["course"]["gate_passed"] is True
    assert (tmp_path / "checkpoint.pt").is_file()


def test_native_course_counters_and_stage_starts_are_read_without_policy_inputs():
    from types import SimpleNamespace
    from supermimic_rl.ppo.environment import CourseEnvironment

    metrics = {"physical_request_outcomes": {"verified_success": 5, "reset_before_verified_success": 1,
                                             "closed_before_verified_success": 2},
               "stair_closed_loop": {"requests_accepted": 4, "skill_exits": 3, "climbed_75pct_of_request": 2},
               "authored_stair_completion": {"ended_by_tile": [1, 3], "ended_complete_by_tile": [0, 2]},
               "recovery": {"counts": {"births": 6, "verified_physical_successes": 2}},
               "episode_birth_family_counts": {"height": 10, "request": 4},
               "level_histogram": [2, 0, 2]}
    environment = object.__new__(CourseEnvironment)
    environment.env = SimpleNamespace(_terrain_course=SimpleNamespace(metrics=lambda: metrics))
    environment.adapter = SimpleNamespace(curriculum=SimpleNamespace(
        stage_names=("a", "b", "c"), stage_budgets=(10, 20, 30), jump_height_level=lambda: 1,
        command_context=lambda: SimpleNamespace(domain=SimpleNamespace(jump_m=(.03, .2)))))
    counters, gauges = environment.course_progress()
    assert counters["course/jump/verified_success"] == 5 and counters["course/stair/authored_ended"] == 4
    assert counters["course/stair/authored_ended_complete"] == 2 and counters["course/recovery/births"] == 6
    assert counters["course/births/request"] == 4 and gauges["course/terrain/level_fraction"] == .5
    assert gauges["course/jump/height_cap_m"] == .2 and gauges["course/jump/height_level"] == 1.
    assert environment.course_stage_starts() == {"a": 0, "b": 10, "c": 30}


def test_restarting_the_hold_keeps_the_coordinate_and_clears_the_old_window():
    course = CourseGates(gates(timeout_updates=1), STARTS)
    course.advance({"rms/height_error_m": .1}, 200)
    course.advance({"rms/height_error_m": .1}, 200)
    restored = CourseGates(gates(timeout_updates=1), STARTS)
    restored.load_state_dict(course.state_dict(), restart_hold=True)
    assert restored.coordinate == 99 and restored.held_updates == 0 and not restored.history
    held = restored.advance({"rms/height_error_m": .01}, 200)
    assert held["gate_passed"] is False   # a fresh window must refill first
    assert restored.advance({"rms/height_error_m": .01}, 200)["gate_passed"] is True


def test_rewind_repeats_a_stage_and_closes_only_later_gates():
    course = CourseGates(gates(window=1) + [{"stage": "terrain", "window": 1, "timeout_updates": 9,
                                             "criteria": [{"metric": "rms/height_error_m", "max": -1.}]}], STARTS)
    for _ in range(4):
        course.advance({"rms/height_error_m": .01}, 200)
    assert course.coordinate == 299 and course.opened == ["jump"] and course.held_updates == 2
    receipt = course.rewind("jump")
    assert receipt["coordinate_before"] == 299 and course.coordinate == 100 and course.opened == ["jump"]
    assert course.held_updates == 0 and not course.history
    with pytest.raises(ValueError):
        course.rewind("terrain")


def test_resume_rewinds_once_and_not_again_from_its_own_checkpoint(tmp_path):
    course = [{"stage": "jump", "window": 1, "timeout_updates": 100,
               "criteria": [{"metric": "rms/forward_error_m_s", "max": 1e9}]},
              {"stage": "terrain", "window": 1, "timeout_updates": 1,
               "criteria": [{"metric": "rms/forward_error_m_s", "max": -1.}]}]
    first = PPOTrainer(GatedEnvironment(), config(updates=40, steps_per_update=10, course_gates=course),
                       tmp_path / "first")
    first.train()
    assert first.curriculum_transitions == 299
    settings = config(updates=60, steps_per_update=10, course_gates=course, course_rewind_stage="jump")
    second = PPOTrainer(GatedEnvironment(), settings, tmp_path / "second", resume=tmp_path / "first" / "checkpoint.pt")
    assert second.curriculum_transitions == 100
    receipt = json.loads((tmp_path / "second" / "course-rewind.json").read_text())
    assert receipt["coordinate_before"] == 299 and receipt["opened"] == ["jump"]
    second.save(tmp_path / "second" / "checkpoint.pt")
    third = PPOTrainer(GatedEnvironment(), settings, tmp_path / "third", resume=tmp_path / "second" / "checkpoint.pt")
    assert third.curriculum_transitions == 100 and not (tmp_path / "third" / "course-rewind.json").exists()


def test_stage_action_std_floor_is_raised_once_on_entering_the_stage(tmp_path):
    import torch

    course = [{"stage": "jump", "window": 1, "timeout_updates": 100,
               "criteria": [{"metric": "rms/forward_error_m_s", "max": 1e9}]}]
    settings = config(updates=8, steps_per_update=10, initial_std=.1, course_gates=course,
                      stage_action_std={"jump": [.3, .3, .3, .3, .05, .05]})
    trainer = PPOTrainer(GatedEnvironment(), settings, tmp_path)
    trainer.train()
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    raised = [record.get("action_std_floor_stage") for record in records]
    assert raised.count("jump") == 1 and raised[0] is None      # flat first, then jump once
    assert trainer.stage_action_std_applied == ["jump"]
    std = trainer.actor.log_std.detach().exp()
    assert (std[:4] >= .29).all()                                # legs raised to the floor (then trained)
    restored = PPOTrainer(GatedEnvironment(), settings, tmp_path / "again", resume=tmp_path / "checkpoint.pt")
    assert restored.stage_action_std_applied == ["jump"]
    with pytest.raises(ValueError, match="six positive"):
        config(course_gates=course, stage_action_std={"jump": [.3]})


def test_a_new_rewind_tag_repeats_the_rewind_once(tmp_path):
    course = [{"stage": "jump", "window": 1, "timeout_updates": 100,
               "criteria": [{"metric": "rms/forward_error_m_s", "max": 1e9}]},
              {"stage": "terrain", "window": 1, "timeout_updates": 1,
               "criteria": [{"metric": "rms/forward_error_m_s", "max": -1.}]}]
    first = PPOTrainer(GatedEnvironment(), config(updates=40, steps_per_update=10, course_gates=course,
                                                   course_rewind_stage="jump"), tmp_path / "first")
    first.train()
    tagged = config(updates=80, steps_per_update=10, course_gates=course, course_rewind_stage="jump",
                    course_rewind_tag="r7")
    second = PPOTrainer(GatedEnvironment(), tagged, tmp_path / "second", resume=tmp_path / "first" / "checkpoint.pt")
    assert second.curriculum_transitions == 100
    second.course.coordinate = 250
    second.save(tmp_path / "second" / "checkpoint.pt")
    third = PPOTrainer(GatedEnvironment(), tagged, tmp_path / "third", resume=tmp_path / "second" / "checkpoint.pt")
    assert third.curriculum_transitions == 250


def test_gates_inside_a_stage_hold_the_coordinate_at_their_fraction():
    course = CourseGates(gates(window=1) + [
        {"stage": "jump", "at_fraction": .5, "window": 1, "timeout_updates": 9,
         "criteria": [{"metric": "rms/height_error_m", "max": .01}]}], STARTS)
    course.advance({"rms/height_error_m": .02}, 120)       # flat: jump gate waits at 99
    course.advance({"rms/height_error_m": .02}, 120)       # passes the jump gate, moves to 199 (jump@0.5 at 200)
    assert course.coordinate == 199 and course.opened == ["jump"]
    held = course.advance({"rms/height_error_m": .02}, 120)
    assert held["gate"] == "jump@0.5" and not held["gate_passed"] and course.coordinate == 199
    passed = course.advance({"rms/height_error_m": .005}, 120)
    assert passed["gate_passed"] and course.opened == ["jump", "jump@0.5"] and course.coordinate == 319
    course.rewind("jump")
    assert course.opened == ["jump"] and course.coordinate == 100
    with pytest.raises(ValueError, match="bounded stage"):
        CourseGates([{"stage": "terrain", "at_fraction": .5, "window": 1, "criteria": [
            {"metric": "rms/height_error_m", "max": .01}]}], STARTS)


def test_report_mode_never_holds_or_stops_and_opens_a_stage_ladder_in_order():
    course = CourseGates(gates(window=1) + [
        {"stage": "jump", "at_fraction": .5, "window": 1, "timeout_updates": 1,
         "criteria": [{"metric": "rms/height_error_m", "max": .01}]}], STARTS, mode="report")
    status = course.advance({"rms/height_error_m": .02}, 120)
    assert course.coordinate == 120 and status["gates"] == {}           # no gate reached yet
    status = course.advance({"rms/height_error_m": .02}, 120)        # jump gate reached and passes
    assert course.opened == ["jump"] and course.coordinate == 240
    for _ in range(5):                                                # far past its timeout: no stop
        status = course.advance({"rms/height_error_m": .02}, 120)
    assert course.coordinate == 840 and course.opened == ["jump"]
    assert status["gates"]["jump@0.5"]["passed"] is False and course.waiting["jump@0.5"] == 5
    course.advance({"rms/height_error_m": .005}, 120)
    assert course.opened == ["jump", "jump@0.5"] and course.waiting == {}
    restored = CourseGates(gates(window=1), STARTS, mode="report")
    with pytest.raises(ValueError, match="mode"):
        CourseGates(gates(), STARTS, mode="skip")
    state = CourseGates(gates(window=1), STARTS, mode="report").state_dict()
    restored.load_state_dict({key: value for key, value in state.items() if key != "waiting"})
    assert restored.waiting == {}


@pytest.mark.parametrize('height_progression', ['performance', 'schedule'])
def test_report_mode_trainer_keeps_training_and_applies_opened_gates(tmp_path, height_progression):
    class LadderEnvironment(GatedEnvironment):
        opened = None

        def set_opened_course_gates(self, keys):
            self.opened = None if keys is None else list(keys)

        def check_course_gates(self, keys):
            self.checked = list(keys)

    course = [{"stage": "jump", "window": 1, "timeout_updates": 1,
               "criteria": [{"metric": "rms/forward_error_m_s", "max": 1e9}]},
              {"stage": "terrain", "window": 1, "timeout_updates": 1,
               "criteria": [{"metric": "rms/forward_error_m_s", "max": -1.}]}]
    environment = LadderEnvironment()
    trainer = PPOTrainer(environment, config(updates=12, steps_per_update=10, course_gates=course,
                                             course_gate_mode="report",
                                             jump_height_progression=height_progression), tmp_path)
    assert environment.opened == ([] if height_progression == 'performance' else None)
    result = trainer.train()
    assert result["status"] == "complete" and trainer.curriculum_transitions == 480
    assert environment.opened == (["jump"] if height_progression == 'performance' else None)
    assert trainer.course.opened == ['jump']
    assert trainer.course.waiting['terrain'] > 0
    record = json.loads((tmp_path / 'metrics.jsonl').read_text().splitlines()[-1])
    assert record['course']['jump_height_progression'] == height_progression
    following = 'schedule' if height_progression == 'performance' else 'performance'
    resumed = PPOTrainer(LadderEnvironment(), config(updates=13, steps_per_update=10, course_gates=course,
        course_gate_mode='report', jump_height_progression=following), tmp_path / 'resumed',
        resume=tmp_path / 'checkpoint.pt')
    assert resumed.course.state_dict() == trainer.course.state_dict()
    assert resumed.updates == trainer.updates and resumed.transitions == trainer.transitions
    assert resumed.resume_migrations[-1]['jump_height_progression'] == {
        'before': height_progression, 'after': following}
    import torch
    for key, value in trainer.actor.state_dict().items():
        torch.testing.assert_close(resumed.actor.state_dict()[key], value, rtol=0, atol=0)
    for key, state in trainer.optimizer.state_dict()['state'].items():
        for field, value in state.items():
            torch.testing.assert_close(resumed.optimizer.state_dict()['state'][key][field], value, rtol=0, atol=0)
    with pytest.raises(ValueError, match="never stops"):
        config(course_gates=course, course_gate_mode="report", stop_after_gate=True)
    with pytest.raises(ValueError, match='jump_height_progression'):
        config(jump_height_progression='skip')


def test_report_mode_ladder_needs_a_fresh_window_per_step_and_rewind_restarts_it():
    ladder = [{"stage": "jump", "at_fraction": fraction, "window": 2, "timeout_updates": 1,
               "criteria": [{"metric": "rms/height_error_m", "max": .01}]} for fraction in (.25, .5, .75)]
    course = CourseGates(ladder, STARTS, mode="report")
    good = {"rms/height_error_m": .001}
    course.advance(good, 400)                      # far past every step at once
    course.advance(good, 1)
    assert course.opened == ["jump@0.25"]          # the next steps wait for windows at the new height
    course.advance(good, 1)
    assert course.opened == ["jump@0.25"]
    course.advance(good, 1)
    assert course.opened == ["jump@0.25", "jump@0.5"]
    course.advance(good, 1)
    course.advance(good, 1)
    assert course.opened == ["jump@0.25", "jump@0.5", "jump@0.75"]
    state = course.state_dict()
    restored = CourseGates(ladder, STARTS, mode="report")
    restored.load_state_dict(state)
    assert restored.opened_at == course.opened_at and restored.records == course.records


def test_report_mode_rejects_a_course_ladder_without_configured_gates():
    from types import SimpleNamespace
    from supermimic_rl.ppo.environment import CourseEnvironment

    environment = object.__new__(CourseEnvironment)
    environment.adapter = SimpleNamespace(curriculum=SimpleNamespace(
        ladder_gate_keys=lambda: ("active_jump@0.25", "active_jump@0.5")))
    environment.check_course_gates(["active_jump@0.25", "active_jump@0.5", "terrain"])
    with pytest.raises(ValueError, match="active_jump@0.5"):
        environment.check_course_gates(["active_jump@0.25", "active_jump@0.3"])


def test_a_report_mode_course_cannot_resume_in_hold_mode():
    reported = CourseGates(gates(window=1), STARTS, mode="report")
    reported.advance({"rms/height_error_m": .1}, 500)
    held = CourseGates(gates(window=1), STARTS)
    with pytest.raises(ValueError, match="hold mode"):
        held.load_state_dict(reported.state_dict())
    CourseGates(gates(window=1), STARTS, mode="report").load_state_dict(held.state_dict())
