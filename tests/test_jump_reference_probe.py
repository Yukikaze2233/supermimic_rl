"""Probe profiles preserve the hardware contract and distinguish missing evidence."""
import importlib.util
import json
from pathlib import Path

from transformer_rl.formal.tasks.jump import JumpReferenceConfig


spec = importlib.util.spec_from_file_location(
    "probe_jump_reference", Path(__file__).parents[1] / "scripts/probe_jump_reference.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_profiles_change_reference_only_and_keep_gated_assist(tmp_path):
    course_path = tmp_path / "course.json"
    course = {"schedule": "jump_first_v1", "base": {
        "stage_transition_budgets": list(range(10, 18)), "domain": {"jump_m": [.03, .4]}}}
    module.write_json(course_path, course)
    factory = {"course_json": str(course_path), "course_sha256": module.digest(course_path),
               "asset_manifest_sha256": "08bf7e00", "runtime_sha256": "pinned-runtime",
               "identified_bundle": "identified/20261031", "num_envs": 4096,
               "reward_extensions": {"jump_flight": 4},
               "jump_takeoff": {"cooldown_s": 1.3},
               "jump_assist": {"probability": 1., "decay": "force",
                               "assist_request_probability": .8, "fade_stage_fraction": .75}}
    factory_path, config_path = tmp_path / "factory.json", tmp_path / "ppo.json"
    module.write_json(factory_path, factory)
    gates = [{"stage": "active_jump", "at_fraction": .25, "criteria": [{"min_count": 200}]}]
    module.write_json(config_path, {"steps_per_update": 24, "course_gates": gates,
                                    "stage_action_std": {"active_jump": [.3] * 4 + [.05] * 2}})
    configs = []
    for profile in ("short", "long"):
        output = tmp_path / profile
        output.mkdir()
        derived, config = module.prepare_probe(factory_path, config_path, output, profile, 400)
        assert derived["num_envs"] == 1024
        for name in ("asset_manifest_sha256", "runtime_sha256", "identified_bundle", "jump_assist"):
            assert derived[name] == factory[name]
        assert derived["jump_takeoff"]["cooldown_s"] == 1.3
        assert derived["reward_extensions"] == {"jump_flight": 4, "jump_missed": 1.}
        assert config["course_gates"] == gates
        assert config["stage_action_std"]["active_jump"] == [.3] * 4 + [.05] * 2
        changed_course = json.loads(Path(derived["course_json"]).read_text())
        budgets = changed_course["base"]["stage_transition_budgets"]
        assert budgets[5] == 400 * 1024 * 24
        assert budgets[:5] + budgets[6:] == list(range(10, 15)) + [16, 17]
        assert changed_course["base"]["domain"] == course["base"]["domain"]
        assert module.digest(Path(derived["course_json"])) == derived["course_sha256"]
        configs.append(derived)
    native = JumpReferenceConfig()
    for field, value in module.PROFILES["short"].items():
        assert value == getattr(native, field)
    assert configs[0]["jump_takeoff"] != configs[1]["jump_takeoff"]
    assert json.loads(factory_path.read_text()) == factory


def test_summary_counts_actual_unassisted_outcomes_and_leaves_unsampled_unknown(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    rows = [
        {"course/jump_assist/ended_rise_0.00-0.10": 9,
         "course/jump_assist/success_rise_0.00-0.10": 5,
         "course/jump_assist/unassisted_ended_rise_0.00-0.10": 2,
         "course/jump_assist/unassisted_success_rise_0.00-0.10": 0},
        {"course/jump_assist/ended_rise_0.00-0.10": 11,
         "course/jump_assist/success_rise_0.00-0.10": 4,
         "course/jump_assist/unassisted_ended_rise_0.00-0.10": 3,
         "course/jump_assist/unassisted_success_rise_0.00-0.10": 1,
         "course/jump_assist/level": .8, "course": {"waiting_for": "active_jump@0.25"}},
    ]
    metrics.write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = module.summarize_probe(metrics)
    low = summary["height_bins_m"]["0.00-0.10"]
    assert low["all"] == {"ended": 20, "physical_success": 9, "success_rate": .45}
    assert low["unassisted"] == {"ended": 5, "physical_success": 1, "success_rate": .2}
    assert summary["height_bins_m"]["0.30-0.40"]["unassisted"]["success_rate"] is None
    assert summary["final_assist_level"] == .8
    assert summary["simulated_seconds_per_environment"] == .96
