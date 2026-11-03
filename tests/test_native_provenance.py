"""Native receipt evidence must not make equivalent cold starts incompatible."""

from contextlib import nullcontext
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from supermimic_rl import cli


def full_vehicle_provenance():
    return {
        "asset_manifest_sha256": "a" * 64,
        "runtime_sha256": "b" * 64,
        "upstream_commit": "c" * 40,
        "sdk_versions": {"isaaclab": "3.0.0"},
        "gimbal": {"pitch_position_rad": [-0.61, 0.52], "yaw_damping": 0.5},
        "full_vehicle": {
            "asset_manifest_sha256": "a" * 64,
            "core_body_names": ["base", "leg"],
            "core_joint_names": ["knee"],
            "auxiliary_body_names": ["yaw_body", "pitch_body"],
            "auxiliary_joint_names": ["yaw", "pitch"],
            "nominal_masses": {"total_mass_kg": 21.44},
            "requested_startup_configuration": {"nominal_fraction": 1.0},
            "gimbal_control": {
                "seed": 67867960,
                "configuration": {"pitch_position_rad": [-0.61, 0.52]},
                "control_joint_signs": [1, -1],
                "effort_limits_nm": [4.5, 4.5],
                "pitch_joint_limits_native_rad": [-0.52, 0.61],
                "pitch_limit_source": "user_pitch_down30_up35",
            },
            "after_core_startup": {
                "native_parameters_sha256": "d" * 64,
                "total_mass_kg_min": 21.440000534057617,
                "properties": {"com_pose_local_xyzw": {
                    "row0": [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]],
                    "sha256": "e" * 64,
                }},
            },
            "core_startup": {"mass_kg_min": 17.000001},
            "auxiliary_property_comparison": {"max_absolute_difference": 0.0},
        },
    }


def test_legacy_native_schema_source_is_byte_identical():
    provenance = {"z": [1, 0.02], "pinned": "same-source", "nested": {"b": 2, "a": 1}}
    assert cli._native_schema_source(provenance) == json.dumps(
        provenance, sort_keys=True, separators=(",", ":"))


def test_known_invalid_row_rejection_preserves_valid_physical_schema():
    original = full_vehicle_provenance()
    current = deepcopy(original)
    current.update(course_schedule=None, numerical_isolation={
        "contract": "physx_finite_coordinate_sampling_row_isolation_v2",
        "native_state_modified": False,
    })
    assert cli._native_schema_source(current) == cli._native_schema_source(original)
    assert "numerical_isolation" in current and "course_schedule" in current
    altered = deepcopy(current)
    altered["numerical_isolation"]["native_state_modified"] = True
    assert cli._native_schema_source(altered) != cli._native_schema_source(original)
    altered = deepcopy(current)
    altered["numerical_isolation"]["contract"] = "different_recovery_policy"
    assert cli._native_schema_source(altered) != cli._native_schema_source(original)
    altered = deepcopy(current)
    altered["course_schedule"] = {"schedule": "jump_first_v1", "request_heading_override": False}
    assert cli._native_schema_source(altered) != cli._native_schema_source(original)


def test_encoder_roundoff_receipt_keeps_coordinate_identity():
    original = {"continuous_coordinates": {"native_period_rad": 6.283185307179586,
                                            "rotation_correction": "twopi"}}
    changed = deepcopy(original)
    changed["continuous_coordinates"]["half_turn_roundoff_rad"] = 7.49e-7
    assert cli._native_schema_source(original) == cli._native_schema_source(changed)
    changed["continuous_coordinates"]["native_period_rad"] *= 2
    assert cli._native_schema_source(original) != cli._native_schema_source(changed)


def test_full_vehicle_schema_ignores_equivalent_sdk_readback_without_losing_evidence():
    original = full_vehicle_provenance()
    saved = deepcopy(original)
    changed = deepcopy(original)
    receipt = changed["full_vehicle"]
    receipt["after_core_startup"]["total_mass_kg_min"] += 0.000001
    receipt["after_core_startup"]["native_parameters_sha256"] = "f" * 64
    com = receipt["after_core_startup"]["properties"]["com_pose_local_xyzw"]
    com["row0"][0][3:] = [-value for value in com["row0"][0][3:]]
    com["sha256"] = "0" * 64
    receipt["core_startup"]["mass_kg_min"] += 0.000001
    receipt["auxiliary_property_comparison"]["max_absolute_difference"] = 0.0000001
    assert cli._native_schema_source(changed) == cli._native_schema_source(original)
    assert original == saved
    assert (changed["full_vehicle"]["after_core_startup"]["properties"]
            != saved["full_vehicle"]["after_core_startup"]["properties"])


def test_execution_schema_binds_model_but_not_sampled_inertia_extrema():
    original = {"execution": {"wheel_model_sha256": "a" * 64,
                              "wheel_inertia_scale_range": [0.9, 1.1]}}
    changed = deepcopy(original)
    changed["execution"]["wheel_inertia_scale_range"] = [0.89, 1.12]
    assert cli._native_schema_source(original) == cli._native_schema_source(changed)
    assert original["execution"]["wheel_inertia_scale_range"] == [0.9, 1.1]
    changed["execution"]["wheel_model_sha256"] = "b" * 64
    assert cli._native_schema_source(original) != cli._native_schema_source(changed)


@pytest.mark.parametrize("path,value", [
    (("asset_manifest_sha256",), "1" * 64),
    (("runtime_sha256",), "2" * 64),
    (("upstream_commit",), "3" * 40),
    (("sdk_versions", "isaaclab"), "3.1.0"),
    (("gimbal", "yaw_damping"), 0.6),
    (("full_vehicle", "gimbal_control", "configuration", "pitch_position_rad"), [-0.5, 0.5]),
    (("full_vehicle", "gimbal_control", "control_joint_signs"), [1, 1]),
    (("full_vehicle", "gimbal_control", "effort_limits_nm"), [5.0, 4.5]),
    (("full_vehicle", "gimbal_control", "pitch_joint_limits_native_rad"), [-0.6, 0.61]),
    (("full_vehicle", "requested_startup_configuration", "nominal_fraction"), 0.5),
])
def test_full_vehicle_schema_preserves_source_and_control_identity(path, value):
    original = full_vehicle_provenance()
    changed = deepcopy(original)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert cli._native_schema_source(changed) != cli._native_schema_source(original)


def test_incomplete_full_vehicle_identity_is_rejected():
    provenance = full_vehicle_provenance()
    del provenance["full_vehicle"]["gimbal_control"]
    with pytest.raises(ValueError, match="authenticated source/control"):
        cli._native_schema_source(provenance)


def test_kamino_schema_binds_solver_and_frames_but_not_native_readback():
    provenance = full_vehicle_provenance()
    provenance.update(physics_backend="kamino", backend={
        "identity": {"padmm_iterations": 200, "inertia_frame": "link_at_com"},
        "readback": {"mass": 21.441739223},
    })
    original = cli._native_schema_source(provenance)
    provenance["backend"]["readback"]["mass"] += 0.000001
    assert cli._native_schema_source(provenance) == original
    provenance["backend"]["identity"]["padmm_iterations"] = 400
    assert cli._native_schema_source(provenance) != original
    del provenance["backend"]["identity"]
    with pytest.raises(ValueError, match="stable backend identity"):
        cli._native_schema_source(provenance)


@pytest.mark.parametrize("command", ["train", "evaluate"])
def test_complete_receipt_is_saved_before_training_or_evaluation_construction(
    tmp_path, monkeypatch, command,
):
    provenance = full_vehicle_provenance()
    output = tmp_path / command
    factory_path = tmp_path / "factory.json"
    factory_path.write_text("{}")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"run": {}, "learning": {}}))
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not-loaded-in-fixture")
    closed = []
    adapter = SimpleNamespace(provenance=provenance, policy_dt_s=0.02,
                              close=lambda: closed.append(True))

    def fail_at_construction(*args, **kwargs):
        assert json.loads((output / "native-provenance.json").read_text()) == provenance
        assert args[0].source == cli._native_schema_source(provenance)
        raise RuntimeError("fixture constructor failure")

    monkeypatch.setattr(cli.importlib, "import_module",
                        lambda name: SimpleNamespace(create=lambda config: adapter))
    monkeypatch.setattr(cli, "native_logs", lambda directory: nullcontext())
    if command == "train":
        # A callable wrapper preserves the real initialization preflight.
        fail_at_construction.validate_initialization = cli.TrainingRun.validate_initialization
        monkeypatch.setattr(cli, "TrainingRun", fail_at_construction)
        extra = ["--config", str(config_path)]
    else:
        monkeypatch.setattr(cli, "evaluate", fail_at_construction)
        extra = ["--checkpoint", str(checkpoint), "--steps", "2"]
    assert cli.main([
        command, "--factory", "fixture:create", "--factory-config", str(factory_path),
        "--output", str(output), *extra,
    ]) == 2
    assert closed == [True]
    assert json.loads((output / "failure.json").read_text())["message"] == "fixture constructor failure"


def test_physx_course_without_coordinate_receipt_keeps_its_identity():
    provenance = {"physics_backend": "physx", "continuous_coordinates": None,
                  "backend": {"identity": {"name": "physx", "position_iterations": 12}}}
    source = cli._native_schema_source(provenance)
    assert json.loads(source)["continuous_coordinates"] is None
