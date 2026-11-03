"""CPU USD checks for the isolated knee-limit finalizer; no native simulator."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

try:
    import numpy as np
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics
    from scipy.spatial.transform import Rotation
except ImportError as error:
    raise unittest.SkipTest("knee finalization checks require the optional CPU USD environment") from error


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/set_knee_candidate_limits.py"
SPEC = importlib.util.spec_from_file_location("knee_candidate_limits_under_test", SCRIPT)
FINALIZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FINALIZER)
REFERENCE = ROOT / "assets/full_vehicle_v2p0p1_source_audited_v2"
AUDIT = ROOT / "outputs/mechanical-audit-20261101"


class KneeCandidateLimitsTest(unittest.TestCase):
    def setUp(self):
        if not (REFERENCE / "model_spec.json").is_file() or not (AUDIT / "limits/surface_contact_onsets.json").is_file():
            self.skipTest("requires the read-only audited mechanical fixture and CAD audit")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.asset = Path(self.temporary.name) / "new-builder-output"
        self.asset.mkdir()
        model = FINALIZER.read_json(REFERENCE / "model_spec.json")
        gimbal = FINALIZER.read_json(REFERENCE / "gimbal_model.json")
        self.mapping = FINALIZER.knee_limit_mapping(model)
        # Recreate the historical input range even if the real reference has
        # already been finalized. Only this disposable fixture is changed.
        for joint in model["joints"]:
            if joint["name"] in self.mapping:
                item = self.mapping[joint["name"]]
                native = sorted((np.deg2rad([40., 107.]) - item["theta0_rad"]) / item["dtheta_dq"])
                joint["limit"]["lower"], joint["limit"]["upper"] = map(str, native)
        self.old_model = copy.deepcopy(model)
        self.write_json("model_spec.json", model)
        self.write_json("gimbal_model.json", gimbal)
        root = ET.parse(REFERENCE / "robot.urdf").getroot()
        root.attrib.pop("version", None)
        for joint in model["joints"]:
            if joint["name"] in self.mapping:
                node = root.find(f"joint[@name='{joint['name']}']/limit")
                for key in ("lower", "upper"):
                    node.set(key, joint["limit"][key])
        self.old_xml = FINALIZER._normalized_xml(root)
        (self.asset / "robot.urdf").write_bytes(ET.tostring(root))
        for name in ("source_full_vehicle.urdf", "gas_spring_binding.json", "full_vehicle_source_provenance.json"):
            (self.asset / name).write_bytes((REFERENCE / name).read_bytes())
        self.write_json("mechanical_limits.json", {
            "knee_inner_angle_deg": [40., 107.], "training_inner_angle_deg": None,
            "source": "explicit_stop_only_candidate_not_measured_hardware_limit", "physics_qualified": False})
        self.write_json("full_vehicle_static_validation.json", {
            "core_joint_contract_unchanged": True, "core_nominal_joint_pos_unchanged": True,
            "core_constraint_anchors_unchanged": True, "urdf_bodies": 21, "urdf_tree_joints": 20})
        (self.asset / "FULL_VEHICLE_CANDIDATE.md").write_text("Historical fixture, not measured limits.\n")
        (self.asset / "tools").mkdir()
        (self.asset / "tools/build_full_vehicle_asset.py").write_text("# Historical builder fixture\n")
        stage = Usd.Stage.CreateNew(str(self.asset / "robot.usda"))
        stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Robot").GetPrim())
        for body in model["bodies"] + gimbal["bodies"]:
            prim = UsdGeom.Xform.Define(stage, "/Robot/" + body["name"]).GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim)
        for item in model["joints"] + gimbal["joints"]:
            joint_type = UsdPhysics.PrismaticJoint if item["type"] == "prismatic" else UsdPhysics.RevoluteJoint
            joint = joint_type.Define(stage, "/Robot/tree_joints/" + item["name"])
            joint.CreateBody0Rel().SetTargets(["/Robot/" + item["parent"]])
            joint.CreateBody1Rel().SetTargets(["/Robot/" + item["child"]])
            joint.CreateExcludeFromArticulationAttr(False)
            if item["name"] in self.mapping:
                joint.CreateLowerLimitAttr(float(np.rad2deg(float(item["limit"]["lower"]))))
                joint.CreateUpperLimitAttr(float(np.rad2deg(float(item["limit"]["upper"]))))
            elif item["name"] == "M_joint2":
                joint.CreateLowerLimitAttr(-30.)
                joint.CreateUpperLimitAttr(35.)
            elif item["type"] == "prismatic":
                joint.CreateLowerLimitAttr(float(item["limit"]["lower"]))
                joint.CreateUpperLimitAttr(float(item["limit"]["upper"]))
        for item in model["constraints"]:
            joint = UsdPhysics.SphericalJoint.Define(stage, "/Robot/loops/" + item["name"])
            joint.CreateBody0Rel().SetTargets(["/Robot/" + item["body0"]])
            joint.CreateBody1Rel().SetTargets(["/Robot/" + item["body1"]])
            joint.CreateExcludeFromArticulationAttr(True)
        stage.GetRootLayer().Save()
        self.old_usd = FINALIZER._normalized_usd(stage.GetRootLayer())
        self.write_json("manifest.json", {
            "asset_revision": FINALIZER.INPUT_REVISION, "gimbal_pitch_limits_deg": [-30., 35.],
            "knee_inner_limits_deg": [40., 107.],
            "source_urdf_sha256": FINALIZER.sha(self.asset / "source_full_vehicle.urdf"),
            "files_sha256": {str(path.relative_to(self.asset)): FINALIZER.sha(path)
                              for path in self.asset.rglob("*") if path.is_file()},
        })

    def write_json(self, relative, value):
        (self.asset / relative).write_bytes(FINALIZER.json_bytes(value))

    def hashes(self):
        return {str(path.relative_to(self.asset)): FINALIZER.sha(path)
                for path in self.asset.rglob("*") if path.is_file()}

    def test_signed_fk_mapping_is_not_inner_degrees_used_as_joint_degrees(self):
        self.assertEqual(self.mapping["L_joint2"]["dtheta_dq"], 1)
        self.assertEqual(self.mapping["R_jonit2"]["dtheta_dq"], -1)
        for name, item in self.mapping.items():
            self.assertGreater(abs(item["native_limits_deg"][0] - 25), 40)
            for endpoint in item["endpoint_checks"]:
                actual = FINALIZER.inner_angle(self.old_model["joints"], item["side"], {name: endpoint["native_q_rad"]})
                self.assertAlmostEqual(np.rad2deg(actual), endpoint["inner_angle_deg"], places=9)

    def test_real_usd_model_urdf_update_preserves_geometry_and_pins_evidence(self):
        gas_before = (self.asset / "gas_spring_binding.json").read_bytes()
        result = FINALIZER.apply_candidate_limits(self.asset, AUDIT)
        self.assertEqual(result["status"], "finalized")
        manifest = FINALIZER.read_json(self.asset / "manifest.json")
        FINALIZER.verify_pins(self.asset, manifest)
        self.assertEqual(manifest["asset_revision"], FINALIZER.FINAL_REVISION)
        self.assertEqual(manifest["knee_inner_limits_deg"], [25., 127.])
        self.assertEqual(manifest["historical_knee_inner_limits_deg"], [40., 107.])
        self.assertEqual(manifest["urdf_format_version"], "1.2")
        self.assertEqual((self.asset / "gas_spring_binding.json").read_bytes(), gas_before)
        model = FINALIZER.read_json(self.asset / "model_spec.json")
        restored = copy.deepcopy(model)
        for joint in restored["joints"]:
            if joint["name"] in FINALIZER.KNEES:
                joint["limit"] = next(item for item in self.old_model["joints"] if item["name"] == joint["name"])["limit"]
        self.assertEqual(restored, self.old_model)
        root = ET.parse(self.asset / "robot.urdf").getroot()
        self.assertEqual(root.get("version"), "1.2")
        self.assertEqual(FINALIZER._normalized_xml(root), self.old_xml)
        stage = Usd.Stage.Open(str(self.asset / "robot.usda"))
        self.assertEqual(FINALIZER._normalized_usd(stage.GetRootLayer()), self.old_usd)
        checks = FINALIZER._check_formats(model, root, stage, self.mapping)
        self.assertEqual(checks["usd_topology"]["tree_joints"], 20)
        self.assertEqual(len(checks["usd_topology"]["external_loops"]), 6)
        validation = FINALIZER.read_json(self.asset / "full_vehicle_static_validation.json")
        self.assertFalse(validation["core_joint_contract_unchanged"])
        receipt = FINALIZER.read_json(self.asset / FINALIZER.RECEIPT)
        self.assertFalse(receipt["measured_hardware_stop"])
        limits = FINALIZER.read_json(self.asset / "mechanical_limits.json")
        self.assertTrue(limits["lower_limit_changed"])
        self.assertTrue(limits["upper_limit_changed"])
        self.assertFalse(receipt["conditional_spring_check"]["hardware_travel_verified"])
        for row in receipt["conditional_spring_check"]["samples"]:
            self.assertGreater(row["margin_to_lower_m"], 0)
            self.assertGreater(row["margin_to_upper_m"], 0)
            self.assertLess(row["closure_max_m"], 1e-7)
        for relative in FINALIZER.EVIDENCE_FILES:
            self.assertEqual(FINALIZER.sha(self.asset / "source_audits" / relative), FINALIZER.sha(AUDIT / relative))
        self.assertEqual((self.asset / "tools/build_full_vehicle_asset.py").read_bytes(),
                         (ROOT / "scripts/build_full_vehicle_asset.py").read_bytes())

    def test_finalized_rerun_verifies_pins_and_changes_no_files(self):
        FINALIZER.apply_candidate_limits(self.asset, AUDIT)
        before = self.hashes()
        self.assertEqual(FINALIZER.apply_candidate_limits(self.asset, AUDIT)["status"], "already_finalized")
        self.assertEqual(self.hashes(), before)
        (self.asset / "model_spec.json").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "pin mismatch"):
            FINALIZER.apply_candidate_limits(self.asset, AUDIT)

    def test_tampered_input_is_rejected_before_any_writes(self):
        (self.asset / "gas_spring_binding.json").write_text("{}")
        before = self.hashes()
        with self.assertRaisesRegex(ValueError, "pin mismatch"):
            FINALIZER.apply_candidate_limits(self.asset, AUDIT)
        self.assertEqual(self.hashes(), before)


    def test_explicit_metadata_migration_changes_only_flags_script_and_manifest(self):
        FINALIZER.apply_candidate_limits(self.asset, AUDIT)
        limits = FINALIZER.read_json(self.asset / "mechanical_limits.json")
        limits["lower_limit_changed"] = False
        limits.pop("upper_limit_changed")
        self.write_json("mechanical_limits.json", limits)
        manifest = FINALIZER.read_json(self.asset / "manifest.json")
        manifest["files_sha256"]["mechanical_limits.json"] = FINALIZER.sha(self.asset / "mechanical_limits.json")
        self.write_json("manifest.json", manifest)
        before = self.hashes()
        result = FINALIZER.migrate_change_flags(self.asset)
        self.assertEqual(result["status"], "change_flags_migrated")
        after = self.hashes()
        changed = {name for name in before if before[name] != after[name]}
        self.assertEqual(changed, {"mechanical_limits.json", "manifest.json"})
        self.assertEqual(FINALIZER.migrate_change_flags(self.asset)["status"], "change_flags_already_current")
        self.assertEqual(self.hashes(), after)
        limits = FINALIZER.read_json(self.asset / "mechanical_limits.json")
        self.assertTrue(limits["lower_limit_changed"])
        self.assertTrue(limits["upper_limit_changed"])

    def test_original_stop107_candidate_revision_is_rejected(self):
        manifest = FINALIZER.read_json(self.asset / "manifest.json")
        manifest["asset_revision"] = "full_vehicle_v2p0p1_stop107_candidate_v1"
        self.write_json("manifest.json", manifest)
        before = self.hashes()
        with self.assertRaisesRegex(ValueError, "Only a new"):
            FINALIZER.apply_candidate_limits(self.asset, AUDIT)
        self.assertEqual(self.hashes(), before)

    def test_failed_commit_restores_original_pins_and_contents(self):
        before = self.hashes()
        actual_write = FINALIZER._atomic_write
        calls = 0
        def fail_once(path, content):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError("Injected commit failure")
            actual_write(path, content)
        with mock.patch.object(FINALIZER, "_atomic_write", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "Injected"):
                FINALIZER.apply_candidate_limits(self.asset, AUDIT)
        self.assertEqual(self.hashes(), before)
        FINALIZER.verify_pins(self.asset, FINALIZER.read_json(self.asset / "manifest.json"))


if __name__ == "__main__":
    unittest.main()
