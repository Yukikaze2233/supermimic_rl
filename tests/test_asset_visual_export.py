"""CPU USD export regression checks; skipped when the optional asset SDK is absent."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

try:
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics
    import trimesh
except ImportError as error:
    raise unittest.SkipTest("asset export checks require the optional USD/mesh environment") from error


BUILDER_PATH = Path(__file__).resolve().parents[1] / "scripts/build_full_vehicle_asset.py"
SPEC = importlib.util.spec_from_file_location("full_vehicle_asset_builder_under_test", BUILDER_PATH)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class AssetVisualExportTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.old, self.out = root / "old", root / "out"
        self.old.mkdir()
        (self.out / "meshes").mkdir(parents=True)
        stage = Usd.Stage.CreateNew(str(self.old / "robot.usda"))
        stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Robot").GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, "/Robot/base_link").GetPrim())
        stage.GetRootLayer().Save()
        (self.old / "robot.urdf").write_text('<robot name="test"><link name="base_link"/></robot>')
        self.bodies = []
        tetrahedron = trimesh.Trimesh(
            vertices=[[0, 0, 0], [.02, 0, 0], [0, .03, 0], [0, 0, .04]],
            faces=[[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], process=False)
        tetrahedron.export(self.out / "collision.obj")
        for index, name in enumerate(("base_link", "M_link1", "M_link2")):
            mesh_origin = np.eye(4)
            mesh_origin[:3, 3] = [.01 * index, .02, -.03]
            tetrahedron.export(self.out / "meshes" / (name + ".stl"))
            self.bodies.append({"name": name, "mass": 1., "com": [0., 0., 0.],
                                "inertia": np.diag([.01, .02, .025]).tolist(),
                                "color": [.3, .4, .5], "mesh": f"meshes/{name}.stl",
                                "mesh_origin": mesh_origin.tolist(),
                                "collisions": [{"type": "mesh", "file": "collision.obj", "origin": np.eye(4).tolist()}]})
        self.gimbal = {"joint_names": ["M_joint1", "M_joint2"], "body_names": ["M_link1", "M_link2"],
                       "view_body": "M_link2", "view_forward_local": [1., 0., 0.],
                       "control_joint_signs": [1, -1], "bodies": self.bodies[1:], "joints": []}
        for name, parent, child, axis in (
            ("M_joint1", "base_link", "M_link1", [0., 0., 1.]),
            ("M_joint2", "M_link1", "M_link2", [0., -1., 0.]),
        ):
            origin = np.eye(4)
            origin[:3, 3] = [0., 0., .1]
            self.gimbal["joints"].append({"name": name, "parent": parent, "child": child,
                                          "type": "continuous", "axis": axis, "origin": origin.tolist(),
                                          "effort_limit_nm": 4.5, "velocity_limit_rad_s": None})
        self.gimbal = BUILDER.configure_gimbal_pitch(self.gimbal)

    def test_real_export_keeps_mesh_reference_geometry_and_pose(self):
        BUILDER.write_usd(self.old, self.out, self.bodies, self.gimbal)
        stage = Usd.Stage.Open(str(self.out / "robot.usda"))
        result = BUILDER.validate_renderable_visuals(stage, [body["name"] for body in self.bodies])
        self.assertEqual(len(result), 3)
        for body in self.bodies:
            visual = stage.GetPrimAtPath(f"/Robot/{body['name']}/Visual")
            self.assertTrue(visual.IsA(UsdGeom.Mesh))
            self.assertTrue(visual.GetMetadata("references").GetAddedOrExplicitItems())
            self.assertGreater(result[body["name"]]["points"], 0)
            self.assertGreater(result[body["name"]]["faces"], 0)
            actual = np.asarray(UsdGeom.Xformable(visual).GetLocalTransformation()).T
            np.testing.assert_allclose(actual, body["mesh_origin"], atol=1e-12)

    def test_validation_rejects_the_original_xform_override_bug(self):
        BUILDER.write_usd(self.old, self.out, self.bodies, self.gimbal)
        stage = Usd.Stage.Open(str(self.out / "robot.usda"))
        stage.SetEditTarget(stage.GetSessionLayer())
        visual = stage.GetPrimAtPath("/Robot/base_link/Visual")
        visual.SetTypeName("Xform")
        self.assertGreater(len(visual.GetAttribute("points").Get()), 0)
        with self.assertRaisesRegex(ValueError, "renderable visual Mesh"):
            BUILDER.validate_renderable_visuals(stage, ["base_link"])

    def test_validation_rejects_an_empty_renderable_mesh(self):
        BUILDER.write_usd(self.old, self.out, self.bodies, self.gimbal)
        stage = Usd.Stage.Open(str(self.out / "robot.usda"))
        stage.SetEditTarget(stage.GetSessionLayer())
        UsdGeom.Mesh(stage.GetPrimAtPath("/Robot/base_link/Visual")).GetPointsAttr().Set([])
        with self.assertRaisesRegex(ValueError, "empty or inconsistent"):
            BUILDER.validate_renderable_visuals(stage, ["base_link"])

    def test_pitch_is_revolute_with_matching_native_bounds_in_usd_and_urdf(self):
        BUILDER.write_urdf(self.old, self.out, self.bodies, self.gimbal)
        BUILDER.write_usd(self.old, self.out, self.bodies, self.gimbal)
        root = ET.parse(self.out / "robot.urdf").getroot()
        self.assertEqual(root.get("version"), "1.2")
        pitch = root.find("joint[@name='M_joint2']")
        self.assertEqual(pitch.get("type"), "revolute")
        np.testing.assert_allclose([float(pitch.find("limit").get(key)) for key in ("lower", "upper")],
                                   np.deg2rad([-30., 35.]), atol=1e-12)
        self.assertIsNone(pitch.find("limit").get("velocity"))
        stage = Usd.Stage.Open(str(self.out / "robot.usda"))
        joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/Robot/tree_joints/M_joint2"))
        self.assertEqual(joint.GetLowerLimitAttr().Get(), -30.)
        self.assertEqual(joint.GetUpperLimitAttr().Get(), 35.)
        self.assertFalse(self.gimbal["pitch_position_contract"]["measured_stop"])
        self.assertEqual(self.gimbal["control_joint_signs"], [1, -1])

    def test_optical_fk_determines_pitch_direction_instead_of_control_sign(self):
        result = BUILDER.pitch_elevation_check(self.gimbal)
        self.assertAlmostEqual(result["samples"]["35.0"]["elevation_deg"], 35.)
        reversed_axis = copy.deepcopy(self.gimbal)
        reversed_axis["joints"][1]["axis"] = [0., 1., 0.]
        with self.assertRaisesRegex(ValueError, "sign no longer"):
            BUILDER.configure_gimbal_pitch(reversed_axis)


if __name__ == "__main__":
    unittest.main()
