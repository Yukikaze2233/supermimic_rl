#!/usr/bin/env python3
"""Finalize a new full-vehicle bundle with CAD-candidate 25--127 degree knees.

This changes neither measured hardware claims nor the historical gas-spring
model. Input pins, named FK geometry, all three limit formats and unchanged
joint geometry are checked before committing the manifest last. Requires the
optional CPU numpy/scipy/usd-core environment, never Isaac or Kit.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


INPUT_REVISION = "full_vehicle_v2p0p1_source_audited_v2"
FINAL_REVISION = INPUT_REVISION + "_cad25_127_pitch_m30_p35"
INNER_LIMITS_DEG = (25., 127.)
LIMIT_SOURCE = "CAD_surface_intersection_candidate_not_measured"
RECEIPT = "knee_candidate_limits_validation.json"
EVIDENCE_FILES = (
    "kinematics/kinematic_audit.json", "kinematics/knee_domain_scan.json",
    "kinematics/knee_domain_scan.csv", "inertials/inertial_audit.json",
    "limits/surface_contact_onsets.json",
)
KNEES = {"L_joint2": "L", "R_jonit2": "R"}
URDF_METADATA = {
    "urdf_format_version": "1.2",
    "urdf_parser_requirement": "URDF 1.2 compatible parser",
    "urdf_omitted_velocity_semantics": "format default infinity; hardware speed unknown, not measured unlimited",
    "native_training_asset_format": "USD",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def checked_relative(root: Path, relative: str) -> Path:
    path = Path(relative)
    result = root / path
    if path.is_absolute() or ".." in path.parts or not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Bundle pin escapes the asset directory: {relative}")
    return result


def verify_pins(asset: Path, manifest: dict) -> None:
    pins = manifest.get("files_sha256")
    if not isinstance(pins, dict) or not pins or "manifest.json" in pins:
        raise ValueError("A nonempty manifest pin set excluding manifest.json is required")
    for relative, expected in pins.items():
        path = checked_relative(asset, relative)
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Existing asset pin mismatch: {relative}")
    actual = {str(path.relative_to(asset)) for path in asset.rglob("*")
              if path.is_file() and path != asset / "manifest.json" and "__pycache__" not in path.parts}
    if actual != set(pins):
        raise ValueError(f"Unpinned or missing asset files: {sorted(actual.symmetric_difference(pins))}")


def fk(joints: list[dict], q: dict | None = None) -> dict[str, np.ndarray]:
    q, poses, remaining = q or {}, {"base_link": np.eye(4)}, list(joints)
    while remaining:
        ready = [joint for joint in remaining if joint["parent"] in poses]
        if not ready:
            raise ValueError("Disconnected or cyclic model_spec joint tree")
        for joint in ready:
            origin, axis = np.asarray(joint["origin"], float), np.asarray(joint["axis"], float)
            if origin.shape != (4, 4) or axis.shape != (3,) or abs(np.linalg.norm(axis) - 1) > 1e-6:
                raise ValueError(f"Invalid model joint transform/axis: {joint['name']}")
            motion = np.eye(4)
            value = q.get(joint["name"], 0.)
            if joint["type"] == "prismatic":
                motion[:3, 3] = axis * value
            elif joint["type"] != "fixed":
                motion[:3, :3] = Rotation.from_rotvec(axis * value).as_matrix()
            poses[joint["child"]] = poses[joint["parent"]] @ origin @ motion
            remaining.remove(joint)
    return poses


def inner_angle(joints: list[dict], side: str, q: dict | None = None) -> float:
    poses = fk(joints, q)
    knee_name = "L_joint2" if side == "L" else "R_jonit2"
    knee = next(joint for joint in joints if joint["name"] == knee_name)
    a, b, c = (poses[side + suffix][:3, 3] for suffix in ("_link1", "_link2", "_link3"))
    axis = poses[side + "_link2"][:3, :3] @ knee["axis"]
    axis /= np.linalg.norm(axis)
    u, v = a - b, c - b
    u, v = u - axis * (u @ axis), v - axis * (v @ axis)
    if min(np.linalg.norm(u), np.linalg.norm(v)) < 1e-6:
        raise ValueError("Degenerate knee pin geometry")
    return math.atan2(float(np.linalg.norm(np.cross(u, v))), float(u @ v))


def knee_limit_mapping(model: dict, limits_deg: tuple[float, float] = INNER_LIMITS_DEG) -> dict:
    """Derive zero and sign from each actual native knee's projected FK."""
    result = {}
    joints = model["joints"]
    for name, side in KNEES.items():
        theta0 = inner_angle(joints, side)
        epsilon = 1e-5
        derivative = (inner_angle(joints, side, {name: epsilon})
                      - inner_angle(joints, side, {name: -epsilon})) / (2 * epsilon)
        sign = 1 if derivative > 0 else -1
        if abs(derivative - sign) > 1e-6:
            raise ValueError(f"Knee angle is not a regular signed native coordinate: {name}")
        theta_bounds = np.deg2rad(limits_deg)
        native = (theta_bounds - theta0) / sign
        samples = []
        for theta, value in zip(limits_deg, native):
            actual = math.degrees(inner_angle(joints, side, {name: float(value)}))
            if abs(actual - theta) > 1e-7:
                raise ValueError(f"FK knee endpoint differs from requested inner angle: {name}")
            samples.append({"inner_angle_deg": theta, "native_q_rad": float(value),
                            "fk_inner_angle_deg": actual})
        result[name] = {"side": side, "theta0_rad": theta0, "theta0_deg": math.degrees(theta0),
                        "dtheta_dq": sign, "native_limits_rad": sorted(native.tolist()),
                        "native_limits_deg": sorted(np.rad2deg(native).tolist()), "endpoint_checks": samples}
    return result


def closure(model: dict, q: dict) -> np.ndarray:
    poses = fk(model["joints"], q)
    residuals = []
    for item in model["constraints"]:
        values = []
        for side in (0, 1):
            pose = poses[item[f"body{side}"]]
            values.append(pose[:3, :3] @ item[f"local_pos{side}_m"] + pose[:3, 3])
        residuals.append(values[0] - values[1])
    return np.asarray(residuals)


def conditional_spring_check(model: dict, mapping: dict, binding: dict,
                             limits_deg: tuple[float, float] = INNER_LIMITS_DEG) -> dict:
    fixed = {"L_joint1", "R_joint1", "L_joint2", "R_jonit2", "L_joint3", "R_joint3"}
    passive = [joint["name"] for joint in model["joints"] if joint["name"] not in fixed]
    if len(passive) != 12 or len(model["constraints"]) != 6:
        raise ValueError("Expected the 18-coordinate, six-loop core")
    rows = []
    for theta in limits_deg:
        target = dict(model["nominal_joint_pos"])
        for name, item in mapping.items():
            target[name] = (math.radians(theta) - item["theta0_rad"]) / item["dtheta_dq"]
        def residual(values):
            return closure(model, {**target, **dict(zip(passive, values))}).ravel()
        solved = least_squares(residual, [target[name] for name in passive],
                               xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=1000)
        target.update(zip(passive, solved.x))
        error = float(np.linalg.norm(solved.fun.reshape(-1, 3), axis=-1).max())
        if error > 1e-7:
            raise ValueError(f"Knee endpoint cannot close the unchanged six loops: {theta} degrees")
        poses = fk(model["joints"], target)
        for side in ("L", "R"):
            spring = binding[side + "_spring_slide"]
            upper = float(spring["full_extension_pin_distance_m"])
            lower = upper - float(spring["stroke_m"])
            # Origins of the reparented barrel and rod are the installed pin centers.
            distance = float(np.linalg.norm(poses[side * 3 + "_link1"][:3, 3]
                                            - poses[side * 3 + "_link2"][:3, 3]))
            if not lower - 1e-8 <= distance <= upper + 1e-8:
                raise ValueError(f"Candidate endpoint exceeds inherited spring travel: {side}/{theta}")
            rows.append({"side": side, "inner_angle_deg": theta, "pin_distance_m": distance,
                         "historical_interval_m": [lower, upper],
                         "margin_to_lower_m": distance - lower, "margin_to_upper_m": upper - distance,
                         "closure_max_m": error, "within_historical_interval": True})
    return {"status": "conditional_on_unchanged_historical_232mm_extension_80mm_stroke_not_new_measurement",
            "hardware_travel_verified": False, "samples": rows}


def _normalized_xml(root: ET.Element) -> bytes:
    root = copy.deepcopy(root)
    root.attrib.pop("version", None)
    for name in KNEES:
        node = root.find(f"joint[@name='{name}']/limit")
        if node is None:
            raise ValueError(f"URDF knee limit is missing: {name}")
        for key in ("lower", "upper"):
            node.set(key, "LIMIT")
    return ET.tostring(root)


def _normalized_usd(layer) -> str:
    from pxr import Sdf

    clone = Sdf.Layer.CreateAnonymous("knee-limit-comparison.usda")
    clone.ImportFromString(layer.ExportToString())
    for name in KNEES:
        for key in ("lowerLimit", "upperLimit"):
            attribute = clone.GetAttributeAtPath(f"/Robot/tree_joints/{name}.physics:{key}")
            if not attribute:
                raise ValueError(f"USD knee limit attribute is missing: {name}/{key}")
            attribute.default = 0.
    return clone.ExportToString()


def _usd_counts(stage) -> dict:
    from pxr import UsdPhysics

    bodies = [prim.GetName() for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    joints = [UsdPhysics.Joint(prim) for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)]
    loops = [joint.GetPrim().GetName() for joint in joints if joint.GetExcludeFromArticulationAttr().Get()]
    result = {"bodies": len(bodies), "tree_joints": len(joints) - len(loops), "external_loops": sorted(loops)}
    if (len(bodies), len(joints) - len(loops), len(loops)) != (21, 20, 6):
        raise ValueError("USD topology must remain 21 bodies / 20 tree joints / 6 loops")
    return result


def _check_formats(model: dict, xml_root: ET.Element, stage, mapping: dict) -> dict:
    from pxr import UsdPhysics

    if len(model["joints"]) != 18 or len(model["bodies"]) != 19 or len(model["constraints"]) != 6:
        raise ValueError("The core model must remain 19 bodies / 18 tree joints / 6 loops")
    if (len(xml_root.findall("link")), len(xml_root.findall("joint"))) != (21, 20):
        raise ValueError("URDF topology must remain 21 bodies / 20 tree joints")
    checked = {}
    for name, item in mapping.items():
        model_joint = next(joint for joint in model["joints"] if joint["name"] == name)
        xml_joint = xml_root.find(f"joint[@name='{name}']")
        usd_joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(f"/Robot/tree_joints/{name}"))
        if model_joint["type"] != "revolute" or xml_joint.get("type") != "revolute" or not usd_joint:
            raise ValueError(f"Expected a passive revolute knee in all representations: {name}")
        model_bounds = [float(model_joint["limit"][key]) for key in ("lower", "upper")]
        xml_bounds = [float(xml_joint.find("limit").get(key)) for key in ("lower", "upper")]
        usd_bounds = [usd_joint.GetLowerLimitAttr().Get(), usd_joint.GetUpperLimitAttr().Get()]
        if (not np.allclose(model_bounds, item["native_limits_rad"], atol=1e-12, rtol=0)
                or not np.allclose(xml_bounds, model_bounds, atol=1e-12, rtol=0)
                or not np.allclose(usd_bounds, item["native_limits_deg"], atol=1e-5, rtol=0)):
            raise ValueError(f"URDF/USD/model knee bounds disagree: {name}")
        checked[name] = {"model_rad": model_bounds, "urdf_rad": xml_bounds, "usd_degrees": usd_bounds}
    pitch = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/Robot/tree_joints/M_joint2"))
    if not pitch or not np.allclose([pitch.GetLowerLimitAttr().Get(), pitch.GetUpperLimitAttr().Get()], [-30, 35], atol=1e-6):
        raise ValueError("This finalizer requires the new user-specified pitch -30/+35 bundle")
    return {"matching_knee_limits": checked, "usd_topology": _usd_counts(stage),
            "urdf_bodies": 21, "urdf_tree_joints": 20}


def _evidence(asset: Path, audit: Path, manifest: dict) -> tuple[dict[str, bytes], dict]:
    source_hash = sha(asset / "source_full_vehicle.urdf")
    if source_hash != manifest["source_urdf_sha256"]:
        raise ValueError("Pinned source URDF does not match the manifest")
    inertial = read_json(audit / "inertials/inertial_audit.json")
    kinematic = read_json(audit / "kinematics/kinematic_audit.json")
    if (inertial.get("source_urdf_sha256") != source_hash
            or source_hash not in kinematic.get("inputs_sha256", {}).values()):
        raise ValueError("Source audits do not belong to this asset source URDF")
    onsets = read_json(audit / "limits/surface_contact_onsets.json")
    if not onsets.get("onsets"):
        raise ValueError("CAD surface-onset evidence is missing")
    poses = audit / "kinematics/knee_domain_scan.json"
    if onsets.get("poses_sha256") != sha(poses):
        raise ValueError("CAD contact onsets were computed from a different pose audit")
    onset_summary = {"format": onsets.get("format"), "meaning": onsets.get("meaning"),
                     "primary_hip_fixed_q_rad": onsets.get("primary_hip_fixed_q_rad"),
                     "onsets": [{key: row[key] for key in ("pair", "final_angle_bracket_deg", "intersection_on_lower_side")}
                                for row in onsets["onsets"]]}
    payload = {"source_audits/" + name: (audit / name).read_bytes() for name in EVIDENCE_FILES}
    payload["source_audits/limits/onset_summary.json"] = json_bytes(onset_summary)
    provenance = read_json(asset / "full_vehicle_source_provenance.json")
    source_record = {"source_urdf_sha256": source_hash,
                     "source_files_sha256": provenance["source_files_sha256"],
                     "evidence_original_sha256": {name: sha(audit / name) for name in EVIDENCE_FILES},
                     "meaning": "Copied source audits; CAD candidates are not measured hardware stops"}
    payload["source_audits/source_hashes.json"] = json_bytes(source_record)
    return payload, onset_summary


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".knee-limit-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _builder_declares_urdf12(content: bytes) -> bool:
    tree = ast.parse(content)
    constants = {target.id: node.value.value for node in tree.body if isinstance(node, ast.Assign)
                 and isinstance(node.value, ast.Constant) for target in node.targets if isinstance(target, ast.Name)}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "set"
                and len(node.args) == 2 and isinstance(node.args[0], ast.Constant) and node.args[0].value == "version"):
            value = node.args[1]
            if ((isinstance(value, ast.Constant) and value.value == "1.2")
                    or (isinstance(value, ast.Name) and constants.get(value.id) == "1.2")):
                return True
    return False


def apply_candidate_limits(asset_dir: Path, audit_dir: Path) -> dict:
    """Apply once to the new builder output; verified finalized reruns are no-ops."""
    from pxr import Sdf, Usd, UsdPhysics

    asset, audit = asset_dir.resolve(), audit_dir.resolve()
    manifest = read_json(asset / "manifest.json")
    verify_pins(asset, manifest)
    original_manifest_bytes = (asset / "manifest.json").read_bytes()
    if manifest.get("asset_revision") == FINAL_REVISION:
        receipt = read_json(asset / RECEIPT)
        if (manifest.get("knee_inner_limits_deg") != list(INNER_LIMITS_DEG)
                or receipt.get("limit_source") != LIMIT_SOURCE):
            raise ValueError("Finalized bundle metadata is inconsistent")
        return {"status": "already_finalized", "asset_revision": FINAL_REVISION,
                "manifest_sha256": sha(asset / "manifest.json")}
    if manifest.get("asset_revision") != INPUT_REVISION or (asset / RECEIPT).exists():
        raise ValueError("Only a new source_audited_v2 builder output may be finalized; old/partial assets are rejected")
    if manifest.get("gimbal_pitch_limits_deg") != [-30., 35.]:
        raise ValueError("New bundle must already include native pitch -30/+35 degrees")
    provenance = read_json(asset / "full_vehicle_source_provenance.json")
    source_root = Path(provenance["source_urdf"]).parent.parent.resolve()
    parent = Path(provenance["parent_asset"]).resolve()
    if asset.is_relative_to(source_root) or asset.is_relative_to(parent):
        raise ValueError("Cannot change the original source or historical parent asset")
    payload, onset_summary = _evidence(asset, audit, manifest)
    model = read_json(asset / "model_spec.json")
    before_model = copy.deepcopy(model)
    mapping = knee_limit_mapping(model)
    binding = read_json(asset / "gas_spring_binding.json")
    spring_check = conditional_spring_check(model, mapping, binding)
    nominal_closure = float(np.linalg.norm(closure(model, model["nominal_joint_pos"]), axis=-1).max())
    if nominal_closure > 1e-7:
        raise ValueError("Existing nominal six-loop closure is invalid")
    xml_root = ET.parse(asset / "robot.urdf").getroot()
    before_xml = _normalized_xml(xml_root)
    xml_root.set("version", "1.2")
    old_bounds = {}
    for name, item in mapping.items():
        joint = next(joint for joint in model["joints"] if joint["name"] == name)
        old_bounds[name] = copy.deepcopy(joint["limit"])
        node = xml_root.find(f"joint[@name='{name}']/limit")
        for key, value in zip(("lower", "upper"), item["native_limits_rad"]):
            joint["limit"][key] = format(value, ".17g") if isinstance(joint["limit"][key], str) else value
            node.set(key, format(value, ".17g"))
    restored_model = copy.deepcopy(model)
    for joint in restored_model["joints"]:
        if joint["name"] in old_bounds:
            joint["limit"] = old_bounds[joint["name"]]
    if restored_model != before_model or _normalized_xml(xml_root) != before_xml:
        raise ValueError("Unexpected model/URDF mutation beyond knee limits and URDF version")
    builder_path = Path(__file__).with_name("build_full_vehicle_asset.py")
    builder_content = builder_path.read_bytes()
    if not _builder_declares_urdf12(builder_content):
        raise ValueError("Copy the updated generic URDF 1.2 builder before finalizing")
    with tempfile.TemporaryDirectory(prefix=".knee-finalize-", dir=asset.parent) as temporary:
        stage_dir = Path(temporary)
        # Relative USD mesh references remain readable without copying source meshes.
        if (asset / "meshes").exists():
            (stage_dir / "meshes").symlink_to(asset / "meshes", target_is_directory=True)
        layer = Sdf.Layer.FindOrOpen(str(asset / "robot.usda"))
        if not layer:
            raise ValueError("Cannot read authored USD")
        before_usd = _normalized_usd(layer)
        layer.Export(str(stage_dir / "robot.usda"))
        stage = Usd.Stage.Open(str(stage_dir / "robot.usda"))
        counts_before = _usd_counts(stage)
        for name, item in mapping.items():
            joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(f"/Robot/tree_joints/{name}"))
            joint.GetLowerLimitAttr().Set(item["native_limits_deg"][0])
            joint.GetUpperLimitAttr().Set(item["native_limits_deg"][1])
        if _normalized_usd(stage.GetRootLayer()) != before_usd:
            raise ValueError("Unexpected USD changes beyond the four knee bound attributes")
        format_checks = _check_formats(model, xml_root, stage, mapping)
        if _usd_counts(stage) != counts_before:
            raise ValueError("USD topology changed while setting knee bounds")
        stage.GetRootLayer().Save()
        payload["robot.usda"] = (stage_dir / "robot.usda").read_bytes()
    payload["robot.urdf"] = ET.tostring(xml_root, encoding="utf-8", xml_declaration=True)
    payload["model_spec.json"] = json_bytes(model)
    historical_limits = read_json(asset / "mechanical_limits.json")
    limits = copy.deepcopy(historical_limits)
    limits.update(knee_inner_angle_deg=list(INNER_LIMITS_DEG), training_inner_angle_deg=list(INNER_LIMITS_DEG),
                  source=LIMIT_SOURCE, measured_hardware_stop=False, physics_qualified=False,
                  lower_limit_changed=historical_limits["knee_inner_angle_deg"][0] != INNER_LIMITS_DEG[0],
                  upper_limit_changed=historical_limits["knee_inner_angle_deg"][1] != INNER_LIMITS_DEG[1],
                  historical_parent_limits=historical_limits, native_knee_mapping=mapping,
                  conditional_spring_check=spring_check, cad_onset_evidence="source_audits/limits/onset_summary.json")
    payload["mechanical_limits.json"] = json_bytes(limits)
    validation = read_json(asset / "full_vehicle_static_validation.json")
    validation.update(core_joint_contract_unchanged=False, core_joint_geometry_unchanged=True,
                      core_joint_limit_only_change=list(KNEES), core_nominal_joint_pos_unchanged=True,
                      core_constraint_anchors_unchanged=True, knee_candidate_validation=RECEIPT,
                      urdf_format_version="1.2")
    payload["full_vehicle_static_validation.json"] = json_bytes(validation)
    receipt = {"format": "cad-knee-candidate-finalization-v1", "limit_source": LIMIT_SOURCE,
               "inner_limits_deg": list(INNER_LIMITS_DEG), "measured_hardware_stop": False,
               "original_manifest_sha256": hashlib.sha256(original_manifest_bytes).hexdigest(),
               "original_asset_revision": manifest["asset_revision"],
               "historical_inner_limits_deg": manifest["knee_inner_limits_deg"],
               "historical_mechanical_limits_source": historical_limits.get("source"),
               "old_native_knee_limits": old_bounds, "native_knee_mapping": mapping,
               "expected_changes_only": True, "nominal_closure_max_m_before_and_after": nominal_closure,
               "loop_origins_axes_nominal_and_gas_binding_unchanged": True,
               "conditional_spring_check": spring_check, "cad_onsets": onset_summary,
               "format_checks": format_checks, **URDF_METADATA,
               "original_pin_count_verified": len(manifest["files_sha256"]),
               "source_urdf_sha256": manifest["source_urdf_sha256"],
               "physics_executed": False, "training_executed": False}
    payload[RECEIPT] = json_bytes(receipt)
    payload["tools/" + Path(__file__).name] = Path(__file__).read_bytes()
    payload["tools/build_full_vehicle_asset.py"] = builder_content
    readme = (asset / "FULL_VEHICLE_CANDIDATE.md").read_text()
    readme = readme.replace(
        "Its knee limits and gas-spring travel remain historical inputs pending new CAD travel review; "
        "they are not newly confirmed mechanical ranges.",
        "Its previous knee range is retained as historical provenance; the active knee range is now the "
        "CAD candidate 25--127 degrees. Gas-spring travel remains a conditional historical input, not a newly measured range.")
    readme += ("\n## CAD candidate knee update\n\n"
               "The active trial knee inner-angle range is 25--127 degrees, derived from the new CAD surface "
               "intersection audit. This is not a measured hardware stop. The previous 40--107 degree range "
               "is retained as historical provenance. Joint origins, axes, six loops, reset pose and gas-spring "
               "parameters are unchanged. The 152--232 mm spring interval remains a conditional historical "
               "input, not a newly measured travel range. See knee_candidate_limits_validation.json.\n\n"
               "robot.urdf declares URDF 1.2 and needs a compatible parser. The omitted gimbal pitch velocity "
               "uses the format default infinity; actual hardware speed remains unknown, not measured unlimited. "
               "Native training consumes the authored USD. Pitch remains the user-specified -30/+35 degrees.\n")
    payload["FULL_VEHICLE_CANDIDATE.md"] = readme.encode()
    manifest.update(asset_revision=FINAL_REVISION, knee_inner_limits_deg=list(INNER_LIMITS_DEG),
                    historical_knee_inner_limits_deg=manifest["knee_inner_limits_deg"],
                    mechanical_scope="CAD knee candidate 25--127 degrees; historical spring travel conditional; user-specified pitch down30/up35; not measured stops",
                    core_mechanical_limit_evidence=LIMIT_SOURCE, knee_candidate_validation=RECEIPT,
                    source_audit_evidence="source_audits/source_hashes.json", **URDF_METADATA)
    pins = dict(manifest["files_sha256"])
    pins.update({name: hashlib.sha256(content).hexdigest() for name, content in payload.items()})
    manifest["files_sha256"] = dict(sorted(pins.items()))
    payload["manifest.json"] = json_bytes(manifest)
    if (asset / "manifest.json").read_bytes() != original_manifest_bytes:
        raise ValueError("Asset manifest changed during preparation")
    verify_pins(asset, json.loads(original_manifest_bytes))
    original_files = {name: (asset / name).read_bytes() if (asset / name).exists() else None for name in payload}
    committed = []
    try:
        for name in sorted(payload, key=lambda name: (name == "manifest.json", name)):
            _atomic_write(asset / name, payload[name])
            committed.append(name)
        verify_pins(asset, manifest)
    except BaseException:
        for name in reversed(committed):
            if original_files[name] is None:
                (asset / name).unlink(missing_ok=True)
            else:
                _atomic_write(asset / name, original_files[name])
        raise
    return {"status": "finalized", "asset_revision": FINAL_REVISION,
            "manifest_sha256": sha(asset / "manifest.json"), "pin_count": len(pins),
            "native_knee_mapping": mapping, "conditional_spring_check": spring_check}


def migrate_change_flags(asset_dir: Path) -> dict:
    """Explicitly repair only the initial finalizer's inherited change flags.

    The original receipt and every numeric/geometry file remain untouched. This
    narrowly scoped migration is never invoked by ordinary idempotent reruns.
    """
    from pxr import Usd

    asset = asset_dir.resolve()
    manifest = read_json(asset / "manifest.json")
    verify_pins(asset, manifest)
    original_manifest = (asset / "manifest.json").read_bytes()
    if (manifest.get("asset_revision") != FINAL_REVISION
            or manifest.get("knee_inner_limits_deg") != list(INNER_LIMITS_DEG)):
        raise ValueError("Change-flag migration only accepts the finalized CAD 25--127 bundle")
    limits = read_json(asset / "mechanical_limits.json")
    receipt = read_json(asset / RECEIPT)
    if (limits.get("source") != LIMIT_SOURCE or receipt.get("limit_source") != LIMIT_SOURCE
            or limits.get("knee_inner_angle_deg") != list(INNER_LIMITS_DEG)
            or limits.get("training_inner_angle_deg") != list(INNER_LIMITS_DEG)):
        raise ValueError("The finalized knee metadata does not match the supported migration")
    model = read_json(asset / "model_spec.json")
    mapping = knee_limit_mapping(model)
    stage = Usd.Stage.Open(str(asset / "robot.usda"))
    checks = _check_formats(model, ET.parse(asset / "robot.urdf").getroot(), stage, mapping)
    script_relative = "tools/" + Path(__file__).name
    script_bytes = Path(__file__).read_bytes()
    if (limits.get("lower_limit_changed") is True and limits.get("upper_limit_changed") is True
            and (asset / script_relative).read_bytes() == script_bytes):
        return {"status": "change_flags_already_current", "manifest_sha256": sha(asset / "manifest.json")}
    changes = {name: {"before": limits.get(name), "after": True}
               for name in ("lower_limit_changed", "upper_limit_changed")}
    limits.update(lower_limit_changed=True, upper_limit_changed=True)
    payload = {"mechanical_limits.json": json_bytes(limits), script_relative: script_bytes}
    manifest["knee_change_flags_metadata_migration"] = {
        "revision": 1, "source_manifest_sha256": hashlib.sha256(original_manifest).hexdigest(),
        "reason": "Initial finalizer retained the historical lower_limit_changed flag; numeric limits were already correct",
        "changes": changes, "numeric_geometry_and_pitch_files_unchanged": True,
    }
    manifest["files_sha256"].update({name: hashlib.sha256(value).hexdigest() for name, value in payload.items()})
    manifest["files_sha256"] = dict(sorted(manifest["files_sha256"].items()))
    payload["manifest.json"] = json_bytes(manifest)
    if (asset / "manifest.json").read_bytes() != original_manifest:
        raise ValueError("Asset changed during metadata migration")
    verify_pins(asset, json.loads(original_manifest))
    originals = {name: (asset / name).read_bytes() for name in payload}
    committed = []
    try:
        for name in ("mechanical_limits.json", script_relative, "manifest.json"):
            _atomic_write(asset / name, payload[name])
            committed.append(name)
        verify_pins(asset, manifest)
        final = read_json(asset / "mechanical_limits.json")
        if final["lower_limit_changed"] is not True or final["upper_limit_changed"] is not True:
            raise ValueError("Metadata migration readback failed")
        _check_formats(read_json(asset / "model_spec.json"), ET.parse(asset / "robot.urdf").getroot(), stage, mapping)
    except BaseException:
        for name in reversed(committed):
            _atomic_write(asset / name, originals[name])
        raise
    return {"status": "change_flags_migrated", "manifest_sha256": sha(asset / "manifest.json"),
            "pin_count": len(manifest["files_sha256"]), "changed_files": list(payload),
            "static_format_checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument("--repair-change-flags", action="store_true",
                        help="Explicit metadata-only repair of the initial finalized bundle")
    args = parser.parse_args()
    if not args.repair_change_flags and args.audit_dir is None:
        parser.error("--audit-dir is required when finalizing an asset")
    result = (migrate_change_flags(args.asset_dir) if args.repair_change_flags
              else apply_candidate_limits(args.asset_dir, args.audit_dir))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
