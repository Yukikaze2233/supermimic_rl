#!/usr/bin/env python3
"""Derive a full-vehicle bundle that differs from the finalized one only in knee limits.

The finalized CAD bundle uses a 25--127 degree knee candidate. The user chose
40--107 degrees on 2026-11-02 to align with the historical stop and the
identified gas-spring geometry domain (40--110 degrees). Only the two passive
knee bounds change, in model_spec, URDF, USD and mechanical_limits; the
mechanism itself (geometry, six loops, nominal pose, gimbal, gas-spring
binding) and every other byte stay identical and are checked. The source
bundle is untouched. Requires the optional CPU numpy/scipy/usd-core environment.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("knee_limits", HERE / "set_knee_candidate_limits.py")
knee = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(knee)

RECEIPT = "knee_limit_variant.json"


def _revision(limits_deg) -> str:
    low, high = (int(value) if float(value).is_integer() else value for value in limits_deg)
    return f"{knee.INPUT_REVISION}_knee{low}_{high}_pitch_m30_p35"


def _set_limits(model: dict, xml_root: ET.Element, stage, mapping: dict) -> None:
    from pxr import UsdPhysics

    for name, item in mapping.items():
        joint = next(joint for joint in model["joints"] if joint["name"] == name)
        node = xml_root.find(f"joint[@name='{name}']/limit")
        for key, value in zip(("lower", "upper"), item["native_limits_rad"]):
            joint["limit"][key] = format(value, ".17g") if isinstance(joint["limit"][key], str) else value
            node.set(key, format(value, ".17g"))
        usd = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(f"/Robot/tree_joints/{name}"))
        usd.GetLowerLimitAttr().Set(item["native_limits_deg"][0])
        usd.GetUpperLimitAttr().Set(item["native_limits_deg"][1])


def derive(source: Path, output: Path, limits_deg: tuple[float, float], limit_source: str) -> dict:
    from pxr import Usd

    source, output = source.resolve(strict=True), output.resolve()
    if output.exists():
        raise FileExistsError("Derive into a new asset directory")
    manifest = knee.read_json(source / "manifest.json")
    knee.verify_pins(source, manifest)
    if manifest.get("asset_revision") != knee.FINAL_REVISION:
        raise ValueError("The source must be the finalized CAD-candidate bundle")
    source_manifest_sha256 = knee.sha(source / "manifest.json")
    model = knee.read_json(source / "model_spec.json")
    mapping = knee.knee_limit_mapping(model, limits_deg)
    spring = knee.conditional_spring_check(model, mapping, knee.read_json(source / "gas_spring_binding.json"),
                                           limits_deg)
    with tempfile.TemporaryDirectory(prefix=".knee-variant-", dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        shutil.copytree(source, staging, symlinks=False)
        before_model, before_xml_root = copy.deepcopy(model), ET.parse(staging / "robot.urdf").getroot()
        before_xml = knee._normalized_xml(before_xml_root)
        stage = Usd.Stage.Open(str(staging / "robot.usda"))
        before_usd, counts = knee._normalized_usd(stage.GetRootLayer()), knee._usd_counts(stage)
        xml_root = copy.deepcopy(before_xml_root)
        _set_limits(model, xml_root, stage, mapping)
        restored = copy.deepcopy(model)
        for joint in restored["joints"]:
            if joint["name"] in mapping:
                joint["limit"] = next(j for j in before_model["joints"] if j["name"] == joint["name"])["limit"]
        if (restored != before_model or knee._normalized_xml(xml_root) != before_xml
                or knee._normalized_usd(stage.GetRootLayer()) != before_usd or knee._usd_counts(stage) != counts):
            raise ValueError("Unexpected change beyond the knee bounds")
        formats = knee._check_formats(model, xml_root, stage, mapping)
        stage.GetRootLayer().Save()
        payload = {"model_spec.json": knee.json_bytes(model),
                   "robot.urdf": ET.tostring(xml_root, encoding="utf-8", xml_declaration=True)}
        parent_limits = knee.read_json(source / "mechanical_limits.json")
        limits = copy.deepcopy(parent_limits)
        limits.update(knee_inner_angle_deg=list(limits_deg), training_inner_angle_deg=list(limits_deg),
                      source=limit_source, measured_hardware_stop=False, physics_qualified=False,
                      lower_limit_changed=parent_limits["knee_inner_angle_deg"][0] != limits_deg[0],
                      upper_limit_changed=parent_limits["knee_inner_angle_deg"][1] != limits_deg[1],
                      historical_parent_limits=parent_limits, native_knee_mapping=mapping,
                      conditional_spring_check=spring)
        payload["mechanical_limits.json"] = knee.json_bytes(limits)
        receipt = {"format": "knee-limit-variant-v1", "limit_source": limit_source,
                   "inner_limits_deg": list(limits_deg), "measured_hardware_stop": False,
                   "source_manifest_sha256": source_manifest_sha256,
                   "source_asset_revision": manifest["asset_revision"],
                   "source_inner_limits_deg": manifest["knee_inner_limits_deg"],
                   "native_knee_mapping": mapping, "conditional_spring_check": spring, "format_checks": formats,
                   "expected_changes_only": True, "physics_executed": False, "training_executed": False}
        payload[RECEIPT] = knee.json_bytes(receipt)
        payload["tools/" + Path(__file__).name] = Path(__file__).read_bytes()
        payload["tools/set_knee_candidate_limits.py"] = (HERE / "set_knee_candidate_limits.py").read_bytes()
        readme = (staging / "FULL_VEHICLE_CANDIDATE.md").read_text()
        readme += (f"\n## Knee limit variant {limits_deg[0]:g}--{limits_deg[1]:g} degrees\n\n"
                   f"Derived from {manifest['asset_revision']} (manifest {source_manifest_sha256}). Only the two "
                   f"passive knee bounds changed; source: {limit_source}. Not a measured hardware stop. "
                   f"See {RECEIPT}.\n")
        payload["FULL_VEHICLE_CANDIDATE.md"] = readme.encode()
        for name, content in payload.items():
            knee._atomic_write(staging / name, content)
        manifest = copy.deepcopy(manifest)
        manifest.update(asset_revision=_revision(limits_deg), knee_inner_limits_deg=list(limits_deg),
                        historical_knee_inner_limits_deg=manifest["knee_inner_limits_deg"],
                        mechanical_scope=(f"knee {limits_deg[0]:g}--{limits_deg[1]:g} degrees ({limit_source}); "
                                          "historical spring travel conditional; user-specified pitch down30/up35; "
                                          "not measured stops"),
                        knee_limit_variant=RECEIPT, knee_limit_variant_source_manifest_sha256=source_manifest_sha256)
        pins = dict(manifest["files_sha256"])
        pins.update({name: knee.sha(staging / name) for name in (*payload, "robot.usda")})
        manifest["files_sha256"] = dict(sorted(pins.items()))
        knee._atomic_write(staging / "manifest.json", knee.json_bytes(manifest))
        knee.verify_pins(staging, manifest)
        knee._check_formats(knee.read_json(staging / "model_spec.json"), ET.parse(staging / "robot.urdf").getroot(),
                            Usd.Stage.Open(str(staging / "robot.usda")), mapping)
        staging.rename(output)
    return {"status": "derived", "asset_directory": str(output), "asset_revision": manifest["asset_revision"],
            "manifest_sha256": knee.sha(output / "manifest.json"), "native_knee_mapping": mapping,
            "conditional_spring_check": spring}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inner-limits-deg", type=float, nargs=2, required=True)
    parser.add_argument("--limit-source", required=True)
    args = parser.parse_args()
    print(json.dumps(derive(args.source, args.output, tuple(args.inner_limits_deg), args.limit_source), indent=2))


if __name__ == "__main__":
    main()
