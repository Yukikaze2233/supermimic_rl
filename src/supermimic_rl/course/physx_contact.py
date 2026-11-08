"""Requested contact stress relative to the unchanged physical terrain atlas.

The source used average mixing. PhysX uses multiply here: put the source pair
coefficient divided by the atlas flat coefficient on the robot. This matches
the source on flat ground and preserves each other tile's relative friction.
It is a derived simulation distribution, not a hardware measurement.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from .materials import MATERIAL_SOURCE_PINS, _digest, sample_ground_friction


def atlas_contact_profile(path, expected_sha256):
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("Contact reference atlas pin differs")
    manifest = json.loads(content)
    flat = [tile for tile in manifest["tiles"] if tile["kind"] == 0 and tile["name"] == "flat"]
    if len(flat) != 1 or manifest.get("material_combine_mode") != "multiply":
        raise ValueError("Contact reference requires one flat tile and multiply atlas materials")
    reference = flat[0]["parameters"]["friction"]
    if type(reference) not in (int, float) or not math.isfinite(reference) or reference <= 0:
        raise ValueError("Contact reference friction must be positive and finite")
    return {
        "schema": "requested_v2_physx_atlas_relative_v1",
        "source_sha256": dict(MATERIAL_SOURCE_PINS),
        "atlas_sha256": expected_sha256, "atlas_reference_tile": flat[0]["name"],
        "atlas_reference_mu": float(reference),
        "source_combine_mode": "average", "native_combine_mode": "multiply",
        "source_ground_mu_range": [.3, .5], "source_wheel_mu": .1, "source_body_mu": .25,
        "native_wheel_mu": "(.1 + source_ground_mu) / (2 * atlas_reference_mu)",
        "native_other_body_mu": "(.25 + source_ground_mu) / (2 * atlas_reference_mu)",
        "effective_pair": "source_average_pair * actual_tile_mu / atlas_reference_mu",
        "sampling": "one CPU uniform ground draw per environment at startup; seed xor 0x434F4E54",
        "nominal_lanes": "preserve original native robot materials using the existing startup nominal mask",
        "mixture": "project nominal/low-grip mixture; differs from source contact-on-all-lanes semantics",
        "terrain_materials_changed": False, "terrain_geometry_changed": False,
        "restitution_unchanged": True, "reset_resampling": False,
        "classification": "derived atlas-relative simulation stress; not measured hardware friction",
        "source_equivalence": "flat reference only; other tiles retain authored relative friction",
    }


def contact_provenance(reader, *, backend):
    """Describe actual application, never infer it from the requested profile."""
    receipt = reader.receipt
    applied = receipt.get("requested_contact_mapping")
    if applied is not None:
        if applied.get("status") != "native_readback_matched":
            raise ValueError("Contact mapping has no successful native readback")
        # Sampled values belong to setup evidence, not the environment identity.
        profile = applied.get("profile")
        if profile is None:  # Existing Kamino receipt embeds its profile.
            from .materials import material_mapping_provenance

            profile = material_mapping_provenance()
        return {**profile, "applied": True, "status": "native_readback_matched"}
    return {
        "applied": False, "status": "requested_contact_mapping_not_applied",
        "physical_materials": "native robot and atlas materials with native combine modes",
        "backend": backend,
        "legacy_wheel_material_applied": (
            receipt.get("startup_randomization", {}).get("status") == "native_readback_matched"),
    }


class PhysxContactScene:
    """Read active USD bindings; the robot tensor getter is checked separately."""

    def __init__(self, native):
        from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade

        self.native = native
        self.stage = native.scene.stage
        self.usd, self.physics, self.shade, self.physx = Usd, UsdPhysics, UsdShade, PhysxSchema
        self.terrain_paths = native._course_collision_receipt["collider_paths"]

    def read(self):
        stage, physics, shade = self.stage, self.physics, self.shade
        default, _ = shade.MaterialBindingAPI(stage.GetPrimAtPath(
            self.native.cfg.sim.physics_prim_path)).ComputeBoundMaterial(materialPurpose="physics")
        if not default:
            raise ValueError("PhysX scene default physics material is unavailable")

        def material(prim):
            bound, _ = shade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
            bound = bound or default
            value = physics.MaterialAPI(bound.GetPrim())
            combine = self.physx.PhysxMaterialAPI(bound.GetPrim()).GetFrictionCombineModeAttr().Get()
            return {"path": str(bound.GetPath()), "combine": combine,
                    "values": [value.GetStaticFrictionAttr().Get(), value.GetDynamicFrictionAttr().Get(),
                               value.GetRestitutionAttr().Get()]}

        # The native articulation reader verifies row/body/shape layout for all
        # environments. Binding inspection is of the replicated source row;
        # report that scope rather than claiming a hidden PhysX combine getter.
        robot_materials, collision_count = {}, 0
        for path in self.native.robot.root_view.link_paths[0]:
            for prim in self.usd.PrimRange(stage.GetPrimAtPath(path), self.usd.TraverseInstanceProxies()):
                if prim.HasAPI(physics.CollisionAPI) and physics.CollisionAPI(
                        prim).GetCollisionEnabledAttr().Get():
                    row = material(prim)
                    robot_materials[row["path"]] = row
                    collision_count += 1
        if not collision_count or collision_count != self.native.robot.root_view.max_shapes:
            raise ValueError("USD contact binding source row differs from native shape count")

        terrain, terrain_values = [], []
        for path in self.terrain_paths:
            row = material(stage.GetPrimAtPath(path))
            terrain_values.append(row["values"])
            terrain.append({"collider": path, **row})
        if not terrain_values:
            raise ValueError("Atlas contact mapping requires native terrain collision partitions")
        if any(row["combine"] != "multiply" for row in [*robot_materials.values(), *terrain]):
            raise ValueError("Atlas-relative contact mapping requires actual multiply bindings")
        enabled = []
        for path in self.native.robot.root_view.prim_paths:
            prim = stage.GetPrimAtPath(path)
            while prim and not prim.HasAPI(self.physx.PhysxArticulationAPI):
                prim = prim.GetParent()
            enabled.append(self.physx.PhysxArticulationAPI(prim).GetEnabledSelfCollisionsAttr().Get()
                           if prim else None)
        return {
            "robot_binding_scope": "native row 0 collision shapes; replicated scene source",
            "replicate_physics": self.native.cfg.scene.replicate_physics,
            "robot_source_shape_count": collision_count,
            "robot_material_bindings": list(robot_materials.values()),
            "terrain_material_bindings": terrain,
            "terrain_readback_source": "active USD collision physics-material bindings; no static Tensor getter",
            "terrain_bound_materials": torch.tensor(terrain_values, dtype=torch.float32),
            "self_collision": {"source": "PhysxArticulationAPI enabledSelfCollisions on native prim_paths",
                               "enabled_rows": enabled.count(True), "disabled_rows": enabled.count(False),
                               "unresolved_rows": enabled.count(None)},
        }


@torch.no_grad()
def apply_physx_contact_mapping(reader, randomization, profile, scene, *, tensor_to_native=None):
    """Apply once through the existing native reader, then verify full tables."""
    if (reader._startup_applied or reader.num_envs != randomization.num_envs
            or reader.robot.root_view is not reader._view):
        raise ValueError("Contact startup ownership, batch or native view differs")
    if reader.receipt["wheel_body_names"] != ["L_link3", "R_link3"]:
        raise ValueError("Contact mapping requires the named native wheel bodies")
    if tensor_to_native is None:
        import warp as wp

        tensor_to_native = lambda value: wp.from_torch(
            value, dtype=wp.int32 if value.dtype == torch.int32 else wp.float32)
    before_scene = scene.read()
    source = reader._read_native().detach().cpu().clone()
    if source.dtype != torch.float32 or not torch.isfinite(source).all():
        raise ValueError("Contact mapping requires finite native float32 properties")
    ground = sample_ground_friction(reader.num_envs, randomization.seed).float()
    wheel_pair, body_pair = (.1 + ground) / 2, (.25 + ground) / 2
    reference = profile["atlas_reference_mu"]
    wheel, body = wheel_pair / reference, body_pair / reference
    target = source.clone()
    target[:, :, :2] = body[:, None, None]
    for start, end in reader._ranges:
        target[:, start:end, :2] = wheel[:, None, None]
    nominal = randomization.nominal.detach().cpu()
    target[nominal] = source[nominal]
    reader._startup_applied = True
    indices = torch.arange(reader.num_envs, dtype=torch.int32)
    reader._view.set_material_properties(tensor_to_native(target.contiguous()), tensor_to_native(indices))
    actual = reader._read_native().detach().cpu()
    if not torch.equal(actual, target):
        raise RuntimeError("Contact material native readback differs from the full requested shape table")
    after_scene = scene.read()
    before_mu, after_mu = (item["terrain_bound_materials"] for item in (before_scene, after_scene))
    if (not torch.equal(before_mu, after_mu)
            or {key: value for key, value in before_scene.items() if key != "terrain_bound_materials"}
            != {key: value for key, value in after_scene.items() if key != "terrain_bound_materials"}):
        raise RuntimeError("Contact mapping changed terrain materials or scene material bindings")
    reader._publish(actual)
    evidence = {key: value for key, value in after_scene.items() if key != "terrain_bound_materials"}
    wheel_actual = actual[:, [start for start, _ in reader._ranges], :2]
    body_indices = [index for index in range(actual.shape[1])
                    if all(not start <= index < end for start, end in reader._ranges)]
    body_actual = actual[:, body_indices, :2]
    # These products use the native readback, not merely the sampled requests.
    stress = ~nominal
    flat_pair_error = ((wheel_actual[stress] * reference - wheel_pair[stress, None, None]).abs().max()
                       if stress.any() else torch.tensor(0.))
    body_pair_error = ((body_actual[stress] * reference - body_pair[stress, None, None]).abs().max()
                       if stress.any() else torch.tensor(0.))
    pair_values = wheel_actual[:, :, None, :] * after_mu[None, None, :, :2]
    body_values = body_actual[:, :, None, :] * after_mu[None, None, :, :2]
    receipt = {
        "profile": profile, "status": "native_readback_matched", "seed": randomization.seed,
        "ground_draw_sha256": _digest(ground), "robot_native_sha256": _digest(actual),
        "original_robot_native_sha256": _digest(source),
        "terrain_bound_sha256": _digest(after_mu), "scene_readback": evidence,
        "robot_environment_count": reader.num_envs, "robot_shapes_per_environment": actual.shape[1],
        "source_ground_draw_range": [float(ground.min()), float(ground.max())],
        "sampled_stress_wheel_mu_range": [float(wheel.min()), float(wheel.max())],
        "sampled_stress_other_body_mu_range": [float(body.min()), float(body.max())],
        "native_wheel_mu_range": [float(wheel_actual.min()), float(wheel_actual.max())],
        "native_other_body_mu_range": [float(body_actual.min()), float(body_actual.max())],
        "nominal_rows": int(nominal.sum()), "low_grip_rows": int(stress.sum()),
        "nominal_mask_sha256": _digest(nominal),
        "original_nominal_native_sha256": _digest(source[nominal]),
        "actual_nominal_native_sha256": _digest(actual[nominal]),
        "nominal_native_properties_unchanged": bool(torch.equal(actual[nominal], source[nominal])),
        "actual_reference_wheel_pair_error_max": float(flat_pair_error),
        "actual_reference_body_pair_error_max": float(body_pair_error),
        "actual_wheel_pair_all_tiles_range": [float(pair_values.min()), float(pair_values.max())],
        "actual_body_pair_all_tiles_range": [float(body_values.min()), float(body_values.max())],
        "terrain_bound_mu_range": [float(after_mu[:, :2].min()), float(after_mu[:, :2].max())],
        "restitution_unchanged": bool(torch.equal(actual[:, :, 2], source[:, :, 2])),
        "terrain_materials_changed": False, "terrain_geometry_changed": False,
        "native_all_robot_shapes_readback": True,
    }
    reader._receipt["requested_contact_mapping"] = receipt
    # Native critic wheel features already contain the randomized robot
    # coefficients. Do not relabel them as the pair coefficient on every tile.
    return receipt


def install_physx_contact_mapping(native, randomization, profile):
    return apply_physx_contact_mapping(native._wheel_materials, randomization, profile,
                                       PhysxContactScene(native))
