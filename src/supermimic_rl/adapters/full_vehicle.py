"""Full articulation lifecycle hooks around the unchanged core locomotion task."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math

import torch

from .core_projection import CoreArticulationProjection
from .mechanics import read_articulation_mechanics, validate_mechanics_backend


def make_full_vehicle_env(
    base_type, asset, gimbal_config, *, controller_type=None,
    native_to_tensor=None, tensor_to_native=None,
    mechanics_backend="physx",
):
    """Return a flat Lab environment subclass with an independently driven gimbal.

    ``asset`` must be the already authenticated upstream AssetDescription. Its
    core model remains the reset/kinematics source; the full native articulation
    stays in the scene. The upstream ``_build_startup_dynamics`` hook is required.
    Optional dependencies support CPU interface tests without importing Isaac.

    This verifies native mass accounting and records native inertial properties.
    It does not qualify the asset or equate SDK principal frames to source link
    frames. No simulator is started until the returned class is instantiated.
    """
    if not isinstance(base_type, type) or not callable(getattr(base_type, "_build_startup_dynamics", None)):
        raise TypeError("full vehicle requires the upstream startup-dynamics hook")
    backend = validate_mechanics_backend(mechanics_backend)
    manifest = deepcopy(asset.manifest)
    model = json.loads(asset.bound_bytes(manifest["auxiliary_model"], maximum_bytes=256 * 1024))
    core_bodies = CoreArticulationProjection._validate_names(manifest["rigid_body_names"], "core body")
    core_joints = CoreArticulationProjection._validate_names(manifest["tree_joint_names"], "core joint")
    aux_bodies = CoreArticulationProjection._validate_names(model["body_names"], "auxiliary body")
    aux_joints = CoreArticulationProjection._validate_names(model["joint_names"], "auxiliary joint")
    if (len(aux_bodies) != 2 or len(aux_joints) != 2
            or tuple(manifest["auxiliary_body_names"]) != aux_bodies
            or tuple(manifest["auxiliary_joint_names"]) != aux_joints
            or set(core_bodies) & set(aux_bodies) or set(core_joints) & set(aux_joints)):
        raise ValueError("full vehicle requires disjoint core axes and two named auxiliary axes")
    nominal_mass = {}
    for key in ("core_mass_kg", "auxiliary_mass_kg", "total_mass_kg"):
        value = manifest[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("full vehicle masses must be finite positive source values")
        nominal_mass[key] = float(value)
    if not math.isclose(nominal_mass["core_mass_kg"] + nominal_mass["auxiliary_mass_kg"],
                        nominal_mass["total_mass_kg"], abs_tol=1e-9, rel_tol=0):
        raise ValueError("source core and auxiliary masses do not sum to the full vehicle mass")
    joints = {joint["name"]: joint for joint in model["joints"]}
    bodies = {body["name"]: body for body in model["bodies"]}
    if (len(joints) != len(model["joints"]) or set(joints) != set(aux_joints)
            or len(bodies) != len(model["bodies"]) or set(bodies) != set(aux_bodies)):
        raise ValueError("auxiliary source records must match their named body and joint axes")
    signs = tuple(model["control_joint_signs"])
    limits = tuple(joints[name]["effort_limit_nm"] for name in aux_joints)
    source_aux_mass = tuple(bodies[name]["mass"] for name in aux_bodies)
    nominal_position = tuple(manifest["nominal_joint_pos"][name] for name in aux_joints)
    if (len(signs) != 2 or any(type(sign) not in (int, float) or sign not in (-1, 1) for sign in signs)
            or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                   for value in (*limits, *source_aux_mass))
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in nominal_position)
            or not math.isclose(sum(source_aux_mass), nominal_mass["auxiliary_mass_kg"],
                                abs_tol=1e-9, rel_tol=0)):
        raise ValueError("auxiliary source requires signed axes, positive limits/masses and nominal positions")
    if controller_type is None:
        from ..gimbal import IndependentGimbalController

        controller_type = IndependentGimbalController

    class FullVehicleEnv(base_type):
        mechanics_backend = backend

        def __init__(self, *args, **kwargs):
            self._gimbal = None
            self._gimbal_model = deepcopy(model)
            self._full_vehicle_receipt = None
            super().__init__(*args, **kwargs)
            self._bind_full_vehicle()
            self._gimbal_seed = int(getattr(self.cfg, "seed", 0) or 0) ^ 67867967
            self._gimbal = controller_type(
                gimbal_config, self.num_envs, self.device, self._gimbal_seed, signs, limits)
            pitch_limits = joints[aux_joints[1]].get("mechanical_position_limits_rad")
            if pitch_limits is not None:
                native_targets = sorted(value * signs[1] for value in self._gimbal.config.pitch_position_rad)
                if native_targets[0] < pitch_limits[0] - 1e-7 or native_targets[1] > pitch_limits[1] + 1e-7:
                    raise ValueError("independent pitch commands exceed the selected asset joint limits")
            self._full_vehicle_receipt["gimbal_control"] = {
                "seed": self._gimbal_seed,
                "configuration": asdict(self._gimbal.config),
                "joint_names": list(aux_joints),
                "control_joint_signs": list(signs),
                "effort_limits_nm": list(limits),
                "command_order": ["relative_yaw_velocity_rad_s", "relative_pitch_position_rad"],
                "command_schedule": "sampled once per physical episode reset",
                "actuation": "native joint effort; reaction supplied by full articulation physics",
                "reset_position_native_rad": list(nominal_position),
                "pitch_joint_limits_native_rad": pitch_limits,
                "pitch_limit_source": joints[aux_joints[1]].get("position_limit_source", "unspecified"),
                "mechanical_position_limits": (
                    "pitch limits explicitly authored in the selected asset; yaw continuous"
                    if pitch_limits is not None else "unknown; command ranges are not mechanical limits"),
            }

        def _setup_scene(self):
            super()._setup_scene()
            self._full_robot = self.robot
            self.robot = CoreArticulationProjection(
                self._full_robot, core_bodies, core_joints,
                native_to_tensor=native_to_tensor, tensor_to_native=tensor_to_native,
                mechanics_backend=backend)

        def _bind_full_vehicle(self):
            # Force the lazy projection to verify the actual backend/public map.
            _ = self.robot.body_names
            full = self._full_robot
            names = tuple(full.body_names)
            joint_names = tuple(full.joint_names)
            if (len(names) != len(core_bodies) + len(aux_bodies)
                    or set(names) != set(core_bodies) | set(aux_bodies)
                    or len(joint_names) != len(core_joints) + len(aux_joints)
                    or set(joint_names) != set(core_joints) | set(aux_joints)
                    or self.robot.num_instances != self.num_envs):
                raise ValueError("native full vehicle axes differ from the authenticated core/auxiliary split")
            self._gimbal_body_indices = torch.tensor(
                [names.index(name) for name in aux_bodies], dtype=torch.long, device=self.device)
            self._gimbal_joint_indices = torch.tensor(
                [joint_names.index(name) for name in aux_joints], dtype=torch.long, device=self.device)
            self._full_backend_body_names = (
                tuple(path.rsplit("/", 1)[-1] for path in full.root_view.link_paths[0])
                if backend == "physx" else names)
            self._gimbal_backend_body_indices = torch.tensor(
                [self._full_backend_body_names.index(name) for name in aux_bodies], dtype=torch.long)

        def _read_full_mechanics(self):
            full = self._full_robot
            parameters = read_articulation_mechanics(
                full, backend=backend, tensor_converter=self.robot._tensor)
            if parameters.body_names != self._full_backend_body_names:
                raise ValueError("full-vehicle mechanical readback body order changed")
            self._mechanical_frames = (parameters.com_frame, parameters.inertia_frame)
            result = {}
            for name, value, tail in (
                ("mass_kg", parameters.mass_kg, ()),
                ("com_pose_local_xyzw", parameters.com_pose_local_xyzw, (7,)),
                ("inertia_kg_m2", parameters.inertia_kg_m2, (9,)),
            ):
                if (tuple(value.shape) != (self.num_envs, len(full.body_names)) + tail
                        or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())):
                    raise ValueError("invalid native full-vehicle mechanical readback: " + name)
                result[name] = value.detach().cpu().clone()
            if not bool((result["mass_kg"] > 0).all()):
                raise ValueError("native full-vehicle body masses must be positive")
            norms = result["com_pose_local_xyzw"][..., 3:].norm(dim=-1)
            if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=0):
                raise ValueError("native full-vehicle COM poses require normalized xyzw quaternions")
            return result

        def _mechanics_receipt(self, values):
            digest = hashlib.sha256()
            properties = {}
            for name, value in values.items():
                digest.update(value.contiguous().numpy().tobytes())
                properties[name] = {
                    "row0": value[0].tolist(),
                    "minimum_by_body": value.amin(dim=0).tolist(),
                    "maximum_by_body": value.amax(dim=0).tolist(),
                    "sha256": hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest(),
                }
            total = values["mass_kg"].sum(-1)
            return {
                "native_parameters_sha256": digest.hexdigest(),
                "body_order": list(self._full_backend_body_names),
                "num_envs": self.num_envs,
                "total_mass_kg_min": float(total.min()),
                "total_mass_kg_max": float(total.max()),
                "properties": properties,
                "com_frame": self._mechanical_frames[0],
                "inertia_frame": self._mechanical_frames[1],
            }

        def _build_startup_dynamics(
            self, robot, binding, num_envs, nominal_mass_kg, seed, config, **kwargs,
        ):
            if (robot is not self.robot or num_envs != self.num_envs
                    or not math.isclose(nominal_mass_kg, nominal_mass["total_mass_kg"],
                                        abs_tol=1e-9, rel_tol=0)):
                raise ValueError("startup dynamics must receive the selected full vehicle nominal mass")
            self._bind_full_vehicle()
            before = self._read_full_mechanics()
            expected_total = torch.full((num_envs,), nominal_mass["total_mass_kg"])
            if not torch.allclose(before["mass_kg"].sum(-1), expected_total, atol=1e-5, rtol=0):
                raise ValueError("native full vehicle nominal mass differs from the selected asset")
            aux_indices = self._gimbal_backend_body_indices
            aux_mass = before["mass_kg"].index_select(1, aux_indices)
            if not torch.allclose(aux_mass, torch.tensor(source_aux_mass).expand_as(aux_mass),
                                  atol=1e-6, rtol=0):
                raise ValueError("native auxiliary body masses differ from the selected asset")
            requested_configuration = asdict(config) if is_dataclass(config) else deepcopy(config)
            all_nominal = config is not None and getattr(config, "nominal_fraction", None) == 1.0
            if backend == "newton" and config is not None and not all_nominal:
                raise ValueError("Newton full-vehicle mechanics currently support nominal startup only")
            # The upstream sampler makes every target exactly nominal here.
            # Avoid a redundant PhysX inertia decomposition and the SDK's stale
            # auxiliary COM-cache publication on its later partial COM write.
            # This does not claim to repair randomized full-vehicle writes.
            effective_config = None if all_nominal else config
            dynamics = super()._build_startup_dynamics(
                robot, binding, num_envs, nominal_mass["core_mass_kg"], seed, effective_config, **kwargs)
            core_receipt = deepcopy(dynamics.receipt)
            after = self._read_full_mechanics()
            aux_comparison = {}
            for name in before:
                previous = before[name].index_select(1, aux_indices)
                current = after[name].index_select(1, aux_indices)
                difference = (previous - current).abs()
                comparison = {"max_absolute_difference": float(difference.max())}
                if name == "com_pose_local_xyzw":
                    left, right = previous[..., 3:], current[..., 3:]
                    left = left / left.norm(dim=-1, keepdim=True)
                    right = right / right.norm(dim=-1, keepdim=True)
                    rotation_error = torch.minimum(
                        (left - right).abs().amax(-1), (left + right).abs().amax(-1))
                    matched = (torch.allclose(previous[..., :3], current[..., :3], atol=1e-6, rtol=0)
                               and bool(rotation_error.le(1e-6).all()))
                    comparison.update({
                        "position_max_absolute_difference_m": float(difference[..., :3].max()),
                        "quaternion_sign_equivalent_max_absolute_difference": float(rotation_error.max()),
                        "position_atol_m": 1e-6, "quaternion_atol": 1e-6,
                    })
                else:
                    rtol = 1e-4 if name == "inertia_kg_m2" else 1e-5
                    matched = torch.allclose(previous, current, atol=1e-7, rtol=rtol)
                    comparison.update({"rtol": rtol, "atol": 1e-7})
                comparison["matched"] = matched
                aux_comparison[name] = comparison
                if not matched:
                    raise RuntimeError(
                        "core startup dynamics changed auxiliary native properties: " + name
                        + f"; comparison={comparison}; before_row0={previous[0].tolist()}"
                        + f"; after_row0={current[0].tolist()}")
            total = after["mass_kg"].sum(-1).to(device=self.device)
            aux_actual = after["mass_kg"].index_select(1, aux_indices).sum(-1).to(device=self.device)
            if not torch.allclose(dynamics.mass_kg_by_environment + aux_actual, total,
                                  atol=1e-5, rtol=0):
                raise RuntimeError("core startup mass accounting differs from the full native readback")
            dynamics.mass_kg_by_environment = total
            dynamics.mass_scale = (total / nominal_mass["total_mass_kg"])[:, None]
            after_receipt = self._mechanics_receipt(after)
            self._full_vehicle_receipt = {
                "status": "native_mass_accounted_and_inertial_properties_recorded",
                "asset_manifest_sha256": asset.manifest_sha256,
                "core_body_names": list(core_bodies), "core_joint_names": list(core_joints),
                "auxiliary_body_names": list(aux_bodies), "auxiliary_joint_names": list(aux_joints),
                "public_body_names": list(self._full_robot.body_names),
                "public_joint_names": list(self._full_robot.joint_names),
                "nominal_masses": nominal_mass.copy(),
                "before_core_startup": self._mechanics_receipt(before),
                "after_core_startup": after_receipt,
                "core_startup": core_receipt,
                "requested_startup_configuration": requested_configuration,
                "write_skipped_all_nominal": all_nominal,
                "randomized_auxiliary_property_writes_validated": False,
                "auxiliary_properties_unchanged_by_core_startup": True,
                "auxiliary_property_comparison": aux_comparison,
                "auxiliary_comparison_contract": "upstream startup tolerances; quaternion signs equivalent",
                "reward_and_critic_mass": "actual full articulation mass including auxiliary bodies",
                "source_com_inertia_validation": "readback recorded; source-frame comparison not performed",
                "physical_qualification": False,
            }
            # Existing reward/critic consumers read these public fields after
            # construction. Preserve the core COM helpers and configuration.
            dynamics._receipt = {
                **core_receipt,
                "status": "native_full_vehicle_mass_accounted",
                "body_order": list(self._full_backend_body_names),
                "mass_kg_min": float(total.min()), "mass_kg_max": float(total.max()),
                "nominal_mass_kg": nominal_mass["total_mass_kg"],
                "native_parameters_sha256": after_receipt["native_parameters_sha256"],
                "randomization_scope": "core only; auxiliary properties unchanged",
                "requested_configuration": requested_configuration,
                "write_skipped_all_nominal": all_nominal,
                "core_startup": core_receipt,
                "physical_qualification": False,
            }
            return dynamics

        def _apply_action(self):
            super()._apply_action()
            if self._gimbal is None:
                return
            native = self._full_robot
            position = self.robot._tensor(native.data.joint_pos).index_select(1, self._gimbal_joint_indices)
            velocity = self.robot._tensor(native.data.joint_vel).index_select(1, self._gimbal_joint_indices)
            effort = self._gimbal.effort(position, velocity)
            native.set_joint_effort_target(effort, joint_ids=self._gimbal_joint_indices)

        def _reset_idx(self, env_ids):
            if self._gimbal is not None:
                ids = (torch.arange(self.num_envs, device=self.device) if env_ids is None
                       else torch.as_tensor(env_ids, dtype=torch.long, device=self.device))
                if ids.numel():
                    self._gimbal.reset(ids)
                    position = torch.tensor(nominal_position, dtype=torch.float32, device=self.device)
                    position = position.expand(len(ids), -1).contiguous()
                    zeros = torch.zeros_like(position)
                    native = self._full_robot
                    native.write_joint_position_to_sim_index(
                        position=position, joint_ids=self._gimbal_joint_indices, env_ids=ids)
                    native.write_joint_velocity_to_sim_index(
                        velocity=zeros, joint_ids=self._gimbal_joint_indices, env_ids=ids)
                    native.set_joint_effort_target(
                        zeros, joint_ids=self._gimbal_joint_indices, env_ids=ids)
            # The parent forwards the scene after writing core q/dq. Auxiliary
            # q/dq must already be installed so its readback describes one reset.
            return super()._reset_idx(env_ids)

        @property
        def full_vehicle_receipt(self):
            return deepcopy(self._full_vehicle_receipt)

    FullVehicleEnv.__name__ = "FullVehicle" + base_type.__name__
    return FullVehicleEnv
