#!/usr/bin/env python3
"""Audit physics backend semantics from recorded substep truth (record_substeps + substep_audit).

Each check is an independent function over a SubstepView:
- body_velocity_convention: is the reported linear velocity the link origin or the COM?
- angular_velocity_convention: does R_k = exp(dt * omega_world) R_{k-1} hold?
- free_flight_momentum: without any contact, total momentum must change by M g dt.
- static_support: quasi-static support loads must carry the robot weight.
- loop_closure / joint_pose_consistency: closed-chain anchors and joint-coordinate FK
  against native body poses, using the pinned upstream kinematics (optional).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

from supermimic_rl.world_model_v2 import so3
from supermimic_rl.world_model_v2.substep_view import SubstepView

GRAVITY = torch.tensor([0., 0., -9.81], dtype=torch.float64)


def rms(value, mask=None):
    value = value if mask is None else value[mask]
    return value.square().mean(0).sqrt().tolist() if len(value) else None


def context_field(view, name):
    offset = 0
    for field in view.schema.context_fields:
        if field.name == name:
            raw = view.storage["start_context"][..., offset:offset + field.size].double()
            return raw.reshape(view.steps, view.num_envs, *field.shape)
        offset += field.size
    raise KeyError(name)


class BodyMotion:
    """Per-body link pose, COM offset and reported twist at every substep."""

    def __init__(self, view):
        pose = view.get("body_link_pose_world_xyzw")                       # T,N,K,B,7
        reported = view.get("body_com_velocity_reported_world")            # T,N,K,B,6
        com_local = context_field(view, "body_com_pose_local_xyzw")[..., :3]  # T,N,B,3
        self.position, self.rotation = pose[..., :3], so3.quaternion_to_matrix(pose[..., 3:])
        self.com_offset = (self.rotation @ com_local[:, :, None, :, :, None]).squeeze(-1)
        self.com = self.position + self.com_offset
        self.linear, self.angular = reported[..., :3], reported[..., 3:]
        self.mass = context_field(view, "body_mass_kg")                     # T,N,B
        self.dt = view.physics_dt_s
        self.within = view.valid[..., None].expand(-1, -1, view.substeps - 1)

    def com_velocity(self, convention):
        if convention == "reported_is_com":
            return self.linear
        return self.linear + torch.linalg.cross(self.angular, self.com_offset)


def body_velocity_convention(view, motion):
    after, before, dt = slice(1, None), slice(None, -1), motion.dt
    result = {}
    for index, name in enumerate(view.body_names):
        link_step = motion.position[:, :, after, index] - motion.position[:, :, before, index]
        com_step = motion.com[:, :, after, index] - motion.com[:, :, before, index]
        linear = motion.linear[:, :, after, index]
        corrected = motion.com_velocity("reported_is_link_origin")[:, :, after, index]
        result[name] = {
            "link_minus_dt_reported_m": rms(link_step - dt * linear, motion.within),
            "com_minus_dt_reported_m": rms(com_step - dt * linear, motion.within),
            "com_minus_dt_reported_plus_w_cross_c_m": rms(com_step - dt * corrected, motion.within),
            "com_offset_norm_m": float(motion.com_offset[view.valid][:, 0, index].norm(dim=-1).mean()),
        }
    return result


def angular_velocity_convention(view, motion):
    after, before = slice(1, None), slice(None, -1)
    result = {}
    for index, name in enumerate(view.body_names):
        omega = motion.angular[:, :, after, index]
        world = so3.exp(motion.dt * omega) @ motion.rotation[:, :, before, index]
        error = so3.angle(world.transpose(-1, -2) @ motion.rotation[:, :, after, index])
        result[name] = rms(error[..., None], motion.within)
    return result


def free_flight_momentum(view, motion):
    loads = torch.cat((view.get("body_normal_load_n"), view.get("wheel_support_load_n")), -1).amax(-1)
    free = (loads[:, :, 1:] < 1e-6) & (loads[:, :, :-1] < 1e-6) & motion.within
    total_mass = motion.mass.sum(-1)[:, :, None]
    result = {"ticks": int(free.sum())}
    for convention in ("reported_is_com", "reported_is_link_origin"):
        momentum = (motion.mass[:, :, None, :, None] * motion.com_velocity(convention)).sum(-2)
        rate = (momentum[:, :, 1:] - momentum[:, :, :-1]) / motion.dt
        relative = (rate - total_mass[..., None] * GRAVITY) / (total_mass[..., None] * 9.81)
        result[convention + "_relative_force_error"] = rms(relative, free)
    return result


def momentum_balance(view, motion):
    """dP/dt = M g + sum of terrain contact forces; tests contact readings at every tick."""
    momentum = (motion.mass[:, :, None, :, None] * motion.com_velocity("reported_is_com")).sum(-2)
    rate = (momentum[:, :, 1:] - momentum[:, :, :-1]) / motion.dt
    weight_vector = motion.mass.sum(-1)[:, :, None, None] * GRAVITY
    contact = view.get("terrain_contact_force_world_n").sum(-2)
    scale = motion.mass.sum(-1)[:, :, None, None] * 9.81
    loaded = (view.get("wheel_support_load_n").sum(-1) > 5)[:, :, 1:] & motion.within
    result = {"ticks": int(loaded.sum())}
    for name, force in (("contact_at_k", contact[:, :, 1:]), ("contact_at_k_minus_1", contact[:, :, :-1])):
        residual = (rate - weight_vector - force) / scale
        result[name + "_relative_residual"] = rms(residual, loaded)
        result[name + "_relative_residual_p99"] = residual[loaded].abs().quantile(.99, dim=0).tolist()
    result["contact_force_over_weight_rms"] = rms(contact[:, :, 1:] / scale, loaded)
    return result


def static_support(view, motion, convention):
    velocity = (motion.mass[:, :, None, :, None] * motion.com_velocity(convention)).sum(-2) \
        / motion.mass.sum(-1)[:, :, None, None]
    base = view.bodies(("base_link",))[0]
    wheel = view.get("wheel_support_load_n")
    forbidden = view.get("body_normal_load_n")[..., view.bodies(("base_link", "M_link1", "M_link2"))]
    quiet = ((velocity.norm(dim=-1) < .05) & (motion.angular[..., base, :].norm(dim=-1) < .2)
             & (wheel > 5).all(-1) & (forbidden < 1).all(-1) & view.valid[..., None])
    weight = motion.mass.sum(-1)[:, :, None] * 9.81
    vertical = view.get("terrain_contact_force_world_n")[..., 2].sum(-1)
    return {"ticks": int(quiet.sum()),
            "wheel_support_over_weight": rms((wheel.sum(-1) / weight)[..., None] - 1, quiet),
            "vertical_terrain_force_over_weight_minus_one": rms((vertical / weight)[..., None] - 1, quiet),
            "mean_wheel_support_over_weight": float((wheel.sum(-1) / weight)[quiet].mean()) if quiet.any() else None}


def upstream_kinematics(upstream_root, asset_directory):
    sys.path.insert(0, str(Path(upstream_root) / "src"))
    from transformer_rl.formal.control.observation import MechanicalBinding
    from transformer_rl.formal.tasks.reset_kinematics import AssetResetKinematics
    directory = Path(asset_directory)
    raw = (directory / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    binding = MechanicalBinding.from_manifest(manifest, manifest["tree_joint_names"],
                                              manifest["rigid_body_names"], include_auxiliary=False)
    return AssetResetKinematics(raw, (directory / "model_spec.json").read_bytes(), binding), binding


def loop_closure_and_joint_pose(view, motion, kinematics, binding, samples=20000):
    core_bodies = view.bodies(binding.native_body_names)
    tree_joints = view.joints(binding.tree_joint_names)
    pose = view.get("body_link_pose_world_xyzw")[..., core_bodies, :][view.valid]   # M,K,19,7
    joints = view.get("joint_position_continuous")[..., tree_joints][view.valid]    # M,K,18
    pose, joints = pose.flatten(0, 1), joints.flatten(0, 1)
    if len(pose) > samples:
        chosen = torch.randperm(len(pose), generator=torch.Generator().manual_seed(0))[:samples]
        pose, joints = pose[chosen], joints[chosen]
    gaps = kinematics.closure_from_poses(pose).norm(dim=-1)                     # M,6
    worst = gaps.amax(-1)
    fk = kinematics.forward(joints)
    base = binding.native_body_names.index("base_link")
    r_base = so3.quaternion_to_matrix(pose[:, base, 3:])
    relative = (r_base.transpose(-1, -2)[:, None] @ (pose[..., :3] - pose[:, base:base + 1, :3])[..., None]).squeeze(-1)
    position_error = (relative - fk.body_position_m).norm(dim=-1)
    rotation_error = so3.angle((r_base.transpose(-1, -2)[:, None] @ so3.quaternion_to_matrix(pose[..., 3:]))
                               .transpose(-1, -2) @ fk.body_rotation)
    quantiles = torch.tensor([.5, .99, .999], dtype=torch.float64)
    return {"samples": len(pose),
            "worst_anchor_gap_mm_quantiles_p50_p99_p999": (worst.quantile(quantiles) * 1e3).tolist(),
            "fraction_worst_gap_over_1mm": float((worst > 1e-3).double().mean()),
            "loop_anchor_gap_mm": {name: {"rms": float(gaps[:, i].square().mean().sqrt() * 1e3),
                                          "max": float(gaps[:, i].max() * 1e3)}
                                   for i, name in enumerate(kinematics.constraint_names)},
            "joint_fk_position_error_mm": {name: float(position_error[:, i].max() * 1e3)
                                           for i, name in enumerate(binding.native_body_names)},
            "joint_fk_rotation_error_rad": {name: float(rotation_error[:, i].max())
                                            for i, name in enumerate(binding.native_body_names)},
            # Worst body per sample: tree-joint consistency between joint readback and body poses.
            "joint_fk_worst_position_mm_p50_p99_p999": (position_error.amax(-1).quantile(quantiles) * 1e3).tolist(),
            "joint_fk_worst_rotation_rad_p50_p99_p999": rotation_error.amax(-1).quantile(quantiles).tolist()}


def loop_gap_conditions(view, kinematics, binding):
    """Where do large closure gaps occur? Conditioned on the whole interval."""
    core = view.bodies(binding.native_body_names)
    pose = view.get("body_link_pose_world_xyzw")[..., core, :]
    T, N, K = pose.shape[:3]
    gaps = kinematics.closure_from_poses(pose.reshape(-1, len(core), 7)).norm(dim=-1).amax(-1).reshape(T, N, K)
    worst = gaps.amax(-1)                                                   # T,N
    weight = view.robot_weight_n()
    impact = view.get("wheel_support_load_n").sum(-1).amax(-1) / weight > 1.5
    joint_speed = view.get("joint_velocity_native").abs().amax(-1).amax(-1)
    contacts = (view.get("body_normal_load_n") > 1.).sum(-1).amax(-1)
    fresh = ~view.follows_previous()
    valid = view.valid
    groups = {"all": valid, "first_interval_of_segment": valid & fresh,
              "impact_peak_over_1p5_weight": valid & impact,
              "joint_speed_over_20_rad_s": valid & (joint_speed > 20),
              "five_or_more_bodies_in_contact": valid & (contacts >= 5),
              "calm": valid & ~fresh & ~impact & (joint_speed < 10)}
    result = {}
    for name, mask in groups.items():
        values = worst[mask]
        result[name] = {"intervals": int(mask.sum()),
                        "worst_gap_mm_p50_p99_max": None if not len(values) else
                        [float(values.median() * 1e3), float(values.quantile(.99) * 1e3), float(values.max() * 1e3)],
                        "fraction_over_1mm": None if not len(values) else float((values > 1e-3).double().mean())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--asset-directory", type=Path)
    args = parser.parse_args()
    view = SubstepView.from_replay_payload(torch.load(args.replay, map_location="cpu", weights_only=True))
    motion = BodyMotion(view)
    report = {"replay": str(args.replay), "transitions": int(view.valid.numel()),
              "body_velocity_convention": body_velocity_convention(view, motion),
              "angular_velocity_convention_rad": angular_velocity_convention(view, motion),
              "free_flight_momentum": free_flight_momentum(view, motion),
              "momentum_balance": momentum_balance(view, motion),
              "loop_gap_conditions": None,
              "static_support": {c: static_support(view, motion, c)
                                 for c in ("reported_is_com", "reported_is_link_origin")}}
    if args.upstream_root is not None:
        kinematics, binding = upstream_kinematics(args.upstream_root, args.asset_directory)
        report["closed_chain"] = loop_closure_and_joint_pose(view, motion, kinematics, binding)
        report["loop_gap_conditions"] = loop_gap_conditions(view, kinematics, binding)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
