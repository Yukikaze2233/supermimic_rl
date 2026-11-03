#!/usr/bin/env python3
"""M0 measurements on recorded substep truth; reads a replay, never a simulator.

1. Integration consistency: does Kamino advance pose and joints from the
   post-update velocity (semi-implicit Euler)?
2. Interval events: how often contact appears or vanishes inside 20 ms, i.e.
   what an endpoint-only view would miss.
3. Endpoint interpolation: error of the per-tick quantities analytic modules
   read when only the two 20 ms endpoints are known, split by regime. This is
   an approximation study, not a parity test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from supermimic_rl.contracts import ACTION_BOUNDS
from supermimic_rl.world_model_v2 import so3
from supermimic_rl.world_model_v2.substep_view import SubstepView

LEG_JOINTS = ("L_joint1", "LL_joint1", "R_joint1", "RR_joint1")
WHEEL_JOINTS = ("L_joint3", "R_joint3")
GIMBAL_JOINTS = ("M_joint1", "M_joint2")
FORBIDDEN_BODIES = ("base_link", "M_link1", "M_link2")
WHEEL_LOADED_N, FORBIDDEN_LOAD_N = 5., 1.
LEG_EFFORT_LIMIT_NM, WHEEL_EFFORT_LIMIT_NM = 40., 4.5
UPRIGHT_TILT_RAD = 0.35


def summary(error):
    error = error.reshape(-1, error.shape[-1]) if error.ndim > 1 else error.reshape(-1, 1)
    if not len(error):
        return {"count": 0}
    return {"count": int(error.shape[0]), "rms": error.square().mean(0).sqrt().tolist(),
            "bias": error.mean(0).tolist(), "max_abs": error.abs().amax(0).tolist()}


def integration_consistency(view):
    """Residual of x_k - x_{k-1} against dt times the velocity at k (semi-implicit) or k-1."""
    dt, within = view.physics_dt_s, view.valid[..., None].expand(-1, -1, view.substeps - 1)
    after, before = slice(1, None), slice(None, -1)

    def residuals(name, position, velocity):
        delta = position[:, :, after] - position[:, :, before]
        return {f"{name}_semi_implicit": summary((delta - dt * velocity[:, :, after])[within]),
                f"{name}_explicit": summary((delta - dt * velocity[:, :, before])[within]),
                f"{name}_step": summary(delta[within])}

    rotation = so3.quaternion_to_matrix(view.get("root_link_orientation_xyzw"))
    origin_velocity = (rotation @ view.get("root_origin_velocity_body_m_s")[..., None]).squeeze(-1)
    result = residuals("link_origin_position_m", view.get("root_link_position_world_m"), origin_velocity)
    if view.has("root_com_position_world_m"):
        result.update(residuals("com_position_m", view.get("root_com_position_world_m"),
                                view.get("root_com_velocity_world_m_s")))
    omega = view.get("root_angular_velocity_body_rad_s")
    for name, index in (("semi_implicit", after), ("explicit", before)):
        predicted = rotation[:, :, before] @ so3.exp(dt * omega[:, :, index])
        error = so3.angle(predicted.transpose(-1, -2) @ rotation[:, :, after])
        result[f"orientation_{name}_rad"] = summary(error[within][:, None])
    q, dq = view.get("joint_position_continuous"), view.get("joint_velocity_native")
    for group, names in (("leg", LEG_JOINTS), ("wheel", WHEEL_JOINTS), ("gimbal", GIMBAL_JOINTS)):
        index = view.joints(names)
        result.update(residuals(f"{group}_joint_rad", q[..., index], dq[..., index]))
    return result


def contact_flags(view):
    loaded = view.get("wheel_support_load_n") > WHEEL_LOADED_N
    touching = view.get("body_normal_load_n")[..., view.bodies(FORBIDDEN_BODIES)] > FORBIDDEN_LOAD_N
    return loaded, touching


def interval_events(view):
    valid = view.valid
    loaded, touching = contact_flags(view)
    rate = lambda flag: float(flag[valid].float().mean())
    result = {
        "wheel_contact_changes_inside_interval": rate(loaded.any(2) != loaded.all(2)),
        "forbidden_contact_any_tick": rate(touching.any(2)),
        "forbidden_contact_inside_but_not_endpoint": rate(touching.any(2) & ~touching[:, :, -1]),
        "forbidden_contact_endpoint_only": rate(touching[:, :, -1] & ~touching[:, :, :-1].any(2)),
        "airborne_any_tick": rate((~loaded).all(-1).any(-1)),
    }
    weight = view.robot_weight_n()
    if weight is not None:
        load = view.get("wheel_support_load_n").sum(-1)
        peak = load.amax(-1) / weight
        result["wheel_load_peak_over_weight"] = summary(peak[valid][:, None])
        result["peak_minus_endpoint_over_weight"] = summary((peak - load[:, :, -1] / weight)[valid][:, None])
    return result


def regimes(view):
    """Masks over intervals with a known start (the previous interval's last tick)."""
    loaded, touching = contact_flags(view)
    effort = view.get("applied_effort_nm").abs()
    weight = view.robot_weight_n()
    actions = view.storage["actions"].double()
    bounds = actions.new_tensor(ACTION_BOUNDS)
    change = torch.zeros(view.steps, view.num_envs, dtype=actions.dtype)
    change[1:] = ((actions[1:] - actions[:-1]).abs() / bounds).amax(-1)
    impact = (torch.zeros_like(change, dtype=torch.bool) if weight is None
              else view.get("wheel_support_load_n").sum(-1).amax(-1) / weight > 1.5)
    airborne = (~loaded).all(-1).any(-1) & loaded.any(-1).any(-1)
    saturated = ((effort[..., :4] >= LEG_EFFORT_LIMIT_NM - .5).any(-1).any(-1)
                 | (effort[..., 4:] >= WHEEL_EFFORT_LIMIT_NM - .05).any(-1).any(-1))
    contact = touching.any(-1).any(-1)
    nominal = ~(impact | airborne | saturated | contact)
    # Closest proxy for controlled locomotion: policy-owned, motors acting,
    # both wheels supported, upright, no forbidden contact in any tick.
    upright = view.get("gravity_body")[..., 2] < -torch.cos(torch.tensor(UPRIGHT_TILT_RAD, dtype=torch.float64))
    controlled = (view.storage["policy_owned"].bool() & (effort[..., :4].amax(-1) > .5).all(-1)
                  & loaded.all(-1).all(-1) & upright.all(-1) & ~contact)
    known = view.follows_previous()
    masks = {"impact": impact, "airborne_transition": airborne, "saturated_effort": saturated,
             "forbidden_contact": contact, "nominal": nominal,
             "nominal_action_change_lt_5pct": nominal & (change < .05),
             "nominal_action_change_gt_30pct": nominal & (change > .3),
             "controlled_upright": controlled & ~impact & ~saturated,
             "all": torch.ones_like(nominal)}
    return {name: mask & known for name, mask in masks.items()}


def interpolation_errors(view, masks):
    """Linear interpolation between the previous and current last ticks, minus the truth."""
    k = view.substeps
    fractions = torch.arange(1, k, dtype=torch.float64) / k
    q, dq = view.get("joint_position_continuous"), view.get("joint_velocity_native")
    quantities = {
        "root_origin_velocity_body_m_s": view.get("root_origin_velocity_body_m_s"),
        "root_angular_velocity_body_rad_s": view.get("root_angular_velocity_body_rad_s"),
        "gravity_body": view.get("gravity_body"),
        "leg_position_rad": q[..., view.joints(LEG_JOINTS)],
        "leg_velocity_rad_s": dq[..., view.joints(LEG_JOINTS)],
        "wheel_velocity_rad_s": dq[..., view.joints(WHEEL_JOINTS)],
        "gimbal_velocity_rad_s": dq[..., view.joints(GIMBAL_JOINTS)],
        "applied_effort_nm": view.get("applied_effort_nm"),
        "wheel_support_load_n": view.get("wheel_support_load_n"),
    }
    result = {}
    for name, value in quantities.items():
        start, end = value[:-1, :, -1], value[1:, :, -1]
        shape = (1, 1, k - 1) + (1,) * (value.ndim - 3)
        guess = start[:, :, None] + fractions.reshape(shape) * (end - start)[:, :, None]
        error = guess - value[1:, :, :-1]
        result[name] = {regime: summary(error[mask[1:]]) for regime, mask in masks.items()}
    return result


def analyze(path):
    view = SubstepView.from_replay_payload(torch.load(path, map_location="cpu", weights_only=True))
    masks = regimes(view)
    return {"replay": str(path), "transitions": int(view.valid.numel()), "valid": int(view.valid.sum()),
            "substeps_per_step": view.substeps, "physics_dt_s": view.physics_dt_s,
            "integration_consistency": integration_consistency(view),
            "interval_events": interval_events(view),
            "regime_counts": {name: int(mask.sum()) for name, mask in masks.items()},
            "endpoint_interpolation": interpolation_errors(view, masks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(analyze(args.replay), indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
