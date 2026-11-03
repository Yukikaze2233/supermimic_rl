#!/usr/bin/env python3
"""M1 parity: recorded substeps through the analytic modules must reproduce the next memories.

For every valid, policy-owned transition that does not end its episode, the
start-snapshot memories advanced with the four recorded physics ticks must
equal the end-snapshot memories:
- sensing: encoder/IMU, gimbal and heading queues (sample, transforms, push);
- execution: the identified wheel delay queue, and the delayed motor torque at
  every tick against the recorded applied effort (outside the speed envelope).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from supermimic_rl.compact_state import named_field
from supermimic_rl.world_model_v2.execution import LegExecution, WheelExecution
from supermimic_rl.world_model_v2.sensing import (
    SensorQueues, SensorTransforms, encoder_imu_sample, gimbal_sample, heading_sample)
from supermimic_rl.world_model_v2.substep_view import SubstepView
from supermimic_rl.world_model_v2.termination import PoseContactLimits, PoseContactTermination

POLICY_JOINTS = ("L_joint1", "LL_joint1", "R_joint1", "RR_joint1", "L_joint3", "R_joint3")
GIMBAL_JOINTS = ("M_joint1", "M_joint2")


def yaw_of(quaternion_xyzw):
    x, y, z, w = quaternion_xyzw.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def check(path, bundle, runtime=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    view = SubstepView.from_replay_payload(payload)
    schema, data = view.schema, view.storage
    rows = view.valid & ~view.ended & data["policy_owned"].bool() & data["end_policy_owned"].bool()
    state0, state1 = data["start_state"][rows].double(), data["end_state"][rows].double()
    context = data["start_context"][rows].double()
    get = lambda name: view.get(name)[rows]
    field = lambda tensor, name, **kw: named_field(tensor, schema, name, **kw)
    transforms = SensorTransforms(
        field(context, "requested.policy_encoder_bias_rad", context=True),
        field(context, "requested.gyro_bias_rad_s", context=True),
        field(context, "requested.imu_sensor_to_body_rotation", context=True).reshape(-1, 3, 3),
        field(context, "gimbal.control_joint_signs", context=True),
        field(context, "gimbal.joint_zero_rad", context=True))
    queues = SensorQueues(field(state0, "sensor.values"), field(state0, "sensor.gimbal_history"),
                          field(state0, "sensor.heading"), field(state0, "sensor.count"))
    q, dq = get("joint_position_continuous"), get("joint_velocity_native")
    policy, gimbal = view.joints(POLICY_JOINTS), view.joints(GIMBAL_JOINTS)
    yaw = yaw_of(get("root_link_orientation_xyzw"))
    for tick in range(view.substeps):
        queues = queues.push(
            encoder_imu_sample(q[:, tick, policy[:4]], dq[:, tick, policy],
                               get("root_angular_velocity_body_rad_s")[:, tick],
                               get("gravity_body")[:, tick], transforms),
            gimbal_sample(q[:, tick, gimbal], dq[:, tick, gimbal], transforms),
            heading_sample(yaw[:, tick]))
    report = {"replay": str(path), "transitions_checked": int(rows.sum())}
    report.update(check_execution(view, rows, state0, state1, context, field, q, dq, policy, bundle))
    for name, predicted in (("sensor.values", queues.values), ("sensor.gimbal_history", queues.gimbal),
                            ("sensor.heading", queues.heading), ("sensor.count", queues.count)):
        error = (predicted - field(state1, name)).abs()
        report[name] = {"max_abs": float(error.max()),
                        "max_abs_by_channel": error.flatten(0, -2).amax(0).tolist() if error.ndim > 2 else None}
    if runtime is not None:
        report["termination"] = check_termination(view, runtime)
    return report


def check_termination(view, runtime):
    """Compare every recorded tick, including terminal intervals, before reset."""
    if not view.has("failure_context_valid"):
        raise ValueError("Collect native failure-clock labels before testing termination parity")
    rows = view.valid & view.get("failure_context_valid").bool().all((-2, -1))
    data, schema = view.storage, view.schema
    state = data["start_state"][rows].double()
    memory = torch.stack([named_field(state, schema, name) for name in
                          ("failure.contact_time_s", "failure.pose_time_s", "failure.failure_reasons")], -1)
    limits = PoseContactLimits.from_runtime(runtime)
    termination = PoseContactTermination(limits, view.physics_dt_s)
    height, gravity = view.get("termination_height_m")[rows], view.get("gravity_body")[rows]
    loads = view.get("body_normal_load_n")[rows][..., view.bodies(limits.forbidden_body_names)]
    recovery = view.get("command_mode")[rows][..., 0] == 3
    allowed = view.get("body_contact_allowed")[rows][..., 0] > .5
    error = memory.new_zeros(2)
    mismatch = torch.zeros((), dtype=torch.long)
    for tick in range(view.substeps):
        result = termination.advance(memory, height[:, tick:tick + 1], gravity[:, tick:tick + 1],
                                     loads[:, tick:tick + 1], recovery=recovery[:, tick],
                                     body_contact_allowed=allowed[:, tick])
        actual_time = torch.cat((view.get("failure_contact_time_s")[rows][:, tick],
                                 view.get("failure_pose_time_s")[rows][:, tick]), -1)
        actual_bits = view.get("failure_reasons")[rows][:, tick, 0].round().long()
        error = torch.maximum(error, (result.memory[:, :2] - actual_time).abs().amax(0))
        mismatch += ((result.memory[:, 2].long() & 7) != (actual_bits & 7)).sum()
        memory = result.memory
    return {"transitions_checked": int(rows.sum()), "max_abs": float(error.max()),
            "contact_time_max_abs_s": float(error[0]), "pose_time_max_abs_s": float(error[1]),
            "pose_contact_reason_mismatches": int(mismatch),
            "scope": "pose/contact clocks and bits 1/2/4; mechanical predicates require separate validation"}


def check_execution(view, rows, state0, state1, context, field, q, dq, policy, bundle):
    wheels = WheelExecution.from_bundle(bundle, view.physics_dt_s)
    history = field(state0, "identified.wheel_torque_history_nm")
    target = field(state1, "target.wheel_velocity_rad_s")
    strength = field(context, "requested.motor_strength", context=True)[:, 4:6]
    start_velocity = field(state0, "joint_velocity_native")[:, policy[4:]]
    applied = view.get("applied_effort_nm")[rows][..., 4:6]
    legs = LegExecution(*(field(context, "controller." + name, context=True) for name in
                          ("nominal", "stiffness", "damping", "effort_bounds")))
    bias = field(context, "requested.pd_encoder_bias_rad", context=True)
    full_strength = field(context, "requested.motor_strength", context=True)
    start_position = field(state0, "joint_position_gimbal_yaw_sine")[:, policy[:4]]
    leg_target = legs.target(view.storage["actions"][rows].double(), start_position + bias[:, :4])
    target_error = (leg_target - field(state1, "target.admitted_leg_position_rad")).abs().amax()
    leg_start_velocity = field(state0, "joint_velocity_native")[:, policy[:4]]
    delayed, recorded, envelope, leg_errors = [], [], [], []
    for tick in range(view.substeps):
        velocity = start_velocity if tick == 0 else dq[:, tick - 1, policy[4:]]
        history = wheels.push(history, wheels.motor_input(target, velocity, strength))
        delayed.append(wheels.delayed(history))
        envelope.append(wheels.applied(history, velocity))
        recorded.append(applied[:, tick])
        position = start_position if tick == 0 else q[:, tick - 1, policy[:4]]
        speed = leg_start_velocity if tick == 0 else dq[:, tick - 1, policy[:4]]
        leg_effort = legs.motor_effort(leg_target, position + bias[:, :4], speed, full_strength)
        leg_errors.append(leg_effort - view.get("applied_effort_nm")[rows][:, tick, :4])
    delayed, recorded = torch.stack(delayed, 1), torch.stack(recorded, 1)
    # The speed envelope may lower the applied torque below the delayed command.
    unclamped = recorded.abs() < delayed.abs() - 1e-4
    effort_error = (delayed - recorded).abs()[~unclamped]
    return {"identified.wheel_torque_history_nm": {
                "max_abs": float((history - field(state1, "identified.wheel_torque_history_nm")).abs().max())},
            "delayed_wheel_torque_vs_applied": {
                "max_abs_outside_envelope": float(effort_error.max()) if effort_error.numel() else None,
                "envelope_limited_fraction": float(unclamped.double().mean())},
            "wheel_effort_including_envelope_max_abs_nm": float((torch.stack(envelope, 1) - recorded).abs().max()),
            "admitted_leg_target_max_abs_rad": float(target_error),
            "leg_effort_max_abs_nm": float(torch.stack(leg_errors, 1).abs().max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--identified-bundle", type=Path, default=Path("configs/identified/20261031"))
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runtime = None if args.runtime is None else json.loads(args.runtime.read_text())
    report = check(args.replay, args.identified_bundle, runtime)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
