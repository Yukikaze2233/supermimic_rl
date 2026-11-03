"""Bind the packed course snapshot schema to the v2 physical state and conditions.

The native snapshot keeps every controller, sensor and task memory. This layout
reads the fields v2 needs by name, once per schema, so no other module depends
on packed offsets: physical state, motion-relevant memories, the sim context
xi, and per-substep supervision labels.
"""
from __future__ import annotations

import re

import torch
from torch import Tensor

from ..contracts import StateSchema
from . import so3
from .physical import POSITION_JOINTS, VELOCITY_JOINTS, PhysicalState, increments
from .records import (CONTACT_BODIES, FORBIDDEN_BODIES, RESULTANT_BODIES, PASSIVE_KNEE_JOINTS,
                      Memory, encode_memory, encode_physical, encode_queues)
from .sensing import SensorQueues

def _names(frame: str, kind: str) -> tuple[str, ...]:
    match = re.search(rf"native {kind} order: ([^;]+)", frame)
    if match is None:
        raise ValueError(f"Field frame does not declare its native {kind} order")
    return tuple(name.strip() for name in match.group(1).split(","))


class _Fields:
    def __init__(self, fields):
        self.slices, offset = {}, 0
        for field in fields:
            self.slices[field.name] = (slice(offset, offset + field.size), field.shape)
            offset += field.size

    def __call__(self, tensor: Tensor, name: str) -> Tensor:
        index, shape = self.slices[name]
        return tensor[..., index].reshape(*tensor.shape[:-1], *shape)

    def flat(self, tensor: Tensor, names) -> Tensor:
        """Concatenate the packed columns of ``names`` without reshaping."""
        return torch.cat([tensor[..., self.slices[name][0]] for name in names], -1)

    def __contains__(self, name):
        return name in self.slices


class SnapshotLayout:
    MEMORY_FIELDS = ("identified.wheel_torque_history_nm", "control._previous_action",
                     "gimbal.controller.target_angles.pitch_rad", "gimbal.controller.target_angles.yaw_sin_cos",
                     "gimbal.controller.yaw_rate_target", "gimbal.controller.mode")
    CONTEXT_FIELDS = ("body_mass_kg", "requested.motor_strength", "requested.pd_encoder_bias_rad",
                      "wheel_material", "identified.wheel_total_inertia_kg_m2")

    def __init__(self, schema: StateSchema, *, record_terrain: bool = False):
        if not schema.substep_fields:
            raise ValueError("v2 requires recorded substep truth (record_substeps=true)")
        self.schema = schema
        self.state, self.context = _Fields(schema.state_fields), _Fields(schema.context_fields)
        self.record_terrain = record_terrain
        if record_terrain and any(name not in self.state for name in ('course.terrain_tile', 'course.tasks.family')):
            raise ValueError('Terrain records require the actual native tile assignment and task family')
        self.substep = _Fields(schema.substep_fields)
        frames = {field.name: field.frame for field in schema.state_fields}
        self.joint_names = _names(frames["joint_velocity_native"], "joint")
        substep_frames = {field.name: field.frame for field in schema.substep_fields}
        self.body_names = _names(substep_frames["body_normal_load_n"], "body")
        self._position_columns = [self.joint_names.index(name) for name in POSITION_JOINTS[:4]]
        self._velocity_columns = [self.joint_names.index(name) for name in VELOCITY_JOINTS]
        self._substep_position = [self.joint_names.index(name) for name in POSITION_JOINTS]
        self._base = self.body_names.index("base_link")
        self._forbidden = [self.body_names.index(name) for name in FORBIDDEN_BODIES]
        self._contact_bodies = [self.body_names.index(name) for name in CONTACT_BODIES]
        self._resultants = [self.body_names.index(name) for name in RESULTANT_BODIES]
        self._knees = [self.joint_names.index(name) for name in PASSIVE_KNEE_JOINTS]
        missing = [name for name in self.MEMORY_FIELDS if name not in self.state]
        missing += [name for name in self.CONTEXT_FIELDS if name not in self.context]
        missing += [name for name in ("body_inertia_kg_m2", "body_com_pose_local_xyzw") if name not in self.context]
        if missing:
            raise ValueError("Snapshot schema lacks v2 fields: " + ", ".join(missing))

    # -- physical state --------------------------------------------------------------
    def physical(self, state: Tensor) -> PhysicalState:
        get = lambda name: self.state(state, name)
        rotation = so3.quaternion_to_matrix(get("root_orientation_xyzw"))
        rotate = lambda vector: (rotation @ vector[..., None]).squeeze(-1)
        phase = get("gimbal.yaw_sin_cos")
        joints = get("joint_position_gimbal_yaw_sine")[..., self._position_columns]
        position = torch.cat((joints, torch.atan2(phase[..., :1], phase[..., 1:]),
                              get("gimbal.pitch_position_rad")), -1)
        return PhysicalState(get("course.root_position_local_m"), rotation,
                             rotate(get("root_linear_velocity_body_m_s")),
                             rotate(get("root_angular_velocity_body_rad_s")),
                             position, get("joint_velocity_native")[..., self._velocity_columns])

    def height_above_ground(self, state: Tensor) -> Tensor:
        return self.state(state, "root_height_above_ground_m")

    def terrain_scan(self, state: Tensor) -> Tensor:
        return self.state(state, "course.terrain_heights_relative_m")

    # -- conditions ------------------------------------------------------------------
    def memory(self, state: Tensor) -> Memory:
        get = lambda name: self.state(state, name)
        phase = get("gimbal.controller.target_angles.yaw_sin_cos")
        return Memory(get("identified.wheel_torque_history_nm"), get("control._previous_action"),
                      get("gimbal.controller.mode").round().long(),
                      get("gimbal.controller.target_angles.pitch_rad")[..., 0],
                      torch.atan2(phase[..., 0], phase[..., 1]), get("gimbal.controller.yaw_rate_target"))

    def motor_strength(self, context: Tensor) -> Tensor:
        return self.context(context, "requested.motor_strength")

    def sim_context(self, context: Tensor) -> Tensor:
        base_com = self.context(context, "body_com_pose_local_xyzw")[..., self._base, :3]
        inertia = self.context(context, "body_inertia_kg_m2").unflatten(-1, (3, 3))
        # The pinned startup DR scales each authored tensor by one independent
        # positive scalar (in addition to mass). Trace records that scalar's
        # actual result without depending on PhysX's choice of principal axes.
        # Log units expose relative changes even for the tiny linkage inertias.
        # A future tensor-shape/orientation DR needs a new condition contract.
        inertia_trace = inertia.diagonal(dim1=-2, dim2=-1).sum(-1)
        return torch.cat((self.context.flat(context, self.CONTEXT_FIELDS), base_com, inertia_trace.log()), -1)

    # -- substep supervision --------------------------------------------------------
    def tick_labels(self, substeps: Tensor, start: PhysicalState, context: Tensor) -> Tensor:
        """``[..., K, TICK_LABEL_DIM]`` per-tick quantities relative to the start state.

        Full physical substeps supply sensors, execution, reward and termination.
        All increments share the start frame; time integration is not autoregressive
        within the 20 ms interval.
        """
        get = lambda name: self.substep(substeps, name)
        q = get("joint_position_continuous")
        rotation = so3.quaternion_to_matrix(get("root_link_orientation_xyzw"))
        rotate = lambda value: (rotation @ value[..., None]).squeeze(-1)
        origin = self.context(context, "environment_origin_world_m")[..., None, :]
        states = PhysicalState(get("root_link_position_world_m") - origin, rotation,
                               rotate(get("root_origin_velocity_body_m_s")),
                               rotate(get("root_angular_velocity_body_rad_s")),
                               q[..., self._substep_position],
                               get("joint_velocity_native")[..., self._velocity_columns])
        return torch.cat((increments(start.with_time_axis(), states), get("wheel_support_load_n"),
            get("body_normal_load_n")[..., self._contact_bodies],
            get("terrain_contact_force_world_n")[..., self._resultants, :].norm(dim=-1),
            q[..., self._knees], get("joint_velocity_native")[..., self._knees]), -1)

    # -- compact records ------------------------------------------------------------
    def encode(self, transition) -> dict[str, Tensor]:
        """One packed ``Transition`` -> the compact v2 record (all float32 ``[N, ...]``)."""
        start, end = transition.start, transition.end
        state, context = start.state, start.context
        physical = self.physical(state)
        get = lambda name: self.state(state, name)
        ctx = lambda name: self.context(context, name)
        if "sensor.uplink_delay_s" in self.context:
            raise ValueError("Identified execution owns aggregate I/O; extra sensor uplink is unsupported")
        queues = SensorQueues(get("sensor.values"), get("sensor.gimbal_history"), get("sensor.heading"),
                              get("sensor.count"))
        transforms = torch.cat((ctx("requested.policy_encoder_bias_rad"), ctx("requested.gyro_bias_rad_s"),
                                ctx("requested.imu_sensor_to_body_rotation"), ctx("gimbal.control_joint_signs"),
                                ctx("gimbal.joint_zero_rad")), -1)
        distribution = (ctx("sensor.noise_distribution") if "sensor.noise_distribution" in self.context
                        else torch.zeros_like(ctx("sensor.nominal")))
        readout = torch.cat((ctx("sensor.lags"), distribution.reshape(-1, 1), ctx("sensor.noise_halfwidth"),
                             ctx("sensor.nominal").reshape(-1, 1), ctx("sensor.noise_scale").reshape(-1, 1),
                             ctx("controller.nominal")), -1)
        dynamic = get("course.dynamic_goal").reshape(-1, 1).to(state.dtype)
        horizon = torch.cat((get("course.imagination_remaining_s").reshape(-1, 1), dynamic), -1)
        estimator = (start.estimator_target if start.estimator_target is not None
                     else state.new_zeros(state.shape[0], 5))
        estimator_valid = (start.estimator_valid if start.estimator_valid is not None
                           else torch.zeros(state.shape[0], dtype=torch.bool, device=state.device))
        # Legacy recordings have no contact-exception flag. They can provide
        # motion labels, but their task semantics are not an imagination seed.
        has_conditions = all(name in self.state for name in ("course.body_contact_allowed", "course.command_mode"))
        allowed = (get("course.body_contact_allowed") if has_conditions
                   else torch.zeros_like(get("failure.contact_time_s")))
        mode = (get("course.command_mode") if "course.command_mode" in self.state
                else get("course.tasks.base_mode"))
        failure_conditions = torch.stack((mode == 3, allowed), -1).to(state.dtype)
        reward_fields = ('reward._previous', 'reward._previous2', 'reward._action_count')
        support_fields = tuple('course.support._' + name for name in (
            'contact', 'airborne', 'unsupported_s', 'landing_remaining_s', 'phase_active',
            'phase_elapsed_s', 'wheel_contact_s', 'base_contact_s'))
        controller_fields = tuple('controller.' + name for name in ('nominal', 'stiffness', 'damping', 'effort_bounds'))
        controller_fields += ('requested.pd_encoder_bias_rad',)
        has_reward = all(name in self.state for name in reward_fields + support_fields)
        has_controller = all(name in self.context for name in controller_fields)
        route_fields = ('course.command_mode', 'course.tasks.terrain_kind', 'course.gimbal.special')
        has_route = all(name in self.state for name in route_fields)
        record = {
            "physical": encode_physical(physical), "height": self.height_above_ground(state),
            "scan": self.terrain_scan(state), "memory": encode_memory(self.memory(state)),
            "queues": encode_queues(queues), "goal": start.goal, "sim_context": self.sim_context(context),
            "motor_strength": self.motor_strength(context), "transforms": transforms, "readout": readout,
            "horizon": horizon, "frame": start.actor_observation, "estimator_target": estimator,
            "estimator_valid": estimator_valid.to(state.dtype)[:, None],
            "end_physical": encode_physical(self.physical(end.state)),
            "end_height": self.height_above_ground(end.state), "action": transition.action,
            "reward": transition.reward[:, None], "ticks": self.tick_labels(transition.substeps, physical, context),
            "failure_memory": self.state.flat(state, ("failure.contact_time_s", "failure.pose_time_s",
                                                       "failure.failure_reasons")),
            "failure_conditions": failure_conditions,
            "failure_conditions_valid": state.new_full((state.shape[0], 1), float(has_conditions)),
            "reward_memory": (self.state.flat(state, reward_fields) if has_reward else state.new_zeros(state.shape[0], 13)),
            "support_memory": (self.state.flat(state, support_fields) if has_reward else state.new_zeros(state.shape[0], 10)),
            "controller_parameters": (self.context.flat(context, controller_fields) if has_controller
                                      else state.new_zeros(state.shape[0], 30)),
            "reward_context_valid": state.new_full((state.shape[0], 1), float(has_reward and has_controller and has_route)),
            "reward_route": self.state.flat(state, route_fields) if has_route else state.new_full((state.shape[0], 3), -1.),
        }
        if self.record_terrain:
            record['terrain_tile'] = get('course.terrain_tile').reshape(-1, 1)
            record['task_family'] = get('course.tasks.family').reshape(-1, 1)
        return record
