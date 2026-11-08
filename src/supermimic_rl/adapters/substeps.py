"""Physics-tick truth inside one policy interval for the v2 world model.

The pinned environment calls ``_observe_substep`` exactly once after each of
its ``decimation`` physics steps. Sensor pushes, reward accumulation and
termination latches read the sample produced there, so the rows recorded here
are exactly the substep states those analytic modules consume. Recording
happens before any auto-reset at the end of the interval; the last row is the
pre-reset physical endpoint.
"""
from __future__ import annotations

from types import MethodType

import torch

from ..contracts import Field


def substep_fields(body_names, joint_names, *, audit=False, disturbances=False) -> tuple[Field, ...]:
    """Measurement superset for M0; the model consumes a validated subset.

    ``audit`` adds every body's link pose and the backend-reported body velocity,
    so velocity/frame conventions can be verified against finite differences.
    """
    body_order = "native body order: " + ",".join(body_names)
    joint_order = "native joint order: " + ",".join(joint_names)
    body_count, joint_count = len(body_names), len(joint_names)
    audit_fields = (
        Field("body_link_pose_world_xyzw", (body_count, 7), "m,1", "link origin, quaternion xyzw; world; " + body_order),
        Field("body_com_velocity_reported_world", (body_count, 6), "m/s,rad/s",
              "backend body_com_vel_w as returned, linear then angular; world; " + body_order),
    ) if audit else ()
    from .disturbances import DISTURBANCE_FIELDS

    return (DISTURBANCE_FIELDS if disturbances else ()) + audit_fields + (
        Field("root_link_position_world_m", (3,), "m", "base_link origin; world"),
        Field("root_link_orientation_xyzw", (4,), "1", "base_link orientation in world; quaternion xyzw"),
        Field("root_origin_velocity_body_m_s", (3,), "m/s", "base_link origin; base_link axes"),
        Field("root_angular_velocity_body_rad_s", (3,), "rad/s", "base_link axes"),
        # The simulator integrates the COM; link-origin motion is derived from it.
        Field("root_com_position_world_m", (3,), "m", "base_link COM; world"),
        Field("root_com_velocity_world_m_s", (3,), "m/s", "base_link COM; world"),
        Field("root_angular_velocity_world_rad_s", (3,), "rad/s", "base_link; world axes"),
        Field("gravity_body", (3,), "1", "unit gravity direction; base_link axes"),
        Field("joint_position_continuous", (joint_count,), "rad or m",
              joint_order + "; continuous coordinates"),
        Field("joint_velocity_native", (joint_count,), "rad/s or m/s", joint_order),
        Field("height_above_ground_m", (1,), "m", "base_link origin above local ground; world Z"),
        Field("ground_at_root_world_z_m", (1,), "m", "world Z"),
        Field("ground_at_wheel_world_z_m", (2,), "m", "world Z; left, right wheel"),
        Field("body_normal_load_n", (body_count,), "N", "scalar terrain normal load; " + body_order),
        Field("wheel_support_load_n", (2,), "N", "upward terrain support; left, right wheel"),
        Field("terrain_contact_force_world_n", (body_count, 3), "N",
              "net terrain force; world; " + body_order),
        Field("applied_effort_nm", (6,), "N*m", "policy-axis effort after the actuator chain; P order"),
        Field("contact_measurement_valid", (1,), "1", "1 when this tick's contact reduction is valid"),
        # The command actually executed in this interval. The endpoint goal may
        # already belong to the next segment, so rewards must not read end.goal.
        Field("command_admitted_body", (4,), "m/s,m/s,rad/s,m",
              "admitted forward, lateral, yaw-rate and height command during this tick"),
        Field("command_mode", (1,), "1", "upstream CommandMode index during this tick"),
        Field("request_height_m", (1,), "m", "admitted jump request height during this tick"),
        Field("termination_height_m", (1,), "m", "actual height supplied to native failure latch"),
        Field("body_contact_allowed", (1,), "1", "intentional base-contact exception supplied to latch"),
        Field("failure_context_valid", (1,), "1", "native failure latch was present"),
        Field("failure_contact_time_s", (1,), "s", "native consecutive forbidden-contact duration"),
        Field("failure_pose_time_s", (1,), "s", "native consecutive failed-pose duration"),
        Field("failure_reasons", (1,), "1", "native latched failure reason bits"),
    )


class SubstepRecorder:
    """Copy every physics tick of an adapter-owned step into preallocated rows."""

    def __init__(self, env, *, audit=False, disturbances=False):
        self.env, self.audit = env, audit
        if type(disturbances) is not bool:
            raise ValueError("disturbances must be boolean")
        from .disturbances import DisturbanceRecorder

        self.disturbances = DisturbanceRecorder(env) if disturbances else None
        self.count = int(env.cfg.decimation)
        if self.count < 1:
            raise ValueError("Substep recording requires a positive decimation")
        self.num_envs, self.device = env.num_envs, torch.device(env.device)
        self.fields = substep_fields(tuple(env._binding.native_body_names),
                                     tuple(env._binding.native_joint_names), audit=audit,
                                     disturbances=disturbances)
        self._values = {field.name: torch.zeros(self.num_envs, self.count, *field.shape,
                                                dtype=torch.float32, device=self.device)
                        for field in self.fields}
        # Clock metadata stays float64 and never enters the float32 learning rows.
        self._timestamps = torch.zeros(self.num_envs, self.count, dtype=torch.float64, device=self.device)
        self._index = None
        self._original = self._had_instance_method = None
        self._installed = False

    def install(self):
        env = self.env
        if self._installed or getattr(env, "_supermimic_substep_recorder", None) is not None:
            raise RuntimeError("The native environment already has a substep recorder")
        if self.disturbances is not None:
            self.disturbances.install()
        original = env._observe_substep
        self._original = original
        self._had_instance_method = "_observe_substep" in vars(env)
        recorder = self

        def observe_then_record(owner, *args, **kwargs):
            result = original(*args, **kwargs)
            recorder._record(owner)
            return result

        env._observe_substep = MethodType(observe_then_record, env)
        env._supermimic_substep_recorder = self
        self._installed = True
        return self

    def begin(self):
        if not self._installed:
            raise RuntimeError("Install the substep recorder before stepping")
        self._index = 0
        if self.disturbances is not None:
            self.disturbances.begin()

    def abort(self):
        self._index = None

    @torch.no_grad()
    def _record(self, env):
        if self._index is None:
            return
        if self._index >= self.count:
            raise RuntimeError("The native step ran more physics substeps than its decimation")
        sample, state = env._last_sample, env._last_state
        for name in ("body_normal_load_n", "wheel_support_load_n", "contact_measurement_valid"):
            if not hasattr(sample, name):
                raise ValueError("Substep truth requires the Lab contact measurement sample: " + name)
        n = self.num_envs
        rows = {
            "root_link_position_world_m": sample.root_link_position_world_m,
            "root_link_orientation_xyzw": sample.root_link_orientation_xyzw,
            "root_origin_velocity_body_m_s": state.velocity_origin_body_m_s,
            "root_angular_velocity_body_rad_s": state.angular_velocity_body_rad_s,
            "root_com_position_world_m": sample.root_com_position_world_m,
            "root_com_velocity_world_m_s": sample.root_com_velocity_world_m_s,
            "root_angular_velocity_world_rad_s": sample.angular_velocity_world_rad_s,
            "gravity_body": state.gravity_body,
            "joint_position_continuous": sample.joint_position_rad,
            "joint_velocity_native": sample.joint_velocity_rad_s,
            "height_above_ground_m": state.height_above_ground_m.reshape(n, 1),
            "ground_at_root_world_z_m": sample.ground_at_root_world_z_m.reshape(n, 1),
            "ground_at_wheel_world_z_m": sample.ground_at_wheel_world_z_m,
            "body_normal_load_n": sample.body_normal_load_n,
            "wheel_support_load_n": sample.wheel_support_load_n,
            "terrain_contact_force_world_n": sample.terrain_contact_force_world_n,
            "applied_effort_nm": env._last_effort,
            "contact_measurement_valid": sample.contact_measurement_valid.reshape(n, 1),
            "command_admitted_body": env._command,
            "command_mode": env._command_mode.reshape(n, 1),
            "request_height_m": env._request_height.reshape(n, 1),
        }
        course = getattr(env, "_terrain_course", None)
        latch = getattr(env, "_failure_latch", None)
        zero = state.height_above_ground_m.new_zeros(n, 1)
        rows.update({
            "termination_height_m": (state.height_above_ground_m if course is None
                                     else course.termination_height(state)).reshape(n, 1),
            "body_contact_allowed": zero if course is None else course.body_contact_allowed.reshape(n, 1),
            "failure_context_valid": torch.full_like(zero, float(latch is not None)),
            "failure_contact_time_s": zero if latch is None else latch.contact_time_s.reshape(n, 1),
            "failure_pose_time_s": zero if latch is None else latch.pose_time_s.reshape(n, 1),
            "failure_reasons": zero if latch is None else latch.failure_reasons.reshape(n, 1),
        })
        if self.audit:
            rows["body_link_pose_world_xyzw"] = env._as_torch(env.robot.data.body_link_pose_w)
            rows["body_com_velocity_reported_world"] = env._as_torch(env.robot.data.body_com_vel_w)
        if self.disturbances is not None:
            rows.update(self.disturbances.tick())
        k = self._index
        for field in self.fields:
            value = rows[field.name]
            if tuple(value.shape) != (n, *field.shape):
                raise ValueError(f"Native substep field {field.name} has shape {tuple(value.shape)}")
            self._values[field.name][:, k].copy_(value)
        self._timestamps[:, k].copy_(sample.timestamp_s.reshape(n))
        self._index = k + 1

    def finish(self):
        """Return owned ``[batch, substeps, ...]`` rows and their float64 clocks."""
        if self._index != self.count:
            recorded = self._index
            self._index = None
            raise RuntimeError(f"Recorded {recorded} of {self.count} physics substeps")
        self._index = None
        return ({name: value.clone() for name, value in self._values.items()},
                self._timestamps.clone())

    def close(self):
        if not self._installed:
            return
        env = self.env
        if self._had_instance_method:
            env._observe_substep = self._original
        else:
            del env.__dict__["_observe_substep"]
        del env.__dict__["_supermimic_substep_recorder"]
        if self.disturbances is not None:
            self.disturbances.close()
        self._installed = False
