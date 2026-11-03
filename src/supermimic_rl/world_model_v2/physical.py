"""Physical state predicted by the v2 network, and its local-frame increments.

The network sees only position/heading-invariant features and predicts one
20 ms increment, following supermimic_vibed/Radiant:

    dp  = H0^T (p1 - p0)        yaw frame of the start pose
    dth = log(R0^T R1)          start body frame
    dv  = R0^T (v1 - v0)        start body frame, base_link origin velocity
    dw  = R0^T (w1 - w0)
    dq  = q1 - q0               active coordinates; gimbal yaw is wrapped
    dqd = qd1 - qd0

Active coordinates are the four coaxial hip motors, gimbal yaw and pitch;
velocities add both wheels. Passive closed-chain joints are not state: they
follow from ``leg_linkage``. Absolute position and heading stay outside the
network and are kept only for terrain lookup and world-heading targets.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor

from . import so3

LEG_JOINTS = ("L_joint1", "LL_joint1", "R_joint1", "RR_joint1")
WHEEL_JOINTS = ("L_joint3", "R_joint3")
GIMBAL_JOINTS = ("M_joint1", "M_joint2")
POSITION_JOINTS = LEG_JOINTS + GIMBAL_JOINTS            # 6
VELOCITY_JOINTS = LEG_JOINTS + WHEEL_JOINTS + GIMBAL_JOINTS  # 8
GIMBAL_YAW_COLUMN = POSITION_JOINTS.index("M_joint1")
INCREMENT_DIM = 3 + 3 + 3 + 3 + len(POSITION_JOINTS) + len(VELOCITY_JOINTS)


@dataclass(frozen=True)
class PhysicalState:
    position: Tensor          # [N, 3] base_link origin in the course tile frame
    rotation: Tensor          # [N, 3, 3] base orientation in world axes
    velocity: Tensor          # [N, 3] base_link origin linear velocity, world axes
    angular_velocity: Tensor  # [N, 3] world axes
    joint_position: Tensor    # [N, 6] POSITION_JOINTS; gimbal yaw wrapped to (-pi, pi]
    joint_velocity: Tensor    # [N, 8] VELOCITY_JOINTS

    def map(self, function) -> PhysicalState:
        return PhysicalState(*(function(getattr(self, f.name)) for f in fields(self)))

    def with_time_axis(self) -> PhysicalState:
        axis = self.position.ndim - 1
        return self.map(lambda value: value.unsqueeze(axis))

    def tick(self, index: int) -> PhysicalState:
        axis = self.position.ndim - 2
        return self.map(lambda value: value.select(axis, index))

    @property
    def gravity_body(self) -> Tensor:
        return -self.rotation[..., 2, :]

    @property
    def velocity_body(self) -> Tensor:
        return (self.rotation.transpose(-1, -2) @ self.velocity[..., None]).squeeze(-1)

    @property
    def angular_velocity_body(self) -> Tensor:
        return (self.rotation.transpose(-1, -2) @ self.angular_velocity[..., None]).squeeze(-1)


def increments(start: PhysicalState, end: PhysicalState) -> Tensor:
    """Regression target ``[N, INCREMENT_DIM]`` from ``start`` to ``end``."""
    heading = so3.yaw_frame(start.rotation)
    rotation_t = start.rotation.transpose(-1, -2)
    dq = end.joint_position - start.joint_position
    dq = torch.cat((dq[..., :GIMBAL_YAW_COLUMN], so3.wrap(dq[..., GIMBAL_YAW_COLUMN:GIMBAL_YAW_COLUMN + 1]),
                    dq[..., GIMBAL_YAW_COLUMN + 1:]), -1)
    return torch.cat((
        (heading.transpose(-1, -2) @ (end.position - start.position)[..., None]).squeeze(-1),
        so3.log(rotation_t @ end.rotation),
        (rotation_t @ (end.velocity - start.velocity)[..., None]).squeeze(-1),
        (rotation_t @ (end.angular_velocity - start.angular_velocity)[..., None]).squeeze(-1),
        dq, end.joint_velocity - start.joint_velocity), -1)


def apply(start: PhysicalState, increment: Tensor) -> PhysicalState:
    """Inverse of :func:`increments`; differentiable in both arguments."""
    dp, dth, dv, dw, dq, dqd = increment.split((3, 3, 3, 3, len(POSITION_JOINTS), len(VELOCITY_JOINTS)), -1)
    heading = so3.yaw_frame(start.rotation)
    q = start.joint_position + dq
    q = torch.cat((q[..., :GIMBAL_YAW_COLUMN], so3.wrap(q[..., GIMBAL_YAW_COLUMN:GIMBAL_YAW_COLUMN + 1]),
                   q[..., GIMBAL_YAW_COLUMN + 1:]), -1)
    return PhysicalState(
        start.position + (heading @ dp[..., None]).squeeze(-1),
        start.rotation @ so3.exp(dth),
        start.velocity + (start.rotation @ dv[..., None]).squeeze(-1),
        start.angular_velocity + (start.rotation @ dw[..., None]).squeeze(-1),
        q, start.joint_velocity + dqd)
