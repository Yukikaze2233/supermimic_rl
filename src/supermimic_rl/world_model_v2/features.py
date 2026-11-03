"""Heading- and position-invariant network inputs for the v2 motion model.

Everything here is a deterministic, differentiable function of the physical
state and code-advanced memories; nothing is predicted. Closed-chain geometry
comes from the 1D leg table, gimbal targets are expressed relative to the base
heading in world modes, and the terrain scan is relative to the base height.
"""
from __future__ import annotations

import torch
from torch import Tensor

from . import so3
from .records import Memory
from .leg_linkage import LegLinkage
from .physical import PhysicalState

GIMBAL_MODES = 3


def physical_features(state: PhysicalState, height_above_ground: Tensor) -> Tensor:
    """Height, tilt (6D), body-frame twist, hip angles, leg shapes, gimbal and all rates (30D)."""
    tilt = so3.to_6d(so3.yaw_frame(state.rotation).transpose(-1, -2) @ state.rotation)
    q = state.joint_position
    left_phi, right_phi = q[..., 1:2] - q[..., 0:1], q[..., 3:4] - q[..., 2:3]
    return torch.cat((height_above_ground, tilt, state.velocity_body, state.angular_velocity_body,
                      q[..., 0:1].sin(), q[..., 0:1].cos(), left_phi,
                      q[..., 2:3].sin(), q[..., 2:3].cos(), right_phi,
                      q[..., 4:5].sin(), q[..., 4:5].cos(), q[..., 5:6], state.joint_velocity), -1)


def linkage_features(linkage: LegLinkage, state: PhysicalState) -> tuple[Tensor, Tensor]:
    """Wheel centres in the base frame, spring slide, knee angle and domain flags (12D)."""
    q = state.joint_position
    columns, valid = [], []
    for side, (primary, secondary) in (("left", (0, 1)), ("right", (2, 3))):
        phi = q[..., secondary] - q[..., primary]
        geometry, inside = linkage.geometry_base(side, q[..., primary], phi)
        slide = geometry["passive_joint"][..., 4:5]
        columns += [geometry['wheel_center_base_m'], slide, geometry["knee_inner_angle_rad"]]
        valid.append(inside)
    inside = torch.stack(valid, -1)
    return torch.cat(columns + [inside.to(q.dtype)], -1), inside


def memory_features(memory: Memory, state: PhysicalState) -> Tensor:
    """Wheel queue, last action and gimbal targets relative to the base heading (19D)."""
    world = memory.gimbal_mode != 0
    relative_yaw = torch.where(world, so3.wrap(memory.gimbal_target_yaw - so3.yaw(state.rotation)),
                               memory.gimbal_target_yaw)
    mode = torch.nn.functional.one_hot(memory.gimbal_mode.clamp(0, GIMBAL_MODES - 1), GIMBAL_MODES)
    return torch.cat((memory.wheel_torque_history.flatten(-2), memory.previous_action, mode.to(relative_yaw.dtype),
                      memory.gimbal_target_pitch[..., None], relative_yaw.sin()[..., None],
                      relative_yaw.cos()[..., None], memory.gimbal_yaw_rate_target[..., None]), -1)


class MotionInputs:
    """Assemble the motion model input; widths are fixed by the layout and table."""

    def __init__(self, linkage: LegLinkage):
        self.linkage = linkage

    def __call__(self, state: PhysicalState, height_above_ground: Tensor, terrain_scan: Tensor,
                 memory: Memory, sim_context: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        linkage, inside = linkage_features(self.linkage, state)
        features = torch.cat((physical_features(state, height_above_ground), linkage,
                              memory_features(memory, state), action, sim_context, terrain_scan), -1)
        return features, inside
