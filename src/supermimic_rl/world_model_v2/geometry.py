"""Asset-derived wheel cylinders and passive knees for differentiable reward geometry.

The selected wheel collision offsets and cylinder axes are parallel to the
wheel joint axes, so their geometry is independent of wheel spin. This is
checked from the source asset before omitting wheel angle from model state.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import Tensor

from . import so3
from .leg_linkage import LegLinkage
from .physical import PhysicalState


@dataclass(frozen=True)
class WheelGeometry:
    link_position_base: Tensor
    centre_world: Tensor
    cylinder_axis_world: Tensor
    bottom_world_z: Tensor
    extension_m: Tensor
    forward_velocity_rad_s: Tensor
    knee_angle_rad: Tensor
    knee_velocity_rad_s: Tensor
    knee_clearance_rad: Tensor
    valid: Tensor

    def tick(self, index: int):
        axis = self.valid.ndim - 1
        return type(self)(*(getattr(self, item.name).select(axis, index) for item in fields(self)))


@dataclass(frozen=True)
class _Side:
    name: str
    primary: int
    secondary: int
    wheel: int
    passive_knee: int
    hip_origin: Tensor
    hip_axis: Tensor
    knee_origin: Tensor
    knee_axis: Tensor
    wheel_origin: Tensor
    wheel_axis: Tensor
    collision_offset: Tensor
    collision_axis: Tensor
    radius: float
    half_length: float
    knee_inner_offset: float
    knee_inner_direction: float


class RewardGeometry:
    def __init__(self, linkage: LegLinkage, asset_directory):
        root = Path(asset_directory)
        manifest_raw, spec_raw = (root / 'manifest.json').read_bytes(), (root / 'model_spec.json').read_bytes()
        manifest, spec = json.loads(manifest_raw), json.loads(spec_raw)
        if (hashlib.sha256(manifest_raw).hexdigest() != linkage.provenance['asset_manifest_sha256']
                or hashlib.sha256(spec_raw).hexdigest() != manifest['files_sha256']['model_spec.json']):
            raise ValueError('Reward geometry and closed-chain table must use the same pinned asset')
        self.linkage = linkage
        self.knee_limits = tuple(math.radians(v) for v in manifest['knee_inner_limits_deg'])
        joints, bodies = {v['name']: v for v in spec['joints']}, {v['name']: v for v in spec['bodies']}
        self.sides = []
        self._device_sides = {}
        for side, primary, secondary, wheel, wheel_joint in (
                ('left', 0, 1, 4, 'L_joint3'), ('right', 2, 3, 5, 'R_joint3')):
            table = linkage.sides[side]
            hip, output = joints[table.primary_joint], joints[wheel_joint]
            knees = [v for v in spec['joints'] if v['parent'] == hip['child'] and v['child'] == output['parent']]
            if hip['parent'] != 'base_link' or len(knees) != 1:
                raise ValueError('Reward wheel geometry requires the declared hip/knee/wheel path')
            knee = knees[0]
            collisions = bodies[output['child']]['collisions']
            if len(collisions) != 1 or collisions[0]['type'] != 'cylinder':
                raise ValueError('Reward geometry requires one authored cylinder per wheel')
            collision = collisions[0]
            matrix = torch.tensor(collision['origin'], dtype=torch.float64)
            axis = torch.tensor(output['axis'], dtype=torch.float64)
            axis /= axis.norm()
            offset, cylinder_axis = matrix[:3, 3], matrix[:3, 2]
            if (torch.linalg.cross(offset, axis).norm() > 1e-9
                    or torch.linalg.cross(cylinder_axis, axis).norm() > 1e-9):
                raise ValueError('Wheel spin changes collision geometry; add wheel phase to the physical state')
            tensor = lambda value: torch.tensor(value, dtype=torch.float64)
            knee_column = table.passive_joint_names.index(knee['name'])
            endpoint = linkage.unpack(table.values[[0, -1]])
            inner = endpoint['knee_inner_angle_rad'][:, 0]
            native = endpoint['passive_joint'][:, knee_column]
            direction = float((inner[1] - inner[0]) / (native[1] - native[0]))
            if abs(abs(direction) - 1.) > 1e-6:
                raise ValueError('Passive knee coordinate does not map affinely to the internal angle')
            direction = 1. if direction > 0 else -1.
            knee_offset = float(inner[0] - direction * native[0])
            self.sides.append(_Side(side, primary, secondary, wheel, table.passive_joint_names.index(knee['name']),
                tensor(hip['origin']), tensor(hip['axis']), tensor(knee['origin']), tensor(knee['axis']),
                tensor(output['origin']), axis, matrix[:3, 3], cylinder_axis, collision['radius'],
                collision['length'] / 2, knee_offset, direction))

    def _sides_for(self, reference: Tensor):
        """Immutable asset transforms stay resident throughout a physics rollout."""
        key = (reference.device, reference.dtype)
        if key not in self._device_sides:
            self._device_sides[key] = [replace(side, **{
                item.name: getattr(side, item.name).to(device=reference.device, dtype=reference.dtype)
                for item in fields(side) if isinstance(getattr(side, item.name), Tensor)}) for side in self.sides]
        return self._device_sides[key]

    def __call__(self, state: PhysicalState, *, passive_knee_position: Tensor | None = None,
                 passive_knee_velocity: Tensor | None = None) -> WheelGeometry:
        """Evaluate ideal linkage, or the two explicitly supplied native knee coordinates.

        The latter is a measured-tree audit path, not a closure correction. It
        separates solver compliance from FK/readback errors without assuming
        that real passive joints exactly follow the ideal LUT.
        """
        if (passive_knee_position is None) != (passive_knee_velocity is None):
            raise ValueError('Supply both passive knee positions and velocities, in native joint axes')
        if passive_knee_position is not None:
            expected = (*state.joint_position.shape[:-1], 2)
            if passive_knee_position.shape != expected or passive_knee_velocity.shape != expected:
                raise ValueError('Passive knee coordinates must match the batch shape and left/right order')
        q, dq = state.joint_position, state.joint_velocity
        link_positions, centres, axes, bottoms, wheel_speeds, knees, knee_speeds, valid = ([] for _ in range(8))
        # Native wheel direction uses world-horizontal body heading, not a tilted body-y axis.
        forward = state.rotation[..., :, 0]
        forward_xy = forward[..., :2] / forward[..., :2].norm(dim=-1, keepdim=True).clamp_min(1e-6)
        left = torch.stack((-forward_xy[..., 1], forward_xy[..., 0], torch.zeros_like(forward_xy[..., 0])), -1)
        for side_index, side in enumerate(self._sides_for(q)):
            phi = q[..., side.secondary] - q[..., side.primary]
            if passive_knee_position is None:
                table = self.linkage.sides[side.name]
                features, inside = self.linkage.evaluate(side.name, phi)
                derivative = self.linkage.unpack(table.derivative(phi))
                knee_q = features['passive_joint'][..., side.passive_knee]
                knee_speed = derivative['knee_inner_angle_rad'][..., 0] * (dq[..., side.secondary] - dq[..., side.primary])
                knee_angle = features['knee_inner_angle_rad'][..., 0]
            else:
                knee_q = passive_knee_position[..., side_index]
                knee_speed = passive_knee_velocity[..., side_index] * side.knee_inner_direction
                knee_angle = knee_q * side.knee_inner_direction + side.knee_inner_offset
                # The measured tree has a defined pose even when the solver's
                # compliant loop leaves the ideal LUT domain. Stop violations
                # belong to physical termination, not a fabricated geometry.
                inside = torch.isfinite(knee_q) & torch.isfinite(knee_speed) & torch.isfinite(q[..., side.primary])
            hip, knee, wheel = side.hip_origin, side.knee_origin, side.wheel_origin
            hip_r = hip[:3, :3] @ so3.exp(side.hip_axis * q[..., side.primary, None])
            knee_p = hip[:3, 3] + (hip_r @ knee[:3, 3, None]).squeeze(-1)
            knee_r = hip_r @ knee[:3, :3] @ so3.exp(side.knee_axis * knee_q[..., None])
            wheel_p = knee_p + (knee_r @ wheel[:3, 3, None]).squeeze(-1)
            wheel_r = knee_r @ wheel[:3, :3]
            centre_base = wheel_p + (wheel_r @ side.collision_offset[..., None]).squeeze(-1)
            centre = state.position + (state.rotation @ centre_base[..., None]).squeeze(-1)
            axis_world = (state.rotation @ wheel_r @ side.collision_axis[..., None]).squeeze(-1)
            joint_world = (state.rotation @ wheel_r @ side.wheel_axis[..., None]).squeeze(-1)
            # norm(xy) equals sqrt(1-z^2) on the unit sphere and has finite
            # derivatives at a vertical cylinder axis (important during flips).
            extent = side.radius * axis_world[..., :2].norm(dim=-1) + side.half_length * axis_world[..., 2].abs()
            link_positions.append(wheel_p); centres.append(centre); axes.append(axis_world)
            bottoms.append(centre[..., 2] - extent)
            wheel_speeds.append(dq[..., side.wheel] * (joint_world * left).sum(-1))
            knees.append(knee_angle); knee_speeds.append(knee_speed); valid.append(inside)
        bottom, knee_angle = torch.stack(bottoms, -1), torch.stack(knees, -1)
        clearance = torch.minimum(knee_angle - self.knee_limits[0], self.knee_limits[1] - knee_angle)
        return WheelGeometry(torch.stack(link_positions, -2), torch.stack(centres, -2), torch.stack(axes, -2),
                             bottom, state.position[..., 2] - bottom.amax(-1), torch.stack(wheel_speeds, -1),
                             knee_angle, torch.stack(knee_speeds, -1), clearance, torch.stack(valid, -1).all(-1))
