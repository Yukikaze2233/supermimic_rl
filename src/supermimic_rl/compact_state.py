"""Named compact physical state and a differentiable clean actor observer.

The projection removes redundant rigid-body poses, coordinate-reader caches,
contact-force samples and the copied observation. It preserves controller,
transport, sensor, reward and failure memories verbatim: their deterministic
update rules belong to a subsequent model implementation, not this projection.
This is an observable model state, not a restorable PhysX solver snapshot.

``analytic_actor_observation`` reproduces instantaneous noiseless clean35. It
does not reproduce a delayed/noisy actor frame from instantaneous physics. A
caller using those sensors must explicitly model their retained histories and
noise, or use a separate observation head trained against that sensor contract.
"""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Sequence

import torch
from torch import Tensor

from .adapters.isaac import RawSnapshot
from .contracts import StateSchema


def named_field(tensor: Tensor, schema: StateSchema, name: str, *, context: bool = False) -> Tensor:
    """Read a shaped field by name, preserving arbitrary leading batch axes."""
    fields = schema.context_fields if context else schema.state_fields
    width = schema.context_dim if context else schema.state_dim
    if tensor.ndim < 1 or tensor.shape[-1] != width:
        raise ValueError(f"Packed {'context' if context else 'state'} width differs from schema")
    offset = 0
    for field in fields:
        if field.name == name:
            return tensor[..., offset:offset + field.size].reshape(tensor.shape[:-1] + field.shape)
        offset += field.size
    raise ValueError(f"Required {'context' if context else 'state'} field is missing: {name}")


def _inverse_rotate(quaternion: Tensor, vector: Tensor) -> Tensor:
    imaginary, scalar = -quaternion[..., :3], quaternion[..., 3:]
    cross = torch.linalg.cross(imaginary, vector)
    return vector + 2 * (scalar * cross + torch.linalg.cross(imaginary, cross))


class CompactStateProjector:
    """Project native ``RawSnapshot`` values before ``PackedEnvironment`` packs.

    Body/joint orders must come from the native binding. The joint arrays remain
    in that exact order, including passive/closed-chain coordinates. Full-vehicle
    adapters expose their two extra axes separately as ``gimbal.position_rad``,
    ``gimbal.velocity_rad_s`` and ``gimbal.effort_nm``; those fields and context
    ``gimbal.command`` pass through unchanged. No auxiliary zero state is made up.

    Terrain forces are omitted from the dynamics state. Contact-dependent reward
    and termination therefore still need validated outcome/contact prediction;
    retaining their counters does not make contact forces analytically known.
    Omitting horizontal position assumes the current flat, translation-invariant
    task; position-dependent terrain needs an explicit additional representation.
    Unknown additional state fields are retained, avoiding silent memory loss.
    """

    _REMOVED = frozenset({
        "body_link_pose_world_xyzw", "body_com_pose_world_xyzw", "body_com_velocity_world",
        "joint_position_native", "continuous._previous_raw", "continuous._position",
        "continuous._turns", "terrain_contact_force_world_n", "clean_observation",
    })
    _ROOT_METADATA = {
        "root_height_above_ground_m": {"units": "m", "frame": "base_link origin above local ground; world Z"},
        "root_orientation_xyzw": {"units": "1", "frame": "base_link orientation in world; quaternion xyzw"},
        "root_linear_velocity_body_m_s": {"units": "m/s", "frame": "base_link origin; base_link axes"},
        "root_angular_velocity_body_rad_s": {"units": "rad/s", "frame": "base_link axes"},
    }

    def __init__(self, body_names: Sequence[str], joint_names: Sequence[str], *, root_body_name: str = "base_link"):
        self.body_names, self.joint_names = tuple(body_names), tuple(joint_names)
        for label, names in (("body", self.body_names), ("joint", self.joint_names)):
            if (not names or any(not isinstance(name, str) or not name for name in names)
                    or len(set(names)) != len(names)):
                raise ValueError(f"Native {label} names must be nonempty and unique")
        if root_body_name not in self.body_names:
            raise ValueError("The named root body is absent from the native binding")
        self.root_index = self.body_names.index(root_body_name)
        self.root_body_name = root_body_name

    def project(self, snapshot: RawSnapshot) -> RawSnapshot:
        """Return an owned compact state without changing source observations."""
        if not isinstance(snapshot, RawSnapshot):
            raise TypeError("Compact projection requires a RawSnapshot")
        state, context = snapshot.state, snapshot.context
        n = snapshot.observation.shape[0]
        required = {
            "body_link_pose_world_xyzw": (len(self.body_names), 7),
            "body_com_pose_world_xyzw": (len(self.body_names), 7),
            "body_com_velocity_world": (len(self.body_names), 6),
            "joint_position_continuous": (len(self.joint_names),),
            "joint_velocity_native": (len(self.joint_names),),
        }
        for name, shape in required.items():
            value = state.get(name)
            if (not isinstance(value, Tensor) or value.shape != (n, *shape)
                    or not value.is_floating_point() or not torch.isfinite(value).all()):
                raise ValueError(f"Missing or invalid native compact source field: {name}")
        ground = context.get("ground_at_root_world_z_m")
        if (not isinstance(ground, Tensor) or ground.shape not in ((n,), (n, 1))
                or not torch.isfinite(ground).all()):
            raise ValueError("Compact projection requires ground_at_root_world_z_m per environment")
        if set(self._ROOT_METADATA).intersection(state):
            raise ValueError("Compact root fields already exist; do not project a snapshot twice")
        link = state["body_link_pose_world_xyzw"][:, self.root_index]
        com = state["body_com_pose_world_xyzw"][:, self.root_index]
        motion = state["body_com_velocity_world"][:, self.root_index]
        quaternion = link[:, 3:]
        norm = quaternion.norm(dim=-1, keepdim=True)
        if (norm - 1).abs().gt(2e-4).any():
            raise ValueError("Native root orientation must be a unit xyzw quaternion")
        quaternion = quaternion / norm
        omega = motion[:, 3:]
        com_position = com[:, :3]
        local_com = context.get('body_com_pose_local_xyzw')
        if local_com is not None:
            if (not isinstance(local_com, Tensor) or local_com.shape != (n, len(self.body_names), 7)
                    or not local_com.is_floating_point() or not torch.isfinite(local_com).all()):
                raise ValueError('Invalid native body_com_pose_local_xyzw context')
            # Match LabStartupDynamics.base_com_position, used by the native
            # velocity label. SDK p + R*c rounds differently at large world xy.
            offset = local_com[:, self.root_index, :3]
            imaginary, scalar = quaternion[:, :3], quaternion[:, 3:]
            cross = torch.linalg.cross(imaginary, offset)
            com_position = link[:, :3] + offset + 2 * (
                scalar * cross + torch.linalg.cross(imaginary, cross))
        origin_velocity = motion[:, :3] - torch.linalg.cross(omega, com_position - link[:, :3])
        projected = {name: value.clone() for name, value in state.items() if name not in self._REMOVED}
        projected.update({
            "root_height_above_ground_m": link[:, 2:3] - ground.reshape(n, 1),
            "root_orientation_xyzw": quaternion,
            "root_linear_velocity_body_m_s": _inverse_rotate(quaternion, origin_velocity),
            "root_angular_velocity_body_rad_s": _inverse_rotate(quaternion, omega),
        })
        # PhysX wraps the continuous yaw encoder at +/-2*pi. Its physical
        # orientation is periodic; a raw residual target would see a 4*pi jump.
        if "gimbal.position_rad" in projected:
            auxiliary_q = projected.pop("gimbal.position_rad")
            if auxiliary_q.shape != (n, 2):
                raise ValueError("gimbal positions must contain native yaw and pitch")
            yaw = auxiliary_q[:, :1]
            projected["gimbal.yaw_sin_cos"] = torch.cat((yaw.sin(), yaw.cos()), dim=-1)
            projected["gimbal.pitch_position_rad"] = auxiliary_q[:, 1:].clone()
        original_metadata = snapshot.field_metadata or {}
        state_metadata = {name: dict(original_metadata.get("state", {}).get(name, {})) for name in projected}
        for name, metadata in self._ROOT_METADATA.items():
            state_metadata[name] = {key: value.replace("base_link", self.root_body_name)
                                    for key, value in metadata.items()}
        if "gimbal.yaw_sin_cos" in projected:
            state_metadata["gimbal.yaw_sin_cos"] = {"units": "1", "frame": "sin,cos of native yaw joint angle"}
            state_metadata["gimbal.pitch_position_rad"] = {"units": "rad", "frame": "native pitch joint axis"}
        metadata = {
            "state": state_metadata,
            "context": {name: dict(value) for name, value in original_metadata.get("context", {}).items()},
        }
        return replace(snapshot, state=projected, context=dict(context), field_metadata=metadata)


def analytic_actor_observation(
    continuous: Tensor,
    context: Tensor,
    goal: Tensor,
    previous_action: Tensor | None,
    schema: StateSchema,
    policy_joint_indices: Tensor | Sequence[int] | None = None,
    nominal: Tensor | Sequence[float] | None = None,
) -> Tensor:
    """Reconstruct clean35 with gradients through physical state and actions.

    ``continuous`` is the complete packed compact state, including its retained
    memories. ``goal`` is the existing encoded 11D command, already scaled. Six
    policy indices select [left pair, right pair, left wheel, right wheel] from
    native joint order; default indices and nominal positions are read by name
    from context ``policy_joint_indices`` and ``controller.nominal``.
    A missing ``previous_action`` uses native memory ``control._previous_action``.

    The common winding of each coaxial pair is removed together, matching
    ``formal.control.ownership.canonical_coaxial_displacement``. This map has its
    unavoidable branch discontinuity at a pair mean of pi; its derivative is
    preserved away from that boundary. Wheel position slots remain exactly zero.
    """
    if previous_action is None:
        previous_action = named_field(continuous, schema, "control._previous_action")
    if (not continuous.is_floating_point() or continuous.ndim < 2
            or context.shape[:-1] != continuous.shape[:-1]
            or context.shape[-1] != schema.context_dim
            or context.device != continuous.device or goal.device != continuous.device
            or previous_action.device != continuous.device
            or goal.shape != (*continuous.shape[:-1], 11)
            or previous_action.shape != (*continuous.shape[:-1], 6)):
        raise ValueError("Clean observation requires matching packed state/context and 11D/6D inputs")
    position_fields = {field.name for field in schema.state_fields}.intersection(
        ("joint_position_continuous", "joint_position_gimbal_yaw_sine"))
    if len(position_fields) != 1:
        raise ValueError("Clean observation needs one unambiguous native joint-position representation")
    # The formal-course phase codec changes only the independent yaw column;
    # all six locomotion policy columns keep their native coordinates/indices.
    q = named_field(continuous, schema, next(iter(position_fields)))
    dq = named_field(continuous, schema, "joint_velocity_native")
    quaternion = named_field(continuous, schema, "root_orientation_xyzw")
    omega = named_field(continuous, schema, "root_angular_velocity_body_rad_s")
    batch = continuous.shape[:-1]
    if (q.ndim != continuous.ndim or dq.shape != q.shape
            or quaternion.shape != (*batch, 4) or omega.shape != (*batch, 3)):
        raise ValueError("Compact joint, root quaternion or root angular-velocity shapes differ")
    if policy_joint_indices is None:
        policy_joint_indices = named_field(context, schema, "policy_joint_indices", context=True)
    indices = torch.as_tensor(policy_joint_indices, device=continuous.device)
    if (indices.shape not in ((6,), (*batch, 6)) or indices.dtype == torch.bool
            or indices.is_complex() or not torch.isfinite(indices).all()
            or (indices.is_floating_point() and not torch.equal(indices, indices.round()))):
        raise ValueError("policy_joint_indices must contain six integer native indices per row")
    indices = indices.to(torch.long).expand(*batch, 6)
    if ((indices < 0).any() or (indices >= q.shape[-1]).any()
            or (indices.sort(dim=-1).values.diff(dim=-1) == 0).any()):
        raise ValueError("policy_joint_indices must be unique and in native joint range")
    if nominal is None:
        nominal = named_field(context, schema, "controller.nominal", context=True)
    nominal = torch.as_tensor(nominal, device=continuous.device, dtype=continuous.dtype)
    if nominal.shape not in ((6,), (*batch, 6)) or not torch.isfinite(nominal).all():
        raise ValueError("controller.nominal must contain six finite policy positions per row")
    policy_q, policy_dq = q.gather(-1, indices), dq.gather(-1, indices)
    pairs = (policy_q[..., :4].double() - nominal[..., :4].double()).unflatten(-1, (2, 2))
    turns = torch.floor(pairs.mean(-1) / (2 * math.pi) + 0.5)
    relative = (pairs - turns[..., None] * (2 * math.pi)).flatten(-2).to(continuous.dtype)
    norm = quaternion.norm(dim=-1, keepdim=True)
    if not torch.isfinite(norm).all() or (norm <= torch.finfo(continuous.dtype).eps).any():
        raise ValueError("Clean observation requires a finite nonzero root quaternion")
    quaternion = quaternion / norm
    gravity_world = torch.zeros_like(omega)
    gravity_world[..., 2] = -1
    gravity = _inverse_rotate(quaternion, gravity_world)
    frame = torch.cat((goal[..., :4], omega * 0.5, gravity, relative,
                       torch.zeros_like(policy_q[..., 4:]), policy_dq * 0.1,
                       previous_action, goal[..., 4:]), dim=-1)
    if not torch.isfinite(frame).all():
        raise ValueError("Clean observation contains nonfinite physical or command values")
    return frame.clamp(-100, 100)
