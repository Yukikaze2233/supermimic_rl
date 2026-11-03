"""Continuous encoder coordinates for the verified Kamino TWOPI export.

Newton's correction first chooses the nearest 2*pi representation, then takes
``wp.mod(corrected, q_j_limit)``. TWOPI's q_j_limit is ``wp.tau``: its actual
discontinuity is 2*pi, despite the enum docstring's [-2*pi, 2*pi] range. This is
different from PhysX's 4*pi representation. No policy target or native state is
modified; only the read-side continuous encoder coordinate accumulates turns.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor


SDK_EVIDENCE = {
    "kinematics/joints.py": {
        "sha256": "63cba26766b672c74a814a68317eafea755c5c8eaf504aa08c0cb1e56ca361c5",
        "lines": [64, 72, 99],
        "rule": "nearest-reference correction by tau, followed by mod(q_j_limit)",
    },
    "core/joints.py": {
        "sha256": "52f275a1cefb883a3571caeda855894c07eb1147c112c17b4bde91c636562e43",
        "lines": [425, 426],
        "rule": "JointCorrectionMode.TWOPI.bound == wp.tau",
    },
}


class NewtonContinuousJointCoordinates:
    """Decode 2*pi native wraps while retaining the upstream sampling guards."""

    period_rad = math.tau

    def __init__(self, initial_position: Tensor, joint_names, continuous_names,
                 physics_dt_s: float):
        joint_names, continuous_names = tuple(joint_names), tuple(continuous_names)
        if (not isinstance(initial_position, Tensor) or initial_position.ndim != 2
                or initial_position.dtype != torch.float32
                or initial_position.shape[1] != len(joint_names)
                or len(set(joint_names)) != len(joint_names)
                or len(set(continuous_names)) != len(continuous_names)
                or not set(continuous_names).issubset(joint_names)
                or not math.isfinite(physics_dt_s) or physics_dt_s <= 0
                or not torch.isfinite(initial_position).all()):
            raise ValueError("Continuous coordinates require exact finite named native rows and a physical clock")
        self.shape, self.device = initial_position.shape, initial_position.device
        self.joint_names, self.continuous_names = joint_names, continuous_names
        self.indices = torch.tensor([joint_names.index(name) for name in continuous_names],
                                    dtype=torch.long, device=self.device)
        self.physics_dt_s = float(physics_dt_s)
        self._previous_raw = initial_position.detach().clone()
        self._position = initial_position.detach().clone()
        self._turns = torch.zeros(self.shape[0], len(continuous_names), dtype=torch.int64, device=self.device)
        self._tick = None

    def _layout(self, value):
        if (not isinstance(value, Tensor) or value.shape != self.shape
                or value.device != self.device or value.dtype != torch.float32):
            raise ValueError("Continuous coordinates differ from the bound native layout")

    @torch.no_grad()
    def read(self, raw_position, velocity, physical_tick):
        self._layout(raw_position)
        self._layout(velocity)
        if type(physical_tick) is not int or physical_tick < 0:
            raise ValueError("Continuous coordinates require an integer physical tick")
        if self._tick == physical_tick:
            return self._position
        if self._tick is not None and physical_tick != self._tick + 1:
            raise RuntimeError("Continuous coordinates lost a physical sample; reset explicitly")
        raw = raw_position.index_select(1, self.indices)
        speed = velocity.index_select(1, self.indices)
        if (not torch.isfinite(raw).all() or not torch.isfinite(speed).all()
                or (speed.abs() * self.physics_dt_s >= math.pi).any()):
            raise RuntimeError("Continuous joint state is nonfinite or angular sampling is ambiguous")
        previous = self._previous_raw.index_select(1, self.indices)
        turns = torch.round((previous.double() - raw.double()) / self.period_rad).long()
        delta = raw.double() - previous.double() + turns.double() * self.period_rad
        # Float32(pi) can round just beyond pi, then the 2*pi correction turns
        # it into just below -pi. Keep that half-turn ambiguous in either sign.
        roundoff = torch.finfo(raw.dtype).eps * self.period_rad
        if (delta.abs() >= math.pi - roundoff).any():
            raise RuntimeError("Continuous joint position increment is ambiguous")
        # Publish new endpoint buffers so pre-reset replay snapshots stay valid.
        next_turns = self._turns + turns
        position = raw_position.detach().clone()
        unwrapped = (raw.double() + next_turns.double() * self.period_rad).float()
        if not torch.isfinite(unwrapped).all():
            raise RuntimeError("Continuous encoder accumulation overflowed its representation")
        position.index_copy_(1, self.indices, unwrapped)
        self._turns.copy_(next_turns)
        self._previous_raw = raw_position.detach().clone()
        self._position = position
        self._tick = physical_tick
        return position

    @torch.no_grad()
    def reset_rows(self, ids, written_position):
        """Seed actual reset coordinates while preserving all continuing rows."""
        if (not isinstance(ids, Tensor) or ids.ndim != 1 or ids.dtype != torch.long
                or ids.device != self.device or written_position.shape != (len(ids), self.shape[1])
                or written_position.dtype != torch.float32 or written_position.device != self.device
                or not torch.isfinite(written_position).all()
                or ((ids < 0) | (ids >= self.shape[0])).any()):
            raise ValueError("Continuous coordinate reset differs from the native reset rows")
        self._previous_raw = self._previous_raw.clone()
        self._position = self._position.clone()
        self._previous_raw.index_copy_(0, ids, written_position)
        self._position.index_copy_(0, ids, written_position)
        self._turns[ids] = 0


def install_newton_coordinates(environment) -> dict | None:
    """Install at construction, using the exact source asset names and clock.

    PhysX retains its original reader. Unknown Newton correction modes are not
    guessed: the inspected course configuration uses Kamino TWOPI explicitly.
    """
    if not getattr(environment, "_is_newton", False):
        return None
    cfg = environment.cfg
    mode = cfg.sim.physics.solver_cfg.rotation_correction
    if getattr(cfg, "physics_backend_name", None) != "kamino" or mode != "twopi":
        raise ValueError("Course encoder adapter is verified for Kamino rotation_correction='twopi' only")
    previous = environment._joint_coordinates
    if previous._turns.any():
        raise RuntimeError("Install the backend encoder before any accumulated simulation turns")
    raw = environment._as_torch(environment.robot.data.joint_pos)
    names = environment._binding.native_joint_names
    continuous = environment._asset.continuous_joint_names()
    environment._joint_coordinates = NewtonContinuousJointCoordinates(
        raw, names, continuous, float(environment.physics_dt))
    receipt = {
        "contract": "kamino_twopi_encoder_2pi_v1", "backend": "kamino",
        "rotation_correction": mode, "native_period_rad": math.tau,
        "sampling_displacement_limit_rad": math.pi,
        "half_turn_roundoff_rad": torch.finfo(raw.dtype).eps * math.tau,
        "physics_dt_s": float(environment.physics_dt),
        "continuous_joint_names": list(continuous), "sdk_evidence": SDK_EVIDENCE,
        "native_state_or_controller_target_modified": False,
    }
    environment.continuous_coordinate_receipt = receipt
    return receipt
