"""Analytic pose/contact failure clocks on the physical substep grid.

The limits come from the pinned native runtime, never from model predictions.
This implements the pose/contact part of PhysicalFailureLatch. Mechanical
reason bits already present in a snapshot stay latched; predicting a new
mechanical failure requires the corresponding geometry/solver measurements.
Discrete events have no gradient, as with other task-state transitions.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PoseContactLimits:
    minimum_normal_height_m: float
    minimum_recovery_height_m: float
    failure_tilt_rad: float
    forbidden_body_load_n: float
    forbidden_contact_confirm_s: float
    pose_failure_confirm_s: float
    forbidden_body_names: tuple[str, ...]

    @classmethod
    def from_runtime(cls, runtime: dict) -> PoseContactLimits:
        source = runtime["termination_limits"]
        values = {name: source[name] for name in cls.__dataclass_fields__}
        values["forbidden_body_names"] = tuple(values["forbidden_body_names"])
        return cls(**values)


@dataclass(frozen=True)
class FailureInterval:
    memory: Tensor          # [N, 3]: contact time, pose time, latched reason bits
    failed: Tensor          # [N], including any previously latched failure
    first_failure: Tensor   # [N], at most one event per interval
    failed_at_tick: Tensor  # [N, K]


class PoseContactTermination:
    def __init__(self, limits: PoseContactLimits, physics_dt_s: float):
        self.limits, self.dt = limits, physics_dt_s
        self.base_index = limits.forbidden_body_names.index("base_link")
        if physics_dt_s <= 0:
            raise ValueError("Termination requires a positive physical clock")

    @torch.no_grad()
    def advance(self, memory: Tensor, height: Tensor, gravity: Tensor, loads: Tensor, *,
                recovery: Tensor, body_contact_allowed: Tensor) -> FailureInterval:
        limits, dt = self.limits, self.dt
        # Packed snapshots store float32 seconds. Recover integer-tick ages
        # before testing a 0.4 s boundary, avoiding a one-tick roundoff delay.
        contact_time = (memory[..., 0].double() / dt).round() * dt
        pose_time = (memory[..., 1].double() / dt).round() * dt
        reasons = memory[..., 2].round().long()
        failed_before = reasons != 0
        failed_ticks = []
        for tick in range(height.shape[-2]):
            recover = recovery[:, tick] if recovery.ndim == 2 else recovery
            allowed = body_contact_allowed[:, tick] if body_contact_allowed.ndim == 2 else body_contact_allowed
            contacts = loads[:, tick].gt(limits.forbidden_body_load_n)
            contacts[:, self.base_index] &= ~allowed
            contact = contacts.any(-1) & ~recover
            contact_time = torch.where(contact, contact_time + dt, 0.)
            minimum = torch.where(recover, limits.minimum_recovery_height_m, limits.minimum_normal_height_m)
            low = height[:, tick, 0] < minimum
            tilt = torch.atan2(gravity[:, tick, :2].norm(dim=-1), -gravity[:, tick, 2])
            tilted = (tilt > limits.failure_tilt_rad) & ~recover
            pose_time = torch.where(low | tilted, pose_time + dt, 0.)
            confirmed = pose_time >= limits.pose_failure_confirm_s - 1e-9
            reasons |= (low & confirmed).long()
            reasons |= (tilted & confirmed).long() * 2
            reasons |= (contact_time >= limits.forbidden_contact_confirm_s - 1e-9).long() * 4
            failed_ticks.append(reasons != 0)
        failed = reasons != 0
        result = torch.stack((contact_time, pose_time, reasons), -1).to(memory.dtype)
        return FailureInterval(result, failed, failed & ~failed_before, torch.stack(failed_ticks, -1))
