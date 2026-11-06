"""Exact exogenous disturbance clocks and COM-velocity steps.

The actor acts on its pre-push sensor history. The motion model must start from
``apply_velocity_push`` and receive all four held wrenches as conditions; these
quantities are not motion targets or actor observations.
"""
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from .physical import PhysicalState


DISTURBANCE_STATE_FIELDS = (
    "disturbance.push_remaining_s", "disturbance.wrench_remaining_s",
    "disturbance.held_wrench_body", "disturbance.scale",
)
DISTURBANCE_RECORD_FIELDS = {
    "disturbance": (9,), "end_disturbance": (9,),
    "velocity_push": (3,), "external_wrench": (4, 6),
}
DISTURBANCE_NOISE_DIM = 4 + 4 * 7


def shift_tick_reference(labels: Tensor, start: PhysicalState, pushed: PhysicalState, *, to_pushed=False):
    """Rebase only twist increments across the instantaneous push; contacts stay physical."""
    rotation_t = start.rotation.transpose(-1, -2)
    dv = (rotation_t @ (pushed.velocity - start.velocity)[..., None]).squeeze(-1)
    dw = (rotation_t @ (pushed.angular_velocity - start.angular_velocity)[..., None]).squeeze(-1)
    sign = -1 if to_pushed else 1
    return torch.cat((labels[..., :6], labels[..., 6:9] + sign * dv[..., None, :],
                      labels[..., 9:12] + sign * dw[..., None, :], labels[..., 12:]), -1)


def apply_velocity_push(state: PhysicalState, delta: Tensor, base_com_local: Tensor) -> PhysicalState:
    """Native world [COM vx, COM vy, wz] jump, represented at the base origin."""
    zero = torch.zeros_like(delta[..., :1])
    dv_com = torch.cat((delta[..., :2], zero), -1)
    dw = torch.cat((zero, zero, delta[..., 2:]), -1)
    offset = (state.rotation @ base_com_local[..., None]).squeeze(-1)
    return replace(state, velocity=state.velocity + dv_com - torch.linalg.cross(dw, offset),
                   angular_velocity=state.angular_velocity + dw)


@dataclass(frozen=True)
class DisturbanceState:
    push_remaining_s: Tensor
    wrench_remaining_s: Tensor
    held_wrench_body: Tensor
    scale: Tensor

    @classmethod
    def decode(cls, value):
        return cls(value[..., 0], value[..., 1], value[..., 2:8], value[..., 8])

    def encode(self):
        return torch.cat((self.push_remaining_s[..., None], self.wrench_remaining_s[..., None],
                          self.held_wrench_body, self.scale[..., None]), -1)

    def advance_prepared(self, config, eligible, noise, *, policy_dt_s=.02):
        """Advance using row-local uniforms drawn before microbatching or retries.

        All branches select from the same prepared realization. Their law and
        clocks match native scheduling; they do not depend on microbatch size.
        """
        if noise.shape != (len(self.scale), DISTURBANCE_NOISE_DIM):
            raise ValueError('Prepared disturbance uniforms require [batch, 32]')
        push_clock, wrench_clock = self.push_remaining_s, self.wrench_remaining_s
        delta = self.held_wrench_body.new_zeros(len(self.scale), 3)
        wrench = self.held_wrench_body
        if config.pushes:
            push_clock = push_clock - policy_dt_s
            due = (push_clock <= 0) & eligible
            limit = delta.new_tensor([config.push_linear_m_s, config.push_linear_m_s, config.push_yaw_rad_s])
            delta = (noise[:, :3] * 2 - 1) * limit * self.scale[:, None] * due[:, None]
            low, high = config.push_interval_s
            push_clock = torch.where(due, low + (high - low) * noise[:, 3], push_clock)
        ticks = []
        for tick in range(4):
            if config.wrenches:
                wrench_clock = wrench_clock - policy_dt_s / 4
                due = wrench_clock <= 0
                values = noise[:, 4 + tick * 7:11 + tick * 7]
                limit = wrench.new_tensor([config.external_force_n] * 3 + [config.external_torque_nm] * 3)
                wrench = torch.where(due[:, None], (values[:, :6] * 2 - 1) * limit * self.scale[:, None], wrench)
                low, high = config.wrench_interval_s
                wrench_clock = torch.where(due, low + (high - low) * values[:, 6], wrench_clock)
            ticks.append(wrench)
        return delta, torch.stack(ticks, 1), DisturbanceState(push_clock, wrench_clock, wrench, self.scale)

    def advance(self, config, eligible, generator, *, policy_dt_s=.02, substeps=4):
        """Match native push-before-action and per-tick wrench draws, including RNG.

        Use the pre-action command mode and recovery-preparation eligibility.
        Draw once for a vector rollout; reuse these draws across microbatches.
        ``config`` is the existing RequestedRandomizationConfig, not new ranges.
        """
        n, device = len(self.scale), self.scale.device
        push_clock, wrench_clock = self.push_remaining_s.clone(), self.wrench_remaining_s.clone()
        delta = self.held_wrench_body.new_zeros(n, 3)
        wrench = self.held_wrench_body.clone()

        def interval(count, bounds):
            low, high = bounds
            return low + (high - low) * torch.rand(count, device=device, generator=generator)

        def uniform(count, width):
            return torch.rand(count, width, device=device, generator=generator) * 2 - 1

        if config.pushes:
            push_clock = push_clock - policy_dt_s
            rows = ((push_clock <= 0) & eligible).nonzero().flatten()
            limit = delta.new_tensor([config.push_linear_m_s, config.push_linear_m_s, config.push_yaw_rad_s])
            delta[rows] = uniform(len(rows), 3) * limit * self.scale[rows, None]
            if rows.numel():
                push_clock[rows] = interval(len(rows), config.push_interval_s)
        ticks = []
        for _ in range(substeps):
            if config.wrenches:
                wrench_clock = wrench_clock - policy_dt_s / substeps
                rows = (wrench_clock <= 0).nonzero().flatten()
                if rows.numel():
                    limit = wrench.new_tensor([config.external_force_n] * 3 + [config.external_torque_nm] * 3)
                    wrench[rows] = uniform(len(rows), 6) * limit * self.scale[rows, None]
                    wrench_clock[rows] = interval(len(rows), config.wrench_interval_s)
            ticks.append(wrench.clone())
        return (delta, torch.stack(ticks, 1),
                DisturbanceState(push_clock, wrench_clock, wrench, self.scale))
