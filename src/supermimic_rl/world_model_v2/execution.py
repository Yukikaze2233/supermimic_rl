"""Wheel command and identified delay queue advanced by code during imagination.

Per physics tick the controller computes the wheel velocity feedback from the
true wheel speed at the start of the tick, caps it, applies the per-episode
motor strength, caps again, and pushes the software-capped input into the
identified aggregate delay queue (left 5.5 ms, right 3.5 ms). The queue is a
motion-relevant memory: the world model receives it as an input, and code
advances it with the predicted tick wheel speeds.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
from torch import Tensor

WHEEL_TARGET_SCALE = 10.0      # rad/s per action unit (OutputFeedbackController.decode)
WHEEL_ACTION_BOUND = 9.0


@dataclass(frozen=True)
class WheelExecution:
    velocity_gain_nm_per_rad_s: float
    torque_cap_nm: float
    history_length: int
    whole_delay: tuple[int, int]
    fractional_delay: tuple[float, float]
    zero_torque_speed: tuple[tuple[float, float], tuple[float, float]] | None = None
    rolloff_slope: tuple[tuple[float, float], tuple[float, float]] | None = None

    @classmethod
    def from_bundle(cls, bundle_directory: str | Path, physics_dt_s: float) -> WheelExecution:
        """Read the pinned identified wheel model; delays are converted on the physics clock."""
        model = json.loads((Path(bundle_directory) / "wheel_model.json").read_text())
        # Same rows and rounding as IdentifiedExecution.
        delays = [model["axes"][name]["aggregate_delay_s"] for name in ("L_joint3", "R_joint3")]
        steps = [delay / physics_dt_s for delay in delays]
        whole = tuple(int(math.floor(step + 1e-10)) for step in steps)
        fraction = tuple(step - w for step, w in zip(steps, whole))
        return cls(float(model["controller_velocity_p_nm_per_rad_s"]), float(model["software_torque_cap_nm"]),
                   max(whole) + 2, whole, fraction,
                   tuple(tuple(model["axes"][name]["high_speed_envelope"]["zero_torque_speed_rad_s"])
                         for name in ("L_joint3", "R_joint3")),
                   tuple(tuple(model["axes"][name]["high_speed_envelope"]["rolloff_slope_nm_per_rad_s"])
                         for name in ("L_joint3", "R_joint3")))

    def target(self, action: Tensor) -> Tensor:
        return action[..., 4:6].clamp(-WHEEL_ACTION_BOUND, WHEEL_ACTION_BOUND) * WHEEL_TARGET_SCALE

    def motor_input(self, target_rad_s: Tensor, wheel_velocity_rad_s: Tensor, strength: Tensor) -> Tensor:
        """Software-capped wheel torque entering the delay queue for one tick."""
        cap = self.torque_cap_nm
        nominal = (self.velocity_gain_nm_per_rad_s * (target_rad_s - wheel_velocity_rad_s)).clamp(-cap, cap)
        return (nominal * strength).clamp(-cap, cap)

    def push(self, history: Tensor, motor_input: Tensor) -> Tensor:
        """Newest-first ``[N, history_length, 2]`` queue after one tick."""
        return torch.cat((motor_input[..., None, :], history[..., :-1, :]), -2)

    def delayed(self, history: Tensor) -> Tensor:
        """Delayed motor torque read right after the newest push."""
        columns = []
        for side in range(2):
            whole, fraction = self.whole_delay[side], self.fractional_delay[side]
            a, b = history[..., whole, side], history[..., whole + 1, side]
            columns.append(a * (1 - fraction) + b * fraction)
        return torch.stack(columns, -1)

    def applied(self, history: Tensor, wheel_velocity_rad_s: Tensor) -> Tensor:
        """Actual motor torque after the identified direction-dependent speed envelope."""
        if self.zero_torque_speed is None or self.rolloff_slope is None:
            raise ValueError("Applied wheel effort requires the identified high-speed envelope")
        speed = wheel_velocity_rad_s
        zero, slope = speed.new_tensor(self.zero_torque_speed), speed.new_tensor(self.rolloff_slope)
        negative = speed < 0
        capacity = (torch.where(negative, slope[:, 1], slope[:, 0])
                    * (torch.where(negative, zero[:, 1], zero[:, 0]) - speed.abs())).clamp(0., self.torque_cap_nm)
        delayed = self.delayed(history)
        bound = torch.where(speed * delayed > 0, capacity, self.torque_cap_nm)
        return delayed.clamp(min=-bound, max=bound)


@dataclass(frozen=True)
class LegExecution:
    """Native identified PD with a target held for the complete policy interval.

    Targets use a common whole-turn winding for each coaxial pair, chosen from
    the start feedback. Rewrapping each motor or each predicted tick changes
    the actual knee target and is not the deployed controller.
    """

    nominal: Tensor
    stiffness: Tensor
    damping: Tensor
    effort_bounds: Tensor
    # Native task.action_bounds for the leg axes (legacy 3 = +-43 deg around nominal).
    leg_action_bound: float = 3.

    def target(self, action: Tensor, feedback_position_rad: Tensor) -> Tensor:
        return self.decode_target(action, feedback_position_rad)[0]

    def decode_target(self, action: Tensor, feedback_position_rad: Tensor) -> tuple[Tensor, Tensor]:
        """Return the held target and the native four-axis admission correction."""
        desired = self.nominal[..., :4] + .25 * action[..., :4].clamp(-self.leg_action_bound, self.leg_action_bound)
        pairs = desired.double().unflatten(-1, (2, 2))
        feedback = feedback_position_rad[..., :4].double().unflatten(-1, (2, 2))
        turns = torch.floor((feedback.mean(-1) - pairs.mean(-1)) / (2 * math.pi) + .5).detach()
        admitted = (pairs + turns[..., None] * (2 * math.pi)).flatten(-2).to(action.dtype)
        unwound = admitted.double().unflatten(-1, (2, 2)) - turns[..., None] * (2 * math.pi)
        correction = (desired.double() - unwound.flatten(-2)).abs().to(action.dtype)
        return admitted, correction

    def motor_effort(self, target: Tensor, feedback_position_rad: Tensor,
                     velocity_rad_s: Tensor, strength: Tensor) -> Tensor:
        requested = (self.stiffness[..., :4] * (target - feedback_position_rad[..., :4])
                     - self.damping[..., :4] * velocity_rad_s[..., :4])
        bounds = self.effort_bounds[..., :4]
        nominal = requested.clamp(min=-bounds, max=bounds)
        return (nominal * strength[..., :4]).clamp(min=-bounds, max=bounds)
