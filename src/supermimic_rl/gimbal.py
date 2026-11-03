"""Independent gimbal servo; chassis reactions are produced by native physics."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class GimbalConfig:
    """SCUT-style relative yaw-rate/pitch-position experiment, not hardware tuning.

    Ranges describe sampled commands, not mechanical stops or motor speed limits.
    The two commands stay fixed within each physical episode.
    """

    yaw_velocity_rad_s: tuple[float, float] = (-1.5, 1.5)
    pitch_position_rad: tuple[float, float] = (-0.15, 0.15)
    yaw_damping: float = 0.5
    pitch_stiffness: float = 20.0
    pitch_damping: float = 0.5

    def __post_init__(self):
        for name in ("yaw_velocity_rad_s", "pitch_position_rad"):
            values = getattr(self, name)
            if (len(values) != 2 or any(not math.isfinite(x) for x in values)
                    or values[0] > values[1]):
                raise ValueError("gimbal command range must be finite and ordered: " + name)
        for name in ("yaw_damping", "pitch_stiffness", "pitch_damping"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError("gimbal servo gains must be finite and nonnegative")


class IndependentGimbalController:
    """Map control coordinates to the actual URDF axes exactly once."""

    def __init__(self, config, num_envs, device, seed, signs, effort_limits):
        self.config = config if isinstance(config, GimbalConfig) else GimbalConfig(**config)
        self.device = torch.device(device)
        self.signs = torch.as_tensor(signs, device=self.device, dtype=torch.float32)
        self.effort_limits = torch.as_tensor(effort_limits, device=self.device, dtype=torch.float32)
        if (self.signs.shape != (2,) or not (self.signs.abs() == 1).all()
                or self.effort_limits.shape != (2,) or not torch.isfinite(self.effort_limits).all()
                or not (self.effort_limits > 0).all()):
            raise ValueError("gimbal requires two signed axes with positive finite effort limits")
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.command = torch.zeros((num_envs, 2), device=self.device)
        self.last_effort = torch.zeros_like(self.command)
        ranges = (self.config.yaw_velocity_rad_s, self.config.pitch_position_rad)
        self.low = self.command.new_tensor([r[0] for r in ranges])
        self.high = self.command.new_tensor([r[1] for r in ranges])

    def reset(self, ids):
        if ids is None:
            ids = torch.arange(len(self.command), device=self.device)
        self.command[ids] = self.low + torch.rand(
            (len(ids), 2), device=self.device, generator=self.generator) * (self.high - self.low)
        self.last_effort[ids] = 0

    def effort(self, position, velocity):
        if position.shape != self.command.shape or velocity.shape != self.command.shape:
            raise ValueError("gimbal feedback must have two native joint axes per environment")
        q, dq = position * self.signs, velocity * self.signs
        yaw = self.config.yaw_damping * (self.command[:, 0] - dq[:, 0])
        pitch = (self.config.pitch_stiffness * (self.command[:, 1] - q[:, 1])
                 - self.config.pitch_damping * dq[:, 1])
        native = torch.stack((yaw, pitch), dim=-1) * self.signs
        self.last_effort = native.clamp(-self.effort_limits, self.effort_limits)
        return self.last_effort
