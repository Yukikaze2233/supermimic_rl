"""One 20 ms world-model step: learned motion plus code-advanced memories.

``WorldModelV2.step`` is shared by multi-step fitting and policy imagination,
so both see the same memory dynamics: the wheel delay queue is pushed with the
predicted tick wheel speeds, the issued action becomes the previous action and
yaw-rate gimbal targets integrate on the physics clock. Exogenous memories
(gimbal commands set at segment boundaries) come from data while fitting and
are held between known boundaries during imagination.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn

from . import so3
from .dynamics import MotionModel
from .execution import WheelExecution
from .features import MotionInputs
from .records import CONTACT_BODIES, FORBIDDEN_BODIES, RESULTANT_BODIES, WHEEL_BODIES, Memory
from .physical import INCREMENT_DIM, PhysicalState, apply

SIM_CONTEXT_DIM = 44 + 21  # mass/controller/material/COM plus actual inertia traces
INPUT_DIM = 30 + 12 + 19 + 6 + SIM_CONTEXT_DIM + 77


@dataclass(frozen=True)
class Ticks:
    """Absolute per-physics-tick quantities ``[N, 4, ...]`` (tick 4 ends the interval)."""

    state: PhysicalState
    wheel_load: Tensor
    body_normal_load: Tensor
    resultant_contact_force: Tensor
    passive_knee_position: Tensor
    passive_knee_velocity: Tensor

    @classmethod
    def decode(cls, relative: Tensor, start: PhysicalState) -> Ticks:
        motion, support, body, resultant, knee_position, knee_velocity = relative.split(
            (INCREMENT_DIM, 2, len(CONTACT_BODIES), len(RESULTANT_BODIES), 2, 2), -1)
        return cls(apply(start.with_time_axis(), motion), support, body, resultant, knee_position, knee_velocity)

    def with_contact_floor(self, body_floor: Tensor, base_floor: Tensor) -> Ticks:
        """Zero predicted non-wheel body loads below calibrated floors (imagination only).

        Load regression noise near zero otherwise reads as contact at the
        native 1-3 N thresholds. A zero floor (wheel bodies, uncalibrated
        channels) leaves the channel untouched; fitting losses always use the
        raw prediction.
        """
        body = torch.where((self.body_normal_load > body_floor) | (body_floor <= 0), self.body_normal_load, 0.)
        resultant = self.resultant_contact_force.clone()
        base = resultant[..., 2]
        resultant[..., 2] = torch.where((base > base_floor) | (base_floor <= 0), base, 0.)
        return replace(self, body_normal_load=body, resultant_contact_force=resultant)

    @property
    def forbidden_load(self):
        return self.body_normal_load[..., [CONTACT_BODIES.index(name) for name in FORBIDDEN_BODIES]]

    @property
    def wheel_body_load(self):
        return self.body_normal_load[..., [CONTACT_BODIES.index(name) for name in WHEEL_BODIES]]

    @property
    def wheel_contact_force(self):
        return self.resultant_contact_force[..., :2]

    @property
    def base_contact_force(self):
        return self.resultant_contact_force[..., 2]

    @property
    def leg_position(self):
        return self.state.joint_position[..., :4]

    @property
    def joint_velocity(self):
        return self.state.joint_velocity

    @property
    def angular_velocity_body(self):
        return self.state.angular_velocity_body

    @property
    def gravity_body(self):
        return self.state.gravity_body

    @property
    def gimbal_position(self):
        return self.state.joint_position[..., 4:]

    @property
    def yaw(self):
        return so3.yaw(self.state.rotation)


@dataclass(frozen=True)
class Prediction:
    state: PhysicalState
    ticks: Ticks
    memory: Memory
    height_above_ground: Tensor
    terrain_scan: Tensor
    linkage_inside: Tensor
    increment: Tensor
    relative_ticks: Tensor
    tick_height_above_ground: Tensor
    tick_terrain_valid: Tensor | None = None


class FlatTerrain:
    """Ground fixed under each environment; heights shift with the base height.

    Valid on flat tiles only. Sloped, rough and step courses require an atlas
    query at the predicted pose (planned for the terrain milestone).
    """

    @staticmethod
    def advance(height: Tensor, scan: Tensor, start: PhysicalState, end: PhysicalState, *, tile=None):
        rise = (end.position[..., 2:3] - start.position[..., 2:3])
        return height + rise, scan - rise


class WorldModelV2(nn.Module):
    def __init__(self, inputs: MotionInputs, wheels: WheelExecution, *,
                 hidden: tuple[int, ...] = (2048, 2048, 2048), physics_dt_s: float = .005, terrain=None):
        super().__init__()
        self.inputs, self.wheels, self.physics_dt_s = inputs, wheels, physics_dt_s
        self.terrain = FlatTerrain() if terrain is None else terrain
        self.motion = MotionModel(INPUT_DIM, hidden)

    def features(self, state, height, scan, memory, sim_context, action):
        return self.inputs(state, height, scan, memory, sim_context, action)

    def step(self, state: PhysicalState, height: Tensor, scan: Tensor, memory: Memory, sim_context: Tensor,
             motor_strength: Tensor, action: Tensor, *, terrain_tile: Tensor | None = None) -> Prediction:
        features, inside = self.features(state, height, scan, memory, sim_context, action)
        increment, relative = self.motion(features)
        ticks = Ticks.decode(relative, state)
        end = ticks.state.tick(-1)
        next_memory = self.advance_memory(memory, state, ticks, motor_strength, action)
        tick_height, tick_scan = self.terrain.advance(height[..., None, :], scan[..., None, :],
                                                      state.with_time_axis(), ticks.state, tile=terrain_tile)
        return Prediction(end, ticks, next_memory, tick_height[..., -1, :], tick_scan[..., -1, :],
                          inside, increment, relative, tick_height,
                          torch.isfinite(tick_height).all(-1) & torch.isfinite(tick_scan).all(-1))

    def advance_memory(self, memory: Memory, start: PhysicalState, ticks: Ticks,
                       motor_strength: Tensor, action: Tensor) -> Memory:
        """Push four ticks into the wheel queue; tick k uses the wheel speed after tick k-1."""
        target = self.wheels.target(action)
        history = memory.wheel_torque_history
        speeds = torch.cat((start.joint_velocity[..., None, 4:6], ticks.joint_velocity[..., :-1, 4:6]), -2)
        for tick in range(speeds.shape[-2]):
            history = self.wheels.push(history, self.wheels.motor_input(target, speeds[..., tick, :],
                                                                          motor_strength))
        rate = memory.gimbal_mode == 2
        target_yaw = torch.where(rate, so3.wrap(memory.gimbal_target_yaw
                                                + memory.gimbal_yaw_rate_target * self.physics_dt_s
                                                * speeds.shape[-2]), memory.gimbal_target_yaw)
        return replace(memory, wheel_torque_history=history, previous_action=action,
                       gimbal_target_yaw=target_yaw)
