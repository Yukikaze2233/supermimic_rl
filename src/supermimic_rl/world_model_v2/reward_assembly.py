"""Assemble and integrate the shared native reward on every predicted physics tick.

Task references and ground queries are explicit inputs. This module does not
invent a task phase, freeze an unknown terrain height, or qualify ideal linkage
geometry as measured simulator geometry. The caller must provide that geometry.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace

import torch
from torch import Tensor

from transformer_rl.formal.tasks.reward import (
    CommandMode, RewardPhysics, RewardReference, RoughDashRewardReference, TerrainRewardReference,
)
from ..reward_kernel import RewardHistory, RewardParameters, physical_reward_densities
from .execution import LegExecution, WheelExecution
from .geometry import WheelGeometry
from .physical import PhysicalState
from .support import SupportInputs, SupportState, advance_support


@dataclass(frozen=True)
class RewardMemory:
    previous_action: Tensor
    previous_action2: Tensor
    action_count: Tensor
    support: SupportState

    @classmethod
    def decode(cls, flat: Tensor):
        previous, previous2, count, support = flat.split((6, 6, 1, 10), -1)
        return cls(previous, previous2, count[..., 0], SupportState.decode(support))

    def encode(self):
        return torch.cat((self.previous_action, self.previous_action2, self.action_count[..., None],
                          self.support.encode()), -1)


@dataclass(frozen=True)
class RewardTick:
    state: PhysicalState
    geometry: WheelGeometry
    root_ground_z: Tensor
    wheel_ground_z: Tensor
    body_normal_load: Tensor
    wheel_support_load: Tensor
    resultant_contact_force: Tensor  # left wheel, right wheel, base resultant norms
    support_valid: Tensor
    terrain_eligible: Tensor
    reference: RewardReference       # exact task producer's command/phase/targets at this tick
    timestamp_s: Tensor
    episode_id: Tensor
    control_epoch: Tensor
    policy_owned: Tensor


@dataclass(frozen=True)
class RewardInterval:
    physical_reward: Tensor
    components: dict[str, Tensor]    # integrated contributions, not per-second densities
    failure_scale: Tensor
    memory: RewardMemory
    wheel_torque_history: Tensor
    invalid: Tensor
    termination_height: Tensor | None = None  # Task/terrain producer's [N, K, 1] height, not a learned label.
    jump_state: object | None = None
    success_credited: Tensor | None = None

    def finish(self, terminated: Tensor, success_credited: Tensor, parameters: RewardParameters) -> Tensor:
        """One-shot events; the task producer owns success identity/ownership checks."""
        success = success_credited & ~terminated
        return (self.physical_reward + terminated * parameters.tuning.failure_event * self.failure_scale
                + success * parameters.tuning.success_event)


class RewardAssembly:
    def __init__(self, parameters: RewardParameters, available_terms: tuple[str, ...],
                 knee_limits: tuple[float, float], support_config, wheels: WheelExecution):
        if 'contact_slip' in available_terms:
            raise ValueError('Imagined contact slip requires an actual contact-point tangent-velocity producer')
        if parameters.tuning.profile != support_config.profile:
            raise ValueError('Reward and support clocks must use the same pinned profile')
        self.parameters, self.available_terms = parameters, available_terms
        self.knee_limits, self.support_config, self.wheels = knee_limits, support_config, wheels

    def physics(self, tick: RewardTick, effort: Tensor, admission_correction: Tensor) -> RewardPhysics:
        state, geometry = tick.state, tick.geometry
        r = state.rotation
        # atan2 avoids asin's infinite derivative at pitch +/-pi/2. Roll is
        # undefined there; use a constant branch at the exact singularity.
        horizontal = r[..., 2, 1:].norm(dim=-1)
        roll_y = torch.where(horizontal > 0, r[..., 2, 1], 0.)
        roll_x = torch.where(horizontal > 0, r[..., 2, 2], 1.)
        roll_pitch = torch.stack((torch.atan2(roll_y, roll_x), torch.atan2(-r[..., 2, 0], horizontal)), -1)
        # The selected native LabEnv calls LegBoundaryGeometry with 0.05 rad.
        boundary = torch.cat(((.05 - geometry.knee_clearance_rad).clamp_min(0), admission_correction), -1)
        return RewardPhysics(
            tick.timestamp_s, tick.episode_id, tick.control_epoch, tick.policy_owned,
            state.velocity_body, state.velocity, state.position[..., 2],
            state.position[..., 2] - tick.root_ground_z, roll_pitch, state.gravity_body,
            state.angular_velocity_body, state.joint_velocity[..., :6], effort, tick.body_normal_load,
            torch.zeros_like(tick.wheel_support_load), boundary,
            wheel_position_base_m=geometry.link_position_base,
            wheel_forward_velocity_rad_s=geometry.forward_velocity_rad_s,
            wheel_contact_force_n=tick.resultant_contact_force[..., :2],
            passive_knee_position_rad=geometry.knee_angle_rad,
            passive_knee_velocity_rad_s=geometry.knee_velocity_rad_s,
            passive_knee_limits_rad=state.position.new_tensor([self.knee_limits] * 2),
            available_terms=self.available_terms)

    def advance_reference(self, tick: RewardTick, support: SupportState) -> tuple[RewardReference, SupportState]:
        reference, geometry = tick.reference, tick.geometry
        ordinary = (reference.mode != int(CommandMode.JUMP)) & (reference.mode != int(CommandMode.RECOVERY))
        evidence = SupportInputs(tick.support_valid, ordinary, tick.state.position[..., 2],
            geometry.centre_world[..., 2], geometry.bottom_world_z, tick.root_ground_z, tick.wheel_ground_z,
            tick.wheel_support_load, tick.resultant_contact_force[..., :2], tick.resultant_contact_force[..., 2])
        phase = advance_support(support, evidence, self.support_config, self.parameters.physics_dt_s)
        terrain_active = tick.terrain_eligible & ordinary & tick.support_valid
        terrain = TerrainRewardReference(terrain_active, geometry.extension_m,
                                          phase.airborne & terrain_active, phase.landing & terrain_active)
        rough = None
        if self.support_config.profile == 'rough_dash':
            rough = RoughDashRewardReference(phase.state.phase_active & ordinary & tick.support_valid,
                geometry.extension_m, phase.state.wheel_contact_s, phase.state.base_contact_s)
        return replace(reference, terrain=terrain, rough_dash=rough), phase.state

    @classmethod
    def _stack_rows(cls, records, *, static_fields=()):
        """Concatenate tick-major rows, retaining explicitly static schema fields.

        A changing optional-reference layout uses the original per-tick kernel
        below. It is uncommon in production, but zero-filling absent references
        would change the set of reward components and their validation semantics.
        """
        values = {}
        for field in fields(records[0]):
            parts = tuple(getattr(record, field.name) for record in records)
            first = parts[0]
            if field.name in static_fields:
                values[field.name] = first
            elif isinstance(first, Tensor):
                if not all(isinstance(part, Tensor) for part in parts):
                    return None
                values[field.name] = torch.cat(parts, dim=0)
            elif is_dataclass(first):
                if not all(type(part) is type(first) for part in parts):
                    return None
                value = cls._stack_rows(parts)
                if value is None:
                    return None
                values[field.name] = value
            elif first is None:
                if any(part is not None for part in parts):
                    return None
                values[field.name] = None
            else:
                values[field.name] = first
        return type(records[0])(**values)

    def integrate(self, start: PhysicalState, ticks: tuple[RewardTick, ...], action: Tensor,
                  memory: RewardMemory, wheel_torque_history: Tensor, legs: LegExecution,
                  pd_encoder_bias: Tensor, motor_strength: Tensor) -> RewardInterval:
        expected = round(self.parameters.tuning.policy_dt_s / self.parameters.physics_dt_s)
        if len(ticks) != expected:
            raise ValueError('Reward integration requires all physical ticks of one policy interval')
        target, correction = legs.decode_target(action, start.joint_position[..., :4] + pd_encoder_bias[..., :4])
        previous, support, queue = start, memory.support, wheel_torque_history
        components = {}
        physics_ticks, reference_ticks, histories = [], [], []
        invalid = torch.zeros(action.shape[0], dtype=torch.bool, device=action.device)
        history = RewardHistory.from_actions(start.joint_velocity[..., :6], action, memory.previous_action,
                                              memory.previous_action2, memory.action_count)
        for tick in ticks:
            wheel_speed = previous.joint_velocity[..., 4:6]
            queue = self.wheels.push(queue, self.wheels.motor_input(self.wheels.target(action), wheel_speed,
                                                                   motor_strength[..., 4:6]))
            leg_effort = legs.motor_effort(target, previous.joint_position[..., :4] + pd_encoder_bias[..., :4],
                                           previous.joint_velocity[..., :4], motor_strength)
            effort = torch.cat((leg_effort, self.wheels.applied(queue, wheel_speed)), -1)
            physics = self.physics(tick, effort, correction)
            reference, support = self.advance_reference(tick, support)
            physics_ticks.append(physics)
            reference_ticks.append(reference)
            histories.append(history)
            invalid |= ~tick.support_valid | ~tick.geometry.valid | ~tick.policy_owned
            invalid |= ((tick.body_normal_load < 0).any(-1) | (tick.resultant_contact_force < 0).any(-1)
                        | (tick.wheel_support_load < 0).any(-1))
            history = replace(history, velocity_rad_s=tick.state.joint_velocity[..., :6])
            previous = tick.state
        references = self._stack_rows(reference_ticks)
        if references is None:
            # Preserve support for callers whose optional task producers change
            # within the interval. Native locomotion uses one stable layout.
            for physics, reference, history in zip(physics_ticks, reference_ticks, histories):
                densities = physical_reward_densities(physics, reference, history, self.parameters)
                for name, value in densities.components.items():
                    contribution = value.double() * self.parameters.physics_dt_s
                    components[name] = components.get(name, 0.) + contribution
                invalid |= densities.invalid
            failure_scale = densities.failure_scale
        else:
            physics = self._stack_rows(physics_ticks, static_fields=(
                'passive_knee_limits_rad', 'available_terms', 'velocity_reference',
                'effort_reference', 'slip_reference', 'boundary_reference'))
            parameters = replace(self.parameters, mass_kg=self.parameters.mass_kg.repeat(expected))
            densities = physical_reward_densities(physics, references, self._stack_rows(histories), parameters)
            # Keep the exact original double-precision tick accumulation order.
            # Only the stateless elementwise densities share one larger batch.
            for name, values in densities.components.items():
                accumulated = 0.
                for value in values.reshape(expected, action.shape[0]).unbind(0):
                    accumulated = accumulated + value.double() * self.parameters.physics_dt_s
                components[name] = accumulated
            invalid |= densities.invalid.reshape(expected, action.shape[0]).any(0)
            failure_scale = densities.failure_scale.reshape(expected, action.shape[0])[-1]
        next_memory = RewardMemory(action, memory.previous_action, (memory.action_count + 1).clamp_max(2), support)
        return RewardInterval(torch.stack(tuple(components.values())).sum(0), components,
                              failure_scale, next_memory, queue, invalid)
