"""Compact per-step records for the v2 learner, and their flat encodings.

A record holds only what fitting and imagination read: the physical state, the
code-advanced memories, the deployed sensor queues, commands, the context
parameters each module needs, the deployed frame, estimator labels and the
interval's tick labels. ``SnapshotLayout`` encodes packed native snapshots
into records; nothing downstream depends on the packed schema.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..formal_observation import SensorReadout
from . import so3
from .physical import INCREMENT_DIM, PhysicalState
from .sensing import SensorQueues, SensorTransforms
from .motion_task import MOTION_TASK_DIM
from .stair_task import STAIR_RECORD_FIELDS
from .stair_source import TOF_RECORD_FIELDS
from .jump_task import JUMP_RECORD_FIELDS, JUMP_TRIGGER_RECORD_FIELDS
from .recovery_task import RECOVERY_RECORD_FIELDS

PHYSICAL_DIM = 3 + 4 + 3 + 3 + 6 + 8
MEMORY_DIM = 6 + 6 + 4
SENSOR_CAPACITY = 3
QUEUE_DIM = SENSOR_CAPACITY * (18 + 5 + 2) + 1
TRANSFORM_DIM = 6 + 3 + 9 + 2 + 2
READOUT_DIM = 4 + 1 + 18 + 1 + 1 + 6
TERRAIN_SCAN_DIM = 77
# Canonical full-vehicle order, independent of the backend's articulation order.
CONTACT_BODIES = ('base_link', 'L_link1', 'L_link2', 'L_link3', 'LLL_link2', 'LLL_link1',
                  'R_link1', 'R_link2', 'R_link3', 'RRR_link2', 'RRR_link1', 'LL_link1',
                  'LL_link2', 'LL_link3', 'LL_link4', 'RR_link1', 'RR_link2', 'RR_link3',
                  'RR_link4', 'M_link1', 'M_link2')
FORBIDDEN_BODIES = ('base_link', 'M_link1', 'M_link2')
WHEEL_BODIES = ('L_link3', 'R_link3')
MODEL_CONTRACT = 'fullticks-knee-v4'
# Support forces, all body patch-normal loads, and wheel/base resultant norms
# have distinct native semantics. Riser contacts must not be called support.
RESULTANT_BODIES = (*WHEEL_BODIES, 'base_link')
CONTACT_LABEL_DIM = 2 + len(CONTACT_BODIES) + len(RESULTANT_BODIES)
PASSIVE_KNEE_JOINTS = ('L_joint2', 'R_jonit2')  # Authored right-side spelling is part of the asset pin.
KNEE_LABEL_DIM = 4  # Native knee positions and velocities, each in left/right order.
TICK_LABEL_DIM = INCREMENT_DIM + CONTACT_LABEL_DIM + KNEE_LABEL_DIM


@dataclass(frozen=True)
class Memory:
    """Motion-relevant controller memory advanced by code, not predicted.

    The runtime gimbal controller is a pure proportional cascade (ki = kd = 0),
    so its targets and mode are its only memory.
    """

    wheel_torque_history: Tensor     # [N, H, 2] newest-first identified queue
    previous_action: Tensor          # [N, 6] last issued action
    gimbal_mode: Tensor              # [N] BODY_FIXED=0, WORLD_HEADING=1, YAW_RATE=2
    gimbal_target_pitch: Tensor      # [N]
    gimbal_target_yaw: Tensor        # [N] controller frame: joint (body mode) or world yaw
    gimbal_yaw_rate_target: Tensor   # [N]


def encode_physical(state: PhysicalState) -> Tensor:
    return torch.cat((state.position, so3.matrix_to_quaternion(state.rotation), state.velocity,
                      state.angular_velocity, state.joint_position, state.joint_velocity), -1)


def decode_physical(flat: Tensor) -> PhysicalState:
    position, quaternion, velocity, angular, q, qd = flat.split((3, 4, 3, 3, 6, 8), -1)
    return PhysicalState(position, so3.quaternion_to_matrix(quaternion), velocity, angular, q, qd)


def encode_memory(memory: Memory) -> Tensor:
    return torch.cat((memory.wheel_torque_history.flatten(-2), memory.previous_action,
                      memory.gimbal_mode.to(memory.previous_action.dtype)[..., None],
                      memory.gimbal_target_pitch[..., None], memory.gimbal_target_yaw[..., None],
                      memory.gimbal_yaw_rate_target[..., None]), -1)


def decode_memory(flat: Tensor) -> Memory:
    history, previous, mode, pitch, yaw, rate = flat.split((6, 6, 1, 1, 1, 1), -1)
    return Memory(history.unflatten(-1, (3, 2)), previous, mode[..., 0].round().long(),
                  pitch[..., 0], yaw[..., 0], rate[..., 0])


def encode_queues(queues: SensorQueues) -> Tensor:
    return torch.cat((queues.values.flatten(-2), queues.gimbal.flatten(-2), queues.heading.flatten(-2),
                      queues.count.to(queues.values.dtype)[..., None]), -1)


def decode_queues(flat: Tensor) -> SensorQueues:
    c = SENSOR_CAPACITY
    values, gimbal, heading, count = flat.split((c * 18, c * 5, c * 2, 1), -1)
    return SensorQueues(values.unflatten(-1, (c, 18)), gimbal.unflatten(-1, (c, 5)),
                        heading.unflatten(-1, (c, 2)), count[..., 0])


def decode_transforms(flat: Tensor) -> SensorTransforms:
    bias, gyro, imu, sign, zero = flat.split((6, 3, 9, 2, 2), -1)
    return SensorTransforms(bias, gyro, imu.unflatten(-1, (3, 3)), sign, zero)


def decode_readout(flat: Tensor) -> SensorReadout:
    ages, gaussian, halfwidth, amplitude_nominal, scale, nominal_q = flat.split((4, 1, 18, 1, 1, 6), -1)
    amplitude = (amplitude_nominal < .5).to(flat.dtype) * scale
    return SensorReadout(ages, gaussian > .5, halfwidth, amplitude, nominal_q)


@dataclass(frozen=True)
class RecordSpec:
    """Field widths of one stored step; ``goal_dim`` and ``context_dim`` follow the schema."""

    goal_dim: int
    sim_context_dim: int

    def fields(self, *, record_terrain: bool = False, record_tasks: bool = False,
               record_stairs: bool = False, record_stair_source: bool = False,
               record_jumps: bool = False, record_recovery: bool = False,
               record_jump_trigger: bool = False) -> dict[str, tuple[int, ...]]:
        if record_stair_source and not (record_terrain and record_tasks and record_stairs):
            raise ValueError('Ideal stair-source records require terrain, command and stair task records')
        if record_jumps and not record_stair_source:
            raise ValueError('Jump records require the complete command, terrain and stair arbitration state')
        if record_recovery and not record_jumps:
            raise ValueError('Recovery records require complete jump and stair arbitration state')
        if record_jump_trigger and not record_jumps:
            raise ValueError('Jump trigger records require the native jump task')
        result = {
            "physical": (PHYSICAL_DIM,), "height": (1,), "scan": (TERRAIN_SCAN_DIM,),
            "memory": (MEMORY_DIM,), "queues": (QUEUE_DIM,), "goal": (self.goal_dim,),
            "sim_context": (self.sim_context_dim,), "motor_strength": (6,),
            "transforms": (TRANSFORM_DIM,), "readout": (READOUT_DIM,), "horizon": (2,),
            "frame": (45,), "estimator_target": (5,), "estimator_valid": (1,),
            "end_physical": (PHYSICAL_DIM,), "end_height": (1,),
            "action": (6,), "reward": (1,), "ticks": (4, TICK_LABEL_DIM),
            "failure_memory": (3,), "failure_conditions": (2,),
            "failure_conditions_valid": (1,),
            "reward_memory": (13,), "support_memory": (10,),
            "controller_parameters": (30,), "reward_context_valid": (1,),
            "reward_route": (3,),  # Native command mode, terrain kind, special gimbal task.
        }
        if record_terrain:
            result['terrain_tile'] = (1,)
            result['task_family'] = (1,)
        if record_tasks:
            result['motion_task'] = (MOTION_TASK_DIM,)
            result['task_record_valid'] = (1,)
        if record_stairs:
            result.update(STAIR_RECORD_FIELDS)
        if record_stair_source:
            result.update(TOF_RECORD_FIELDS)
            result['task_estimate'] = (5,)
            result['stair_source_valid'] = (1,)
        if record_jumps:
            result.update(JUMP_RECORD_FIELDS)
            result['jump_record_valid'] = (1,)
        if record_jump_trigger:
            result.update(JUMP_TRIGGER_RECORD_FIELDS)
        if record_recovery:
            result.update(RECOVERY_RECORD_FIELDS)
            result['recovery_record_valid'] = (1,)
        return result
