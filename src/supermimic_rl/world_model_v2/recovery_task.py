"""Recorded recovery lifecycle, sharing the native policy/physics separation.

The actor owns all recovery actions. This module only advances task memory and
the recovery-to-tracking command boundary; it generates no actions or rewards.
Physical evidence certifies success separately from the prior-estimate handoff.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .motion_task import MotionTask
from .physical import PhysicalState

if TYPE_CHECKING:
    from transformer_rl.formal.lab_env.recovery_lifecycle import RecoveryLifecycleConfig
    from transformer_rl.formal.tasks.protocol import CompletedEstimate


RECOVERY_FLAG_NAMES = ('active', '_verified')
RECOVERY_CLOCK_NAMES = (
    'now_s', 'policy_elapsed_s', 'estimated_stable_s', 'physical_stable_s', '_last_estimate_s',
)
RECOVERY_RECORD_FIELDS = {'recovery_flags': (2,), 'recovery_clock': (5,)}
RECOVERY_RECORD_DTYPES = {'recovery_flags': torch.bool, 'recovery_clock': torch.float64}
RECOVERY_FEATURE_DIM = 7


@dataclass(frozen=True)
class RecoveryHandoffConfig:
    duration_s: float
    forward_m_s: float
    yaw_rad_s: float

    def __post_init__(self):
        if any(not math.isfinite(value) or value <= 0 for value in vars(self).values()):
            raise ValueError('Recovery handoff uses the captured positive command-pool bounds')


@dataclass(frozen=True)
class RecoveryTaskConfig:
    lifecycle: RecoveryLifecycleConfig
    handoff: RecoveryHandoffConfig | None

    @classmethod
    def capture(cls, native):
        course = native._terrain_course
        pool = course.tasks.command_pool
        handoff = None if pool is None else RecoveryHandoffConfig(
            pool.config.handoff_s, pool.config.handoff_forward_m_s, pool.config.handoff_yaw_rad_s)
        result = cls(course.recovery.config, handoff)
        if float(native.physics_dt) != result.lifecycle.physics_dt_s:
            raise ValueError('Recovery lifecycle and native physics clocks differ')
        return result

    @classmethod
    def from_dict(cls, value):
        from transformer_rl.formal.lab_env.recovery_lifecycle import RecoveryLifecycleConfig

        handoff = value['handoff']
        return cls(RecoveryLifecycleConfig(**value['lifecycle']),
                   None if handoff is None else RecoveryHandoffConfig(**handoff))

    def __post_init__(self):
        if self.lifecycle.policy_dt_s != .02 or self.lifecycle.physics_dt_s != .005:
            raise ValueError('Recovery imagination uses the recorded 200/50 Hz four-tick contract')


@dataclass(frozen=True)
class RecoveryTaskState:
    flags: Tensor
    clock: Tensor

    @classmethod
    @torch.no_grad()
    def capture(cls, native):
        return cls._pack(native._clock(), native._terrain_course.recovery)

    @classmethod
    def _pack(cls, now, owner):
        return cls(torch.stack([getattr(owner, name) for name in RECOVERY_FLAG_NAMES], -1).clone(),
                   torch.stack([now, *(getattr(owner, name) for name in RECOVERY_CLOCK_NAMES[1:])],
                               -1).clone())

    @classmethod
    def decode(cls, record):
        return cls(*(record[name] for name in RECOVERY_RECORD_FIELDS))

    def record(self):
        return dict(zip(RECOVERY_RECORD_FIELDS, (self.flags, self.clock)))

    @property
    def active(self):
        return self.flags[:, 0]

    @property
    def verified(self):
        return self.flags[:, 1]

    @property
    def now_s(self):
        return self.clock[:, 0]

    def critic_features(self, config: RecoveryTaskConfig):
        """Relative ages and causal latches; no absolute world/policy clock."""
        valid = torch.isfinite(self.clock[:, 4])
        age = torch.where(valid, self.now_s - self.clock[:, 4], 0.)
        lifecycle = config.lifecycle
        return torch.stack((self.active, self.verified,
            self.clock[:, 1] / lifecycle.timeout_s,
            self.clock[:, 2] / lifecycle.stable_s,
            self.clock[:, 3] / lifecycle.stable_s,
            valid, age / lifecycle.estimate_max_age_s), -1).float()

    def _restore(self, config: RecoveryTaskConfig):
        from transformer_rl.formal.lab_env.recovery_lifecycle import RecoveryLifecycle

        n, device = len(self.flags), self.flags.device
        for name, value in self.record().items():
            if (value.shape != (n, *RECOVERY_RECORD_FIELDS[name])
                    or value.device != device or value.dtype != RECOVERY_RECORD_DTYPES[name]):
                raise ValueError('Invalid recorded recovery field: ' + name)
        owner = RecoveryLifecycle(n, device=device, config=config.lifecycle)
        for column, name in enumerate(RECOVERY_FLAG_NAMES):
            setattr(owner, name, self.flags[:, column].clone())
        for column, name in enumerate(RECOVERY_CLOCK_NAMES[1:], 1):
            setattr(owner, name, self.clock[:, column].clone())
        return owner


@dataclass(frozen=True)
class RecoveryBoundary:
    state: RecoveryTaskState
    handoff: Tensor
    timeout: Tensor                  # Native truncation, not a failure event.
    verified_success: Tensor         # Diagnostic rising edge; never a reward payment.


@dataclass(frozen=True)
class RecoveryMotionBoundary:
    motion: MotionTask
    task_family: Tensor
    segment_end: Tensor              # [N, 1], in the native float32 episode clock.


class RecoveryTask:
    def __init__(self, config: RecoveryTaskConfig):
        self.config = config

    def boundary_time(self, state: RecoveryTaskState):
        dt = self.config.lifecycle.physics_dt_s
        return ((state.now_s / dt).round() + 4) * dt

    @torch.no_grad()
    def observe_interval(self, state: RecoveryTaskState, *, physical: PhysicalState,
                         wheel_support_load: Tensor, base_normal_load: Tensor,
                         base_com_offset: Tensor, command_height: Tensor,
                         ground_z: Tensor, policy_owned: Tensor) -> RecoveryTaskState:
        """Use all four real/predicted ticks with the old interval's command.

        Wheel evidence is the upward patch-normal support load the native
        lifecycle receives (``wheel_ground_force_n``); base evidence is its
        normal load. COM vertical speed includes the rotational offset.
        Lifecycle predicates stop gradients; shared physical rewards do not.
        """
        n = len(state.flags)
        expected = ((physical.position, (n, 4, 3)), (wheel_support_load, (n, 4, 2)),
                    (base_normal_load, (n, 4)), (base_com_offset, (n, 3)),
                    (command_height, (n,)), (ground_z, (n, 4)))
        if any(value.shape != shape or value.device != state.flags.device for value, shape in expected):
            raise ValueError('Recovery evidence needs four ticks and the recorded native row layout')
        if (policy_owned.shape not in ((n,), (n, 4)) or policy_owned.dtype != torch.bool
                or policy_owned.device != state.flags.device):
            raise ValueError('Recovery evidence requires interval or per-tick policy ownership')
        owner = state._restore(self.config)
        offset_world = (physical.rotation @ base_com_offset[:, None, :, None]).squeeze(-1)
        com_velocity = physical.velocity + torch.linalg.cross(physical.angular_velocity, offset_world)
        gravity = physical.gravity_body
        tilt = torch.atan2(gravity[..., :2].norm(dim=-1), -gravity[..., 2])
        angular_speed = physical.angular_velocity_body.norm(dim=-1)
        for tick in range(4):
            owner.observe_physics(
                height_m=physical.position[:, tick, 2] - ground_z[:, tick],
                command_height_m=command_height, tilt_rad=tilt[:, tick],
                angular_speed_rad_s=angular_speed[:, tick], velocity_z_m_s=com_velocity[:, tick, 2],
                wheel_load_n=wheel_support_load[:, tick], base_load_n=base_normal_load[:, tick],
                policy_owned=policy_owned if policy_owned.ndim == 1 else policy_owned[:, tick])
        return RecoveryTaskState._pack(state.now_s, owner)

    @torch.no_grad()
    def finish_interval(self, state: RecoveryTaskState, *, now_s: Tensor,
                        estimate: CompletedEstimate | None, policy_owned: Tensor,
                        failed: Tensor) -> RecoveryBoundary:
        """After physical failure latching, before next command admission.

        ``estimate`` is the preceding completed noisy actor estimate. Physical
        certification cannot select the next mode. A timeout is reported
        separately because the native environment treats it as a time limit.
        """
        owner = state._restore(self.config)
        boundary = owner.finish_interval(now_s, estimate, policy_owned=policy_owned, failed=failed)
        return RecoveryBoundary(RecoveryTaskState._pack(now_s, owner), boundary.handoff,
                                boundary.timeout, owner._verified & ~state.verified)

    def apply_handoff(self, boundary: RecoveryBoundary, motion: MotionTask, *,
                      task_family: Tensor, segment_end: Tensor,
                      handoff_uniform: Tensor | None = None) -> RecoveryMotionBoundary:
        """Reuse native begin_tracking with prepared whole-batch randomness.

        Call before MotionTask.advance_admission. Native begin_tracking leaves
        stream.current unchanged until that admission, and anchors the new
        segment deadline to the old episode elapsed time. Continuous command
        height remains differentiable; random planar targets do not.
        """
        from transformer_rl.formal.control.observation import CommandMode
        from transformer_rl.formal.lab_env.course_tasks import LabCourseTasks

        n, device = len(boundary.handoff), boundary.handoff.device
        if (task_family.shape != (n,) or task_family.dtype != torch.long
                or task_family.device != device or segment_end.shape != (n, 1)
                or segment_end.dtype != torch.float32 or segment_end.device != device):
            raise ValueError('Recovery handoff needs native family and segment deadline rows')
        managed = self.config.handoff is not None
        if ((motion.route[:, 4] > .5) != managed).any():
            raise ValueError('Recovery handoff configuration differs from the recorded command pool')
        pool = None
        if managed:
            if (handoff_uniform is None or handoff_uniform.shape != (n, 2)
                    or handoff_uniform.dtype != torch.float32 or handoff_uniform.device != device
                    or not torch.isfinite(handoff_uniform).all()
                    or ((handoff_uniform < 0) | (handoff_uniform >= 1)).any()):
                raise ValueError('Recovery handoff requires prepared [N,2] uniform draws in [0,1)')
            config = self.config.handoff
            rows = boundary.handoff.nonzero().flatten()
            planar = (handoff_uniform.detach()[rows] * 2 - 1) * handoff_uniform.new_tensor(
                [config.forward_m_s, config.yaw_rad_s])
            # LabCourseTasks owns the mutation semantics; only its random draw
            # is replaced by the caller's common microbatch realization.
            pool = SimpleNamespace(config=SimpleNamespace(handoff_s=config.duration_s),
                handoff=lambda count, generator, device: (planar[:, 0], planar[:, 1]))
        owner = SimpleNamespace(
            command_pool=pool, generator=None, device=device, family=task_family.clone(),
            stream=SimpleNamespace(current=motion.current.clone()),
            trajectory=SimpleNamespace(start=motion.start.clone(), end=motion.end.clone(),
                                       kind=motion.route[:, 0].long().clone()),
            elapsed=motion.timing[:, 0].clone(), segment_clock=motion.timing[:, 1].clone(),
            segment_end_s=segment_end[:, 0].clone(),
            heading_enabled=(motion.heading[:, 0] > .5).clone(),
            heading_valid=(motion.heading[:, 1] > .5).clone())
        LabCourseTasks.begin_tracking(owner, boundary.handoff)
        base_mode = torch.where(boundary.handoff, int(CommandMode.NORMAL), motion.route[:, 3])
        route = torch.cat((owner.trajectory.kind[:, None], motion.route[:, 1:3],
                           base_mode[:, None], motion.route[:, 4:]), -1)
        following = replace(motion, start=owner.trajectory.start, end=owner.trajectory.end,
            timing=torch.cat((motion.timing[:, :1], owner.segment_clock[:, None], motion.timing[:, 2:]), -1),
            route=route, heading=torch.cat((owner.heading_enabled[:, None], owner.heading_valid[:, None],
                                           motion.heading[:, 2:]), -1))
        return RecoveryMotionBoundary(following, owner.family, owner.segment_end_s[:, None])
