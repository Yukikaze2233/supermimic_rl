"""Platform caller coordination at one known, completed policy boundary.

This is the native request order, not a contact predictor. The caller supplies
the jump state after interval physics, and all random draws for this boundary.
Recorded future evidence and unknown external packets are deliberately absent.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import torch

from ..course.platform_intent import (
    PlatformIntentPhase,
    PlatformIntentState,
    consume_platform_intent,
    prepare_platform_intent,
)
from .jump_task import (
    JUMP_CLOCK_NAMES,
    JUMP_ID_NAMES,
    JUMP_VALUE_NAMES,
    JumpAdvance,
    JumpTask,
    JumpTaskState,
)
from .stair_source import IdealStairSource, StairSourceBoundary
from .stair_task import StairPacket, StairTask


@dataclass(frozen=True)
class PlatformBoundary:
    intent: PlatformIntentState
    jump: JumpAdvance
    source: StairSourceBoundary

    @property
    def boundary(self):
        return self.source.boundary


@dataclass(frozen=True)
class PlatformTask:
    """Stateless coordinator sharing the existing jump and stair task objects."""

    jump_task: JumpTask
    stair_task: StairTask
    stair_source: IdealStairSource

    def __post_init__(self):
        if self.jump_task.config.trigger is None:
            raise ValueError('Platform intent requires the recorded native take-off gate')
        if self.jump_task.config.request.policy_dt_s != self.stair_task.config.request.policy_dt_s:
            raise ValueError('Platform jump and stair tasks require the same policy boundary')

    def _prepare(self, intent, jump, motion, stair, ground_z, now):
        from transformer_rl.formal.lab_env.skill_fsm import StairState

        # Match JumpTriggerGate.ready before admission. Using the new command
        # here could accept during a height slew from a different support stance.
        gate = self.jump_task.config.trigger
        last_end, _ = jump._trigger_state()
        supported = jump.values[:, JUMP_VALUE_NAMES.index('_last_supported')].bool()
        height = jump.values[:, JUMP_VALUE_NAMES.index('_last_world_height')]
        dt = self.jump_task.config.request.policy_dt_s
        ready = (supported & (height - ground_z - motion.current[:, 3]).abs().le(gate.height_tolerance_m)
                 & (motion.timing[:, 0] + dt >= gate.min_episode_s)
                 & (now - last_end >= gate.cooldown_s))
        return prepare_platform_intent(intent, now_s=now, elapsed_s=motion.timing[:, 0],
            policy_dt_s=dt, timeout_s=self.jump_task.config.request.timeout_s,
            physical_id=jump.physical_id,
            physical_accepted_at_s=jump.clock[:, JUMP_CLOCK_NAMES.index('_physical_accepted_at_s')],
            stair_active_id=stair.ids[:, 1], stair_idle=stair.ids[:, 2] == int(StairState.NORMAL),
            native_ready=ready, admitted_height=motion.current[:, 3],
            max_seen_id=jump.ids[:, JUMP_ID_NAMES.index('protocol.max_seen_id')],
            next_id=jump.ids[:, JUMP_ID_NAMES.index('next_id')])

    def advance(self, intent: PlatformIntentState, jump: JumpTaskState, *, motion, stair, source,
                physical, wheel_centres, wheel_radius, terrain_tile, queues, readout, goal,
                estimate, task_family, ground_z, interval_uniform, latency, draws,
                retain_peak_gradient=False, request_gimbal_exclusion=False) -> PlatformBoundary:
        """Advance admission -> jump -> caller consumption -> ToF/stair.

        ``jump`` must contain the just-completed interval's physical evidence;
        ``motion`` still has its old admitted command and episode clock. All
        geometry, including ``ground_z``, is tile-local. The returned boundary
        keeps the original command-resampling/known-future rules: ownership of
        a platform plan does not make a future sampled velocity command known.

        Explicit mixed ownership consumes the existing full-batch jump draw,
        including when all rows are owned or inactive. No RNG is accessed here.
        Caller plan/reset and landing-state propagation remain with the rollout.
        """
        from ..course.gimbal_arbitration import request_owns_chassis

        now = self.jump_task.boundary_time(jump)
        prepared = self._prepare(intent, jump, motion, stair, ground_z, now)
        following, admitted = motion.advance_admission(queues, readout,
            self.jump_task.config.request.policy_dt_s,
            height_override=prepared.height_override_m,
            height_override_mask=prepared.height_override_mask)
        jumped = self.jump_task.advance(jump, now_s=now, estimate=estimate, admitted=admitted,
            base_mode=following.route[:, 3].long(), task_family=task_family,
            elapsed_s=following.timing[:, 0], ground_z=ground_z,
            blocked_by_stair=stair.ids[:, 1] >= 0, interval_uniform=interval_uniform,
            packet=prepared.jump_packet, packet_owned=prepared.packet_owned,
            previous_command_height=motion.current[:, 3], previous_elapsed_s=motion.timing[:, 0],
            retain_peak_gradient=retain_peak_gradient)
        condition = SimpleNamespace(active_id=jumped.state.active_id, response=jumped.response,
                                    finished_id=jumped.finished_id, end_reason=jumped.end_reason)
        following_intent = consume_platform_intent(prepared, condition, physical_id=jumped.state.physical_id)
        # NativePlatformIntent applies the cancel after consuming the actual
        # jump result. Expired/finished callers must not cancel another request.
        cancel = torch.where(following_intent.phase == int(PlatformIntentPhase.PENDING),
                             prepared.stair_packet.cancel_id, -1)
        packet = replace(StairPacket.empty(len(jump.ids), jump.ids.device), cancel_id=cancel)
        advanced = self.stair_source.advance(source, stair, task=self.stair_task, motion=motion,
            physical=physical, wheel_centres=wheel_centres, wheel_radius=wheel_radius,
            terrain_tile=terrain_tile, queues=queues, readout=readout, goal=goal, estimate=estimate,
            physics_dt_s=self.jump_task.config.physics_dt_s, latency=latency, draws=draws,
            external=packet, reservation=following_intent.reservation,
            admission=(replace(following, current=jumped.command), jumped.command),
            base_mode=jumped.mode, jump_active=jumped.state.active_id >= 0,
            request_height=jumped.request_height, request_clock=jumped.request_clock,
            request_owned=(None if not request_gimbal_exclusion else
                           request_owns_chassis(task_family, jumped.state.physical_id)))
        return PlatformBoundary(following_intent, jumped, advanced)
