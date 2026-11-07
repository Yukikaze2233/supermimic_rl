"""Recorded stair requests and their shared deployment state machine.

The caller supplies actual operator/sensor packets and the preceding deployed
estimate. This module does not manufacture requests from terrain truth, assume
future operator input, or equate an estimated release with physical success.
Protocol IDs and clocks retain their native int64/float64 precision in replay.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .motion_task import MotionTask
    from transformer_rl.formal.lab_env.course_tasks import StairRequestConfig
    from transformer_rl.formal.lab_env.skill_fsm import StairFSMConfig, StairHeightAssistConfig
    from transformer_rl.formal.tasks.protocol import CompletedEstimate


STAIR_RECORD_FIELDS = {
    'stair_ids': (4,),         # Watermark, active request, FSM state, remaining hint ticks.
    'stair_clock': (3,),       # Boundary, acceptance, previous protocol call (float64).
    'stair_values': (6,),      # Request height; FSM clocks/heights; permanent contact exception.
    'stair_parameters': (16,),
}
STAIR_RECORD_DTYPES = {'stair_ids': torch.int64, 'stair_clock': torch.float64}
STAIR_FEATURE_DIM = 30


@dataclass(frozen=True)
class StairTaskConfig:
    request: StairRequestConfig
    fsm: StairFSMConfig
    height_assist: StairHeightAssistConfig | None
    estimate_max_age_s: float
    use_mode_labels: bool

    @classmethod
    def capture(cls, native):
        course = native._terrain_course
        policy = getattr(native.cfg, 'terrain_policy_spec', None)
        return cls(course.stair_requests.config, native._skill_fsm.stair.config,
                   getattr(native.cfg, 'stair_height_assist_spec', None),
                   course.tasks.protocol.config.estimate_max_age_s,
                   policy is None or policy.use_mode_labels)

    @classmethod
    def from_dict(cls, value):
        from transformer_rl.formal.lab_env.course_tasks import StairRequestConfig
        from transformer_rl.formal.lab_env.skill_fsm import StairFSMConfig, StairHeightAssistConfig

        return cls(StairRequestConfig(**value['request']), StairFSMConfig(**value['fsm']),
                   None if value['height_assist'] is None else StairHeightAssistConfig(**value['height_assist']),
                   value['estimate_max_age_s'], value['use_mode_labels'])

    def signature(self, like):
        """Only fields consumed by deployment_step, in a versioned fixed order."""
        request, fsm, assist = self.request, self.fsm, self.height_assist
        return like.new_tensor([
            request.policy_dt_s, request.min_height_m, request.max_height_m, request.timeout_s,
            fsm.policy_dt_s, fsm.curb_max_m, fsm.step_up_max_m, fsm.stair_max_m,
            fsm.ledge_max_m, fsm.failure_tilt_rad,
            *([0., 0., 0., 0.] if assist is None else
              [assist.enabled, assist.raise_by_m, assist.max_height_m, assist.hold_s]),
            self.estimate_max_age_s, self.use_mode_labels,
        ])


@dataclass(frozen=True)
class StairTaskState:
    ids: Tensor
    clock: Tensor
    values: Tensor
    parameters: Tensor

    @classmethod
    @torch.no_grad()
    def capture(cls, native):
        course, fsm = native._terrain_course, native._skill_fsm.stair
        protocol = course.stair_requests
        values = torch.stack((protocol.height_m, fsm.state_time, fsm.success_time,
                              fsm.reference_height, fsm.target_height,
                              course._stair_course_contact_allowed), -1).float()
        return cls(torch.stack((protocol.max_seen_id, protocol.active_id, fsm.state,
                                fsm._height_assist_steps_remaining), -1).clone(),
                   torch.stack((native._clock(), protocol.accepted_at_s, protocol.last_time_s), -1).clone(),
                   values, StairTaskConfig.capture(native).signature(values).expand(len(values), -1).clone())

    @classmethod
    def decode(cls, record):
        return cls(*(record[name] for name in STAIR_RECORD_FIELDS))

    def record(self):
        return dict(zip(STAIR_RECORD_FIELDS, (self.ids, self.clock, self.values, self.parameters)))

    def critic_features(self):
        """Relative clocks and request state, without absolute time or large IDs."""
        active = self.ids[:, 1] >= 0
        age = torch.where(active, self.clock[:, 0] - self.clock[:, 1], 0.)
        since = torch.where(torch.isfinite(self.clock[:, 2]), self.clock[:, 0] - self.clock[:, 2], 0.)
        return torch.cat((active[:, None], torch.nn.functional.one_hot(self.ids[:, 2], 4),
            self.ids[:, 3:4], age[:, None], since[:, None], self.values, self.parameters), -1).to(self.values)


@dataclass(frozen=True)
class StairPacket:
    """One known boundary's request; completion must come from its actual source."""
    request_id: Tensor
    height_m: Tensor
    cancel_id: Tensor
    completed_id: Tensor
    safety_abort: Tensor

    @classmethod
    def empty(cls, num_envs, device):
        ids = torch.full((num_envs,), -1, dtype=torch.int64, device=device)
        return cls(ids, torch.zeros(num_envs, device=device), ids.clone(), ids.clone(),
                   torch.zeros(num_envs, dtype=torch.bool, device=device))


@dataclass(frozen=True)
class StairAdvance:
    state: StairTaskState
    command: Tensor
    mode: Tensor
    body_contact_allowed: Tensor
    response: Tensor
    finished_id: Tensor
    end_reason: Tensor
    entered: Tensor
    exited: Tensor
    protocol_active_id: Tensor  # Before an immediate FSM exit clears the protocol.


@dataclass(frozen=True)
class StairMotionBoundary:
    motion: MotionTask
    stair: StairAdvance
    goal: Tensor
    failure_conditions: Tensor


class StairTask:
    def __init__(self, config: StairTaskConfig):
        self.config = config

    @staticmethod
    def completed_estimate(history, estimate, available_at_s):
        """Use the same noisy 35D slice and SI estimate as CourseAdapter."""
        from transformer_rl.formal.tasks.protocol import CompletedEstimate

        raw = torch.cat((estimate[..., :3], torch.zeros_like(estimate[..., :3]), estimate[..., 3:5]), -1)
        return CompletedEstimate._from_actor_observation(history[:, -1, :35], raw, available_at_s)

    def boundary_time(self, state, physics_dt_s):
        dt = self.config.request.policy_dt_s
        substeps = round(dt / physics_dt_s)
        if substeps < 1 or abs(substeps * physics_dt_s - dt) > 1e-12:
            raise ValueError('Task boundary requires the actual integer physics decimation')
        return ((state.clock[:, 0] / physics_dt_s).round() + substeps) * physics_dt_s

    def estimate_inputs(self, now_s, estimate, like):
        if estimate is None:
            return (torch.zeros(len(like), dtype=torch.bool, device=like.device),
                    like.new_zeros(len(like), 2), like.new_zeros(len(like)))
        age = now_s - estimate.available_at_s
        contact, tilt = estimate.wheel_contact_probability, estimate.tilt_rad
        ready = (estimate.valid & torch.isfinite(age) & (age > 0)
                 & (age <= self.config.estimate_max_age_s + 1e-9)
                 & torch.isfinite(contact).all(-1) & ((contact >= 0) & (contact <= 1)).all(-1)
                 & torch.isfinite(tilt) & (tilt >= 0))
        return ready, torch.where(ready[:, None], contact, 0.), torch.where(ready, tilt, 0.)

    def advance_motion(self, state, packet, *, motion, queues, readout, goal, estimate, physics_dt_s,
                       admission=None, base_mode=None, jump_active=None, request_height=None, request_clock=None,
                       request_owned=None):
        """Compose native command admission -> request FSM -> gimbal task.

        Called after the preceding interval's reward/termination. The returned
        exception belongs to the next interval, and must also reach its reward.
        Packet generation and future resampling belong to their separate source.
        """
        from transformer_rl.formal.control.observation import CommandMode

        if admission is None and ((motion.route[:, 3] > int(CommandMode.SLOPE)).any()
                                  or (goal[:, 7:9] > .5).any()):
            raise ValueError('Stair motion composition requires ordinary commands; jump/recovery need their own task state')
        dt = self.config.request.policy_dt_s
        # Native _clock() multiplies an integer physical tick, not repeated
        # float addition. Preserve that grid at exact timeout boundaries.
        now = self.boundary_time(state, physics_dt_s)
        following, admitted = (motion.advance_admission(queues, readout, dt) if admission is None else admission)
        base_mode = following.route[:, 3].long() if base_mode is None else base_mode
        jump_active = base_mode == int(CommandMode.JUMP) if jump_active is None else jump_active
        result = self.advance(state, packet, now_s=now, estimate=estimate, admitted=admitted,
                              base_mode=base_mode, jump_active=jump_active)
        following, next_goal = following.finish_command(result.command, goal, mode=result.mode,
                                                         request_owned=request_owned)
        if request_height is not None:
            if request_clock is None:
                raise ValueError('Jump goal needs both the native requested height and visible clock')
            next_goal = torch.cat((next_goal[:, :9], request_height[:, None] * 5.,
                                   request_clock[:, None].clamp_max(5), next_goal[:, 11:]), -1)
        conditions = torch.stack((result.mode == int(CommandMode.RECOVERY), result.body_contact_allowed), -1)
        return StairMotionBoundary(following, result, next_goal, conditions.to(goal.dtype))

    def advance(self, state: StairTaskState, packet: StairPacket, *, now_s: Tensor,
                estimate: CompletedEstimate | None, admitted: Tensor, base_mode: Tensor,
                jump_active: Tensor) -> StairAdvance:
        """Advance after physics and command admission, before the next actor frame.

        Discrete decisions use the pinned native protocol and deployment FSM.
        Fresh per-call state prevents native in-place operations from changing
        replay or a previous imagined node. Continuous height hints keep their
        derivative through command admission and the latched target height.
        """
        from transformer_rl.formal.control.observation import CommandMode
        from transformer_rl.formal.lab_env.course_tasks import StairRequestProtocol
        from transformer_rl.formal.lab_env.skill_fsm import StairFSM
        from transformer_rl.formal.tasks.protocol import RequestEnd

        n, device = len(state.values), state.values.device
        for name, value in state.record().items():
            dtype = STAIR_RECORD_DTYPES.get(name, torch.float32)
            if value.shape != (n, *STAIR_RECORD_FIELDS[name]) or value.dtype != dtype or value.device != device:
                raise ValueError('Invalid recorded stair field: ' + name)
        expected = self.config.signature(state.values).expand_as(state.parameters)
        if not torch.equal(state.parameters, expected):
            raise ValueError('Recorded stair parameters differ from the native task configuration')
        if (admitted.shape != (n, 4) or admitted.dtype != torch.float32 or admitted.device != device
                or base_mode.shape != (n,) or base_mode.dtype != torch.int64 or base_mode.device != device
                or packet.safety_abort.shape != (n,) or packet.safety_abort.dtype != torch.bool
                or packet.safety_abort.device != device):
            raise ValueError('Stair task requires native admitted commands, modes and abort flags')
        protocol = StairRequestProtocol(n, device=device, config=self.config.request)
        protocol.max_seen_id = state.ids[:, 0].clone()
        protocol.active_id = state.ids[:, 1].clone()
        protocol.height_m = state.values[:, 0].clone()
        protocol.accepted_at_s = state.clock[:, 1].clone()
        protocol.last_time_s = state.clock[:, 2].clone()
        fsm = StairFSM(n, config=self.config.fsm, device=device)
        fsm.state = state.ids[:, 2].clone()
        fsm._height_assist_steps_remaining = state.ids[:, 3].clone()
        for index, name in enumerate(('state_time', 'success_time', 'reference_height', 'target_height'), 1):
            setattr(fsm, name, state.values[:, index].clone())

        ready, contact, tilt = self.estimate_inputs(now_s, estimate, admitted)
        condition = protocol.step(now_s=now_s, request_id=packet.request_id, height_m=packet.height_m,
                                  cancel_id=packet.cancel_id, jump_active=jump_active,
                                  ready=ready & ~packet.safety_abort, completed_id=packet.completed_id)
        finished = condition.finished_id >= 0
        released = finished & (condition.end_reason == int(RequestEnd.ESTIMATED_COMPLETE))
        reset_rows = (finished & ~released).nonzero().flatten()
        # Empty indexed resets still launch writes and validate ranges on CUDA.
        if reset_rows.numel():
            fsm.reset(reset_rows)
        ordinary = (base_mode != int(CommandMode.JUMP)) & (base_mode != int(CommandMode.RECOVERY))
        output = fsm.deployment_step(
            step_height_m=condition.height_m, wheel_contact_probability=contact, tilt_rad=tilt,
            command_height_m=admitted[:, 3], manual_request=condition.active,
            manual_height_m=condition.height_m, completed=released,
            height_assist=self.config.height_assist,
            height_assist_allowed=(condition.height_m > 0) & ordinary)
        protocol.clear(output.exited)
        active = output.mode != int(CommandMode.NORMAL)
        height = admitted[:, 3]
        if self.config.height_assist is not None and self.config.height_assist.enabled:
            height = torch.where(active & ordinary, output.height_command_m, height)
        mode = torch.where(active, output.mode, base_mode) if self.config.use_mode_labels else base_mode
        body_allowed = ((state.values[:, 5] > .5) | output.body_contact_allowed)
        body_allowed &= (mode != int(CommandMode.JUMP)) & (mode != int(CommandMode.RECOVERY))
        following = StairTaskState(
            torch.stack((protocol.max_seen_id, protocol.active_id, fsm.state,
                         fsm._height_assist_steps_remaining), -1),
            torch.stack((now_s, protocol.accepted_at_s, protocol.last_time_s), -1),
            torch.stack((protocol.height_m, fsm.state_time, fsm.success_time,
                         fsm.reference_height, fsm.target_height, state.values[:, 5]), -1), state.parameters)
        return StairAdvance(following, torch.cat((admitted[:, :3], height[:, None]), -1), mode,
                            body_allowed, condition.response, condition.finished_id, condition.end_reason,
                            output.entered, output.exited, condition.active_id)
