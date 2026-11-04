"""Optional recorded jump tasks, with the pinned deployment/physics semantics.

Request acceptance uses the preceding completed estimator output. Physical
flight/landing evidence is reward-only. The visible request clock advances at
50 Hz and is held across each four-tick reference, exactly as in LabCourseTasks.
All heights here use the terrain tile's coordinates, like PhysicalState.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .physical import PhysicalState

if TYPE_CHECKING:
    from supermimic_rl.course.jump_takeoff import JumpTakeoffConfig
    from transformer_rl.formal.lab_env.course_tasks import RequestCommand
    from transformer_rl.formal.tasks.jump import JumpReferenceConfig
    from transformer_rl.formal.tasks.protocol import CompletedEstimate, RequestProtocolConfig


# Qualified names describe the original owner, not new learned coordinates.
JUMP_ID_NAMES = (
    'protocol.max_seen_id', 'protocol.active_id', 'protocol.air_ticks',
    'next_id', 'physical_id', 'air_ticks', 'paid_id', 'success_id', 'phase',
)
JUMP_CLOCK_NAMES = (
    'now_s', 'protocol.accepted_at_s', 'protocol.last_time_s',
    'protocol.landing_since_s', '_physical_accepted_at_s',
)
JUMP_VALUE_NAMES = (
    'protocol.height_m', 'protocol.cancelled', 'protocol.flight_seen',
    'next_request_s', 'rise', 'active_rise', 'latched_height', 'ground_anchor',
    'request_clock', '_physical_cancelled', 'flight_seen', 'ascent_seen',
    'supported_start', '_last_supported', '_last_world_height', 'peak_rise', 'landed_s',
)
JUMP_RECORD_FIELDS = {'jump_ids': (9,), 'jump_clock': (5,), 'jump_values': (17,)}
JUMP_RECORD_DTYPES = {'jump_ids': torch.int64, 'jump_clock': torch.float64}
JUMP_FEATURE_DIM = 32
JUMP_TRIGGER_RECORD_VERSION = 1
JUMP_TRIGGER_RECORD_FIELDS = {'jump_trigger_clock': (1,), 'jump_trigger_flags': (2,)}
JUMP_TRIGGER_RECORD_DTYPES = {'jump_trigger_clock': torch.float64, 'jump_trigger_flags': torch.bool}
JUMP_TRIGGER_FEATURE_DIM = 3
# First-real-flight latch of the shared jump reward: (latch request id, lifted, closed).
JUMP_FLIGHT_LATCH_RECORD_FIELDS = {'jump_flight_latch': (3,)}
JUMP_FLIGHT_LATCH_RECORD_DTYPES = {'jump_flight_latch': torch.int64}
_BOOL_NAMES = {
    'protocol.cancelled', 'protocol.flight_seen', '_physical_cancelled',
    'flight_seen', 'ascent_seen', 'supported_start', '_last_supported',
}


@dataclass(frozen=True)
class JumpTaskConfig:
    request: RequestProtocolConfig
    reference: JumpReferenceConfig
    physics_dt_s: float = .005
    trigger: JumpTakeoffConfig | None = None
    trigger_record_version: int = JUMP_TRIGGER_RECORD_VERSION
    # The native shared reward owns a first-real-flight latch (course.reward.FirstFlightLatch).
    flight_latch: bool = False

    @classmethod
    def capture(cls, native):
        tasks = native._terrain_course.tasks
        gate = getattr(native, 'supermimic_jump_trigger', None)
        return cls(tasks.protocol.config, tasks.reference.config, float(native.physics_dt),
                   None if gate is None else gate.config,
                   flight_latch=hasattr(tasks, 'supermimic_flight_latch'))

    @classmethod
    def from_dict(cls, value):
        from transformer_rl.formal.tasks.jump import JumpReferenceConfig
        from transformer_rl.formal.tasks.protocol import RequestProtocolConfig
        from supermimic_rl.course.jump_takeoff import JumpTakeoffConfig

        return cls(RequestProtocolConfig(**value['request']), JumpReferenceConfig(**value['reference']),
                   value['physics_dt_s'],
                   None if value.get('trigger') is None else JumpTakeoffConfig(**value['trigger']),
                   value.get('trigger_record_version', JUMP_TRIGGER_RECORD_VERSION),
                   value.get('flight_latch', False))

    def __post_init__(self):
        if self.request.policy_dt_s != .02 or self.physics_dt_s != .005:
            raise ValueError('Jump imagination uses the recorded 200/50 Hz four-tick contract')
        if self.trigger_record_version != JUMP_TRIGGER_RECORD_VERSION:
            raise ValueError('Unknown jump trigger record version')


@dataclass(frozen=True)
class JumpTaskState:
    ids: Tensor
    clock: Tensor
    values: Tensor
    trigger_clock: Tensor | None = None  # last_end_s, absolute float64 clock
    trigger_flags: Tensor | None = None  # known state, was_active
    flight_latch: Tensor | None = None   # int64 (latch request id, lifted, closed)

    @classmethod
    @torch.no_grad()
    def capture(cls, native):
        course = native._terrain_course
        if getattr(course, '_pending_requests', {}).get('jump') is not None:
            raise ValueError('Jump capture cannot discard a pending external request packet')
        gate = getattr(native, 'supermimic_jump_trigger', None)
        trigger = None if gate is None else (
            gate.last_end_s[:, None], torch.stack((torch.ones_like(gate.was_active), gate.was_active), -1))
        latch = getattr(course.tasks, 'supermimic_flight_latch', None)
        flight_latch = None if latch is None else torch.stack(
            (latch.request.long(), latch.lifted.long(), latch.closed.long()), -1)
        return cls._pack(native._clock(), course.tasks.protocol, course.tasks,
                         origin_z=course.router.origins[:, 2], trigger=trigger, flight_latch=flight_latch)

    @classmethod
    def _pack(cls, now, protocol, owner, *, origin_z=None, trigger=None, flight_latch=None):
        def get(name):
            if name == 'now_s':
                return now
            value = (getattr(protocol, name.split('.', 1)[1]) if name.startswith('protocol.')
                     else getattr(owner, name))
            if origin_z is not None and name in ('ground_anchor', '_last_world_height'):
                value = value - origin_z
            return value

        captured_trigger = (None, None) if trigger is None else tuple(value.clone() for value in trigger)
        return cls(torch.stack([get(name) for name in JUMP_ID_NAMES], -1).clone(),
                   torch.stack([get(name) for name in JUMP_CLOCK_NAMES], -1).clone(),
                   torch.stack([get(name).float() for name in JUMP_VALUE_NAMES], -1).clone(),
                   *captured_trigger, None if flight_latch is None else flight_latch.clone())

    @classmethod
    def decode(cls, record):
        if ('jump_trigger_clock' in record) != ('jump_trigger_flags' in record):
            raise ValueError('Incomplete jump trigger record')
        return cls(*(record[name] for name in JUMP_RECORD_FIELDS),
                   record.get('jump_trigger_clock'), record.get('jump_trigger_flags'),
                   record.get('jump_flight_latch'))

    def record(self):
        result = dict(zip(JUMP_RECORD_FIELDS, (self.ids, self.clock, self.values)))
        if self.trigger_clock is not None:
            result.update(jump_trigger_clock=self.trigger_clock, jump_trigger_flags=self.trigger_flags)
        if self.flight_latch is not None:
            result['jump_flight_latch'] = self.flight_latch
        return result

    def _flight_latch_state(self):
        latch, n = self.flight_latch, len(self.ids)
        if (latch.shape != (n, 3) or latch.dtype != torch.int64 or latch.device != self.ids.device
                or ((latch[:, 1:] != 0) & (latch[:, 1:] != 1)).any()):
            raise ValueError('Invalid first-flight latch record')
        return latch[:, 0], latch[:, 1].bool(), latch[:, 2].bool()

    def _trigger_state(self):
        n, device = len(self.ids), self.ids.device
        if (self.trigger_clock is None or self.trigger_flags is None
                or self.trigger_clock.shape != (n, 1) or self.trigger_clock.dtype != torch.float64
                or self.trigger_flags.shape != (n, 2) or self.trigger_flags.dtype != torch.bool
                or self.trigger_clock.device != device or self.trigger_flags.device != device
                or not self.trigger_flags[:, 0].all()
                or not (torch.isfinite(self.trigger_clock) | torch.isneginf(self.trigger_clock)).all()):
            raise ValueError('Jump trigger needs known native clock/state; migrate legacy replay explicitly')
        return self.trigger_clock[:, 0], self.trigger_flags[:, 1]

    def trigger_critic_features(self, motion, config):
        last_end, was_active = self._trigger_state()
        gate = config.trigger
        if gate is None:
            raise ValueError('Trigger critic features require an enabled gate')
        cooldown = (gate.cooldown_s - (self.now_s - last_end)).clamp_min(0.)
        episode = (gate.min_episode_s - motion.timing[:, 0]).clamp_min(0.)
        return torch.stack((cooldown / max(gate.cooldown_s, .02),
                            episode / max(gate.min_episode_s, .02), was_active), -1).to(self.values)

    @property
    def active_id(self):
        return self.ids[:, 1]

    @property
    def physical_id(self):
        return self.ids[:, 4]

    @property
    def now_s(self):
        return self.clock[:, 0]

    @property
    def request_clock(self):
        return self.values[:, 8]

    @property
    def next_request_s(self):
        return self.values[:, 3]

    def critic_features(self, physical: PhysicalState, motion, config: JumpTaskConfig):
        """Relative privileged memory; never request IDs or absolute time/xyz.

        Equal current poses can have different peak/landing/payment histories.
        These features let the optional jump critic distinguish those states
        without changing the actor's existing visible request interface.
        """
        request, reference = config.request, config.reference
        active, physical_active = self.active_id >= 0, self.physical_id >= 0
        landing = torch.isfinite(self.clock[:, 3])
        age = torch.where(active, self.now_s - self.clock[:, 1], 0.)
        landing_age = torch.where(landing, self.now_s - self.clock[:, 3], 0.)
        physical_age = torch.where(physical_active, self.now_s - self.clock[:, 4], 0.)
        z, scale = physical.position[:, 2], reference.jump_max_m
        value = self.values
        scalars = torch.stack((
            active, age / request.timeout_s, landing, landing_age / request.stable_landing_s,
            value[:, 1], value[:, 2], self.ids[:, 2] / request.flight_confirm_ticks,
            (self.next_request_s - motion.timing[:, 0]) / 8., value[:, 4] / scale,
            value[:, 5] / scale, value[:, 6] / reference.support_height_max_m,
            (value[:, 7] - z) / scale,
            physical_active, physical_age / request.timeout_s,
            value[:, 9], value[:, 10], value[:, 11], value[:, 12], value[:, 13],
            (value[:, 14] - z) / scale, value[:, 15] / scale,
            value[:, 16] / 2., self.ids[:, 5] / 2.,
            physical_active & (self.physical_id <= self.ids[:, 6]), self.ids[:, 7] >= 0,
            active & physical_active & (self.active_id == self.physical_id),
        ), -1).to(value)
        return torch.cat((scalars, torch.nn.functional.one_hot(self.ids[:, 8], 6).to(value)), -1)

    def _restore(self, config):
        """Fresh native owners prevent in-place protocols from mutating replay."""
        from transformer_rl.formal.tasks.protocol import RequestProtocol

        protocol = RequestProtocol(len(self.ids), self.ids.device, config.request)
        owner = SimpleNamespace()
        for names, tensor in ((JUMP_ID_NAMES, self.ids), (JUMP_CLOCK_NAMES, self.clock),
                              (JUMP_VALUE_NAMES, self.values)):
            for index, name in enumerate(names):
                if name == 'now_s':
                    continue
                value = tensor[:, index].clone()
                if name in _BOOL_NAMES:
                    value = value.bool()
                if name.startswith('protocol.'):
                    setattr(protocol, name.split('.', 1)[1], value.detach())
                else:
                    setattr(owner, name, value)
        owner.active_id = protocol.active_id.clone()
        owner.physical_outcome_counts = self.ids.new_zeros(3)
        return protocol, owner


@dataclass(frozen=True)
class JumpAdvance:
    state: JumpTaskState
    command: Tensor
    mode: Tensor
    request_height: Tensor
    request_clock: Tensor
    response: Tensor
    finished_id: Tensor
    end_reason: Tensor
    deferred: Tensor


@dataclass(frozen=True)
class JumpInterval:
    state: JumpTaskState
    phase: Tensor
    height_world_z_m: Tensor
    velocity_world_z_m_s: Tensor
    success_request_id: Tensor
    success_credited: Tensor
    reference_stage: Tensor
    ascent_seen: Tensor
    flight_closed: Tensor | None = None   # per tick; None without a recorded first-flight latch


class JumpTask:
    def __init__(self, config: JumpTaskConfig):
        from transformer_rl.formal.tasks.jump import JumpReferenceProgram

        self.config = config
        self.program = JumpReferenceProgram(config.reference)

    def boundary_time(self, state):
        dt = self.config.physics_dt_s
        return ((state.now_s / dt).round() + 4) * dt

    def reference(self, state):
        return self.program.sample(state.values[:, 6], state.values[:, 5],
            state.request_clock.double(), state.values[:, 7], active=state.active_id >= 0)

    def advance(self, state: JumpTaskState, *, now_s: Tensor, estimate: CompletedEstimate | None,
                admitted: Tensor, base_mode: Tensor, task_family: Tensor, elapsed_s: Tensor,
                ground_z: Tensor, blocked_by_stair: Tensor, interval_uniform: Tensor | None = None,
                packet: RequestCommand | None = None, previous_command_height: Tensor | None = None,
                previous_elapsed_s: Tensor | None = None) -> JumpAdvance:
        """Advance after old-interval reward, and after motion command admission.

        ``elapsed_s`` is MotionTask's already advanced episode clock. Random
        intervals are supplied once per full imagined batch, before microbatches
        or retries. Unknown future external packets are not synthesized.
        """
        from transformer_rl.formal.control.observation import CommandMode
        from transformer_rl.formal.tasks.curriculum import TaskFamily
        from transformer_rl.formal.tasks.protocol import CompletedEstimate, RequestEnd, RequestResponse

        protocol, owner = state._restore(self.config)
        n, device = len(state.ids), state.ids.device
        postponed = torch.zeros(n, dtype=torch.bool, device=device)
        trigger = None
        if self.config.trigger is not None:
            last_end, was_active = state._trigger_state()
            trigger = (last_end[:, None], state.trigger_flags)
            if packet is None:
                if (previous_command_height is None or previous_elapsed_s is None
                        or previous_command_height.shape != (n,) or previous_elapsed_s.shape != (n,)):
                    raise ValueError('Gated jumps need the command height and episode clock before admission')
                gate, dt = self.config.trigger, self.config.request.policy_dt_s
                ready = (owner._last_supported
                         & (owner._last_world_height - ground_z - previous_command_height).abs().le(
                             gate.height_tolerance_m)
                         & (previous_elapsed_s + dt >= gate.min_episode_s)
                         & (now_s - last_end >= gate.cooldown_s))
                due = ((task_family == int(TaskFamily.REQUEST))
                       & (previous_elapsed_s + dt >= owner.next_request_s) & (protocol.active_id < 0))
                postponed = due & ~ready
                owner.next_request_s = torch.where(postponed, previous_elapsed_s + 2 * dt,
                                                   owner.next_request_s)
        owner.success_id.fill_(-1)
        if estimate is None:
            estimate = CompletedEstimate(admitted.new_zeros(n), admitted.new_zeros(n, 2),
                admitted.new_zeros(n), admitted.new_zeros(n), now_s.clone(),
                torch.zeros(n, dtype=torch.bool, device=device))
        if packet is None:
            if (interval_uniform is None or interval_uniform.shape != (n,)
                    or interval_uniform.device != device or interval_uniform.dtype != torch.float32):
                raise ValueError('Automatic jump requests require one recorded/prepared uniform draw per row')
            due = ((task_family == int(TaskFamily.REQUEST)) & ~blocked_by_stair
                   & (elapsed_s >= owner.next_request_s) & (protocol.active_id < 0))
            request_id = torch.where(due, owner.next_id, -1)
            owner.next_id = owner.next_id + due.long()
            owner.next_request_s = torch.where(due, elapsed_s + (3 + 5 * interval_uniform),
                                               owner.next_request_s)
            requested_height = owner.rise
            cancel_id = torch.full_like(request_id, -1)
            safety_abort = torch.zeros(n, dtype=torch.bool, device=device)
            allowed = task_family == int(TaskFamily.REQUEST)
        else:
            packet.validate(n, device)
            request_id, requested_height = packet.request_id, packet.height_m
            cancel_id = (torch.full_like(request_id, -1) if packet.cancel_id is None else packet.cancel_id)
            safety_abort = (torch.zeros(n, dtype=torch.bool, device=device)
                            if packet.safety_abort is None else packet.safety_abort)
            allowed = base_mode != int(CommandMode.RECOVERY)
        condition = protocol.step(now_s=now_s, estimate=estimate, request_id=request_id,
            requested_height_m=requested_height, cancel_id=cancel_id,
            allow_new=allowed & ~blocked_by_stair, safety_abort=safety_abort)

        # This is the LabCourseTasks.step glue surrounding the shared protocol:
        # an estimated completion does not close unverified physical evidence.
        owner.next_id = torch.maximum(owner.next_id, protocol.max_seen_id + 1)
        accepted = condition.response == int(RequestResponse.ACCEPTED)
        matching = condition.finished_id == owner.physical_id
        closed = ((matching & (condition.end_reason != int(RequestEnd.ESTIMATED_COMPLETE))
                   & (condition.end_reason != int(RequestEnd.NONE)))
                  | ((owner.physical_id >= 0)
                     & (now_s - owner._physical_accepted_at_s >= self.config.request.timeout_s)))
        owner.physical_id.masked_fill_(closed, -1)
        owner.physical_id = torch.where(accepted, condition.active_id, owner.physical_id)
        owner._physical_accepted_at_s = torch.where(accepted, now_s, owner._physical_accepted_at_s)
        owner._physical_cancelled |= ((owner.physical_id == protocol.active_id) & protocol.cancelled)
        owner._physical_cancelled |= (owner.physical_id >= 0) & blocked_by_stair
        owner._physical_cancelled.masked_fill_(accepted, False)
        owner.latched_height = torch.where(accepted, admitted[:, 3], owner.latched_height)
        owner.active_rise = torch.where(accepted, condition.height_m, owner.active_rise)
        owner.ground_anchor = torch.where(accepted, ground_z, owner.ground_anchor)
        supported = (owner._last_supported
                     & (owner._last_world_height - ground_z - admitted[:, 3]).abs().le(.015))
        owner.supported_start = torch.where(accepted, supported, owner.supported_start)
        for name in ('flight_seen', 'ascent_seen', 'air_ticks', 'peak_rise', 'landed_s'):
            value = getattr(owner, name)
            setattr(owner, name, torch.where(accepted, torch.zeros_like(value), value))
        owner.active_id, owner.request_clock = condition.active_id, condition.clock_s
        mode = torch.where(condition.jump_mode, int(CommandMode.JUMP), base_mode)
        height = torch.where(condition.jump_mode, owner.latched_height, admitted[:, 3])
        command = torch.cat((admitted[:, :3], height[:, None]), -1)
        if trigger is not None:
            active = protocol.active_id >= 0
            last_end = torch.where(was_active & ~active, now_s, last_end)
            trigger = (last_end[:, None], torch.stack((state.trigger_flags[:, 0], active), -1))
        return JumpAdvance(JumpTaskState._pack(now_s, protocol, owner, trigger=trigger,
                                               flight_latch=state.flight_latch), command, mode,
            condition.height_m, condition.clock_s, condition.response, condition.finished_id, condition.end_reason,
            postponed)

    @staticmethod
    def com_velocity(physical: PhysicalState, base_com_offset: Tensor) -> Tensor:
        """Native jump evidence uses base COM velocity, unlike reward tracking."""
        offset = base_com_offset
        while offset.ndim < physical.position.ndim:
            offset = offset.unsqueeze(-2)
        world_offset = (physical.rotation @ offset[..., None]).squeeze(-1)
        return physical.velocity + torch.linalg.cross(physical.angular_velocity, world_offset)

    def observe_interval(self, state: JumpTaskState, *, physical: PhysicalState,
                         wheel_support_load: Tensor, base_com_offset: Tensor, base_mode: Tensor,
                         command_height: Tensor, ground_z: Tensor) -> JumpInterval:
        """Integrate four physical samples and return their native reward targets.

        ``wheel_support_load`` is the two wheels' upward patch-normal support
        load, exactly what the native task receives (``wheel_ground_force_n``,
        the measured ``wheel_support_load_n``); a resultant norm would also
        count horizontal riser and friction forces as support. Physical event transitions are deliberately
        detached; the caller's continuous motion/reward graph remains intact.
        """
        from transformer_rl.formal.lab_env.course_tasks import LabCourseTasks
        from transformer_rl.formal.tasks.reward import RewardPhase
        from ..reward_kernel import advance_flight_latch

        n = len(state.ids)
        if physical.position.shape != (n, 4, 3) or wheel_support_load.shape != (n, 4, 2):
            raise ValueError('Jump evidence requires four physical ticks and two wheel support loads')
        if ground_z.shape != (n, 4) or command_height.shape != (n,) or base_com_offset.shape != (n, 3):
            raise ValueError('Jump reference requires four ground heights and one native base COM offset')
        with torch.no_grad():
            protocol, owner = state._restore(self.config)
            owner.base_mode = base_mode
            velocity_z = self.com_velocity(physical, base_com_offset)[..., 2]
            gravity = physical.gravity_body
            tilt = torch.atan2(gravity[..., :2].norm(dim=-1), -gravity[..., 2])
            speed = physical.angular_velocity_body.norm(dim=-1)
            phases, ascents, closed_ticks = [], [], []
            latch = None if state.flight_latch is None else state._flight_latch_state()
            for tick in range(4):
                LabCourseTasks.observe_physics(owner, wheel_support_load[:, tick], velocity_z[:, tick],
                    tilt[:, tick], physical.position[:, tick, 2], angular_speed_rad_s=speed[:, tick],
                    dt=self.config.physics_dt_s)
                phases.append(owner.phase.clone())
                ascents.append(owner.ascent_seen.clone())
                if latch is not None:
                    # Same update as the native FirstFlightLatch after observe_physics.
                    request, lifted, closed = latch
                    changed = owner.physical_id != request
                    lifted, closed = advance_flight_latch(lifted & ~changed, closed & ~changed,
                        active=owner.physical_id >= 0, ascent_seen=owner.ascent_seen, air_ticks=owner.air_ticks)
                    latch = (owner.physical_id.clone(), lifted, closed)
                    closed_ticks.append(closed.clone())
            trigger = (None if state.trigger_clock is None else (state.trigger_clock, state.trigger_flags))
            flight_latch = None if latch is None else torch.stack(
                (latch[0].long(), latch[1].long(), latch[2].long()), -1)
            following = JumpTaskState._pack(state.now_s, protocol, owner, trigger=trigger, flight_latch=flight_latch)
            phase = torch.stack(phases, -1)
            success = (owner.success_id >= 0) & (owner.success_id > state.ids[:, 6])
        curve = self.reference(state)
        landing = (phase == int(RewardPhase.LANDING)) | (phase == int(RewardPhase.SETTLING))
        height = torch.where(landing, ground_z + command_height[:, None], curve.height_world_z_m[:, None])
        velocity = torch.where(landing, 0., curve.velocity_world_z_m_s[:, None])
        return JumpInterval(following, phase, height, velocity, owner.success_id.clone(), success,
                            curve.clock_stage[:, None].expand(-1, 4), torch.stack(ascents, -1),
                            torch.stack(closed_ticks, -1) if closed_ticks else None)
