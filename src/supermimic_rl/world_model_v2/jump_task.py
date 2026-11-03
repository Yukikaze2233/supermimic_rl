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
_BOOL_NAMES = {
    'protocol.cancelled', 'protocol.flight_seen', '_physical_cancelled',
    'flight_seen', 'ascent_seen', 'supported_start', '_last_supported',
}


@dataclass(frozen=True)
class JumpTaskConfig:
    request: RequestProtocolConfig
    reference: JumpReferenceConfig
    physics_dt_s: float = .005

    @classmethod
    def capture(cls, native):
        if getattr(native, 'supermimic_jump_trigger', None) is not None:
            # Imagination replays automatic request scheduling; it must not
            # silently ignore the stability gate that defers native requests.
            raise ValueError('The world-model jump task does not mirror the jump trigger gate yet')
        tasks = native._terrain_course.tasks
        return cls(tasks.protocol.config, tasks.reference.config, float(native.physics_dt))

    @classmethod
    def from_dict(cls, value):
        from transformer_rl.formal.tasks.jump import JumpReferenceConfig
        from transformer_rl.formal.tasks.protocol import RequestProtocolConfig

        return cls(RequestProtocolConfig(**value['request']), JumpReferenceConfig(**value['reference']),
                   value['physics_dt_s'])

    def __post_init__(self):
        if self.request.policy_dt_s != .02 or self.physics_dt_s != .005:
            raise ValueError('Jump imagination uses the recorded 200/50 Hz four-tick contract')


@dataclass(frozen=True)
class JumpTaskState:
    ids: Tensor
    clock: Tensor
    values: Tensor

    @classmethod
    @torch.no_grad()
    def capture(cls, native):
        course = native._terrain_course
        if getattr(course, '_pending_requests', {}).get('jump') is not None:
            raise ValueError('Jump capture cannot discard a pending external request packet')
        return cls._pack(native._clock(), course.tasks.protocol, course.tasks,
                         origin_z=course.router.origins[:, 2])

    @classmethod
    def _pack(cls, now, protocol, owner, *, origin_z=None):
        def get(name):
            if name == 'now_s':
                return now
            value = (getattr(protocol, name.split('.', 1)[1]) if name.startswith('protocol.')
                     else getattr(owner, name))
            if origin_z is not None and name in ('ground_anchor', '_last_world_height'):
                value = value - origin_z
            return value

        return cls(torch.stack([get(name) for name in JUMP_ID_NAMES], -1).clone(),
                   torch.stack([get(name) for name in JUMP_CLOCK_NAMES], -1).clone(),
                   torch.stack([get(name).float() for name in JUMP_VALUE_NAMES], -1).clone())

    @classmethod
    def decode(cls, record):
        return cls(*(record[name] for name in JUMP_RECORD_FIELDS))

    def record(self):
        return dict(zip(JUMP_RECORD_FIELDS, (self.ids, self.clock, self.values)))

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


@dataclass(frozen=True)
class JumpInterval:
    state: JumpTaskState
    phase: Tensor
    height_world_z_m: Tensor
    velocity_world_z_m_s: Tensor
    success_request_id: Tensor
    success_credited: Tensor


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
                packet: RequestCommand | None = None) -> JumpAdvance:
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
        return JumpAdvance(JumpTaskState._pack(now_s, protocol, owner), command, mode,
            condition.height_m, condition.clock_s, condition.response, condition.finished_id, condition.end_reason)

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
            phases = []
            for tick in range(4):
                LabCourseTasks.observe_physics(owner, wheel_support_load[:, tick], velocity_z[:, tick],
                    tilt[:, tick], physical.position[:, tick, 2], angular_speed_rad_s=speed[:, tick],
                    dt=self.config.physics_dt_s)
                phases.append(owner.phase.clone())
            following = JumpTaskState._pack(state.now_s, protocol, owner)
            phase = torch.stack(phases, -1)
            success = (owner.success_id >= 0) & (owner.success_id > state.ids[:, 6])
        curve = self.reference(state)
        landing = (phase == int(RewardPhase.LANDING)) | (phase == int(RewardPhase.SETTLING))
        height = torch.where(landing, ground_z + command_height[:, None], curve.height_world_z_m[:, None])
        velocity = torch.where(landing, 0., curve.velocity_world_z_m_s[:, None])
        return JumpInterval(following, phase, height, velocity, owner.success_id.clone(), success)
