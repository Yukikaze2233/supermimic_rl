"""The native ideal position-assisted stair source, with explicit random draws.

This is the existing simulation model, not a calibrated hardware ToF model.
Detection, completion and response binding use the pinned native implementation.
Only random-dependent arm/step operations are functionalized so a whole actor
batch draws once and microbatches/gradient checks consume the same realization.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from . import so3
from .stair_task import StairMotionBoundary, StairPacket, StairTaskConfig

if TYPE_CHECKING:
    from transformer_rl.formal.lab_env.stair_trigger import TofTriggerConfig

TOF_RECORD_FIELDS = {'tof_ids': (4,), 'tof_flags': (4,), 'tof_values': (13,), 'tof_parameters': (15,),
                     'tof_segment_end': (1,)}
TOF_RECORD_DTYPES = {'tof_ids': torch.int64, 'tof_flags': torch.bool}
TOF_FEATURE_DIM = 38


@dataclass(frozen=True)
class StairScenario:
    """Serializable native task/source settings shared by collection and imagination."""
    task: StairTaskConfig
    tof: TofTriggerConfig
    min_height_m: float
    max_height_m: float
    timeout_retry: bool = False

    def __post_init__(self):
        if type(self.timeout_retry) is not bool:
            raise ValueError('Stair timeout retry must be a boolean')

    @classmethod
    def capture(cls, native):
        trigger = native._terrain_course.tof_trigger
        return cls(StairTaskConfig.capture(native), trigger.config, trigger.min_height_m, trigger.max_height_m,
                   getattr(native._terrain_course, 'supermimic_stair_timeout_retry', False))

    @classmethod
    def from_dict(cls, value):
        from transformer_rl.formal.lab_env.stair_trigger import TofTriggerConfig

        retry = value.get('timeout_retry', False)
        expected = ('supermimic-native-stair-source-timeout-retry-v1' if retry
                    else 'supermimic-native-stair-source-v1')
        if type(retry) is not bool or value['format'] != expected:
            raise ValueError('Unknown native stair scenario format')
        tof = dict(value['tof'])
        for key in ('trigger_distance_m', 'early_distance_m', 'latency_intervals'):
            tof[key] = tuple(tof[key])
        return cls(StairTaskConfig.from_dict(value['task']), TofTriggerConfig(**tof),
                   value['min_height_m'], value['max_height_m'], retry)

    def to_dict(self):
        fields = asdict(self)
        if not self.timeout_retry:
            fields.pop('timeout_retry')
        return {'format': ('supermimic-native-stair-source-timeout-retry-v1' if self.timeout_retry
                           else 'supermimic-native-stair-source-v1'), **fields,
                'scope': 'existing ideal position-assisted simulator source; not calibrated hardware ToF'}

    def build(self, terrain):
        from .stair_task import StairTask

        return StairTask(self.task), IdealStairSource(self.tof, terrain=terrain,
            min_height_m=self.min_height_m, max_height_m=self.max_height_m, timeout_retry=self.timeout_retry)


@dataclass(frozen=True)
class StairSourceState:
    ids: Tensor       # Next request ID, countdown, pending source, completion ID.
    flags: Tensor     # Armed, early, riser-cleared, finite last-riser position.
    values: Tensor    # Distance/height; 5 XY pairs; stable-support clock.
    parameters: Tensor
    segment_end: Tensor  # Exact episode-clock deadline; gimbal resamples on this managed boundary.

    @classmethod
    @torch.no_grad()
    def capture(cls, native):
        course = native._terrain_course
        if (course._pending_requests['stair'] is not None or course.manual_stair_request.any()
                or (course.tof_request_id >= 0).any() or (course.stair_cancel_id >= 0).any()):
            raise ValueError('Ideal-source collection cannot discard a pending external stair packet')
        return cls.from_trigger(course.tof_trigger, course.stair_next_id, course.router.origins[:, :2],
                                segment_end=course.tasks.segment_end_s[:, None])

    @classmethod
    def from_trigger(cls, trigger, next_id, origin=None, *, segment_end=None):
        if origin is None:
            origin = torch.zeros_like(trigger.pending_riser_xy)
        last_valid = torch.isfinite(trigger.last_riser_xy).all(-1)
        local = lambda value: value - origin
        values = torch.cat((trigger.trigger_distance_m[:, None], trigger.measured_height_m[:, None],
            local(trigger.pending_riser_xy), trigger.pending_direction_xy,
            torch.where(last_valid[:, None], local(trigger.last_riser_xy), 0.),
            local(trigger._completion_riser_xy), trigger._completion_direction_xy,
            trigger._completion_stable_s[:, None]), -1)
        if segment_end is None:
            segment_end = torch.full_like(trigger.trigger_distance_m[:, None], float('inf'))
        return cls(torch.stack((next_id, trigger.countdown, trigger.pending_source, trigger._completion_id), -1).clone(),
            torch.stack((trigger.armed, trigger.early, trigger.riser_cleared, last_valid), -1).clone(),
            values, IdealStairSource.signature(trigger.config, trigger.min_height_m,
                                             trigger.max_height_m, values).expand(len(values), -1).clone(),
                   segment_end.clone())

    @classmethod
    def decode(cls, record):
        return cls(*(record[key] for key in TOF_RECORD_FIELDS))

    def record(self):
        return dict(zip(TOF_RECORD_FIELDS, (self.ids, self.flags, self.values, self.parameters, self.segment_end)))

    def critic_features(self, physical, stair, motion):
        """Only robot-relative tracked geometry; request identity is relational."""
        yaw = so3.yaw(physical.rotation)
        cosine, sine = yaw.cos(), yaw.sin()

        def rotate(xy):
            return torch.stack((cosine * xy[:, 0] + sine * xy[:, 1],
                                -sine * xy[:, 0] + cosine * xy[:, 1]), -1)

        def point(xy, valid):
            return torch.where(valid[:, None], rotate(xy - physical.position[:, :2]), 0.)

        pending = self.ids[:, 1] >= 0
        bound = (self.ids[:, 3] >= 0) & (self.ids[:, 3] == stair.ids[:, 1])
        values = torch.cat((self.values[:, :1], torch.where(pending[:, None], self.values[:, 1:2], 0.),
            point(self.values[:, 2:4], pending), torch.where(pending[:, None], rotate(self.values[:, 4:6]), 0.),
            point(self.values[:, 6:8], self.flags[:, 3]), point(self.values[:, 8:10], bound),
            torch.where(bound[:, None], rotate(self.values[:, 10:12]), 0.),
            torch.where(bound[:, None], self.values[:, 12:13], 0.)), -1)
        managed = motion.route[:, 4] > .5
        remaining = torch.where(managed[:, None], self.segment_end - motion.timing[:, :1], 0.)
        return torch.cat((self.flags, self.ids[:, 1:2], torch.nn.functional.one_hot(self.ids[:, 2], 3),
                          bound[:, None], values, self.parameters, remaining), -1).to(self.values)

    def boundary_known(self, motion, stair, policy_dt_s, jump=None):
        """No new random command is needed at the next native task boundary.

        Native tasks increment their float32 episode clock, then resample if
        due and unblocked. Gimbal commands share that exact managed boundary.
        An expired segment remains predictable while a stair request blocks it.
        """
        managed = motion.route[:, 4] > .5
        due = motion.timing[:, 0] + policy_dt_s >= self.segment_end[:, 0]
        # Native command-pool resampling is paused throughout RECOVERY,
        # including intervals with no stair or jump request.
        blocked = (stair.ids[:, 1] >= 0) | (motion.route[:, 3] == 3)
        if jump is not None:
            blocked = blocked | (jump.active_id >= 0)
        return ~managed | ~due | blocked


@dataclass(frozen=True)
class StairSourceBoundary:
    boundary: StairMotionBoundary
    source: StairSourceState
    packet: StairPacket
    fired: Tensor


class IdealStairSource:
    def __init__(self, config, *, min_height_m, max_height_m, terrain, timeout_retry=False):
        if type(timeout_retry) is not bool:
            raise ValueError('Stair timeout retry must be a boolean')
        self.config, self.min_height_m, self.max_height_m = config, min_height_m, max_height_m
        self.terrain = terrain
        self.timeout_retry = timeout_retry

    @staticmethod
    def signature(config, minimum, maximum, like):
        return like.new_tensor([*config.trigger_distance_m, config.early_probability,
            *config.early_distance_m, config.height_noise_m, *config.latency_intervals,
            config.sample_spacing_m, config.edge_window_m, config.minimum_travel_speed_m_s,
            config.riser_identity_m, config.sensor_range_m, minimum, maximum])

    def restore(self, state):
        from transformer_rl.formal.lab_env.stair_trigger import SimulatedTofTrigger

        n, device = len(state.values), state.values.device
        for name, value in state.record().items():
            if (value.shape != (n, *TOF_RECORD_FIELDS[name]) or value.device != device
                    or value.dtype != TOF_RECORD_DTYPES.get(name, torch.float32)):
                raise ValueError('Invalid recorded ideal stair-source field: ' + name)
        expected = self.signature(self.config, self.min_height_m, self.max_height_m, state.values)
        if not torch.equal(state.parameters, expected.expand(n, -1)):
            raise ValueError('Recorded ideal-source simulation parameters differ')
        trigger = SimulatedTofTrigger(n, min_height_m=self.min_height_m, max_height_m=self.max_height_m,
                                      config=self.config, device=device)
        for index, name in enumerate(('countdown', 'pending_source', '_completion_id'), 1):
            setattr(trigger, name, state.ids[:, index].clone())
        for index, name in enumerate(('armed', 'early', 'riser_cleared')):
            setattr(trigger, name, state.flags[:, index].clone())
        trigger.trigger_distance_m = state.values[:, 0].clone()
        trigger.measured_height_m = state.values[:, 1].clone()
        for index, name in enumerate(('pending_riser_xy', 'pending_direction_xy', 'last_riser_xy',
                                      '_completion_riser_xy', '_completion_direction_xy')):
            setattr(trigger, name, state.values[:, 2 + 2 * index:4 + 2 * index].clone())
        trigger.last_riser_xy = torch.where(state.flags[:, 3:4], trigger.last_riser_xy, float('inf'))
        trigger._completion_stable_s = state.values[:, 12].clone()
        return trigger

    def draw(self, batch, horizon, *, device, generator=None):
        """Native draw order, including unconditional draws on inactive rows.

        These are imagined exogenous realizations, not predictions of the
        native collector's future RNG. Generate before splitting actor batches.
        """
        latency, uniform = [], []
        low, high = self.config.latency_intervals
        for _ in range(horizon):
            latency.append(torch.randint(low, high + 1, (batch,), device=device, generator=generator))
            uniform.append(torch.stack([torch.rand(batch, device=device, generator=generator)
                                        for _ in range(7)], -1))
        return {'tof_latency': torch.stack(latency, 1), 'tof_uniform': torch.stack(uniform, 1)}

    @torch.no_grad()
    def arm(self, trigger, mask, draws):
        early = draws[:, 0] < self.config.early_probability
        early_low, early_high = self.config.early_distance_m
        low, high = self.config.trigger_distance_m
        distance = torch.where(early, early_low + (early_high - early_low) * draws[:, 1],
                                low + (high - low) * draws[:, 2])
        trigger.trigger_distance_m.copy_(torch.where(mask, distance, trigger.trigger_distance_m))
        trigger.early.copy_(torch.where(mask, early, trigger.early))
        trigger.armed |= mask
        trigger.countdown.masked_fill_(mask, -1)

    @torch.no_grad()
    def step(self, trigger, beam, *, idle, latency, draws):
        found, distance, height = trigger.detect(beam.height_m, beam.valid)
        riser = beam.origin_xy + beam.direction_xy * distance.clamp_max(1e3)[:, None]
        same = found & ((riser - trigger.last_riser_xy).norm(dim=-1) < self.config.riser_identity_m)
        trigger.riser_cleared |= ~same
        in_band = (height >= self.min_height_m) & (height <= self.max_height_m + 1e-3)
        detected = (trigger.armed & found & trigger.riser_cleared
                    & (distance <= trigger.trigger_distance_m) & in_band)
        noise = (draws[:, 0] * 2 - 1) * self.config.height_noise_m
        source = torch.where(beam.forward, torch.where(trigger.early, 1, 0), 2)
        for name, value in (('measured_height_m', height + noise), ('pending_source', source),
                            ('countdown', latency)):
            old = getattr(trigger, name)
            old.copy_(torch.where(detected, value, old))
        for name, value in (('pending_riser_xy', riser), ('pending_direction_xy', beam.direction_xy)):
            old = getattr(trigger, name)
            old.copy_(torch.where(detected[:, None], value, old))
        trigger.armed &= ~detected
        due = (trigger.countdown == 0) & idle
        current = (found & in_band & ((riser - trigger.pending_riser_xy).norm(dim=-1) < self.config.riser_identity_m)
                   & ((beam.direction_xy * trigger.pending_direction_xy).sum(-1) > .9))
        fire, stale = due & current, due & ~current
        trigger.countdown.copy_(torch.where(trigger.countdown > 0, trigger.countdown - 1, trigger.countdown))
        trigger.countdown.masked_fill_(due, -1)
        trigger.last_riser_xy.copy_(torch.where(fire[:, None], trigger.pending_riser_xy, trigger.last_riser_xy))
        trigger.riser_cleared &= ~fire
        self.arm(trigger, stale, draws[:, 1:4])
        return fire, torch.where(fire, trigger.measured_height_m.clamp(self.min_height_m, self.max_height_m), 0.)

    @torch.no_grad()
    def profile(self, physical, tile, goal):
        from transformer_rl.formal.lab_env.stair_trigger import TofBeam

        forward = goal[:, 0]  # The interval's old final command, not the newly admitted one.
        travel = forward.sign() * (forward.abs() >= self.config.minimum_travel_speed_m_s)
        yaw = so3.yaw(physical.rotation)
        direction = torch.stack((yaw.cos(), yaw.sin()), -1) * travel[:, None]
        distances = self.config.sample_distances(physical.position.device)
        xy = physical.position[:, None, :2] + direction[:, None] * distances[None, :, None]
        points = torch.cat((xy, torch.zeros_like(xy[..., :1])), -1)
        ground = self.terrain.height_at(points, tile)
        return TofBeam(ground.height, ground.valid & (travel != 0)[:, None],
                       physical.position[:, :2], direction, travel > 0)

    def advance(self, source, stair, *, task, motion, physical, wheel_centres, wheel_radius,
                terrain_tile, queues, readout, goal, estimate, physics_dt_s, latency, draws, external=None,
                admission=None, base_mode=None, jump_active=None, request_height=None, request_clock=None,
                reservation=None, request_owned=None):
        """Complete one boundary: detect -> release -> protocol/FSM -> rearm -> goal.

        Caller reservation suppresses only new ToF submissions. Detection,
        latency, existing-request completion and external packets still advance.
        """
        from transformer_rl.formal.tasks.protocol import RequestResponse

        trigger = self.restore(source)
        n = len(source.values)
        if reservation is not None and (not isinstance(reservation, Tensor)
                or reservation.shape != (n,) or reservation.device != source.values.device
                or reservation.dtype != torch.bool):
            raise ValueError('Stair reservation requires one bool per source row')
        external = StairPacket.empty(n, source.values.device) if external is None else external
        now = task.boundary_time(stair, physics_dt_s)
        ready, contact, tilt = task.estimate_inputs(now, estimate, source.values)
        owned = ((external.request_id != -1) | (external.cancel_id != -1)
                 | external.safety_abort | (external.completed_id != -1))
        idle = ready & ~owned & (stair.ids[:, 1] < 0)
        if jump_active is not None:
            idle = idle & ~jump_active
        if reservation is not None:
            idle = idle & ~reservation
        beam = self.profile(physical, terrain_tile, goal)
        fired, measured = self.step(trigger, beam, idle=idle, latency=latency, draws=draws)
        next_id = torch.maximum(source.ids[:, 0], stair.ids[:, 0] + 1)
        relative = wheel_centres[..., :2] - beam.origin_xy[:, None]
        rear = -(relative * trigger._completion_direction_xy[:, None]).sum(-1) + wheel_radius
        completed = trigger.completion(beam, active_id=stair.ids[:, 1], wheel_contact_probability=contact,
            tilt_rad=tilt, ready=ready, clearance_m=rear.amax(-1).clamp_min(0), policy_dt_s=task.config.request.policy_dt_s)
        matched = (external.completed_id >= 0) & (external.completed_id == stair.ids[:, 1])
        packet = StairPacket(torch.where(fired, next_id, external.request_id),
            torch.where(fired, measured, external.height_m), external.cancel_id,
            torch.where(matched, external.completed_id, completed), external.safety_abort)
        boundary = task.advance_motion(stair, packet, motion=motion, queues=queues, readout=readout,
            goal=goal, estimate=estimate, physics_dt_s=physics_dt_s, admission=admission,
            base_mode=base_mode, jump_active=jump_active, request_height=request_height,
            request_clock=request_clock, request_owned=request_owned)
        result = boundary.stair
        accepted = result.response == int(RequestResponse.ACCEPTED)
        trigger.record_responses(fired, accepted, active_id=result.protocol_active_id)
        if self.timeout_retry:
            from ..course.stair_retry import allow_timeout_retry

            allow_timeout_retry(trigger, result.finished_id >= 0, result.end_reason)
        self.arm(trigger, (fired & ~accepted) | (result.finished_id >= 0) | result.exited, draws[:, 4:7])
        following = StairSourceState.from_trigger(trigger, torch.maximum(next_id, result.state.ids[:, 0] + 1),
                                                  segment_end=source.segment_end)
        return StairSourceBoundary(boundary, following, packet, fired)
