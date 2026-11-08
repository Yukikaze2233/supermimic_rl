"""Recorded caller intent for shared native/imagined platform-jump boundaries.

The caller supplies the target and earliest episode time. ToF remains a stair
source, while the existing request protocol owns admission, IDs and timeout.
This module is tensor-only: no environment reads, random draws or future packets.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from transformer_rl.formal.lab_env.course_tasks import RequestCommand
    from transformer_rl.formal.tasks.protocol import RequestCondition


PLATFORM_INTENT_FORMAT = 'platform-jump-intent-v1'
PLATFORM_INTENT_RECORD_VERSION = 1
PLATFORM_TARGET_IDS = ('101:profile:1', '102:profile:1')
PLATFORM_INTENT_RECORD_FIELDS = {
    'platform_intent_ids': (3,),       # target index, phase, bound request ID
    'platform_intent_clock': (1,),     # earliest episode time, float64
    'platform_intent_values': (2,),    # requested rise and support height
}
PLATFORM_INTENT_RECORD_DTYPES = {
    'platform_intent_ids': torch.int64,
    'platform_intent_clock': torch.float64,
    'platform_intent_values': torch.float32,
}


class PlatformIntentPhase(IntEnum):
    INACTIVE = 0
    WAITING = 1
    PENDING = 2
    ACCEPTED = 3
    PHYSICAL_TAIL = 4
    DONE = 5


@dataclass(frozen=True)
class PlatformIntentState:
    target_index: Tensor
    earliest_episode_s: Tensor
    rise_m: Tensor
    support_height_m: Tensor
    phase: Tensor
    bound_request_id: Tensor

    @classmethod
    def inactive(cls, count: int, device='cpu') -> PlatformIntentState:
        ids = torch.full((count,), -1, dtype=torch.int64, device=device)
        values = torch.zeros(count, dtype=torch.float32, device=device)
        return cls(ids, torch.zeros(count, dtype=torch.float64, device=device), values,
                   values.clone(), torch.zeros_like(ids), ids.clone())

    @classmethod
    def plan(cls, *, target_index: Tensor, earliest_episode_s: Tensor,
             rise_m: Tensor, support_height_m: Tensor) -> PlatformIntentState:
        enabled = target_index >= 0
        result = cls(target_index.clone(), earliest_episode_s.clone(), rise_m.clone(),
                     support_height_m.clone(), torch.where(enabled, int(PlatformIntentPhase.WAITING),
                                                          int(PlatformIntentPhase.INACTIVE)),
                     torch.full_like(target_index, -1))
        result.validate(values=True)
        return result

    def validate(self, *, values=False):
        n, device = len(self.target_index), self.target_index.device
        for name in ('target_index', 'earliest_episode_s', 'rise_m', 'support_height_m',
                     'phase', 'bound_request_id'):
            value = getattr(self, name)
            dtype = (torch.float64 if name == 'earliest_episode_s' else torch.float32
                     if name in ('rise_m', 'support_height_m') else torch.int64)
            if value.shape != (n,) or value.device != device or value.dtype != dtype:
                raise ValueError('Platform intent needs consistently typed one-dimensional rows')
        if not values:
            return
        enabled = self.target_index >= 0
        waiting = (self.phase == int(PlatformIntentPhase.WAITING)) | (self.phase == int(PlatformIntentPhase.PENDING))
        accepted = ((self.phase == int(PlatformIntentPhase.ACCEPTED))
                    | (self.phase == int(PlatformIntentPhase.PHYSICAL_TAIL)))
        invalid = ((self.target_index < -1) | (self.target_index >= len(PLATFORM_TARGET_IDS))
                   | (self.phase < 0) | (self.phase > int(PlatformIntentPhase.DONE))
                   | (self.bound_request_id < -1)
                   | (enabled != (self.phase != int(PlatformIntentPhase.INACTIVE)))
                   | (waiting & (self.bound_request_id >= 0))
                   | (accepted & (self.bound_request_id < 0))
                   | ~torch.isfinite(self.earliest_episode_s) | (self.earliest_episode_s < 0)
                   | ~torch.isfinite(self.rise_m) | ~torch.isfinite(self.support_height_m)
                   | (enabled & ((self.rise_m <= 0) | (self.support_height_m <= 0)))
                   | (~enabled & ((self.bound_request_id != -1) | (self.earliest_episode_s != 0)
                                  | (self.rise_m != 0) | (self.support_height_m != 0))))
        if invalid.any():
            raise ValueError('Invalid platform intent plan or request history')

    @property
    def owned(self):
        return ((self.phase != int(PlatformIntentPhase.INACTIVE))
                & (self.phase != int(PlatformIntentPhase.DONE)))

    @property
    def reservation(self):
        return self.owned & (self.phase != int(PlatformIntentPhase.WAITING))

    def reset(self, mask: Tensor) -> PlatformIntentState:
        """Rearm the same explicit plan after a real episode reset.

        To remove or replace a plan, supply newly sampled intent rows instead.
        Resetting caller history never resets the protocol's ID watermark.
        """
        _row(mask, self.phase, torch.bool, 'reset mask')
        initial = torch.where(self.target_index >= 0, int(PlatformIntentPhase.WAITING),
                              int(PlatformIntentPhase.INACTIVE))
        return replace(self, phase=torch.where(mask, initial, self.phase),
                       bound_request_id=torch.where(mask, -1, self.bound_request_id))

    def record(self):
        """Independent snapshots; recording does not expose mutable state views."""
        return {'platform_intent_ids': torch.stack((self.target_index, self.phase, self.bound_request_id), -1),
                'platform_intent_clock': self.earliest_episode_s[:, None].clone(),
                'platform_intent_values': torch.stack((self.rise_m, self.support_height_m), -1)}

    @classmethod
    def decode(cls, record, *, version=PLATFORM_INTENT_RECORD_VERSION) -> PlatformIntentState:
        if type(version) is not int or version != PLATFORM_INTENT_RECORD_VERSION:
            raise ValueError('Unknown platform intent record version')
        if not all(name in record for name in PLATFORM_INTENT_RECORD_FIELDS):
            raise ValueError('Missing platform intent record fields; legacy rows must be explicitly inactive')
        ids = record['platform_intent_ids']
        for name, shape in PLATFORM_INTENT_RECORD_FIELDS.items():
            value = record[name]
            if (value.shape != (len(ids), *shape) or value.device != ids.device
                    or value.dtype != PLATFORM_INTENT_RECORD_DTYPES[name]):
                raise ValueError('Invalid platform intent record layout')
        clock, values = record['platform_intent_clock'], record['platform_intent_values']
        result = cls(ids[:, 0].clone(), clock[:, 0].clone(), values[:, 0].clone(), values[:, 1].clone(),
                     ids[:, 1].clone(), ids[:, 2].clone())
        # Replay layout is checked here without a device-to-host value sync.
        # Explicit plans and checkpoint restoration validate semantic values.
        result.validate()
        return result

    def state_dict(self):
        return {'format': PLATFORM_INTENT_FORMAT, **self.record()}

    @classmethod
    def from_state_dict(cls, state):
        if (set(state) != {'format', *PLATFORM_INTENT_RECORD_FIELDS}
                or state.get('format') != PLATFORM_INTENT_FORMAT):
            raise ValueError('Platform intent state format differs')
        result = cls.decode(state)
        result.validate(values=True)
        return result


@dataclass(frozen=True)
class PlatformIntentPreparation:
    state: PlatformIntentState
    jump_packet: RequestCommand
    packet_owned: Tensor
    stair_packet: RequestCommand
    reservation: Tensor
    height_override_mask: Tensor
    height_override_m: Tensor


def _row(value, like, dtype, name):
    if (not isinstance(value, Tensor) or value.shape != like.shape
            or value.device != like.device or value.dtype != dtype):
        raise ValueError(f'Platform intent {name} requires matching {dtype} rows')


def prepare_platform_intent(state: PlatformIntentState, *, now_s: Tensor, elapsed_s: Tensor,
                            policy_dt_s: float, timeout_s: float, physical_id: Tensor,
                            physical_accepted_at_s: Tensor, stair_active_id: Tensor,
                            stair_idle: Tensor, native_ready: Tensor, admitted_height: Tensor,
                            max_seen_id: Tensor, next_id: Tensor) -> PlatformIntentPreparation:
    """Prepare before command admission/protocol steps, using current evidence.

    ``elapsed_s`` is the preceding episode clock, before its policy-dt advance.
    The caller passes the existing gate's readiness and actual stair idle state.
    A cancellation and jump submission cannot occur for one row in this call.
    Accepted deadlines use the existing physical request's acceptance clock.
    """
    from transformer_rl.formal.lab_env.course_tasks import RequestCommand

    state.validate()
    for name, value, dtype in (
        ('now_s', now_s, torch.float64), ('elapsed_s', elapsed_s, torch.float32),
        ('physical_id', physical_id, torch.int64),
        ('physical_accepted_at_s', physical_accepted_at_s, torch.float64),
        ('stair_active_id', stair_active_id, torch.int64), ('stair_idle', stair_idle, torch.bool),
        ('native_ready', native_ready, torch.bool), ('admitted_height', admitted_height, torch.float32),
        ('max_seen_id', max_seen_id, torch.int64), ('next_id', next_id, torch.int64)):
        _row(value, state.phase, dtype, name)
    episode_s = elapsed_s.double() + policy_dt_s
    bound = ((state.phase == int(PlatformIntentPhase.ACCEPTED))
             | (state.phase == int(PlatformIntentPhase.PHYSICAL_TAIL)))
    expired = torch.where(bound, now_s >= physical_accepted_at_s + timeout_s - 1e-9,
                          episode_s >= state.earliest_episode_s + timeout_s - 1e-9)
    physical_ended = bound & (physical_id != state.bound_request_id)
    ended = state.owned & (expired | physical_ended)
    phase = torch.where(ended, int(PlatformIntentPhase.DONE), state.phase)
    due = ((phase == int(PlatformIntentPhase.WAITING))
           & (episode_s + 1e-9 >= state.earliest_episode_s))
    phase = torch.where(due, int(PlatformIntentPhase.PENDING), phase)
    following = replace(state, phase=phase)
    pending = phase == int(PlatformIntentPhase.PENDING)
    cancel = torch.where(pending & (stair_active_id >= 0), stair_active_id, -1)
    submit = (pending & (stair_active_id < 0) & stair_idle & native_ready
              & (admitted_height == state.support_height_m))
    request = torch.where(submit, torch.maximum(max_seen_id + 1, next_id), -1)
    rise = torch.where(submit, state.rise_m, 0.)
    return PlatformIntentPreparation(following, RequestCommand(request, rise), following.owned,
        RequestCommand(torch.full_like(request, -1), torch.zeros_like(rise), cancel_id=cancel),
        following.reservation, following.owned, state.support_height_m.clone())


def consume_platform_intent(prepared: PlatformIntentPreparation, condition: RequestCondition,
                            *, physical_id: Tensor) -> PlatformIntentState:
    """Consume actual protocol results after native physical-request bookkeeping.

    Estimated completion keeps the reservation while the same physical request
    remains live. Physical success, cancellation, timeout or reset closes it.
    Rejected IDs remain pending; the next preparation uses the updated watermark.
    """
    from transformer_rl.formal.tasks.protocol import RequestEnd, RequestResponse

    state = prepared.state
    _row(physical_id, state.phase, torch.int64, 'physical_id')
    for name in ('active_id', 'response', 'finished_id', 'end_reason'):
        _row(getattr(condition, name), state.phase, torch.int64, name)
    submitted = prepared.packet_owned & (prepared.jump_packet.request_id >= 0)
    accepted = (submitted & (condition.response == int(RequestResponse.ACCEPTED))
                & (condition.active_id == prepared.jump_packet.request_id))
    bound = torch.where(accepted, condition.active_id, state.bound_request_id)
    phase = torch.where(accepted, int(PlatformIntentPhase.ACCEPTED), state.phase)
    live = ((phase == int(PlatformIntentPhase.ACCEPTED))
            | (phase == int(PlatformIntentPhase.PHYSICAL_TAIL)))
    matching_end = live & (bound >= 0) & (condition.finished_id == bound)
    estimated = matching_end & (condition.end_reason == int(RequestEnd.ESTIMATED_COMPLETE))
    phase = torch.where(estimated, int(PlatformIntentPhase.PHYSICAL_TAIL), phase)
    ended = ((matching_end & (condition.end_reason != int(RequestEnd.NONE)) & ~estimated)
             | (live & (physical_id != bound)))
    phase = torch.where(ended, int(PlatformIntentPhase.DONE), phase)
    return replace(state, phase=phase, bound_request_id=bound)
