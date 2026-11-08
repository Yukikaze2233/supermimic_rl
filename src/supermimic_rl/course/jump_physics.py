"""Shared single-tick jump evidence and its sole physical-success payment.

Flat requests retain the pinned native rules. Selected-platform requests add
direct-landing evidence supplied by the caller's geometry/contact observer;
this module neither guesses terrain geometry nor changes request admission.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum

import torch
from torch import Tensor


class PlatformLandingPhase(IntEnum):
    BEFORE_FLIGHT = 0
    FLIGHT = 1
    TARGET_SINGLE_WHEEL = 2
    TARGET_BOTH_WHEELS = 3
    MISSED = 4
    UNKNOWN_CONTACT = 5


@dataclass(frozen=True)
class PlatformLandingState:
    """Request-bound direct-landing history, separate from wheel unloading.

    Successful/inactive requests retain their final phase for reporting. An
    episode reset clears it explicitly; a new request cannot inherit it.
    """

    request_id: Tensor
    free_ticks: Tensor
    phase: Tensor

    @classmethod
    def empty(cls, like: Tensor) -> PlatformLandingState:
        return cls(torch.full_like(like, -1, dtype=torch.int64),
                   torch.zeros_like(like, dtype=torch.int64),
                   torch.zeros_like(like, dtype=torch.int64))

    def reset(self, mask: Tensor) -> PlatformLandingState:
        return PlatformLandingState(torch.where(mask, -1, self.request_id),
                                    torch.where(mask, 0, self.free_ticks),
                                    torch.where(mask, 0, self.phase))

    def validate(self, shape, device):
        for field in fields(self):
            value = getattr(self, field.name)
            if value.shape != shape or value.device != device or value.dtype != torch.int64:
                raise ValueError('Platform landing history requires matching int64 request rows')


@dataclass(frozen=True)
class PlatformLandingEvidence:
    """Already evaluated physical evidence for one tick or four recorded ticks.

    ``full_body_clear`` certifies full-body contact departure and positive wheel
    clearance, not merely two unloaded wheels. ``wheel_touch`` includes every
    significant wheel contact; ``target_wheel_touch`` additionally certifies
    contact with the selected target's exposed surface. ``nonwheel_contact``
    identifies a significant non-wheel contact, including body-supported poses.
    The geometry observer owns these definitions and their source identity.

    After both wheels have touched the target, moving continuation is allowed:
    ``landing_support_valid`` must certify current legitimate wheel support,
    and ``landing_world_height`` is current support ground plus command height.
    Neither a root ray nor a matching height alone supplies this certificate.
    """

    selected: Tensor
    measurement_valid: Tensor
    full_body_clear: Tensor
    wheel_touch: Tensor
    target_wheel_touch: Tensor
    nonwheel_contact: Tensor
    landing_support_valid: Tensor
    landing_world_height: Tensor

    def validate(self, shape, device):
        for field in fields(self):
            value = getattr(self, field.name)
            expected = (*shape, 2) if field.name in ('wheel_touch', 'target_wheel_touch') else shape
            correct_dtype = (value.is_floating_point() if field.name == 'landing_world_height'
                             else value.dtype == torch.bool)
            if value.shape != expected or value.device != device or not correct_dtype:
                raise ValueError('Platform landing evidence requires matching typed physical rows')

    def tick(self, index: int) -> PlatformLandingEvidence:
        return PlatformLandingEvidence(**{field.name: getattr(self, field.name)[:, index]
                                           for field in fields(self)})


def advance_platform_landing(state: PlatformLandingState, evidence: PlatformLandingEvidence,
                             *, physical_id: Tensor, ascent_seen: Tensor) -> PlatformLandingState:
    """Advance direct landing qualification without mutating previous history.

    A wrong first landing cannot be repaired by subsequently driving onto the
    target. Missing measurements during the direct approach leave qualification
    unknown, rather than claiming a missed landing or fabricating clear air.
    No outcome here terminates the physical environment or request protocol.
    """
    state.validate(physical_id.shape, physical_id.device)
    evidence.validate(physical_id.shape, physical_id.device)
    active = physical_id >= 0
    bound = active & (state.request_id == physical_id)
    selected = active & evidence.selected
    new_request = selected & ~bound
    # A new flat request also discards an earlier platform request's history.
    reset = active & ~bound
    request_id = torch.where(reset, -1, state.request_id)
    request_id = torch.where(new_request, physical_id, request_id)
    phase = torch.where(reset, 0, state.phase)
    free_ticks = torch.where(reset, 0, state.free_ticks)
    owned = bound | selected
    valid = evidence.measurement_valid & evidence.selected
    wheel_touch = evidence.wheel_touch
    target_touch = evidence.target_wheel_touch & wheel_touch
    free = (owned & valid & evidence.full_body_clear & ~wheel_touch.any(-1)
            & ~evidence.nonwheel_contact)
    free_ticks = torch.where(free, free_ticks + 1, 0)
    before = phase == int(PlatformLandingPhase.BEFORE_FLIGHT)
    flight = owned & before & (free_ticks >= 2) & ascent_seen
    phase = torch.where(flight, int(PlatformLandingPhase.FLIGHT), phase)
    approaching = owned & ((phase == int(PlatformLandingPhase.FLIGHT))
                           | (phase == int(PlatformLandingPhase.TARGET_SINGLE_WHEEL)))
    bad_contact = evidence.nonwheel_contact | (wheel_touch & ~target_touch).any(-1)
    phase = torch.where(approaching & valid & bad_contact,
                        int(PlatformLandingPhase.MISSED), phase)
    phase = torch.where(approaching & ~valid, int(PlatformLandingPhase.UNKNOWN_CONTACT), phase)
    direct = approaching & valid & ~bad_contact
    phase = torch.where(direct & target_touch.any(-1),
                        int(PlatformLandingPhase.TARGET_SINGLE_WHEEL), phase)
    phase = torch.where(direct & target_touch.all(-1),
                        int(PlatformLandingPhase.TARGET_BOTH_WHEELS), phase)
    return PlatformLandingState(request_id, free_ticks, phase)


def observe_jump_physics(owner, force, velocity_z, tilt, world_height, *, angular_speed_rad_s,
                         dt=.005, platform_state: PlatformLandingState | None = None,
                         platform_evidence: PlatformLandingEvidence | None = None):
    """Update the existing native owner once and return optional platform history.

    Install this before native flight-latch/reward observers. Those observers
    still run once afterwards; this function does not advance the flight latch.
    The platform branch changes qualification at the sole success/payment point,
    never by retracting an already paid flat success.
    """
    from transformer_rl.formal.control.observation import CommandMode
    from transformer_rl.formal.tasks.reward import RewardPhase

    if (platform_state is None) != (platform_evidence is None):
        raise ValueError('Platform landing needs both request history and physical evidence')
    if platform_state is not None:
        platform_state.validate(owner.physical_id.shape, owner.physical_id.device)
        platform_evidence.validate(owner.physical_id.shape, owner.physical_id.device)
    active = owner.physical_id >= 0
    air = (force <= 2).all(-1)
    owner.air_ticks.copy_(torch.where(active & air, owner.air_ticks + 1, 0))
    owner.flight_seen |= active & (owner.air_ticks >= 2)
    actual_rise = world_height - owner.ground_anchor - owner.latched_height
    owner.ascent_seen |= active & (velocity_z > .15) & (actual_rise > .01)
    owner.peak_rise.copy_(torch.where(active, torch.maximum(owner.peak_rise, actual_rise), owner.peak_rise))
    contact = (force >= 5).all(-1)
    tolerance = torch.minimum(torch.full_like(owner.active_rise, .03), owner.active_rise * .5)
    achieved = (owner.peak_rise - owner.active_rise).abs() <= tolerance + 1e-6
    quiet = (velocity_z.abs() <= .15) & (tilt <= .25) & (angular_speed_rad_s <= .5)
    owner._last_supported.copy_(contact & quiet)
    owner._last_world_height.copy_(world_height)
    landing = actual_rise.abs() <= .05
    if platform_state is not None:
        platform_state = advance_platform_landing(platform_state, platform_evidence,
                                                  physical_id=owner.physical_id,
                                                  ascent_seen=owner.ascent_seen)
        selected = active & (platform_evidence.selected | (platform_state.request_id == owner.physical_id))
        target_landing = ((platform_state.phase == int(PlatformLandingPhase.TARGET_BOTH_WHEELS))
                          & platform_evidence.selected & platform_evidence.measurement_valid
                          & platform_evidence.landing_support_valid & ~platform_evidence.nonwheel_contact
                          & ((world_height - platform_evidence.landing_world_height).abs() <= .05))
        landing = torch.where(selected, target_landing, landing)
    stable = (active & owner.supported_start & owner.flight_seen & owner.ascent_seen & achieved
              & contact & quiet & landing)
    owner.landed_s.copy_(torch.where(stable, owner.landed_s + dt, 0))
    owner.phase.copy_(torch.where(
        owner.active_id >= 0,
        torch.where(air & owner.flight_seen, int(RewardPhase.FLIGHT),
                    torch.where(owner.flight_seen, int(RewardPhase.LANDING), int(RewardPhase.TAKEOFF))),
        torch.where(owner.base_mode == int(CommandMode.RECOVERY), int(RewardPhase.RECOVERY),
                    int(RewardPhase.SUPPORT))))
    success = (stable & ~owner._physical_cancelled & (owner.landed_s >= 2.)
               & (owner.physical_id > owner.paid_id))
    owner.physical_outcome_counts[0].add_(success.sum())
    owner.success_id.copy_(torch.where(success, owner.physical_id, owner.success_id))
    owner.paid_id.copy_(torch.where(success, owner.physical_id, owner.paid_id))
    owner.physical_id.masked_fill_(success, -1)
    return platform_state
