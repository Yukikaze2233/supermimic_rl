"""Physical support/flight/landing clocks shared by real-state checks and imagination.

The source configuration is supplied by native provenance, never inferred from
absence of support. In rough_dash a riser collision, base contact or insufficient
clearance cannot qualify flight. Discrete phase transitions stop gradients;
geometry is evaluated separately so extension-height reward stays differentiable.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor


@dataclass(frozen=True)
class SupportState:
    contact: Tensor
    airborne: Tensor
    unsupported_s: Tensor
    landing_remaining_s: Tensor
    phase_active: Tensor
    phase_elapsed_s: Tensor
    wheel_contact_s: Tensor
    base_contact_s: Tensor

    @classmethod
    def from_native(cls, native):
        return cls(*(getattr(native, '_' + field.name).clone() for field in fields(cls)))

    @classmethod
    def decode(cls, flat: Tensor):
        values = flat.split((2, 1, 1, 1, 1, 1, 2, 1), -1)
        contact, air, unsupported, landing, active, elapsed, wheel, base = values
        return cls(contact > .5, air[..., 0] > .5, unsupported[..., 0], landing[..., 0],
                   active[..., 0] > .5, elapsed[..., 0], wheel, base[..., 0])

    def encode(self):
        columns = (self.contact, self.airborne[..., None], self.unsupported_s[..., None],
                   self.landing_remaining_s[..., None], self.phase_active[..., None],
                   self.phase_elapsed_s[..., None], self.wheel_contact_s, self.base_contact_s[..., None])
        return torch.cat([value.to(self.unsupported_s.dtype) for value in columns], -1)


@dataclass(frozen=True)
class SupportInputs:
    valid: Tensor
    eligible: Tensor
    root_z: Tensor
    centre_z: Tensor
    bottom_z: Tensor
    root_ground_z: Tensor
    wheel_ground_z: Tensor
    wheel_support_load: Tensor
    wheel_resultant_force: Tensor
    base_resultant_force: Tensor


@dataclass(frozen=True)
class SupportResult:
    state: SupportState
    airborne: Tensor
    landing: Tensor
    just_landed: Tensor


@torch.no_grad()
def advance_support(state: SupportState, evidence: SupportInputs, config, dt: float) -> SupportResult:
    """One completed physical tick; configuration is the actual TerrainSupportConfig."""
    valid = evidence.valid
    load = evidence.wheel_support_load
    contact = valid[:, None] & ((load >= config.support_on_n)
                               | (state.contact & (load >= config.support_off_n)))
    supported = contact.any(-1)
    if config.profile == 'legacy':
        unsupported = valid & ~supported
        unsupported_s = torch.where(unsupported, state.unsupported_s + dt, 0.)
        just_landed = valid & supported & state.airborne
        airborne = unsupported & (state.airborne | (unsupported_s >= config.airborne_confirm_s - 1e-6))
        remaining = torch.where(just_landed, config.landing_duration_s,
                                (state.landing_remaining_s - dt).clamp_min(0.))
        remaining = torch.where(airborne | ~valid, 0., remaining)
        landing = valid & ~airborne & (remaining > 1e-6)
        next_state = SupportState(contact, airborne, unsupported_s, remaining, state.phase_active,
                                  state.phase_elapsed_s, state.wheel_contact_s, state.base_contact_s)
        return SupportResult(next_state, airborne, landing, just_landed)
    if config.profile != 'rough_dash':
        raise ValueError('Unsupported native terrain support profile')
    measurable = valid & evidence.eligible
    natural_enter = (measurable & ~supported
                     & ~(evidence.wheel_resultant_force > config.wheel_landing_force_n).any(-1)
                     & (evidence.base_resultant_force < config.base_contact_force_n)
                     & (evidence.root_z - evidence.root_ground_z > config.body_clearance_enter_m)
                     & ((evidence.bottom_z - evidence.wheel_ground_z) > config.wheel_clearance_enter_m).all(-1))
    unsupported_s = torch.where(natural_enter & ~state.phase_active, state.unsupported_s + dt, 0.)
    enter = natural_enter & ~state.phase_active & (unsupported_s >= config.airborne_confirm_s - 1e-6)
    continuing = measurable & state.phase_active
    low = evidence.centre_z - evidence.wheel_ground_z < config.wheel_landing_center_height_m
    countable = (evidence.wheel_resultant_force > config.wheel_landing_force_n) & low
    wheel_time = torch.where(continuing[:, None], torch.where(
        countable, state.wheel_contact_s + dt, torch.where(low, state.wheel_contact_s, 0.)), 0.)
    base_time = torch.where(continuing, state.base_contact_s
                            + (evidence.base_resultant_force > config.base_contact_force_n) * dt, 0.)
    elapsed = torch.where(continuing, state.phase_elapsed_s + dt, 0.)
    exit_phase = ((wheel_time >= config.wheel_landing_exit_s - 1e-6).any(-1)
                  | (base_time >= config.base_contact_exit_s - 1e-6)
                  | (elapsed >= config.maximum_phase_s - 1e-6))
    active = measurable & (state.phase_active | enter) & ~exit_phase
    airborne = (active & ~supported & (evidence.base_resultant_force < config.base_contact_force_n)
                & ~(evidence.wheel_resultant_force > config.wheel_landing_force_n).any(-1))
    just_landed = active & supported & state.airborne
    landing = active & ~airborne
    next_state = SupportState(contact, airborne, unsupported_s, state.landing_remaining_s,
                              active, elapsed, wheel_time, base_time)
    return SupportResult(next_state, airborne, landing, just_landed)
