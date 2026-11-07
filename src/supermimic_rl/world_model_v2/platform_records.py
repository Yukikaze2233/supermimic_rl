"""Typed policy-start caller/landing history, separate from motion labels.

Missing legacy records mean unknown, not an inactive caller. A known inactive
row is captured explicitly with the same typed state as an active request.
Contact validity belongs to each recorded physics tick, not the start snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..course.jump_physics import PlatformLandingState
from ..course.platform_intent import (
    PLATFORM_INTENT_RECORD_DTYPES, PLATFORM_INTENT_RECORD_FIELDS, PlatformIntentState,
)


PLATFORM_RECORD_VERSION = 1
PLATFORM_STATE_RECORD_FIELDS = {
    **PLATFORM_INTENT_RECORD_FIELDS,
    'platform_landing_ids': (3,),  # Request ID, consecutive full-clear ticks, landing phase.
    'platform_record_valid': (1,),
}
PLATFORM_RECORD_FIELDS = {**PLATFORM_STATE_RECORD_FIELDS, 'contact_valid_ticks': (4, 1)}
PLATFORM_RECORD_DTYPES = {
    **PLATFORM_INTENT_RECORD_DTYPES,
    'platform_landing_ids': torch.int64,
    'platform_record_valid': torch.bool,
    'contact_valid_ticks': torch.bool,
}


def validate_platform_fields(fields):
    """Permit the unchanged legacy layout or the complete opt-in record."""
    if not set(fields).intersection(PLATFORM_RECORD_FIELDS):
        return
    if any(name not in fields or tuple(fields[name]) != shape
           for name, shape in PLATFORM_RECORD_FIELDS.items()):
        raise ValueError('Platform replay requires the complete typed caller, landing and contact record')


@dataclass(frozen=True)
class PlatformRecordState:
    intent: PlatformIntentState
    landing: PlatformLandingState
    valid: Tensor  # [N, 1]: captured state is known; this does not certify a landing.

    @classmethod
    def capture(cls, native, landing_capture, *, count: int, device):
        caller = getattr(native, 'supermimic_platform_intent', None)
        if caller is None or not callable(getattr(caller, 'capture', None)):
            raise ValueError('Platform collection requires an explicitly installed native caller')
        if not callable(landing_capture):
            raise ValueError('Platform collection requires the actual landing owner capture')
        intent = PlatformIntentState.decode(caller.capture())
        if intent.phase.shape != (count,) or intent.phase.device != torch.device(device):
            raise ValueError('Platform caller differs from the native collection rows/device')
        landing = landing_capture()
        if not isinstance(landing, PlatformLandingState):
            raise ValueError('Landing capture must return the actual PlatformLandingState')
        landing.validate((count,), intent.phase.device)
        # Native buffers may be mutated by substeps and auto-reset immediately
        # afterwards. Never let a start record alias those live tensors.
        landing = PlatformLandingState(landing.request_id.clone(), landing.free_ticks.clone(), landing.phase.clone())
        return cls(intent, landing, torch.ones(count, 1, device=intent.phase.device, dtype=torch.bool))

    def record(self):
        return {**self.intent.record(),
                'platform_landing_ids': torch.stack(
                    (self.landing.request_id, self.landing.free_ticks, self.landing.phase), -1),
                'platform_record_valid': self.valid.clone()}

    @classmethod
    def decode(cls, record, *, version=PLATFORM_RECORD_VERSION):
        if type(version) is not int or version != PLATFORM_RECORD_VERSION:
            raise ValueError('Unknown platform record version')
        if not all(name in record for name in PLATFORM_STATE_RECORD_FIELDS):
            raise ValueError('Missing platform state records are unknown, not inactive')
        intent = PlatformIntentState.decode(record)
        n, device = len(intent.phase), intent.phase.device
        for name, shape in PLATFORM_STATE_RECORD_FIELDS.items():
            value = record[name]
            if (value.shape != (n, *shape) or value.device != device
                    or value.dtype != PLATFORM_RECORD_DTYPES[name]):
                raise ValueError('Invalid typed platform state record: ' + name)
        ids = record['platform_landing_ids']
        landing = PlatformLandingState(*(ids[:, index].clone() for index in range(3)))
        return cls(intent, landing, record['platform_record_valid'].clone())
