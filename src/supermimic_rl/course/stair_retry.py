"""Opt-in retries after a completed timeout of the ideal simulated stair source."""
from __future__ import annotations


def timeout_retry_receipt() -> dict:
    return {
        'contract': 'ideal_tof_same_riser_timeout_retry_v1',
        'end_reason': 'RequestEnd.TIMEOUT after recording the finished request',
        'next_request': 'new ID through the existing detector, latency and admission',
        'state_change': 'riser_cleared only; no extension of the finished request or height hint',
    }


def allow_timeout_retry(trigger, ended, reason) -> None:
    """Unlock the existing detector only after a request has actually timed out."""
    from transformer_rl.formal.tasks.protocol import RequestEnd

    trigger.riser_cleared |= ended & (reason == int(RequestEnd.TIMEOUT))


def install_stair_timeout_retry(native, *, enabled: bool = False) -> dict | None:
    """Keep native end accounting and rearming; unlock its same-riser latch."""
    if type(enabled) is not bool:
        raise ValueError('Stair timeout retry must be a boolean')
    if not enabled:
        return None
    course = native._terrain_course
    if getattr(course, 'supermimic_stair_timeout_retry', False):
        return timeout_retry_receipt()
    record_end = course._record_stair_end

    def record_then_unlock(ended, reason):
        # The original request remains a failure in native telemetry. This
        # permits a later detection, never a continuation of its old hint.
        record_end(ended, reason)
        allow_timeout_retry(course.tof_trigger, ended, reason)

    course._record_stair_end = record_then_unlock
    course.supermimic_stair_timeout_retry = True
    return timeout_retry_receipt()
