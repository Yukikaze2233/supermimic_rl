"""SCUT-style stair flag: STAIR in the actor's mode one-hot only while the ToF stair state machine is active.

The native ``use_mode_labels`` switch shows both the ToF state-machine label
and labels derived from the routed terrain tile; the latter is ground truth the
robot does not have. This keeps the former and drops the latter, so the flag
comes from the request FSM instead of a terrain ID. The simulation source is
still an idealized forward terrain profile; its position-assisted completion
needs a separate deployment sensor contract.
"""
from __future__ import annotations

from dataclasses import replace


def install_tof_stair_label(native, *, height_hint_lifetime="fixed") -> dict:
    """Keep the native FSM and optionally bind its hint to request expiry.

    A request owns a finite deadline even if no completion packet arrives.
    Using that existing duration keeps its latched height until release, while
    retaining native completion/cancel/abort handling and exact tick counting.
    The default neither changes configuration nor adds a provenance field.
    """
    if height_hint_lifetime not in ("fixed", "request"):
        raise ValueError('Stair height hint lifetime must be fixed or request')
    policy = getattr(native.cfg, 'terrain_policy_spec', None)
    if policy is None or not policy.use_mode_labels:
        raise ValueError('The ToF stair label needs runtime terrain_policy.use_mode_labels = true')
    receipt = {'stair_label': 'STAIR set by the ToF stair state machine only; terrain-kind labels disabled'}
    if height_hint_lifetime == "request":
        assist = getattr(native.cfg, 'stair_height_assist_spec', None)
        if assist is None or not assist.enabled:
            raise ValueError('Request-lifetime stair hints require an enabled native height assist')
        timeout = native._terrain_course.stair_requests.config.timeout_s
        native.cfg.stair_height_assist_spec = replace(assist, hold_s=timeout)
        receipt['height_hint'] = {
            'contract': 'stair_request_lifetime_hint_v1',
            'runtime_hold_s': assist.hold_s,
            'effective_hold_s': timeout,
            'duration_source': 'StairRequestProtocol.config.timeout_s',
            'height_reference': 'native request-entry target, unchanged raise and maximum height',
            'release': 'native completion, cancellation, timeout or abort; replayed IDs do not refresh',
            'priority': 'native jump-active abort; no raised output in base JUMP or RECOVERY mode',
        }
    native._terrain_course.tasks.terrain_mode_labels = False
    return receipt
