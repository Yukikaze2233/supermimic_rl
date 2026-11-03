"""SCUT-style stair flag: STAIR in the actor's mode one-hot only while the ToF stair state machine is active.

The native ``use_mode_labels`` switch shows both the ToF state-machine label
and labels derived from the routed terrain tile; the latter is ground truth the
robot does not have. This keeps the former and drops the latter, so the flag
comes from the request FSM instead of a terrain ID. The simulation source is
still an idealized forward terrain profile; its position-assisted completion
needs a separate deployment sensor contract.
"""
from __future__ import annotations


def install_tof_stair_label(native) -> dict:
    policy = getattr(native.cfg, 'terrain_policy_spec', None)
    if policy is None or not policy.use_mode_labels:
        raise ValueError('The ToF stair label needs runtime terrain_policy.use_mode_labels = true')
    native._terrain_course.tasks.terrain_mode_labels = False
    return {'stair_label': 'STAIR set by the ToF stair state machine only; terrain-kind labels disabled'}
