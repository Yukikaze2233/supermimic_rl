"""Preserve learning state when enabling reviewed flight geometry and rest rewards."""
from copy import deepcopy
import hashlib
import json

from .reward_continuity import _upgrade_reward_pair


FORMAT = 'jump-flight-geometry-and-rest-v1'
SOURCE_KERNEL_SHA256 = '459439a4f56c19d3115ed55ad415386129db7a5325a603849fdc0ae8094b3b61'
# Shared by the integrated source and its formal deployment backport.
TARGET_KERNEL_SHA256 = '8480365bebf37c79e4c9035e05d52557442f0dd5e799d0263bd6a472514c71e9'
GEOMETRY_RECEIPT_SHA256 = 'f186eaab8a206917d11ed2fe2572f70d3e68e61db7bef8bdd2b071a3b4a785b4'
GIMBAL_RECEIPT_SHA256 = '9b135d9704909b42e0a09c22bc1e69cb8449f0b0ccced4a3013ae8cab59281eb'


def verify_flight_reward(source, target):
    """Allow these reward changes without reinterpreting old measured transitions."""
    if not isinstance(source, dict) or not isinstance(target, dict):
        raise ValueError('Flight reward migration needs complete native sources')
    before, after = source.get('reward_kernel'), target.get('reward_kernel')
    if (not isinstance(before, dict) or not isinstance(after, dict)
            or before.get('implementation') != 'shared_physical_reward_densities'
            or before.get('kernel_sha256') != SOURCE_KERNEL_SHA256
            or len(TARGET_KERNEL_SHA256) != 64 or len(GEOMETRY_RECEIPT_SHA256) != 64):
        raise ValueError('Flight reward migration requires the reviewed source and target')
    old = before.get('extensions', {})
    if (old.get('jump_flight_clearance', False) is not False
            or old.get('jump_tuck', 0.) != 0.
            or old.get('jump_flight', 0.) <= 0.
            or old.get('stationary_quiet') != 1.
            or any(old.get(name) is not True for name in (
                'jump_peak_completion', 'jump_physical_tail', 'jump_extension_height_floor'))):
        raise ValueError('Expected the current flight/settling reward before geometry activation')
    geometry = after.get('jump_flight_geometry')
    raw = json.dumps(geometry, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if hashlib.sha256(raw).hexdigest() != GEOMETRY_RECEIPT_SHA256:
        raise ValueError('Flight geometry differs from the reviewed asset-derived receipt')
    arbitration = target.get('request_gimbal_exclusion')
    raw = json.dumps(arbitration, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if (source.get('request_gimbal_exclusion') is not None
            or hashlib.sha256(raw).hexdigest() != GIMBAL_RECEIPT_SHA256):
        raise ValueError('Request/gimbal arbitration differs from the reviewed owner rule')
    schedule = source.get('course_schedule')
    if (not isinstance(schedule, dict) or schedule.get('schedule') != 'jump_first_v1'
            or schedule.get('request_motion') not in (None, 'birth_forward_held_v1')
            or schedule.get('request_motion_resume') not in (None, 'cold_physics_reset_repopulates_cache')
            or schedule.get('spin_motion') is not None):
        raise ValueError('Expected the current request/spin command sampler source')
    expected = deepcopy(source)
    expected['reward_kernel'] = {
        **before, 'kernel_sha256': TARGET_KERNEL_SHA256,
        'extensions': {**old, 'jump_flight_clearance': True, 'jump_tuck': 1.7},
        'jump_flight_geometry': deepcopy(geometry),
    }
    # Recorded segments already contain their actual commands. Only future
    # native draws change; imagination still truncates at an unknown segment.
    expected['course_schedule'] = {**schedule,
        'request_motion': 'birth_forward_held_v1',
        'request_motion_resume': 'cold_physics_reset_repopulates_cache',
        'spin_motion': 'zero_forward_spin_segments_v1'}
    expected['request_gimbal_exclusion'] = deepcopy(arbitration)
    if target != expected:
        raise ValueError('Flight reward migration cannot change physics, course, control or other weights')
    return {'format': FORMAT, 'before': deepcopy(source), 'after': deepcopy(target)}


def verify_saved_proof(proof):
    if not isinstance(proof, dict) or proof.get('format') != FORMAT:
        raise ValueError('Unknown flight reward proof')
    try:
        checked = verify_flight_reward(proof['before'], proof['after'])
    except (KeyError, TypeError) as error:
        raise ValueError('Incomplete flight reward proof') from error
    if checked != proof:
        raise ValueError('Flight reward proof has unregistered changes')
    return checked


def upgrade_flight_reward(saved, dataset, target_schema):
    """Retain all physical truth; exclude only old task-incompatible Actor starts."""
    from ...course.gimbal_arbitration import conflicting_request_starts

    updated, migrated, report = _upgrade_reward_pair(saved, dataset, target_schema,
        target_kernel=TARGET_KERNEL_SHA256, verify=verify_flight_reward,
        proof_key='jump_flight_geometry_reward')
    replay, storage = dataset['replay'], dataset['replay']['storage']
    conflict = conflicting_request_starts(storage)
    flags = storage['task_record_valid']
    excluded = conflict & flags[..., 0].bool()
    # Copy only this small admission field. In particular, do not manufacture
    # corrected commands/history, discard dynamics labels or copy the ring.
    if excluded.any():
        migrated = {**migrated, 'replay': {**replay, 'storage': {
            **storage, 'task_record_valid': flags.masked_fill(excluded[..., None], 0.)}}}
    report['request_gimbal_conflicting_starts'] = int(conflict.sum())
    report['request_gimbal_newly_excluded_starts'] = int(excluded.sum())
    report['historical_physical_task_records_and_validity'] = (
        'Physical truth/validity and historical commands/H5 unchanged; only conflicting '
        'REQUEST/special-gimbal Actor-start task_record_valid entries cleared in a copied field')
    report['world_model_supervision_preserved'] = True
    return updated, migrated, report
