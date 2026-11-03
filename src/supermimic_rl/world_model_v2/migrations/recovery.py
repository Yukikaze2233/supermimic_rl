"""Explicit recovery-memory expansion of the calibrated chassis35 jump branch."""
from dataclasses import asdict
import hashlib
import json

import torch

from ...contracts import StateSchema
from ...course.jump_first import SCHEDULE, make_course_curriculum
from .critic import append_critic_features
from ..imagination import Critic
from ..jump_task import JumpTaskConfig
from ..motion_task import MOTION_TASK_DIM
from ..online import OnlineConfig
from ..records import MODEL_CONTRACT
from ..recovery_task import (
    RECOVERY_FEATURE_DIM, RECOVERY_RECORD_DTYPES, RECOVERY_RECORD_FIELDS, RecoveryTaskConfig,
)


def _place_course(saved, source, course_raw, coordinate, num_envs):
    if type(coordinate) is not int or coordinate < 0:
        raise ValueError('Recovery branch needs an explicit nonnegative course coordinate')
    course = json.loads(course_raw)
    if (course.get('schedule') != SCHEDULE or set(course) != {'schedule', 'base'}
            or hashlib.sha256(course_raw).hexdigest() != source['course_sha256']
            or course != source['curriculum']):
        raise ValueError('Recovery expansion requires the unchanged pinned jump-first course')
    old = saved['environment']['curriculum']
    sampler = make_course_curriculum(num_envs, 'cpu', 0, course)
    sampler.load_state_dict(old)
    # Course position is an explicit lookup coordinate. Actual transitions,
    # sample counts, allocation credits and generator states are not rewritten.
    placed = make_course_curriculum(num_envs, 'cpu', 0, course)
    placed.set_progress(coordinate, old['actor_updates'])
    receipt = {
        'source_coordinate': old['real_transitions'], 'target_coordinate': coordinate,
        'target_stage': placed.stage_names[placed.stage],
        'course_sha256': source['course_sha256'], 'sampler_credits_reset': False,
        'sampling_counts_preserved': True, 'sampler_rng_preserved': True,
        'coordinate_is_capability_promotion': False,
    }
    return {**saved['environment'], 'curriculum': {**old, 'real_transitions': coordinate}}, receipt


def upgrade_recovery_tasks(saved, dataset, reference, *, course_raw, course_coordinate,
                           world_only_cycles=0):
    """Append seven critic features without reinterpreting old physical records.

    The native reference supplies the actual lifecycle/handoff configuration.
    No world state, observation layout, physical source or course is changed.
    Old transitions retain world-model supervision, but lack recovery memory
    and cannot seed the newly enabled task producer.
    """
    config = OnlineConfig.from_dict(saved['config'])
    if saved.get('jump_scenario') is not None and JumpTaskConfig.from_dict(saved['jump_scenario']).trigger is not None:
        raise ValueError('Expand recovery before the optional jump trigger critic columns')
    if (saved.get('format') != 'supermimic-online-v2' or saved.get('model_contract') != MODEL_CONTRACT
            or not config.record_stair_source or not config.record_jumps or config.record_recovery
            or config.actor_frame_dim != 35 or saved.get('task_norm_pending', False)
            or saved.get('critic_norm_from') is not None or saved.get('recovery_scenario') is not None):
        raise ValueError('Recovery expansion requires the calibrated chassis35 jump/stair/source branch')
    schema = StateSchema.from_dict(saved['schema'])
    target_schema = StateSchema.from_dict(reference['schema'])
    if (schema.fingerprint != saved['schema_fingerprint'] or dataset['cycle'] != saved['cycle']
            or dataset['schema_fingerprint'] != schema.fingerprint
            or reference.get('format') != 'supermimic-world-v2-dataset-v1'
            or reference.get('model_contract') != MODEL_CONTRACT
            or reference['schema_fingerprint'] != target_schema.fingerprint):
        raise ValueError('Recovery expansion requires a paired checkpoint and native reference')
    if schema.to_dict() != target_schema.to_dict():
        raise ValueError('Recovery expansion cannot change the physical source, tensor contract or course')
    if (json.dumps(saved['stair_scenario'], sort_keys=True)
            != json.dumps(reference['stair_source_config'], sort_keys=True)
            or json.dumps(saved['jump_scenario'], sort_keys=True)
            != json.dumps(reference['jump_task_config'], sort_keys=True)):
        raise ValueError('Recovery expansion cannot replace the native jump/stair task configuration')
    if 'recovery_task_config' not in reference:
        raise ValueError('Recovery reference lacks its recorded native lifecycle configuration')
    recovery = RecoveryTaskConfig.from_dict(reference['recovery_task_config'])
    recorded = reference['replay']['storage']
    flag = recorded.get('recovery_record_valid')
    if (flag is None or flag.ndim != 3 or flag.shape[-1] != 1 or flag.dtype != torch.float32
            or any(name not in recorded or recorded[name].shape != (*flag.shape[:2], *shape)
                   or recorded[name].dtype != RECOVERY_RECORD_DTYPES[name]
                   for name, shape in RECOVERY_RECORD_FIELDS.items())):
        raise ValueError('Recovery reference needs complete native recovery-task records')
    valid, owned = recorded.get('valid'), recorded.get('policy_owned')
    if (valid is None or owned is None or valid.shape != flag.shape[:2] or owned.shape != flag.shape[:2]
            or valid.dtype != torch.bool or owned.dtype != torch.bool
            or not ((flag[..., 0] > .5) & valid & owned).any()):
        raise ValueError('Recovery reference needs actual valid policy-owned native records')
    if type(world_only_cycles) is not int or world_only_cycles < 0:
        raise ValueError('World-only adaptation cycles must be nonnegative')
    storage = dataset['replay']['storage']
    if any(name in storage for name in (*RECOVERY_RECORD_FIELDS, 'recovery_record_valid')):
        raise ValueError('Recovery expansion cannot reinterpret existing recovery fields')
    if any(name not in storage for name in ('jump_ids', 'jump_clock', 'jump_values', 'jump_record_valid')):
        raise ValueError('Recovery expansion requires the complete recorded-jump replay')
    environment, course_receipt = _place_course(saved, json.loads(schema.source), course_raw,
                                               course_coordinate, dataset['replay']['num_envs'])
    base = Critic.feature_size(schema.goal_dim, storage['sim_context'].shape[-1],
                              task_dim=MOTION_TASK_DIM, terrain=True, stairs=True, jumps=True)
    learner = append_critic_features(saved['learner'], expected_dim=base, extra_dim=RECOVERY_FEATURE_DIM)
    leading = storage['frame'].shape[:2]
    additions = {name: torch.zeros(*leading, *shape, device=storage['frame'].device,
                                  dtype=RECOVERY_RECORD_DTYPES[name])
                 for name, shape in RECOVERY_RECORD_FIELDS.items()}
    additions['recovery_record_valid'] = storage['frame'].new_zeros(*leading, 1)
    next_config = OnlineConfig.from_dict({**asdict(config), 'record_recovery': True,
        'course_coordinate': course_coordinate, 'policy_start_cycle': saved['cycle'] + world_only_cycles})
    receipt = {
        'migration': 'recorded-jumps-to-recorded-recovery-v1', 'cycle': saved['cycle'],
        'critic_input_before': base, 'critic_input_after': base + RECOVERY_FEATURE_DIM,
        'critic_initialization': 'new weights/Adam moments zero; new mean zero/variance one; old columns unchanged',
        'critic_norm': 'calibrate only new columns from fresh valid native records before policy learning',
        'legacy_recovery_records_valid': 0, 'old_physical_transitions_retained': leading[0] * leading[1],
        'schema_fingerprint': schema.fingerprint, 'course': course_receipt,
        'policy_start_cycle': next_config.policy_start_cycle,
        'preserved': ['actor35 and estimator', 'world model and Adam', 'jump/stair producer configurations',
                      'old critic columns/normalization/Adam', 'other optimizers, update counters and RNGs',
                      'physical replay, boundaries and sequence IDs', 'collection counters and teacher schedule',
                      'course sample counts, sampler allocation credits and RNG'],
        'legacy_policy_starts': 'unrecorded recovery memories remain invalid, including ordinary goals',
        'scope': 'explicit recovery producer; no source change, invented task history or skill qualification',
    }
    upgraded = {**saved, 'config': asdict(next_config), 'learner': learner, 'environment': environment,
        'recovery_scenario': asdict(recovery), 'critic_norm_from': base,
        'resume_migrations': [*saved.get('resume_migrations', []), receipt]}
    replay = {**dataset, 'replay': {**dataset['replay'], 'storage': {**storage, **additions}}}
    return upgraded, replay, receipt
