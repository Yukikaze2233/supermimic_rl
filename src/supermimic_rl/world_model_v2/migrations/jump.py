"""Explicit recorded-jump branch; old physical data never invents task memory."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json

import torch

from ...contracts import StateSchema
from ...course.jump_first import SCHEDULE, make_course_curriculum
from .critic import append_critic_features
from ..imagination import Critic
from ..jump_task import JUMP_FEATURE_DIM, JUMP_RECORD_DTYPES, JUMP_RECORD_FIELDS, JumpTaskConfig
from ..motion_task import MOTION_TASK_DIM
from ..online import OnlineConfig
from ..records import MODEL_CONTRACT


def _migrate_course(saved, source, target, source_raw, target_raw, coordinate, num_envs):
    """Only the pinned jump-first reorder and explicit exposure-budget change."""
    if type(coordinate) is not int or coordinate < 0:
        raise ValueError('Jump branch needs an explicit nonnegative course coordinate')
    courses = []
    for provenance, raw in ((source, source_raw), (target, target_raw)):
        course = json.loads(raw)
        if (hashlib.sha256(raw).hexdigest() != provenance['course_sha256']
                or provenance['curriculum'] != course):
            raise ValueError('Jump migration course source pin differs')
        courses.append(course)
    old_course, new_course = courses
    ignored = ('course_sha256', 'curriculum', 'course_schedule')
    if ({key: value for key, value in source.items() if key not in ignored}
            != {key: value for key, value in target.items() if key not in ignored}):
        raise ValueError('Jump migration cannot change physics, execution, terrain or reward source contracts')
    changed = old_course != new_course
    if changed:
        if ('schedule' in old_course or new_course.get('schedule') != SCHEDULE
                or set(new_course) != {'schedule', 'base'}):
            raise ValueError('Only the explicit legacy-to-jump-first course migration is supported')
        old_base, new_base = dict(old_course), dict(new_course['base'])
        old_base.pop('stage_transition_budgets')
        new_base.pop('stage_transition_budgets')
        if old_base != new_base:
            raise ValueError('Jump-first migration may reorder exposure and change budgets, not task domains')
        expected_schedule = {
            'schedule': SCHEDULE, 'request_heading_override': False, 'native_handoff_preserved': True,
            'early_jump_domain': list(new_course['base']['initial_domain']['jump_m']),
        }
        if source.get('course_schedule') is not None or target.get('course_schedule') != expected_schedule:
            raise ValueError('Jump-first native command-pool receipt differs')
    elif source.get('course_schedule') != target.get('course_schedule'):
        raise ValueError('Unchanged course cannot reinterpret its command-pool receipt')
    old = saved['environment']['curriculum']
    old_sampler = make_course_curriculum(num_envs, 'cpu', 0, old_course)
    if (old['format'] != old_sampler._FORMAT or old['config'] != old_sampler.config.to_dict()
            or old['num_envs'] != num_envs):
        raise ValueError('Saved curriculum does not match its pinned source course')
    new_sampler = make_course_curriculum(num_envs, 'cpu', 0, new_course)
    new_sampler.set_progress(coordinate, old['actor_updates'])
    if changed:
        current = new_sampler.state_dict()
        # Stage order and mixture weights change allocation credits, not the
        # historical counts or the underlying random-number stream.
        current['sampler']['rng'] = old['sampler']['rng']
        current['sample_counts'] = deepcopy(old['sample_counts'])
    else:
        current = {**old, 'real_transitions': coordinate}
    receipt = {
        'source_coordinate': old['real_transitions'], 'target_coordinate': coordinate,
        'target_stage': new_sampler.stage_names[new_sampler.stage],
        'source_course_sha256': source['course_sha256'], 'target_course_sha256': target['course_sha256'],
        'jump_first_migration': changed, 'sampler_credits_reset': changed,
        'sampling_counts_preserved': True, 'sampler_rng_preserved': True,
        'coordinate_is_capability_promotion': False,
    }
    return {**saved['environment'], 'curriculum': current}, receipt


def upgrade_jump_tasks(saved, dataset, reference, *, source_course_raw, target_course_raw,
                       course_coordinate, world_only_cycles=0):
    """Append 32 critic features and real jump records to the current stair branch."""
    config = OnlineConfig.from_dict(saved['config'])
    if (saved.get('format') != 'supermimic-online-v2' or saved.get('model_contract') != MODEL_CONTRACT
            or not config.record_stair_source or config.record_jumps or config.actor_frame_dim != 35
            or saved.get('task_norm_pending', False) or saved.get('critic_norm_from') is not None):
        raise ValueError('Jump expansion requires the calibrated chassis35 motion/stair/source branch')
    old_schema = StateSchema.from_dict(saved['schema'])
    target_schema = StateSchema.from_dict(reference['schema'])
    if (old_schema.fingerprint != saved['schema_fingerprint'] or dataset['cycle'] != saved['cycle']
            or dataset['schema_fingerprint'] != old_schema.fingerprint
            or reference.get('format') != 'supermimic-world-v2-dataset-v1'
            or reference.get('model_contract') != MODEL_CONTRACT
            or reference['schema_fingerprint'] != target_schema.fingerprint):
        raise ValueError('Jump migration requires a paired checkpoint and authenticated native reference')
    old_fields, new_fields = old_schema.to_dict(), target_schema.to_dict()
    source, target = json.loads(old_fields.pop('source')), json.loads(new_fields.pop('source'))
    if old_fields != new_fields:
        raise ValueError('Jump migration cannot change the physical/observation tensor contract')
    if json.dumps(saved['stair_scenario'], sort_keys=True) != json.dumps(reference['stair_source_config'], sort_keys=True):
        raise ValueError('Jump migration cannot replace the native stair protocol/source')
    jump_config = JumpTaskConfig.from_dict(reference['jump_task_config'])
    reference_data = reference['replay']['storage']
    if (any(name not in reference_data or reference_data[name].shape[2:] != shape
            or reference_data[name].dtype != JUMP_RECORD_DTYPES.get(name, torch.float32)
            for name, shape in JUMP_RECORD_FIELDS.items())
            or 'jump_record_valid' not in reference_data or not (reference_data['jump_record_valid'] > .5).any()):
        raise ValueError('Jump reference needs actual valid native jump-task records')
    if type(world_only_cycles) is not int or world_only_cycles < 0:
        raise ValueError('World-only adaptation cycles must be nonnegative')
    storage = dataset['replay']['storage']
    if any(name in storage for name in (*JUMP_RECORD_FIELDS, 'jump_record_valid')):
        raise ValueError('Jump migration cannot reinterpret existing jump fields')
    environment, course_receipt = _migrate_course(saved, source, target, source_course_raw, target_course_raw,
        course_coordinate, dataset['replay']['num_envs'])
    base = Critic.feature_size(old_schema.goal_dim, storage['sim_context'].shape[-1],
                              task_dim=MOTION_TASK_DIM, terrain=True, stairs=True)
    learner = append_critic_features(saved['learner'], expected_dim=base, extra_dim=JUMP_FEATURE_DIM)
    leading = storage['frame'].shape[:2]
    additions = {name: torch.zeros(*leading, *shape, device=storage['frame'].device,
                                  dtype=JUMP_RECORD_DTYPES.get(name, torch.float32))
                 for name, shape in JUMP_RECORD_FIELDS.items()}
    additions['jump_record_valid'] = storage['frame'].new_zeros(*leading, 1)
    next_config = OnlineConfig.from_dict({**asdict(config), 'record_jumps': True,
        'course_coordinate': course_coordinate, 'policy_start_cycle': saved['cycle'] + world_only_cycles})
    receipt = {
        'migration': 'stair-source-to-recorded-jumps-v1', 'cycle': saved['cycle'],
        'critic_input_before': base, 'critic_input_after': base + JUMP_FEATURE_DIM,
        'critic_initialization': 'new weights/Adam moments zero; new mean zero/variance one; old columns unchanged',
        'critic_norm': 'calibrate only new columns from fresh valid native records before policy learning',
        'legacy_jump_records_valid': 0, 'old_physical_transitions_retained': leading[0] * leading[1],
        'source_schema': old_schema.fingerprint, 'target_schema': target_schema.fingerprint,
        'course': course_receipt, 'policy_start_cycle': next_config.policy_start_cycle,
        'preserved': ['actor35 and estimator', 'world model and Adam', 'old critic columns/normalization/Adam',
                      'other optimizers, update counters and RNGs', 'physical replay, boundaries and sequence IDs',
                      'collection counters and teacher schedule'],
        'legacy_policy_starts': 'unrecorded jump memories remain invalid, including ordinary goals',
        'scope': 'explicit jump producer and course branch; no old request state or skill qualification invented',
    }
    upgraded = {**saved, 'schema': target_schema.to_dict(), 'schema_fingerprint': target_schema.fingerprint,
        'config': asdict(next_config), 'learner': learner, 'environment': environment,
        'jump_scenario': asdict(jump_config), 'critic_norm_from': base,
        'resume_migrations': [*saved.get('resume_migrations', []), receipt]}
    replay = {**dataset, 'schema_fingerprint': target_schema.fingerprint,
              'replay': {**dataset['replay'], 'storage': {**storage, **additions}}}
    if 'schema' in dataset:
        replay['schema'] = target_schema.to_dict()
    return upgraded, replay, receipt
