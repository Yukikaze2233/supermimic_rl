"""Explicit a05 to R7 handoff: retain physical learning, replace task semantics."""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import inspect
import json
import math

import torch

from ...contracts import StateSchema
from ...course.curriculum import CourseCurriculumConfig
from ...course.jump_first import SCHEDULE, make_course_curriculum
from ..imagination import Critic
from ..jump_task import (
    JUMP_FLIGHT_LATCH_RECORD_DTYPES, JUMP_FLIGHT_LATCH_RECORD_FIELDS,
    JUMP_TRIGGER_FEATURE_DIM, JUMP_TRIGGER_RECORD_DTYPES, JUMP_TRIGGER_RECORD_FIELDS, JumpTaskConfig,
)
from ..motion_task import MOTION_TASK_DIM
from ..online import OnlineConfig
from ..records import MODEL_CONTRACT, RecordSpec
from ..teacher import file_sha256
from .teacher import rebind_teacher_source


TASK_VALID_FLAGS = ('task_record_valid', 'stair_source_valid', 'jump_record_valid', 'recovery_record_valid')
OPTIONAL_SOURCE_KEYS = ('jump_takeoff', 'stair_label', 'jump_assist')
SEMANTIC_SOURCE_KEYS = ('runtime_sha256', 'course_sha256', 'curriculum', 'reward_kernel',
                        'jump_takeoff', 'stair_label', 'course_schedule', 'numerical_isolation')


def _source(value):
    result = json.loads(value)
    for name in OPTIONAL_SOURCE_KEYS:
        result.setdefault(name, None)
    return result


def _check_runtime(source, target, source_raw, target_raw):
    runtimes = []
    for provenance, raw in ((source, source_raw), (target, target_raw)):
        if hashlib.sha256(raw).hexdigest() != provenance['runtime_sha256']:
            raise ValueError('Task handoff runtime SHA256 differs')
        runtime = json.loads(raw)
        disturbances = runtime['requested_randomization']
        if any(disturbances.get(name) != 0 for name in (
                'external_force_n', 'external_torque_nm', 'push_linear_m_s', 'push_yaw_rad_s')):
            raise ValueError('Task handoff cannot model external disturbances')
        runtimes.append(runtime)
    old, new = runtimes
    if (old['terrain_policy']['use_mode_labels'] is not False
            or new['terrain_policy']['use_mode_labels'] is not True):
        raise ValueError('Task handoff requires enabling the ToF mode label')
    old['terrain_policy']['use_mode_labels'] = True
    if old != new:
        raise ValueError('Task handoff may change only runtime terrain_policy.use_mode_labels')


def _course(saved, source, target, source_raw, target_raw, coordinate, num_envs, sampler_device):
    courses = []
    for provenance, raw in ((source, source_raw), (target, target_raw)):
        course = json.loads(raw)
        if (hashlib.sha256(raw).hexdigest() != provenance['course_sha256']
                or course != provenance['curriculum'] or course.get('schedule') != SCHEDULE
                or set(course) != {'schedule', 'base'}):
            raise ValueError('Task handoff needs the pinned jump-first source and target courses')
        courses.append(course)
    changed = {'stage_weights', 'recovery_postures', 'recovery_posture_stages',
               'jump_height_stage', 'jump_height_gates'}
    if ({k: v for k, v in courses[0]['base'].items() if k not in changed}
            != {k: v for k, v in courses[1]['base'].items() if k not in changed}
            or courses[1]['base'].get('jump_height_stage') != 'active_jump'
            or not courses[1]['base'].get('jump_height_gates')):
        raise ValueError('Task handoff only replaces v5 task mixtures and difficulty gates')
    old = saved['environment']['curriculum']
    sampler_device = torch.device(sampler_device)
    if sampler_device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA sampler migration requires an available CUDA device')
    original = make_course_curriculum(num_envs, sampler_device, 0, courses[0])
    expected_rng = original.sampler.generator.get_state()
    saved_rng = old['sampler']['rng']
    if saved_rng.shape != expected_rng.shape or saved_rng.dtype != expected_rng.dtype:
        raise ValueError(f'Saved sampler RNG does not match sampler_device={sampler_device}: '
                         f'{saved_rng.numel()} bytes versus {expected_rng.numel()}; '
                         'select the original native sampler device, without converting its RNG')
    # c15 saved its config before these three optional defaults were added.
    # Normalize only that known historical omission, then retain the native
    # strict source/configuration and sampler-state validation.
    defaults = {'jump_height_stage': 'combined_dr', 'jump_height_gates': [],
                'recovery_posture_stages': {}}
    expected = original.config.to_dict()
    missing = expected.keys() - old['config'].keys()
    if not missing <= defaults.keys() or old['config'].keys() - expected.keys():
        raise ValueError('Source curriculum checkpoint has unknown or missing nonlegacy fields')
    normalized = CourseCurriculumConfig.from_dict({**{k: defaults[k] for k in missing}, **old['config']}).to_dict()
    original.load_state_dict({**old, 'config': normalized})
    current = make_course_curriculum(num_envs, sampler_device, 0, courses[1])
    current.set_progress(coordinate, old['actor_updates'])
    state = current.state_dict()
    state['sampler']['rng'] = old['sampler']['rng']
    state['sample_counts'] = deepcopy(old['sample_counts'])
    current.load_state_dict(state)
    receipt = {'source_coordinate': old['real_transitions'], 'target_coordinate': coordinate,
        'target_stage': current.stage_names[current.stage], 'sampler_credits_reset': True,
        'sampling_counts_preserved': True, 'sampler_rng_preserved': True,
        'sampler_device': str(sampler_device), 'sampler_rng_bytes': saved_rng.numel(),
        'source_checkpoint_defaults_materialized': {k: defaults[k] for k in sorted(missing)},
        'coordinate_is_capability_promotion': False}
    return {**saved['environment'], 'curriculum': state}, receipt


def upgrade_task_semantics(saved, dataset, reference, *, source_runtime_raw, target_runtime_raw,
                           source_course_raw, target_course_raw, source_teacher_path, target_teacher_path,
                           course_coordinate, critic_warmup_cycles, sampler_device='cpu'):
    """Rebind a complete ungated a05 recovery pair to an observed R7 producer.

    Legacy task clocks, goals and frames are retained as historical data, never
    rewritten into new-policy starts. The world model still uses their physical
    action/tick supervision. New native rows alone calibrate and train the fresh
    critic. This does not qualify any skill or resume simulator state.
    ``sampler_device`` must match the native RNG backend; RNG bytes are retained.
    """
    config = OnlineConfig.from_dict(saved['config'])
    if (saved.get('format') != 'supermimic-online-v2' or saved.get('model_contract') != MODEL_CONTRACT
            or not all((config.record_tasks, config.record_stair_source, config.record_jumps, config.record_recovery))
            or config.actor_frame_dim != 35 or saved.get('task_norm_pending', False)
            or saved.get('critic_norm_from') is not None):
        raise ValueError('Task handoff requires the calibrated complete a05 chassis35 branch')
    if (type(course_coordinate) is not int or course_coordinate < 0
            or type(critic_warmup_cycles) is not int or critic_warmup_cycles < 1):
        raise ValueError('Explicit nonnegative coordinate and positive critic warmup cycles required')
    schema, target_schema = (StateSchema.from_dict(item['schema']) for item in (saved, reference))
    if (schema.fingerprint != saved['schema_fingerprint'] or dataset.get('cycle') != saved['cycle']
            or dataset['schema_fingerprint'] != schema.fingerprint
            or reference.get('format') != 'supermimic-world-v2-dataset-v1'
            or reference.get('model_contract') != MODEL_CONTRACT
            or reference['schema_fingerprint'] != target_schema.fingerprint):
        raise ValueError('Task handoff requires a paired checkpoint and authenticated native reference')
    old_fields, new_fields = schema.to_dict(), target_schema.to_dict()
    source, target = (_source(fields.pop('source')) for fields in (old_fields, new_fields))
    if (old_fields != new_fields or any(p['jump_assist'] is not None for p in (source, target))
            or {k: v for k, v in source.items() if k not in SEMANTIC_SOURCE_KEYS}
            != {k: v for k, v in target.items() if k not in SEMANTIC_SOURCE_KEYS}):
        raise ValueError('Task handoff cannot change physical, sensor, execution or asset contracts')
    old_schedule, new_schedule = source.get('course_schedule'), target.get('course_schedule')
    if old_schedule != new_schedule and (not isinstance(old_schedule, dict)
            or {**old_schedule, 'request_yaw_rate': 0.} != new_schedule):
        raise ValueError('Task handoff may only add the zero-yaw jump request schedule')
    old_isolation, new_isolation = source.get('numerical_isolation'), target.get('numerical_isolation')
    if old_isolation != new_isolation:
        # This repair changes only rejected rows, which already supply no
        # model targets. Keep the explicit receipt in the new source rather
        # than globally treating arbitrary simulator corrections as equivalent.
        expected_isolation = {
            'contract': 'physx_finite_coordinate_sampling_row_isolation_v3',
            'native_period_rad': 4 * math.pi,
            'scope': 'finite ambiguous position/speed samples; reject the entire policy interval',
            'rejected_rows': 'restore interval-start pose/joints with zero velocity, then normal episode reset',
            'native_state_modified': 'rejected rows only; never used for learning',
            'unknown_or_nonfinite_errors': 'raise',
        }
        if old_isolation is not None or new_isolation != expected_isolation:
            raise ValueError('Task handoff cannot change the dynamics of retained simulator rows')
    _check_runtime(source, target, source_runtime_raw, target_runtime_raw)
    old_reward, new_reward = source['reward_kernel'], target['reward_kernel']
    if ({k: v for k, v in old_reward.items() if k not in ('kernel_sha256', 'extensions')}
            != {k: v for k, v in new_reward.items() if k not in ('kernel_sha256', 'extensions')}
            or new_reward.get('implementation') != 'shared_physical_reward_densities'):
        raise ValueError('Task handoff cannot replace the native physical reward contract')
    from ...reward_kernel import RewardExtensions, physical_reward_densities
    if new_reward.get('kernel_sha256') != file_sha256(inspect.getfile(physical_reward_densities)):
        raise ValueError('Target reward fingerprint differs from the migration code')
    RewardExtensions(**new_reward['extensions'])
    old_jump, new_jump = (JumpTaskConfig.from_dict(value) for value in
                          (saved['jump_scenario'], reference['jump_task_config']))
    if old_jump.trigger is not None or new_jump.trigger is None or source['jump_takeoff'] is not None:
        raise ValueError('Task handoff requires an ungated source and gated target')
    expected_reference = replace(old_jump.reference,
        compression_depth_max_m=new_jump.trigger.compression_depth_max_m,
        compression_duration_s=new_jump.trigger.compression_duration_s,
        nominal_extension_duration_s=new_jump.trigger.nominal_extension_duration_s)
    receipt = target['jump_takeoff']
    if (old_jump.request != new_jump.request or old_jump.physics_dt_s != new_jump.physics_dt_s
            or new_jump.reference != expected_reference or not isinstance(receipt, dict)
            or any(receipt.get(k) != v for k, v in asdict(new_jump.trigger).items())
            or receipt.get('reference') != asdict(new_jump.reference)):
        raise ValueError('Target jump receipt/configuration does not bind the declared trigger and reference')
    stair = deepcopy(saved['stair_scenario'])
    if stair['task']['use_mode_labels'] is not False or source['stair_label'] is not None:
        raise ValueError('Task handoff expects the old unlabeled stair producer')
    stair['task']['use_mode_labels'] = True
    if (stair != reference['stair_source_config']
            or target['stair_label'] != {'stair_label':
                'STAIR set by the ToF stair state machine only; terrain-kind labels disabled'}
            or saved['recovery_scenario'] != reference['recovery_task_config']):
        raise ValueError('Task handoff cannot change ToF/recovery except the declared stair label')
    storage, recorded = dataset['replay']['storage'], reference['replay']['storage']
    if any(name in storage for name in JUMP_TRIGGER_RECORD_FIELDS):
        raise ValueError('Task handoff cannot overwrite existing trigger records')
    if old_jump.flight_latch or any(name in storage for name in JUMP_FLIGHT_LATCH_RECORD_FIELDS):
        raise ValueError('Task handoff expects legacy data without first-flight records')
    fields = RecordSpec(schema.goal_dim, storage['sim_context'].shape[-1]).fields(
        record_tasks=True, record_terrain=True, record_stairs=True, record_stair_source=True,
        record_jumps=True, record_recovery=True, record_jump_trigger=True,
        record_jump_flight_latch=new_jump.flight_latch)
    valid = recorded['valid'] & recorded['policy_owned']
    for name, shape in fields.items():
        if name not in recorded or recorded[name].shape != (*valid.shape, *shape):
            raise ValueError('Target dataset lacks complete native task/physical records: ' + name)
    for name in TASK_VALID_FLAGS:
        if name not in storage:
            raise ValueError('Source replay lacks complete task validity flags')
        valid &= recorded[name][..., 0] > .5
    new_dtypes = {**JUMP_TRIGGER_RECORD_DTYPES,
                  **(JUMP_FLIGHT_LATCH_RECORD_DTYPES if new_jump.flight_latch else {})}
    for name, dtype in new_dtypes.items():
        if recorded[name].dtype != dtype:
            raise ValueError('Native jump memory dtype differs: ' + name)
    valid &= recorded['jump_trigger_flags'][..., 0]
    clock = recorded['jump_trigger_clock'][..., 0]
    valid &= torch.isfinite(clock) | torch.isneginf(clock)
    if not valid.any():
        raise ValueError('Target dataset needs valid owned native task starts with known gate history')
    environment, course_receipt = _course(saved, source, target, source_course_raw, target_course_raw,
                                          course_coordinate, dataset['replay']['num_envs'], sampler_device)
    teacher_source = rebind_teacher_source(saved, source, target, source_teacher_path, target_teacher_path)
    base = Critic.feature_size(schema.goal_dim, storage['sim_context'].shape[-1], task_dim=MOTION_TASK_DIM,
                               terrain=True, stairs=True, jumps=True, recovery=True)
    if saved['learner']['critic']['net.0.weight'].shape[1] != base:
        raise ValueError('Source critic dimensions differ from the complete a05 task branch')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        critic = Critic(base + JUMP_TRIGGER_FEATURE_DIM, config.critic_hidden, task_dim=MOTION_TASK_DIM, terrain=True,
                        stairs=True, jumps=True, recovery=True, jump_trigger=True,
                        input_clip=config.learning.critic_input_clip)
    optimizer = torch.optim.Adam(critic.parameters(), lr=config.learning.critic_learning_rate)
    learner = {**saved['learner'], 'critic': critic.state_dict(),
        'optimizers': {**saved['learner']['optimizers'], 'critic': optimizer.state_dict()}}
    additions = {name: torch.zeros_like(storage[name]) for name in TASK_VALID_FLAGS}
    new_fields = {**JUMP_TRIGGER_RECORD_FIELDS,
                  **(JUMP_FLIGHT_LATCH_RECORD_FIELDS if new_jump.flight_latch else {})}
    for name, shape in new_fields.items():
        additions[name] = torch.zeros(*storage['frame'].shape[:2], *shape, device=storage['frame'].device,
                                      dtype=new_dtypes[name])
    if new_jump.flight_latch:
        # Historical physical rows remain useful supervision, but their
        # missing request memory cannot become an imagined starting state.
        # TASK_VALID_FLAGS above keep these placeholders out until overwritten
        # by a fresh native transition carrying its actual first-flight latch.
        additions['jump_flight_latch'][..., 0] = -1
    next_config = replace(config, course_coordinate=course_coordinate, critic_warmup=True,
                          policy_start_cycle=saved['cycle'] + critic_warmup_cycles)
    report = {'migration': 'a05-to-r7-task-semantics-v1', 'cycle': saved['cycle'],
        'source_schema': schema.fingerprint, 'target_schema': target_schema.fingerprint,
        'source_native_contract': json.loads(schema.source),
        'changed_source_contracts': [k for k in sorted(source.keys() | target.keys()) if source.get(k) != target.get(k)],
        'runtime_changes': ['terrain_policy.use_mode_labels'], 'course': course_receipt,
        'critic_initialization': 'new weights and Adam; all normalization columns calibrated from fresh native rows',
        'critic_input_before': base, 'critic_input_after': base + JUMP_TRIGGER_FEATURE_DIM,
        'critic_warmup_cycles': critic_warmup_cycles, 'policy_start_cycle': next_config.policy_start_cycle,
        'guard_reset': True, 'update_counters': 'lifetime counters retained; critic state reset at this cycle',
        'teacher_weights_identical': True, 'teacher_qualification': teacher_source['qualified_for'],
        'teacher_population_and_schedule_preserved': True,
        'legacy_task_valid_rows_invalidated': {name: int((storage[name] > .5).sum()) for name in TASK_VALID_FLAGS},
        'old_dynamics_rows_retained': int(storage['valid'].sum()), 'legacy_gate_histories_inferred': 0,
        'first_flight_records_added': new_jump.flight_latch, 'legacy_flight_histories_inferred': 0,
        'rejected_row_repair_added': old_isolation != new_isolation,
        'legacy_record_semantics': 'source_native_contract; physical data retained, all task starts invalid',
        'new_native_state': 'cold simulator/gate reset; collector offsets episode IDs; no capability promotion'}
    upgraded = {**saved, 'schema': target_schema.to_dict(), 'schema_fingerprint': target_schema.fingerprint,
        'config': asdict(next_config), 'learner': learner, 'environment': environment, 'teacher_source': teacher_source,
        'stair_scenario': reference['stair_source_config'], 'jump_scenario': asdict(new_jump),
        'recovery_scenario': reference['recovery_task_config'], 'guard': None, 'critic_norm_from': 0,
        'task_norm_pending': False, 'resume_migrations': [*saved.get('resume_migrations', []), report]}
    replay = {**dataset, 'schema_fingerprint': target_schema.fingerprint,
        'replay': {**dataset['replay'], 'storage': {**storage, **additions}}}
    if 'schema' in dataset:
        replay['schema'] = target_schema.to_dict()
    return upgraded, replay, report
