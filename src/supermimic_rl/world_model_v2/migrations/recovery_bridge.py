"""Authenticate side-roll birth sampling without rewriting historical dynamics."""
from copy import deepcopy
import hashlib
import json

import torch

from ...contracts import StateSchema
from ...course.jump_first import SCHEDULE, make_course_curriculum
from ...course.recovery_bridge import RECOVERY_VARIANTS, receipt as bridge_receipt
from ..course_progression import CourseProgressionConfig
from ..records import MODEL_CONTRACT


FORMAT = 'recovery-roll-bridge-sampling-v1'
IDENTITY_KEYS = ('course_sha256', 'curriculum', 'recovery_roll_bridge')


def verify_recovery_bridge_sampling(source, target, *, source_course, target_course):
    """Certify the one-way opt-in against both original course byte strings."""
    courses = []
    for provenance, raw in ((source, source_course), (target, target_course)):
        if not isinstance(raw, bytes):
            raise ValueError('Recovery bridge migration requires the original course bytes')
        course = json.loads(raw)
        if (hashlib.sha256(raw).hexdigest() != provenance.get('course_sha256')
                or course != provenance.get('curriculum')
                or set(course) != {'schedule', 'base'} or course['schedule'] != SCHEDULE):
            raise ValueError('Recovery bridge course bytes, source pin or schedule differ')
        make_course_curriculum(1, 'cpu', 0, course)
        courses.append(deepcopy(course))
    before, after = courses
    if (before['base'].pop('recovery_roll_bridge', False) is not False
            or after['base'].pop('recovery_roll_bridge', False) is not True
            or before != after):
        raise ValueError('Recovery bridge permits only the course switch false to true')
    if ('recovery_roll_bridge' in source
            or target.get('recovery_roll_bridge') != bridge_receipt()):
        raise ValueError('Recovery bridge requires its exact new birth receipt')
    if ({key: value for key, value in source.items() if key not in IDENTITY_KEYS}
            != {key: value for key, value in target.items() if key not in IDENTITY_KEYS}):
        raise ValueError('Recovery bridge cannot change physics, rewards, observers or task producers')
    return {
        'format': FORMAT, 'before': deepcopy(source), 'after': deepcopy(target),
        'source_course_utf8': source_course.decode('utf-8'),
        'target_course_utf8': target_course.decode('utf-8'),
    }


def verify_saved_proof(proof):
    """Reconstruct an exact proof for chronological source-lineage consumers."""
    if not isinstance(proof, dict) or proof.get('format') != FORMAT:
        raise ValueError('Unknown recovery bridge sampling proof')
    try:
        checked = verify_recovery_bridge_sampling(
            proof['before'], proof['after'],
            source_course=proof['source_course_utf8'].encode('utf-8'),
            target_course=proof['target_course_utf8'].encode('utf-8'))
    except (KeyError, AttributeError, TypeError) as error:
        raise ValueError('Incomplete recovery bridge sampling proof') from error
    if checked != proof:
        raise ValueError('Recovery bridge proof differs from its authenticated pair')
    return checked


def upgrade_recovery_roll_bridge(saved, dataset, target_schema, *, source_course,
                                 target_course, sampler_device='cpu'):
    """Preserve a complete pair; append six zero birth labels to the old ledger.

    Source compatibility for retained teacher/auxiliary artifacts is checked by
    the source-lineage consumers before launch. Their original metadata and
    qualification scopes are preserved here, never re-labelled as new evidence.
    """
    old = StateSchema.from_dict(saved['schema'])
    new = target_schema if isinstance(target_schema, StateSchema) else StateSchema.from_dict(target_schema)
    learner = saved.get('learner', {})
    optimizer_names = {'model', 'actor', 'critic', 'estimator'}
    if (saved.get('format') != 'supermimic-online-v2' or saved.get('model_contract') != MODEL_CONTRACT
            or saved.get('schema_fingerprint') != old.fingerprint
            or dataset.get('schema_fingerprint') != old.fingerprint
            or dataset.get('cycle') != saved.get('cycle')
            or not {'model', 'actor', 'critic', 'updates', 'skipped', 'optimizers'} <= learner.keys()
            or set(learner['optimizers']) != optimizer_names
            or set(learner['updates']) != optimizer_names or set(learner['skipped']) != optimizer_names):
        raise ValueError('Recovery bridge requires a complete paired checkpoint and all four optimizers')
    if 'schema' in dataset and StateSchema.from_dict(dataset['schema']) != old:
        raise ValueError('Recovery bridge replay schema differs from its checkpoint')
    before, after = old.to_dict(), new.to_dict()
    source, target = (json.loads(value.pop('source')) for value in (before, after))
    if before != after:
        raise ValueError('Recovery bridge cannot change the physical or observation tensor contract')
    proof = verify_recovery_bridge_sampling(source, target, source_course=source_course,
                                           target_course=target_course)
    config = saved['config']
    if not all(config.get(name, False) for name in
               ('record_tasks', 'record_stair_source', 'record_jumps', 'record_recovery')):
        raise ValueError('Recovery bridge requires the existing full recorded task producer')
    if any(saved.get(name) is None for name in ('stair_scenario', 'jump_scenario', 'recovery_scenario')):
        raise ValueError('Recovery bridge requires the unchanged recorded task signatures')
    teacher = saved.get('teacher_source')
    if not isinstance(teacher, dict) or not {'native_source', 'qualification', 'qualified_for'} <= teacher.keys():
        raise ValueError('Recovery bridge requires the retained teacher source and qualification')
    environment, device = saved['environment'], torch.device(sampler_device)
    current = environment['curriculum']
    coordinate = current['real_transitions']
    progression = config.get('course_progression')
    if progression is not None:
        if coordinate != CourseProgressionConfig(**progression).coordinate(max(0, saved['collection_steps'] - 1)):
            raise ValueError('Recovery bridge course coordinate differs from its collection-clock mapping')
    elif coordinate != config.get('course_coordinate'):
        raise ValueError('Recovery bridge course coordinate differs from its fixed placement')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA sampler state must be migrated on an available CUDA device')
    count = dataset['replay']['num_envs']
    original = make_course_curriculum(count, device, 0, source['curriculum'])
    expected_rng, actual_rng = original.sampler.generator.get_state(), current['sampler']['rng']
    if expected_rng.shape != actual_rng.shape or expected_rng.dtype != actual_rng.dtype:
        raise ValueError('Recovery bridge must preserve the original sampler RNG device')
    original.load_state_dict(current)
    replacement = make_course_curriculum(count, device, 0, target['curriculum'])
    replacement.set_progress(coordinate, current['actor_updates'])
    state = replacement.state_dict()
    counts = deepcopy(current['sample_counts'])
    if len(counts['recovery_posture']) != 16 or len(state['sample_counts']['recovery_posture']) != 22:
        raise ValueError('Recovery bridge requires the original 16 labels and six new variants')
    counts['recovery_posture'].extend([0] * 6)
    state['sample_counts'] = counts
    # Mixtures are identical, so retain both quota debts and the RNG stream.
    state['sampler'] = current['sampler']
    replacement.load_state_dict(state)
    report = {
        'migration': FORMAT, 'cycle': saved['cycle'], 'recovery_roll_bridge_sampling': proof,
        'source_schema': old.fingerprint, 'target_schema': new.fingerprint,
        'course_coordinate': coordinate, 'sampler_device': str(device),
        'birth_label_names': list(RECOVERY_VARIANTS), 'appended_birth_counts': [0] * 6,
        'legacy_posture_counts_preserved': True, 'family_ordinal_preserved': True,
        'sampler_rng_and_quota_preserved': True, 'learner_optimizers_and_rng_preserved': True,
        'historical_truth_and_start_validity_preserved': True,
        'teacher_qualification_scope_unchanged': True, 'additional_warmup_cycles': 0,
        'target_schema_origin': 'authenticated birth-sampling configuration; no new native evaluation',
        'pending_integration': ['retained teacher source-lineage check', 'auxiliary source-lineage check',
                                'native birth readback before formal launch'],
        'scope': 'Future recovery births only. Old 16 labels are retained and six zero counters appended; '
                 'no historical births are reclassified, and no physical/task/reward labels are rewritten.',
    }
    upgraded = {**saved, 'schema': new.to_dict(), 'schema_fingerprint': new.fingerprint,
                'environment': {**environment, 'curriculum': state},
                'resume_migrations': [*saved.get('resume_migrations', ()), report]}
    replay = {**dataset, 'schema_fingerprint': new.fingerprint}
    if 'schema' in replay:
        replay['schema'] = new.to_dict()
    return upgraded, replay, report
