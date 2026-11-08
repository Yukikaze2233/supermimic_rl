"""Change future course sampling without rewriting recorded task conditions."""
from copy import deepcopy
import hashlib
import json

import torch

from ...contracts import StateSchema
from ...course.jump_first import SCHEDULE, make_course_curriculum
from ..course_progression import CourseProgressionConfig
from ..records import MODEL_CONTRACT


FORMAT = 'course-sampling-only-v1'
IDENTITY_KEYS = ('course_sha256', 'curriculum', 'course_schedule')
SAMPLING_FIELDS = ('stage_weights', 'jump_height_gates', 'recovery_posture_stages')


def _identity(source):
    return {key: deepcopy(source[key]) for key in IDENTITY_KEYS}


def verify_course_sampling(source, target, *, source_course, target_course):
    """Authenticate an explicit pair; reject changes outside future sampling.

    Raw JSON is retained in the receipt so later compatibility checks can
    authenticate the original byte pins, rather than accepting arbitrary hash
    substitutions. No native evaluation or new capability is inferred here.
    """
    courses = []
    for provenance, raw in ((source, source_course), (target, target_course)):
        if not isinstance(raw, bytes):
            raise ValueError('Sampling migration requires the original course bytes')
        course = json.loads(raw)
        if (hashlib.sha256(raw).hexdigest() != provenance.get('course_sha256')
                or course != provenance.get('curriculum')
                or set(course) != {'schedule', 'base'} or course['schedule'] != SCHEDULE):
            raise ValueError('Course bytes, source pin or jump-first configuration differ')
        # Parse the public configuration, including domains, weights and stages.
        make_course_curriculum(1, 'cpu', 0, course)
        schedule = provenance.get('course_schedule')
        if (not isinstance(schedule, dict) or schedule.get('schedule') != SCHEDULE
                or schedule.get('early_jump_domain') != course['base']['initial_domain']['jump_m']):
            raise ValueError('Course schedule receipt does not match its initial jump domain')
        courses.append(course)
    if ({key: value for key, value in source.items() if key not in IDENTITY_KEYS}
            != {key: value for key, value in target.items() if key not in IDENTITY_KEYS}):
        raise ValueError('Sampling migration cannot change physics, observations, rewards or task producers')
    before, after = [deepcopy(course['base']) for course in courses]
    for name in SAMPLING_FIELDS:
        before.pop(name, None)
        after.pop(name, None)
    before['initial_domain'].pop('jump_m')
    after['initial_domain'].pop('jump_m')
    if before != after:
        raise ValueError('Only task mixtures, initial jump range and sampling ladders may change')
    old_schedule, new_schedule = [deepcopy(p['course_schedule']) for p in (source, target)]
    old_schedule.pop('early_jump_domain')
    new_schedule.pop('early_jump_domain')
    if old_schedule != new_schedule:
        raise ValueError('Sampling migration cannot change command-pool semantics')
    # Gate records contain evidence indexed by stage/fraction. Moving these
    # keys needs a separate, deliberate evidence migration, not a pin alias.
    old_keys = [value[0] for value in courses[0]['base'].get('jump_height_gates', ())]
    new_keys = [value[0] for value in courses[1]['base'].get('jump_height_gates', ())]
    if old_keys != new_keys:
        raise ValueError('Changed jump ladder keys require a separate gate-evidence migration')
    return {
        'format': FORMAT, 'before': _identity(source), 'after': _identity(target),
        'source_course_utf8': source_course.decode('utf-8'),
        'target_course_utf8': target_course.decode('utf-8'),
    }


def compatible_course_source(recorded, live, migrations):
    """Undo registered sampling identities, leaving unrelated changes visible.

    Callers must still compare the result with the recorded source, or pass it
    through their explicit terrain/disturbance compatibility checks. This
    function does not certify the whole source and never edits either input.
    """
    target = deepcopy(live)
    for migration in reversed(migrations):
        if all(recorded.get(key) == target.get(key) for key in IDENTITY_KEYS):
            break
        if (migration.get('format') != FORMAT
                or any(target.get(key) != migration.get('after', {}).get(key) for key in IDENTITY_KEYS)):
            continue
        proof = verify_course_sampling(
            migration['before'], migration['after'],
            source_course=migration['source_course_utf8'].encode('utf-8'),
            target_course=migration['target_course_utf8'].encode('utf-8'))
        if proof != migration:
            raise ValueError('Sampling compatibility receipt differs from its authenticated pair')
        target.update(deepcopy(migration['before']))
    return target


def upgrade_course_sampling(saved, dataset, target_schema, *, source_course, target_course,
                            sampler_device='cpu'):
    """Derive a complete recovery pair; all physical/task labels remain intact.

    ``target_schema`` is an explicit new configuration identity, not a claimed
    native evaluation. The caller must install compatibility for the retained
    teacher and auxiliary datasets before launching this candidate.
    """
    old = StateSchema.from_dict(saved['schema'])
    new = target_schema if isinstance(target_schema, StateSchema) else StateSchema.from_dict(target_schema)
    if (saved.get('format') != 'supermimic-online-v2' or saved.get('model_contract') != MODEL_CONTRACT
            or saved.get('schema_fingerprint') != old.fingerprint
            or dataset.get('schema_fingerprint') != old.fingerprint
            or dataset.get('cycle') != saved.get('cycle')
            or 'learner' not in saved or 'optimizers' not in saved['learner']):
        raise ValueError('Sampling migration requires a complete paired online checkpoint')
    if 'schema' in dataset and StateSchema.from_dict(dataset['schema']) != old:
        raise ValueError('Replay schema differs from its recorded fingerprint')
    before, after = old.to_dict(), new.to_dict()
    source, target = json.loads(before.pop('source')), json.loads(after.pop('source'))
    if before != after:
        raise ValueError('Sampling migration cannot change the observation or physical tensor contract')
    proof = verify_course_sampling(source, target, source_course=source_course, target_course=target_course)
    config = saved['config']
    if not all(config.get(name, False) for name in
               ('record_tasks', 'record_stair_source', 'record_jumps', 'record_recovery')):
        raise ValueError('Sampling migration requires the existing full recorded task producer')
    if any(saved.get(name) is None for name in ('stair_scenario', 'jump_scenario', 'recovery_scenario')):
        raise ValueError('Sampling migration requires recorded task configuration signatures')
    environment = saved['environment']
    current = environment['curriculum']
    coordinate = current['real_transitions']
    progression = config.get('course_progression')
    if progression is not None:
        clock = CourseProgressionConfig(**progression)
        if coordinate != clock.coordinate(max(0, saved['collection_steps'] - 1)):
            raise ValueError('Saved sampling coordinate differs from its collection-clock mapping')
    elif coordinate != config.get('course_coordinate'):
        raise ValueError('Saved sampling coordinate differs from its fixed course placement')
    device = torch.device(sampler_device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA sampler state must be migrated on an available CUDA device')
    count = dataset['replay']['num_envs']
    original = make_course_curriculum(count, device, 0, source['curriculum'])
    expected_rng, actual_rng = original.sampler.generator.get_state(), current['sampler']['rng']
    if expected_rng.shape != actual_rng.shape or expected_rng.dtype != actual_rng.dtype:
        raise ValueError('Sampler migration must preserve its original RNG device')
    original.load_state_dict(current)
    replacement = make_course_curriculum(count, device, 0, target['curriculum'])
    replacement.set_progress(coordinate, current['actor_updates'])
    state = replacement.state_dict()
    mixture_changed = original.stage_weights != replacement.stage_weights
    # Changed mixtures have new fractional quota debts. Height/posture-only
    # changes can retain those debts as well as the complete random stream.
    if mixture_changed:
        state['sampler']['rng'] = current['sampler']['rng']
    else:
        state['sampler'] = current['sampler']
    state['sample_counts'] = deepcopy(current['sample_counts'])
    replacement.load_state_dict(state)
    previous = tuple(m['course_sampling'] for m in saved.get('resume_migrations', ())
                     if isinstance(m, dict) and 'course_sampling' in m)
    teacher_source = saved.get('teacher_source')
    teacher = teacher_source.get('native_source') if isinstance(teacher_source, dict) else None
    if not isinstance(teacher, dict) or compatible_course_source(teacher, source, previous) != teacher:
        raise ValueError('Retained teacher source needs an explicit compatible sampling lineage')
    report = {
        'migration': FORMAT, 'cycle': saved['cycle'], 'course_sampling': proof,
        'source_schema': old.fingerprint, 'target_schema': new.fingerprint,
        'course_coordinate': coordinate, 'target_stage': replacement.stage_names[replacement.stage],
        'sampler_device': str(device), 'sampler_credits_reset': mixture_changed,
        'sampling_counts_preserved': True, 'sampler_rng_preserved': True,
        'historical_labels_preserved': True, 'teacher_source_preserved': True,
        'teacher_evaluation_scope_unchanged': True,
        'target_schema_origin': 'authenticated future-sampling configuration; no new native evaluation',
        'pending_integration': ['teacher source compatibility', 'auxiliary dataset source compatibility'],
        'scope': 'Future task sampling only; full learner, task signatures, replay truth, '
                 'collection/course clocks and auxiliary splits/RNG retained. No capability promotion.',
    }
    upgraded = {**saved, 'schema': new.to_dict(), 'schema_fingerprint': new.fingerprint,
                'environment': {**environment, 'curriculum': state},
                'resume_migrations': [*saved.get('resume_migrations', ()), report]}
    replay = {**dataset, 'schema_fingerprint': new.fingerprint}
    if 'schema' in replay:
        replay['schema'] = new.to_dict()
    return upgraded, replay, report
