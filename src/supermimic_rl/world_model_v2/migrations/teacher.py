"""Validate unchanged teacher weights against a new native evaluation."""
import json
from pathlib import Path

import torch

from ...chassis_observation import LEGACY_COMMANDS
from ..teacher import PPO_TEACHER_FORMAT, file_sha256


def _source(value):
    result = json.loads(value)
    for name in ('jump_takeoff', 'stair_label', 'jump_assist'):
        result.setdefault(name, None)
    return result


def rebind_teacher_source(saved, source, target, source_path, target_path):
    """Same evaluated PPO weights, with the new artifact's own qualification scope."""
    before = saved['teacher_source']
    if file_sha256(source_path) != before['teacher_artifact']['sha256']:
        raise ValueError('Source teacher artifact differs from the saved checkpoint')
    old, new = [torch.load(path, map_location='cpu', weights_only=True, mmap=True)
                for path in (source_path, target_path)]
    # Sampling-only handoffs deliberately retained the original teacher and
    # evaluation. Authenticate that lineage before rebinding its unchanged
    # weights to a newly evaluated reward/task source.
    course_migrations = tuple(entry['course_sampling'] for entry in saved.get('resume_migrations', ())
                              if isinstance(entry, dict) and 'course_sampling' in entry)
    original_source = source
    if course_migrations:
        from .course import compatible_course_source

        original_source = compatible_course_source(old['source']['native_source'], source, course_migrations)
    for payload, native in ((old, original_source), (new, target)):
        metadata = payload['source']
        report = metadata['qualification']['report']
        if (payload.get('format') != PPO_TEACHER_FORMAT or metadata.get('format') != PPO_TEACHER_FORMAT
                or not metadata.get('qualified_for')
                or _source(json.dumps(metadata['native_source'])) != native
                or _source(json.dumps(report['native_source'])) != native
                or report.get('checkpoint_sha256') != metadata['initial_checkpoint']['sha256']
                or report.get('transferred_source') is not None
                or report.get('steps', 0) <= 0 or report.get('simulated_seconds', 0) <= 0):
            raise ValueError('Teacher needs its own native evaluation on the pinned task source')
    if old['source'] != {k: v for k, v in before.items() if k != 'teacher_artifact'}:
        raise ValueError('Source teacher metadata differs from the online checkpoint')
    # Early exports described the raw slice before its behavior acquired a
    # versioned name. Accept that wording change only for the identical legacy
    # observation projection; the actual contract and feature transform still pin it.
    histories = {payload['source']['history_semantics'] for payload in (old, new)}
    legacy_histories = {
        'native obs_buf policy [N,5,42], oldest first, first 35 channels; native reset filling',
        'native obs_buf policy [N,5,42], oldest first; versioned chassis_frame preprocessing; native reset filling',
    }
    same_history = (len(histories) == 1 or histories == legacy_histories
                    and all(payload['source'].get('chassis_command_contract', LEGACY_COMMANDS) == LEGACY_COMMANDS
                            for payload in (old, new)))
    if (any(old['source']['initial_checkpoint'][key] != new['source']['initial_checkpoint'][key]
            for key in ('sha256', 'updates', 'transitions'))
            or any(old['source'][key] != new['source'][key] for key in
                   ('history_length', 'observation_dim', 'action_bounds', 'action_semantics',
                    'estimate_semantics'))
            or not same_history
            or old['source'].get('chassis_command_contract', LEGACY_COMMANDS)
            != new['source'].get('chassis_command_contract', LEGACY_COMMANDS)
            or old['source'].get('actor_features') != new['source'].get('actor_features')
            or any(old.get(k) != new.get(k) for k in ('actor_hidden', 'estimator_hidden', 'mean_squash'))
            or old['actor'].keys() != new['actor'].keys()
            or any(not torch.equal(value, new['actor'][name]) for name, value in old['actor'].items())):
        raise ValueError('Task handoff must retain exactly the same teacher weights')
    return {**new['source'], 'teacher_artifact': {
        'path': str(Path(target_path).resolve()), 'sha256': file_sha256(target_path)}}

