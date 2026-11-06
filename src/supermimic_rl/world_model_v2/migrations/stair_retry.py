"""Authenticate future ToF timeout retries without relabeling historical data."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import torch

from ...contracts import StateSchema
from ...course.stair_retry import timeout_retry_receipt
from ..records import MODEL_CONTRACT
from ..stair_source import StairScenario


FORMAT = 'stair-timeout-retry-v1'


def _json_identity(value):
    """Stable JSON identity also rejects non-finite or non-serializable proof data."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def verify_stair_timeout_retry(source, target, source_scenario, target_scenario, *,
                              runtime_raw, course_raw):
    """Authorize only the fixed False -> True source policy, with original pins.

    Complete source/scenario identities and their unchanged input bytes are
    retained. This is a configuration proof, not a new native evaluation or
    permission to reinterpret old one-shot source states as retry trajectories.
    """
    if not isinstance(source, dict) or not isinstance(target, dict):
        raise ValueError('ToF retry migration requires complete source dictionaries')
    if 'stair_timeout_retry' in source:
        raise ValueError('ToF retry migration only enables the original disabled policy')
    expected = {**source, 'stair_timeout_retry': timeout_retry_receipt()}
    if _json_identity(target) != _json_identity(expected):
        raise ValueError('Only the fixed ToF timeout retry receipt may change in the source')
    for name, raw in (('runtime', runtime_raw), ('course', course_raw)):
        if (not isinstance(raw, bytes)
                or hashlib.sha256(raw).hexdigest() != source.get(name + '_sha256')):
            raise ValueError('ToF retry migration requires unchanged pinned ' + name + ' bytes')
        if not isinstance(json.loads(raw), dict):
            raise ValueError('ToF retry migration requires an original JSON ' + name + ' object')
    if _json_identity(json.loads(course_raw)) != _json_identity(source.get('curriculum')):
        raise ValueError('Original course bytes differ from the complete source curriculum')
    try:
        before = StairScenario.from_dict(source_scenario)
        after = StairScenario.from_dict(target_scenario)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('ToF retry migration requires complete recorded stair scenarios') from error
    if (before.timeout_retry or not after.timeout_retry
            or _json_identity(source_scenario) != _json_identity(before.to_dict())
            or _json_identity(target_scenario) != _json_identity(replace(before, timeout_retry=True).to_dict())):
        raise ValueError('Only the stair scenario format and disabled-to-enabled retry flag may change')
    return json.loads(_json_identity({
        'format': FORMAT, 'before': source, 'after': target,
        'source_scenario': source_scenario, 'target_scenario': target_scenario,
        'runtime_utf8': runtime_raw.decode('utf-8'), 'course_utf8': course_raw.decode('utf-8'),
    }))


def verify_saved_proof(proof):
    """Revalidate one exact ledger proof before source-lineage compatibility."""
    if not isinstance(proof, dict) or proof.get('format') != FORMAT:
        raise ValueError('Unknown ToF timeout retry migration proof')
    try:
        checked = verify_stair_timeout_retry(
            proof['before'], proof['after'], proof['source_scenario'], proof['target_scenario'],
            runtime_raw=proof['runtime_utf8'].encode('utf-8'), course_raw=proof['course_utf8'].encode('utf-8'))
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('Incomplete ToF timeout retry migration proof') from error
    if _json_identity(proof) != _json_identity(checked):
        raise ValueError('ToF timeout retry proof differs from its authenticated identities')
    return checked


def upgrade_stair_timeout_retry(saved, dataset, target_schema, *, target_scenario, runtime_raw, course_raw):
    """Derive a full recovery pair; invalidate only old imagined-source starts.

    Teacher qualification, auxiliary sources and all training state are kept
    unchanged. Their loaders must separately authenticate this source lineage
    before the pair can launch. No replay-wide copy or native evaluation runs.
    """
    old = StateSchema.from_dict(saved['schema'])
    new = target_schema if isinstance(target_schema, StateSchema) else StateSchema.from_dict(target_schema)
    if (saved.get('format') != 'supermimic-online-v2' or saved.get('model_contract') != MODEL_CONTRACT
            or saved.get('schema_fingerprint') != old.fingerprint
            or dataset.get('schema_fingerprint') != old.fingerprint
            or type(saved.get('cycle')) is not int or saved['cycle'] < 0
            or dataset.get('cycle') != saved['cycle']
            or set(saved.get('learner', {}).get('optimizers', {})) != {'actor', 'critic', 'model', 'estimator'}):
        raise ValueError('ToF retry migration requires a complete paired online checkpoint')
    if 'schema' in dataset and StateSchema.from_dict(dataset['schema']) != old:
        raise ValueError('Replay schema differs from its recorded fingerprint')
    if not all(saved['config'].get(name, False) for name in ('record_tasks', 'record_stair_source')):
        raise ValueError('ToF retry migration requires the existing recorded stair source')
    before, after = old.to_dict(), new.to_dict()
    source, target = [json.loads(fields.pop('source')) for fields in (before, after)]
    if before != after:
        raise ValueError('ToF retry migration cannot change physical or observation tensor contracts')
    proof = verify_stair_timeout_retry(source, target, saved.get('stair_scenario'), target_scenario,
                                      runtime_raw=runtime_raw, course_raw=course_raw)
    replay = dataset['replay']
    storage = replay['storage']
    valid = storage.get('stair_source_valid')
    if (not isinstance(valid, torch.Tensor) or valid.ndim != 3
            or valid.shape[1:] != (replay['num_envs'], 1) or valid.dtype != torch.float32
            or valid.shape[0] != len(replay['sequence'])):
        raise ValueError('ToF retry migration requires recorded source-start validity')
    report = {
        'migration': FORMAT, 'cycle': saved['cycle'], 'stair_timeout_retry': proof,
        'source_schema': old.fingerprint, 'target_schema': new.fingerprint,
        'legacy_source_sequence_before': replay['total_steps'],
        'legacy_stair_starts_excluded': int((valid > .5).sum()),
        'historical_labels': 'physical, actions, rewards, request IDs/parameters/clocks and ToF flags unchanged',
        'learner_optimizers_rng_curriculum_and_quotas': 'all preserved',
        'teacher_source_preserved': True, 'teacher_evaluation_scope_unchanged': True,
        'auxiliary_sources_splits_counts_and_rng': 'all preserved',
        'target_schema_origin': 'authenticated future-source policy; no new native evaluation',
        'additional_warmup_cycles': 0,
        'pending_integration': ['teacher source compatibility', 'auxiliary dataset source compatibility'],
    }
    updated = {**saved, 'schema': new.to_dict(), 'schema_fingerprint': new.fingerprint,
               'stair_scenario': deepcopy(target_scenario),
               'resume_migrations': [*saved.get('resume_migrations', ()), report]}
    migrated = {**dataset, 'schema_fingerprint': new.fingerprint,
                'replay': {**replay, 'storage': {**storage, 'stair_source_valid': torch.zeros_like(valid)}}}
    if 'schema' in migrated:
        migrated['schema'] = new.to_dict()
    return updated, migrated, report
