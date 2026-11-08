"""Preserve complete learning pairs across byte-authenticated reward changes."""
import hashlib
import inspect
import json
from pathlib import Path

import torch

from ...contracts import StateSchema
from ...reward_kernel import physical_reward_densities
from ..jump_task import (
    JUMP_FLIGHT_LATCH_RECORD_DTYPES, JUMP_FLIGHT_LATCH_RECORD_FIELDS,
    JUMP_RECORD_DTYPES, JUMP_RECORD_FIELDS, JumpTaskConfig,
)
from ..records import MODEL_CONTRACT


def _kernel_sha256():
    return hashlib.sha256(Path(inspect.getfile(physical_reward_densities)).read_bytes()).hexdigest()


def _upgrade_reward_pair(saved, dataset, target_schema, *, target_kernel, verify, proof_key):
    """Shared mechanics only; each caller authenticates one fixed source change.

    Native-reference parity and retained teacher/auxiliary restoration belong
    to the handoff preflight. No teacher qualification, historical label,
    optimizer, RNG, sampler or task state is changed here.
    """
    if _kernel_sha256() != target_kernel:
        raise ValueError('Executing reward kernel differs from the reviewed reward target')
    old = StateSchema.from_dict(saved['schema'])
    new = target_schema if isinstance(target_schema, StateSchema) else StateSchema.from_dict(target_schema)
    names = {'model', 'actor', 'critic', 'estimator'}
    learner = saved.get('learner', {})
    if (saved.get('format') != 'supermimic-online-v2' or saved.get('model_contract') != MODEL_CONTRACT
            or saved.get('schema_fingerprint') != old.fingerprint
            or dataset.get('schema_fingerprint') != old.fingerprint
            or type(saved.get('cycle')) is not int or saved['cycle'] < 0
            or dataset.get('cycle') != saved['cycle']
            or set(learner.get('optimizers', {})) != names
            or set(learner.get('updates', {})) != names
            or set(learner.get('skipped', {})) != names):
        raise ValueError('Reward continuity requires a complete pair and all four learning states')
    if 'schema' in dataset and StateSchema.from_dict(dataset['schema']) != old:
        raise ValueError('Replay schema differs from its recorded fingerprint')
    before, after = old.to_dict(), new.to_dict()
    source, target = (json.loads(fields.pop('source')) for fields in (before, after))
    if before != after:
        raise ValueError('Reward continuity cannot change physical or observation tensor layouts')
    proof = verify(source, target)
    if not all(saved['config'].get(key, False) for key in
               ('record_tasks', 'record_stair_source', 'record_jumps', 'record_recovery')):
        raise ValueError('Reward continuity requires the existing full recorded task producer')
    if (not JumpTaskConfig.from_dict(saved['jump_scenario']).flight_latch
            or any(saved.get(key) is None for key in ('stair_scenario', 'recovery_scenario'))):
        raise ValueError('Reward continuity needs unchanged recorded request and first-flight memory')
    replay = dataset['replay']
    shapes = {**JUMP_RECORD_FIELDS, **JUMP_FLIGHT_LATCH_RECORD_FIELDS}
    dtypes = {**JUMP_RECORD_DTYPES, **JUMP_FLIGHT_LATCH_RECORD_DTYPES}
    for name, shape in shapes.items():
        value = replay['storage'].get(name)
        if (not isinstance(value, torch.Tensor)
                or value.shape != (len(replay['sequence']), replay['num_envs'], *shape)
                or value.dtype != dtypes.get(name, torch.float32)):
            raise ValueError('Reward continuity requires the original recorded request field: ' + name)
    teacher = saved.get('teacher_source')
    if not isinstance(teacher, dict) or not {'native_source', 'qualification', 'qualified_for'} <= teacher.keys():
        raise ValueError('Reward continuity requires the retained original teacher identity and qualification')
    report = {
        'migration': proof['format'], 'cycle': saved['cycle'], proof_key: proof,
        'source_schema': old.fingerprint, 'target_schema': new.fingerprint,
        'legacy_reward_sequence_before': replay['total_steps'],
        'legacy_reward_labels': 'unchanged native measurements; never actor/critic targets',
        'historical_physical_task_records_and_validity': 'unchanged',
        'learners_optimizers_normalizers_and_rng': 'all preserved',
        'curriculum_sampler_quotas_and_clocks': 'all preserved',
        'teacher_source_and_qualification': 'original artifact retained, no new capability claim',
        'auxiliary_sources_splits_counts_and_rng': 'all preserved',
        'additional_warmup_cycles': 0,
        'pending_integration': ['native reward reference parity', 'retained teacher source-lineage check',
                                'auxiliary source-lineage and restoration check'],
    }
    upgraded = {**saved, 'schema': new.to_dict(), 'schema_fingerprint': new.fingerprint,
                'resume_migrations': [*saved.get('resume_migrations', ()), report]}
    migrated = {**dataset, 'schema_fingerprint': new.fingerprint}
    if 'schema' in migrated:
        migrated['schema'] = new.to_dict()
    return upgraded, migrated, report
