"""Authenticate the two reviewed settling changes without relabeling old data."""
from copy import deepcopy
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


FORMAT = 'settling-reward-continuity-v1'
SOURCE_KERNEL_SHA256 = '8e86f622f8b408f1784e10da88e7fbb661210e0af70df363294052836d8c7838'
# Combined stationary quiet and physical-tail kernel from the reviewed changes.
# Historical proofs bind this file, not the implementation running in the future.
TARGET_KERNEL_SHA256 = '7ab9e5481d13cfc029852aa0ee44c0a3cf4cdb5672b3b269e8878cff4257850e'


def verify_settling_reward_continuity(source, target):
    """Allow only quiet=1 and physical-tail=True on the known peak-aware kernel."""
    if not isinstance(source, dict) or not isinstance(target, dict):
        raise ValueError('Settling reward proof requires complete source objects')
    before, after = source.get('reward_kernel'), target.get('reward_kernel')
    if (not isinstance(before, dict) or not isinstance(after, dict)
            or before.get('implementation') != 'shared_physical_reward_densities'
            or before.get('kernel_sha256') != SOURCE_KERNEL_SHA256
            or len(TARGET_KERNEL_SHA256) != 64):
        raise ValueError('Settling reward requires the pinned source and reviewed target kernels')
    extensions, changed = before.get('extensions'), after.get('extensions')
    if (not isinstance(extensions, dict) or not isinstance(changed, dict)
            or extensions.get('jump_peak_completion') is not True
            or type(extensions.get('stationary_quiet', 0.)) not in (int, float)
            or extensions.get('stationary_quiet', 0.) != 0.
            or extensions.get('jump_physical_tail', False) is not False
            or type(changed.get('stationary_quiet')) not in (int, float)
            or changed.get('stationary_quiet') != 1.
            or changed.get('jump_physical_tail') is not True):
        raise ValueError('Settling reward permits only the fixed opt-ins on existing peak completion')
    expected = deepcopy(source)
    expected['reward_kernel'] = {
        **before, 'kernel_sha256': TARGET_KERNEL_SHA256,
        'extensions': {**extensions, 'stationary_quiet': 1., 'jump_physical_tail': True},
    }
    if target != expected:
        raise ValueError('Settling reward cannot change other weights or any physical/task/source identity')
    return {'format': FORMAT, 'before': deepcopy(source), 'after': deepcopy(target)}


def verify_saved_proof(proof):
    """Check the fixed historical kernel pair independently of the running file."""
    if not isinstance(proof, dict) or proof.get('format') != FORMAT:
        raise ValueError('Unknown settling reward proof')
    try:
        checked = verify_settling_reward_continuity(proof['before'], proof['after'])
    except (KeyError, TypeError) as error:
        raise ValueError('Incomplete settling reward proof') from error
    if proof != checked:
        raise ValueError('Settling reward proof contains unregistered fields or source changes')
    return checked


def upgrade_settling_reward_continuity(saved, dataset, target_schema):
    """Return a complete in-memory pair, preserving teacher, learners and labels.

    The native target reference and retained teacher/auxiliary restoration are
    checked by the handoff preflight. This function neither qualifies a teacher
    nor changes its source, writes a checkpoint, samples data or warms a learner.
    """
    current = hashlib.sha256(Path(inspect.getfile(physical_reward_densities)).read_bytes()).hexdigest()
    if current != TARGET_KERNEL_SHA256:
        raise ValueError('Executing reward kernel differs from the reviewed settling target')
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
        raise ValueError('Settling reward requires a complete pair and all four learning states')
    if 'schema' in dataset and StateSchema.from_dict(dataset['schema']) != old:
        raise ValueError('Replay schema differs from its recorded fingerprint')
    before, after = old.to_dict(), new.to_dict()
    source, target = (json.loads(fields.pop('source')) for fields in (before, after))
    if before != after:
        raise ValueError('Settling reward cannot change physical or observation tensor layouts')
    proof = verify_settling_reward_continuity(source, target)
    if not all(saved['config'].get(key, False) for key in
               ('record_tasks', 'record_stair_source', 'record_jumps', 'record_recovery')):
        raise ValueError('Settling reward requires the existing full recorded task producer')
    if (not JumpTaskConfig.from_dict(saved['jump_scenario']).flight_latch
            or any(saved.get(key) is None for key in ('stair_scenario', 'recovery_scenario'))):
        raise ValueError('Settling reward needs unchanged recorded request and first-flight memory')
    replay = dataset['replay']
    shapes = {**JUMP_RECORD_FIELDS, **JUMP_FLIGHT_LATCH_RECORD_FIELDS}
    dtypes = {**JUMP_RECORD_DTYPES, **JUMP_FLIGHT_LATCH_RECORD_DTYPES}
    for name, shape in shapes.items():
        value = replay['storage'].get(name)
        if (not isinstance(value, torch.Tensor)
                or value.shape != (len(replay['sequence']), replay['num_envs'], *shape)
                or value.dtype != dtypes.get(name, torch.float32)):
            raise ValueError('Settling reward requires the original recorded request field: ' + name)
    teacher = saved.get('teacher_source')
    if not isinstance(teacher, dict) or not {'native_source', 'qualification', 'qualified_for'} <= teacher.keys():
        raise ValueError('Settling reward requires the retained original teacher identity and qualification')
    report = {
        'migration': FORMAT, 'cycle': saved['cycle'], 'settling_reward_continuity': proof,
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
