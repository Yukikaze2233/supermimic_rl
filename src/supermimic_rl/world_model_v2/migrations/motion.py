"""Explicit foundation-to-command-task checkpoint migration, preserving learning.

Old physical transitions remain world-model data. Their unrecorded command
controller state is marked unavailable, never reconstructed from the goal.
Only newly captured task records can become task-aware imagination starts.
"""
from ...contracts import StateSchema
from .critic import append_critic_features
from ..imagination import Critic
from ..motion_task import MOTION_TASK_DIM
from ..records import MODEL_CONTRACT


def upgrade_motion_tasks(saved, dataset):
    if (saved.get('format') != 'supermimic-online-v2' or saved['model_contract'] != MODEL_CONTRACT
            or saved['config'].get('record_tasks', False)):
        raise ValueError('Migration requires a foundation checkpoint without motion-task records')
    schema = StateSchema.from_dict(saved['schema'])
    if (schema.fingerprint != saved['schema_fingerprint'] or dataset['schema_fingerprint'] != schema.fingerprint
            or dataset['cycle'] != saved['cycle']):
        raise ValueError('Migration checkpoint and replay generation differ')
    storage = dataset['replay']['storage']
    if 'motion_task' in storage or 'task_record_valid' in storage:
        raise ValueError('Foundation migration cannot reinterpret existing task records')
    base = Critic.feature_size(schema.goal_dim, storage['sim_context'].shape[-1])
    learner = append_critic_features(saved['learner'], expected_dim=base, extra_dim=MOTION_TASK_DIM)
    leading = storage['frame'].shape[:2]
    augmented = {**storage,
        'motion_task': storage['frame'].new_zeros(*leading, MOTION_TASK_DIM),
        'task_record_valid': storage['frame'].new_zeros(*leading, 1)}
    upgraded_replay = {**dataset, 'replay': {**dataset['replay'], 'storage': augmented}}
    receipt = {
        'migration': 'foundation-to-motion-tasks-v1', 'cycle': saved['cycle'],
        'critic_input_before': base, 'critic_input_after': base + MOTION_TASK_DIM,
        'preserved': ['world model', 'actor and estimator', 'old critic weights and moments',
                      'other optimizers', 'update counts', 'RNGs', 'physical replay and boundaries'],
        'old_task_records_valid': 0, 'old_physical_transitions_retained': leading[0] * leading[1],
        'task_norm': 'initialize only new feature columns from fresh native task records before policy learning',
        'scope': 'enables recording and learning known command trajectories; no course/skill promotion',
    }
    upgraded = {**saved, 'config': {**saved['config'], 'record_tasks': True},
        'learner': learner,
        'task_norm_pending': True,
        'resume_migrations': [*saved.get('resume_migrations', []), receipt]}
    return upgraded, upgraded_replay, receipt
