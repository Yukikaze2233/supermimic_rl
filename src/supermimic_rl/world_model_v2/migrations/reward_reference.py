"""Keep learned dynamics and task memory across explicit analytic reward changes."""
from dataclasses import asdict
import inspect
import json

import torch

from ...contracts import StateSchema
from ...reward_kernel import RewardExtensions, physical_reward_densities
from ..jump_task import JumpTaskConfig
from ..online import CHECKPOINT_FORMAT
from ..records import MODEL_CONTRACT
from ..teacher import file_sha256
from .teacher import rebind_teacher_source


# The first-flight HOLD fix, before commit 4467228 corrected premature LANDING targets.
SOURCE_KERNEL_SHA256 = '9ebd10778d84c4aa7ab987648ee283e9dcde4154f1ba60646c36807b61b41902'
TARGET_REFERENCE = 'first_real_flight_landing_v1'
# Shared first-real-flight kernel before the opt-in recovery settling term.
RECOVERY_SOURCE_KERNEL_SHA256 = '10ac1a471ef34100ba2828d871a6f08cf7cd8992908b57d6f02e757d55bd52ce'
# Recovery settling kernel, before the ordinary acceleration coefficient became configurable.
ACCELERATION_SOURCE_KERNEL_SHA256 = '35d7e113f8cda0fa65289fc9b2b6e7c0ba1271b93b4f8c710dbace1fe384a51a'


def _upgrade_reward(saved, dataset, reference, *, source_teacher_path, target_teacher_path, change):
    """Change only the analytic target contract; retain all tensors and learning state.

    Stored rewards remain historical measurements. Model fitting consumes physical
    tick labels; actor/critic targets are freshly calculated in imagination. The
    sequence boundary identifies the old reward labels without adding a field to
    every transition or pretending that the old native producer used the fix.
    """
    schema, target_schema = (StateSchema.from_dict(item['schema']) for item in (saved, reference))
    if (saved.get('format') != CHECKPOINT_FORMAT or saved.get('model_contract') != MODEL_CONTRACT
            or reference.get('format') != 'supermimic-world-v2-dataset-v1'
            or reference.get('model_contract') != MODEL_CONTRACT
            or saved['schema_fingerprint'] != schema.fingerprint
            or reference['schema_fingerprint'] != target_schema.fingerprint
            or dataset['schema_fingerprint'] != schema.fingerprint
            or dataset['cycle'] != saved['cycle']):
        raise ValueError('Reward-reference migration requires a paired checkpoint and native reference')
    if 'schema' in dataset and StateSchema.from_dict(dataset['schema']).fingerprint != schema.fingerprint:
        raise ValueError('Replay schema differs from its checkpoint')
    old_fields, new_fields = schema.to_dict(), target_schema.to_dict()
    source, target = (json.loads(fields.pop('source')) for fields in (old_fields, new_fields))
    if (old_fields != new_fields
            or {k: v for k, v in source.items() if k != 'reward_kernel'}
            != {k: v for k, v in target.items() if k != 'reward_kernel'}):
        raise ValueError('Reward-reference migration cannot change physical, observation or task contracts')
    before, after = source['reward_kernel'], target['reward_kernel']
    expected = {**before, 'kernel_sha256': file_sha256(inspect.getfile(physical_reward_densities)),
                'extensions': asdict(RewardExtensions(**before['extensions']))}
    if change == 'jump-reward-reference-v1':
        expected['jump_tracking_reference'] = TARGET_REFERENCE
        if (before.get('kernel_sha256') != SOURCE_KERNEL_SHA256
                or 'jump_tracking_reference' in before
                or not RewardExtensions(**before['extensions']).needs_jump_progress):
            raise ValueError('Only the known premature-landing reference correction is supported')
    elif change == 'recovery-quiet-reward-v1':
        expected['extensions']['recovery_quiet'] = 1.
        if (before.get('kernel_sha256') != RECOVERY_SOURCE_KERNEL_SHA256
                or before.get('jump_tracking_reference') != TARGET_REFERENCE
                or before['extensions'].get('recovery_quiet', 0.) != 0.):
            raise ValueError('Recovery settling migration needs the known unextended flight kernel')
    elif change == 'ordinary-leg-acceleration-v1':
        expected['extensions']['ordinary_leg_acceleration'] = 1e-7
        if (before.get('kernel_sha256') != ACCELERATION_SOURCE_KERNEL_SHA256
                or before['extensions'].get('ordinary_leg_acceleration') is not None):
            raise ValueError('Acceleration migration requires the known native-coefficient kernel')
    else:
        raise ValueError('Unknown analytic reward migration')
    if after != expected or before.get('implementation') != 'shared_physical_reward_densities':
        raise ValueError('Analytic reward migration cannot change unrelated weights or kernel contracts')
    jump = JumpTaskConfig.from_dict(saved['jump_scenario'])
    if (not jump.flight_latch or jump != JumpTaskConfig.from_dict(reference['jump_task_config'])
            or saved['stair_scenario'] != reference['stair_source_config']
            or saved['recovery_scenario'] != reference['recovery_task_config']):
        raise ValueError('Jump/reference, stair and recovery state layouts must remain identical')
    storage, recorded = dataset['replay']['storage'], reference['replay']['storage']
    if (storage.keys() != recorded.keys()
            or any(value.shape[2:] != recorded[name].shape[2:] or value.dtype != recorded[name].dtype
                   for name, value in storage.items())):
        raise ValueError('Native reference and replay record layouts differ')
    latch = storage.get('jump_flight_latch')
    if latch is None or latch.dtype != torch.int64 or latch.shape[-1] != 3:
        raise ValueError('Reward-reference correction needs recorded first-flight memory')
    teacher_source = rebind_teacher_source(saved, source, target, source_teacher_path, target_teacher_path)
    report = {
        'migration': change, 'cycle': saved['cycle'],
        'source_schema': schema.fingerprint, 'target_schema': target_schema.fingerprint,
        'source_reward_kernel': before, 'target_reward_kernel': after,
        'legacy_reward_sequence_before': dataset['replay']['total_steps'],
        'legacy_reward_labels': 'unchanged native measurements; never actor/critic targets',
        'learner_and_optimizers': 'all preserved, including critic and normalizers',
        'task_records_and_validity': 'preserved; no histories inferred or invalidated',
        'curriculum_rng_and_counters': 'preserved; native simulator cold reset',
        'teacher_qualification': teacher_source['qualified_for'], 'teacher_weights_identical': True,
        'additional_warmup_cycles': 0,
    }
    upgraded = {**saved, 'schema': target_schema.to_dict(), 'schema_fingerprint': target_schema.fingerprint,
                'teacher_source': teacher_source,
                'resume_migrations': [*saved.get('resume_migrations', []), report]}
    replay = {**dataset, 'schema_fingerprint': target_schema.fingerprint}
    if 'schema' in dataset:
        replay['schema'] = target_schema.to_dict()
    return upgraded, replay, report


def upgrade_reward_reference(saved, dataset, reference, *, source_teacher_path, target_teacher_path):
    """Apply the first-real-flight LANDING reference correction to a complete pair."""
    return _upgrade_reward(saved, dataset, reference, source_teacher_path=source_teacher_path,
                           target_teacher_path=target_teacher_path, change='jump-reward-reference-v1')


def upgrade_recovery_quiet(saved, dataset, reference, *, source_teacher_path, target_teacher_path):
    """Enable the reviewed 1/s RECOVERY settling term without changing learning state."""
    return _upgrade_reward(saved, dataset, reference, source_teacher_path=source_teacher_path,
                           target_teacher_path=target_teacher_path, change='recovery-quiet-reward-v1')


def upgrade_ordinary_acceleration(saved, dataset, reference, *, source_teacher_path, target_teacher_path):
    """Use the reviewed 1e-7 locomotion coefficient while retaining learning state."""
    return _upgrade_reward(saved, dataset, reference, source_teacher_path=source_teacher_path,
                           target_teacher_path=target_teacher_path, change='ordinary-leg-acceleration-v1')
