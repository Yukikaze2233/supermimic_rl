"""Recorded recovery migration fixture shared by explicit handoff tests."""
from dataclasses import asdict

import torch

from supermimic_rl.world_model_v2.migrations.jump import upgrade_jump_tasks
from supermimic_rl.world_model_v2.records import MODEL_CONTRACT
from supermimic_rl.world_model_v2.recovery_task import RecoveryTaskConfig, RecoveryTaskState
from .jump_upgrade import fixture as jump_upgrade_fixture
from .recovery_task import native_recovery


def fixture():
    saved, dataset, reference, options, boundary = jump_upgrade_fixture()
    saved, dataset, _ = upgrade_jump_tasks(saved, dataset, reference, **options)
    saved['critic_norm_from'] = None
    saved['teacher_source'] = {'fixture': 'own-foundation', 'sha256': 'unchanged-teacher'}
    saved['mixing'] = {'calls': 1234, 'teacher_fraction': .15}
    for name, seed in (('collection_rng', 111), ('mixing_rng', 222), ('validation_rng', 333), ('torch_rng', 444)):
        saved[name] = torch.Generator().manual_seed(seed).get_state()
    native = native_recovery(2)
    reference_data = {name: value[None] for name, value in RecoveryTaskState.capture(native).record().items()}
    reference_data.update(recovery_record_valid=torch.ones(1, 2, 1),
                          valid=torch.ones(1, 2, dtype=torch.bool),
                          policy_owned=torch.ones(1, 2, dtype=torch.bool))
    reference = {
        'format': 'supermimic-world-v2-dataset-v1', 'model_contract': MODEL_CONTRACT,
        'schema': saved['schema'], 'schema_fingerprint': saved['schema_fingerprint'],
        'stair_source_config': saved['stair_scenario'], 'jump_task_config': saved['jump_scenario'],
        'recovery_task_config': asdict(RecoveryTaskConfig.capture(native)),
        'replay': {'storage': reference_data},
    }
    return saved, dataset, reference, {
        'course_raw': options['target_course_raw'], 'course_coordinate': 125, 'world_only_cycles': 3,
    }, boundary

