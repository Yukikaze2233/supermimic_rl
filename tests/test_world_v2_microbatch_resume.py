"""Resume may repartition the same policy batch without replacing its recipe."""
from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.world_model_v2.online import CHECKPOINT_FORMAT, OnlineConfig, OnlineTrainer
from supermimic_rl.world_model_v2.records import MODEL_CONTRACT


@pytest.mark.parametrize('size', [0, -1, 2048., True, None])
def test_microbatch_keeps_existing_positive_integer_validation(size):
    with pytest.raises(ValueError, match='imagination_microbatch_size'):
        OnlineConfig(imagination_microbatch_size=size)


def check_fixture(before=4096, after=2048):
    previous = OnlineConfig(cycles=100, imagination_batch_size=4096,
                            imagination_microbatch_size=before, actor_updates=24)
    trainer = object.__new__(OnlineTrainer)
    trainer.env = SimpleNamespace(schema=SimpleNamespace(fingerprint='unchanged-source'))
    trainer.config = replace(previous, imagination_microbatch_size=after)
    trainer.teacher_source = {'checkpoint': 'same-teacher', 'qualification': 'data-only'}
    trainer.stair_scenario = trainer.jump_scenario = trainer.recovery_scenario = None
    saved = {'format': CHECKPOINT_FORMAT, 'model_contract': MODEL_CONTRACT,
             'schema_fingerprint': 'unchanged-source', 'config': asdict(previous), 'cycle': 10,
             'teacher_source': deepcopy(trainer.teacher_source), 'stair_scenario': None,
             'jump_scenario': None, 'recovery_scenario': None}
    return trainer, saved


@pytest.mark.parametrize('before,after', [(4096, 2048), (2048, 4096), (4096, 4096), (512, 256)])
def test_microbatch_change_preserves_saved_configuration_and_rng(before, after):
    trainer, saved = check_fixture(before, after)
    if before == 512:
        saved['config'].pop('imagination_microbatch_size')
    original = deepcopy(saved)
    rng = torch.get_rng_state().clone()
    trainer._check_saved(saved)
    assert trainer.imagination_microbatch_changed == (
        None if before == after else {'before': before, 'after': after})
    assert trainer.imagination_horizon_changed is None
    assert saved == original
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert OnlineConfig().imagination_microbatch_size == 512
    assert OnlineConfig.from_dict(asdict(trainer.config)) == trainer.config


@pytest.mark.parametrize('field,value', [
    ('imagination_batch_size', 2048), ('actor_updates', 12), ('model_batch_size', 1024),
    ('model_updates', 1), ('fit_horizon', 16), ('replay_steps', 2000), ('seed', 1),
    ('model_hidden', (32,)), ('actor_hidden', (32,)),
])
def test_microbatch_permission_does_not_change_total_work_or_other_recipe(field, value):
    trainer, saved = check_fixture()
    trainer.config = replace(trainer.config, **{field: value})
    with pytest.raises(ValueError, match='training recipe'):
        trainer._check_saved(saved)


@pytest.mark.parametrize('field,value', [
    ('schema_fingerprint', 'another-source'), ('teacher_source', {'checkpoint': 'different'}),
    ('stair_scenario', {'changed': True}), ('recovery_scenario', {'changed': True}),
])
def test_microbatch_permission_preserves_source_and_task_contracts(field, value):
    trainer, saved = check_fixture()
    saved[field] = value
    with pytest.raises(ValueError):
        trainer._check_saved(saved)
