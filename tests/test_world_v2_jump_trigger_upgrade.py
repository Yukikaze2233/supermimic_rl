"""Gate expansion preserves dynamics and values without manufacturing cooldown history."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import runpy

import pytest
import torch

from supermimic_rl.contracts import StateSchema
from supermimic_rl.course.jump_takeoff import install_jump_takeoff
from supermimic_rl.world_model_v2.imagination import Critic
from supermimic_rl.world_model_v2.jump_task import JumpTaskConfig, JumpTaskState
from supermimic_rl.world_model_v2.migrations.jump import upgrade_jump_tasks
from supermimic_rl.world_model_v2.migrations.jump_trigger import upgrade_jump_trigger
from supermimic_rl.world_model_v2.online import OnlineConfig, policy_start_candidates
from supermimic_rl.world_model_v2.replay import CompactReplay
from support.world_v2.jump_upgrade import fixture as jump_fixture
from support.world_v2.jump_imagination import fixture as imagined_fixture
from support.world_v2.jump_task import native_jumps


def fixture():
    saved, dataset, reference, options, boundary = jump_fixture()
    saved, dataset, _ = upgrade_jump_tasks(saved, dataset, reference, **options)
    saved['critic_norm_from'] = None
    dataset['replay']['storage']['jump_record_valid'].fill_(1.)
    native = native_jumps(2)
    receipt = install_jump_takeoff(native, {})
    source = json.loads(saved['schema']['source'])
    schema = StateSchema.from_dict({**saved['schema'], 'source': json.dumps({**source, 'jump_takeoff': receipt})})
    recorded = {name: value[None] for name, value in JumpTaskState.capture(native).record().items()}
    recorded.update(jump_record_valid=torch.ones(1, 2, 1), valid=torch.ones(1, 2, dtype=torch.bool),
                    policy_owned=torch.ones(1, 2, dtype=torch.bool))
    reference = {**reference, 'schema': schema.to_dict(), 'schema_fingerprint': schema.fingerprint,
        'jump_task_config': asdict(JumpTaskConfig.capture(native)), 'replay': {'storage': recorded}}
    return saved, dataset, reference, boundary


def test_gate_migration_preserves_critic_adam_and_world_data_and_requires_fresh_starts():
    saved, dataset, reference, boundary = fixture()
    original = deepcopy(reference['replay']['storage'])
    upgraded, replay, report = upgrade_jump_trigger(saved, dataset, reference)
    before = Critic(401, (16, 8), task_dim=40, terrain=True, stairs=True, jumps=True)
    after = Critic(404, (16, 8), task_dim=40, terrain=True, stairs=True, jumps=True, jump_trigger=True)
    before.load_state_dict(saved['learner']['critic'])
    after.load_state_dict(upgraded['learner']['critic'])
    x = torch.randn(5, 401)
    torch.testing.assert_close(after(torch.cat((x, torch.randn(5, 3)), -1)), before(x))
    assert not after.net[0].weight[:, 401:].any()
    first = saved['learner']['optimizers']['critic']['param_groups'][0]['params'][0]
    for name in ('exp_avg', 'exp_avg_sq'):
        old = saved['learner']['optimizers']['critic']['state'][first][name]
        new = upgraded['learner']['optimizers']['critic']['state'][first][name]
        assert torch.equal(new[:, :401], old) and not new[:, 401:].any()
    assert upgraded['critic_norm_from'] == 401
    assert report['legacy_gate_histories_inferred'] == 0
    assert report['legacy_valid_task_rows_invalidated'] == 6
    assert (dataset['replay']['storage']['jump_record_valid'] == 1).all()
    assert 'jump_trigger_clock' not in dataset['replay']['storage']
    for name, value in original.items():
        assert torch.equal(value, reference['replay']['storage'][name])
    for name in ('model', 'actor', 'updates'):
        assert upgraded['learner'][name] is saved['learner'][name]
    old_ring, new_ring = CompactReplay.from_state_dict(dataset['replay']), CompactReplay.from_state_dict(replay['replay'])
    torch.testing.assert_close(old_ring.window_candidates(1), new_ring.window_candidates(1))
    config = OnlineConfig.from_dict(upgraded['config'])
    assert policy_start_candidates(new_ring, config, [0, 1], .02).numel() == 0
    _, actual, _ = imagined_fixture(gated=True)
    record = {name: value[-1].clone() for name, value in replay['replay']['storage'].items()
              if name not in ('segment_age', *boundary)}
    record.update({name: value for name, value in actual.items() if name in record})
    new_ring.add(record, boundary)
    assert policy_start_candidates(new_ring, config, [0, 1], .02).numel() == 2
    assert new_ring.data['jump_trigger_clock'].dtype == torch.float64
    assert new_ring.data['jump_trigger_flags'].dtype == torch.bool


@pytest.mark.parametrize('damage', ['unknown', 'clock_dtype', 'source', 'version', 'reference'])
def test_gate_migration_refuses_unknown_target_or_unrelated_source_changes(damage):
    saved, dataset, reference, _ = fixture()
    if damage == 'unknown':
        reference['replay']['storage']['jump_trigger_flags'].zero_()
    elif damage == 'clock_dtype':
        reference['replay']['storage']['jump_trigger_clock'] = reference['replay']['storage']['jump_trigger_clock'].float()
    elif damage == 'source':
        source = json.loads(reference['schema']['source'])
        schema = StateSchema.from_dict({**reference['schema'], 'source': json.dumps({**source, 'mass': 99.})})
        reference.update(schema=schema.to_dict(), schema_fingerprint=schema.fingerprint)
    elif damage == 'version':
        reference['jump_task_config']['trigger_record_version'] = 99
    else:
        reference['jump_task_config']['reference']['compression_depth_max_m'] = .17
    with pytest.raises(ValueError):
        upgrade_jump_trigger(saved, dataset, reference)


def test_cli_writes_a_new_pair_and_preserves_original_files(tmp_path, monkeypatch):
    saved, dataset, reference, _ = fixture()
    saved['replay_file'] = 'old-replay.pt'
    checkpoint, replay, target = [tmp_path / name for name in ('old.pt', 'old-replay.pt', 'reference.pt')]
    for value, path in ((saved, checkpoint), (dataset, replay), (reference, target)):
        torch.save(value, path)
    originals = [path.read_bytes() for path in (checkpoint, replay, target)]
    output = tmp_path / 'gated'
    monkeypatch.setattr('sys.argv', ['upgrade_online_jump_trigger', '--checkpoint', str(checkpoint),
        '--reference-dataset', str(target), '--output', str(output)])
    runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/upgrade_online_jump_trigger.py'), run_name='__main__')
    assert originals == [path.read_bytes() for path in (checkpoint, replay, target)]
    result = torch.load(output / 'resume-a.pt', weights_only=True)
    assert result['jump_scenario'] == reference['jump_task_config']
    assert result['resume_migrations'][-1]['legacy_gate_histories_inferred'] == 0
    assert (output / 'replay-a.pt').is_file()
