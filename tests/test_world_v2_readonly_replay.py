"""Memory-mapped audit views preserve validation and cannot append to training data."""
import hashlib
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.world_model_v2.data_report import report
from supermimic_rl.world_model_v2.model import Ticks
from supermimic_rl.world_model_v2.records import RecordSpec, TICK_LABEL_DIM
from supermimic_rl.world_model_v2.replay import CompactReplay


def snapshot(steps):
    fields = RecordSpec(16, 65).fields()
    replay = CompactReplay(fields, 2, 5, 'cpu')
    boundary = {name: torch.ones(2, dtype=torch.bool) for name in
                ('valid', 'policy_owned', 'end_policy_owned')}
    boundary.update({name: torch.zeros(2, dtype=torch.bool) for name in ('terminated', 'truncated')})
    boundary.update({name: torch.zeros(2, dtype=torch.long) for name in
                    ('start_episode_id', 'end_episode_id', 'start_control_epoch', 'end_control_epoch')})
    for step in range(steps):
        record = {name: torch.zeros(2, *shape) for name, shape in fields.items()}
        record['physical'][:, 6] = record['end_physical'][:, 6] = 1.
        record['failure_conditions_valid'].fill_(1.)
        record['frame'].fill_(step)
        record['action'].fill_(step * .1)
        replay.add(record, boundary)
    return replay.state_dict()


@pytest.mark.parametrize('steps', [3, 8])
def test_mmap_alias_sampling_matches_default_copy_and_reports_leave_source_unchanged(tmp_path, steps):
    path = tmp_path / 'replay.pt'
    torch.save(snapshot(steps), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    saved = torch.load(path, weights_only=True, mmap=True)
    before = {name: value.clone() for name, value in saved['storage'].items()}
    original_sequence = saved['sequence'].clone()
    view = CompactReplay.from_state_dict(saved, read_only=True)
    copied = CompactReplay.from_state_dict(saved)
    assert view.read_only and not copied.read_only
    assert view.capacity == min(steps, 5) and copied.capacity == 5
    assert view.sequence.data_ptr() == saved['sequence'].data_ptr()
    for name, value in view.data.items():
        assert value.data_ptr() == saved['storage'][name].data_ptr()
        assert copied.data[name].data_ptr() != value.data_ptr()
    for horizon in (1, 3):
        torch.testing.assert_close(view.window_candidates(horizon, [1]), copied.window_candidates(horizon, [1]))
        windows = view.sample_windows(8, horizon, generator=torch.Generator().manual_seed(8), env_ids=[1])
        expected = copied.sample_windows(8, horizon, generator=torch.Generator().manual_seed(8), env_ids=[1])
        for name, value in windows.items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
            value.zero_()  # Samples own storage and may be changed by a consumer.
    starts, frames = view.sample_starts(8, 5, generator=torch.Generator().manual_seed(8), env_ids=[1])
    expected_starts, expected_frames = copied.sample_starts(8, 5, generator=torch.Generator().manual_seed(8), env_ids=[1])
    torch.testing.assert_close(frames, expected_frames, rtol=0, atol=0)
    for name in starts:
        torch.testing.assert_close(starts[name], expected_starts[name], rtol=0, atol=0)

    class StaticModel:
        def step(self, state, height, scan, memory, context, strength, action):
            return SimpleNamespace(state=state, height_above_ground=height, terrain_scan=scan,
                                   memory=memory, ticks=Ticks.decode(torch.zeros(len(action), 4, TICK_LABEL_DIM), state))

    result = report(view, fit=SimpleNamespace(model=StaticModel()), env_ids=[1], horizon=3, batch=8,
                    generator=torch.Generator().manual_seed(9))
    assert result['populations']['all']['coverage']['transitions'] == min(steps, 5) * 2
    with pytest.raises(RuntimeError, match='read-only'):
        view.add({}, {})
    for name, value in saved['storage'].items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    torch.testing.assert_close(saved['sequence'], original_sequence, rtol=0, atol=0)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize('damage', ['dtype', 'shape', 'boundary', 'chronology', 'capacity', 'environments'])
def test_readonly_loading_runs_full_snapshot_validation(damage):
    saved = snapshot(3)
    if damage == 'dtype':
        saved['storage']['action'] = saved['storage']['action'].double()
    elif damage == 'shape':
        saved['storage']['action'] = saved['storage']['action'][:2]
    elif damage == 'boundary':
        del saved['storage']['valid']
    elif damage == 'chronology':
        saved['sequence'][1] = saved['sequence'][0]
    elif damage == 'capacity':
        saved['capacity'] = 2
    else:
        saved['num_envs'] = 0
    for read_only in (False, True):
        with pytest.raises(ValueError):
            CompactReplay.from_state_dict(saved, read_only=read_only)


def test_empty_audit_view_and_forbidden_storage_overrides():
    saved = snapshot(0)
    view = CompactReplay.from_state_dict(saved, read_only=True)
    assert len(view) == 0 and not view.window_candidates(1).numel()
    assert report(view)['populations']['all']['coverage'] == {'transitions': 0}
    for options in ({'capacity_steps': 10}, {'env_ids': [0]}, {'device': 'cuda:0'}):
        with pytest.raises(ValueError, match='CPU storage'):
            CompactReplay.from_state_dict(saved, read_only=True, **options)
