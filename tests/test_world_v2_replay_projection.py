"""Batch projection and reusable output buffers preserve replay sampling exactly."""
import pytest
import torch

from supermimic_rl.world_model_v2.replay import CompactReplay


def fixture():
    replay = CompactReplay({'ticks': (4, 56), 'scan': (77,), 'unused': (128,)}, 4, 12, 'cpu')
    boundary = {name: torch.ones(4, dtype=torch.bool) for name in ('valid', 'policy_owned', 'end_policy_owned')}
    boundary.update({name: torch.zeros(4, dtype=torch.bool) for name in ('terminated', 'truncated')})
    boundary.update({name: torch.zeros(4, dtype=torch.long) for name in
                    ('start_episode_id', 'end_episode_id', 'start_control_epoch', 'end_control_epoch')})
    for step in range(15):
        values = {}
        for name in ('ticks', 'scan', 'unused'):
            shape = replay.data[name].shape[1:]
            values[name] = torch.arange(torch.tensor(shape).prod()).reshape(shape).float() + step * 1000
        replay.add(values, boundary)
    return replay


@pytest.mark.parametrize('preallocated', [False, True])
def test_projection_keeps_draw_rng_chronology_and_environment_partition(preallocated):
    replay = fixture()
    fields = ('ticks', 'scan', 'valid', 'start_episode_id')
    first, second = (torch.Generator().manual_seed(7) for _ in range(2))
    buffers = ({name: torch.empty(8, 8, *replay.data[name].shape[2:], dtype=replay.data[name].dtype)
                for name in fields} if preallocated else None)
    for _ in range(3):
        expected = replay.sample_windows(8, 8, generator=first, env_ids=[0, 2])
        actual = replay.sample_windows(8, 8, generator=second, env_ids=[0, 2], field_names=fields, out=buffers)
        assert set(actual) == set(fields)
        if buffers is not None:
            assert actual is buffers
        for name in fields:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        assert torch.equal(first.get_state(), second.get_state())


def test_projection_rejects_invalid_buffers_before_consuming_rng():
    replay, generator = fixture(), torch.Generator().manual_seed(1)
    before = generator.get_state()
    for kwargs in ({'field_names': ['missing']}, {'field_names': ['ticks', 'ticks']},
                   {'field_names': ['ticks'], 'out': {}},
                   {'field_names': ['ticks'], 'out': {'ticks': torch.empty(2, 8, 4, 56, dtype=torch.float64)}}):
        with pytest.raises(ValueError, match='Replay'):
            replay.sample_windows(2, 8, generator=generator, **kwargs)
        assert torch.equal(before, generator.get_state())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Pinned transfer requires CUDA')
def test_reused_pinned_projection_transfers_complete_windows_exactly():
    replay = fixture()
    fields = ('ticks', 'valid')
    buffers = {name: torch.empty(4, 8, *replay.data[name].shape[2:], dtype=replay.data[name].dtype,
                                pin_memory=True) for name in fields}
    for seed in (3, 4):
        expected = replay.sample_windows(4, 8, generator=torch.Generator().manual_seed(seed), env_ids=[0, 2])
        actual = replay.sample_windows(4, 8, generator=torch.Generator().manual_seed(seed), env_ids=[0, 2],
                                        field_names=fields, out=buffers)
        moved = {name: value.to('cuda', non_blocking=True) for name, value in actual.items()}
        torch.cuda.synchronize()
        for name in fields:
            assert actual[name].is_pinned()
            torch.testing.assert_close(moved[name].cpu(), expected[name], rtol=0, atol=0)
