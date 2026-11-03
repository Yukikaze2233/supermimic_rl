"""Contact sampling changes fitting coverage without changing native labels or boundaries."""
import pytest
import torch

from supermimic_rl.world_model_v2.physical import INCREMENT_DIM
from supermimic_rl.world_model_v2.records import TICK_LABEL_DIM
from supermimic_rl.world_model_v2.replay import CompactReplay


def append(replay, step, off=(), *, boundary=None, sequence=None):
    n = replay.num_envs
    ticks = torch.full((n, 4, TICK_LABEL_DIM), 40.)
    for env, tick, load in off:
        ticks[env, tick, INCREMENT_DIM:INCREMENT_DIM + 2] = load
    record = {'ticks': ticks, 'identity': torch.stack((torch.full((n,), step), torch.arange(n)), -1).float()}
    edges = {name: torch.ones(n, dtype=torch.bool) for name in ('valid', 'policy_owned', 'end_policy_owned')}
    edges.update({name: torch.zeros(n, dtype=torch.bool) for name in ('terminated', 'truncated')})
    edges.update({name: torch.zeros(n, dtype=torch.long) for name in
                  ('start_episode_id', 'end_episode_id', 'start_control_epoch', 'end_control_epoch')})
    for name, value in (boundary or {}).items():
        edges[name][0] = value
    replay.add(record, edges, sequence_id=sequence)


def replay_with_crossing(*, boundary_kind=None):
    replay = CompactReplay({'ticks': (4, TICK_LABEL_DIM), 'identity': (2,)}, 2, 18, 'cpu')
    for step in range(18):
        off = [(1, tick, 0.) for tick in range(4)]  # Held-out environment has abundant events.
        if step == 7:
            off += [(0, 3, 2.)]
        if step == 8:
            off += [(0, 0, 2.)]
        edges = {}
        if boundary_kind == 'invalid' and step == 8:
            edges['valid'] = False
        if boundary_kind in ('terminated', 'truncated') and step == 7:
            edges[boundary_kind] = True
        if boundary_kind in ('episode_id', 'control_epoch') and step >= 8:
            edges['start_' + boundary_kind] = edges['end_' + boundary_kind] = 1
        sequence = step + int(boundary_kind == 'sequence' and step >= 8)
        append(replay, step, off, boundary=edges, sequence=sequence)
    return replay


def test_two_ticks_cross_policy_boundary_and_holdout_stays_separate():
    replay = replay_with_crossing()
    assert replay.contact_event_window_candidates(1, [0]).numel() == 0
    assert replay.contact_event_window_candidates(8, [0]).tolist() == [start * 2 for start in range(1, 8)]
    assert (replay.contact_event_window_candidates(8, [1]) % 2 == 1).all()
    assert replay.contact_event_window_candidates(8, [1]).numel() == 11


@pytest.mark.parametrize('kind', ['invalid', 'terminated', 'truncated', 'episode_id', 'control_epoch', 'sequence'])
def test_no_event_pair_crosses_a_native_or_replay_boundary(kind):
    replay = replay_with_crossing(boundary_kind=kind)
    assert not replay.contact_event_window_candidates(8, [0]).numel()


def test_event_requires_adjacent_true_wheel_support_loads_not_other_contact_columns():
    replay = CompactReplay({'ticks': (4, TICK_LABEL_DIM), 'identity': (2,)}, 1, 4, 'cpu')
    append(replay, 0, [(0, 0, 0.), (0, 2, 0.)])  # Separated ticks are not a duration.
    append(replay, 1, [(0, 0, 2.), (0, 1, 2.)])  # Inclusive native off threshold.
    append(replay, 2, [(0, 0, 2.01), (0, 1, 2.01)])
    append(replay, 3)
    replay.data['ticks'][..., INCREMENT_DIM + 2:] = 0.  # Body/resultant loads are irrelevant.
    assert replay.contact_event_window_candidates(1).tolist() == [1]


def test_lazy_contact_cache_tracks_ring_overwrites_and_serialized_chronology():
    replay = CompactReplay({'ticks': (4, TICK_LABEL_DIM), 'identity': (2,)}, 1, 10, 'cpu')
    for step in range(10):
        append(replay, step, [(0, 0, 0.), (0, 1, 0.)] if step == 4 else [])
    assert replay.contact_event_window_candidates(8).numel() == 3
    cache = replay._contact_off_ticks
    for step in range(10, 16):
        append(replay, step)
    assert replay._contact_off_ticks is cache
    assert not replay.contact_event_window_candidates(8).numel()
    append(replay, 16, [(0, 3, 0.)])
    append(replay, 17, [(0, 0, 0.)])
    restored = CompactReplay.from_state_dict(replay.state_dict(), read_only=True)
    torch.testing.assert_close(replay.contact_event_window_candidates(8),
                               restored.contact_event_window_candidates(8))
    assert replay.contact_event_window_candidates(8).tolist() == [2]


@pytest.mark.parametrize('preallocated', [False, True])
def test_mix_has_uniform_prefix_reuses_buffers_and_never_draws_held_out_envs(preallocated):
    replay = replay_with_crossing()
    names = ('identity', 'ticks')
    buffers = ({name: torch.empty(12, 8, *replay.data[name].shape[2:]) for name in names}
               if preallocated else None)
    rng, expected_rng = (torch.Generator().manual_seed(17) for _ in range(2))
    expected_uniform = replay.sample_windows(9, 8, generator=expected_rng, env_ids=[0], field_names=names)
    expected_event = replay.sample_windows(3, 8, generator=expected_rng, env_ids=[0], field_names=names,
                                          candidates=replay.contact_event_window_candidates(8, [0]))
    actual, uniform_count = replay.sample_contact_windows(12, 8, contact_event_fraction=.25,
        generator=rng, env_ids=[0], field_names=names, out=buffers)
    assert uniform_count == 9
    if buffers is not None:
        assert actual is buffers
    assert not actual['identity'][..., 1].any()
    for name in names:
        torch.testing.assert_close(actual[name][:9], expected_uniform[name], rtol=0, atol=0)
        torch.testing.assert_close(actual[name][9:], expected_event[name], rtol=0, atol=0)
    assert torch.equal(rng.get_state(), expected_rng.get_state())
    assert ((actual['identity'][9:, 0, 0] >= 1) & (actual['identity'][9:, 0, 0] <= 7)).all()


@pytest.mark.parametrize('fraction', [0., .25])
def test_zero_fraction_or_empty_pool_keeps_original_draw_and_rng(fraction):
    replay = replay_with_crossing(boundary_kind='invalid')
    rng, expected_rng = (torch.Generator().manual_seed(27) for _ in range(2))
    expected = replay.sample_windows(8, 8, generator=expected_rng, env_ids=[0])
    actual, uniform_count = replay.sample_contact_windows(8, 8, contact_event_fraction=fraction,
                                                         generator=rng, env_ids=[0])
    assert uniform_count == 8
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    assert torch.equal(rng.get_state(), expected_rng.get_state())
    if fraction == 0:
        assert getattr(replay, '_contact_off_ticks', None) is None


@pytest.mark.parametrize('fraction', [-.1, 1., float('nan'), float('inf')])
def test_invalid_fraction_is_rejected_without_rng_draws(fraction):
    replay, rng = replay_with_crossing(), torch.Generator().manual_seed(3)
    before = rng.get_state()
    with pytest.raises(ValueError, match='fraction'):
        replay.sample_contact_windows(8, 8, contact_event_fraction=fraction, generator=rng)
    assert torch.equal(before, rng.get_state())


def test_cached_windows_match_direct_tick_duration_audit_after_wrap_and_restore():
    replay = CompactReplay({'ticks': (4, TICK_LABEL_DIM), 'identity': (2,)}, 2, 13, 'cpu')
    generator = torch.Generator().manual_seed(71)
    for step in range(29):
        off = [(env, tick, 0.) for env, tick in (torch.rand(2, 4, generator=generator) < .3).nonzero().tolist()]
        append(replay, step, off, boundary={'truncated': step in (17, 25)})
        if step == 12:
            replay.contact_event_window_candidates(8)
    for source in (replay, CompactReplay.from_state_dict(replay.state_dict(), read_only=True)):
        for horizon in (1, 2, 8):
            expected = []
            for candidate in source.window_candidates(horizon, [0]).tolist():
                start, env = divmod(candidate, source.num_envs)
                slots = (torch.arange(horizon) + start + source.head - source.size) % source.capacity
                raw = source.data['ticks'][slots, env, :, INCREMENT_DIM:INCREMENT_DIM + 2].reshape(-1, 2)
                off = (raw <= 2.).all(-1)
                if (off[:-1] & off[1:]).any():
                    expected.append(candidate)
            assert source.contact_event_window_candidates(horizon, [0]).tolist() == expected


def test_mixed_buffer_rejection_precedes_draws():
    replay, generator = replay_with_crossing(), torch.Generator().manual_seed(2)
    before = generator.get_state()
    with pytest.raises(ValueError, match='full mixed batch'):
        replay.sample_contact_windows(8, 8, contact_event_fraction=.25, generator=generator,
            env_ids=[0], field_names=['identity'], out={'identity': torch.empty(9, 8, 2)})
    assert torch.equal(before, generator.get_state())


def test_partition_pools_are_reused_until_replay_changes(monkeypatch):
    replay = replay_with_crossing()
    original = replay._allowed
    partitions = []

    def allowed(candidates, ids):
        if ids is not None:
            partitions.append(tuple(ids))
        return original(candidates, ids)

    monkeypatch.setattr(replay, '_allowed', allowed)
    replay.sample_contact_windows(8, 8, contact_event_fraction=.25, env_ids=[0])
    first = len(partitions)
    replay.sample_contact_windows(8, 8, contact_event_fraction=.25, env_ids=torch.tensor([0]))
    assert len(partitions) == first
    replay.sample_contact_windows(8, 8, contact_event_fraction=.25, env_ids=[1])
    assert len(partitions) > first
    before_append = len(partitions)
    append(replay, 18)
    replay.sample_contact_windows(8, 8, contact_event_fraction=.25, env_ids=[0])
    assert len(partitions) > before_append
