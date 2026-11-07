"""In-memory migration preserves old truth and never invents contact evidence."""
from copy import deepcopy

import pytest
import torch

from supermimic_rl.world_model_v2.migrations.platform import (
    append_platform_history, append_platform_learner,
)
from supermimic_rl.world_model_v2.platform_contacts import PLATFORM_CONTACT_PRODUCER
from supermimic_rl.world_model_v2.platform_features import PLATFORM_FEATURE_DIM
from supermimic_rl.world_model_v2.platform_records import (
    PLATFORM_RECORD_DTYPES, PLATFORM_RECORD_FIELDS, PlatformRecordState, decode_platform_interval,
)
from supermimic_rl.world_model_v2.replay import CompactReplay
from support.world_v2.jump_imagination import fixture as old_fixture
from test_world_v2_platform_learning import assert_equal, fixture as platform_fixture
from test_world_v2_readonly_replay import snapshot


@pytest.mark.parametrize('steps', [0, 3, 8])
@pytest.mark.parametrize('known', [False, True])
def test_history_preserves_ring_and_unknown_measurements_through_save(steps, known):
    replay = CompactReplay.from_state_dict(snapshot(steps))
    tensors = dict(replay.data)
    prior = deepcopy(replay.state_dict())
    sequence = replay.sequence
    windows = replay.window_candidates(2).clone()
    report = append_platform_history(replay, caller_known_inactive=known)
    assert report['allocated_bytes'] == 217 * replay.capacity * replay.num_envs
    assert report['populated_transitions'] == len(replay)
    assert replay.sequence is sequence
    for name, value in tensors.items():
        assert replay.data[name] is value
    saved = replay.state_dict()
    for name, value in prior['storage'].items():
        torch.testing.assert_close(saved['storage'][name], value, atol=0, rtol=0)
    torch.testing.assert_close(replay.window_candidates(2), windows, atol=0, rtol=0)
    for name, shape in PLATFORM_RECORD_FIELDS.items():
        value = saved['storage'][name]
        assert value.dtype == PLATFORM_RECORD_DTYPES[name]
        assert value.shape == (replay.size, replay.num_envs, *shape)
    if replay.size:
        record = {name: value.flatten(0, 1) for name, value in saved['storage'].items()}
        state, ticks = PlatformRecordState.decode(record), decode_platform_interval(record)
        assert (state.valid == known).all()
        state.intent.validate(values=True)
        assert (state.intent.target_index == -1).all()
        assert (state.landing.request_id == -1).all()
        assert not ticks.evidence.measurement_valid.any()
        assert not ticks.evidence.selected.any()
        assert not ticks.evidence.landing_support_valid.any()
        assert torch.isnan(ticks.evidence.landing_world_height).all()
        assert (ticks.state.request_id == -1).all()
    restored = CompactReplay.from_state_dict(saved)
    assert_equal(saved, restored.state_dict())


def test_history_never_changes_an_existing_platform_record_or_audit_view():
    replay = CompactReplay.from_state_dict(snapshot(3))
    append_platform_history(replay, caller_known_inactive=False)
    prior = deepcopy(replay.state_dict())
    with pytest.raises(ValueError, match='overwritten'):
        append_platform_history(replay, caller_known_inactive=True)
    assert_equal(prior, replay.state_dict())
    view = CompactReplay.from_state_dict(snapshot(3), read_only=True)
    with pytest.raises(ValueError, match='read-only'):
        append_platform_history(view, caller_known_inactive=True)


def test_learner_keeps_value_function_adam_counts_and_online_rng(tmp_path):
    old, _, _ = old_fixture(gated=True)
    new, sample, command = platform_fixture(tmp_path)
    dimension = old.critic.net[0].in_features
    features = torch.randn(3, dimension)
    old.optimizers['critic'].zero_grad(set_to_none=True)
    old.critic(features).sum().backward()
    old.optimizers['critic'].step()
    old.updates['critic'] = 1
    old.critic.norm.count.fill_(1234)
    old.critic.norm.mean.copy_(torch.linspace(-1, 1, dimension))
    old.critic.norm.var.fill_(.7)
    new.fit_platform_contact(sample, command_height=command)
    head = {
        'format': 'supermimic-platform-contact-head-v1',
        'producer': PLATFORM_CONTACT_PRODUCER,
        'model': new.platform_contact.state_dict(),
        'optimizer': new.optimizers['platform_contact'].state_dict(),
        'updates': new.updates['platform_contact'],
        'skipped': new.skipped['platform_contact'],
        'rng': torch.zeros(2, dtype=torch.uint8),
    }
    before = deepcopy(old.state_dict())
    rng = torch.get_rng_state().clone()
    migrated = append_platform_learner(before, head, expected_critic_dim=dimension)
    torch.testing.assert_close(rng, torch.get_rng_state(), atol=0, rtol=0)
    for key in ('model', 'actor'):
        assert_equal(before[key], migrated[key])
    for key in ('model', 'actor', 'estimator'):
        assert_equal(before['optimizers'][key], migrated['optimizers'][key])
    assert_equal(head['model'], migrated['platform_contact'])
    assert_equal(head['optimizer'], migrated['optimizers']['platform_contact'])
    assert migrated['updates']['platform_contact'] == 1
    assert migrated['updates']['critic'] == 1
    new.load_state_dict(migrated)
    assert_equal(migrated, new.state_dict())
    appended = torch.randn(3, PLATFORM_FEATURE_DIM)
    torch.testing.assert_close(old.critic(features), new.critic(torch.cat((features, appended), -1)),
                               atol=1e-7, rtol=1e-6)
    assert (new.critic.norm.mean[dimension:] == 0).all()
    assert (new.critic.norm.var[dimension:] == 1).all()
    assert new.critic.norm.count == old.critic.norm.count
    prior_adam = before['optimizers']['critic']
    current = migrated['optimizers']['critic']
    first = prior_adam['param_groups'][0]['params'][0]
    for key in ('exp_avg', 'exp_avg_sq'):
        torch.testing.assert_close(current['state'][first][key][:, :dimension],
                                   prior_adam['state'][first][key], atol=0, rtol=0)
        assert (current['state'][first][key][:, dimension:] == 0).all()
    assert_equal(before, old.state_dict())
    with pytest.raises(ValueError, match='four learner'):
        append_platform_learner(migrated, head, expected_critic_dim=dimension)


def test_incomplete_head_does_not_initialize_or_mutate_learner():
    old, _, _ = old_fixture(gated=True)
    before = deepcopy(old.state_dict())
    with pytest.raises(ValueError, match='complete contact head'):
        append_platform_learner(before, {}, expected_critic_dim=old.critic.net[0].in_features)
    assert_equal(before, old.state_dict())
