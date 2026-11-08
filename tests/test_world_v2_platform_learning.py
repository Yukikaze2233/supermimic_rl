"""Contact supervision has its own optimizer and survives complete learner saves."""
from copy import deepcopy

import pytest
import torch

from supermimic_rl.course.jump_physics import PlatformLandingState
from supermimic_rl.world_model_v2.imagination import Critic
from supermimic_rl.world_model_v2.jump_task import JumpTaskState
from supermimic_rl.world_model_v2.learning import Learner
from supermimic_rl.world_model_v2.motion_task import MOTION_TASK_DIM
from supermimic_rl.world_model_v2.platform_records import PlatformRecordState
from supermimic_rl.world_model_v2.platform_task import PlatformTask
from support.world_v2.jump_imagination import fixture as jump_fixture
from test_world_v2_platform_contacts import inputs, record, set_output


def fixture(tmp_path):
    old, start, _ = jump_fixture(gated=True)
    contact, ticks, arguments = inputs(tmp_path)
    # This fitter test uses the explicit two-tile triangle fixture; it does not
    # execute motion rollouts or claim native landing accuracy.
    old.model.terrain = contact.geometry.terrain
    prior = old.imagination
    prior.stair_source.terrain = old.model.terrain
    prior.reward.terrain = old.model.terrain
    routes = dict(task_dim=MOTION_TASK_DIM, terrain=True, stairs=True,
                  jumps=True, jump_trigger=True, platforms=True)
    critic = Critic(Critic.feature_size(16, 65, **routes), (16,), **routes)
    learner = Learner(old.model, old.actor, critic, prior.termination, old.config,
        reward=prior.reward, stair_task=prior.stair_task, stair_source=prior.stair_source,
        jump_task=prior.jump_task,
        platform_task=PlatformTask(prior.jump_task, prior.stair_task, prior.stair_source),
        platform_contact=contact, platform_geometry=contact.geometry)
    truth = contact.predict(ticks, **arguments)
    sample = record(contact, truth, arguments, ticks=ticks)
    platform = PlatformRecordState(arguments['intent'],
        PlatformLandingState.empty(arguments['intent'].bound_request_id),
        torch.ones(2, 1, dtype=torch.bool))
    sample = {**start, **sample, **platform.record()}
    JumpTaskState.decode(sample).physical_id.copy_(7)
    # Labels were constructed independently before changing model predictions.
    set_output(contact, logits=(-4., -4., -4.))
    return learner, sample, arguments['command_height']


def assert_equal(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal(a, b)
    elif isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0, equal_nan=True)
    else:
        assert left == right


def test_head_really_updates_without_changing_actor_motion_or_critic(tmp_path):
    learner, sample, command = fixture(tmp_path)
    before = deepcopy(learner.state_dict())
    result = learner.fit_platform_contact(sample, command_height=command)
    assert result['platform_contact/updated']
    assert result['platform_contact/labels'] == result['platform_contact/height_labels'] == 8
    assert learner.updates['platform_contact'] == 1 and learner.skipped['platform_contact'] == 0
    assert learner.optimizers['platform_contact'].param_groups[0]['lr'] == learner.config.model_learning_rate
    assert not torch.equal(before['platform_contact']['net.0.bias'], learner.platform_contact.net[0].bias)
    after = learner.state_dict()
    for name in ('actor', 'model', 'critic'):
        assert_equal(before[name], after[name])
    for name in ('actor', 'model', 'critic', 'estimator'):
        assert_equal(before['optimizers'][name], after['optimizers'][name])
        assert learner.updates[name] == 0


def test_unknown_evidence_does_not_update_head_or_moments(tmp_path):
    learner, sample, command = fixture(tmp_path)
    learner.fit_platform_contact(sample, command_height=command)
    sample['contact_valid_ticks'].zero_()
    before = deepcopy(learner.state_dict())
    result = learner.fit_platform_contact(sample, command_height=command)
    assert result['platform_contact/labels'] == 0 and not result['platform_contact/updated']
    assert_equal(before, learner.state_dict())


def test_contact_weights_normalizers_and_adam_resume_exactly(tmp_path):
    learner, sample, command = fixture(tmp_path)
    learner.fit_platform_contact(sample, command_height=command)
    saved = deepcopy(learner.state_dict())
    child = tmp_path / 'restored'
    child.mkdir()
    restored, _, _ = fixture(child)
    restored.load_state_dict(saved)
    assert_equal(saved, restored.state_dict())
    learner.fit_platform_contact(sample, command_height=command)
    restored.fit_platform_contact(sample, command_height=command)
    assert_equal(learner.state_dict(), restored.state_dict())
    missing = {k: v for k, v in saved.items() if k != 'platform_contact'}
    with pytest.raises(ValueError, match='platform contact contracts'):
        restored.load_state_dict(missing)


def test_cached_supervision_matches_recomputed_head_loss_and_gradients(tmp_path, monkeypatch):
    model, ticks, arguments = inputs(tmp_path)
    initial = model.predict(ticks, **arguments)
    labels = model.targets(record(model, initial, arguments, ticks=ticks),
                          command_height=arguments['command_height'])
    model.observe(initial, labels)
    live = model.predict(ticks, **arguments)
    live_loss = sum(model.losses(live, labels).values())
    expected = torch.autograd.grad(live_loss, tuple(model.parameters()))
    monkeypatch.setattr(model.clearance, 'at', lambda *a, **k: pytest.fail('Repeated cylinder query'))
    cached_loss = sum(model.losses_from_features(initial.features.detach(), labels,
                                                available=initial.available).values())
    actual = torch.autograd.grad(cached_loss, tuple(model.parameters()))
    torch.testing.assert_close(cached_loss, live_loss, rtol=0, atol=0)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_legacy_learner_does_not_add_contact_checkpoint_state():
    learner, _, _ = jump_fixture(gated=True)
    saved = deepcopy(learner.state_dict())
    assert 'platform_contact' not in saved and 'platform_contact' not in learner.optimizers
    learner.load_state_dict(saved)
    assert_equal(saved, learner.state_dict())
