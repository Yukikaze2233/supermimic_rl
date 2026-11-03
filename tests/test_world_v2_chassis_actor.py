"""Chassis-only history actors keep gimbal physics outside the policy interface."""
from types import SimpleNamespace
from pathlib import Path
import runpy

import pytest
import torch

from supermimic_rl.world_model_v2.migrations.actor import restrict_actor_to_chassis
from supermimic_rl.world_model_v2.online import OnlineConfig, OnlineTrainer
from supermimic_rl.world_model_v2.policy import EstimatorActor, export_onnx

from support.world_v2.motion_upgrade import fixture as checkpoint_fixture
from support.world_v2.stair_imagination import fixture as imagination_fixture
from support.world_v2.chassis_actor import migrated_fixture


def test_chassis_actor_and_estimator_ignore_every_gimbal_channel_and_keep_input_gradients():
    torch.manual_seed(12)
    actor = EstimatorActor(frame_dim=35, estimator_hidden=(8,), actor_hidden=(16,))
    history = torch.randn(3, 5, 45, requires_grad=True)
    changed = history.detach().clone()
    changed[..., 35:] = float('nan')
    torch.testing.assert_close(actor(history), actor(changed), rtol=0, atol=0)
    torch.testing.assert_close(actor.estimate(history), actor.estimate(changed), rtol=0, atol=0)
    torch.testing.assert_close(actor(history), actor(history[..., :35]), rtol=0, atol=0)
    actor(history).square().sum().backward()
    assert torch.isfinite(history.grad).all()
    assert history.grad[..., :35].abs().sum() > 0
    assert not history.grad[..., 35:].any()
    assert all(parameter.grad is None for parameter in actor.estimator.parameters())
    actor.estimator_loss(history.detach(), torch.zeros(3, 3), torch.zeros(3, 2)).backward()
    assert all(parameter.grad is not None for parameter in actor.estimator.parameters())
    for invalid in (torch.zeros(3, 5, 36), torch.zeros(3, 4, 35), torch.zeros(35)):
        with pytest.raises(ValueError, match='Actor requires'):
            actor(invalid)


def test_legacy_config_stays_formal45_and_cannot_silently_resume_as_chassis35():
    assert OnlineConfig.from_dict({}).actor_frame_dim == 45
    assert EstimatorActor(estimator_hidden=(8,), actor_hidden=(16,)).frame_dim == 45
    saved, _, _, _, _, _, _ = checkpoint_fixture()
    trainer = object.__new__(OnlineTrainer)
    trainer.env = SimpleNamespace(schema=SimpleNamespace(fingerprint=saved['schema_fingerprint']))
    trainer.teacher_source = trainer.stair_scenario = trainer.jump_scenario = trainer.recovery_scenario = None
    trainer.config = OnlineConfig(cycles=33)
    trainer._check_saved(saved)
    trainer.config = OnlineConfig(cycles=33, actor_frame_dim=35)
    with pytest.raises(ValueError, match='training recipe'):
        trainer._check_saved(saved)
    for invalid in (34, 36, 45., True):
        with pytest.raises(ValueError, match='frame dimension'):
            OnlineConfig(actor_frame_dim=invalid)


def test_explicit_branch_selects_each_frame_columns_resets_only_affected_adam_and_preserves_physics():
    saved, dataset, old_actor = migrated_fixture()
    old_weights = {name: value.clone() for name, value in old_actor.state_dict().items()}
    upgraded, retained_data, report = restrict_actor_to_chassis(saved, dataset)
    new_actor = EstimatorActor(frame_dim=35, estimator_hidden=(8,), actor_hidden=(16,))
    new_actor.load_state_dict(upgraded['learner']['actor'])
    assert retained_data is dataset
    assert dataset['replay']['storage']['frame'].shape[-1] == 45
    assert upgraded['learner']['model'] is saved['learner']['model']
    assert upgraded['learner']['critic'] is saved['learner']['critic']
    for name in ('model', 'critic'):
        assert upgraded['learner']['optimizers'][name] is saved['learner']['optimizers'][name]
    for name, module in (('actor', new_actor.decoder), ('estimator', new_actor.estimator)):
        assert saved['learner']['optimizers'][name]['state']
        assert upgraded['learner']['optimizers'][name]['state'] == {}
        optimizer = torch.optim.Adam(module.parameters())
        optimizer.load_state_dict(upgraded['learner']['optimizers'][name])
        assert not optimizer.state
    assert upgraded['learner']['updates'] == saved['learner']['updates']
    assert report['policy_equivalent'] is False
    assert upgraded['config']['actor_frame_dim'] == 35
    assert 'actor_frame_dim' not in saved['config']
    for name, value in old_actor.state_dict().items():
        torch.testing.assert_close(value, old_weights[name], rtol=0, atol=0)
    history = torch.randn(3, 5, 35)
    zero_gimbal = torch.cat((history, torch.zeros(3, 5, 10)), -1)
    # It is an initialization equivalent only to zeroed legacy gimbal inputs,
    # not to the ancestor policy on actual moving-gimbal observations.
    torch.testing.assert_close(new_actor(history), old_actor(zero_gimbal), rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(new_actor.estimate(history), old_actor.estimate(zero_gimbal), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(new_actor.decoder[0].weight[:, -5:], old_actor.decoder[0].weight[:, -5:],
                               rtol=0, atol=0)
    with pytest.raises(ValueError, match='legacy formal45'):
        restrict_actor_to_chassis(upgraded, dataset)
    wrong_generation = {**dataset, 'cycle': saved['cycle'] + 1}
    with pytest.raises(ValueError, match='paired unchanged'):
        restrict_actor_to_chassis(saved, wrong_generation)


def test_chassis_actor_keeps_full_gimbal_world_and_stair_rollout_differentiable():
    learner, start, frames = imagination_fixture(sensor_noise=True)
    learner.actor = EstimatorActor(frame_dim=35, estimator_hidden=(8,), actor_hidden=(16,))
    learner.imagination.actor = learner.actor
    learner.optimizers['actor'] = torch.optim.Adam(learner.actor.decoder.parameters())
    learner.optimizers['estimator'] = torch.optim.Adam(learner.actor.estimator.parameters())
    start['estimator_valid'].fill_(1.)
    original = {name: value.clone() for name, value in start.items()}
    result = learner.update_policy(start, frames, microbatch_size=1)
    assert result['actor/updated'] and result['estimator/updated']
    assert result['imagination/accepted_trajectories'] == 2
    assert frames.shape[-1] == 45
    for name, value in start.items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


def test_chassis_onnx_exports_only_five_by_35_and_matches_training(tmp_path):
    pytest.importorskip('onnx')
    runtime = pytest.importorskip('onnxruntime')
    actor = EstimatorActor(frame_dim=35, estimator_hidden=(8,), actor_hidden=(16,))
    path = export_onnx(actor, tmp_path / 'chassis35.onnx')
    session = runtime.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    assert session.get_inputs()[0].shape == [1, 5, 35]
    for scale in (0., 1., 5.):
        history = torch.randn(1, 5, 35) * scale
        actual = torch.from_numpy(session.run(None, {'history': history.numpy()})[0])
        torch.testing.assert_close(actual, actor(history), atol=1e-6, rtol=1e-5)


def test_branch_script_preserves_source_pair_and_loads_a_native_chassis_actor(tmp_path, monkeypatch):
    saved, dataset, _ = migrated_fixture()
    saved['replay_file'] = 'source-replay.pt'
    checkpoint, replay = tmp_path / 'source.pt', tmp_path / saved['replay_file']
    torch.save(saved, checkpoint)
    torch.save(dataset, replay)
    source_bytes = checkpoint.read_bytes()
    output = tmp_path / 'chassis-branch'
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr('sys.argv', ['upgrade_online_chassis_actor', '--checkpoint', str(checkpoint),
                                   '--output', str(output)])
    runpy.run_path(str(root / 'scripts/upgrade_online_chassis_actor.py'), run_name='__main__')
    assert checkpoint.read_bytes() == source_bytes
    assert replay.stat().st_ino == (output / 'replay-a.pt').stat().st_ino
    load_actor = runpy.run_path(str(root / 'scripts/train_world_v2.py'))['read_actor']
    actor, artifact = load_actor(output / 'actor-000032.pt', 'cpu')
    assert actor.frame_dim == 35 and artifact['actor_branch']['policy_equivalent'] is False
    assert actor(torch.zeros(1, 5, 35)).shape == (1, 6)
    assert (output / 'resume.json').is_file()
