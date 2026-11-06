"""Native-curve parity, estimator isolation and portable actor-feature semantics."""
from dataclasses import asdict, replace
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.actor_features import ActorFeatures, JUMP_REFERENCE_FEATURES, ONNX_FEATURE_METADATA
from supermimic_rl.ppo.policy import TeacherActor
from supermimic_rl.ppo.trainer import PPOTrainer, load_actor
from supermimic_rl.world_model_v2.policy import EstimatorActor
from supermimic_rl.world_model_v2.student_transfer import max_action_difference, ppo_actor_from_student
from transformer_rl.formal.tasks.jump import JumpReferenceConfig, JumpReferenceProgram
from support.ppo import TargetEnvironment, config


def feature_config(reference=None):
    reference = reference or JumpReferenceConfig(compression_depth_max_m=.1,
        compression_duration_s=.3, nominal_extension_duration_s=.2)
    return {'version': JUMP_REFERENCE_FEATURES, 'reference': asdict(reference)}


def history():
    value = torch.randn(6, 5, 35)
    value[..., 3] = 1.6
    value[..., 28:33] = 0
    value[0, :, 28] = 1
    value[1:, :, 32] = 1
    value[..., 33] = .5
    value[..., 34] = torch.tensor([0., .15, .35, .55, 1.])
    return value


@pytest.mark.parametrize('reference', [JumpReferenceConfig(), JumpReferenceConfig(
    compression_depth_max_m=.1, compression_duration_s=.3, nominal_extension_duration_s=.2)])
def test_height_feature_matches_native_curve_including_support_limited_extension(reference):
    baseline, rise, clock = torch.meshgrid(
        torch.linspace(reference.support_height_min_m, reference.support_height_max_m, 7),
        torch.tensor([.03, .1, .4]), torch.linspace(0, 2, 121), indexing='ij')
    frames = torch.zeros(baseline.numel(), 5, 35)
    frames[..., 3] = baseline.flatten()[:, None] * 5
    frames[..., 32] = 1
    frames[..., 33] = rise.flatten()[:, None] * 5
    frames[..., 34] = clock.flatten()[:, None]
    raw = frames.clone()
    flat = frames.reshape(-1, 35)
    # Native task inputs precede the float32 observation scale/unscale round trip.
    native = JumpReferenceProgram(reference).sample(baseline.flatten().repeat_interleave(5),
        rise.flatten().repeat_interleave(5),
        flat[:, 34].double(), torch.zeros(len(flat)), active=torch.ones(len(flat), dtype=torch.bool))
    changed = ActorFeatures(feature_config(reference))(frames)
    torch.testing.assert_close(changed[..., 3].flatten(), native.height_world_z_m * 5, rtol=0, atol=5e-7)
    torch.testing.assert_close(changed[..., 4:], raw[..., 4:], rtol=0, atol=0)
    torch.testing.assert_close(changed[..., :3], raw[..., :3], rtol=0, atol=0)
    torch.testing.assert_close(frames, raw, rtol=0, atol=0)
    assert native.extension_time_limited_by_support_range.any()


def test_only_decoder_features_change_and_student_keeps_observation_gradients():
    raw = history()
    student = EstimatorActor(5, (8,), (16,), frame_dim=35, actor_features=feature_config())
    plain = EstimatorActor(5, (8,), (16,), frame_dim=35)
    plain.load_state_dict(student.state_dict())
    torch.testing.assert_close(student.estimate(raw), plain.estimate(raw), rtol=0, atol=0)
    torch.testing.assert_close(student(raw[:1]), plain(raw[:1]), rtol=0, atol=0)
    ppo = ppo_actor_from_student(student)
    torch.testing.assert_close(ppo.estimator(raw), student.estimate(raw), rtol=0, atol=0)
    torch.testing.assert_close(ppo(raw), student(raw), rtol=0, atol=0)
    assert max_action_difference(student, ppo, samples=100) == 0
    raw.requires_grad_()
    student(raw).sum().backward()
    assert torch.isfinite(raw.grad).all() and raw.grad[..., 4:28].abs().sum() > 0
    assert all(p.grad is None for p in student.estimator.parameters())
    assert any(p.grad is not None for p in student.decoder.parameters())
    ppo.zero_grad()
    ppo(raw).sum().backward()
    assert all(p.grad is None for p in ppo.estimator.parameters())


@pytest.mark.parametrize('kind', ['ppo', 'student'])
def test_onnx_includes_features_and_preserves_raw_history_interface(tmp_path, kind):
    onnx = pytest.importorskip('onnx')
    ort = pytest.importorskip('onnxruntime')
    if kind == 'ppo':
        from supermimic_rl.ppo.evaluation import export_onnx
        actor = TeacherActor((16,), (8,), actor_features=feature_config())
    else:
        from supermimic_rl.world_model_v2.policy import export_onnx
        actor = EstimatorActor(5, (8,), (16,), frame_dim=35, actor_features=feature_config())
    path = export_onnx(actor, tmp_path / f'{kind}.onnx')
    model = onnx.load(str(path))
    metadata = {x.key:x.value for x in model.metadata_props}
    assert json.loads(metadata[ONNX_FEATURE_METADATA]) == feature_config()
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    for frame in history():
        raw = frame[None]
        actual = session.run(None, {'history':raw.numpy()})[0]
        torch.testing.assert_close(torch.from_numpy(actual), actor(raw), rtol=1e-5, atol=1e-6)


def test_ppo_activation_preserves_complete_learning_state_and_roundtrips(tmp_path):
    prior = PPOTrainer(TargetEnvironment(num_envs=4), config(updates=1), tmp_path / 'old')
    prior.train()
    saved = prior.state_dict()
    saved['config'].pop('actor_features')  # A real legacy checkpoint has no new config field.
    path = tmp_path / 'legacy.pt'
    torch.save(saved, path)
    environment = TargetEnvironment(num_envs=4)
    native_reference = JumpReferenceProgram(JumpReferenceConfig(**feature_config()['reference']))
    environment.env = SimpleNamespace(_terrain_course=SimpleNamespace(
        tasks=SimpleNamespace(reference=native_reference)))
    trainer = PPOTrainer(environment, replace(config(updates=2), actor_features=feature_config()),
                         tmp_path / 'new', resume=path)
    actual = trainer.state_dict()
    for name in ('actor', 'critic', 'optimizer', 'estimator_optimizer', 'torch_rng',
                 'learning_rate', 'updates', 'transitions'):
        torch.testing.assert_close(actual[name], saved[name], rtol=0, atol=0)
    assert trainer.resume_migrations[-1]['actor_features'] == feature_config()
    trainer.train()
    actor = load_actor(tmp_path / 'new/checkpoint.pt')
    sample = history()
    torch.testing.assert_close(actor(sample), trainer.actor(sample), rtol=0, atol=0)
    assert actor.features.configuration == feature_config()
    assert trainer.updates == 2


def test_reference_metadata_must_match_the_native_curve():
    features = ActorFeatures(feature_config())
    with pytest.raises(ValueError, match='native jump reference differ'):
        features.validate_reference(JumpReferenceConfig())
    with pytest.raises(ValueError, match='complete native jump reference'):
        ActorFeatures({'version': JUMP_REFERENCE_FEATURES, 'reference': {}})


def test_world_resume_activates_features_without_rewriting_recorded_data():
    from supermimic_rl.world_model_v2.online import OnlineConfig, OnlineTrainer, CHECKPOINT_FORMAT
    from supermimic_rl.world_model_v2.records import MODEL_CONTRACT

    old = OnlineConfig(actor_frame_dim=35, record_tasks=True, record_stair_source=True,
                       record_jumps=True, course_coordinate=0)
    saved = {'format': CHECKPOINT_FORMAT, 'model_contract': MODEL_CONTRACT, 'schema_fingerprint': 'same',
             'config': asdict(old), 'cycle': 1, 'teacher_source': None, 'stair_scenario': None,
             'jump_scenario': None, 'recovery_scenario': None}
    saved['config'].pop('actor_features')
    trainer = object.__new__(OnlineTrainer)
    trainer.config = replace(old, actor_features=feature_config())
    trainer.env = SimpleNamespace(schema=SimpleNamespace(fingerprint='same'))
    trainer.teacher_source = trainer.stair_scenario = trainer.jump_scenario = trainer.recovery_scenario = None
    before = deepcopy(saved)
    trainer._check_saved(saved)
    assert trainer.actor_features_activated and saved == before
    trainer.config = replace(trainer.config, model_updates=old.model_updates + 1)
    with pytest.raises(ValueError, match='training recipe'):
        trainer._check_saved(saved)


def test_imagined_update_keeps_original_anchor_and_optimizer_state():
    from support.world_v2.jump_imagination import fixture

    learner, start, frames = fixture()
    learner.set_anchor(learner.actor, torch.full((6,), .2))
    learner.update_policy(start, frames, microbatch_size=2)
    saved = deepcopy(learner.state_dict())
    saved['anchor'].pop('actor_features')
    expected_anchor = learner.anchor[0](frames).detach()
    learner.actor.features = ActorFeatures(feature_config(learner.imagination.jump_task.config.reference))
    learner.load_state_dict(saved)
    assert learner.actor.features.configuration is not None
    assert learner.anchor[0].features.configuration is None
    torch.testing.assert_close(learner.anchor[0](frames), expected_anchor, rtol=0, atol=0)
    torch.testing.assert_close(learner.state_dict()['optimizers'], saved['optimizers'], rtol=0, atol=0)
    result = learner.update_policy(start, frames, microbatch_size=2)
    assert result['actor/updated'] == 1 and result['actor/skipped'] == 0
    # A future reference policy with enabled features also round-trips them.
    learner.set_anchor(learner.actor, torch.full((6,), .2))
    learner.load_state_dict(deepcopy(learner.state_dict()))
    assert learner.anchor[0].features.configuration == learner.actor.features.configuration
