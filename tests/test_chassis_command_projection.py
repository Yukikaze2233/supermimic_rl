"""The 35D controller must see the same horizontal target its reward uses."""
from copy import deepcopy
from dataclasses import asdict, replace
import math
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.chassis_observation import (
    HEADING_COMMANDS, LEGACY_COMMANDS, ONNX_COMMAND_METADATA, chassis_frame,
)
from supermimic_rl.ppo.environment import CourseEnvironment
from supermimic_rl.ppo.trainer import PPOTrainer, load_actor
from supermimic_rl.world_model_v2.policy import EstimatorActor, export_onnx
from supermimic_rl.world_model_v2.student_transfer import ppo_actor_from_student
from supermimic_rl.world_model_v2.teacher import PinnedPpoTeacher, native_teacher_decision
from support.ppo import TargetEnvironment, config


def frames():
    result = torch.zeros(4, 5, 45, dtype=torch.float64)
    result[..., 3] = 1.6
    result[..., 9] = -1.
    result[..., 28] = 1.
    result[..., 36] = 1.
    result[..., 37] = .5
    direction = torch.tensor([0., math.pi, 0., 0.], dtype=result.dtype)
    yaw = torch.linspace(-.4, .4, 5, dtype=result.dtype)[None].expand(4, -1).clone()
    yaw[2] = math.pi / 2
    result[..., 38] = direction.sin()[:, None]
    result[..., 39] = direction.cos()[:, None]
    result[..., 40], result[..., 41] = yaw.sin(), yaw.cos()
    result[3, :, 35:37] = torch.tensor([1., 0.])
    result[3, :, :2] = torch.tensor([.3, -.2])
    return result


def test_opposite_hidden_targets_become_distinct_and_ordinary_channels_are_exact():
    raw = frames()
    before = raw.clone()
    legacy = chassis_frame(raw, LEGACY_COMMANDS)
    actual = chassis_frame(raw, HEADING_COMMANDS)
    assert torch.equal(legacy[0], legacy[1])
    torch.testing.assert_close(actual[0, :, :2], -actual[1, :, :2])
    torch.testing.assert_close(actual[2, :, :2], torch.tensor([0., .5], dtype=raw.dtype).expand(5, -1))
    assert torch.equal(actual[3], legacy[3])
    assert torch.equal(actual[..., 2:], raw[..., 2:35])
    assert torch.equal(raw, before)
    assert torch.equal(actual, chassis_frame(raw[..., :42], HEADING_COMMANDS))
    assert torch.equal(actual, chassis_frame(actual, HEADING_COMMANDS))  # Prepared 35D is not rotated twice.


def test_heading_rotation_matches_native_reward_for_tilted_chassis():
    from transformer_rl.formal.lab_env.gimbal import GimbalMeasurement
    from transformer_rl.formal.tasks.gimbal_course import GimbalRewardReference, gimbal_tracking_densities
    from supermimic_rl.world_model_v2 import so3

    raw = frames()[:3, -1]
    rotation = so3.exp(torch.tensor([[.3, -.5, 1.], [-.4, .7, -2.], [.2, .3, -.5]], dtype=raw.dtype))
    root_yaw = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
    yaw = torch.atan2(raw[:, 40], raw[:, 41])
    measure = GimbalMeasurement(None, None, torch.stack((yaw, yaw * 0), -1), None, None,
                                root_yaw, None, None)
    command = chassis_frame(raw, HEADING_COMMANDS)[:, :2]
    # A horizontal heading basis derived independently from the physical body pose.
    forward = torch.cat((rotation[:, :2, 0], torch.zeros(3, 1, dtype=raw.dtype)), -1)
    forward = forward / forward.norm(dim=-1, keepdim=True)
    lateral = torch.linalg.cross(torch.tensor([0., 0., 1.], dtype=raw.dtype).expand_as(forward), forward)
    velocity = command[:, :1] * forward + command[:, 1:] * lateral
    reference = GimbalRewardReference(torch.ones(3, dtype=torch.bool), measure.task_heading,
        raw[:, 37], torch.atan2(raw[:, 38], raw[:, 39]))
    _, _, errors = gimbal_tracking_densities(reference, velocity)
    torch.testing.assert_close(errors['gimbal_velocity_m_s'], torch.zeros(3, dtype=raw.dtype), atol=1e-14, rtol=0)


def test_command_projection_keeps_imagined_yaw_and_speed_gradients():
    raw = frames()

    def project(yaw, speed):
        value = raw.clone()
        value[..., 37] = speed
        value[..., 40], value[..., 41] = yaw.sin(), yaw.cos()
        return chassis_frame(value, HEADING_COMMANDS)

    yaw = torch.full((4, 5), .4, dtype=torch.float64, requires_grad=True)
    speed = torch.full((4, 5), .5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(project, (yaw, speed))
    derivative = torch.autograd.grad(project(yaw, speed)[..., 0].sum(), yaw)[0]
    assert derivative[:3].abs().min() > 0
    assert derivative[3].eq(0).all()


def test_ppo_student_teacher_and_native_evaluation_share_the_history_projection():
    raw = frames().float()
    native = SimpleNamespace(num_envs=4, device='cpu',
        obs_buf={'policy': raw[..., :42].flatten(-2), 'critic': torch.zeros(4, 182)})
    adapter = SimpleNamespace(env=native, _health=None)
    env = CourseEnvironment(adapter)
    env.set_chassis_command_contract(HEADING_COMMANDS)
    history = env._observation().history
    student = EstimatorActor(frame_dim=35, estimator_hidden=(8,), actor_hidden=(16,),
                             chassis_command_contract=HEADING_COMMANDS)
    actor = ppo_actor_from_student(student)
    wrapped_environment = SimpleNamespace(num_envs=4, adapter=adapter)
    script = Path(__file__).resolve().parents[1] / 'scripts/evaluate_ppo_skills.py'
    wrapper = runpy.run_path(str(script))['NativeHistoryActor'](actor, wrapped_environment)
    assert torch.equal(wrapper._history(), history)
    torch.testing.assert_close(student._flat(raw), history.flatten(-2), rtol=0, atol=0)
    with torch.no_grad():
        expected = student(raw)
        torch.testing.assert_close(wrapper(raw), expected)
        teacher = PinnedPpoTeacher(actor, {})
        action, estimate = native_teacher_decision(teacher, wrapped_environment)
        torch.testing.assert_close(action, expected)
        torch.testing.assert_close(estimate, student.estimate(raw))


def test_onnx_keeps_35_inputs_and_records_preprocessing_contract(tmp_path):
    onnx = pytest.importorskip('onnx')
    from onnx.reference import ReferenceEvaluator
    from supermimic_rl.ppo.evaluation import export_onnx as export_ppo

    student = EstimatorActor(frame_dim=35, estimator_hidden=(8,), actor_hidden=(16,),
                             chassis_command_contract=HEADING_COMMANDS).eval()
    raw = frames()[:1].float()
    prepared = chassis_frame(raw, HEADING_COMMANDS)
    for name, actor, export in [('student', student, export_onnx),
                                ('ppo', ppo_actor_from_student(student), export_ppo)]:
        path = export(actor, tmp_path / (name + '.onnx'))
        model = onnx.load(str(path))
        assert [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim] == [1, 5, 35]
        assert {p.key: p.value for p in model.metadata_props}[ONNX_COMMAND_METADATA] == HEADING_COMMANDS
        session = ReferenceEvaluator(model)
        actual = session.run(None, {'history': prepared.numpy()})[0]
        with torch.no_grad():
            torch.testing.assert_close(torch.tensor(actual), student(raw), atol=1e-6, rtol=1e-6)


def test_ppo_full_resume_changes_only_command_semantics_and_retains_learning_state(tmp_path):
    seed = PPOTrainer(TargetEnvironment(num_envs=4), config(updates=1), tmp_path / 'seed')
    seed.train()
    path = tmp_path / 'seed/checkpoint.pt'
    saved = torch.load(path, weights_only=True)
    resumed = PPOTrainer(TargetEnvironment(num_envs=4),
        replace(seed.config, updates=2, chassis_command_contract=HEADING_COMMANDS), tmp_path / 'new', resume=path)
    actual = resumed.state_dict()
    for key in ('actor', 'critic', 'optimizer', 'estimator_optimizer', 'learning_rate', 'updates', 'transitions'):
        torch.testing.assert_close(actual[key], saved[key], rtol=0, atol=0)
    assert actual['resume_migrations'][-1]['learner_rng_preserved']
    assert load_actor(path).chassis_command_contract == LEGACY_COMMANDS
    resumed.save(tmp_path / 'new.pt')
    assert load_actor(tmp_path / 'new.pt').chassis_command_contract == HEADING_COMMANDS


def test_world_resume_projects_existing_raw_replay_without_recipe_change():
    from supermimic_rl.world_model_v2.online import OnlineConfig, OnlineTrainer, CHECKPOINT_FORMAT
    from supermimic_rl.world_model_v2.records import MODEL_CONTRACT

    old = OnlineConfig(actor_frame_dim=35)
    saved = {'format': CHECKPOINT_FORMAT, 'model_contract': MODEL_CONTRACT, 'schema_fingerprint': 'same',
             'config': asdict(old), 'cycle': 1, 'teacher_source': None, 'stair_scenario': None,
             'jump_scenario': None, 'recovery_scenario': None}
    trainer = object.__new__(OnlineTrainer)
    trainer.config = replace(old, chassis_command_contract=HEADING_COMMANDS)
    trainer.env = SimpleNamespace(schema=SimpleNamespace(fingerprint='same'))
    trainer.teacher_source = trainer.stair_scenario = trainer.jump_scenario = trainer.recovery_scenario = None
    before = deepcopy(saved)
    trainer._check_saved(saved)
    assert trainer.chassis_commands_activated and saved == before
    trainer.config = replace(trainer.config, model_updates=old.model_updates + 1)
    with pytest.raises(ValueError, match='training recipe'):
        trainer._check_saved(saved)
