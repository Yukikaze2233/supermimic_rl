"""Own PPO teachers preserve deployed native history, clipping, estimates and provenance."""
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.contracts import ACTION_BOUNDS
from supermimic_rl.ppo.policy import TeacherActor
from supermimic_rl.ppo.trainer import CHECKPOINT_FORMAT, PPOConfig
from supermimic_rl.world_model_v2.teacher import (
    export_ppo_teacher_payload, file_sha256, load_native_teacher, native_teacher_decision,
)

ROOT = Path(__file__).resolve().parents[1]


def fixture(tmp_path):
    config = PPOConfig(actor_hidden=(16,), estimator_hidden=(8,), critic_hidden=(16,))
    actor = TeacherActor(config.actor_hidden, config.estimator_hidden)
    with torch.no_grad():
        actor.network[-1].bias.copy_(torch.tensor([5., -6., .5, -.5, 20., -20.]))
        actor.estimator.network[-1].bias[-2:] = torch.tensor([2.3, -1.7])
    checkpoint = tmp_path / 'ppo.pt'
    torch.save({'format': CHECKPOINT_FORMAT, 'config': asdict(config), 'actor': actor.state_dict(),
                'updates': 100, 'transitions': 100 * 4096 * 24}, checkpoint)
    source = {'asset_manifest_sha256': 'a' * 64, 'runtime_sha256': 'b' * 64, 'course_sha256': 'c' * 64,
              'upstream_commit': 'd' * 40, 'observer_contract': 'full45_physical_sensor_history_v1',
              'physics_backend': 'physx', 'physics_dt_s': .005, 'policy_dt_s': .02}
    evaluation = tmp_path / 'evaluation.json'
    evaluation.write_text(json.dumps({'checkpoint': str(checkpoint), 'checkpoint_sha256': file_sha256(checkpoint),
        'native_source': source, 'steps': 100, 'environments': 4, 'simulated_seconds': 8., 'terminations': 0}))
    return actor, checkpoint, evaluation, source


def test_own_teacher_preserves_native_mean_clipping_logits_and_history_resets(tmp_path):
    actor, checkpoint, evaluation, source = fixture(tmp_path)
    payload = export_ppo_teacher_payload(checkpoint, evaluation, ['foundation'])
    artifact = tmp_path / 'teacher.pt'
    torch.save(payload, artifact)
    teacher = load_native_teacher(artifact, file_sha256(artifact), device='cpu', native_provenance=source)
    assert not teacher.training and all(not parameter.requires_grad for parameter in teacher.parameters())
    native = torch.randn(4, 5, 42)
    native[..., 35:] = float('nan')
    environment = SimpleNamespace(num_envs=4, adapter=SimpleNamespace(env=SimpleNamespace(
        obs_buf={'policy': native.flatten(-2)})))
    for reset in (False, True):
        if reset:
            native[0] = native[0, -1].clone()[None]
        history = native[..., :35]
        actual_action, actual_estimate = native_teacher_decision(teacher, environment)
        torch.testing.assert_close(actual_action, actor.clip(actor(history)), rtol=0, atol=0)
        torch.testing.assert_close(actual_estimate, actor.estimator(history), rtol=0, atol=0)
        assert (actual_action.abs() <= torch.tensor(ACTION_BOUNDS)).all()
        assert actual_estimate[:, -2:].max() > 1 and actual_estimate[:, -2:].min() < 0
    metadata = teacher.source_metadata
    metadata['qualified_for'].append('not-qualified')
    assert teacher.source_metadata['qualified_for'] == ['foundation']
    assert payload.keys() == {'format', 'source', 'actor', 'actor_hidden', 'estimator_hidden', 'mean_squash'}
    with pytest.raises(ValueError, match='5, 35'):
        teacher.action_and_estimate(torch.zeros(4, 5, 45))


def test_teacher_refuses_missing_evaluation_binding_changed_physics_and_raw_ppo(tmp_path):
    _, checkpoint, evaluation, source = fixture(tmp_path)
    with pytest.raises(ValueError, match='exported artifact'):
        load_native_teacher(checkpoint, file_sha256(checkpoint), device='cpu', native_provenance=source)
    payload = export_ppo_teacher_payload(checkpoint, evaluation, ['foundation'])
    artifact = tmp_path / 'teacher.pt'
    torch.save(payload, artifact)
    with pytest.raises(ValueError, match='SHA256 differs'):
        load_native_teacher(artifact, '0' * 64, device='cpu', native_provenance=source)
    with pytest.raises(ValueError, match='native collection source differ'):
        load_native_teacher(artifact, file_sha256(artifact), device='cpu',
                            native_provenance={**source, 'asset_manifest_sha256': 'f' * 64})
    with pytest.raises(ValueError, match='reviewed native capability scope'):
        export_ppo_teacher_payload(checkpoint, evaluation, [])
    report = json.loads(evaluation.read_text())
    for field, value in (('checkpoint_sha256', None), ('steps', 0), ('transferred_source', {})):
        evaluation.write_text(json.dumps({**report, field: value}))
        with pytest.raises(ValueError, match='native evaluation bound'):
            export_ppo_teacher_payload(checkpoint, evaluation, ['foundation'])


def test_legacy_teacher_source_metadata_remains_unchanged(tmp_path, monkeypatch):
    from supermimic_rl import transferred_actor

    source = {'format': 'existing-cp18026', 'initial_checkpoint': {'sha256': 'legacy'}}

    class Legacy(torch.nn.Module):
        history_length = 5

        def __init__(self, config):
            super().__init__()
            assert config == {'fixture': True}
            self.source_metadata = source

        def load_initial_checkpoint(self, path, expected):
            assert file_sha256(path) == expected

    monkeypatch.setattr(transferred_actor, 'TransferredActor', Legacy)
    checkpoint = tmp_path / 'legacy.pt'
    torch.save({'recipe': {'policy': {'fixture': True}}}, checkpoint)
    loaded = load_native_teacher(checkpoint, file_sha256(checkpoint), device='cpu', native_provenance={})
    assert loaded.source_metadata is source


@pytest.mark.parametrize('mutate_checkpoint', [False, True])
def test_ppo_evaluation_pins_exact_weight_bytes_and_native_source(tmp_path, monkeypatch, mutate_checkpoint):
    from supermimic_rl import cli
    from supermimic_rl.ppo import evaluation as native_evaluation

    _, checkpoint, _, source = fixture(tmp_path)
    closed = []
    environment = SimpleNamespace(device='cpu', adapter=SimpleNamespace(provenance=source),
                                  close=lambda: closed.append(True))
    program = runpy.run_path(str(ROOT / 'scripts/train_ppo.py'))
    evaluate = program['evaluate']
    evaluate.__globals__['_native_environment'] = lambda *args: environment
    monkeypatch.setattr(cli, 'native_logs', lambda *args: nullcontext())

    def run(*args, **kwargs):
        if mutate_checkpoint:
            with checkpoint.open('ab') as stream:
                stream.write(b'changed-during-evaluation')
        return {'steps': 100, 'simulated_seconds': 8.}

    monkeypatch.setattr(native_evaluation, 'evaluate', run)
    output = tmp_path / 'native-evaluation'
    args = SimpleNamespace(output=output, factory_config=tmp_path / 'factory.json', factory='fixture',
                           checkpoint=checkpoint, steps=100)
    if mutate_checkpoint:
        with pytest.raises(ValueError, match='changed during native evaluation'):
            evaluate(args)
        assert (output / 'failure-traceback.txt').is_file()
        assert not (output / 'evaluation.json').exists()
    else:
        evaluate(args)
        report = json.loads((output / 'evaluation.json').read_text())
        assert report['checkpoint_sha256'] == file_sha256(checkpoint)
        assert report['native_source'] == source
    assert closed == [True]


def test_export_command_creates_the_artifact_and_hashed_manifest(tmp_path, monkeypatch):
    _, checkpoint, evaluation, source = fixture(tmp_path)
    output = tmp_path / 'own-teacher.pt'
    monkeypatch.setattr('sys.argv', ['export_ppo_teacher', '--checkpoint', str(checkpoint),
        '--evaluation', str(evaluation), '--qualified-for', 'foundation', '--output', str(output)])
    runpy.run_path(str(ROOT / 'scripts/export_ppo_teacher.py'), run_name='__main__')
    manifest = json.loads(output.with_suffix('.manifest.json').read_text())
    assert manifest['sha256'] == file_sha256(output)
    teacher = load_native_teacher(output, manifest['sha256'], device='cpu', native_provenance=source)
    assert teacher.source_metadata['initial_checkpoint']['sha256'] == file_sha256(checkpoint)
