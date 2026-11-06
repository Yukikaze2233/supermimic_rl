"""Pinned native teachers for real V2 collection, retaining their deployed input history."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from ..contracts import action_bounds_from_source
from ..chassis_observation import LEGACY_COMMANDS, chassis_frame
from ..ppo.policy import FRAME_DIM, HISTORY_LENGTH, TeacherActor

PPO_TEACHER_FORMAT = 'supermimic-native-ppo-teacher-v1'


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class PinnedPpoTeacher(nn.Module):
    """Own 5x35 PPO deterministic mean plus velocity/contact logits; no PPO updates."""

    history_length = HISTORY_LENGTH

    def __init__(self, actor, source):
        super().__init__()
        self.actor, self._source = actor, deepcopy(source)

    @property
    def source_metadata(self):
        return deepcopy(self._source)

    @torch.no_grad()
    def action_and_estimate(self, history):
        if history.ndim != 3 or history.shape[1:] != (HISTORY_LENGTH, FRAME_DIM):
            raise ValueError('Own PPO teacher requires native chronological [batch, 5, 35] history')
        estimate = self.actor.estimator(history)
        return self.actor.clip(self.actor(history, estimate)), estimate


def export_ppo_teacher_payload(checkpoint, evaluation_path, qualified_for):
    """Bind caller-reviewed native evidence to the exact evaluated PPO weights.

    Scope is declared by the caller after reviewing native results. Exporting
    does not invent a skill threshold or turn a completed evaluation into a pass.
    """
    from ..ppo.trainer import CHECKPOINT_FORMAT, PPOConfig

    checkpoint, evaluation_path = Path(checkpoint).resolve(strict=True), Path(evaluation_path).resolve(strict=True)
    evaluation_raw = evaluation_path.read_bytes()
    report = json.loads(evaluation_raw)
    digest = file_sha256(checkpoint)
    if (report.get('checkpoint_sha256') != digest
            or report.get('transferred_source') is not None
            or not isinstance(report.get('native_source'), dict)
            or report.get('steps', 0) <= 0 or report.get('simulated_seconds', 0) <= 0):
        raise ValueError('Teacher export requires native evaluation bound to this own PPO checkpoint SHA256')
    source = report['native_source']
    for name in ('asset_manifest_sha256', 'runtime_sha256', 'course_sha256', 'upstream_commit'):
        if not source.get(name):
            raise ValueError('Native evaluation lacks teacher source pin: ' + name)
    scopes = list(dict.fromkeys(qualified_for))
    if not scopes or any(not isinstance(value, str) or not value.strip() for value in scopes):
        raise ValueError('Declare the reviewed native capability scope for this teacher')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if saved.get('format') != CHECKPOINT_FORMAT:
        raise ValueError('Teacher export requires this robot own PPO checkpoint')
    config = PPOConfig.from_dict(saved['config'])
    with torch.random.fork_rng(devices=[]):
        actor = TeacherActor(config.actor_hidden, config.estimator_hidden, config.initial_std,
                             mean_squash=config.mean_squash, action_bounds=config.action_bounds,
                             chassis_command_contract=config.chassis_command_contract,
                             actor_features=config.actor_features)
    actor.load_state_dict(saved['actor'], strict=True)
    bounds = action_bounds_from_source(source)
    if (not all(torch.isfinite(value).all() for value in actor.state_dict().values())
            or not torch.equal(actor.action_bounds, torch.tensor(bounds))):
        raise ValueError('Own PPO teacher contains invalid weights or different action bounds')
    if file_sha256(checkpoint) != digest:
        raise ValueError('Own PPO checkpoint changed during teacher export')
    metadata = {'format': PPO_TEACHER_FORMAT, 'initial_checkpoint': {'path': str(checkpoint), 'sha256': digest,
        'updates': saved['updates'], 'transitions': saved['transitions']},
        'history_length': HISTORY_LENGTH, 'observation_dim': FRAME_DIM, 'action_bounds': list(bounds),
        'chassis_command_contract': config.chassis_command_contract,
        'actor_features': config.actor_features,
        'action_semantics': ('own PPO deterministic mean followed by existing per-axis hard clipping; '
                             f'mean squash {config.mean_squash}'),
        'history_semantics': ('native obs_buf policy [N,5,42], oldest first; versioned chassis_frame '
                              'preprocessing; native reset filling'),
        'estimate_semantics': 'body velocity xyz and two wheel-contact logits; no privileged labels',
        'native_source': source, 'qualified_for': scopes,
        'qualification': {'declaration': 'caller-reviewed scope; no automatic success threshold',
            'evaluation': str(evaluation_path), 'evaluation_sha256': hashlib.sha256(evaluation_raw).hexdigest(),
            'report': report},
        'transfer_scope': 'actor and estimator only; no critic, PPO optimizer or exploration sampling'}
    return {'format': PPO_TEACHER_FORMAT, 'source': metadata, 'actor': actor.state_dict(),
            'actor_hidden': list(config.actor_hidden), 'estimator_hidden': list(config.estimator_hidden),
            'mean_squash': config.mean_squash}


def load_native_teacher(path, expected_sha256, *, device, native_provenance, course_migrations=()):
    """Load pinned weights; sampling compatibility never expands qualification."""
    path = Path(path).resolve(strict=True)
    if file_sha256(path) != expected_sha256:
        raise ValueError('Native teacher checkpoint SHA256 differs')
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if payload.get('format') != PPO_TEACHER_FORMAT:
        from ..transferred_actor import TransferredActor
        if 'recipe' not in payload:
            raise ValueError('Own PPO teacher needs an exported artifact with native evaluation provenance')
        teacher = TransferredActor(payload['recipe']['policy'])
        teacher.load_initial_checkpoint(path, expected_sha256)
    else:
        from ..cli import _native_schema_source
        source = payload['source']
        if (source.get('format') != PPO_TEACHER_FORMAT or source.get('history_length') != HISTORY_LENGTH
                or source.get('observation_dim') != FRAME_DIM or not source.get('action_bounds')
                or not source.get('qualified_for')):
            raise ValueError('Own PPO teacher deployment contract differs')
        native = json.loads(_native_schema_source(native_provenance))
        compatible = native
        if course_migrations:
            from .migrations.course import compatible_course_source

            compatible = compatible_course_source(source['native_source'], native, course_migrations)
        if source['native_source'] != compatible:
            raise ValueError('Own PPO teacher evaluation and native collection source differ')
        if tuple(source['action_bounds']) != action_bounds_from_source(native):
            raise ValueError('Own PPO teacher action bounds differ from the native collection')
        with torch.random.fork_rng(devices=[]):
            # Artifacts exported before the bounded mean carry no squash field.
            actor = TeacherActor(tuple(payload['actor_hidden']), tuple(payload['estimator_hidden']),
                                 mean_squash=payload.get('mean_squash', 'none'),
                                 action_bounds=tuple(source['action_bounds']),
                                 chassis_command_contract=source.get('chassis_command_contract', LEGACY_COMMANDS),
                                 actor_features=source.get('actor_features'))
        actor.load_state_dict(payload['actor'], strict=True)
        teacher = PinnedPpoTeacher(actor, {**source, 'teacher_artifact': {'path': str(path), 'sha256': expected_sha256}})
        if file_sha256(path) != expected_sha256:
            raise ValueError('Own PPO teacher artifact changed during loading')
    return teacher.to(device).eval().requires_grad_(False)


def native_teacher_decision(teacher, environment):
    """Read the exact noisy native history used by PPO and its native evaluation.

    Collector history can reset on control ownership transitions as well as
    episode resets. It remains world-model data, not a replacement PPO input.
    """
    native = environment.adapter.env
    history = native.obs_buf['policy'].reshape(environment.num_envs, teacher.history_length, 42)
    actor = teacher.actor if isinstance(teacher, PinnedPpoTeacher) else teacher
    return teacher.action_and_estimate(chassis_frame(
        history, getattr(actor, 'chassis_command_contract', LEGACY_COMMANDS)))
