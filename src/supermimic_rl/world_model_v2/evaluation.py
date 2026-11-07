"""Evaluate the history student in native physics, without its world model."""
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from ..evaluation import native_tracking_sample, terminal_failure_masks
from ..ppo.trainer import EpisodeStatistics
from .collection import Collector


@torch.no_grad()
def evaluate_native(environment, actor, steps, *, on_progress=None,
                    record_dataset: Path | None = None, recording_source=None):
    if steps < 1:
        raise ValueError('Native evaluation requires positive physical exposure')
    if record_dataset is not None and Path(record_dataset).exists():
        raise FileExistsError('Use a new native evaluation dataset path')
    if environment.current is None:
        environment.reset()
    owner = environment.adapter
    platform = getattr(owner.env, 'supermimic_platform_course', None)
    platform_rows = None
    features = getattr(actor, 'features', None)
    if features is not None and features.configuration is not None:
        features.validate_reference(owner.env._terrain_course.tasks.reference.config)
    begin = getattr(owner, 'evaluation_begin', None)
    fixed_suite = callable(begin) and getattr(owner, '_evaluation_configuration', None) is not None
    if fixed_suite:
        begin(environment.schema)
        environment.current = environment._pack(owner.current)
    if platform is not None:
        # Fixed evaluation may replace the initial training routes and reset
        # every caller. Capture the resulting episode plans, not those of the
        # discarded setup birth. Later resets may change mixed-course plans.
        rows = platform.platform_rows
        if (not isinstance(rows, torch.Tensor) or rows.shape != (environment.num_envs,)
                or rows.dtype != torch.bool):
            raise ValueError('Platform evaluation requires explicit planned native rows')
        platform_rows = rows.nonzero().flatten().cpu().tolist()
    mixed_platform = getattr(getattr(platform, 'plan', None), 'sampler', None) is not None
    statistics = EpisodeStatistics(environment.num_envs, environment.device)
    masks, latest = {}, {}
    actor.eval()

    def decision(history):
        sample = native_tracking_sample(environment)
        latest['native_tracking'] = sample
        error = sample.actual - sample.target
        state = collector.layout.physical(environment.current.state)
        gravity = state.gravity_body
        latest['tracking'] = {'forward_error_m_s': error[:, 0], 'yaw_rate_error_rad_s': error[:, 1],
            'height_error_m': error[:, 2], 'tilt_rad': torch.atan2(gravity[:, :2].norm(dim=-1), -gravity[:, 2])}
        return actor(history), actor.estimate(history)

    def measured(transition):
        statistics.record(SimpleNamespace(reward=transition.reward, terminated=transition.terminated,
            truncated=transition.truncated, tracking=latest['tracking'], valid=transition.valid,
            numerical_invalid=getattr(getattr(owner, 'env', None), '_numerical_invalid', None)))
        found = terminal_failure_masks(environment.schema, SimpleNamespace(
            end=transition.end, terminated=transition.terminated & transition.valid))
        for key, value in (found or {}).items():
            masks[key] = masks.get(key, 0) + value
        if fixed_suite:
            owner.evaluation_step(transition, latest['native_tracking'])

    started = time.monotonic()

    def progress(record):
        if on_progress is not None and (record['steps'] % 100 == 0 or record['steps'] == steps):
            on_progress({**record, **statistics.summary(), 'elapsed_s': time.monotonic() - started})

    recording = record_dataset is not None
    source_rng = []
    source_config = None
    if recording:
        from .stair_source import StairScenario

        source_config = StairScenario.capture(owner.env).to_dict()
    recording_options = ({'record_terrain': True, 'record_tasks': True, 'record_stairs': True,
        'record_stair_source': True, 'record_jumps': True, 'record_recovery': True} if recording else {})
    if recording and platform is not None:
        recording_options.update(platform.recording_options())
    collector = Collector(environment, decision, capacity_steps=steps if recording else 1,
        action_noise_std=0., advance_training_progress=False,
        **recording_options)
    try:
        collector.collect(steps, on_progress=progress, on_transition=measured,
            on_source_rng=((lambda sequence, state: source_rng.append((sequence, state.cpu())))
                           if recording else None))
    finally:
        finish = getattr(owner, 'evaluation_end', None)
        if fixed_suite and callable(finish):
            finish()
    seconds = steps * environment.num_envs * environment.policy_dt_s
    valid_seconds = statistics.count * environment.policy_dt_s
    report = {**statistics.summary(), 'steps': steps, 'environments': environment.num_envs,
              'simulated_seconds': seconds, 'elapsed_s': time.monotonic() - started,
              'failures_per_env_minute': 60 * statistics.terminated / seconds, 'termination_masks': masks,
              'valid_simulated_seconds': valid_seconds,
              'failures_per_valid_env_minute': (60 * statistics.terminated / valid_seconds
                                               if valid_seconds else None),
              'scope': 'native physics policy evaluation; no teacher, world model or gradient updates; '
                       'action sampling follows the caller'}
    if fixed_suite:
        report['task_suite'] = owner.evaluation_summary()
    if platform is not None:
        summary = getattr(platform, 'summary', None)
        report['platform_diagnostics'] = (summary() if callable(summary) else {
            'status': 'unknown', 'reason': 'The native platform owner has no diagnostic summary'})
        report['platform_evaluation_scope'] = {
            'planned_env_ids': platform_rows, 'recorded_physics_ticks': recording,
            'capability_source': 'platform_diagnostics only; final snapshots are not a success rate',
            'legacy_jump_events': 'Unchanged native jump pulses and aggregate counters are not '
                                  'an all-task or selected-platform success numerator'}
        if mixed_platform:
            report['platform_evaluation_scope'].update(
                planned_env_ids_scope='initial episode plans after evaluation_begin only',
                episode_assignment_source='per-interval typed platform intent; reset may change membership')
        if fixed_suite:
            suite = dict(report['task_suite'])
            if 'jump_diagnostics' in suite:
                diagnostic = dict(suite.pop('jump_diagnostics'))
                diagnostic['platform_interpretation'] = {
                    'planned_env_ids': platform_rows, 'includes_platform_rows': bool(platform_rows),
                    'scope': 'Flat-anchor jump conditions include selected-platform rows; '
                             'they do not qualify platform flight, landing or success. '
                             'Original values are retained without recomputing outcomes.'}
                if mixed_platform:
                    diagnostic['platform_interpretation'].update(
                        includes_platform_rows=None,
                        initial_includes_platform_rows=bool(platform_rows),
                        planned_env_ids_scope='initial episode plans only; not a full-run exclusion mask',
                        scope='Mixed-episode membership must come from typed per-interval intent. '
                              'Initial plans do not partition these aggregate flat-anchor outcomes. '
                              'Original values are retained and do not qualify platform success.')
                suite['flat_anchor_jump_diagnostics'] = diagnostic
            suite['success_semantics'] = (suite.get('success_semantics', '') +
                ' Selected-platform capability is reported separately in platform_diagnostics; '
                'legacy jump events and mixed counters do not certify all-task success.')
            report['task_suite'] = suite
    if recording:
        from .dataset import save_native_dataset

        report['dataset'] = save_native_dataset(collector, record_dataset, report.copy(),
            source_rng=source_rng, stair_source_config=source_config, source=recording_source)
    return report
