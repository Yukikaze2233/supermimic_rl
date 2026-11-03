"""Evaluate the history student in native physics, without its world model."""
from types import SimpleNamespace
import time

import torch

from ..evaluation import native_tracking_sample, terminal_failure_masks
from ..ppo.trainer import EpisodeStatistics
from .collection import Collector


@torch.no_grad()
def evaluate_native(environment, actor, steps, *, on_progress=None):
    if steps < 1:
        raise ValueError('Native evaluation requires positive physical exposure')
    if environment.current is None:
        environment.reset()
    owner = environment.adapter
    begin = getattr(owner, 'evaluation_begin', None)
    fixed_suite = callable(begin) and getattr(owner, '_evaluation_configuration', None) is not None
    if fixed_suite:
        begin(environment.schema)
        environment.current = environment._pack(owner.current)
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

    collector = Collector(environment, decision, capacity_steps=1, action_noise_std=0., advance_training_progress=False)
    try:
        collector.collect(steps, on_progress=progress, on_transition=measured)
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
              'scope': 'native physics student only; no teacher, world model, exploration or gradient updates'}
    if fixed_suite:
        report['task_suite'] = owner.evaluation_summary()
    return report
