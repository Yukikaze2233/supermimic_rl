"""Fixed native PPO skill evaluation without a learned-model state layout.

The policy owns its real native observation history. This loop sends its
estimate back before stepping, and keeps the packed pre-reset transition for
physical metrics. No replay, optimizer or training-course clock is advanced.
"""
from __future__ import annotations

from time import monotonic
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from ..compact_state import named_field, _inverse_rotate
from ..evaluation import native_tracking_sample, terminal_failure_masks
from .trainer import EpisodeStatistics


@torch.no_grad()
def evaluate_native_policy(environment, policy, steps, *, on_progress=None):
    """Evaluate a NativeHistoryActor; its ignored history argument is ``None``."""
    if type(steps) is not int or steps < 1:
        raise ValueError("Native evaluation requires positive integer physical exposure")
    if environment.current is None:
        environment.reset()
    owner = environment.adapter
    fixed_suite = (callable(getattr(owner, "evaluation_begin", None))
                   and getattr(owner, "_evaluation_configuration", None) is not None)
    statistics = EpisodeStatistics(environment.num_envs, environment.device)
    masks = {}
    policy.eval()
    started = monotonic()
    try:
        if fixed_suite:
            owner.evaluation_begin(environment.schema)
            environment.current = environment._pack(owner.current)
        bounds = environment.current.state.new_tensor(environment.schema.action_bounds)
        for index in range(steps):
            before = environment.current
            sample = native_tracking_sample(environment)
            error = sample.actual - sample.target
            quaternion = F.normalize(named_field(before.state, environment.schema, "root_orientation_xyzw"), dim=-1)
            down = quaternion.new_tensor((0., 0., -1.)).expand(environment.num_envs, 3)
            gravity = _inverse_rotate(quaternion, down)
            tracking = {"forward_error_m_s": error[:, 0], "yaw_rate_error_rad_s": error[:, 1],
                        "height_error_m": error[:, 2],
                        "tilt_rad": torch.atan2(gravity[:, :2].norm(dim=-1), -gravity[:, 2])}
            action = policy(None)
            estimate = policy.estimate(None)
            if action.shape != (environment.num_envs, 6) or estimate.shape != (environment.num_envs, 5):
                raise ValueError("Native policy requires six actions and five deployed estimates")
            raw_estimate = torch.cat((estimate[:, :3], torch.zeros_like(estimate[:, :3]), estimate[:, 3:5]), -1)
            environment.set_policy_estimate(before.actor_observation, raw_estimate)
            transition = environment.step(action.clamp(-bounds, bounds))
            statistics.record(SimpleNamespace(
                reward=transition.reward, terminated=transition.terminated, truncated=transition.truncated,
                tracking=tracking, valid=transition.valid,
                numerical_invalid=getattr(getattr(owner, "env", None), "_numerical_invalid", None)))
            found = terminal_failure_masks(environment.schema, SimpleNamespace(
                end=transition.end, terminated=transition.terminated & transition.valid))
            for key, value in (found or {}).items():
                masks[key] = masks.get(key, 0) + value
            if fixed_suite:
                owner.evaluation_step(transition, sample)
            if on_progress is not None and ((index + 1) % 100 == 0 or index + 1 == steps):
                on_progress({"steps": index + 1, "transitions": (index + 1) * environment.num_envs,
                             **statistics.summary(), "elapsed_s": monotonic() - started})
    finally:
        finish = getattr(owner, "evaluation_end", None)
        if fixed_suite and callable(finish):
            finish()
    seconds = steps * environment.num_envs * environment.policy_dt_s
    valid_seconds = statistics.count * environment.policy_dt_s
    report = {**statistics.summary(), "steps": steps, "environments": environment.num_envs,
              "simulated_seconds": seconds, "elapsed_s": monotonic() - started,
              "failures_per_env_minute": 60 * statistics.terminated / seconds,
              "termination_masks": masks, "valid_simulated_seconds": valid_seconds,
              "failures_per_valid_env_minute": (60 * statistics.terminated / valid_seconds
                                                if valid_seconds else None),
              "scope": "native physics PPO policy evaluation; no teacher, world model, replay or gradient updates; "
                       "action sampling follows the caller"}
    if fixed_suite:
        report["task_suite"] = owner.evaluation_summary()
    return report
