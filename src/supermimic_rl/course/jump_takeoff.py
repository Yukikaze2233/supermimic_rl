"""SCUT-style jump take-off task: stability-gated triggering and a crouch-then-long-push reference.

SCUT's jump state machine enters PUSH only when the robot is settled (episode
time, height and velocity tracking, cooldown) and drives a long push stroke.
The native course already has the request protocol, phases and reference
program; this module changes only:

* triggering: a due automatic request is postponed until the robot is in the
  native supported stance (both wheels loaded, quiet, height within the native
  supported-start tolerance), past ``min_episode_s`` and ``cooldown_s`` after
  the previous request ended. Externally supplied requests are not gated.
* reference: a deeper, longer crouch and extension, so the take-off needs about
  half the acceleration of the native 4 cm / 0.1 s profile and its height and
  vertical-speed tracking rewards stay informative through the whole push.

It defines the task, so training, evaluation and world-model collection all
install it (the assist in ``jump_assist`` is training-only).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

import torch


@dataclass(frozen=True)
class JumpTakeoffConfig:
    min_episode_s: float = 1.
    cooldown_s: float = 1.
    height_tolerance_m: float = .015       # native supported-start tolerance
    compression_depth_max_m: float = .10
    compression_duration_s: float = .30
    nominal_extension_duration_s: float = .20

    def __post_init__(self):
        values = asdict(self).values()
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('Jump take-off settings must be finite and nonnegative')
        if not (self.height_tolerance_m > 0 and self.compression_depth_max_m > 0
                and self.compression_duration_s > 0 and self.nominal_extension_duration_s > 0):
            raise ValueError('Jump take-off tolerance, depth and durations must be positive')


class JumpTriggerGate:
    """Postpone due automatic requests until the native supported stance holds."""

    def __init__(self, tasks, config: JumpTakeoffConfig):
        from transformer_rl.formal.tasks.curriculum import TaskFamily

        self.tasks, self.config, self.request_family = tasks, config, int(TaskFamily.REQUEST)
        device = tasks.elapsed.device
        self.last_end_s = torch.full(tasks.elapsed.shape, -math.inf, dtype=torch.float64, device=device)
        self.was_active = torch.zeros(tasks.elapsed.shape, dtype=torch.bool, device=device)
        self.deferred = torch.zeros((), dtype=torch.long, device=device)

    def ready(self, now, ground):
        t, c = self.tasks, self.config
        height = (t._last_world_height - ground - t.stream.current[:, 3]).abs() <= c.height_tolerance_m
        return (t._last_supported & height & (t.elapsed + t.policy_dt_s >= c.min_episode_s)
                & (now - self.last_end_s >= c.cooldown_s))

    def defer(self, now, ground):
        """Before the native step: a request that would become due now waits one more interval."""
        t = self.tasks
        due = ((t.family == self.request_family) & (t.elapsed + t.policy_dt_s >= t.next_request_s)
               & (t.protocol.active_id < 0))
        postpone = due & ~self.ready(now, ground)
        t.next_request_s.copy_(torch.where(postpone, t.elapsed + 2 * t.policy_dt_s, t.next_request_s))
        self.deferred += postpone.sum()

    def observe(self, now):
        """After the native step: remember when each request ended (absolute clock)."""
        active = self.tasks.protocol.active_id >= 0
        self.last_end_s = torch.where(self.was_active & ~active, now, self.last_end_s)
        self.was_active = active


def install_jump_takeoff(native, configuration: dict) -> dict:
    """Replace the reference program and gate automatic requests; return the receipt."""
    from transformer_rl.formal.tasks.jump import JumpReferenceProgram

    config = JumpTakeoffConfig(**configuration)
    tasks = native._terrain_course.tasks
    reference = replace(tasks.reference.config, compression_depth_max_m=config.compression_depth_max_m,
                        compression_duration_s=config.compression_duration_s,
                        nominal_extension_duration_s=config.nominal_extension_duration_s)
    tasks.reference = JumpReferenceProgram(reference)
    gate = JumpTriggerGate(tasks, config)
    native_step = tasks.step

    def step(now, estimate, ground, *, command=None, **kwargs):
        if command is None:
            gate.defer(now, ground)
        result = native_step(now, estimate, ground, command=command, **kwargs)
        gate.observe(now)
        return result

    tasks.step = step
    native.supermimic_jump_trigger = gate
    return {**asdict(config), 'reference': asdict(reference),
            'trigger': 'automatic requests wait for the native supported stance, episode time and cooldown; '
                       'external requests are not gated'}
