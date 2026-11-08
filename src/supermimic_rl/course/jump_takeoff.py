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
        values = (getattr(self, name) for name in JumpTakeoffConfig.__dataclass_fields__)
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('Jump take-off settings must be finite and nonnegative')
        if not (self.height_tolerance_m > 0 and self.compression_depth_max_m > 0
                and self.compression_duration_s > 0 and self.nominal_extension_duration_s > 0):
            raise ValueError('Jump take-off tolerance, depth and durations must be positive')

    @classmethod
    def from_dict(cls, configuration):
        configuration = dict(configuration)
        pending = configuration.pop('automatic_pending_admission', False)
        if type(pending) is not bool:
            raise ValueError('automatic_pending_admission must be a bool')
        return (PendingJumpTakeoffConfig(**configuration) if pending else cls(**configuration))


@dataclass(frozen=True)
class PendingJumpTakeoffConfig(JumpTakeoffConfig):
    """An explicit opt-in leaves legacy trigger/config serialization unchanged."""

    automatic_pending_admission: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.automatic_pending_admission is not True:
            raise ValueError('Use JumpTakeoffConfig for legacy automatic scheduling')


def retry_pending_automatic(next_request_s, elapsed_s, response, automatic, policy_dt_s):
    """Retain automatic intent after temporary rejection without replaying its ID.

    The protocol has already consumed the ID; native/imagined selection generates
    a new one next tick. Existing interval random draws remain consumed unchanged.
    """
    from transformer_rl.formal.tasks.protocol import RequestResponse

    retry = automatic & (response == int(RequestResponse.NOT_READY))
    return torch.where(retry, elapsed_s + policy_dt_s, next_request_s)


class JumpTriggerGate:
    """Postpone due automatic requests until the native supported stance holds."""

    def __init__(self, tasks, config: JumpTakeoffConfig):
        from transformer_rl.formal.tasks.curriculum import TaskFamily

        self.tasks, self.config, self.request_family = tasks, config, int(TaskFamily.REQUEST)
        device = tasks.elapsed.device
        self.last_end_s = torch.full(tasks.elapsed.shape, -math.inf, dtype=torch.float64, device=device)
        self.was_active = torch.zeros(tasks.elapsed.shape, dtype=torch.bool, device=device)
        self.deferred = torch.zeros((), dtype=torch.long, device=device)

    def ready(self, now, ground, *, after_admission=False):
        t, c = self.tasks, self.config
        height = (t._last_world_height - ground - t.stream.current[:, 3]).abs() <= c.height_tolerance_m
        elapsed = t.elapsed if after_admission else t.elapsed + t.policy_dt_s
        return (t._last_supported & height & (elapsed >= c.min_episode_s)
                & (now - self.last_end_s >= c.cooldown_s))

    def defer(self, now, ground, *, automatic_mask=None, after_admission=False, blocked=None):
        """Postpone a due request by one interval using the selected admission clock."""
        t = self.tasks
        elapsed = t.elapsed if after_admission else t.elapsed + t.policy_dt_s
        due = ((t.family == self.request_family) & (elapsed >= t.next_request_s)
               & (t.protocol.active_id < 0))
        if automatic_mask is not None:
            if (not isinstance(automatic_mask, torch.Tensor) or automatic_mask.shape != due.shape
                    or automatic_mask.dtype != torch.bool or automatic_mask.device != due.device):
                raise ValueError('Automatic request ownership requires one bool per native row')
            due &= automatic_mask
        if blocked is not None:
            due &= ~blocked
        postpone = due & ~self.ready(now, ground, after_admission=after_admission)
        next_tick = t.elapsed + (t.policy_dt_s if after_admission else 2 * t.policy_dt_s)
        t.next_request_s.copy_(torch.where(postpone, next_tick, t.next_request_s))
        self.deferred += postpone.sum()

    def observe(self, now):
        """After the native step: remember when each request ended (absolute clock)."""
        active = self.tasks.protocol.active_id >= 0
        self.last_end_s = torch.where(self.was_active & ~active, now, self.last_end_s)
        self.was_active = active


class PendingAutomaticRequests:
    """Observe admission and protocol results inside one native task step.

    Instance-local hooks also support the old native source without its newer
    request-selection entry point. No external packet is made automatic.
    """

    def __init__(self, tasks, gate):
        self.tasks, self.gate = tasks, gate
        self._tick = None
        self._native_step, self._stream_step = tasks.step, tasks.stream.step
        self._protocol_step = tasks.protocol.step
        tasks.step, tasks.stream.step, tasks.protocol.step = self.step, self.admit, self.respond

    def step(self, now, estimate, ground, *, command=None, packet_owned=None, **kwargs):
        if packet_owned is not None:
            from .request_selection import validate_packet_owned

            validate_packet_owned(packet_owned, packet=command,
                                  count=len(self.tasks.elapsed), device=self.tasks.elapsed.device)
        automatic = (torch.ones_like(self.tasks.elapsed, dtype=torch.bool) if command is None
                     else torch.zeros_like(self.tasks.elapsed, dtype=torch.bool) if packet_owned is None
                     else ~packet_owned)
        if packet_owned is not None:
            kwargs['packet_owned'] = packet_owned
        blocked = kwargs.get('blocked_by_stair')
        if blocked is not None:
            blocked = torch.as_tensor(blocked, dtype=torch.bool, device=self.tasks.elapsed.device)
        previous = self._tick
        self._tick = (now, ground, automatic, blocked)
        try:
            result = self._native_step(now, estimate, ground, command=command, **kwargs)
            self.gate.observe(now)
            return result
        finally:
            self._tick = previous

    def admit(self, target):
        result = self._stream_step(target)
        if self._tick is not None:
            now, ground, automatic, blocked = self._tick
            self.gate.defer(now, ground, automatic_mask=automatic, blocked=blocked, after_admission=True)
        return result

    def respond(self, **kwargs):
        result = self._protocol_step(**kwargs)
        if self._tick is not None:
            automatic = self._tick[2] & (kwargs['request_id'] >= 0)
            self.tasks.next_request_s.copy_(retry_pending_automatic(
                self.tasks.next_request_s, self.tasks.elapsed, result.response,
                automatic, self.tasks.policy_dt_s))
        return result


def install_jump_takeoff(native, configuration: dict) -> dict:
    """Replace the reference program and gate automatic requests; return the receipt."""
    from transformer_rl.formal.tasks.jump import JumpReferenceProgram

    config = JumpTakeoffConfig.from_dict(configuration)
    tasks = native._terrain_course.tasks
    reference = replace(tasks.reference.config, compression_depth_max_m=config.compression_depth_max_m,
                        compression_duration_s=config.compression_duration_s,
                        nominal_extension_duration_s=config.nominal_extension_duration_s)
    tasks.reference = JumpReferenceProgram(reference)
    gate = JumpTriggerGate(tasks, config)
    native_step = tasks.step

    def step(now, estimate, ground, *, command=None, packet_owned=None, **kwargs):
        if packet_owned is not None:
            from .request_selection import validate_packet_owned

            validate_packet_owned(packet_owned, packet=command,
                                  count=len(tasks.elapsed), device=tasks.elapsed.device)
            gate.defer(now, ground, automatic_mask=~packet_owned)
            kwargs['packet_owned'] = packet_owned
        if command is None:
            gate.defer(now, ground)
        result = native_step(now, estimate, ground, command=command, **kwargs)
        gate.observe(now)
        return result

    if isinstance(config, PendingJumpTakeoffConfig):
        PendingAutomaticRequests(tasks, gate)
    else:
        tasks.step = step
    native.supermimic_jump_trigger = gate
    receipt = {**asdict(config), 'reference': asdict(reference),
            'trigger': 'automatic requests wait for the native supported stance, episode time and cooldown; '
                       'external requests are not gated'}
    if isinstance(config, PendingJumpTakeoffConfig):
        receipt['automatic_scheduling'] = {
            'version': 'admitted_height_pending_retry_v1',
            'gate_height': 'current_tick_admitted',
            'not_ready': 'retry_next_policy_tick_with_new_id',
            'random_draws': 'unchanged_native_interval_vector_each_automatic_tick',
            'external_requests': 'unchanged',
        }
    return receipt
