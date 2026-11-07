"""Bind shared request selection to a native task's post-admission entry point.

Install before the take-off gate. A caller-owned packet is local to one step;
its persistent intent belongs in the recorded platform task, not this adapter.
"""
from __future__ import annotations

from types import MethodType

import torch

from .jump_physics import observe_jump_physics
from .request_selection import select_jump_request, validate_packet_owned


class NativeJumpTaskAdapter:
    """Bind shared task rules before the existing take-off and flight observers."""

    def __init__(self, tasks):
        if not callable(getattr(tasks, '_select_request', None)):
            raise ValueError('The native source needs the private request-selection entry point')
        if hasattr(tasks, 'supermimic_flight_latch'):
            raise ValueError('Install shared jump physics before the native first-flight observer')
        self.tasks = tasks
        self._packet_owned = None
        self._native_step = tasks.step
        tasks._select_request = MethodType(self._select_request, tasks)
        tasks.step = self.step
        # Platform callers retain the optional returned history. Ordinary native
        # calls need no new state; the outer flight-latch observer still runs once.
        tasks.observe_physics = MethodType(observe_jump_physics, tasks)

    def _select_request(self, tasks, command, blocked):
        # This native entry point runs after command-segment draws and admission.
        # Draw the full vector even when no automatic request is due, as before.
        uniform = (torch.rand(tasks.num_envs, generator=tasks.generator, device=tasks.device)
                   if command is None or self._packet_owned is not None else None)
        selected = select_jump_request(
            task_family=tasks.family, base_mode=tasks.base_mode, blocked_by_stair=blocked,
            elapsed_s=tasks.elapsed, active_id=tasks.protocol.active_id,
            next_id=tasks.next_id, next_request_s=tasks.next_request_s, rise_m=tasks.rise,
            interval_uniform=uniform, packet=command, packet_owned=self._packet_owned)
        tasks.next_id.copy_(selected.next_id)
        tasks.next_request_s.copy_(selected.next_request_s)
        external = None if command is None else selected.packet_owned
        return (selected.request_id, selected.requested_height_m, selected.cancel_id,
                selected.safety_abort, selected.allow_new, external)

    def step(self, now, estimate, ground, *, command=None, packet_owned=None, **kwargs):
        validate_packet_owned(packet_owned, packet=command,
                              count=self.tasks.num_envs, device=self.tasks.device)
        self._packet_owned = packet_owned
        try:
            return self._native_step(now, estimate, ground, command=command, **kwargs)
        finally:
            self._packet_owned = None
