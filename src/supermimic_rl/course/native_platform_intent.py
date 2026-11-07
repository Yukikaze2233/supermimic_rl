"""Explicit platform callers around the existing native task boundary.

Installation is opt-in. This adapter records caller intent only; it neither
samples plans nor changes the native physical success or flight observers.
"""
from __future__ import annotations

from dataclasses import fields, replace

import torch

from .command_override import validate_height_override
from .jump_task_adapter import NativeJumpTaskAdapter
from .platform_intent import (
    PlatformIntentPhase, PlatformIntentState, consume_platform_intent, prepare_platform_intent,
)
from .request_selection import validate_packet_owned


class NativePlatformIntent:
    """Bind explicit, restartable plans outside an already installed jump gate.

    Install the shared task adapter before flight observers, then the take-off
    gate, then this caller. Capture caller history at complete policy boundaries.
    ``load_state_dict`` restores only caller history. A cold simulator restart
    still needs its real task reset; this is not a native-state checkpoint.
    """

    def __init__(self, native, *, task_adapter: NativeJumpTaskAdapter,
                 state: PlatformIntentState):
        self.native = native
        self.course = native._terrain_course
        self.tasks = self.course.tasks
        self.task_adapter = task_adapter
        self.gate = getattr(native, 'supermimic_jump_trigger', None)
        selector = getattr(self.tasks, '_select_request', None)
        if (not isinstance(task_adapter, NativeJumpTaskAdapter)
                or task_adapter.tasks is not self.tasks
                or getattr(selector, '__self__', None) is not self.tasks
                or getattr(selector, '__func__', None) != task_adapter._select_request):
            raise ValueError('Platform intent requires the matching native jump task adapter')
        if self.gate is None or self.gate.tasks is not self.tasks:
            raise ValueError('Install the native take-off gate before platform intent')
        self._check_state(state)
        self.state = PlatformIntentState.decode(state.record())
        self._patches = []
        self._pending_cancel = None

    def _check_state(self, state):
        state.validate(values=True)
        if (state.phase.shape != self.tasks.elapsed.shape
                or state.phase.device != self.tasks.elapsed.device):
            raise ValueError('Platform intent must match the native task rows and device')

    def capture(self):
        """Capture independent tensors without altering the native task state."""
        return self.state.record()

    def state_dict(self):
        return self.state.state_dict()

    def load_state_dict(self, saved):
        restored = PlatformIntentState.from_state_dict(saved)
        self._check_state(restored)
        self.state = restored
        self._pending_cancel = None

    def set_plan(self, state: PlatformIntentState, *, rows=None):
        """Replace explicit plans, without resetting protocol ID watermarks.

        Supply full typed metadata even for inactive rows. Use this after the
        real reset when the next episode should have a different caller plan.
        """
        state.validate(values=True)
        if ((state.bound_request_id != -1).any()
                or ((state.phase != int(PlatformIntentPhase.INACTIVE))
                    & (state.phase != int(PlatformIntentPhase.WAITING))).any()):
            raise ValueError('A new plan must be inactive or waiting, without request history')
        if rows is None:
            self._check_state(state)
            self.state = PlatformIntentState.decode(state.record())
        else:
            if (not isinstance(rows, torch.Tensor) or rows.ndim != 1
                    or rows.dtype != torch.long or rows.device != self.tasks.device
                    or state.phase.shape != rows.shape or state.phase.device != rows.device
                    or (rows < 0).any() or (rows >= self.tasks.num_envs).any()
                    or rows.unique().numel() != rows.numel()):
                raise ValueError('Replacement plans require unique native row indices')
            values = {}
            for field in fields(state):
                values[field.name] = getattr(self.state, field.name).clone()
                values[field.name][rows] = getattr(state, field.name)
            self.state = PlatformIntentState(**values)
        self._pending_cancel = None

    def _patch(self, owner, name, callback):
        present, previous = name in vars(owner), vars(owner).get(name)
        setattr(owner, name, callback)
        self._patches.append((owner, name, present, previous, callback))

    def install(self):
        if self._patches or hasattr(self.native, 'supermimic_platform_intent'):
            raise ValueError('Native platform intent is already installed')
        self._native_step = self.tasks.step
        self._native_tof = self.course._tof_command
        self._native_stair = self.course.stair_requests.step
        self._native_advance = self.course.advance_skill
        self._native_reset = self.course.reset_commands
        for owner, name, callback in (
            (self.tasks, 'step', self._step),
            (self.course, '_tof_command', self._tof_command),
            (self.course.stair_requests, 'step', self._stair_step),
            (self.course, 'advance_skill', self._advance_skill),
            (self.course, 'reset_commands', self._reset_commands),
            (self.native, 'supermimic_platform_intent', self),
        ):
            self._patch(owner, name, callback)
        return self

    def close(self):
        # Do not silently remove an observer installed outside this adapter.
        if any(getattr(owner, name) is not callback
               for owner, name, _, _, callback in self._patches):
            raise RuntimeError('Close outer native observers before platform intent')
        for owner, name, present, previous, _ in reversed(self._patches):
            if present:
                setattr(owner, name, previous)
            else:
                delattr(owner, name)
        self._patches.clear()
        self._pending_cancel = None

    def _merge_packet(self, prepared, command, packet_owned):
        from transformer_rl.formal.lab_env.course_tasks import RequestCommand

        validate_packet_owned(packet_owned, packet=command,
                              count=self.tasks.num_envs, device=self.tasks.device)
        if command is None:
            return prepared, prepared.jump_packet, prepared.packet_owned
        command.validate(self.tasks.num_envs, self.tasks.device)
        owned = prepared.packet_owned
        external = torch.ones_like(owned) if packet_owned is None else packet_owned
        cancel = (torch.full_like(command.request_id, -1) if command.cancel_id is None
                  else command.cancel_id)
        abort = (torch.zeros_like(owned) if command.safety_abort is None
                 else command.safety_abort)
        stop = external & ((cancel != -1) | abort)
        # Never combine an external stop with a newly submitted caller request.
        request = torch.where(owned, prepared.jump_packet.request_id, command.request_id)
        request = torch.where(owned & stop, -1, request)
        height = torch.where(owned, prepared.jump_packet.height_m, command.height_m)
        height = torch.where(owned & stop, 0., height)
        packet = RequestCommand(request, height,
                                cancel_id=(None if command.cancel_id is None
                                           else torch.where(external, cancel, -1)),
                                safety_abort=None if command.safety_abort is None else external & abort,
                                source=command.source,
                                completed_id=command.completed_id)
        unaccepted = ((prepared.state.phase == int(PlatformIntentPhase.WAITING))
                      | (prepared.state.phase == int(PlatformIntentPhase.PENDING)))
        matching_draft = ((prepared.jump_packet.request_id >= 0)
                          & (cancel == prepared.jump_packet.request_id))
        caller_cancelled = unaccepted & external & (abort | matching_draft)
        # A caller can be withdrawn before the protocol has an active ID. This
        # closes intent only; it does not manufacture a protocol finished event.
        prepared = replace(prepared, state=replace(prepared.state,
            phase=torch.where(caller_cancelled, int(PlatformIntentPhase.DONE), prepared.state.phase)))
        # Consume the actual submitted packet, never a suppressed draft ID.
        prepared = replace(prepared, jump_packet=replace(prepared.jump_packet,
            request_id=torch.where(stop, -1, prepared.jump_packet.request_id),
            height_m=torch.where(stop, 0., prepared.jump_packet.height_m)))
        # An existing whole-batch external packet must not introduce RNG draws.
        return prepared, packet, None if packet_owned is None else external | owned

    def _step(self, now, estimate, ground, *, command=None, packet_owned=None,
              height_override=None, height_override_mask=None, **kwargs):
        from transformer_rl.formal.lab_env.skill_fsm import StairState

        self._pending_cancel = None
        prepared = prepare_platform_intent(self.state, now_s=now,
            elapsed_s=self.tasks.elapsed, policy_dt_s=self.tasks.policy_dt_s,
            timeout_s=self.tasks.protocol.config.timeout_s, physical_id=self.tasks.physical_id,
            physical_accepted_at_s=self.tasks._physical_accepted_at_s,
            stair_active_id=self.course.stair_requests.active_id,
            stair_idle=self.native._skill_fsm.stair.state == int(StairState.NORMAL),
            native_ready=self.gate.ready(now, ground), admitted_height=self.tasks.stream.current[:, 3],
            max_seen_id=self.tasks.protocol.max_seen_id, next_id=self.tasks.next_id)
        prepared, command, packet_owned = self._merge_packet(prepared, command, packet_owned)
        validate_height_override(self.tasks.stream.current, height_override, height_override_mask)
        if height_override is None:
            height_override, height_override_mask = prepared.height_override_m, prepared.height_override_mask
        else:
            height_override = torch.where(prepared.height_override_mask,
                                          prepared.height_override_m, height_override)
            height_override_mask = height_override_mask | prepared.height_override_mask
        protocol = self.tasks.protocol
        original = protocol.step
        present, previous = 'step' in vars(protocol), vars(protocol).get('step')
        observed = []

        def record_condition(*args, **call_kwargs):
            condition = original(*args, **call_kwargs)
            observed.append(condition)
            return condition

        protocol.step = record_condition
        try:
            result = self._native_step(now, estimate, ground, command=command,
                packet_owned=packet_owned, height_override=height_override,
                height_override_mask=height_override_mask, **kwargs)
        finally:
            if present:
                protocol.step = previous
            else:
                del protocol.step
        if len(observed) != 1:
            raise RuntimeError('The native jump step must produce exactly one protocol condition')
        self.state = consume_platform_intent(prepared, observed[0], physical_id=self.tasks.physical_id)
        self._pending_cancel = torch.where(
            self.state.phase == int(PlatformIntentPhase.PENDING), prepared.stair_packet.cancel_id, -1)
        return result

    def _tof_command(self, request_id, height_m, cancel_id, safety_abort, ready,
                     completed_id=None):
        return self._native_tof(request_id, height_m, cancel_id, safety_abort,
                                ready & ~self.state.reservation, completed_id=completed_id)

    def _stair_step(self, **kwargs):
        try:
            if self._pending_cancel is not None:
                kwargs['cancel_id'] = torch.where(self._pending_cancel >= 0,
                    self._pending_cancel, kwargs['cancel_id'])
            return self._native_stair(**kwargs)
        finally:
            self._pending_cancel = None

    def _advance_skill(self, *args, **kwargs):
        try:
            return self._native_advance(*args, **kwargs)
        finally:
            self._pending_cancel = None

    def _reset_commands(self, ids, *args, **kwargs):
        result = self._native_reset(ids, *args, **kwargs)
        mask = torch.zeros_like(self.state.phase, dtype=torch.bool)
        mask[ids] = True
        self.state = self.state.reset(mask)
        self._pending_cancel = None
        return result
