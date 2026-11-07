"""Evaluation-only manual jumps on fixed authored terrain.

The caller owns the goal, trigger time and command values. Existing protocols
own admission and IDs; this observer never substitutes actions or certifies a
platform landing. Future manual intent is NOT part of the training schema.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import torch


@dataclass(frozen=True)
class ManualPlatformJumpIntent:
    env_id: int
    target_id: str
    jump_at_episode_s: float
    rise_m: float
    support_height_m: float

    def __post_init__(self):
        if (type(self.env_id) is not int or self.env_id < 0
                or self.target_id not in ('101:profile:1', '102:profile:1')
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       for value in (self.jump_at_episode_s, self.rise_m, self.support_height_m))
                or self.jump_at_episode_s < 0 or self.rise_m <= 0 or self.support_height_m <= 0):
            raise ValueError('Manual platform jump needs an explicit row, target, time, rise and support height')


class ManualPlatformJumpEvaluation:
    """Context around existing ``evaluate_native`` with a fixed TERRAIN suite.

    One intent per row avoids turning a vector-wide external packet into an
    accidental override of automatic requests in a mixed REQUEST population.
    Support height passes through the existing command stream. Rejections may
    consume IDs; a subsequent submission always uses the native watermark.
    """

    FORMAT = 'manual-platform-jump-evaluation-v1'
    DATA_USE = {
        'joint_task_imagination_supported': False,
        'joint_imagination_forbidden': True,
        'permitted_training_use': 'physical dynamics supervision only, subject to ordinary source checks',
        'missing_from_training_record': 'manual intent, pending/future packets and reservation state',
        'native_flat_success_reinterpreted': False,
    }

    def __init__(self, adapter, *, intents, catalog_path, catalog_sha256, sidecar_path):
        self.adapter, self.env = adapter, adapter.env
        self.intents = tuple(intents)
        if (not self.intents or any(not isinstance(i, ManualPlatformJumpIntent) for i in self.intents)
                or sorted(i.env_id for i in self.intents) != list(range(adapter.num_envs))):
            raise ValueError('Supply exactly one explicit intent for every fixed TERRAIN evaluation row')
        self.intents = tuple(sorted(self.intents, key=lambda i: i.env_id))
        raw = Path(catalog_path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != catalog_sha256:
            raise ValueError('Platform catalog pin differs')
        self.catalog = json.loads(raw)
        if self.catalog['format'] != 'authored-platform-triangle-targets-v1':
            raise ValueError('Unsupported platform target catalog')
        targets = {t['target_id']: t for t in self.catalog['targets']}
        if len(targets) != len(self.catalog['targets']):
            raise ValueError('Duplicate target identities')
        self.targets = tuple(targets[i.target_id] for i in self.intents)
        if any(not t['initial_landing_enabled'] or t['profile_index'] != 1 for t in self.targets):
            raise ValueError('Only the two reviewed first platforms are initial evaluation targets')
        self.catalog_sha256, self.sidecar_path = catalog_sha256, Path(sidecar_path)
        self.events, self.states = [], [None] * adapter.num_envs
        self.reserved = torch.zeros(adapter.num_envs, dtype=torch.bool, device=adapter.device)
        self._outer, self._native = [], []
        self._entered = False
        self._begun = False
        self._finish_suite = None

    def recording_source(self, source):
        """Pass this to evaluate_native(recording_source=...), never omit the use restriction."""
        return {**source, 'manual_platform_jump': self._metadata()}

    def _metadata(self):
        return {'format': self.FORMAT, 'catalog_sha256': self.catalog_sha256,
                'intents': [asdict(i) for i in self.intents], 'data_use': deepcopy(self.DATA_USE),
                'trigger': {'jump_at_episode_s': 'earliest request boundary, not guaranteed submission time',
                    'readiness': 'existing native JumpTriggerGate.ready after exact admitted caller height',
                    'uses_privileged_native_supported_start': True,
                    'deployment_controller_implemented': False,
                    'reservation': 'pending until earliest time + native timeout; accepted until native acceptance + timeout'},
                'sidecar_path': str(self.sidecar_path)}

    @staticmethod
    def _patch(restore, owner, name, callback):
        restore.append((owner, name, owner.__dict__.get(name), name in owner.__dict__))
        setattr(owner, name, callback)

    @staticmethod
    def _unpatch(restore):
        for owner, name, previous, existed in reversed(restore):
            if existed:
                setattr(owner, name, previous)
            else:
                delattr(owner, name)
        restore.clear()

    def __enter__(self):
        if self._entered or getattr(self.adapter, '_evaluation_configuration', None) is None:
            raise ValueError('Manual platform jumps require one fresh fixed-suite evaluation context')
        self.sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        with self.sidecar_path.open('x') as stream:
            stream.write(json.dumps({'event': 'metadata', **self._metadata()}) + '\n')
        self._entered = True
        begin, summarize, step = (self.adapter.evaluation_begin, self.adapter.evaluation_summary,
                                  self.adapter.evaluation_step)

        def beginning(schema):
            if self._begun:
                raise RuntimeError('One evaluation begin per manual-jump context')
            self._begun = True
            begin(schema)
            try:
                self._install_native(self.adapter._course_evaluator.plan)
            except BaseException:
                self.adapter.evaluation_end()
                raise
            finish = self.adapter.evaluation_end
            suite_finished = False

            def finished():
                nonlocal suite_finished
                if suite_finished:
                    return None
                suite_finished = True
                self._close_native('evaluation_end')
                return finish()

            self._finish_suite = finished
            self._patch(self._outer, self.adapter, 'evaluation_end', finished)

        def summary():
            return {**summarize(), 'manual_platform_jump': self.summary()}

        def observed_step(transition, tracking):
            for row in range(self.adapter.num_envs):
                if bool(transition.terminated[row] or transition.truncated[row] or not transition.valid[row]):
                    self._emit('interval_end', row, episode_id=int(transition.start.episode_id[row]),
                               terminated=bool(transition.terminated[row]), truncated=bool(transition.truncated[row]),
                               valid=bool(transition.valid[row]))
            return step(transition, tracking)

        self._patch(self._outer, self.adapter, 'evaluation_begin', beginning)
        self._patch(self._outer, self.adapter, 'evaluation_summary', summary)
        self._patch(self._outer, self.adapter, 'evaluation_step', observed_step)
        return self

    def _install_native(self, plan):
        from transformer_rl.formal.tasks.curriculum import TaskFamily

        if self._native:
            raise RuntimeError('Manual evaluation is already installed')
        self.course = course = self.env._terrain_course
        self.tasks = tasks = course.tasks
        self.gate = getattr(self.env, 'supermimic_jump_trigger', None)
        if not callable(getattr(self.gate, 'ready', None)):
            raise ValueError('Manual evaluation requires the existing native jump supported-start gate')
        if (any(plan.scenarios[name]['family'] != 'TERRAIN' for name in plan.row_names)
                or not (tasks.family == int(TaskFamily.TERRAIN)).all()):
            raise ValueError('Manual platform evaluation must use only fixed TERRAIN rows')
        pins = self.catalog['source']
        if (course.router.identity != pins['atlas_manifest']['sha256']
                or course.router.manifest['files']['geometry.npz']['sha256'] != pins['atlas_geometry']['sha256']):
            raise ValueError('Native terrain and manual target catalog differ')
        for intent in self.intents:
            request, reference = tasks.protocol.config, tasks.reference.config
            if (not request.height_min_m <= intent.rise_m <= request.height_max_m
                    or not reference.support_height_min_m <= intent.support_height_m <= reference.support_height_max_m):
                raise ValueError('Caller jump values exceed the existing native request/reference domain')
        self.timeout = float(tasks.protocol.config.timeout_s)
        advance, stream, tof = course.advance_skill, tasks.stream.step, course._tof_command
        jump_step, stair_step, reset = tasks.protocol.step, course.stair_requests.step, course.reset_commands

        def advanced():
            self._prepare_boundary()
            result = advance()
            for row, state in enumerate(self.states):
                if state is not None and state.pop('accepted_pending_trace', False):
                    self._emit('accepted_baseline', row, request_id=state['request_id'],
                               admitted_support_height_m=float(tasks.latched_height[row]),
                               physical_supported_start=bool(tasks.supported_start[row]),
                               reservation_deadline_native_s=state['deadline_clock_s'])
            return result

        def commanded(requested):
            target = requested.clone()
            for row, state in enumerate(self.states):
                if state is not None and state['stage'] != 'done':
                    target[row, 3] = self.intents[row].support_height_m
            return stream(target)

        def reserved_tof(request_id, height_m, cancel_id, safety_abort, ready, completed_id=None):
            # The original detector/countdown still runs. Only new submissions
            # are reserved; cancellation is processed by stair_requests.step.
            return tof(request_id, height_m, cancel_id, safety_abort, ready & ~self.reserved,
                       completed_id=completed_id)

        def jump_observed(*args, **kwargs):
            result = jump_step(*args, **kwargs)
            self._observe_jump(result, kwargs['request_id'])
            return result

        def stair_observed(*args, **kwargs):
            result = stair_step(*args, **kwargs)
            for row in (kwargs['cancel_id'] >= 0).nonzero().flatten().tolist():
                self._emit('stair_cancel_response', row, cancel_id=int(kwargs['cancel_id'][row]),
                           active_id=int(result.active_id[row]), finished_id=int(result.finished_id[row]),
                           end_reason=int(result.end_reason[row]))
            return result

        def resetting(ids, *args, **kwargs):
            for row in ids.tolist():
                self._end(row, 'native_reset')
                self.states[row] = None
            return reset(ids, *args, **kwargs)

        for owner, name, callback in ((course, 'advance_skill', advanced), (tasks.stream, 'step', commanded),
                (course, '_tof_command', reserved_tof), (tasks.protocol, 'step', jump_observed),
                (course.stair_requests, 'step', stair_observed), (course, 'reset_commands', resetting)):
            self._patch(self._native, owner, name, callback)

    def _emit(self, event, row, **values):
        state = self.states[row]
        record = {'event': event, 'env_id': row, 'target_id': self.intents[row].target_id,
                  'episode_id': state['episode_id'] if state is not None else int(self.env._episode_id[row]),
                  'native_clock_s': float(self.env._clock()[row]), **values}
        self.events.append(record)
        with self.sidecar_path.open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')

    def _end(self, row, reason):
        state = self.states[row]
        if state is not None and state['stage'] != 'done':
            self._emit('intent_end', row, reason=reason, request_id=state.get('request_id'),
                       request_was_accepted='request_id' in state,
                       initiation_outcome='initiated' if 'request_id' in state else 'not_initiated')
            state['stage'] = 'done'
        self.reserved[row] = False

    def _prepare_boundary(self):
        from transformer_rl.formal.lab_env.course_tasks import RequestCommand
        from transformer_rl.formal.lab_env.skill_fsm import StairState

        n, device = self.adapter.num_envs, self.adapter.device
        request = torch.full((n,), -1, dtype=torch.int64, device=device)
        rise = torch.zeros(n, device=device)
        cancel = torch.full_like(request, -1)
        native_ready = self.gate.ready(self.env._clock(), self.env._ground_at_root)
        for row, intent in enumerate(self.intents):
            episode = int(self.env._episode_id[row])
            state = self.states[row]
            if state is None or state['episode_id'] != episode:
                self._end(row, 'episode_changed')
                if int(self.course.router.tile[row]) != self.targets[row]['tile_id']:
                    raise ValueError('Fixed evaluation reset selected a different target terrain tile')
                state = {'episode_id': episode, 'stage': 'waiting',
                         'deadline_s': intent.jump_at_episode_s + self.timeout}
                self.states[row] = state
                self._emit('intent_begin', row, **asdict(intent))
            episode_s = float(self.tasks.elapsed[row]) + float(self.tasks.policy_dt_s)
            if state['stage'] == 'done':
                continue
            expired = (float(self.env._clock()[row]) >= state['deadline_clock_s'] - 1e-9
                       if 'deadline_clock_s' in state else episode_s >= state['deadline_s'] - 1e-9)
            if expired:
                self._end(row, 'explicit_window_timeout')
                continue
            if episode_s + 1e-9 < intent.jump_at_episode_s:
                continue
            self.reserved[row] = True
            if state['stage'] in ('accepted', 'physical_tail'):
                continue
            state['stage'] = 'pending'
            stair_id = int(self.course.stair_requests.active_id[row])
            if stair_id >= 0:
                cancel[row] = stair_id
                self._pending_reason(row, 'stair_cancel_then_confirm_idle', episode_s)
                self._emit('stair_cancel_packet', row, cancel_id=stair_id, episode_s=episode_s)
            elif int(self.env._skill_fsm.stair.state[row]) == int(StairState.NORMAL):
                current = self.tasks.stream.current[row, 3]
                if not bool(current == current.new_tensor(intent.support_height_m)):
                    self._pending_reason(row, 'caller_height_not_yet_admitted', episode_s)
                    continue
                if not bool(native_ready[row]):
                    self._pending_reason(row, 'native_supported_start_not_ready', episode_s)
                    continue
                request[row] = max(int(self.tasks.protocol.max_seen_id[row]) + 1, int(self.tasks.next_id[row]))
                rise[row] = intent.rise_m
                self._emit('jump_packet', row, request_id=int(request[row]), rise_m=intent.rise_m,
                           support_height_m=intent.support_height_m, episode_s=episode_s,
                           delay_after_earliest_s=max(0., episode_s - intent.jump_at_episode_s))
            else:
                self._pending_reason(row, 'stair_fsm_not_idle', episode_s)
        if bool((cancel >= 0).any()):
            self.course.submit_request('stair', RequestCommand(torch.full_like(request, -1),
                                                              torch.zeros_like(rise), cancel_id=cancel))
        if bool((request >= 0).any()):
            self.course.submit_request('jump', RequestCommand(request, rise))

    def _pending_reason(self, row, reason, episode_s):
        state = self.states[row]
        if state.get('pending_reason') != reason:
            self._emit('pending', row, reason=reason, episode_s=episode_s,
                       delay_after_earliest_s=max(0., episode_s - self.intents[row].jump_at_episode_s),
                       admitted_support_height_m=float(self.tasks.stream.current[row, 3]))
            state['pending_reason'] = reason

    def _observe_jump(self, result, submitted):
        from transformer_rl.formal.tasks.protocol import RequestEnd, RequestResponse

        for row in (submitted >= 0).nonzero().flatten().tolist():
            state = self.states[row]
            self._emit('jump_response', row, request_id=int(submitted[row]), response=int(result.response[row]),
                       response_name=RequestResponse(int(result.response[row])).name.lower())
            if int(result.response[row]) == int(RequestResponse.ACCEPTED):
                state.update(stage='accepted', request_id=int(result.active_id[row]), accepted_pending_trace=True,
                             deadline_clock_s=float(self.env._clock()[row]) + self.timeout)
        for row in (result.finished_id >= 0).nonzero().flatten().tolist():
            state = self.states[row]
            if state is None or state.get('request_id') != int(result.finished_id[row]):
                continue
            reason = RequestEnd(int(result.end_reason[row]))
            self._emit('jump_protocol_end', row, request_id=int(result.finished_id[row]), end_reason=reason.name.lower())
            if reason == RequestEnd.ESTIMATED_COMPLETE:
                state['stage'] = 'physical_tail'
            else:
                self._end(row, reason.name.lower())

    def _close_native(self, reason):
        if self._native:
            for row in range(self.adapter.num_envs):
                self._end(row, reason)
            self._unpatch(self._native)

    def summary(self):
        return {**self._metadata(), 'events': deepcopy(self.events),
                'platform_landing_success': None,
                'scope': 'Caller-authored evaluation packets and raw native responses only; '
                         'use recorded physics for independent platform landing evidence. '
                         'Native flat-jump success is unchanged and does not certify platform landing.'}

    def __exit__(self, exc_type, exc, traceback):
        try:
            self._close_native('evaluation_exception' if exc_type is not None else 'evaluation_context_end')
            if self._finish_suite is not None:
                self._finish_suite()
        finally:
            self._unpatch(self._outer)
