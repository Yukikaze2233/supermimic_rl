"""Measured zero-command drift over continuous native policy intervals."""
from __future__ import annotations

import math

import torch

from ..compact_state import named_field


class StationaryEvaluation:
    """Keep net displacement distinct from travel and absolute velocity.

    Local world positions share a fixed terrain origin within an episode. Never
    integrate across resets, invalid intervals, task changes or control handoffs.
    These are diagnostics, not new success thresholds or motion constraints.
    """

    FIELDS = (
        'course.root_position_local_m', 'course.command_mode',
        'course.recovery.active', 'course.gimbal.special',
        'course.tasks.physical_id.valid', 'course.protocol.active',
    )

    def __init__(self, schema, row_names, *, normal_mode):
        self.schema, self.row_names = schema, tuple(row_names)
        self.normal_mode = normal_mode
        self.active = {}
        self.completed = []
        self.excluded = {}
        self.intervals = 0

    def _finish(self, row, reason):
        if row in self.active:
            self.completed.append(self._report(self.active.pop(row), reason))

    @staticmethod
    def _report(segment, reason):
        result = dict(segment)
        delta = [b - a for a, b in zip(result['start_xy_m'], result['end_xy_m'])]
        result.update(net_xy_m=delta, net_displacement_m=math.hypot(*delta),
                      mean_abs_forward_m_s=result.pop('abs_forward_integral_m') / result['duration_s'],
                      mean_abs_yaw_rad_s=result.pop('abs_yaw_integral_rad') / result['duration_s'],
                      end_reason=reason)
        return result

    def observe_transition(self, transition, tracking, valid):
        context = getattr(tracking, 'course', None)
        command = None if context is None else context.get('stationary_admitted_command')
        actual = None if tracking is None else getattr(tracking, 'actual', None)
        if command is None or actual is None:
            for row in range(len(self.row_names)):
                self._finish(row, 'missing_measurement')
            self.excluded['missing_measurement'] = self.excluded.get('missing_measurement', 0) + len(self.row_names)
            self.intervals += 1
            return
        command, actual = command.detach().cpu(), actual.detach().cpu()
        start, end = transition.start, transition.end
        states = [{name: named_field(snapshot.state, self.schema, name).detach().cpu()
                   for name in self.FIELDS} for snapshot in (start, end)]
        positions = [value[self.FIELDS[0]][:, :2].double() for value in states]
        eligible = valid.detach().cpu().clone()
        reasons = ['eligible'] * len(self.row_names)
        checks = [('invalid_interval', eligible.clone()),
                  ('nonzero_command', command[:, :3].eq(0).all(-1)),
                  ('nonfinite_measurement', torch.isfinite(command).all(-1) & torch.isfinite(actual).all(-1)),
                  ('unowned', start.policy_owned.cpu() & end.policy_owned.cpu()),
                  ('episode_boundary', start.episode_id.cpu() == end.episode_id.cpu()),
                  ('control_boundary', start.control_epoch.cpu() == end.control_epoch.cpu())]
        for values in states:
            checks.extend([
                ('nonfinite_measurement', torch.stack([torch.isfinite(value).reshape(len(self.row_names), -1).all(-1)
                                                       for value in values.values()]).all(0)),
                ('other_mode', values['course.command_mode'].reshape(-1).eq(self.normal_mode)),
                ('recovery', ~values['course.recovery.active'].reshape(-1).bool()),
                ('gimbal_translation', ~values['course.gimbal.special'].reshape(-1).bool()),
                ('active_jump', ~values['course.tasks.physical_id.valid'].reshape(-1).bool()
                               & ~values['course.protocol.active'].reshape(-1).bool()),
            ])
        # One exclusion reason per interval; no hidden denominator overlap.
        eligible.fill_(True)
        for label, passed in checks:
            for row in (eligible & ~passed).nonzero().flatten().tolist():
                reasons[row] = label
            eligible &= passed
        terminated, truncated = transition.terminated.cpu(), transition.truncated.cpu()
        dt = self.schema.policy_dt_s
        for row, name in enumerate(self.row_names):
            if not eligible[row]:
                self._finish(row, reasons[row])
                self.excluded[reasons[row]] = self.excluded.get(reasons[row], 0) + 1
                continue
            episode, epoch = int(start.episode_id[row]), int(start.control_epoch[row])
            p0, p1 = positions[0][row].tolist(), positions[1][row].tolist()
            previous = self.active.get(row)
            if previous is not None and (previous['episode_id'] != episode or previous['control_epoch'] != epoch
                    or previous['end_interval'] != self.intervals - 1 or previous['end_xy_m'] != p0):
                self._finish(row, 'discontinuous_endpoint')
            if row not in self.active:
                self.active[row] = dict(scenario=name, env_id=row, episode_id=episode, control_epoch=epoch,
                    start_interval=self.intervals, end_interval=self.intervals, start_xy_m=p0, end_xy_m=p0,
                    samples=0, duration_s=0., path_m=0., abs_forward_integral_m=0., abs_yaw_integral_rad=0.)
            segment = self.active[row]
            segment['end_interval'], segment['end_xy_m'] = self.intervals, p1
            segment['samples'] += 1
            segment['duration_s'] += dt
            segment['path_m'] += math.dist(p0, p1)
            segment['abs_forward_integral_m'] += abs(float(actual[row, 0])) * dt
            segment['abs_yaw_integral_rad'] += abs(float(actual[row, 1])) * dt
            if terminated[row] or truncated[row]:
                self._finish(row, 'terminated' if terminated[row] else 'truncated')
        self.intervals += 1

    def summary(self):
        return {'format': 'supermimic-zero-command-drift-v1',
                'selection': 'Exact admitted vx=vy=yaw=0, NORMAL, policy-owned, no special gimbal translation, recovery or physical/protocol jump at either endpoint. Independent gimbal motion is included.',
                'position_semantics': 'Native root-link XY in the fixed episode terrain frame; endpoint displacement and policy-rate polyline length, not integrated absolute speed. Segments never cross resets, invalid rows or control handoffs.',
                'velocity_semantics': 'Absolute pre-action heading-forward origin speed and world-up yaw rate; not signed net displacement. Path is sampled at policy frequency.',
                'excluded_intervals': dict(self.excluded),
                'segments': self.completed + [self._report(value, 'active_at_evaluation_end')
                                              for value in self.active.values()]}
