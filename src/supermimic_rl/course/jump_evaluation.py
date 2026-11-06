"""Evaluation-only physical-request evidence, independent of jump reward phases."""
from __future__ import annotations

import torch

from ..reward_kernel import JUMP_PEAK_TOLERANCE_MAX_M, JUMP_PEAK_TOLERANCE_TARGET_FRACTION


class JumpEvaluation:
    """Observe native decisions without changing their state, actions or RNG.

    Small per-environment accumulators stay on device. Request boundaries copy
    summaries, not trajectories, to the report. Protocol completion and physical
    certification are separate: an estimated handoff need not end verification.
    """

    CONDITIONS = ('supported_start', 'flight_seen', 'ascent_seen', 'peak_in_target_window',
                  'wheel_support', 'vertical_speed', 'tilt', 'angular_speed',
                  'landing_height', 'not_cancelled')
    VALUES = ('actual_rise_m', 'velocity_z_m_s', 'tilt_rad', 'angular_speed_rad_s',
              'left_wheel_load_n', 'right_wheel_load_n')
    CRITERIA = {'flight_max_wheel_load_n': 2., 'flight_confirm_ticks': 2,
                'ascent_min_velocity_z_m_s': .15, 'ascent_min_rise_m': .01,
                'peak_tolerance_max_m': JUMP_PEAK_TOLERANCE_MAX_M,
                'peak_tolerance_target_fraction': JUMP_PEAK_TOLERANCE_TARGET_FRACTION,
                'landing_min_wheel_load_n': 5., 'landing_max_vertical_speed_m_s': .15,
                'landing_max_tilt_rad': .25, 'landing_max_angular_speed_rad_s': .5,
                'landing_height_tolerance_m': .05, 'required_stable_landing_s': 2.}

    def __init__(self, env, row_names):
        self.env, self.tasks = env, env._terrain_course.tasks
        self.row_names = tuple(row_names)
        n, device = self.tasks.physical_id.shape[0], self.tasks.physical_id.device
        if len(self.row_names) != n:
            raise ValueError('Jump evaluation needs one scenario name per environment')
        self._restore, self.records, self.pending = [], {}, set()
        self.keys = [None] * n
        self.request = torch.full((n,), -1, dtype=torch.long, device=device)
        self.counts = torch.zeros(n, 4, dtype=torch.long, device=device)
        self.failed = torch.zeros(n, len(self.CONDITIONS), dtype=torch.float64, device=device)
        self.only_failed = torch.zeros_like(self.failed)
        self.landing_failed = torch.zeros_like(self.failed)
        self.landing_candidate_s = torch.zeros(n, dtype=torch.float64, device=device)
        self.physical_s = torch.zeros_like(self.landing_candidate_s)
        self.max_landed = torch.zeros_like(self.landing_candidate_s)
        self.max_upward_speed = torch.zeros_like(self.landing_candidate_s)
        self.unloaded_s = torch.zeros_like(self.landing_candidate_s)
        self.max_unloaded_s = torch.zeros_like(self.landing_candidate_s)
        self.total_unloaded_s = torch.zeros_like(self.landing_candidate_s)
        self.clearance_ticks = torch.zeros(n, dtype=torch.long, device=device)
        self.max_minimum_clearance = torch.full((n,), -torch.inf, device=device)
        self.max_unloaded_clearance = torch.full_like(self.max_minimum_clearance, -torch.inf)
        self.target = torch.zeros(n, device=device)
        self.peak = torch.zeros_like(self.target)
        self.landed = torch.zeros_like(self.target)
        self.flags = torch.zeros(n, 4, dtype=torch.bool, device=device)
        self.last = torch.full((n, len(self.VALUES)), float('nan'), device=device)
        self.last_good = torch.zeros(n, len(self.CONDITIONS), dtype=torch.bool, device=device)
        self.success = torch.zeros(n, dtype=torch.bool, device=device)
        self.impulse = torch.zeros(n, device=device)
        self.impulse_known = torch.zeros(n, dtype=torch.bool, device=device)
        self.assist = getattr(env, 'supermimic_jump_assist', None)

    def install(self):
        from transformer_rl.formal.tasks.protocol import RequestEnd

        if self._restore:
            raise RuntimeError('Jump evaluation is already installed')
        tasks, protocol = self.tasks, self.tasks.protocol
        step, observe, reset, protocol_step = tasks.step, tasks.observe_physics, tasks.reset, protocol.step

        def stepped(now, *args, **kwargs):
            self._flush_successes()
            self._capture_impulse()
            result = step(now, *args, **kwargs)
            current = tasks.physical_id.cpu().tolist()
            for row, request in enumerate(current):
                key = self.keys[row]
                if key is not None and request != key[2]:
                    elapsed = float(now[row]) - self.records[key]['accepted_at_s']
                    if elapsed >= tasks.protocol.config.timeout_s:
                        reason = 'physical_timeout'
                    elif request >= 0:
                        reason = 'superseded_by_new_request'
                    else:
                        reason = 'protocol_closed'
                    self._end(row, reason)
                if request >= 0 and self.keys[row] is None:
                    self._begin(row, request, 'native_acceptance')
            return result

        def observed(force, velocity_z, tilt, world_height, *, angular_speed_rad_s, dt=.005):
            active = (tasks.physical_id >= 0) & (tasks.physical_id == self.request)
            self._capture_impulse()
            result = observe(force, velocity_z, tilt, world_height,
                             angular_speed_rad_s=angular_speed_rad_s, dt=dt)
            self._observe(active, force, velocity_z, tilt, world_height, angular_speed_rad_s, dt)
            return result

        def reset_rows(ids, *args, **kwargs):
            self._flush_successes()
            self._capture_impulse()
            for row in ids.cpu().tolist():
                self._protocol_end(row, int(protocol.active_id[row]), RequestEnd.RESET.name.lower())
                self._end(row, 'physical_reset')
            return reset(ids, *args, **kwargs)

        def protocol_stepped(*args, **kwargs):
            condition = protocol_step(*args, **kwargs)
            finished = condition.finished_id
            for row in (finished >= 0).nonzero().flatten().cpu().tolist():
                self._protocol_end(row, int(finished[row]),
                                   RequestEnd(int(condition.end_reason[row])).name.lower())
            return condition

        for owner, name, callback in ((tasks, 'step', stepped), (tasks, 'observe_physics', observed),
                                      (tasks, 'reset', reset_rows), (protocol, 'step', protocol_stepped)):
            self._restore.append((owner, name, owner.__dict__.get(name), name in owner.__dict__))
            setattr(owner, name, callback)
        for row in (tasks.physical_id >= 0).nonzero().flatten().cpu().tolist():
            self._begin(row, int(tasks.physical_id[row]), 'initial_active_snapshot')

    def close(self):
        for owner, name, previous, existed in reversed(self._restore):
            if existed:
                setattr(owner, name, previous)
            else:
                delattr(owner, name)
        self._restore.clear()

    def _begin(self, row, request, source):
        tasks = self.tasks
        key = (row, int(self.env._episode_id[row]), request)
        if key in self.records:
            raise ValueError('Jump evaluation requires unique episode/request identities')
        self.keys[row] = key
        self.records[key] = {'env_id': row, 'episode_id': key[1], 'request_id': request,
            'scenario': self.row_names[row], 'source': source,
            'accepted_at_s': float(tasks._physical_accepted_at_s[row]),
            'physical_end_reason': None, 'protocol_end_reason': None,
            'observed_intervals': 0, 'invalid_intervals': 0, 'boundary_valid': None}
        self.request[row] = request
        for tensor in (self.counts, self.failed, self.only_failed, self.landing_failed,
                       self.landing_candidate_s, self.physical_s, self.max_landed,
                       self.max_upward_speed, self.success, self.impulse, self.last_good,
                       self.unloaded_s, self.max_unloaded_s, self.total_unloaded_s, self.clearance_ticks):
            tensor[row] = 0
        self.max_minimum_clearance[row] = -torch.inf
        self.max_unloaded_clearance[row] = -torch.inf
        self.last[row] = float('nan')
        self.target[row] = tasks.active_rise[row]
        self.peak[row], self.landed[row] = tasks.peak_rise[row], tasks.landed_s[row]
        self.max_landed[row] = tasks.landed_s[row]
        self.flags[row] = torch.stack((tasks.supported_start[row], tasks.flight_seen[row],
                                      tasks.ascent_seen[row], tasks._physical_cancelled[row]))
        # An absent assist module certifies zero injected impulse. An installed
        # module with missing or mismatched request metadata is unknown.
        self.impulse_known[row] = self.assist is None
        self._capture_impulse()

    def _capture_impulse(self):
        if self.assist is None:
            return
        request, impulse = (getattr(self.assist, name, None) for name in ('request', 'impulse'))
        if (not isinstance(request, torch.Tensor) or not isinstance(impulse, torch.Tensor)
                or request.shape != self.request.shape or impulse.shape != self.impulse.shape):
            return
        matching = (self.request >= 0) & (request == self.request)
        finite = torch.isfinite(impulse) & (impulse >= 0)
        self.impulse = torch.where(matching & finite, impulse, self.impulse)
        self.impulse_known = torch.where(matching, finite, self.impulse_known)

    def _observe(self, active, force, velocity_z, tilt, height, angular_speed, dt):
        tasks = self.tasks
        rise = height - tasks.ground_anchor - tasks.latched_height
        values = torch.cat((torch.stack((rise, velocity_z, tilt, angular_speed), -1), force), -1)
        finite = torch.isfinite(values).all(-1) & torch.isfinite(tasks.peak_rise)
        valid = active & finite
        tolerance = torch.minimum(torch.full_like(tasks.active_rise, .03), tasks.active_rise * .5)
        good = torch.stack((tasks.supported_start, tasks.flight_seen, tasks.ascent_seen,
            (tasks.peak_rise - tasks.active_rise).abs() <= tolerance + 1e-6,
            (force >= 5).all(-1), velocity_z.abs() <= .15, tilt <= .25,
            angular_speed <= .5, rise.abs() <= .05, ~tasks._physical_cancelled), -1)
        candidate = valid & good[:, :4].all(-1)
        missed = ~good
        self.counts[:, 0] += active.long()
        self.counts[:, 1] += valid.long()
        self.counts[:, 2] += (active & ~finite).long()
        owned = getattr(self.env, '_interval_policy_owned', None)
        if isinstance(owned, torch.Tensor):
            self.counts[:, 3] += (active & ~owned).long()
        self._observe_unloading(valid, owned, force, dt)
        self.physical_s += active * dt
        self.failed += (valid[:, None] & missed) * dt
        self.only_failed += (valid[:, None] & missed & (missed.sum(-1) == 1)[:, None]) * dt
        self.landing_failed += (candidate[:, None] & missed) * dt
        self.landing_candidate_s += candidate * dt
        self.max_landed = torch.where(active, torch.maximum(self.max_landed, tasks.landed_s), self.max_landed)
        self.max_upward_speed = torch.where(valid, torch.maximum(self.max_upward_speed,
                                                                velocity_z.double().clamp_min(0)),
                                            self.max_upward_speed)
        self.peak = torch.where(active, tasks.peak_rise, self.peak)
        self.landed = torch.where(active, tasks.landed_s, self.landed)
        flags = torch.stack((tasks.supported_start, tasks.flight_seen, tasks.ascent_seen,
                             tasks._physical_cancelled), -1)
        self.flags = torch.where(active[:, None], flags, self.flags)
        self.last = torch.where(active[:, None], values, self.last)
        self.last_good = torch.where(active[:, None], good, self.last_good)
        self.success |= active & (tasks.success_id == self.request) & (tasks.physical_id < 0)

    def _observe_unloading(self, valid, owned, force, dt):
        """Report request-scoped physics without changing native success decisions.

        The native support observer has already measured the current tick's
        wheel cylinders before tasks.observe_physics runs. Its geometry is
        distinct from its terrain-only airborne flag, which excludes JUMP.
        """
        measured = valid & (force >= 0).all(-1)
        if isinstance(owned, torch.Tensor):
            measured &= owned
        unloaded = measured & (force <= self.CRITERIA['flight_max_wheel_load_n']).all(-1)
        self.unloaded_s = torch.where(unloaded, self.unloaded_s + dt, 0.)
        self.max_unloaded_s = torch.maximum(self.max_unloaded_s, self.unloaded_s)
        self.total_unloaded_s += unloaded.to(self.total_unloaded_s.dtype) * dt
        course = self.env._terrain_course
        support = getattr(course, '_support_sample', None)
        bottom = getattr(support, 'wheel_bottom_world_z_m', None)
        support_valid = getattr(support, 'valid', None)
        ground = getattr(course, 'wheel_ground', None)
        if (not isinstance(bottom, torch.Tensor) or not isinstance(ground, torch.Tensor)
                or not isinstance(support_valid, torch.Tensor)
                or bottom.shape != force.shape or ground.shape != force.shape
                or support_valid.shape != valid.shape):
            return
        clearance = bottom - ground
        geometry_valid = measured & support_valid & torch.isfinite(clearance).all(-1)
        self.clearance_ticks += geometry_valid.long()
        # The lower of the two wheel clearances prevents a lifted single wheel
        # from being reported as whole-vehicle clearance.
        minimum = clearance.amin(-1)
        self.max_minimum_clearance = torch.where(geometry_valid,
            torch.maximum(self.max_minimum_clearance, minimum), self.max_minimum_clearance)
        self.max_unloaded_clearance = torch.where(geometry_valid & unloaded,
            torch.maximum(self.max_unloaded_clearance, minimum), self.max_unloaded_clearance)

    def _protocol_end(self, row, request, reason):
        if request < 0:
            return
        key = (row, int(self.env._episode_id[row]), request)
        if key in self.records:
            self.records[key]['protocol_end_reason'] = reason

    def _flush_successes(self):
        for row in self.success.nonzero().flatten().cpu().tolist():
            self._end(row, 'verified_success')

    @staticmethod
    def _number(value):
        return float(value) if bool(torch.isfinite(value)) else None

    def _snapshot(self, row):
        counts = self.counts[row].cpu().tolist()
        flags = self.flags[row].cpu().tolist()
        target, peak = self._number(self.target[row]), self._number(self.peak[row])
        tolerance = None if target is None else min(.03, .5 * target)
        impulse = self._number(self.impulse[row]) if bool(self.impulse_known[row]) else None
        return {'physical_ticks': counts[0], 'finite_physical_ticks': counts[1],
            'nonfinite_physical_ticks': counts[2], 'non_policy_owned_physical_ticks': counts[3],
            'policy_ownership_observed': isinstance(getattr(self.env, '_interval_policy_owned', None), torch.Tensor),
            'physical_observation_s': float(self.physical_s[row]),
            'supported_start': flags[0], 'flight_seen': flags[1], 'ascent_seen': flags[2],
            'physical_cancelled': flags[3], 'peak_rise_m': peak, 'requested_rise_m': target,
            'peak_tolerance_m': tolerance,
            'peak_in_target_window': (None if peak is None or target is None else
                                      abs(peak - target) <= tolerance + 1e-6),
            'max_landed_s': float(self.max_landed[row]), 'landed_at_end_s': float(self.landed[row]),
            'max_upward_com_velocity_m_s': float(self.max_upward_speed[row]),
            'max_continuous_both_wheels_unloaded_s': float(self.max_unloaded_s[row]),
            'total_both_wheels_unloaded_s': float(self.total_unloaded_s[row]),
            'wheel_clearance_observed_ticks': int(self.clearance_ticks[row]),
            'max_minimum_wheel_clearance_m': self._number(self.max_minimum_clearance[row]),
            'max_minimum_wheel_clearance_while_unloaded_m': self._number(self.max_unloaded_clearance[row]),
            'actual_assist_impulse_n_s': impulse,
            'assisted': None if impulse is None else impulse > 0,
            'native_verified_physical_success': bool(self.success[row]),
            'failed_condition_s': dict(zip(self.CONDITIONS, self.failed[row].cpu().tolist())),
            'only_failed_condition_s': dict(zip(self.CONDITIONS, self.only_failed[row].cpu().tolist())),
            'landing_candidate_s': float(self.landing_candidate_s[row]),
            'landing_failed_condition_s': dict(zip(self.CONDITIONS, self.landing_failed[row].cpu().tolist())),
            'last_physical_values': dict(zip(self.VALUES, [self._number(v) for v in self.last[row].cpu()])),
            'last_physical_conditions': (dict(zip(self.CONDITIONS, self.last_good[row].cpu().tolist()))
                                         if counts[0] else None)}

    def _end(self, row, reason):
        key = self.keys[row]
        if key is None:
            return
        self.records[key].update(self._snapshot(row), physical_end_reason=reason)
        self.pending.add(key)
        self.keys[row] = None
        self.request[row] = -1
        self.success[row] = False

    def observe_transition(self, transition, valid):
        """Associate request outcomes with their pre-reset physical interval."""
        self._flush_successes()
        starts, ends = transition.start.episode_id.cpu().tolist(), transition.end.episode_id.cpu().tolist()
        for key in set(self.pending) | {key for key in self.keys if key is not None}:
            row, episode, _ = key
            if starts[row] != episode:
                continue
            record = self.records[key]
            trusted = bool(valid[row]) and ends[row] == episode
            record['observed_intervals'] += 1
            record['invalid_intervals'] += int(not trusted)
            if key in self.pending:
                record['boundary_valid'] = trusted
                self.pending.remove(key)

    def summary(self):
        # A success on the final substep is complete even if no next step/reset
        # calls the deferred boundary flush. Keep summary itself read-only.
        current = {key: row for row, key in enumerate(self.keys) if key is not None}
        requests, groups = [], {}
        for key, saved in self.records.items():
            record = dict(saved)
            if key in current:
                row = current[key]
                record.update(self._snapshot(row), physical_end_reason=(
                    'verified_success' if bool(self.success[row]) else 'active_at_evaluation_end'))
            record['had_invalid'] = bool(record['invalid_intervals'] or record['nonfinite_physical_ticks']
                                         or record['boundary_valid'] is False)
            record['validated_boundary'] = record['boundary_valid'] is True and not record['had_invalid']
            requests.append(record)
            group = groups.setdefault(record['scenario'], {'requests': 0, 'physical_end_counts': {},
                'protocol_end_counts': {}, 'verified_successes': 0, 'validated_verified_successes': 0,
                'flight_seen': 0, 'ascent_seen': 0, 'peak_in_target_window': 0,
                'actual_assisted_requests': 0, 'actual_unassisted_requests': 0, 'assist_unknown_requests': 0})
            group['requests'] += 1
            for field, counts in (('physical_end_reason', 'physical_end_counts'),
                                   ('protocol_end_reason', 'protocol_end_counts')):
                label = record[field] or 'not_observed'
                group[counts][label] = group[counts].get(label, 0) + 1
            group['verified_successes'] += int(record['native_verified_physical_success'])
            group['validated_verified_successes'] += int(record['native_verified_physical_success']
                                                        and record['validated_boundary'])
            for name in ('flight_seen', 'ascent_seen', 'peak_in_target_window'):
                group[name] += int(bool(record[name]))
            group['assist_unknown_requests' if record['assisted'] is None else
                  'actual_assisted_requests' if record['assisted'] else 'actual_unassisted_requests'] += 1
        return {'criteria': dict(self.CRITERIA), 'physical_timeout_s': self.tasks.protocol.config.timeout_s,
            'by_scenario': groups, 'requests': requests,
            'scope': 'Observed native physical requests keyed by environment, episode and request ID. Protocol endings and physical certification are separate; active requests at evaluation end are incomplete, not failed.',
            'condition_units': 'Seconds of finite active physical samples; conditions can overlap. Landing-candidate seconds additionally require supported start, flight, ascent and peak within tolerance. Maxima include every observed substep.',
            'unloading_scope': 'Request-scoped consecutive finite policy-owned physics ticks with both wheel support loads <=2 N. Invalid, unowned or supported ticks break the run. Duration and measured cylinder-bottom clearance are descriptive, not additional success gates. Clearance is the lower of the two wheels above their current terrain columns; missing geometry is null. Neither metric alone rules out other body contacts.',
            'validity': 'Raw native outcomes are retained. Validated boundaries require an observed valid ending interval and no earlier invalid interval/nonfinite physical sample. Initial active snapshots have partial history.',
            'assist_scope': 'Zero impulse is known when no assist module is installed; installed modules require matching request and finite nonnegative impulse metadata.'}
