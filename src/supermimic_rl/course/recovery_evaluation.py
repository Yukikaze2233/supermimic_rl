"""Evaluation-only, episode-scoped observations of the native recovery lifecycle."""
from __future__ import annotations

from dataclasses import asdict

import torch


class RecoveryEvaluation:
    """Read native predicates/clocks without changing decisions, state or RNG.

    Tensor accumulators retain only per-environment summaries. Native reset and
    finish hooks preserve events before auto-reset; the completed transition
    subsequently supplies validity for that same episode, not the newborn one.
    """

    CONDITIONS = ('height', 'tilt', 'angular_speed', 'vertical_speed', 'wheel_support', 'base_clear')
    VALUES = ('height_error_m', 'tilt_rad', 'angular_speed_rad_s', 'velocity_z_m_s',
              'left_wheel_load_n', 'right_wheel_load_n', 'base_load_n')
    ESTIMATE_CONDITIONS = ('valid', 'fresh_clock', 'new_sample', 'wheel_contact',
                           'tilt', 'angular_speed', 'vertical_speed')

    def __init__(self, env, row_names):
        self.env, self.native = env, env._terrain_course.recovery
        self.row_names = tuple(row_names)
        n, device = self.native.num_envs, self.native.device
        self._restore, self.records, self.pending = [], {}, set()
        self.pending_birth = set()
        self.keys = [None] * n
        self.after_handoff = torch.zeros(n, dtype=torch.bool, device=device)
        self.counts = torch.zeros(n, 4, dtype=torch.long, device=device)
        self.failed = torch.zeros(n, len(self.CONDITIONS), dtype=torch.long, device=device)
        self.only_failed = torch.zeros_like(self.failed)
        self.maximum = torch.zeros(n, 2, dtype=torch.float64, device=device)
        self.last = torch.full((n, len(self.VALUES)), float('nan'), device=device)
        self.last_good = torch.zeros_like(self.failed, dtype=torch.bool)
        self.last_valid = torch.zeros(n, dtype=torch.bool, device=device)
        self.estimate_intervals = torch.zeros(n, dtype=torch.long, device=device)
        self.estimate_failed = torch.zeros(n, len(self.ESTIMATE_CONDITIONS), dtype=torch.long, device=device)
        self.last_estimate_good = torch.zeros_like(self.estimate_failed, dtype=torch.bool)

    def install(self):
        owner = self.native
        observe, finish, reset = owner.observe_physics, owner.finish_interval, owner.reset

        def observed(**values):
            self._bind_pending_births()
            active = owner.active.clone()
            result = observe(**values)
            self._observe(active, values)
            return result

        def finished(now_s, estimate, *, policy_owned, failed):
            self._bind_pending_births()
            active = owner.active.clone()
            self._observe_estimate(active & policy_owned & ~failed, now_s, estimate)
            result = finish(now_s, estimate, policy_owned=policy_owned, failed=failed)
            self.maximum[:, 1] = torch.maximum(self.maximum[:, 1], owner.estimated_stable_s)
            for label, mask in (('handoff', result.handoff), ('timeout', result.timeout),
                                ('failed', active & failed)):
                for row in mask.nonzero().flatten().cpu().tolist():
                    self._end(row, label, estimate=estimate, now_s=now_s)
            self.after_handoff |= result.handoff
            return result

        def reset_rows(ids, recovery):
            self._bind_pending_births()
            rows = ids.cpu().tolist()
            for row in rows:
                if self.keys[row] is not None:
                    self._end(row, 'reset_before_completion')
            result = reset(ids, recovery)
            self.after_handoff[ids] = False
            # Native reset_commands runs before env._episode_id increments.
            # Bind after the enclosing reset returns, never guess the next ID.
            self.pending_birth.update(ids[recovery].cpu().tolist())
            return result

        for name, callback in (('observe_physics', observed), ('finish_interval', finished), ('reset', reset_rows)):
            self._restore.append((name, owner.__dict__.get(name), name in owner.__dict__))
            setattr(owner, name, callback)
        self._begin(owner.active.nonzero().flatten(), 'initial_active_snapshot')

    def close(self):
        for name, value, existed in reversed(self._restore):
            if existed:
                setattr(self.native, name, value)
            else:
                delattr(self.native, name)
        self._restore.clear()

    def _begin(self, ids, source):
        for row in ids.cpu().tolist():
            key = (row, int(self.env._episode_id[row]))
            if key in self.records:
                raise ValueError('Recovery evaluation requires unique native episode identities')
            self.keys[row] = key
            self.records[key] = {'env_id': row, 'episode_id': key[1], 'scenario': self.row_names[row],
                                 'birth_source': source, 'outcome': None, 'invalid_intervals': 0,
                                 'observed_intervals': 0, 'boundary_valid': None}
        for value in (self.counts, self.failed, self.only_failed, self.maximum, self.last_good,
                      self.last_valid, self.estimate_intervals, self.estimate_failed, self.last_estimate_good):
            value[ids] = 0
        self.last[ids] = float('nan')

    def _bind_pending_births(self):
        if self.pending_birth:
            ids = torch.tensor(sorted(self.pending_birth), dtype=torch.long, device=self.native.device)
            self._begin(ids, 'native_reset')
            self.pending_birth.clear()

    def _observe(self, active, values):
        c = self.native.config
        height_error = values['height_m'] - values['command_height_m']
        tilt, angular, vertical = (values[name] for name in
                                   ('tilt_rad', 'angular_speed_rad_s', 'velocity_z_m_s'))
        wheels, base = values['wheel_load_n'], values['base_load_n']
        measurements = torch.cat((torch.stack((height_error, tilt, angular, vertical), -1),
                                  wheels, base[:, None]), -1)
        finite = torch.isfinite(measurements).all(-1)
        good = torch.stack((height_error.abs() <= c.height_tolerance_m,
            (tilt >= 0) & (tilt <= c.tilt_rad),
            (angular >= 0) & (angular <= c.angular_speed_rad_s),
            vertical.abs() <= c.vertical_speed_m_s,
            (wheels >= c.wheel_load_n).all(-1), (base >= 0) & (base <= c.base_load_n)), -1)
        owned = values['policy_owned']
        valid = active & owned & finite
        self.counts[:, 0] += active.long()
        self.counts[:, 1] += valid.long()
        self.counts[:, 2] += (active & ~finite).long()
        self.counts[:, 3] += (active & ~owned).long()
        misses = ~good
        self.failed += (valid[:, None] & misses).long()
        self.only_failed += (valid[:, None] & misses & (misses.sum(-1) == 1)[:, None]).long()
        self.maximum[:, 0] = torch.maximum(self.maximum[:, 0], self.native.physical_stable_s)
        self.last = torch.where(active[:, None], measurements, self.last)
        self.last_good = torch.where(active[:, None], good, self.last_good)
        self.last_valid = torch.where(active, valid, self.last_valid)

    def _observe_estimate(self, eligible, now_s, estimate):
        # Read the previous timestamp before native finish_interval updates it.
        c = self.native.config
        good = torch.zeros_like(self.last_estimate_good)
        if estimate is not None:
            age = now_s - estimate.available_at_s
            probability = estimate.wheel_contact_probability
            good = torch.stack((estimate.valid,
                torch.isfinite(age) & (age > 0) & (age <= c.estimate_max_age_s + 1e-9),
                estimate.available_at_s > self.native._last_estimate_s,
                torch.isfinite(probability).all(-1) & (probability >= c.contact_probability).all(-1)
                    & (probability <= 1).all(-1),
                (estimate.tilt_rad >= 0) & (estimate.tilt_rad <= c.tilt_rad),
                (estimate.angular_speed_rad_s >= 0) & (estimate.angular_speed_rad_s <= c.angular_speed_rad_s),
                estimate.velocity_z_m_s.abs() <= c.vertical_speed_m_s), -1)
        self.estimate_intervals += eligible.long()
        self.estimate_failed += (eligible[:, None] & ~good).long()
        self.last_estimate_good = torch.where(eligible[:, None], good, self.last_estimate_good)

    @staticmethod
    def _number(value):
        return float(value) if bool(torch.isfinite(value)) else None

    def _snapshot(self, row):
        counts = self.counts[row].cpu().tolist()
        values = self.last[row].cpu()
        c = self.native
        return {'physical_ticks': counts[0], 'valid_owned_physical_ticks': counts[1],
            'nonfinite_physical_ticks': counts[2], 'non_policy_owned_physical_ticks': counts[3],
            'failed_condition_ticks': dict(zip(self.CONDITIONS, self.failed[row].cpu().tolist())),
            'only_failed_condition_ticks': dict(zip(self.CONDITIONS, self.only_failed[row].cpu().tolist())),
            'max_physical_stable_s': float(self.maximum[row, 0]),
            'max_estimated_stable_s': float(self.maximum[row, 1]),
            'physical_stable_at_end_s': float(c.physical_stable_s[row]),
            'estimated_stable_at_end_s': float(c.estimated_stable_s[row]),
            'policy_elapsed_at_end_s': float(c.policy_elapsed_s[row]),
            'native_verified_physical_success': bool(c._verified[row]),
            'eligible_estimate_intervals': int(self.estimate_intervals[row]),
            'failed_estimate_condition_intervals': dict(zip(
                self.ESTIMATE_CONDITIONS, self.estimate_failed[row].cpu().tolist())),
            'last_estimate_conditions': (dict(zip(self.ESTIMATE_CONDITIONS,
                self.last_estimate_good[row].cpu().tolist())) if self.estimate_intervals[row] else None),
            'last_physical_valid_owned': bool(self.last_valid[row]),
            'last_physical_values': dict(zip(self.VALUES, [self._number(v) for v in values])),
            'last_physical_conditions': (dict(zip(self.CONDITIONS, self.last_good[row].cpu().tolist()))
                                         if counts[0] else None)}

    def _end(self, row, outcome, *, estimate=None, now_s=None):
        key = self.keys[row]
        if key is None:
            return
        record = self.records[key]
        record.update(self._snapshot(row), outcome=outcome)
        if estimate is not None:
            record['last_estimate'] = {
                name: self._number(getattr(estimate, name)[row]) for name in
                ('tilt_rad', 'angular_speed_rad_s', 'velocity_z_m_s', 'available_at_s')}
            record['last_estimate'].update(valid=bool(estimate.valid[row]),
                age_s=self._number(now_s[row] - estimate.available_at_s[row]),
                wheel_contact_probability=[self._number(v) for v in estimate.wheel_contact_probability[row]])
        self.pending.add(key)
        self.keys[row] = None

    def observe_transition(self, transition, valid):
        """Attach validity after auto-reset using the completed interval's ID."""
        episodes = transition.start.episode_id.cpu().tolist()
        end_episodes = transition.end.episode_id.cpu().tolist()
        for row, episode in enumerate(episodes):
            key = (row, episode)
            record = self.records.get(key)
            if record is None or record['outcome'] is not None and key not in self.pending:
                continue
            trusted = bool(valid[row]) and episode == end_episodes[row]
            record['observed_intervals'] += 1
            record['invalid_intervals'] += int(not trusted)
            if key in self.pending:
                record['boundary_valid'] = trusted
                self.pending.remove(key)
        # First attribute the completed interval to its old episode, then bind
        # newborns against the IDs published by the now-complete native reset.
        self._bind_pending_births()

    def sample(self):
        self._bind_pending_births()
        return {'recovery_active': self.native.active.detach().clone(),
                'recovery_after_handoff': self.after_handoff.detach().clone()}

    def summary(self):
        self._bind_pending_births()
        episodes, groups = [], {}
        current = {key: row for row, key in enumerate(self.keys) if key is not None}
        for key, saved in self.records.items():
            record = dict(saved)
            if key in current:
                record.update(self._snapshot(current[key]), outcome='active_at_evaluation_end')
            record['had_invalid'] = bool(record['invalid_intervals'] or record['nonfinite_physical_ticks']
                                         or record['boundary_valid'] is False)
            record['validated_boundary'] = record['boundary_valid'] is True and not record['had_invalid']
            episodes.append(record)
            group = groups.setdefault(record['scenario'], {'births': 0, 'raw_outcomes': {},
                'validated_outcomes': {}, 'validated_verified_episodes': 0, 'episodes_with_invalid': 0})
            group['births'] += 1
            group['episodes_with_invalid'] += int(record['had_invalid'])
            outcome = record['outcome']
            group['raw_outcomes'][outcome] = group['raw_outcomes'].get(outcome, 0) + 1
            if record['validated_boundary']:
                group['validated_outcomes'][outcome] = group['validated_outcomes'].get(outcome, 0) + 1
                group['validated_verified_episodes'] += int(record['native_verified_physical_success'])
        return {'configuration': asdict(self.native.config), 'by_scenario': groups, 'episodes': episodes,
            'scope': 'Per native recovery episode, including initial active births and automatic resets. Physical evidence stops at native active=false; no post-handoff certification is invented.',
            'validity': 'Raw native outcomes are retained. Validated boundaries exclude any invalid interval/nonfinite physical tick in that recovery episode and require a valid observed ending interval. Non-policy-owned ticks are counted separately and excluded from predicate statistics.',
            'condition_units': 'Failed/only-failed counts are physical ticks, not independent trials; maxima include every observed substep, not just policy endpoints.'}
