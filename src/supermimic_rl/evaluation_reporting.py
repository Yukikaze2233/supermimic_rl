"""Keep platform and flat-anchor evaluation evidence distinct in either loop."""
from __future__ import annotations

import torch


class PlatformEvaluationReport:
    """Capture post-begin initial membership; never use it as a full-run mask."""

    def __init__(self, platform, num_envs):
        self.platform = platform
        self.rows = None
        self.mixed = getattr(getattr(platform, 'plan', None), 'sampler', None) is not None
        if platform is not None:
            rows = platform.platform_rows
            if (not isinstance(rows, torch.Tensor) or rows.shape != (num_envs,) or rows.dtype != torch.bool):
                raise ValueError('Platform evaluation requires explicit planned native rows')
            self.rows = rows.nonzero().flatten().cpu().tolist()

    def apply(self, report, *, recorded_physics_ticks):
        platform, platform_rows, mixed_platform = self.platform, self.rows, self.mixed
        fixed_suite = 'task_suite' in report
        recording = recorded_physics_ticks
        if platform is not None:
            summary = getattr(platform, 'summary', None)
            report['platform_diagnostics'] = (summary() if callable(summary) else {
                'status': 'unknown', 'reason': 'The native platform owner has no diagnostic summary'})
            report['platform_evaluation_scope'] = {
                'planned_env_ids': platform_rows, 'recorded_physics_ticks': recording,
                'capability_source': 'platform_diagnostics only; final snapshots are not a success rate',
                'legacy_jump_events': 'Unchanged native jump pulses and aggregate counters are not '
                                      'an all-task or selected-platform success numerator'}
            if mixed_platform:
                report['platform_evaluation_scope'].update(
                    planned_env_ids_scope='initial episode plans after evaluation_begin only',
                    episode_assignment_source='per-interval typed platform intent; reset may change membership')
            if fixed_suite:
                suite = dict(report['task_suite'])
                if 'jump_diagnostics' in suite:
                    diagnostic = dict(suite.pop('jump_diagnostics'))
                    diagnostic['platform_interpretation'] = {
                        'planned_env_ids': platform_rows, 'includes_platform_rows': bool(platform_rows),
                        'scope': 'Flat-anchor jump conditions include selected-platform rows; '
                                 'they do not qualify platform flight, landing or success. '
                                 'Original values are retained without recomputing outcomes.'}
                    if mixed_platform:
                        diagnostic['platform_interpretation'].update(
                            includes_platform_rows=None,
                            initial_includes_platform_rows=bool(platform_rows),
                            planned_env_ids_scope='initial episode plans only; not a full-run exclusion mask',
                            scope='Mixed-episode membership must come from typed per-interval intent. '
                                  'Initial plans do not partition these aggregate flat-anchor outcomes. '
                                  'Original values are retained and do not qualify platform success.')
                    suite['flat_anchor_jump_diagnostics'] = diagnostic
                suite['success_semantics'] = (suite.get('success_semantics', '') +
                    ' Selected-platform capability is reported separately in platform_diagnostics; '
                    'legacy jump events and mixed counters do not certify all-task success.')
                report['task_suite'] = suite
