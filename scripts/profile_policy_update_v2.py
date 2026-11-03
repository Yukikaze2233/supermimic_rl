#!/usr/bin/env python3
"""Profile disposable saved-checkpoint policy updates through the existing audit.

Pass audit arguments after ``--``. No checkpoint, optimizer, or replay artifact
is written. Section timers synchronize CUDA by default; their inclusive times
overlap and must not be added together. PyTorch profiling is optional because a
complete imagined update can generate a large event stream.
"""
from __future__ import annotations

import argparse
import cProfile
from contextlib import ExitStack
from functools import wraps
import json
from pathlib import Path
import pstats
import runpy
import sys
import time
from unittest.mock import patch

import torch

from supermimic_rl.world_model_v2.imagination import Imagination
from supermimic_rl.world_model_v2.learning import Learner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile-output', type=Path, required=True)
    parser.add_argument('--torch-profile', action='store_true')
    parser.add_argument('--record-shapes', action='store_true')
    parser.add_argument('--trace', action='store_true', help='Export optional Chrome trace (can be large)')
    parser.add_argument('--no-section-sync', action='store_true')
    parser.add_argument('--top', type=int, default=40)
    parser.add_argument('audit_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    audit_args = args.audit_args[1:] if args.audit_args[:1] == ['--'] else args.audit_args
    if not audit_args or args.top < 1:
        parser.error('Supply audit arguments after -- and a positive --top')
    if (args.record_shapes or args.trace) and not args.torch_profile:
        parser.error('--record-shapes/--trace require --torch-profile')
    if args.profile_output.exists():
        raise FileExistsError(args.profile_output)
    args.profile_output.parent.mkdir(parents=True, exist_ok=True)
    result = {'scope': 'disposable-copy update profile; no native physics or retained weights',
              'cuda_section_sync': not args.no_section_sync,
              'torch_profiler': args.torch_profile, 'record_shapes': args.record_shapes,
              'timing_scope': 'includes cProfile/range/profiler instrumentation overhead; not an uninstrumented benchmark',
              'sections_are_inclusive': True, 'updates': []}
    active = False
    device = None
    sections = {}
    optimizer_context = []

    def synchronize():
        if device is not None and device.type == 'cuda':
            torch.cuda.synchronize(device)

    def section(original, label, *, optimizer=False):
        @wraps(original)
        def wrapped(*values, **kwargs):
            if not active:
                return original(*values, **kwargs)
            name = label(*values, **kwargs) if callable(label) else label
            if not args.no_section_sync:
                synchronize()
            started = time.perf_counter()
            if optimizer:
                optimizer_context.append(values[1])
            try:
                with torch.profiler.record_function(name):
                    return original(*values, **kwargs)
            finally:
                if optimizer:
                    optimizer_context.pop()
                if not args.no_section_sync:
                    synchronize()
                row = sections.setdefault(name, {'calls': 0, 'seconds': 0.})
                row['calls'] += 1
                row['seconds'] += time.perf_counter() - started
        return wrapped

    original_update = Learner.update_policy

    @wraps(original_update)
    def profiled_update(learner, *values, **kwargs):
        nonlocal active, device, sections
        device = next(learner.actor.parameters()).device
        synchronize()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        sections = {}
        cpu_profile = cProfile.Profile()
        index = len(result['updates']) + 1
        row = {'attempt': index, 'device': str(device), 'sections': sections}
        profiler = None
        started = time.perf_counter()
        active = True
        try:
            with ExitStack() as stack:
                if args.torch_profile:
                    activities = [torch.profiler.ProfilerActivity.CPU]
                    if device.type == 'cuda':
                        activities.append(torch.profiler.ProfilerActivity.CUDA)
                    profiler = stack.enter_context(torch.profiler.profile(
                        activities=activities, record_shapes=args.record_shapes,
                        profile_memory=False, with_stack=False))
                cpu_profile.enable()
                try:
                    return original_update(learner, *values, **kwargs)
                finally:
                    synchronize()
                    cpu_profile.disable()
                    row['update_wall_s'] = time.perf_counter() - started
        finally:
            active = False
            stats = pstats.Stats(cpu_profile)
            row['cprofile'] = [
                {'file': key[0], 'line': key[1], 'function': key[2],
                 'primitive_calls': value[0], 'calls': value[1],
                 'self_seconds': value[2], 'cumulative_seconds': value[3]}
                for key, value in sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)[:args.top]]
            if device.type == 'cuda':
                row['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
                row['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
            if profiler is not None:
                averages = profiler.key_averages(group_by_input_shape=args.record_shapes)
                prefix = args.profile_output.with_suffix('')
                outputs = {}
                for suffix, order in [('cpu', 'self_cpu_time_total'), ('cuda', 'self_device_time_total')]:
                    path = Path(f'{prefix}-update{index}-{suffix}.txt')
                    path.write_text(averages.table(sort_by=order, row_limit=args.top) + '\n')
                    outputs[suffix] = str(path)
                if args.trace:
                    path = Path(f'{prefix}-update{index}-trace.json')
                    profiler.export_chrome_trace(str(path))
                    outputs['trace'] = str(path)
                row['torch_tables'] = outputs
            result['updates'].append(row)
            args.profile_output.write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps({'profile': str(args.profile_output), 'attempt': index,
                              'seconds': row.get('update_wall_s'), 'sections': sections}), flush=True)

    rollout_name = lambda *values, **kwargs: (
        'policy/rollout_screen' if kwargs.get('screen_invalid') else 'policy/rollout_gradient')
    backward_name = lambda *values, **kwargs: 'policy/backward_' + (
        optimizer_context[-1] if optimizer_context else 'actor')
    audit_script = Path(__file__).with_name('check_online_world_v2.py')
    with ExitStack() as stack:
        stack.enter_context(patch.object(Learner, 'update_policy', profiled_update))
        stack.enter_context(patch.object(Learner, '_usable_rollout',
            section(Learner._usable_rollout, 'policy/usable_rollout')))
        stack.enter_context(patch.object(Imagination, 'rollout', section(Imagination.rollout, rollout_name)))
        stack.enter_context(patch.object(Learner, '_step', section(Learner._step,
            lambda *values, **kwargs: 'policy/optimizer_' + values[1], optimizer=True)))
        stack.enter_context(patch.object(Learner, 'fit_estimator',
            section(Learner.fit_estimator, 'policy/estimator')))
        stack.enter_context(patch.object(torch.Tensor, 'backward', section(torch.Tensor.backward, backward_name)))
        stack.enter_context(patch.object(sys, 'argv', [str(audit_script), *audit_args]))
        runpy.run_path(str(audit_script), run_name='__main__')


if __name__ == '__main__':
    main()
