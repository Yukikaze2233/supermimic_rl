#!/usr/bin/env python3
"""Export online learning curves while separating mixed collection from student evidence."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.metrics:
        previous_teacher = previous_student = 0
        for line in path.read_text().splitlines():
            row = json.loads(line)
            teacher_total = row['teacher_transitions_since_cold_start']
            student_total = row['student_transitions_since_cold_start']
            teacher = row.get('teacher_transitions', teacher_total - previous_teacher)
            student = row.get('student_transitions', student_total - previous_student)
            previous_teacher, previous_student = teacher_total, student_total
            row = {**row, 'source_metrics': str(path), 'measured_teacher_fraction': teacher / (teacher + student)}
            rows.append(row)
    rows.sort(key=lambda row: row['cycle'])
    updates = np.array([row['cycle'] for row in rows])
    if len(set(updates)) != len(updates):
        raise ValueError('Select one authoritative metric row per online cycle')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(3, 2, figsize=(12, 10), layout='constrained')
    figure.suptitle('Full-vehicle SuperMimic | native collection + imagined policy learning', fontsize=15)
    panels = axes.flatten()
    panels[0].plot(updates, [r['real_transitions'] / 1e6 for r in rows], color='#23679c')
    panels[0].set(title='Online native transitions', ylabel='Million transitions (seed data excluded)')
    panels[1].plot(updates, [r['collection']['reward_mean'] for r in rows], color='#23679c')
    panels[1].set(title='Collection reward: teacher + student mixture', ylabel='Mean reward / 20 ms')
    panels[2].plot(updates, [r['measured_teacher_fraction'] * 100 for r in rows], label='Measured teacher share')
    panels[2].plot(updates, [r['teacher_target_fraction'] * 100 for r in rows], '--', label='Configured target')
    panels[2].set(title='Actual controller allocation', ylabel='Teacher transitions (%)', ylim=(0, 100))
    panels[2].legend(frameon=False)
    previous_transitions = 0
    failures = []
    for row in rows:
        delta = row['real_transitions'] - previous_transitions
        failures.append(1000 * row['collection']['terminations'] / delta)
        previous_transitions = row['real_transitions']
    panels[3].plot(updates, failures, color='#ad4c34')
    panels[3].set(title='Collection failures: teacher + student mixture', ylabel='Terminations / 1,000 transitions')
    timings = [[r['seconds'][key] for r in rows] for key in ('collection', 'world', 'policy')]
    panels[4].stackplot(updates, *timings, labels=['Native collection', 'World fitting', 'Policy + critic + estimator'],
                        colors=['#7da9bf', '#72a688', '#d7aa60'])
    panels[4].set(title='Time per 4096 x 24 collection/update cycle', ylabel='Seconds')
    panels[4].legend(frameon=False, fontsize=8, loc='upper left')
    panels[5].plot(updates, [(r['policy'].get('imagination/survival')
                             if r['policy'].get('imagination/survival') is not None else float('nan')) * 100
                            for r in rows], color='#7566a6')
    panels[5].set(title='Imagined H16 survival (model estimate)', ylabel='Surviving imagined rows (%)', ylim=(0, 100))
    warmup_spans = []
    for row in rows:
        if row.get('policy_learning', True):
            continue
        cycle = row['cycle']
        if warmup_spans and cycle == warmup_spans[-1][1] + 1:
            warmup_spans[-1][1] = cycle
        else:
            warmup_spans.append([cycle, cycle])
    for axis in panels:
        axis.set_xlabel('Online cycle (24 native vector steps)')
        axis.set_xlim(max(0, updates.min() - 1), updates.max() + 1)
        for first, last in warmup_spans:
            axis.axvspan(first - .5, last + .5, color='#888888', alpha=.12, zorder=0)
        axis.grid(alpha=.2)
    caption = 'Mixed reward and imagined survival do not establish native student capability.'
    if warmup_spans:
        caption += '\nShaded cycles: world model + estimator warmup; actor decoder and critic frozen.'
    figure.supxlabel(caption, fontsize=10)
    args.output.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output / 'learning-curves.svg')
    figure.savefig(args.output / 'learning-curves.png', dpi=150)
    report = {'sources': [str(path.resolve()) for path in args.metrics], 'cycles': len(rows),
              'last_actor_updates': rows[-1]['updates']['actor'], 'last_real_transitions': rows[-1]['real_transitions'],
              'scope': 'training diagnostics only; capability requires separate student-only native evaluation',
              'metrics': rows}
    (args.output / 'curve-data.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'metrics'}))


if __name__ == '__main__':
    main()
