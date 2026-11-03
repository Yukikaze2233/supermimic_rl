#!/usr/bin/env python3
"""Create the explicit a05 to R7 task handoff from an actual native target dataset."""
import argparse
import json
from pathlib import Path

import torch

from supermimic_rl.runtime import atomic_save
from supermimic_rl.world_model_v2.migrations.task_semantics import upgrade_task_semantics
from supermimic_rl.world_model_v2.teacher import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'reference-dataset', 'source-runtime', 'target-runtime',
                 'source-course', 'target-course', 'source-teacher', 'target-teacher', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--course-coordinate', type=int, required=True)
    parser.add_argument('--critic-warmup-cycles', type=int, required=True)
    parser.add_argument('--sampler-device', default='cpu',
                        help='Original native sampler RNG device, e.g. cuda:0 for CUDA Philox checkpoints')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Task migration requires a new output directory')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    source_replay = (args.checkpoint.parent / saved['replay_file']).resolve(strict=True)
    dataset = torch.load(source_replay, map_location='cpu', weights_only=True, mmap=True)
    reference = torch.load(args.reference_dataset, map_location='cpu', weights_only=True, mmap=True)
    upgraded, replay, receipt = upgrade_task_semantics(saved, dataset, reference,
        **{name + '_raw': getattr(args, name).read_bytes() for name in
           ('source_runtime', 'target_runtime', 'source_course', 'target_course')},
        source_teacher_path=args.source_teacher, target_teacher_path=args.target_teacher,
        course_coordinate=args.course_coordinate, critic_warmup_cycles=args.critic_warmup_cycles,
        sampler_device=args.sampler_device)
    for name in ('checkpoint', 'reference_dataset', 'source_runtime', 'target_runtime',
                 'source_course', 'target_course', 'source_teacher', 'target_teacher'):
        path = getattr(args, name)
        receipt[name] = {'path': str(path.resolve()), 'sha256': file_sha256(path)}
    receipt['source_replay'] = str(source_replay)
    args.output.mkdir(parents=True)
    upgraded['replay_file'] = 'replay-a.pt'
    atomic_save(replay, args.output / 'replay-a.pt')
    atomic_save(upgraded, args.output / 'resume-a.pt')
    (args.output / 'migration.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (args.output / 'online-config.json').write_text(json.dumps(upgraded['config'], indent=2) + '\n')
    (args.output / 'resume.json').write_text(json.dumps({
        'slot': 'a', 'cycle': upgraded['cycle'], 'checkpoint': 'resume-a.pt', 'replay_file': 'replay-a.pt',
        'physics_resume': 'cold native reset; all legacy task starts invalid; fresh critic warmup'}, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
