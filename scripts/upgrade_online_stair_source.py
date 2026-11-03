#!/usr/bin/env python3
"""Copy a complete motion checkpoint into a pinned terrain/source continuation."""
import argparse
import json
from pathlib import Path

import torch

from supermimic_rl.runtime import atomic_save
from supermimic_rl.world_model_v2.migrations.stair import upgrade_stair_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference-dataset', type=Path, required=True)
    parser.add_argument('--atlas', type=Path, required=True)
    parser.add_argument('--source-runtime', type=Path, required=True)
    parser.add_argument('--target-runtime', type=Path, required=True)
    parser.add_argument('--course-coordinate', type=int, required=True)
    parser.add_argument('--world-only-cycles', type=int, required=True)
    parser.add_argument('--microbatch-size', type=int, default=512)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Stair migration must preserve its original recovery pair')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    source_replay = args.checkpoint.parent / saved['replay_file']
    dataset = torch.load(source_replay, map_location='cpu', weights_only=True, mmap=True)
    reference = torch.load(args.reference_dataset, map_location='cpu', weights_only=True, mmap=True)
    upgraded, replay, receipt = upgrade_stair_source(saved, dataset, reference, args.atlas.read_bytes(),
        course_coordinate=args.course_coordinate, world_only_cycles=args.world_only_cycles,
        source_runtime_raw=args.source_runtime.read_bytes(), target_runtime_raw=args.target_runtime.read_bytes(),
        microbatch_size=args.microbatch_size)
    args.output.mkdir(parents=True)
    upgraded['replay_file'] = 'replay-a.pt'
    atomic_save(replay, args.output / 'replay-a.pt')
    atomic_save(upgraded, args.output / 'resume-a.pt')
    receipt.update(source_checkpoint=str(args.checkpoint.resolve()), source_replay=str(source_replay.resolve()),
                   reference_dataset=str(args.reference_dataset.resolve()))
    (args.output / 'migration.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (args.output / 'online-config.json').write_text(json.dumps(upgraded['config'], indent=2) + '\n')
    (args.output / 'resume.json').write_text(json.dumps({'slot': 'a', 'cycle': upgraded['cycle'],
        'checkpoint': 'resume-a.pt', 'replay_file': 'replay-a.pt',
        'physics_resume': 'explicit cold reset at the selected course coordinate; old source states invalid'}, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
