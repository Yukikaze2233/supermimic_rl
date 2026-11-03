#!/usr/bin/env python3
"""Copy a complete foundation recovery pair into an explicit task-aware continuation."""
import argparse
import json
from pathlib import Path

import torch

from supermimic_rl.runtime import atomic_save
from supermimic_rl.world_model_v2.migrations.motion import upgrade_motion_tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Migration must create a separate recovery directory')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    replay_path = args.checkpoint.parent / saved['replay_file']
    dataset = torch.load(replay_path, map_location='cpu', weights_only=True, mmap=True)
    upgraded, replay, receipt = upgrade_motion_tasks(saved, dataset)
    args.output.mkdir(parents=True)
    upgraded['replay_file'] = 'replay-a.pt'
    atomic_save(replay, args.output / 'replay-a.pt')
    atomic_save(upgraded, args.output / 'resume-a.pt')
    receipt = {**receipt, 'source_checkpoint': str(args.checkpoint.resolve()), 'source_replay': str(replay_path.resolve())}
    (args.output / 'migration.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (args.output / 'online-config.json').write_text(json.dumps(upgraded['config'], indent=2) + '\n')
    (args.output / 'resume.json').write_text(json.dumps({
        'slot': 'a', 'checkpoint': 'resume-a.pt', 'replay_file': 'replay-a.pt', 'cycle': upgraded['cycle'],
        'physics_resume': 'explicit cold reset; old transitions have no task imagination state'}, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
