#!/usr/bin/env python3
"""Create an explicitly gated continuation while preserving old dynamics replay."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from supermimic_rl.runtime import atomic_save
from supermimic_rl.world_model_v2.migrations.jump_trigger import upgrade_jump_trigger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference-dataset', type=Path, required=True)
    parser.add_argument('--world-only-cycles', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Gate migration must preserve its original recovery pair')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    source_replay = (args.checkpoint.parent / saved['replay_file']).resolve(strict=True)
    dataset = torch.load(source_replay, map_location='cpu', weights_only=True, mmap=True)
    reference = torch.load(args.reference_dataset, map_location='cpu', weights_only=True, mmap=True)
    upgraded, replay, receipt = upgrade_jump_trigger(
        saved, dataset, reference, world_only_cycles=args.world_only_cycles)
    for name, path in (('source_checkpoint', args.checkpoint), ('reference_dataset', args.reference_dataset)):
        with path.open('rb') as stream:
            receipt[name + '_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
        receipt[name] = str(path.resolve())
    receipt['source_replay'] = str(source_replay)
    args.output.mkdir(parents=True)
    upgraded['replay_file'] = 'replay-a.pt'
    atomic_save(replay, args.output / 'replay-a.pt')
    atomic_save(upgraded, args.output / 'resume-a.pt')
    (args.output / 'migration.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (args.output / 'online-config.json').write_text(json.dumps(upgraded['config'], indent=2) + '\n')
    (args.output / 'resume.json').write_text(json.dumps({
        'slot': 'a', 'cycle': upgraded['cycle'], 'checkpoint': 'resume-a.pt', 'replay_file': 'replay-a.pt',
        'physics_resume': 'cold native reset; legacy gate histories invalid, fresh rows required'}, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
