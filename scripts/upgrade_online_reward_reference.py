#!/usr/bin/env python3
"""Migrate a full online pair to the verified jump reward-reference correction."""
import argparse
import json
from pathlib import Path

import torch

from supermimic_rl.runtime import atomic_save
from supermimic_rl.world_model_v2.migrations.reward_reference import upgrade_recovery_quiet, upgrade_reward_reference
from supermimic_rl.world_model_v2.teacher import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'reference-dataset', 'source-teacher', 'target-teacher', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--change', choices=('jump-reference', 'recovery-quiet'), default='jump-reference')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Reward-reference migration requires a new output directory')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    source_replay = (args.checkpoint.parent / saved['replay_file']).resolve(strict=True)
    dataset = torch.load(source_replay, map_location='cpu', weights_only=True, mmap=True)
    reference = torch.load(args.reference_dataset, map_location='cpu', weights_only=True, mmap=True)
    migrate = upgrade_recovery_quiet if args.change == 'recovery-quiet' else upgrade_reward_reference
    upgraded, replay, receipt = migrate(saved, dataset, reference,
        source_teacher_path=args.source_teacher, target_teacher_path=args.target_teacher)
    for name in ('checkpoint', 'reference_dataset', 'source_teacher', 'target_teacher'):
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
        'physics_resume': 'cold native reset; learning state retained; no new warmup'}, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
