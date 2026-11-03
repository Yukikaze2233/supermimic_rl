#!/usr/bin/env python3
"""Create an explicit recorded-recovery branch from a complete chassis35 jump pair."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from supermimic_rl.runtime import atomic_save
from supermimic_rl.world_model_v2.migrations.recovery import upgrade_recovery_tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference-dataset', type=Path, required=True)
    parser.add_argument('--course', type=Path, required=True,
                        help='The unchanged pinned jump-first course used by both source and reference')
    parser.add_argument('--course-coordinate', type=int, required=True)
    parser.add_argument('--world-only-cycles', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Recovery expansion must use a new output directory')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    source_replay = (args.checkpoint.parent / saved['replay_file']).resolve(strict=True)
    dataset = torch.load(source_replay, map_location='cpu', weights_only=True, mmap=True)
    reference = torch.load(args.reference_dataset, map_location='cpu', weights_only=True, mmap=True)
    upgraded, replay, receipt = upgrade_recovery_tasks(saved, dataset, reference,
        course_raw=args.course.read_bytes(), course_coordinate=args.course_coordinate,
        world_only_cycles=args.world_only_cycles)
    for name, path in (('source_checkpoint', args.checkpoint), ('reference_dataset', args.reference_dataset)):
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        receipt[name], receipt[name + '_sha256'] = str(path.resolve()), digest
    receipt['source_replay'] = str(source_replay)
    args.output.mkdir(parents=True)
    upgraded['replay_file'] = 'replay-a.pt'
    atomic_save(replay, args.output / 'replay-a.pt')
    atomic_save(upgraded, args.output / 'resume-a.pt')
    actor = {key: value for key, value in upgraded.items() if key in (
        'model_contract', 'schema', 'schema_fingerprint', 'config', 'cycle', 'resume_migrations')}
    actor.update(format='supermimic-online-v2-actor', actor=upgraded['learner']['actor'],
                 updates=upgraded['learner']['updates'])
    atomic_save(actor, args.output / f"actor-{upgraded['cycle']:06d}.pt")
    (args.output / 'migration.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (args.output / 'online-config.json').write_text(json.dumps(upgraded['config'], indent=2) + '\n')
    (args.output / 'resume.json').write_text(json.dumps({
        'slot': 'a', 'checkpoint': 'resume-a.pt', 'replay_file': 'replay-a.pt', 'cycle': upgraded['cycle'],
        'physics_resume': 'explicit cold reset at selected course coordinate; old recovery records invalid'},
        indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
