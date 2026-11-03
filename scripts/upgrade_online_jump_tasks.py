#!/usr/bin/env python3
"""Create an explicit recorded-jump branch from a complete chassis35 stair pair."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from supermimic_rl.runtime import atomic_save
from supermimic_rl.world_model_v2.migrations.actor import inject_fitted_world
from supermimic_rl.world_model_v2.migrations.jump import upgrade_jump_tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference-dataset', type=Path, required=True)
    parser.add_argument('--source-course', type=Path, required=True)
    parser.add_argument('--target-course', type=Path, required=True)
    parser.add_argument('--course-coordinate', type=int, required=True)
    parser.add_argument('--world-only-cycles', type=int, default=0)
    parser.add_argument('--world-fit-checkpoint', type=Path,
                        help='Join WM/Adam fitted on this exact source pair before changing the task schema')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Jump migration must use a new recovery directory')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    source_replay = (args.checkpoint.parent / saved['replay_file']).resolve(strict=True)
    dataset = torch.load(source_replay, map_location='cpu', weights_only=True, mmap=True)
    reference = torch.load(args.reference_dataset, map_location='cpu', weights_only=True, mmap=True)
    if args.world_fit_checkpoint is not None:
        world = torch.load(args.world_fit_checkpoint, map_location='cpu', weights_only=True, mmap=True)
        if not Path(world['source']['replay']).samefile(source_replay):
            raise ValueError('World fit and jump migration must use the same immutable paired replay')
        with args.checkpoint.open('rb') as stream:
            checkpoint_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
        saved, world_receipt = inject_fitted_world(saved, dataset, world,
            source_checkpoint_sha256=checkpoint_sha256)
        with args.world_fit_checkpoint.open('rb') as stream:
            world_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
        world_receipt.update(checkpoint=str(args.world_fit_checkpoint.resolve()),
                             checkpoint_sha256=world_sha256)
    upgraded, replay, receipt = upgrade_jump_tasks(saved, dataset, reference,
        source_course_raw=args.source_course.read_bytes(), target_course_raw=args.target_course.read_bytes(),
        course_coordinate=args.course_coordinate, world_only_cycles=args.world_only_cycles)
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
        'physics_resume': 'explicit cold reset at the selected course coordinate; old jump records invalid'},
        indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
