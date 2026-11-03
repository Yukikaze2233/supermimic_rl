#!/usr/bin/env python3
"""Create a distinct chassis-only actor branch from a complete recovery pair."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import torch

from supermimic_rl.runtime import atomic_save
from supermimic_rl.world_model_v2.migrations.actor import inject_fitted_world, restrict_actor_to_chassis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay-storage', choices=('hardlink', 'copy'), default='hardlink',
                        help='Replay bytes are unchanged; hardlink avoids duplicating the immutable checkpoint')
    parser.add_argument('--replay-device', choices=('learner', 'cpu'),
                        help='Optional explicit residency change for the new policy branch')
    parser.add_argument('--imagination-microbatch-size', type=int)
    parser.add_argument('--sampling-seed', type=int,
                        help='Required for CPU replay migration; validation uses this seed plus one')
    parser.add_argument('--world-fit-checkpoint', type=Path,
                        help='Inject WM/Adam/counters fitted on this exact online pair; retain policy ancestry')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Actor branch must use a new recovery directory')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    replay_path = (args.checkpoint.parent / saved['replay_file']).resolve(strict=True)
    dataset = torch.load(replay_path, map_location='cpu', weights_only=True, mmap=True)
    upgraded, _, receipt = restrict_actor_to_chassis(saved, dataset, replay_device=args.replay_device,
        imagination_microbatch_size=args.imagination_microbatch_size, sampling_seed=args.sampling_seed)
    with args.checkpoint.open('rb') as stream:
        checkpoint_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    receipt.update(source_checkpoint=str(args.checkpoint.resolve()), source_replay=str(replay_path),
                   source_checkpoint_sha256=checkpoint_sha256,
                   replay_file_storage=args.replay_storage)
    if args.world_fit_checkpoint is not None:
        world = torch.load(args.world_fit_checkpoint, map_location='cpu', weights_only=True, mmap=True)
        upgraded, world_receipt = inject_fitted_world(upgraded, dataset, world,
            source_checkpoint_sha256=checkpoint_sha256)
        if not Path(world['source']['replay']).samefile(replay_path):
            raise ValueError('World fitting and actor migration must use the same immutable paired replay')
        with args.world_fit_checkpoint.open('rb') as stream:
            world_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
        world_receipt.update(checkpoint=str(args.world_fit_checkpoint.resolve()), checkpoint_sha256=world_sha256)
        receipt['world_fit'] = world_receipt
    args.output.mkdir(parents=True)
    target_replay = args.output / 'replay-a.pt'
    if args.replay_storage == 'hardlink':
        os.link(replay_path, target_replay)
    else:
        shutil.copy2(replay_path, target_replay)
    upgraded['replay_file'] = target_replay.name
    atomic_save(upgraded, args.output / 'resume-a.pt')
    actor = {key: value for key, value in upgraded.items() if key in (
        'model_contract', 'schema', 'schema_fingerprint', 'config', 'cycle')}
    actor.update(format='supermimic-online-v2-actor', actor=upgraded['learner']['actor'],
                 updates=upgraded['learner']['updates'], actor_branch=receipt)
    atomic_save(actor, args.output / f"actor-{upgraded['cycle']:06d}.pt")
    (args.output / 'migration.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (args.output / 'online-config.json').write_text(json.dumps(upgraded['config'], indent=2) + '\n')
    (args.output / 'resume.json').write_text(json.dumps({
        'slot': 'a', 'checkpoint': 'resume-a.pt', 'replay_file': target_replay.name, 'cycle': upgraded['cycle'],
        'physics_resume': 'explicit cold reset; newly initialized chassis-only actor branch'}, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
