#!/usr/bin/env python3
"""Export an own PPO teacher with exact native evaluation evidence and declared scope."""
import argparse
import json
from pathlib import Path

from supermimic_rl.runtime import atomic_save
from supermimic_rl.world_model_v2.teacher import export_ppo_teacher_payload, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--qualified-for', action='append', required=True,
                        help='Reviewed native capability scope, e.g. foundation; repeat for additional scopes')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new path for the pinned teacher artifact')
    payload = export_ppo_teacher_payload(args.checkpoint, args.evaluation, args.qualified_for)
    atomic_save(payload, args.output)
    receipt = {'path': str(args.output.resolve()), 'sha256': file_sha256(args.output), 'source': payload['source']}
    args.output.with_suffix('.manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'teacher': receipt['path'], 'sha256': receipt['sha256'],
                      'qualified_for': payload['source']['qualified_for']}))


if __name__ == '__main__':
    main()
