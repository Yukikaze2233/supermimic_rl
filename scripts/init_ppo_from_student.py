#!/usr/bin/env python3
"""Write a PPO actor-initialization artifact from a 5x35 V2 student actor."""
import argparse
import hashlib
import json
from pathlib import Path

from supermimic_rl.runtime import atomic_save
from supermimic_rl.world_model_v2.student_transfer import actor_init_payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, required=True, help="V2 actor-XXXXXX.pt (chassis 35D)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-bounds", type=float, nargs=6,
                        help="Target PPO bounds (legs then wheels); default keeps the student's legacy bounds")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new path for the initialization artifact")
    payload = (actor_init_payload(args.student) if args.action_bounds is None
               else actor_init_payload(args.student, tuple(args.action_bounds)))
    atomic_save(payload, args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    receipt = {"path": str(args.output.resolve()), "sha256": digest, "source": payload["source"],
               "equivalence": payload["equivalence"]}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
