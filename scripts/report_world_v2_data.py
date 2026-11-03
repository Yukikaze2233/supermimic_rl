#!/usr/bin/env python3
"""Per-skill coverage and world-model prediction report for recorded V2 data.

Reads a compact/packed dataset (optionally with an offline world checkpoint)
or a complete online pair (its replay and world model). Prediction uses the
held-out environment rows of the original fit. No Isaac, no training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from supermimic_rl.world_model_v2.data_report import report
from supermimic_rl.world_model_v2.records import MODEL_CONTRACT


def offline_world_source(path, schema):
    """Load a fitted model independently of the replay's online generation."""
    world = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    contract = world.get("model_contract", {})
    if (world.get("format") != "supermimic-offline-world-v2"
            or contract.get("version") != MODEL_CONTRACT
            or contract.get("schema_fingerprint") != schema.fingerprint):
        raise ValueError("World checkpoint and dataset contracts differ")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    source = {"kind": "offline_world_checkpoint", "path": str(path.resolve()),
              "sha256": digest, "iteration": world["iteration"],
              "training_source": world.get("source"),
              "training_env_ids": world["configuration"]["train_env_ids"],
              "validation_env_ids": world["configuration"]["validation_env_ids"],
              "contact_floor": "none; original online floors belong to different model weights"}
    return world["model"], contract["hidden"], world["configuration"]["validation_fraction"], source


def checkpoint_contact_floor(learner, device):
    """Restore the same saved floors the online learner uses in imagination."""
    floor = learner.get('contact_floor')
    return None if floor is None else (floor['body'].to(device), floor['base'].to(device))


def pinned_termination(schema, runtime_path):
    """Use recorded runtime limits, never guessed contact confirmation times."""
    if runtime_path is None:
        return None
    from supermimic_rl.world_model_v2.termination import PoseContactLimits, PoseContactTermination

    raw = runtime_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != json.loads(schema.source)['runtime_sha256']:
        raise ValueError('Contact audit runtime differs from the recorded SHA256')
    return PoseContactTermination(PoseContactLimits.from_runtime(json.loads(raw)),
                                  schema.policy_dt_s / schema.substeps_per_step)


def _model(schema, args, hidden, state, device):
    from supermimic_rl.world_model_v2.execution import WheelExecution
    from supermimic_rl.world_model_v2.features import MotionInputs
    from supermimic_rl.world_model_v2.fitting import WorldModelFit
    from supermimic_rl.world_model_v2.leg_linkage import LegLinkage
    from supermimic_rl.world_model_v2.model import WorldModelV2

    source = json.loads(schema.source)
    terrain = None
    if args.terrain_atlas is not None:
        from supermimic_rl.world_model_v2.terrain import AtlasTerrain
        terrain = AtlasTerrain.load(args.terrain_atlas, expected_sha256=source["terrain_sha256"], device=device)
    model = WorldModelV2(
        MotionInputs(LegLinkage.load(args.linkage, expected_asset_manifest_sha256=source.get("asset_manifest_sha256"))),
        WheelExecution.from_bundle(args.identified_bundle, schema.policy_dt_s / schema.substeps_per_step),
        hidden=tuple(hidden), terrain=terrain).to(device)
    model.load_state_dict(state)
    return WorldModelFit(model.eval())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, help="Compact or packed native dataset")
    parser.add_argument("--world-checkpoint", type=Path,
                        help="Offline fit *.checkpoint.pt; overrides the model for either replay source")
    parser.add_argument("--online-checkpoint", type=Path, help="Complete online pair (replay and world model)")
    parser.add_argument("--linkage", type=Path, required=True)
    parser.add_argument("--identified-bundle", type=Path, default=Path("configs/identified/20261031"))
    parser.add_argument("--terrain-atlas", type=Path)
    parser.add_argument("--runtime-json", type=Path,
                        help="Pinned runtime JSON to audit the native contact confirmation clock")
    parser.add_argument("--validation-fraction", type=float, default=.125)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.replay is None) == (args.online_checkpoint is None):
        parser.error("Choose --replay or --online-checkpoint")
    from supermimic_rl.contracts import StateSchema
    from supermimic_rl.world_model_v2.replay import CompactReplay

    state, hidden, contact_floor, pair_identity = None, None, None, {}
    if args.online_checkpoint is not None:
        saved = torch.load(args.online_checkpoint, map_location="cpu", weights_only=True, mmap=True)
        schema = StateSchema.from_dict(saved["schema"])
        dataset = torch.load(args.online_checkpoint.parent / saved["replay_file"], map_location="cpu",
                             weights_only=True, mmap=True)
        if dataset["schema_fingerprint"] != schema.fingerprint or dataset["cycle"] != saved["cycle"]:
            raise ValueError("Online checkpoint and replay generation differ")
        pair_identity = {'checkpoint_cycle': saved['cycle'], 'replay_file': saved['replay_file']}
        replay = CompactReplay.from_state_dict(dataset["replay"], "cpu", read_only=True)
        state, hidden = saved["learner"]["model"], saved["config"]["model_hidden"]
        contact_floor = checkpoint_contact_floor(saved['learner'], args.device)
        fraction = saved["config"].get("validation_fraction", args.validation_fraction)
    else:
        from supermimic_rl.world_model_v2.ingest import load_recorded_replay
        replay, schema = load_recorded_replay(args.replay, "cpu", record_terrain=args.terrain_atlas is not None)
        fraction = args.validation_fraction
    if args.world_checkpoint is not None:
        state, hidden, fraction, model_source = offline_world_source(args.world_checkpoint, schema)
        # Contact floors were calibrated against the original online weights.
        # Report the override's raw loads instead of silently reusing that filter.
        contact_floor = None
        pair_identity["prediction_model"] = model_source
    held = max(1, round(replay.num_envs * fraction))
    valid_ids = list(range(replay.num_envs - held, replay.num_envs))
    if args.world_checkpoint is not None:
        if (model_source['validation_env_ids'] != valid_ids
                or model_source['training_env_ids'] != list(range(replay.num_envs - held))):
            raise ValueError('World checkpoint environment split differs from the reported replay')
        if args.online_checkpoint is not None:
            with args.online_checkpoint.open('rb') as stream:
                origin_digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            origin = model_source['training_source'] or {}
            if origin.get('kind') != 'online_pair' or origin.get('checkpoint_sha256') != origin_digest:
                raise ValueError('World override must originate from the selected online pair for held-out reporting')
    fit = None if state is None else _model(schema, args, hidden, state, args.device)
    result = report(replay, fit=fit, env_ids=valid_ids, horizon=args.horizon, batch=args.batch,
                    generator=torch.Generator().manual_seed(args.seed), device=args.device,
                    contact_floor=contact_floor, termination=pinned_termination(schema, args.runtime_json))
    result.update(schema_fingerprint=schema.fingerprint, validation_env_ids=[valid_ids[0], valid_ids[-1]],
                  source=str((args.online_checkpoint or args.replay).resolve()), **pair_identity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: entry["coverage"].get("transitions") for name, entry in result["populations"].items()}))


if __name__ == "__main__":
    main()
