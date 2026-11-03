"""Convert recorded native transitions into the compact V2 replay contract."""
from __future__ import annotations

from pathlib import Path

import torch

from ..contracts import Snapshot, StateSchema, Transition
from ..replay import BOUNDARY_FIELDS
from .layout import SnapshotLayout
from .replay import CompactReplay
from .records import MODEL_CONTRACT


def load_recorded_replay(path: Path, device="cpu", *, record_terrain=False) -> tuple[CompactReplay, StateSchema]:
    """Read packed data once; preserve terminal endpoints, ownership and time gaps.

    CPU mmap avoids simultaneously allocating the old ring and its unused capacity.
    World fitting and online collection use the exact same SnapshotLayout encoder.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    schema = StateSchema.from_dict(payload["schema"])
    if schema.fingerprint != payload["schema_fingerprint"]:
        raise ValueError("Recorded schema fingerprint differs")
    if payload.get('format') == 'supermimic-world-v2-dataset-v1':
        if payload.get('model_contract') != MODEL_CONTRACT:
            raise ValueError('Compact dataset world-model contract differs')
        replay = CompactReplay.from_state_dict(payload['replay'], device)
        if record_terrain and 'terrain_tile' not in replay.data:
            raise ValueError('Compact data has no terrain route; re-encode packed data or collect with --record-terrain')
        return replay, schema
    layout = SnapshotLayout(schema, record_terrain=record_terrain)
    storage, sequence = payload["storage"], payload["sequence"]
    order = sequence.argsort()
    replay = None
    snapshot_names = ("state", "observation", "goal", "context", "episode_id", "control_epoch",
                      "actor_observation", "estimator_target", "estimator_valid")
    for slot in order.tolist():
        endpoints = []
        for endpoint in ("start", "end"):
            values = {name: storage[f"{endpoint}_{name}"][slot].to(device) for name in snapshot_names}
            ownership = "policy_owned" if endpoint == "start" else "end_policy_owned"
            endpoints.append(Snapshot(**values, policy_owned=storage[ownership][slot].to(device)))
        transition = Transition(
            start=endpoints[0], end=endpoints[1], action=storage["actions"][slot].to(device),
            reward=storage["rewards"][slot].to(device), terminated=storage["terminated"][slot].to(device),
            truncated=storage["truncated"][slot].to(device), valid=storage["valid"][slot].to(device),
            substeps=storage["substeps"][slot].to(device))
        record = layout.encode(transition)
        if replay is None:
            shapes = {name: tuple(value.shape[1:]) for name, value in record.items()}
            replay = CompactReplay(shapes, payload["num_envs"], len(order), device)
        boundary = {name: storage[name][slot].to(device) for name in BOUNDARY_FIELDS}
        replay.add(record, boundary, sequence_id=int(sequence[slot]))
    if replay is None:
        raise ValueError("Recorded replay is empty")
    return replay, schema
