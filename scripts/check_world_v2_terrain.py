#!/usr/bin/env python3
"""Audit predicted-pose atlas queries against native snapshots and collision facets.

No simulator or policy is run. Native packed frames verify scan/frame alignment;
authored facet centroids cover every atlas kind and verify local height gradients.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from supermimic_rl.contracts import StateSchema
from supermimic_rl.world_model_v2.geometry import RewardGeometry
from supermimic_rl.world_model_v2.layout import SnapshotLayout
from supermimic_rl.world_model_v2.leg_linkage import LegLinkage
from supermimic_rl.world_model_v2.terrain import AtlasTerrain


def errors(value):
    value = value.detach()
    finite = torch.isfinite(value)
    flat = value[finite].abs().double()
    return {'finite': int(finite.sum()), 'nonfinite': int((~finite).sum()),
            'rms': float(flat.square().mean().sqrt()) if len(flat) else None,
            'max': float(flat.max()) if len(flat) else None}


def check_native(terrain, payload, schema, linkage, asset, maximum):
    layout = SnapshotLayout(schema, record_terrain=True)
    raw = payload['storage']
    candidates = raw['valid'].flatten().nonzero().flatten()
    selected = candidates[torch.linspace(0, len(candidates) - 1, min(maximum, len(candidates))).long()]
    values = lambda name: raw[name].flatten(0, 1)[selected].to(terrain.query.device)
    packed, context = values('start_state'), values('start_context')
    state = layout.physical(packed)
    tile = layout.state(packed, 'course.terrain_tile')
    observation = terrain.observe(state, tile)
    origin = layout.context(context, 'environment_origin_world_m')
    assigned_origin = terrain.origins[tile.long()]
    geometry = RewardGeometry(LegLinkage.load(linkage), asset)
    q = layout.state(packed, 'joint_position_gimbal_yaw_sine')
    dq = layout.state(packed, 'joint_velocity_native')
    from supermimic_rl.world_model_v2.records import PASSIVE_KNEE_JOINTS
    knees = [layout.joint_names.index(name) for name in PASSIVE_KNEE_JOINTS]
    wheel = geometry(state, passive_knee_position=q[..., knees], passive_knee_velocity=dq[..., knees])
    wheel_ground = terrain.height_at(wheel.centre_world, tile)
    actual_wheel = layout.context(context, 'ground_at_wheel_world_z_m') - origin[:, 2:3]
    return {'frames': len(selected), 'valid_scans': int(observation.valid.sum()),
            'tile_origin_m': errors(origin - assigned_origin),
            'height_above_ground_m': errors(observation.height_above_ground - layout.height_above_ground(packed)),
            'scan_relative_m': errors(observation.scan_relative - layout.terrain_scan(packed)),
            'wheel_ground_m': errors(wheel_ground.height - actual_wheel),
            'scope': 'recorded native states, measured passive knees; no learned dynamics'}


def check_facets(terrain, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    with np.load(manifest_path.parent / 'geometry.npz', allow_pickle=False) as source:
        vertices = source['vertices_m']
        faces = source['triangles']
        face_tile = source['face_tile']
    triangles = torch.from_numpy(vertices[faces])
    tile = torch.from_numpy(face_tile).long()
    a, b, c = triangles.unbind(-2)
    cross = torch.linalg.cross(b - a, c - a)
    edges = torch.stack((b - a, c - b, a - c), -2)[..., :2].norm(dim=-1)
    # Stay far enough inside each XY triangle that finite differences do not
    # cross its boundary; vertical risers have no downward-column derivative.
    clearance = cross[:, 2].abs() / (3 * edges.max(-1).values.clamp_min(1e-12))
    eligible = clearance > .025
    selected = []
    for index in range(len(manifest['tiles'])):
        choices = (eligible & (tile == index)).nonzero().flatten()
        if len(choices):
            selected.append(choices[torch.linspace(0, len(choices) - 1, min(16, len(choices))).long()])
    selected = torch.cat(selected)
    tile = tile[selected].to(terrain.query.device)
    points = triangles[selected].mean(-2).to(terrain.query.device) - terrain.origins[tile]
    points.requires_grad_()
    result = terrain.height_at(points, tile)
    result.height[result.valid].sum().backward()
    numerical = []
    for axis in (0, 1):
        delta = torch.zeros_like(points)
        delta[:, axis] = .005
        plus = terrain.height_at(points + delta, tile)
        minus = terrain.height_at(points - delta, tile)
        # Use the actual float32 atlas displacement. Large atlas translations
        # quantize an ideal +/-5 mm shift before the native kernel sees it.
        positive = (points.detach() + delta + terrain.origins[tile])[:, axis]
        negative = (points.detach() - delta + terrain.origins[tile])[:, axis]
        numerical.append((plus.height - minus.height) / (positive - negative))
    numerical = torch.stack(numerical, -1)
    kind = torch.tensor([t['kind'] for t in manifest['tiles']], device=tile.device)[tile]
    reports = {}
    for value in kind.unique().tolist():
        chosen = kind == value
        reports[str(value)] = {'points': int(chosen.sum()), 'valid': int(result.valid[chosen].sum()),
            'gradient_error_m_per_m': errors(points.grad[chosen, :2] - numerical[chosen])}
    return {'tiles_sampled': int(tile.unique().numel()), 'atlas_tiles': len(manifest['tiles']),
            'per_kind': reports, 'vertical_derivative': errors(points.grad[:, 2]),
            'scope': 'facet interiors only; no derivative asserted at a step edge or triangle boundary'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('replay', type=Path, help='Packed native replay with terrain context')
    parser.add_argument('--atlas', type=Path, default=Path('assets/rmuc2027_course_atlas/manifest.json'))
    parser.add_argument('--asset', type=Path, default=Path('assets/full_vehicle_v2p0p1_source_audited_v2'))
    parser.add_argument('--linkage', type=Path, default=Path('configs/linkage/leg_linkage_e726d23c.json'))
    parser.add_argument('--max-frames', type=int, default=1024)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    payload = torch.load(args.replay, map_location='cpu', weights_only=True, mmap=True)
    schema = StateSchema.from_dict(payload['schema'])
    if schema.fingerprint != payload['schema_fingerprint']:
        raise ValueError('Native replay schema differs')
    terrain = AtlasTerrain.load(args.atlas, expected_sha256=json.loads(schema.source)['terrain_sha256'], device=args.device)
    report = {'atlas_sha256': terrain.manifest_sha256, 'device': str(terrain.query.device),
        'native': check_native(terrain, payload, schema, args.linkage, args.asset, args.max_frames),
        'facets': check_facets(terrain, args.atlas)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
