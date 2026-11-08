"""Create a derived atlas with consistently oriented procedural step risers.

The source atlas remains immutable. Exact-coordinate adjacency connects the
generator's duplicated seam vertices; upward platform faces fix the outside.
Only vertical face winding may change, never geometry, materials or face IDs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import copy
import json
from pathlib import Path

import numpy as np


STEP_TILES = frozenset(('step_up', 'step_down', 'stairs', 'two_platforms'))


def riser_flips(vertices, faces):
    """Find vertical faces inconsistent with their upward platform neighbours."""
    points = np.asarray(vertices, dtype=np.float64)[faces]
    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    if not np.isfinite(points).all() or (np.linalg.norm(normals, axis=1) == 0).any():
        raise ValueError('Step surface has nonfinite or degenerate triangles')
    horizontal = (normals[:, :2] == 0).all(axis=1)
    vertical = normals[:, 2] == 0
    if not (horizontal | vertical).all() or (normals[horizontal, 2] <= 0).any():
        raise ValueError('Expected upward platforms and vertical risers only')
    # Welding is for topology checks only; no vertex is moved or merged in USD.
    _, welded = np.unique(points.reshape(-1, 3), axis=0, return_inverse=True)
    welded = welded.reshape(-1, 3)
    edges = defaultdict(list)
    for face, indices in enumerate(welded):
        for a, b in zip(indices, np.roll(indices, -1)):
            edges[min(a, b), max(a, b)].append((face, a < b))
    neighbours = defaultdict(list)
    for incident in edges.values():
        if len(incident) > 2:
            raise ValueError('Step surface has a non-manifold seam')
        if len(incident) == 2:
            (a, forward_a), (b, forward_b) = incident
            # Adjacent faces must traverse their common edge in opposite ways.
            different_flip = forward_a == forward_b
            neighbours[a].append((b, different_flip))
            neighbours[b].append((a, different_flip))
    flips = np.full(len(faces), -1, dtype=np.int8)
    flips[horizontal] = 0
    queue = deque(np.flatnonzero(horizontal).tolist())
    while queue:
        a = queue.popleft()
        for b, different_flip in neighbours[a]:
            expected = int(flips[a]) ^ int(different_flip)
            if flips[b] < 0:
                flips[b] = expected
                queue.append(b)
            elif flips[b] != expected:
                raise ValueError('Step surface orientation conflicts with its platforms')
    if (flips < 0).any():
        raise ValueError('A riser is disconnected from the declared platforms')
    return np.flatnonzero(flips == 1)


def repair_atlas(source, output):
    from transformer_rl.formal.tasks.terrain_assets import file_pin, write_collision_usda
    from transformer_rl.formal.tasks.terrain_query import build_query_index

    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError('Derived atlas must be outside the source atlas')
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest['format'] != 'terrain_course_atlas_v1':
        raise ValueError('Unsupported terrain atlas')
    for name, pin in manifest['files'].items():
        if Path(name).name != name or file_pin(source / name) != pin:
            raise ValueError(f'Source atlas file identity differs: {name}')
    with np.load(source / 'geometry.npz', allow_pickle=False) as raw:
        arrays = {name: raw[name].copy() for name in raw.files}
    original_faces = arrays['triangles'].copy()
    changed, tiles = [], []
    for tile_id, tile in enumerate(manifest['tiles']):
        if tile['name'] not in STEP_TILES:
            continue
        face_ids = np.flatnonzero(arrays['face_tile'] == tile_id)
        flips = face_ids[riser_flips(arrays['vertices_m'], original_faces[face_ids])]
        arrays['triangles'][flips] = original_faces[flips][:, [0, 2, 1]]
        assert not len(riser_flips(arrays['vertices_m'], arrays['triangles'][face_ids]))
        changed.extend(flips.tolist())
        tiles.append({'tile_id': tile_id, 'name': tile['name'],
                      'changed_face_ids': flips.tolist()})
    with np.load(source / 'query_index.npz', allow_pickle=False) as raw:
        old_index = {name: raw[name].copy() for name in raw.files}
    index = build_query_index(arrays['vertices_m'], arrays['triangles'], arrays['face_tile'],
                              len(manifest['tiles']), cell_size_m=float(old_index['cell_size_m']))
    assert all(np.array_equal(index[name], old_index[name]) for name in index)
    derived = copy.deepcopy(manifest)
    derived['derived_riser_winding_repair'] = {
        'source_manifest': file_pin(source / 'manifest.json'),
        'operation': 'vertical_winding_consistent_with_upward_platforms',
        'changed_face_ids': changed, 'changed_face_count': len(changed),
        'coordinates_materials_face_ids_unchanged': True,
        'authored_source_tiles_unchanged': True,
        'physics_qualification_performed': False,
    }
    output.mkdir(parents=True)
    np.savez_compressed(output / 'geometry.npz', **arrays)
    np.savez_compressed(output / 'query_index.npz', **index)
    write_collision_usda(output / 'terrain.usda', arrays['vertices_m'], arrays['triangles'],
                         static_friction=.7, dynamic_friction=.7,
                         face_material=arrays['face_material'],
                         material_parameters=arrays['material_parameters'])
    derived['files'] = {name: file_pin(output / name) for name in manifest['files']}
    (output / 'manifest.json').write_text(json.dumps(derived, indent=2) + '\n')
    return {'source_manifest': file_pin(source / 'manifest.json'),
            'candidate_manifest': file_pin(output / 'manifest.json'),
            'changed_face_count': len(changed), 'tiles': tiles,
            'query_index_array_equal': True, 'physics_qualification_performed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = repair_atlas(args.source, args.output)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'tiles'}))


if __name__ == '__main__':
    main()
