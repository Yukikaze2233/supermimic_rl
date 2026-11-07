"""Surface targets retain exact footprints, holes, overlays and facet gradients."""
from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pytest
import torch

from supermimic_rl.course.platform_geometry import PlatformGeometry


def rectangle(x0, y0, x1, y1, z=.2):
    a, b, c, d = ([x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z])
    return [[a, b, c], [a, c, d]]


class Terrain:
    manifest_sha256 = 'atlas-pin'

    def __init__(self, height=.2, valid=True, origin=(0., 0., 0.)):
        self.origins = torch.tensor([origin, origin])
        self.height, self.valid = height, valid

    def height_at(self, points, tile):
        z = self.height(points) if callable(self.height) else torch.full_like(points[..., 0], self.height)
        return SimpleNamespace(height=z, valid=torch.full_like(z, self.valid, dtype=torch.bool))


def fixture(tmp_path, triangles, *, terrain=None, roi=((-5., -5.), (5., 5.)), target_tile=0):
    target = {'target_id': 'synthetic', 'tile_id': target_tile, 'triangle_ids': list(range(len(triangles))),
              'source_roi_xy_m': roi, 'sparse_query': {'numerical_z_tolerance_m': float(4 * np.finfo(np.float32).eps)}}
    faces = {str(i): {'tile_id': target_tile, 'source_part_role': 'prepared_source',
                     'tile_local_vertices_m': triangle} for i, triangle in enumerate(triangles)}
    data = {'format': 'authored-platform-triangle-targets-v1',
            'source': {'atlas_manifest': {'sha256': 'atlas-pin'}, 'atlas_geometry': {'sha256': 'geometry-pin'}},
            'targets': [target], 'face_catalog': faces}
    path = tmp_path / 'catalog.json'
    raw = json.dumps(data).encode()
    path.write_bytes(raw)
    return PlatformGeometry.load(path, expected_sha256=hashlib.sha256(raw).hexdigest(),
                                 terrain=terrain or Terrain(), target_ids=('synthetic',))


def query(geometry, xy, *, target=0, tile=0, requires_grad=False):
    points = torch.tensor([[*p, 123.] for p in xy], requires_grad=requires_grad)
    sample = geometry.query(points, torch.full((len(xy),), tile), torch.full((len(xy),), target))
    return points, sample


def test_internal_diagonal_is_not_an_edge_and_distance_gradient_is_local(tmp_path):
    geometry = fixture(tmp_path, rectangle(0, 0, 2, 2))
    points, sample = query(geometry, [[1., 1.], [.3, .8], [-.2, .8]], requires_grad=True)
    assert sample.inside.tolist() == [True, True, False]
    torch.testing.assert_close(sample.edge_distance, torch.tensor([1., .3, -.2]))
    assert torch.isnan(sample.height[-1])
    sample.edge_distance[1:].sum().backward()
    torch.testing.assert_close(points.grad[1:], torch.tensor([[1., 0., 0.], [1., 0., 0.]]))


def test_hole_and_disconnected_island_are_not_filled(tmp_path):
    triangles = (rectangle(0, 0, 3, 1) + rectangle(0, 2, 3, 3)
                 + rectangle(0, 1, 1, 2) + rectangle(2, 1, 3, 2) + rectangle(5, 0, 6, 1))
    geometry = fixture(tmp_path, triangles, roi=((-1, -1), (7, 4)))
    _, sample = query(geometry, [[1.5, 1.5], [.75, 1.5], [4., .5], [5.5, .5]])
    assert sample.inside.tolist() == [False, True, False, True]
    torch.testing.assert_close(sample.edge_distance, torch.tensor([-.5, .25, -1., .5]))


def test_overlapping_rectangles_remove_their_internal_edges(tmp_path):
    geometry = fixture(tmp_path, rectangle(0, 0, 2, 2) + rectangle(1, 0, 3, 2))
    _, sample = query(geometry, [[1.5, 1.], [2., 1.], [.5, .5]])
    torch.testing.assert_close(sample.edge_distance, torch.tensor([1., 1., .5]))
    assert sample.inside.all()


def test_roi_clips_real_faces_and_becomes_an_actual_target_boundary(tmp_path):
    geometry = fixture(tmp_path, rectangle(0, 0, 3, 2), roi=((.5, .5), (2.5, 1.5)))
    _, sample = query(geometry, [[.75, 1.], [.25, 1.], [1., 1.]])
    assert sample.inside.tolist() == [True, False, True]
    torch.testing.assert_close(sample.edge_distance, torch.tensor([.25, -.25, .5]))


def test_source_triangle_union_rejects_bounding_box_padding(tmp_path):
    geometry = fixture(tmp_path, [[[0, 0, .2], [2, 0, .2], [0, 2, .2]]])
    _, sample = query(geometry, [[1.5, 1.5], [.25, .25]])
    assert sample.inside.tolist() == [False, True]
    torch.testing.assert_close(sample.edge_distance, torch.tensor([-2**-.5, .25]))


def test_highest_layer_and_external_occlusion_are_separate_from_footprint(tmp_path):
    geometry = fixture(tmp_path, rectangle(0, 0, 2, 2) + rectangle(.5, .5, 1.5, 1.5, .21),
                       terrain=Terrain(lambda p: torch.where((p[..., 0] >= .5) & (p[..., 0] <= 1.5), .21, .2)))
    _, sample = query(geometry, [[1., 1.], [.25, .25]])
    assert sample.inside.all()
    torch.testing.assert_close(sample.height, torch.tensor([.21, .2]))
    torch.testing.assert_close(sample.edge_distance, torch.tensor([1., .25]))
    geometry.terrain.height = .4
    _, covered = query(geometry, [[1., 1.]])
    assert covered.in_footprint.item() and not covered.inside.item()
    assert covered.edge_distance.item() == 1


def test_plane_height_gradient_matches_exact_facet_with_world_quantization(tmp_path):
    triangles = rectangle(0, 0, 2, 2)
    for triangle in triangles:
        for vertex in triangle:
            vertex[2] = .2 + .1 * vertex[0] - .05 * vertex[1]
    terrain = Terrain(lambda p: .2 + .1 * p[..., 0] - .05 * p[..., 1])
    geometry = fixture(tmp_path, triangles, terrain=terrain)
    points, sample = query(geometry, [[.3, .8]], requires_grad=True)
    assert sample.inside.item()
    sample.height.sum().backward()
    torch.testing.assert_close(points.grad, torch.tensor([[.1, -.05, 0.]]))


def test_routes_batch_extra_axes_nonfinite_and_boundary_gradients(tmp_path):
    geometry = fixture(tmp_path, rectangle(0, 0, 2, 2))
    points = torch.tensor([[[.5, .5, 0.], [2., .5, 0.]], [[.5, .5, 0.], [float('nan'), 0., 0.]]],
                          requires_grad=True)
    sample = geometry.query(points, torch.tensor([0, 1]), torch.tensor([0, 0]))
    assert sample.inside.tolist() == [[True, True], [False, False]]
    sample.edge_distance[0].sum().backward()
    assert torch.isfinite(points.grad).all()
    _, inactive = query(geometry, [[.5, .5]], target=-1)
    assert not inactive.inside.item() and torch.isnan(inactive.height).all()
    with pytest.raises(ValueError, match='integer catalog'):
        query(geometry, [[.5, .5]], target=.5)


def test_atlas_miss_cannot_gain_an_invented_surface(tmp_path):
    geometry = fixture(tmp_path, rectangle(0, 0, 2, 2), terrain=Terrain(valid=False))
    _, sample = query(geometry, [[1., 1.]])
    assert sample.in_footprint.item() and not sample.inside.item() and not sample.atlas_valid.item()


def test_catalog_pin_and_atlas_identity_are_checked(tmp_path):
    geometry = fixture(tmp_path, rectangle(0, 0, 1, 1))
    path = tmp_path / 'catalog.json'
    with pytest.raises(ValueError, match='catalog pin'):
        PlatformGeometry.load(path, expected_sha256='wrong', terrain=geometry.terrain)
    geometry.terrain.manifest_sha256 = 'different'
    with pytest.raises(ValueError, match='physical atlas'):
        PlatformGeometry.load(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              terrain=geometry.terrain)


def test_target_footprint_does_not_gain_native_world_rounding_padding(tmp_path):
    geometry = fixture(tmp_path, rectangle(0, 0, 2, 2), terrain=Terrain(origin=(48., 144., 0.)))
    _, sample = query(geometry, [[2.000001, 1.], [1., 1.]])
    assert not sample.in_footprint[0]
    assert sample.inside[1]


def test_actual_pinned_sparse_probes_when_catalog_is_supplied(tmp_path):
    """Optional CPU-only check of the existing 36 probes, never a terrain scan.

    Uses the pinned native NumPy triangle oracle through AtlasTerrain's same
    world-ray input boundary. This does not claim Warp/GPU validation.
    """
    import importlib
    import os
    from pathlib import Path
    import sys

    from supermimic_rl.world_model_v2.terrain import AtlasTerrain

    filename = os.environ.get('SUPERMIMIC_PLATFORM_CATALOG')
    if not filename:
        pytest.skip('Supply the existing ignored catalog for its one-time sparse CPU audit')
    path = Path(filename)
    raw = path.read_bytes()
    pin = '51370f75d555bc7ba32143467b934149ea6a184ccf90cdad9a86fc211316195e'
    assert hashlib.sha256(raw).hexdigest() == pin
    catalog = json.loads(raw)
    payloads = {}
    for name in ('atlas_manifest', 'atlas_geometry', 'offline_query_source'):
        source = catalog['source'][name]
        value = Path(source['path']).read_bytes()
        assert hashlib.sha256(value).hexdigest() == source['sha256']
        payloads[name] = value
    manifest = json.loads(payloads['atlas_manifest'])
    source_root = Path(catalog['source']['offline_query_source']['path']).parents[3]
    sys.path.insert(0, str(source_root))
    try:
        native = importlib.import_module('transformer_rl.formal.tasks.terrain_courses')
    finally:
        sys.path.pop(0)
    data = np.load(catalog['source']['atlas_geometry']['path'])
    vertices, faces, face_tiles = data['vertices_m'], data['triangles'], data['face_tile']

    class OfflineQuery:
        tile_count = len(manifest['tiles'])
        device = torch.device('cpu')
        _ceiling_m = float(vertices[:, 2].max() + 1)

        def __call__(self, rays, tiles):
            points = rays.detach().numpy().reshape(-1, 3)
            routes = tiles.numpy().reshape(-1)
            height = np.full(len(points), np.nan)
            for route in np.unique(routes):
                selected = routes == route
                height[selected] = native.surface_height(vertices, faces[face_tiles == route], points[selected, :2])
            height = torch.tensor(height.reshape(tiles.shape), dtype=torch.float32)
            valid = torch.isfinite(height)
            normal = torch.zeros(*tiles.shape, 3)
            normal[..., 2] = valid.float()
            return height, valid, normal, torch.ones_like(height)

    origins = torch.tensor([tile['origin_m'] for tile in manifest['tiles']])
    terrain = AtlasTerrain(OfflineQuery(), origins, manifest_sha256=catalog['source']['atlas_manifest']['sha256'])
    targets = catalog['targets']
    geometry = PlatformGeometry.load(path, expected_sha256=pin, terrain=terrain,
                                     target_ids=tuple(t['target_id'] for t in targets))
    points, routes, target_routes, expected = [], [], [], []
    for index, target in enumerate(targets):
        for probe in target['sparse_query']['probes']:
            points.append([*probe['tile_local_xy_m'], -.5])
            routes.append(target['tile_id'])
            target_routes.append(index)
            expected.append(probe['target_union_height_m'])
    assert len(points) == 36
    sample = geometry.query(torch.tensor(points), torch.tensor(routes), torch.tensor(target_routes))
    assert sample.inside.all(), torch.nonzero(~sample.inside).flatten().tolist()
    assert (sample.edge_distance > 0).all()
    error = (sample.height.double() - torch.tensor(expected, dtype=torch.float64)).abs()
    assert (error <= geometry.height_tolerance[torch.tensor(target_routes)]).all()
    padding = [[*t['sparse_query']['negative_control']['tile_local_xy_m'], 1.] for t in targets]
    control = geometry.query(torch.tensor(padding), torch.tensor([t['tile_id'] for t in targets]),
                             torch.arange(len(targets)))
    assert control.atlas_valid.all() and not control.inside.any()
    assert (control.edge_distance < 0).all() and torch.isnan(control.height).all()
    report = {'catalog_sha256': pin, 'probe_count': 36, 'inside': int(sample.inside.sum()),
              'padding_controls': 4, 'padding_rejected': int((~control.inside).sum()),
              'max_height_error_m': float(error.max()),
              'boundary_segments_per_target': geometry.edge_mask.sum(-1).tolist(),
              'backend': 'CPU pinned native NumPy surface_height through AtlasTerrain; not Warp/GPU',
              'target_ids': geometry.target_ids, 'source': catalog['source']}
    if os.environ.get('SUPERMIMIC_PLATFORM_REPORT'):
        Path(os.environ['SUPERMIMIC_PLATFORM_REPORT']).write_text(json.dumps(report, indent=2) + '\n')


def test_boundary_piece_collapsing_at_float32_precision_remains_a_point(tmp_path):
    geometry = fixture(tmp_path, rectangle(0, 0, 2, 2))
    geometry.boundaries[0, 0, 1] = geometry.boundaries[0, 0, 0]
    points, sample = query(geometry, [[1., 1.]], requires_grad=True)
    assert torch.isfinite(sample.edge_distance).all()
    sample.edge_distance.sum().backward()
    assert torch.isfinite(points.grad).all()


def test_crossing_triangle_edges_keep_only_exposed_subsegments(tmp_path):
    geometry = fixture(tmp_path, [[[0, 0, .2], [2, 0, .2], [0, 2, .2]],
                                  [[0, 0, .2], [2, 0, .2], [2, 2, .2]]])
    _, sample = query(geometry, [[1., .5], [1., 1.5], [.25, 1.5]])
    assert sample.inside.tolist() == [True, False, True]
    torch.testing.assert_close(sample.edge_distance,
                               torch.tensor([.5, -.5 * 2**-.5, .25 * 2**-.5]))
