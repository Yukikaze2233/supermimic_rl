"""The native finite-cylinder solver receives true CSR triangles, never columns."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

pytest.importorskip('transformer_rl.formal.control.wheel_clearance')
from transformer_rl.formal.control.wheel_clearance import CylinderTerrainGap
from transformer_rl.formal.tasks.terrain_query import build_query_index

from supermimic_rl.course.atlas_cylinder_gap import AtlasCylinderTerrainGap


def rectangle(x0, y0, x1, y1, z=0.):
    a, b, c, d = ([x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z])
    return [[a, b, c], [a, c, d]]


def terrain(meshes, origins=None, cell_size=.25, device='cpu'):
    origins = np.zeros((len(meshes), 3)) if origins is None else np.asarray(origins)
    vertices, face_tiles = [], []
    for tile, triangles in enumerate(meshes):
        vertices.extend((np.asarray(triangles) + origins[tile]).reshape(-1, 3))
        face_tiles.extend([tile] * len(triangles))
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.arange(len(vertices)).reshape(-1, 3).astype(np.int32)
    owners = np.asarray(face_tiles, dtype=np.int32)
    index = build_query_index(vertices, faces, owners, len(meshes), cell_size_m=cell_size)
    size = float(index.pop('cell_size_m'))
    device = torch.device(device)
    index.update(vertices=vertices, faces=faces)
    query = SimpleNamespace(_buffers={k: torch.tensor(v, device=device) for k, v in index.items()},
                            _source_face_tiles=owners, tile_count=len(meshes), device=device,
                            cell_size_m=size)
    return SimpleNamespace(query=query, origins=torch.tensor(origins, dtype=torch.float32, device=device),
                           manifest_sha256='synthetic')


def geometry(meshes, **kwargs):
    return AtlasCylinderTerrainGap(terrain(meshes, **kwargs), (.1, .1), (.08, .08), tile_ids=tuple(range(len(meshes))))


def poses(points, axis=(0., 1., 0.), dtype=torch.float32, device='cpu'):
    centres = torch.tensor(points, dtype=dtype, device=device)[:, None].expand(-1, 2, -1).clone()
    axes = torch.tensor(axis, dtype=dtype, device=device).expand_as(centres).clone()
    return centres, axes


def test_flat_whole_cylinder_witness_and_reversed_winding():
    faces = rectangle(-2, -2, 2, 2)
    faces[0] = faces[0][::-1]
    gap = geometry([faces])
    centre, axis = poses([[0., 0., .3], [0., 0., .04]])
    sample = gap.at(centre, axis, torch.tensor([0, 0]))
    assert sample.valid.all() and sample.domain_covered.all()
    torch.testing.assert_close(sample.vertical_gap_m, torch.tensor([[.2, .2], [-.06, -.06]]))
    torch.testing.assert_close(sample.cylinder_point_local_m[..., 2], centre[..., 2].double() - .1)
    assert torch.equal(sample.terrain_point_local_m[..., 2], torch.zeros((2, 2), dtype=torch.float64))
    assert gap.coverage[0]['missing_projected_area_m2'] == 0


def test_step_rim_sees_high_surface_while_centre_column_remains_low():
    gap = geometry([rectangle(-2, -2, 0, 2) + rectangle(0, -2, 2, 2, .2)])
    centre, axis = poses([[-.02, 0., .3]])
    sample = gap.at(centre, axis, torch.tensor([0]))
    expected = .3 - (.1**2 - .02**2)**.5 - .2
    torch.testing.assert_close(sample.vertical_gap_m, torch.full((1, 2), expected), atol=2e-8, rtol=0)
    assert (sample.vertical_gap_m < .003).all()  # Centre column alone incorrectly gives 200 mm.
    assert (sample.terrain_point_local_m[..., 0].abs() < 1e-8).all()
    assert (sample.terrain_point_local_m[..., 2] > .19).all()


def test_all_overlays_scatter_to_the_true_upper_envelope():
    gap = geometry([rectangle(-2, -2, 2, 2) + rectangle(-.05, -.05, .05, .05, .2)
                    + rectangle(-.04, -.04, .04, .04, .15)])
    centre, axis = poses([[0., 0., .35]])
    sample = gap.at(centre, axis, torch.tensor([0]))
    torch.testing.assert_close(sample.vertical_gap_m, torch.full((1, 2), .05), atol=3e-8, rtol=0)
    assert (sample.triangle_index >= 2).all() and (sample.triangle_index < 4).all()


def test_tilted_plane_and_tilted_cylinder_match_analytic_support():
    faces = np.asarray(rectangle(-2, -2, 2, 2))
    faces[:, :, 2] = .1 + .2 * faces[:, :, 0] - .07 * faces[:, :, 1]
    gap = geometry([faces])
    centre, axis = poses([[.2, -.3, .5]], axis=(.3, .4, 0.8660254037844386), dtype=torch.float64)
    sample = gap.at(centre, axis, torch.tensor([0]))
    normal = torch.tensor([-.2, .07, 1.], dtype=torch.float64)
    along = (normal * axis[0, 0]).sum()
    expected = (normal * centre[0, 0]).sum() - .1 - .04 * along.abs() - .1 * (normal - along * axis[0, 0]).norm()
    torch.testing.assert_close(sample.vertical_gap_m.double(), expected.expand(1, 2), atol=3e-8, rtol=0)


def test_unknown_hole_is_not_filled_by_rectangle_or_csr_padding():
    faces = rectangle(-2, -2, 2, -.2) + rectangle(-2, .2, 2, 2)
    faces += rectangle(-2, -.2, -.2, .2) + rectangle(.2, -.2, 2, .2)
    gap = geometry([faces])
    centre, axis = poses([[0., 0., .3], [-.27, 0., .3], [-.5, 0., .3], [1.95, 0., .3]])
    sample = gap.at(centre, axis, torch.zeros(4, dtype=torch.long))
    assert sample.domain_covered[:, 0].tolist() == [False, False, True, False]
    assert torch.isnan(sample.vertical_gap_m[[0, 1, 3]]).all()
    assert sample.valid[2].all()
    assert gap.coverage[0]['missing_projected_area_m2'] == pytest.approx(.16, abs=1e-8)


def test_coverage_checks_true_cylinder_not_just_its_aabb_against_holes():
    # For a vertical cylinder the XY footprint is a disk. This corner hole
    # intersects its AABB but lies outside the disk, so domain remains known.
    faces = rectangle(-2, -2, 2, .08) + rectangle(-2, .09, 2, 2)
    faces += rectangle(-2, .08, .08, .09) + rectangle(.09, .08, 2, .09)
    gap = geometry([faces])
    centre, axis = poses([[0., 0., .3]], axis=(0., 0., 1.))
    sample = gap.at(centre, axis, torch.tensor([0]))
    assert sample.valid.all() and sample.domain_covered.all()


def test_disconnected_triangle_and_submillimetre_hole_remain_unknown():
    gap = geometry([rectangle(-1, -1, -.0001, 1) + rectangle(.0001, -1, 1, 1)])
    centre, axis = poses([[0., 0., .3], [.4, 0., .3]])
    sample = gap.at(centre, axis, torch.tensor([0, 0]))
    assert sample.domain_covered[:, 0].tolist() == [False, True]
    assert gap.coverage[0]['missing_projected_area_m2'] > 0


def test_csr_cells_deduplicate_faces_and_keep_tile_ownership():
    gap = geometry([rectangle(-2, -2, 2, 2), rectangle(-2, -2, 2, 2, .4)],
                   origins=((0., 0., 0.), (64., 128., 2.)), cell_size=.02)
    centre, axis = poses([[0., 0., .6], [0., 0., .6]])
    sample = gap.at(centre, axis, torch.tensor([0, 1]))
    torch.testing.assert_close(sample.vertical_gap_m, torch.tensor([[.5, .5], [.1, .1]]), atol=2e-7, rtol=0)
    rows, faces = gap._candidate_pairs(centre[:, 0].double(), torch.tensor([[.1, .04, .1]]).double().expand(2, -1),
                                      torch.tensor([0, 1]))
    assert len(faces) == 4
    assert torch.stack((rows, faces), -1).unique(dim=0).shape[0] == 4
    assert (sample.triangle_index[0] < 2).all() and (sample.triangle_index[1] >= 2).all()


def test_world_float32_geometry_translation_is_not_round_tripped_through_local_float32():
    faces = rectangle(-1, -1, 1, 1, .2)
    gap = geometry([faces], origins=((123.25, 201.5, 3.125),))
    centre, axis = poses([[.000001, -.000002, .31]], dtype=torch.float64)
    sample = gap.at(centre, axis, torch.tensor([0]))
    actual_z = float(np.float32(.2 + 3.125)) - 3.125
    torch.testing.assert_close(sample.vertical_gap_m.double(), torch.full((1, 2), .31 - .1 - actual_z,
                                                                        dtype=torch.float64), atol=1e-8, rtol=0)
    torch.testing.assert_close(sample.cylinder_point_local_m[..., 0], centre[..., 0])


def test_inactive_rows_do_not_expand_csr_or_require_geometry_values(monkeypatch):
    gap = geometry([rectangle(-1, -1, 1, 1)])
    centre, axis = poses([[float('nan'), 0., 0.]])
    monkeypatch.setattr(gap, '_candidate_pairs', lambda *args: pytest.fail('inactive rows must not query'))
    sample = gap.at(centre, axis, torch.tensor([-1]), torch.tensor([False]))
    assert not sample.valid.any() and not sample.domain_covered.any()
    assert torch.isnan(sample.vertical_gap_m).all() and (sample.triangle_index == -1).all()


def test_empty_batch_and_selected_subset():
    gap = geometry([rectangle(-1, -1, 1, 1)])
    empty = torch.empty((0, 2, 3))
    assert gap.at(empty, empty, torch.empty(0, dtype=torch.long)).vertical_gap_m.shape == (0, 2)
    centre, axis = poses([[0., 0., .3], [float('nan'), 0., 0.]])
    result = gap.at(centre, axis, torch.tensor([0, -1]), torch.tensor([True, False]))
    assert result.valid[0].all() and not result.valid[1].any()


def test_invalid_selected_axis_or_tile_and_missing_multitile_provenance_are_rejected():
    t = terrain([rectangle(-1, -1, 1, 1), rectangle(-1, -1, 1, 1)])
    gap = AtlasCylinderTerrainGap(t, (.1, .1), (.08, .08), tile_ids=(0,))
    centre, axis = poses([[0., 0., .3]])
    with pytest.raises(ValueError, match='precomputed'):
        gap.at(centre, axis, torch.tensor([1]))
    with pytest.raises(ValueError, match='normalized'):
        gap.at(centre, axis * 2, torch.tensor([0]))
    t.query._source_face_tiles = None
    with pytest.raises(ValueError, match='provenance'):
        AtlasCylinderTerrainGap(t, (.1,), (.08,), tile_ids=(0,))


def test_only_vertical_faces_do_not_supply_a_floor():
    t = terrain([rectangle(-1, -1, 1, 1)])
    t.query._buffers['vertices'][:, 0] = 0
    with pytest.raises(ValueError, match='vertically queryable'):
        AtlasCylinderTerrainGap(t, (.1,), (.08,), tile_ids=(0,))


def test_corrupt_csr_cannot_borrow_a_different_tile_face():
    gap = geometry([rectangle(-1, -1, 1, 1), rectangle(-1, -1, 1, 1)])
    gap._buffers['candidates'].fill_(2)
    centre, axis = poses([[0., 0., .3]])
    with pytest.raises(ValueError, match='ownership'):
        gap.at(centre, axis, torch.tensor([0]))


def test_exact_csr_boundary_and_source_vertex_stay_finite():
    gap = geometry([rectangle(-1, -1, 0, 1) + rectangle(0, -1, 1, 1, .2)])
    centre, axis = poses([[-.1, 0., .3], [0., 0., .3]], dtype=torch.float64)
    sample = gap.at(centre, axis, torch.tensor([0, 0]))
    assert sample.valid.all()
    torch.testing.assert_close(sample.vertical_gap_m, torch.tensor([[.1, .1], [0., 0.]]), atol=1e-8, rtol=0)


def test_actual_pinned_two_tile_sparse_queries_when_assets_supplied():
    """Use existing actual geometry/CSR and URDF, without Warp or simulation."""
    path, urdf_path = os.environ.get('SUPERMIMIC_GAP_ATLAS'), os.environ.get('SUPERMIMIC_GAP_URDF')
    if not path or not urdf_path:
        pytest.skip('Supply existing pinned atlas and wheel URDF for the sparse CPU audit')
    path = Path(path)
    raw = path.read_bytes()
    manifest_sha = '8e34c6e3cb90ec48643f3d74d124c4eaaa829ff39f6e37bd11405023da08a594'
    assert hashlib.sha256(raw).hexdigest() == manifest_sha
    manifest = json.loads(raw)
    payloads = {}
    for name in ('geometry.npz', 'query_index.npz'):
        payload = (path.parent / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == manifest['files'][name]['sha256']
        payloads[name] = np.load(path.parent / name)
    data, index = payloads['geometry.npz'], payloads['query_index.npz']
    buffers = {key: torch.tensor(index[key]) for key in index.files if key != 'cell_size_m'}
    buffers.update(vertices=torch.tensor(data['vertices_m']), faces=torch.tensor(data['triangles']))
    query = SimpleNamespace(_buffers=buffers, _source_face_tiles=data['face_tile'], tile_count=len(manifest['tiles']),
                            device=torch.device('cpu'), cell_size_m=float(index['cell_size_m']))
    t = SimpleNamespace(query=query, origins=torch.tensor([tile['origin_m'] for tile in manifest['tiles']]),
                        manifest_sha256=manifest_sha)
    raw_urdf = Path(urdf_path).read_bytes()
    from supermimic_rl.world_model_v2.records import WHEEL_BODIES
    links = {link.attrib['name']: link for link in ET.fromstring(raw_urdf).findall('link')}
    cylinders = [links[name].find('collision/geometry/cylinder') for name in WHEEL_BODIES]
    assert len(cylinders) == 2 and all(c is not None for c in cylinders)
    radii = [float(c.attrib['radius']) for c in cylinders]
    lengths = [float(c.attrib['length']) for c in cylinders]
    gap = AtlasCylinderTerrainGap(t, radii, lengths)
    # Deterministic approach, top and edge poses only; no replay was reread.
    points = [[-1., 0., .4], [.3, 0., .5], [-.0084, 0., .2 + radii[0]],
              [-1., 0., .4], [.4, -.25, .5], [.000363, .071220, .3]]
    centre, axis = poses(points, dtype=torch.float64)
    tile = torch.tensor([101, 101, 101, 102, 102, 102])
    sample = gap.at(centre, axis, tile)
    # Independent exhaustive true-face call checks CSR candidate completeness
    # at these six poses, not at a sampled full-domain grid.
    expected = []
    for row in range(len(points)):
        tri = data['vertices_m'][data['triangles'][data['face_tile'] == int(tile[row])]].astype(np.float64)
        tri -= np.asarray(manifest['tiles'][int(tile[row])]['origin_m'])
        area = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])[:, 2]
        tri = tri[abs(area) > 1e-14]
        area = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])[:, 2]
        tri[area < 0] = tri[area < 0][:, [0, 2, 1]]
        tri = torch.tensor(tri)
        values, _ = CylinderTerrainGap._minimum(centre[row, 0].expand(len(tri), -1),
            axis[row, 0].expand(len(tri), -1), torch.full((len(tri),), radii[0], dtype=torch.float64),
            torch.full((len(tri),), lengths[0] / 2, dtype=torch.float64), tri)
        expected.append(float(values.min()))
    known = sample.valid[:, 0]
    torch.testing.assert_close(sample.vertical_gap_m[known, 0].double(), torch.tensor(expected, dtype=torch.float64)[known],
                               atol=2e-8, rtol=0)
    assert known.any(), gap.coverage
    # One largest real missing triangle is a geometric negative control, not
    # a domain scan. Independently query all true projections at its centroid.
    holes = gap._holes[gap._hole_tiles == 102]
    areas = torch.linalg.cross(holes[:, 1] - holes[:, 0], holes[:, 2] - holes[:, 0])[:, 2]
    hole_point = holes[areas.argmax()].mean(0)
    hole_tri = data['vertices_m'][data['triangles'][data['face_tile'] == 102]].astype(np.float64)
    hole_tri -= np.asarray(manifest['tiles'][102]['origin_m'])
    u, v = hole_tri[:, 1, :2] - hole_tri[:, 0, :2], hole_tri[:, 2, :2] - hole_tri[:, 0, :2]
    den = u[:, 0]*v[:, 1] - u[:, 1]*v[:, 0]
    regular = abs(den) > 1e-14
    u, v, den, hole_tri = u[regular], v[regular], den[regular], hole_tri[regular]
    offset = hole_point[:2].numpy() - hole_tri[:, 0, :2]
    s = (offset[:, 0]*v[:, 1] - offset[:, 1]*v[:, 0]) / den
    q = (u[:, 0]*offset[:, 1] - u[:, 1]*offset[:, 0]) / den
    assert not ((s >= 0) & (q >= 0) & (s+q <= 1)).any()
    hc, ha = poses([[float(hole_point[0]), float(hole_point[1]), .5]], dtype=torch.float64)
    negative = gap.at(hc, ha, torch.tensor([102]))
    assert not negative.domain_covered.any() and not negative.valid.any()
    report = {'atlas_manifest_sha256': manifest_sha, 'geometry_sha256': manifest['files']['geometry.npz']['sha256'],
              'query_index_sha256': manifest['files']['query_index.npz']['sha256'],
              'urdf_sha256': hashlib.sha256(raw_urdf).hexdigest(), 'radii_m': radii, 'lengths_m': lengths,
              'coverage': gap.coverage, 'tile': tile.tolist(), 'centres_local_m': points,
              'gap_m': torch.nan_to_num(sample.vertical_gap_m, nan=1e30).tolist(),
              'domain_covered': sample.domain_covered.tolist(), 'valid': sample.valid.tolist(),
              'exhaustive_face_minima_m': expected,
              'missing_domain_control_102_xy_m': hole_point[:2].tolist(),
              'missing_domain_control_actual_triangle_hit': False,
              'backend': 'CPU original active-set solver, actual atlas buffers; not Warp/GPU/native cooking validation'}
    if os.environ.get('SUPERMIMIC_GAP_REPORT'):
        Path(os.environ['SUPERMIMIC_GAP_REPORT']).write_text(json.dumps(report, indent=2) + '\n')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_matches_cpu_for_selected_rows():
    faces = rectangle(-1, -1, 0, 1) + rectangle(0, -1, 1, 1, .2)
    cpu, gpu = geometry([faces]), geometry([faces], device='cuda')
    centre, axis = poses([[-.02, 0., .3], [.2, 0., .4]])
    routes = torch.zeros(2, dtype=torch.long)
    expected = cpu.at(centre, axis, routes)
    result = gpu.at(centre.cuda(), axis.cuda(), routes.cuda())
    torch.testing.assert_close(result.vertical_gap_m.cpu(), expected.vertical_gap_m)
    assert torch.equal(result.domain_covered.cpu(), expected.domain_covered)
