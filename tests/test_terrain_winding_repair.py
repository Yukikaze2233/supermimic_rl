"""Step collision orientation follows the free side, including reverse travel."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    'terrain_winding_repair', Path(__file__).parents[1] / 'scripts/repair_terrain_winding.py')
repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair)


def step_surface(height, direction):
    # Deliberately duplicate seam vertices like the upstream generator.
    points = np.array([[-1, -1, 0], [0, -1, 0], [0, 1, 0], [-1, 1, 0],
                       [0, -1, height], [1, -1, height], [1, 1, height], [0, 1, height],
                       [0, -1, 0], [0, 1, 0], [0, 1, height], [0, -1, height]], float)
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
                      [8, 9, 10], [8, 10, 11]])
    if direction < 0:
        points[:, 0] *= -1
        faces[:4] = faces[:4, [0, 2, 1]]
    return points, faces


@pytest.mark.parametrize('height', [.15, -.15])
@pytest.mark.parametrize('direction', [1, -1])
def test_riser_faces_lower_platform_and_repair_is_idempotent(height, direction):
    points, faces = step_surface(height, direction)
    before = faces.copy()
    flips = repair.riser_flips(points, faces)
    faces[flips] = faces[flips][:, [0, 2, 1]]
    tri = points[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    assert (normals[:4, 2] > 0).all()
    assert (normals[4:, 0] * direction * height < 0).all()
    np.testing.assert_array_equal(faces[:4], before[:4])
    assert not len(repair.riser_flips(points, faces))


def test_disconnected_wall_is_not_guessed():
    points, faces = step_surface(.15, 1)
    points[8:, 0] += .01
    with pytest.raises(ValueError, match='disconnected'):
        repair.riser_flips(points, faces)


def test_conflicting_platform_does_not_reverse_ground():
    points, faces = step_surface(.15, 1)
    faces[0] = faces[0, [0, 2, 1]]
    with pytest.raises(ValueError, match='upward'):
        repair.riser_flips(points, faces)


def test_derived_atlas_keeps_source_geometry_and_height_queries(tmp_path):
    source = Path(__file__).parents[1] / 'assets/rmuc2027_course_atlas'
    if not (source / 'manifest.json').is_file():
        pytest.skip('Pinned native atlas is not installed')
    output = tmp_path / 'derived'
    report = repair.repair_atlas(source, output)
    assert report['query_index_array_equal']
    with np.load(source / 'geometry.npz') as before, np.load(output / 'geometry.npz') as after:
        for name in before.files:
            if name != 'triangles':
                np.testing.assert_array_equal(after[name], before[name])
        authored = before['face_tile'] >= 101
        np.testing.assert_array_equal(after['triangles'][authored], before['triangles'][authored])
        changed = (before['triangles'] != after['triangles']).any(axis=1)
        assert changed.sum() == report['changed_face_count'] > 0
        tri = before['vertices_m'][before['triangles'][changed]]
        assert (np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])[:, 2] == 0).all()
    from transformer_rl.formal.tasks.terrain_courses import surface_height
    points = np.array([[12 + x, 84 + y] for x in (-1., 1.499, 1.501, 3.)
                       for y in (-.3, 0., .3)])
    results = []
    for path in (source, output):
        with np.load(path / 'geometry.npz') as arrays:
            faces = arrays['triangles'][arrays['face_tile'] == 58]
            results.append(surface_height(arrays['vertices_m'], faces, points))
    np.testing.assert_array_equal(*results)
