"""Real asset transforms with native pose readback; no simulator or ideal FK."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from transformer_rl.formal.control.observation import MechanicalBinding
from transformer_rl.formal.control.wheel_clearance import CylinderGap
from transformer_rl.formal.lab_env.settings import load_asset_description

from supermimic_rl.course.native_platform_clearance import NativePlatformClearance
from supermimic_rl.world_model_v2.so3 import quaternion_to_matrix


ASSET = Path(__file__).resolve().parents[1] / 'assets/full_vehicle_v2p0p1_source_audited_v2_knee40_107'


class Query:
    """Record the exact arguments delivered to the independently tested solver."""

    def __init__(self, terrain, radii_m, lengths_m, *, tile_ids):
        self.terrain, self.radii_m, self.lengths_m = terrain, radii_m, lengths_m
        self.tile_ids, self.calls, self.args = tile_ids, 0, None

    def at(self, centres, axes, tile, selected):
        self.calls += 1
        self.args = tuple(value.clone() for value in (centres, axes, tile, selected))
        valid = selected[:, None].expand(-1, 2).clone()
        gap = torch.where(valid, 0., torch.nan)
        point = torch.where(valid[..., None], centres, torch.nan)
        return CylinderGap(gap, point, point.clone(), torch.where(valid, 0, -1), valid, valid.clone())


def native_fixture():
    asset = load_asset_description(ASSET)
    manifest = asset.manifest
    # Include both gimbal bodies and deliberately permute native body order.
    names = tuple(reversed(manifest['rigid_body_names'] + manifest['auxiliary_body_names']))
    joints = manifest['tree_joint_names'] + manifest['auxiliary_joint_names']
    binding = MechanicalBinding.from_manifest(manifest, joints, names)
    origins = torch.zeros(103, 3)
    origins[101], origins[102] = torch.tensor([15., -2., .4]), torch.tensor([-8., 4., -.1])
    router = SimpleNamespace(identity='atlas-source', origins=origins.clone(), tile=torch.tensor([101, 102, 0]))
    pose = torch.zeros(3, len(names), 7)
    pose[..., 6] = 1.
    for row in range(2):
        for side, index in enumerate(binding.wheel_body_indices):
            pose[row, index, :3] = origins[router.tile[row]] + torch.tensor([.21, .3 * (1 - 2 * side), .27])
            q = torch.tensor([.2 + row * .1, -.3, .1 + side * .1, .8])
            pose[row, index, 3:] = q / q.norm()
    pose[2].fill_(torch.nan)  # Unrelated inactive row is not queried or certified.
    native = SimpleNamespace(_asset=asset, _binding=binding, num_envs=3, physics_dt=.005,
        _episode_id=torch.tensor([10, 11, 12]), _physical_tick=4,
        _terrain_course=SimpleNamespace(router=router),
        robot=SimpleNamespace(data=SimpleNamespace(body_link_pose_w=SimpleNamespace(torch=pose))))
    native._as_torch = lambda value: value if isinstance(value, torch.Tensor) else value.torch
    native._clock = lambda: torch.full((3,), native._physical_tick * .005, dtype=torch.float64)
    terrain = SimpleNamespace(origins=origins, manifest_sha256=router.identity)
    selection = torch.tensor([True, True, False])
    binder = NativePlatformClearance(native, terrain, selected_capture=lambda: selection.clone(), query_factory=Query)
    return native, binder, selection


def test_actual_named_body_poses_use_pinned_full_vehicle_cylinders():
    native, binder, _ = native_fixture()
    result = binder.capture()
    assert len(native._binding.native_body_names) == 21
    spec = json.loads(native._asset.bound_bytes('model_spec.json'))
    bodies = {item['name']: item for item in spec['bodies']}
    pose = native.robot.data.body_link_pose_w.torch
    for side, index in enumerate(native._binding.wheel_body_indices):
        name = native._binding.native_body_names[index]
        collision = bodies[name]['collisions'][0]
        assert collision['type'] == 'cylinder'
        transform = torch.tensor(collision['origin'], dtype=torch.float64)
        rotation = quaternion_to_matrix(pose[:2, index, 3:].double())
        expected = (pose[:2, index, :3].double() + (rotation @ transform[:3, 3, None]).squeeze(-1)
                    - binder.terrain.origins[native._terrain_course.router.tile[:2]].double())
        centre, axis, _, _ = binder.query.args
        torch.testing.assert_close(centre[:2, side], expected, rtol=0, atol=1e-12)
        torch.testing.assert_close(axis[:2, side], (rotation @ transform[:3, 2, None]).squeeze(-1),
                                   rtol=0, atol=1e-12)
        assert binder.query.radii_m[side] == collision['radius']
        assert binder.query.lengths_m[side] == collision['length']
    assert result.source_geometry_bound and not result.native_collision_equivalence_verified
    assert result.urdf_sha256 == native._asset.manifest['files_sha256'][native._asset.manifest['urdf']]
    assert result.body_names == binder.wheels.body_names
    assert result.physical_tick == 4 and result.model_epoch == 0
    torch.testing.assert_close(result.timestamp_s, torch.full((3,), .02, dtype=torch.float64), rtol=0, atol=0)
    assert result.geometry.valid.tolist() == [[True, True], [True, True], [False, False]]
    assert torch.isnan(result.geometry.vertical_gap_m[2]).all()
    assert binder.query.args[0].dtype == binder.query.args[1].dtype == torch.float64


def test_same_tick_capture_is_cached_even_with_fresh_selection_tensor():
    _, binder, _ = native_fixture()
    first = binder.capture()
    assert binder.capture() is first and binder.query.calls == 1


@pytest.mark.parametrize('change', ['pose', 'episode', 'tile', 'selection', 'tick'])
def test_cache_never_crosses_same_tick_reset_route_pose_or_selection_change(change):
    native, binder, selection = native_fixture()
    before = binder.capture()
    old_centres = binder.query.args[0].clone()
    if change == 'pose':
        native.robot.data.body_link_pose_w.torch[0, native._binding.wheel_body_indices[0], 0] += .01
    elif change == 'episode':
        native._episode_id[0] += 1
    elif change == 'tile':
        native._terrain_course.router.tile[0] = 102
    elif change == 'selection':
        selection[0] = False
    else:
        native._physical_tick += 1
    after = binder.capture()
    assert after is not before and binder.query.calls == 2
    if change == 'pose':
        assert (binder.query.args[0][0, 0, 0] - old_centres[0, 0, 0]).item() == pytest.approx(.01, abs=1e-6)
    if change == 'tile':
        delta = binder.terrain.origins[101].double() - binder.terrain.origins[102].double()
        torch.testing.assert_close(binder.query.args[0][0] - old_centres[0], delta.expand(2, -1), rtol=0, atol=1e-12)
        torch.testing.assert_close(after.environment_origins_world_m[0], binder.terrain.origins[102])
    if change == 'selection':
        assert not after.geometry.valid[0].any()


def test_all_inactive_rows_remain_unqueried_and_pose_nan_does_not_become_clearance():
    native, binder, selection = native_fixture()
    selection.zero_()
    native.robot.data.body_link_pose_w.torch.fill_(torch.nan)
    native._terrain_course.router.tile.fill_(-1)
    result = binder.capture()
    assert not binder.query.args[3].any() and not result.geometry.valid.any()
    assert torch.isnan(result.geometry.vertical_gap_m).all()
    assert binder.capture() is result


def test_selected_invalid_pose_is_not_replaced_by_ideal_kinematics():
    native, binder, _ = native_fixture()
    native.robot.data.body_link_pose_w.torch[0, native._binding.wheel_body_indices[0], 3:] = 0.
    with pytest.raises(ValueError, match='normalized xyzw'):
        binder.capture()
    assert binder.query.calls == 0


def test_new_asset_binding_requires_a_new_binder():
    native, binder, _ = native_fixture()
    binder.capture()
    native._asset = replace(native._asset)
    with pytest.raises(RuntimeError, match='binding changed'):
        binder.capture()


def test_atlas_mismatch_is_rejected_before_constructing_a_query():
    native, binder, _ = native_fixture()
    terrain = SimpleNamespace(origins=binder.terrain.origins, manifest_sha256='different')
    with pytest.raises(ValueError, match='same bound atlas'):
        NativePlatformClearance(native, terrain, selected_capture=lambda: torch.ones(3, dtype=torch.bool),
                                query_factory=Query)


def test_timestamp_snapshot_does_not_alias_native_clock_or_cache_identity():
    native, binder, _ = native_fixture()
    result = binder.capture()
    native._physical_tick = 100
    assert result.physical_tick == 4 and (result.timestamp_s == .02).all()
    assert (binder.capture().timestamp_s == .5).all()
