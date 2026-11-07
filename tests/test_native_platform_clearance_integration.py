"""Pinned native wheel poses -> actual atlas cylinders -> landing evidence."""
from types import SimpleNamespace

import torch

from transformer_rl.formal.control.observation import MechanicalBinding
from transformer_rl.formal.lab_env.settings import load_asset_description
from transformer_rl.formal.simulation.ground import WheelCollisionCentres

from supermimic_rl.course.atlas_cylinder_gap import AtlasCylinderTerrainGap
from supermimic_rl.course.native_platform_clearance import NativePlatformClearance
from supermimic_rl.course.native_platform_landing import PlatformContactPatches, PlatformLandingEvidenceBuilder
from test_atlas_cylinder_gap import rectangle, terrain
from test_native_platform_clearance import ASSET
from test_platform_geometry import Terrain, fixture


def integrated_step(tmp_path):
    atlas = terrain([rectangle(-2, -2, 2, 2),
                     rectangle(-2, -2, 0, 2) + rectangle(0, -2, 2, 2, .2)],
                    origins=((0., 0., 0.), (5.25, -2.5, 1.125)))
    # Reuse the existing exact analytic column fixture for this two-plane
    # mesh. The clearance itself uses the actual triangle CSR/active-set solve.
    columns = Terrain(lambda point: torch.where(point[..., 0] >= 0, .2, 0.))
    columns.origins, columns.query = atlas.origins, atlas.query
    geometry = fixture(tmp_path, rectangle(0, -2, 2, 2, .2), terrain=columns, target_tile=1)
    asset = load_asset_description(ASSET)
    m = asset.manifest
    binding = MechanicalBinding.from_manifest(m, m['tree_joint_names'] + m['auxiliary_joint_names'],
        list(reversed(m['rigid_body_names'] + m['auxiliary_body_names'])))
    wheels = WheelCollisionCentres.from_urdf(asset.bound_bytes(m['urdf']), binding)
    tile = torch.tensor([1, 1, 0])  # Duplicate tiles, with N != T and an inactive row.
    selected = torch.tensor([True, True, False])
    router = SimpleNamespace(identity=columns.manifest_sha256, origins_table=columns.origins,
                             origins=columns.origins[tile].clone(), tile=tile)
    pose = torch.zeros(3, len(binding.native_body_names), 7)
    pose[..., 6] = 1.
    rotation = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]], dtype=torch.float64)
    centre = torch.zeros(3, 2, 3, dtype=torch.float64)
    for row in range(2):
        for side, body in enumerate(binding.wheel_body_indices):
            radius = wheels.radius_m[side]
            centre[row, side] = torch.tensor([-.02, .3 * (1 - 2 * side),
                                              .2 + (radius**2 - .02**2)**.5 + .0002])
            offset = rotation @ torch.tensor(wheels.local_xyz_m[side], dtype=torch.float64)
            pose[row, body, :3] = centre[row, side] + router.origins[row].double() - offset
            pose[row, body, 3:] = torch.tensor([2**-.5, 0., 0., 2**-.5])
    pose[2].fill_(torch.nan)
    native = SimpleNamespace(_asset=asset, _binding=binding, num_envs=3,
        _episode_id=torch.tensor([7, 9, 1]), _physical_tick=12, physics_dt=.005,
        _terrain_course=SimpleNamespace(router=router),
        robot=SimpleNamespace(data=SimpleNamespace(body_link_pose_w=SimpleNamespace(torch=pose))))
    native._as_torch = lambda value: value.torch
    native._clock = lambda: torch.full((3,), native._physical_tick * .005, dtype=torch.float64)
    capture = NativePlatformClearance(native, columns, selected_capture=lambda: selected, tile_ids=(1,))
    builder = PlatformLandingEvidenceBuilder(geometry, binding.wheel_body_indices)
    return native, capture, builder, selected, centre


def contact_rows(native):
    rows = torch.tensor([0, 0, 1, 1])
    body = torch.tensor(native._binding.wheel_body_indices).repeat(2)
    local = torch.tensor([[0., .3, .2], [0., -.3, .2]]).repeat(2, 1)
    points = local + native._terrain_course.router.origins[rows]
    return PlatformContactPatches(rows, body, points, torch.tensor([[0., 0., 1.]]).repeat(4, 1),
                                  torch.full((4,), 20.))


def evidence(builder, native, selected, sample, patches, loads):
    return builder.build(patches=patches, body_load_n=loads,
        contact_valid=torch.ones(3, dtype=torch.bool), tile=native._terrain_course.router.tile,
        selected=selected, target_index=torch.tensor([0, 0, -1]),
        command_height_m=torch.full((3,), .32), clearance=sample.geometry)


def test_real_cylinders_at_step_rim_produce_correct_world_landing_height(tmp_path):
    native, capture, builder, selected, centres = integrated_step(tmp_path)
    assert isinstance(capture.query, AtlasCylinderTerrainGap)  # No mocked query result.
    sample = capture.capture()
    assert sample.geometry.valid.tolist() == [[True, True], [True, True], [False, False]]
    torch.testing.assert_close(sample.geometry.vertical_gap_m[:2], torch.full((2, 2), .0002),
                               atol=3e-7, rtol=0)
    assert (centres[:2, :, 0] < 0).all()
    centre_column_gap = centres[:2, :, 2] - torch.tensor(capture.wheels.radius_m)
    assert (centre_column_gap > .19).all()  # Incorrect centre ray would miss the 200 mm riser.
    torch.testing.assert_close(sample.geometry.terrain_point_local_m[:2, :, 0], torch.zeros(2, 2).double(),
                               atol=1e-8, rtol=0)
    loads = torch.zeros(3, len(native._binding.native_body_names))
    loads[:2, list(native._binding.wheel_body_indices)] = 20.
    result = evidence(builder, native, selected, sample, contact_rows(native), loads)
    assert result.target_wheel_touch[:2].all() and result.landing_support_valid[:2].all()
    torch.testing.assert_close(result.landing_world_height[:2], torch.full((2,), 1.125 + .2 + .32),
                               atol=2e-7, rtol=0)
    assert not result.full_body_clear.any()
    assert not result.measurement_valid[2] and not result.landing_support_valid[2]
    assert torch.isnan(sample.geometry.vertical_gap_m[2]).all()
    assert torch.isnan(result.landing_world_height[2])


def test_nonwheel_support_prevents_flight_despite_real_positive_wheel_clearance(tmp_path):
    native, capture, builder, selected, _ = integrated_step(tmp_path)
    # Raise both actual wheel link poses; all gap calculations remain real.
    pose = native.robot.data.body_link_pose_w.torch
    pose[:2, list(native._binding.wheel_body_indices), 2] += .1
    sample = capture.capture()
    assert (sample.geometry.vertical_gap_m[:2] > .1).all()
    base = native._binding.native_body_names.index('base_link')
    loads = torch.zeros(3, len(native._binding.native_body_names))
    loads[0, base] = 20.
    patches = PlatformContactPatches(torch.tensor([0]), torch.tensor([base]),
        native._terrain_course.router.origins[:1].clone(), torch.tensor([[0., 0., 1.]]), torch.tensor([20.]))
    result = evidence(builder, native, selected, sample, patches, loads)
    assert not result.wheel_touch.any()
    assert result.nonwheel_contact.tolist() == [True, False, False]
    assert result.full_body_clear.tolist() == [False, True, False]
    assert result.measurement_valid.tolist() == [True, True, False]
    assert not result.landing_support_valid.any()
