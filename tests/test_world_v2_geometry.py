"""Compare reward geometry against the full source tree, including wheel spin and derivatives."""
from pathlib import Path
import json

import pytest
import torch

pytest.importorskip('transformer_rl.formal.tasks.reset_kinematics')
from transformer_rl.formal.control.observation import MechanicalBinding
from transformer_rl.formal.tasks.reset_kinematics import AssetResetKinematics
from supermimic_rl.world_model_v2 import so3
from supermimic_rl.world_model_v2.geometry import RewardGeometry
from supermimic_rl.world_model_v2.leg_linkage import LegLinkage
from supermimic_rl.world_model_v2.physical import PhysicalState

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / 'assets/full_vehicle_v2p0p1_source_audited_v2'


def fixture():
    linkage = LegLinkage.load(ROOT / 'configs/linkage/leg_linkage_e726d23c.json')
    geometry = RewardGeometry(linkage, ASSET)
    manifest = json.loads((ASSET / 'manifest.json').read_text())
    binding = MechanicalBinding.from_manifest(manifest, manifest['tree_joint_names'], manifest['rigid_body_names'],
                                              include_auxiliary=False)
    native = AssetResetKinematics((ASSET / 'manifest.json').read_bytes(), (ASSET / 'model_spec.json').read_bytes(), binding)
    return geometry, native, binding


@pytest.mark.parametrize('measured_knees', [False, True])
def test_wheel_geometry_matches_full_tree_and_is_independent_of_wheel_spin(measured_knees):
    geometry, native, binding = fixture()
    n, generator = 24, torch.Generator().manual_seed(15)
    tree = native.nominal_tree_position[None].repeat(n, 1)
    active = torch.zeros(n, 6, dtype=torch.float64)
    for side in geometry.sides:
        table = geometry.linkage.sides[side.name]
        phi = torch.linspace(table.phi_min + .01, table.phi_max - .01, n)
        primary = torch.randn(n, dtype=torch.float64, generator=generator) * 2
        active[:, side.primary], active[:, side.secondary] = primary, primary + phi
        tree[:, binding.tree_joint_names.index(table.primary_joint)] = primary
        tree[:, binding.tree_joint_names.index(table.secondary_joint)] = primary + phi
        passive, _ = geometry.linkage.evaluate(side.name, phi.double())
        for column, name in enumerate(table.passive_joint_names):
            tree[:, binding.tree_joint_names.index(name)] = passive['passive_joint'][:, column]
        if measured_knees:
            knee = table.passive_joint_names[side.passive_knee]
            tree[:, binding.tree_joint_names.index(knee)] += .1
    tree[:, list(binding.wheel_tree_indices)] = torch.randn(n, 2, dtype=torch.float64, generator=generator) * 15
    rotation = so3.exp(torch.randn(n, 3, dtype=torch.float64, generator=generator) * 2)
    state = PhysicalState(torch.randn(n, 3, dtype=torch.float64, generator=generator), rotation,
                          torch.zeros(n, 3).double(), torch.zeros(n, 3).double(), active,
                          torch.randn(n, 8, dtype=torch.float64, generator=generator))
    knee_indices = [binding.tree_joint_names.index(geometry.linkage.sides[side.name].passive_joint_names[side.passive_knee])
                    for side in geometry.sides]
    measured = ({'passive_knee_position': tree[:, knee_indices],
                 'passive_knee_velocity': torch.zeros(n, 2).double()} if measured_knees else {})
    result, truth = geometry(state, **measured), native.forward(tree)
    for side_index, side in enumerate(geometry.sides):
        body = binding.wheel_body_indices[side_index]
        link = truth.body_position_m[:, body]
        centre_base = link + (truth.body_rotation[:, body] @ side.collision_offset[:, None]).squeeze(-1)
        centre = state.position + (rotation @ centre_base[..., None]).squeeze(-1)
        axis = (rotation @ truth.body_rotation[:, body] @ side.collision_axis[:, None]).squeeze(-1)
        torch.testing.assert_close(result.link_position_base[:, side_index], link, atol=1e-8, rtol=0)
        torch.testing.assert_close(result.centre_world[:, side_index], centre, atol=1e-8, rtol=0)
        torch.testing.assert_close(result.cylinder_axis_world[:, side_index], axis, atol=1e-8, rtol=0)
    assert result.valid.all()


def test_passive_knee_velocity_matches_differentiated_shape_and_geometry_gradient_is_finite():
    geometry, _, _ = fixture()
    q = torch.zeros(2, 6, dtype=torch.float64)
    for side in geometry.sides:
        table = geometry.linkage.sides[side.name]
        q[:, side.primary] = table.reference_primary_rad
        q[:, side.secondary] = table.reference_primary_rad + (table.phi_min + table.phi_max) / 2
    q.requires_grad_()
    dq = torch.arange(16).reshape(2, 8).double() / 10
    rotation = so3.exp(torch.tensor([[0., 0., 0.], [torch.pi / 2, 0., 0.]], dtype=torch.float64))
    state = PhysicalState(torch.zeros(2, 3).double(), rotation, torch.zeros(2, 3).double(),
                          torch.zeros(2, 3).double(), q, dq)
    result = geometry(state)
    derivative, = torch.autograd.grad(result.knee_angle_rad.sum(), q, retain_graph=True)
    for index, side in enumerate(geometry.sides):
        expected = (derivative[:, [side.primary, side.secondary]] * dq[:, [side.primary, side.secondary]]).sum(-1)
        torch.testing.assert_close(result.knee_velocity_rad_s[:, index], expected, atol=1e-8, rtol=1e-6)
    gradient, = torch.autograd.grad(result.extension_m.sum() + result.forward_velocity_rad_s.sum(), q)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
