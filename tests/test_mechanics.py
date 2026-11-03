"""Mechanical property contracts without loading either simulator backend."""

from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.adapters.mechanics import read_articulation_mechanics


class NewtonArticulation:
    def __init__(self):
        self.body_names = ("aux", "base", "wheel")
        self.num_instances = 2
        self.data = SimpleNamespace(
            body_mass=SimpleNamespace(torch=torch.tensor([[1., 4., 2.], [1.1, 4.1, 2.1]])),
            body_com_pos_b=SimpleNamespace(torch=torch.arange(18).float().reshape(2, 3, 3)),
            body_inertia=SimpleNamespace(torch=torch.arange(54).float().reshape(2, 3, 9)),
        )

    @property
    def root_view(self):
        raise AssertionError("Newton reader must not access PhysX root_view")


def test_newton_reads_public_named_axes_and_declares_identity_com_frame():
    robot = NewtonArticulation()
    result = read_articulation_mechanics(robot, backend="newton", body_names=("wheel", "base"))
    assert result.body_names == ("wheel", "base")
    assert torch.equal(result.mass_kg, robot.data.body_mass.torch[:, [2, 1]])
    assert torch.equal(result.com_pose_local_xyzw[..., :3], robot.data.body_com_pos_b.torch[:, [2, 1]])
    assert torch.equal(result.com_pose_local_xyzw[..., 3:], torch.tensor([0., 0., 0., 1.]).expand(2, 2, 4))
    assert torch.equal(result.inertia_kg_m2, robot.data.body_inertia.torch[:, [2, 1]])
    assert "link-aligned COM axes" in result.com_frame
    assert "expressed in link axes" in result.inertia_frame
    assert "Newton" in result.source
    result.mass_kg.zero_()
    result.com_pose_local_xyzw.zero_()
    assert robot.data.body_mass.torch.min() > 0
    assert robot.data.body_com_pos_b.torch.count_nonzero() == 17


def test_newton_matrix_inertia_shape_is_preserved_as_full_nine_components():
    robot = NewtonArticulation()
    robot.data.body_inertia.torch = robot.data.body_inertia.torch.reshape(2, 3, 3, 3)
    result = read_articulation_mechanics(robot, backend="newton")
    assert result.inertia_kg_m2.shape == (2, 3, 9)
    assert torch.equal(result.inertia_kg_m2, robot.data.body_inertia.torch.flatten(-2))


def test_physx_keeps_backend_order_and_real_com_quaternion():
    mass = torch.tensor([[2., 1., 4.]])
    com = torch.arange(21).float().reshape(1, 3, 7)
    inertia = torch.arange(27).float().reshape(1, 3, 9)
    robot = SimpleNamespace(
        body_names=("base", "aux", "wheel"), backend_body_names=("wheel", "aux", "base"),
        root_view=SimpleNamespace(
            link_paths=[["/Robot/wheel", "/Robot/aux", "/Robot/base"]],
            get_masses=lambda: mass, get_coms=lambda: com, get_inertias=lambda: inertia),
    )
    result = read_articulation_mechanics(robot)
    assert result.body_names == robot.backend_body_names
    assert torch.equal(result.mass_kg, mass)
    assert torch.equal(result.com_pose_local_xyzw, com)
    assert torch.equal(result.inertia_kg_m2, inertia)
    assert "PhysX" in result.inertia_frame
    gathered = read_articulation_mechanics(robot, body_names=("base", "wheel"))
    assert torch.equal(gathered.mass_kg, mass[:, [2, 0]])


@pytest.mark.parametrize("field,value", [
    ("body_mass", torch.ones(2, 2)),
    ("body_mass", torch.zeros(2, 3)),
    ("body_mass", torch.ones(2, 3, dtype=torch.float64)),
    ("body_com_pos_b", torch.ones(2, 3, 7)),
    ("body_inertia", torch.full((2, 3, 9), float("nan"))),
])
def test_invalid_newton_properties_are_rejected(field, value):
    robot = NewtonArticulation()
    setattr(robot.data, field, value)
    with pytest.raises(ValueError):
        read_articulation_mechanics(robot, backend="newton")


@pytest.mark.parametrize("names", [[], ["base", "base"], ["missing"]])
def test_invalid_named_selections_are_rejected(names):
    with pytest.raises(ValueError, match="selected mechanical body names"):
        read_articulation_mechanics(NewtonArticulation(), backend="newton", body_names=names)


def test_backend_is_explicit_and_never_inferred_from_missing_physx_properties():
    with pytest.raises(ValueError, match="explicitly"):
        read_articulation_mechanics(NewtonArticulation(), backend="auto")
    with pytest.raises(AssertionError, match="PhysX root_view"):
        read_articulation_mechanics(NewtonArticulation())
