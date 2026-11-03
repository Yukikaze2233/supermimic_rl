"""1D closed-chain leg table: Hermite accuracy, gradients, domain masks and rigid hip rotation."""
import json
import math
from pathlib import Path

import pytest
import torch

from supermimic_rl.world_model_v2.leg_linkage import LegLinkage, LinkageSide

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "configs/linkage/leg_linkage_e726d23c.json"
ASSET = ROOT / "assets/full_vehicle_v2p0p1_source_audited_v2"


def cubic_side(knots=11):
    phi = torch.linspace(-1., 2., knots, dtype=torch.float64)
    values = torch.stack((phi ** 3 - phi, 2 * phi), -1)
    slopes = torch.stack((3 * phi ** 2 - 1, torch.full_like(phi, 2.)), -1)
    return LinkageSide("left", "L_joint1", "LL_joint1", -1., 2., values, slopes, 0.,
                       torch.tensor([0., 1., 0.], dtype=torch.float64), torch.zeros(3, dtype=torch.float64))


def test_hermite_reproduces_cubics_and_their_gradient():
    side = cubic_side()
    phi = torch.tensor([-.93, .051, 1.37, 2.], dtype=torch.float64, requires_grad=True)
    values, valid = side.evaluate(phi)
    torch.testing.assert_close(values[:, 0], phi ** 3 - phi)
    torch.testing.assert_close(values[:, 1], 2 * phi)
    values[:, 0].sum().backward()
    torch.testing.assert_close(phi.grad, 3 * phi.detach() ** 2 - 1)
    assert valid.all()


def test_queries_outside_the_domain_are_finite_but_invalid():
    values, valid = cubic_side().evaluate(torch.tensor([-1.5, 3.], dtype=torch.float64))
    assert not valid.any() and torch.isfinite(values).all()


def test_wheel_center_rotates_rigidly_about_the_hip_axis():
    side = cubic_side()
    linkage = LegLinkage({"left": side}, ("wheel_center_reference_base_m",), {"wheel_center_reference_base_m": 2},
                         {})
    # Give the table a 3D centre: (x, 0, z) at every knot.
    knots = side.values.shape[0]
    center = torch.stack((torch.full((knots,), .2), torch.zeros(knots), torch.full((knots,), -.1)), -1).double()
    side = LinkageSide("left", "L_joint1", "LL_joint1", -1., 2., center, torch.zeros_like(center), 0.,
                       torch.tensor([0., 1., 0.], dtype=torch.float64), torch.zeros(3, dtype=torch.float64))
    linkage = LegLinkage({"left": side}, ("wheel_center_reference_base_m",), {"wheel_center_reference_base_m": 3}, {})
    angle = torch.tensor([0., math.pi / 2], dtype=torch.float64)
    position, valid = linkage.wheel_center_base("left", angle, torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(position[0], torch.tensor([.2, 0., -.1], dtype=torch.float64))
    # +90 deg about +y maps x -> -z and z -> x.
    torch.testing.assert_close(position[1], torch.tensor([-.1, 0., -.2], dtype=torch.float64))
    assert valid.all()


def test_loading_rejects_foreign_formats_and_assets(tmp_path):
    path = tmp_path / "table.json"
    path.write_text(json.dumps({"format": "other"}))
    with pytest.raises(ValueError, match="format"):
        LegLinkage.load(path)
    if TABLE.exists():
        with pytest.raises(ValueError, match="different asset"):
            LegLinkage.load(TABLE, expected_asset_manifest_sha256="0" * 64)


@pytest.mark.skipif(not (TABLE.exists() and ASSET.exists()), reason="pinned asset and built table required")
def test_built_table_matches_the_asset_nominal_assembly():
    manifest = json.loads((ASSET / "manifest.json").read_text())
    import hashlib
    linkage = LegLinkage.load(TABLE, expected_asset_manifest_sha256=hashlib.sha256(
        (ASSET / "manifest.json").read_bytes()).hexdigest())
    payload = json.loads(TABLE.read_text())
    nominal = manifest["nominal_joint_pos"]
    for name, side in linkage.sides.items():
        phi = torch.tensor([nominal[side.secondary_joint] - nominal[side.primary_joint]], dtype=torch.float64)
        features, valid = linkage.evaluate(name, phi)
        expected = torch.tensor([nominal[joint] for joint in payload["sides"][name]["passive_joints"]],
                                dtype=torch.float64)
        assert valid.all()
        torch.testing.assert_close(features["passive_joint"][0], expected, atol=1e-8, rtol=0)
    for side in payload["provenance"]["validation"].values():
        assert side["max_interpolated_closure_residual_m"] < 1e-9
