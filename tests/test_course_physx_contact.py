"""Real native reader contracts with CPU tensor views, without launching Isaac."""
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

UPSTREAM = Path(os.environ.get("SUPERMIMIC_UPSTREAM_ROOT", "/tmp/rmcs-supermimic-native-41e8280")) / "src"
if not (UPSTREAM / "transformer_rl/formal/lab_env/materials.py").is_file():
    pytest.skip("Pinned native material reader is required", allow_module_level=True)
sys.path.insert(0, str(UPSTREAM))

from transformer_rl.formal.lab_env.materials import WheelMaterialReader  # noqa: E402
from supermimic_rl.course.physx_contact import (  # noqa: E402
    apply_physx_contact_mapping, atlas_contact_profile, contact_provenance,
)
from supermimic_rl.course.randomization import CourseRandomization, RequestedRandomizationConfig  # noqa: E402


class MaterialView:
    def __init__(self, num_envs):
        self.count, self.max_shapes, self.writes = num_envs, 6, 0
        self.link_paths = [[f"/World/envs/env_{row}/Robot/{name}" for name in
                            ("base_link", "L_link3", "R_link3", "gimbal")] for row in range(num_envs)]
        self.values = torch.tensor([[.8, .6, .12], [1., 1., .02], [1., 1., .02],
                                    [1., 1., .03], [1., 1., .03], [.5, .4, .09]]).repeat(num_envs, 1, 1)
        self.corrupt = None

    def get_material_properties(self):
        return self.values

    def set_material_properties(self, values, indices):
        assert values.dtype == torch.float32 and values.device.type == "cpu" and values.is_contiguous()
        assert indices.dtype == torch.int32 and indices.tolist() == list(range(self.count))
        self.writes += 1
        self.values.copy_(values)
        if self.corrupt:
            self.corrupt(self.values)


class Scene:
    def __init__(self):
        self.values = torch.tensor([[.2, .2, .1], [.7, .7, 0.], [1.1, 1.1, .2]])
        self.reads = 0
        self.corrupt = False

    def read(self):
        self.reads += 1
        if self.corrupt and self.reads > 1:
            self.values[0, 0] += .1
        return {"terrain_bound_materials": self.values.clone(), "robot_source_shape_count": 6,
                "self_collision": {"disabled_rows": 8}, "native_combine_mode": "multiply"}


def setup(tmp_path, *, reference=.7):
    atlas = tmp_path / "manifest.json"
    atlas.write_text(json.dumps({"material_combine_mode": "multiply", "tiles": [
        {"name": "flat", "kind": 0, "parameters": {"friction": reference}}]}))
    digest = hashlib.sha256(atlas.read_bytes()).hexdigest()
    profile = atlas_contact_profile(atlas, digest)
    view = MaterialView(8)
    source = SimpleNamespace(create_rigid_body_view=lambda path: SimpleNamespace(
        max_shapes={"base_link": 1, "L_link3": 2, "R_link3": 2, "gimbal": 1}[path.rsplit("/", 1)[-1]]))
    robot = SimpleNamespace(body_names=["base_link", "L_link3", "R_link3", "gimbal"],
                            root_view=view, _physics_sim_view=source)
    reader = WheelMaterialReader(robot, (1, 2), num_envs=8, device="cpu", native_to_tensor=lambda x: x)
    randomization = CourseRandomization(8, 71, config=RequestedRandomizationConfig(nominal_fraction=.25))
    return reader, randomization, profile, Scene(), atlas


def apply(reader, randomization, profile, scene):
    return apply_physx_contact_mapping(reader, randomization, profile, scene, tensor_to_native=lambda x: x)


def test_flat_source_pair_and_authored_tile_ratios_from_full_native_readback(tmp_path):
    reader, randomization, profile, scene, _ = setup(tmp_path)
    randomization.nominal[[1, 6]] = True  # Exercise both lanes without relying on an eight-row Bernoulli draw.
    before = reader._read_native().clone()
    before_scene = scene.values.clone()
    rng = torch.get_rng_state().clone()
    dr_rng = randomization.generator.get_state().clone()
    result = apply(reader, randomization, profile, scene)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(dr_rng, randomization.generator.get_state())
    ground = .3 + .2 * torch.rand(8, generator=torch.Generator().manual_seed(71 ^ 0x434F4E54))
    wheel, body = (.1 + ground) / 2, (.25 + ground) / 2
    actual = reader._read_native()
    stress = ~randomization.nominal
    assert stress.any() and randomization.nominal.any()
    torch.testing.assert_close(actual[stress, 1:5, :2] * .7, wheel[stress, None, None].expand(-1, 4, 2))
    torch.testing.assert_close(actual[stress][:, [0, 5], :2] * .7, body[stress, None, None].expand(-1, 2, 2))
    for terrain in scene.values[:, 0]:
        torch.testing.assert_close(actual[stress, 1, 0] * terrain, wheel[stress] * terrain / .7)
    assert torch.equal(actual[:, :, 2], before[:, :, 2])
    assert torch.equal(actual[randomization.nominal], before[randomization.nominal])
    assert torch.equal(before_scene, scene.values)
    assert ground.unique().numel() == 8
    assert reader._view.writes == 1 and scene.reads == 2
    assert result["actual_reference_wheel_pair_error_max"] < 3e-8
    assert result["restitution_unchanged"] and result["native_all_robot_shapes_readback"]
    assert result["nominal_native_properties_unchanged"]
    assert result["nominal_rows"] == int(randomization.nominal.sum())
    assert result["low_grip_rows"] == int(stress.sum())
    assert not result["terrain_materials_changed"]
    assert contact_provenance(reader, backend="physx")["applied"]
    torch.testing.assert_close(reader.features()[stress][:, [0, 1, 3, 4]],
                               (wheel[stress] / .7)[:, None].expand(-1, 4))
    with pytest.raises(ValueError, match="startup ownership"):
        apply(reader, randomization, profile, scene)


@pytest.mark.parametrize("shape,column", [(0, 0), (2, 1), (5, 0), (3, 2)])
def test_partial_native_write_never_publishes_applied_or_new_critic(tmp_path, shape, column):
    reader, dr, profile, scene, _ = setup(tmp_path)
    previous = reader.features().clone()
    reader._view.corrupt = lambda values: values[:, shape, column].add_(.01)
    with pytest.raises(RuntimeError, match="full requested shape table"):
        apply(reader, dr, profile, scene)
    assert not contact_provenance(reader, backend="physx")["applied"]
    assert torch.equal(reader.features(), previous)


def test_terrain_readback_change_is_detected_without_success_receipt(tmp_path):
    reader, dr, profile, scene, _ = setup(tmp_path)
    scene.corrupt = True
    with pytest.raises(RuntimeError, match="changed terrain"):
        apply(reader, dr, profile, scene)
    assert not contact_provenance(reader, backend="physx")["applied"]


def test_reference_is_pinned_atlas_value_not_a_hardcoded_mu(tmp_path):
    reader, dr, profile, scene, atlas = setup(tmp_path, reference=.8)
    apply(reader, dr, profile, scene)
    assert profile["atlas_reference_mu"] == .8
    assert profile["atlas_sha256"] == hashlib.sha256(atlas.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="pin differs"):
        atlas_contact_profile(atlas, "0" * 64)


@pytest.mark.parametrize("reference", [0., -1., float("nan")])
def test_invalid_atlas_reference_is_not_silently_replaced(tmp_path, reference):
    with pytest.raises(ValueError, match="positive and finite"):
        setup(tmp_path, reference=reference)


def test_default_physx_receipt_reports_unapplied_and_does_not_change_native_data(tmp_path):
    reader, _, _, _, _ = setup(tmp_path)
    original = reader._read_native().clone()
    receipt = contact_provenance(reader, backend="physx")
    assert receipt["applied"] is False and not receipt["legacy_wheel_material_applied"]
    assert "native robot and atlas" in receipt["physical_materials"]
    assert "kamino" not in json.dumps(receipt)
    assert reader._view.writes == 0 and torch.equal(original, reader._read_native())


def test_sampled_floats_are_setup_evidence_not_environment_identity(tmp_path):
    first, dr, profile, scene, _ = setup(tmp_path)
    apply(first, dr, profile, scene)
    second, other, profile, scene, _ = setup(tmp_path)
    other.seed += 1
    apply(second, other, profile, scene)
    assert first.receipt["requested_contact_mapping"]["ground_draw_sha256"] != second.receipt[
        "requested_contact_mapping"]["ground_draw_sha256"]
    assert contact_provenance(first, backend="physx") == contact_provenance(second, backend="physx")


def test_all_nominal_preserves_arbitrary_original_properties_and_consumes_same_contact_draw(tmp_path):
    reader, dr, profile, scene, _ = setup(tmp_path)
    dr.nominal[:] = True
    before = reader._read_native().clone()
    result = apply(reader, dr, profile, scene)
    assert torch.equal(reader._read_native(), before)
    assert result["nominal_rows"] == 8 and result["low_grip_rows"] == 0
    assert result["actual_reference_wheel_pair_error_max"] == 0.
    other, other_dr, other_profile, other_scene, _ = setup(tmp_path)
    other_dr.nominal[:] = False
    stress_result = apply(other, other_dr, other_profile, other_scene)
    assert result["ground_draw_sha256"] == stress_result["ground_draw_sha256"]
    assert stress_result["low_grip_rows"] == 8 and stress_result["nominal_rows"] == 0
