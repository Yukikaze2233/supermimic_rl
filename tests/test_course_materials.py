"""Exercise the real 41e selected-shape reader without importing Isaac."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


UPSTREAM_ROOT = Path(os.environ.get(
    "SUPERMIMIC_UPSTREAM_ROOT",
    str(Path(__file__).resolve().parents[2] / "transformer_rl-worktrees/full-vehicle-gimbal"),
))
UPSTREAM = UPSTREAM_ROOT / "src"
if not (UPSTREAM / "transformer_rl/formal/lab_env/newton_properties.py").is_file():
    pytest.skip("Pinned Newton upstream modules are required", allow_module_level=True)
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from transformer_rl.formal.lab_env.materials import NewtonWheelMaterialReader  # noqa: E402
from supermimic_rl.course.materials import (  # noqa: E402
    make_material_reader_type, material_mapping_provenance, sample_ground_friction,
)
from supermimic_rl.course.randomization import (  # noqa: E402
    CourseRandomization, RequestedRandomizationConfig,
)


class NativeModel:
    def __init__(self, num_envs, *, groups_disabled=True, filters_disabled=True):
        # The selected shape order is intentionally interleaved across bodies.
        self.shape_world = torch.cat((torch.full((2,), -1), torch.arange(num_envs).repeat_interleave(6)))
        local_body = torch.tensor([0, 1, 2, 0, 2, 1])
        self.shape_body = torch.cat((torch.full((2,), -1),
                                    (local_body + 3 * torch.arange(num_envs)[:, None]).flatten()))
        self.shape_collision_group = (-(self.shape_world + 1) if groups_disabled
                                      else torch.ones_like(self.shape_world))
        self.shape_label = ["/World/ground/plane", "/World/ground/stairs"] + [
            f"/World/envs/env_{i}/Robot/shape_{j}" for i in range(num_envs) for j in range(6)]
        self.shape_material_mu = torch.full((2 + num_envs * 6,), .7)
        self.shape_material_mu[:2] = torch.tensor([.2, 1.1])
        self.shape_material_restitution = torch.zeros_like(self.shape_material_mu)
        self.filters_disabled = filters_disabled
        self.filter_calls = 0

    def shape_collision_filter_mask(self, pairs):
        self.filter_calls += 1
        return torch.full((len(pairs),), self.filters_disabled).numpy()


class NativeView:
    def __init__(self, model, num_envs, *, ignore_write=False):
        self.count, self.shape_count = num_envs, 6
        self.link_names = ["R_link3", "base_link", "L_link3"]
        self.body_shapes = [(2, 4), (0, 3), (1, 5)]
        # Selected row order need not equal the global shape/world order.
        self.indices = (torch.arange(num_envs * 6).reshape(num_envs, 6) + 2).flip(0)
        self.ignore_write = ignore_write

    def get_attribute(self, name, model):
        return getattr(model, name)[self.indices].reshape(1, self.count, self.shape_count)

    def set_attribute(self, name, model, value):
        if not self.ignore_write:
            getattr(model, name)[self.indices] = value.reshape(self.count, self.shape_count)


def construct(*, num_envs=8, nominal=1., groups_disabled=True, filters_disabled=True,
              mix="multiply", ignore_write=False):
    model = NativeModel(num_envs, groups_disabled=groups_disabled, filters_disabled=filters_disabled)
    view = NativeView(model, num_envs, ignore_write=ignore_write)
    robot = SimpleNamespace(body_names=["base_link", "L_link3", "R_link3"], root_view=view)
    randomization = CourseRandomization(num_envs, seed=71,
        config=RequestedRandomizationConfig(nominal_fraction=nominal))
    notifications = []
    reader_type = make_material_reader_type(NewtonWheelMaterialReader, randomization,
        solver_config=SimpleNamespace(materials=SimpleNamespace(friction_mix_mode=mix)))
    reader = reader_type(robot, (1, 2), num_envs=num_envs, device="cpu",
                         model=model, notify=notifications.append)
    return reader, model, view, randomization, notifications


def test_source_draw_and_effective_contact_are_exact_in_selected_row_order():
    reader, model, view, randomization, notifications = construct()
    source_rng = torch.Generator().manual_seed(71 ^ 0x434F4E54)
    source_ground = .3 + (.5 - .3) * torch.rand(8, generator=source_rng)
    assert torch.equal(sample_ground_friction(8, 71), source_ground)
    mu = view.get_attribute("shape_material_mu", model).reshape(8, 6)
    wheel, body = (.1 + source_ground) / 2, (.25 + source_ground) / 2
    assert torch.equal(mu[:, [1, 5, 2, 4]], wheel[:, None].expand(-1, 4))
    assert torch.equal(mu[:, [0, 3]], body[:, None].expand(-1, 2))
    assert torch.equal(model.shape_material_mu[:2], torch.ones(2))
    assert torch.equal(reader.features()[:, [0, 1, 3, 4]], wheel[:, None].expand(-1, 4))
    assert notifications == ["SHAPE_PROPERTIES"]
    # Nominal dynamics lanes still receive the source ground distribution.
    assert randomization.nominal.all() and source_ground.unique().numel() == 8
    context = randomization.context_parameters()
    assert torch.equal(context["source_ground_friction"][:, 0], source_ground)
    assert torch.equal(context["effective_body_ground_friction"][:, 0], body)
    receipt = reader.receipt["requested_contact_mapping"]
    assert receipt["status"] == "native_readback_matched"
    assert receipt["terrain_geometry_changed"] is False
    assert receipt["terrain_authored_mu_range"] == pytest.approx([.2, 1.1])
    assert receipt["self_collision"]["all_robot_self_collisions_disabled"]
    assert model.filter_calls == 0  # Negative equal groups suffice.
    with pytest.raises(ValueError, match="wheel_material to null"):
        reader.randomize_startup(None, seed=123)


def test_native_explicit_filter_pairs_are_checked_when_groups_allow_self_contact():
    reader, model, _, _, _ = construct(groups_disabled=False)
    assert model.filter_calls > 0
    assert reader.receipt["requested_contact_mapping"]["self_collision"][
        "unfiltered_cross_body_pair_count_range"] == [0, 0]


@pytest.mark.parametrize("kwargs,message", [
    ({"mix": "average"}, "multiply"),
    ({"groups_disabled": False, "filters_disabled": False}, "disabled native"),
    ({"ignore_write": True}, "readback"),
])
def test_incompatible_solver_or_native_write_does_not_pass(kwargs, message):
    with pytest.raises((ValueError, RuntimeError), match=message):
        construct(**kwargs)


def test_material_identity_excludes_sampled_native_floats():
    a, _, _, _, _ = construct(num_envs=2)
    b, _, _, _, _ = construct(num_envs=3)
    assert a.receipt["requested_contact_mapping"]["ground_draw_sha256"] != b.receipt[
        "requested_contact_mapping"]["ground_draw_sha256"]
    assert material_mapping_provenance()["native_combine_mode"] == "multiply"
    assert RequestedRandomizationConfig().provenance()["contact_material"] == material_mapping_provenance()
