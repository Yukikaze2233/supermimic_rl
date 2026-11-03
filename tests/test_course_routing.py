"""Exercise the actual frozen atlas through the source routing implementation."""
from copy import deepcopy
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path(os.environ.get("SUPERMIMIC_UPSTREAM_ROOT",
    ROOT.parent / "transformer_rl-worktrees/full-vehicle-gimbal")) / "src"
if not (UPSTREAM / "transformer_rl/formal/tasks/terrain_courses.py").exists():
    pytest.skip("Pinned upstream terrain modules are required", allow_module_level=True)
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from transformer_rl.formal.tasks.terrain_courses import TerrainCourseRouter  # noqa: E402
from supermimic_rl.course.routing import (  # noqa: E402
    AuthoredEndpointRouter, authored_tile_keys, install_authored_routing,
)


def router(count=16, seed=7):
    return TerrainCourseRouter(ROOT / "assets/rmuc2027_course_atlas/manifest.json", count, seed=seed)


def requested(count, height=.2):
    return {"slope_rad": torch.full((count,), .1), "roughness_m": torch.full((count,), .01),
            "step_up_m": torch.full((count,), height), "step_down_m": torch.full((count,), .1)}


def reset(value, kind, *, height=.2, stage=4, cap=.2):
    count = len(kind)
    return value.reset(torch.arange(count), torch.tensor(kind), stage=stage, command_forward=torch.ones(count),
                       requested=requested(count, height), level_cap=9, step_up_cap_m=cap)


def test_existing_source_has_probabilistic_authored_coverage_not_endpoint_guarantee():
    source = router(count=256)
    reset(source, [3, 7] * 128)
    actual = source.authored[source.tile]
    assert actual.any(), "The source already includes real authored ceiling exploration"
    assert (~actual).any(), "Endpoint requests remain proficiency-limited outside exploration"
    assert not source.level.any()


def test_endpoint_requests_place_actual_named_meshes_even_at_zero_proficiency():
    value = AuthoredEndpointRouter(router())
    origins, ground = reset(value, [3, 7] * 8)
    keys = value.authored_tile_keys
    assert keys == {"road_step": 101, "central_steps": 102}
    assert value.tile.tolist() == [keys["road_step"], keys["central_steps"]] * 8
    assert not value.level.any(), "Authored exposure must not forge acquired proficiency"
    torch.testing.assert_close(origins, value.origins_table[value.tile])
    torch.testing.assert_close(ground, value.spawn_z[value.tile])
    report = value.actual_metrics()
    assert report["authored_reset_counts"] == {"road_step": 8, "central_steps": 8}
    assert report["explicit_endpoint_reset_counts"] == {"road_step": 8, "central_steps": 8}
    assert value.receipt()["authored_tiles"]["central_steps"]["profile_m"] == [0., .2, .35, .3]


@pytest.mark.parametrize("stage,height,cap", [(3, .2, .2), (4, .1, .2), (4, .2, .1)])
def test_non_endpoint_or_unadmitted_stage_remains_exactly_source_routing(stage, height, cap):
    source = router()
    value = AuthoredEndpointRouter(router())
    # Include authored kinds and ordinary non-authored kinds in the same batch.
    kinds = [0, 1, 2, 3, 4, 5, 6, 7] * 2
    reset(source, kinds, stage=stage, height=height, cap=cap)
    reset(value, kinds, stage=stage, height=height, cap=cap)
    for name in ("tile", "level", "origins", "direction"):
        assert torch.equal(getattr(source, name), getattr(value, name))
    assert torch.equal(source.generator.get_state(), value.generator.get_state())
    assert sum(value.actual_metrics()["explicit_endpoint_reset_counts"].values()) == 0


def test_other_terrain_kinds_keep_seeded_source_choices_and_do_not_consume_extra_rng():
    source = router()
    value = AuthoredEndpointRouter(router())
    kinds = [0, 1, 2, 3, 4, 5, 6, 7] * 2
    reset(source, kinds)
    reset(value, kinds)
    ordinary = ~torch.isin(torch.tensor(kinds), torch.tensor([3, 7]))
    assert torch.equal(source.tile[ordinary], value.tile[ordinary])
    assert torch.equal(source.generator.get_state(), value.generator.get_state())
    assert torch.equal(source.level, value.level)


def test_routing_checkpoint_restores_real_counts_and_future_route_choices():
    value = AuthoredEndpointRouter(router())
    reset(value, [3, 7] * 8)
    state = value.state_dict()
    restored = AuthoredEndpointRouter(router(seed=19))
    restored.load_state_dict(state)
    assert restored.actual_metrics() == value.actual_metrics()
    reset(value, [0, 1, 2, 3, 4, 5, 6, 7] * 2, height=.1)
    reset(restored, [0, 1, 2, 3, 4, 5, 6, 7] * 2, height=.1)
    assert torch.equal(value.tile, restored.tile)
    assert restored.actual_metrics() == value.actual_metrics()
    damaged = value.state_dict()
    damaged["forced_counts"][0] = 1_000_000
    before = value.actual_metrics()
    with pytest.raises(ValueError, match="exceed actual"):
        value.load_state_dict(damaged)
    assert before == value.actual_metrics()


def test_asset_lookup_rejects_generic_equal_stairs_and_installation_is_idempotent():
    source = router()
    damaged = deepcopy(source.manifest)
    damaged["tiles"][102]["profile_m"] = [0., .2, .4]
    with pytest.raises(ValueError, match="profile"):
        authored_tile_keys(SimpleNamespace(manifest=damaged, step_table=source.step_table))
    env = SimpleNamespace(_terrain_course=SimpleNamespace(router=source))
    installed = install_authored_routing(env)
    assert env._terrain_course.router is installed is install_authored_routing(env)
    assert env.authored_routing_receipt["atlas_sha256"] == source.identity
