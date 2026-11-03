"""Exercise real-exposure scheduling and native birth contracts without Isaac."""
from copy import deepcopy
from dataclasses import fields, replace
import io
import json
import os
from pathlib import Path
import sys

import pytest
import torch


UPSTREAM_ROOT = Path(os.environ.get(
    "SUPERMIMIC_UPSTREAM_ROOT",
    str(Path(__file__).resolve().parents[2] / "transformer_rl-worktrees/full-vehicle-gimbal"),
))
UPSTREAM = UPSTREAM_ROOT / "src"
if not (UPSTREAM / "transformer_rl/formal/lab_env/formal_curriculum.py").is_file():
    pytest.skip("Pinned upstream task modules are required", allow_module_level=True)
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from transformer_rl.formal.lab_env.formal_curriculum import (  # noqa: E402
    FormalTaskCurriculum, LabTaskDrawSampler,
)
from transformer_rl.formal.tasks.curriculum import TaskFamily  # noqa: E402
from transformer_rl.formal.tasks.geometry import TerrainKind  # noqa: E402
from transformer_rl.formal.tasks.recovery import RECOVERY_NAMES  # noqa: E402
from supermimic_rl.course.curriculum import (  # noqa: E402
    CourseCurriculumConfig, REQUIRED_RECOVERY, STAGE_NAMES, SuperMimicCurriculum,
)


def configuration(**kwargs):
    return CourseCurriculumConfig(stage_transition_budgets=(100,) * 8, **kwargs)


def at_stage(stage, fraction=.5, *, num_envs=64):
    course = SuperMimicCurriculum(num_envs, seed=11, config=configuration())
    course.set_progress(stage * 100 + int(fraction * 100), 3)
    return course


def assert_draw_equal(left, right):
    for item in fields(left):
        assert torch.equal(getattr(left, item.name), getattr(right, item.name)), item.name


def test_only_real_transitions_advance_stages_and_compatibility_coordinate():
    course = SuperMimicCurriculum(4, config=configuration())
    course.set_progress(0, 100_000)
    assert course.stage == 0
    assert course.budget.total_updates == 0
    for stage in range(8):
        course.set_progress(stage * 100, 100_000)
        assert course.stage == stage
        assert course.stage_name == STAGE_NAMES[stage]
        assert course.budget.total_updates == course.config.source_schedule_coordinates[stage]
    course.set_progress(803, 100_001)
    assert course.stage == 7
    assert course.status == "training_budget_complete"
    assert course.real_transitions == 803
    assert sum(course.budget.consumed) == 800
    assert course.budget.total_updates == 35000
    with pytest.raises(RuntimeError, match="set_progress"):
        course.begin_update()
    for real, actor in ((802, 100_001), (803, 0), (True, 100_001), (804.0, 100_001)):
        with pytest.raises(ValueError, match="monotonic"):
            course.set_progress(real, actor)


def test_native_draw_mapper_uses_subclass_and_sampling_does_not_spend_exposure():
    course = at_stage(2)
    assert isinstance(course, FormalTaskCurriculum)
    sampler = LabTaskDrawSampler(course)
    result = sampler(torch.arange(64))
    assert result.command.shape == (64, 4)
    assert result.mode.shape == result.phase.shape == (64,)
    assert course.real_transitions == 250 and course.actor_updates == 3
    assert sum(course.sample_counts["family"]) == 64
    assert result.command[:, 1].count_nonzero() == 0


def test_single_row_batches_cover_every_required_recovery_pose():
    course = at_stage(6, num_envs=1)
    observed = set()
    for _ in range(48):
        draw = course.sample([0])
        if draw.family.item() == int(TaskFamily.RECOVERY):
            observed.add(RECOVERY_NAMES[draw.recovery_posture.item()])
            assert not draw.nominal.item()
    assert set(REQUIRED_RECOVERY) <= observed
    assert sum(course.sample_counts["recovery_posture"]) == course.sample_counts["family"][5]


def test_jump_births_and_command_pool_have_standing_then_moving_exposure():
    standing = at_stage(5, .20, num_envs=256)
    assert standing.command_context().request_motion_scale == 0
    draw = standing.sample(torch.arange(256))
    request = draw.family == int(TaskFamily.REQUEST)
    assert request.any()
    assert draw.requested_command[request, :3].count_nonzero() == 0
    assert draw.jump_height_m[request].min() >= standing.domain.jump_m[0]
    moving = at_stage(5, .85, num_envs=256)
    assert moving.command_context().request_motion_scale == 1
    draw = moving.sample(torch.arange(256))
    request = draw.family == int(TaskFamily.REQUEST)
    vx = draw.requested_command[request, 0]
    assert (vx == 0).any() and (vx.abs() > 0).any()
    assert all(value > 0 for value in moving.sample_counts["jump_motion"])
    assert not draw.nominal[request].any()


def test_authored_stair_requests_and_final_rehearsal_have_nonzero_coverage():
    course = at_stage(7, num_envs=256)
    draw = course.sample(torch.arange(256))
    terrain = draw.family == int(TaskFamily.TERRAIN)
    for kind in (TerrainKind.STEP_UP, TerrainKind.TWO_PLATFORMS):
        rows = terrain & (draw.terrain_kind == int(kind))
        assert rows.any()
        assert torch.isclose(draw.step_up_m[rows], torch.tensor(.20)).any()
    assert all(count > 0 for count in course.sample_counts["family"])
    assert course.terrain_level_cap(12) == 11
    assert course.terrain_step_up_cap_m() == .20
    # These checks prove requested exposure, not tile placement or step success.


def test_domains_and_terrain_caps_follow_exposure_not_actor_updates():
    course = at_stage(0, 0)
    assert course._effective_domain().forward_m_s == (-.5, .5)
    assert course.terrain_step_up_cap_m() == 0
    assert course.terrain_level_cap(12) == 0
    course.set_progress(300, 3)
    assert course._effective_domain().forward_m_s == (-2.7, 2.7)
    assert course.terrain_step_up_cap_m() == .05
    assert course.command_context().update == 5000
    course.set_progress(350, 3)
    assert .05 < course.terrain_step_up_cap_m() < .20
    assert 0 < course.terrain_level_cap(12) < 11
    course.set_progress(400, 3)
    assert course.terrain_level_cap(12) == 11
    assert course.terrain_step_up_cap_m() == .20
    with pytest.raises(ValueError, match="atlas"):
        course.terrain_level_cap(0)


def test_checkpoint_weights_only_resume_preserves_rng_quota_and_round_robin():
    original = at_stage(7, num_envs=5)
    for ids in ([0], [4, 1], [3, 2, 0], [2]):
        original.sample(ids)
    stream = io.BytesIO()
    torch.save(original.state_dict(), stream)
    stream.seek(0)
    state = torch.load(stream, weights_only=True)
    resumed = SuperMimicCurriculum(5, seed=999, config=configuration())
    resumed.load_state_dict(state)
    for ids in ([4], [1, 0], [3, 2, 4, 1], [0]):
        assert_draw_equal(original.sample(ids), resumed.sample(ids))
    assert original.sample_counts == resumed.sample_counts
    assert original.real_transitions == resumed.real_transitions
    assert original.actor_updates == resumed.actor_updates


def test_invalid_resume_is_atomic_and_different_configuration_is_rejected():
    original = at_stage(7)
    original.sample(torch.arange(64))
    before = original.state_dict()
    bad = deepcopy(before)
    bad["sample_counts"]["terrain"][0] += 1
    with pytest.raises(ValueError, match="counts disagree"):
        original.load_state_dict(bad)
    assert original.sample_counts == before["sample_counts"]
    assert torch.equal(original.sampler.generator.get_state(), before["sampler"]["rng"])
    other = SuperMimicCurriculum(64, config=replace(configuration(), moving_jump_min_fraction=.20))
    with pytest.raises(ValueError, match="identity"):
        other.load_state_dict(before)


@pytest.mark.parametrize("ids", [[], [0, 0], [-1], [64], [1.0], [True], [[1]]])
def test_invalid_reset_rows_do_not_consume_rng_or_ledger(ids):
    course = at_stage(7)
    before = course.state_dict()
    with pytest.raises(ValueError, match="Reset IDs"):
        course.sample(ids)
    assert course.sample_counts == before["sample_counts"]
    assert torch.equal(course.sampler.generator.get_state(), before["sampler"]["rng"])


def test_configuration_requires_complete_coverage_and_serializes_as_json():
    config = configuration()
    assert CourseCurriculumConfig.from_dict(config.to_dict()) == config
    for bad in (
        {"stage_transition_budgets": [1] * 7},
        {"source_schedule_coordinates": list(range(9))},
        {"recovery_postures": ["left_side_90", "right_side_90"]},
        {"full_terrain_kinds": ["FLAT", "STEP_UP"]},
    ):
        payload = config.to_dict()
        payload.update(bad)
        with pytest.raises(ValueError):
            CourseCurriculumConfig.from_dict(payload)
    payload = config.to_dict()
    payload["stage_weights"][6] = [1, 0, 0, 0, 0, 0]
    with pytest.raises(ValueError, match="cannot be disabled"):
        CourseCurriculumConfig.from_dict(payload)


def test_runnable_default_json_matches_checked_design():
    path = Path(__file__).resolve().parents[1] / "configs/gpufree/course.json"
    config = CourseCurriculumConfig.from_dict(json.loads(path.read_text()))
    assert config == CourseCurriculumConfig()
    assert sum(config.stage_transition_budgets) == 26_476_544
