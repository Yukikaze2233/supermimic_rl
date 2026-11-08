"""ToF-only labels use the pinned native reset, request protocol and stair FSM."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.course.stair_label import install_tof_stair_label
from transformer_rl.formal.control.observation import CommandMode
from transformer_rl.formal.lab_env.course_tasks import LabCourseTasks, RequestCommand
from transformer_rl.formal.lab_env.formal_curriculum import (
    FormalCurriculumCapabilities, FormalTaskCurriculum, LabTaskDrawSampler,
)
from transformer_rl.formal.tasks.curriculum import TaskFamily
from transformer_rl.formal.tasks.geometry import TerrainKind
from transformer_rl.formal.tasks.protocol import RequestResponse
from transformer_rl.formal.tasks.trajectory import CommandTrajectory, DriveEnvelope

from support.world_v2.stair_task import estimate_at, native_stairs


def course_fixture(n=2, *, hint_lifetime='fixed'):
    native = native_stairs(n, labels=True, assist=True)
    tasks = LabCourseTasks(n, DriveEnvelope(.08, .38, 90.), policy_dt_s=.02)
    native._terrain_course.tasks = tasks
    tasks.stream.current[:] = torch.tensor([.8, 0., .2, .30])
    tasks.trajectory = CommandTrajectory(
        tasks.stream.current.clone(), tasks.stream.current.clone(),
        torch.full((n,), 5.), torch.zeros(n, dtype=torch.long))
    install_tof_stair_label(native, height_hint_lifetime=hint_lifetime)
    return native


def advance(native, tick):
    native.time.fill_(tick * .02)
    native._completed_estimate = estimate_at(native.time - .02, native.num_envs)
    native._terrain_course.advance_skill()


@pytest.mark.parametrize('policy', [None, SimpleNamespace(use_mode_labels=False)])
def test_label_install_requires_observable_mode_path(policy):
    native = native_stairs(1)
    native.cfg.terrain_policy_spec = policy
    with pytest.raises(ValueError, match='use_mode_labels = true'):
        install_tof_stair_label(native)


def test_routed_terrain_truth_never_creates_stair_or_slope_actor_labels():
    kinds = torch.tensor([int(kind) for kind in (
        TerrainKind.FLAT, TerrainKind.STEP_UP, TerrainKind.STEP_DOWN,
        TerrainKind.STAIRS, TerrainKind.TWO_PLATFORMS, TerrainKind.SLOPE)])
    n = len(kinds)
    native = course_fixture(n)
    curriculum = FormalTaskCurriculum(
        n, capabilities=FormalCurriculumCapabilities(True, True, True))
    sample = LabTaskDrawSampler(curriculum)(torch.arange(n))
    sample = replace(sample, draw=replace(
        sample.draw, family=torch.full((n,), int(TaskFamily.TERRAIN))))
    tasks = native._terrain_course.tasks
    tasks.reset(torch.arange(n), sample, native.time, terrain_kind=kinds)
    assert not tasks.terrain_mode_labels
    assert tasks.base_mode.eq(int(CommandMode.NORMAL)).all()
    torch.testing.assert_close(tasks.terrain_kind, kinds)
    advance(native, 1)
    assert native._command_mode.eq(int(CommandMode.NORMAL)).all()
    assert native._terrain_course.stair_requests.active_id.eq(-1).all()


@pytest.mark.parametrize('ending', ['complete', 'cancel', 'timeout', 'safety_abort', 'tilt'])
@pytest.mark.parametrize('lifetime', ['fixed', 'request'])
def test_tof_label_and_height_hint_exit_together_on_native_request_end(ending, lifetime):
    native = course_fixture(hint_lifetime=lifetime)
    course = native._terrain_course
    course.tof_request_id.fill_(7)
    course.tof_step_height_m.fill_(.20)
    advance(native, 1)
    assert native._command_mode.eq(int(CommandMode.STAIR)).all()
    assert native._base_command_mode.eq(int(CommandMode.NORMAL)).all()
    torch.testing.assert_close(native._command[:, :3], native._command_base[:, :3])
    torch.testing.assert_close(native._command[:, 3], torch.full((2,), .37))
    assert course.tof_request_id.eq(-1).all()

    tick = 2
    if ending == 'complete':
        course._pending_requests['stair'] = RequestCommand(
            torch.full((2,), -1), torch.zeros(2), completed_id=torch.full((2,), 7))
    elif ending == 'cancel':
        course.stair_cancel_id.fill_(7)
    elif ending == 'timeout':
        tick = 12
    elif ending == 'safety_abort':
        course._pending_requests['stair'] = RequestCommand(
            torch.full((2,), -1), torch.zeros(2), safety_abort=torch.ones(2, dtype=torch.bool))
    if ending == 'tilt':
        native.time.fill_(tick * .02)
        native._completed_estimate = estimate_at(native.time - .02, 2, tilt=.9)
        course.advance_skill()
    else:
        advance(native, tick)
    assert course.stair_requests.active_id.eq(-1).all()
    assert native._command_mode.eq(int(CommandMode.NORMAL)).all()
    torch.testing.assert_close(native._command, native._command_base)
    assert not native._skill_fsm.stair.active.any()
    advance(native, tick + 1)
    assert native._command_mode.eq(int(CommandMode.NORMAL)).all()


@pytest.mark.parametrize('lifetime', ['fixed', 'request'])
def test_new_jump_wins_same_boundary_tof_request_and_is_not_relabelled(lifetime):
    native = course_fixture(hint_lifetime=lifetime)
    course = native._terrain_course
    course._pending_requests['jump'] = RequestCommand(torch.full((2,), 17), torch.full((2,), .10))
    course.tof_request_id.fill_(23)
    course.tof_step_height_m.fill_(.20)
    advance(native, 1)
    assert course.tasks.active_id.eq(17).all()
    assert course.stair_requests.active_id.eq(-1).all()
    assert course._last_stair_response.eq(int(RequestResponse.BUSY)).all()
    assert native._command_mode.eq(int(CommandMode.JUMP)).all()
    torch.testing.assert_close(native._command, native._command_base)
    assert not course.body_contact_allowed.any()
    # A fresh ToF packet on a continuing jump cannot override or cancel it.
    course.tof_request_id.fill_(24)
    advance(native, 2)
    assert native._command_mode.eq(int(CommandMode.JUMP)).all()
    assert course.tasks.active_id.eq(17).all()
    assert not course.tasks._physical_cancelled.any()


@pytest.mark.parametrize('lifetime', ['fixed', 'request'])
def test_existing_stair_request_finishes_before_a_new_jump_can_start(lifetime):
    native = course_fixture(hint_lifetime=lifetime)
    course = native._terrain_course
    course.tof_request_id.fill_(7)
    course.tof_step_height_m.fill_(.20)
    advance(native, 1)
    course._pending_requests['jump'] = RequestCommand(torch.full((2,), 17), torch.full((2,), .10))
    advance(native, 2)
    assert course.tasks.active_id.eq(-1).all()
    assert course.tasks.protocol.max_seen_id.eq(17).all()
    assert native._command_mode.eq(int(CommandMode.STAIR)).all()
    course.stair_cancel_id.fill_(7)
    advance(native, 3)
    course._pending_requests['jump'] = RequestCommand(torch.full((2,), 18), torch.full((2,), .10))
    advance(native, 4)
    assert course.tasks.active_id.eq(18).all()
    assert native._command_mode.eq(int(CommandMode.JUMP)).all()
