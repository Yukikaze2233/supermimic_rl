"""REQUEST birth targets reach native segments, recording and fixed evaluation."""
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from transformer_rl.formal.lab_env.course_tasks import LabCourseTasks
from transformer_rl.formal.lab_env.formal_curriculum import LabTaskDrawSampler
from transformer_rl.formal.tasks.command_pool import CommandPool, SegmentKind
from transformer_rl.formal.tasks.curriculum import TaskFamily
from transformer_rl.formal.tasks.gimbal_course import GimbalCourse, GimbalCourseConfig
from transformer_rl.formal.tasks.trajectory import DriveEnvelope

from supermimic_rl.course.curriculum import CourseCurriculumConfig
from supermimic_rl.course.jump_first import JumpFirstCurriculum, install_jump_first_command_pool
from supermimic_rl.formal_observation import SensorReadout
from supermimic_rl.world_model_v2.motion_task import MotionTask
from supermimic_rl.world_model_v2.sensing import SensorQueues, heading_sample


def native_course(*, stage=2, consumed=85, count=32, requests_only=True, installed=True):
    options = {'stage_transition_budgets': (100,) * 8}
    if requests_only:
        weights = list(CourseCurriculumConfig().stage_weights)
        weights[5] = (0., 0., 0., 0., 1., 0.)
        options.update(stage_weights=weights, mixture_blend_fraction=.1)
    course = JumpFirstCurriculum(count, seed=17, config=CourseCurriculumConfig(**options))
    course.set_progress(stage * 100 + consumed, 0)
    tasks = LabCourseTasks(count, DriveEnvelope(.06, .34), command_pool=CommandPool(),
                          command_context=course.command_context, policy_dt_s=.02,
                          terrain_mode_labels=False)
    if installed:
        install_jump_first_command_pool(tasks, course)
    return course, tasks


def reset_rows(course, tasks, ids=None):
    ids = torch.arange(course.num_envs) if ids is None else ids
    task = LabTaskDrawSampler(course)(ids)
    tasks.reset(ids, task, torch.zeros(course.num_envs, dtype=torch.float64),
                terrain_kind=task.terrain_kind, terrain_step_height_m=task.draw.step_up_m)
    return task


@pytest.mark.parametrize('consumed', [20, 50, 85])
def test_birth_split_and_minimum_survive_initial_and_later_native_segments(consumed):
    course, tasks = native_course(consumed=consumed)
    birth = reset_rows(course, tasks)
    expected = birth.command[:, 0].clone()
    scale = course._request_motion_scale()
    if scale:
        assert (expected == 0).sum() == course.num_envs // 2
        bound = min(-course._effective_domain().forward_m_s[0], course._effective_domain().forward_m_s[1])
        assert expected[expected != 0].abs().min() >= bound * course.config.moving_jump_min_fraction * scale
    else:
        assert expected.eq(0).all()
    assert torch.equal(tasks.trajectory.end[:, 0], expected)
    for tick in range(1, 5):
        tasks.segment_end_s.zero_()  # Exercise native step -> _resample_segments -> _apply_segment.
        admission, _, _, _ = tasks.step(torch.full((course.num_envs,), tick * .02, dtype=torch.float64),
                                        None, torch.zeros(course.num_envs))
        assert torch.equal(admission.requested[:, 0], expected)
        assert admission.requested[:, 1:3].eq(0).all()
        assert torch.equal(tasks.segment_kind == int(SegmentKind.STAND), expected == 0)
        assert not tasks.heading_enabled.any()
    # Partial reset changes only the reset rows, using the existing global birth ordinal.
    rows = torch.tensor([7, 1, 3])
    expected[rows] = reset_rows(course, tasks, rows).command[:, 0]
    tasks.segment_end_s.zero_()
    tasks._resample_segments(None)
    assert torch.equal(tasks.trajectory.end[:, 0], expected)


def test_nonrequest_segments_rng_and_recovery_handoff_are_unchanged():
    course, tasks = native_course(stage=7, count=256, requests_only=False)
    original_course, original = native_course(stage=7, count=256, requests_only=False, installed=False)
    birth, reference = reset_rows(course, tasks), reset_rows(original_course, original)
    assert torch.equal(birth.command, reference.command)
    other = birth.draw.family != int(TaskFamily.REQUEST)
    assert other.any() and (~other).any()
    assert (birth.draw.family == int(TaskFamily.TERRAIN)).any()
    for _ in range(3):
        for name in ('start', 'end', 'duration_s', 'kind'):
            assert torch.equal(getattr(tasks.trajectory, name)[other], getattr(original.trajectory, name)[other])
        for name in ('segment_kind', 'segment_end_s', 'heading_enabled', 'heading_offset_rad'):
            assert torch.equal(getattr(tasks, name)[other], getattr(original, name)[other])
        assert torch.equal(tasks.generator.get_state(), original.generator.get_state())
        tasks.segment_end_s.zero_()
        original.segment_end_s.zero_()
        tasks._resample_segments(None)
        original._resample_segments(None)
    recovery = birth.draw.family == int(TaskFamily.RECOVERY)
    assert recovery.any()
    tasks.begin_tracking(recovery)
    original.begin_tracking(recovery)
    assert torch.equal(tasks.trajectory.end[recovery], original.trajectory.end[recovery])
    assert torch.equal(tasks.generator.get_state(), original.generator.get_state())


def test_cold_resume_rebuilds_episode_targets_from_restored_sampling_state():
    course, tasks = native_course()
    reset_rows(course, tasks)
    course_state, native_state = course.state_dict(), tasks.state_dict()
    restored_course, restored = native_course()
    restored_course.load_state_dict(course_state)
    now = torch.zeros(course.num_envs, dtype=torch.float64)
    restored.restore_after_physical_reset(native_state, now)
    # Both paths now start fresh physical episodes, as CourseAdapter.load_training_state_dict does.
    tasks.restore_after_physical_reset(native_state, now)
    left, right = reset_rows(course, tasks), reset_rows(restored_course, restored)
    for item in fields(left.draw):
        assert torch.equal(getattr(left.draw, item.name), getattr(right.draw, item.name))
    assert torch.equal(tasks.trajectory.end, restored.trajectory.end)
    for owner in (tasks, restored):
        owner.segment_end_s.zero_()
        owner._resample_segments(None)
    assert torch.equal(tasks.trajectory.end, restored.trajectory.end)
    assert torch.equal(tasks.generator.get_state(), restored.generator.get_state())


def test_active_jump_and_manual_trajectory_keep_the_native_segment_owner():
    course, tasks = native_course(count=4)
    reset_rows(course, tasks)
    # External command owners write a trajectory; an in-flight request prevents stochastic resampling.
    manual = torch.tensor([[-.7, 0., 0., .31], [.8, 0., 0., .33]])
    tasks.trajectory.start[:2] = tasks.trajectory.end[:2] = manual
    tasks.protocol.active_id[:2] = torch.tensor([101, 102])
    tasks.segment_end_s.zero_()
    tasks._resample_segments(None)
    assert torch.equal(tasks.trajectory.end[:2], manual)
    tasks.protocol.active_id[:2] = -1
    tasks._resample_segments(None)
    assert not torch.equal(tasks.trajectory.end[:2], manual)


def test_fixed_evaluation_replaces_training_segments_and_restores_wrapper():
    from supermimic_rl.course.evaluation import CourseEvaluationConfig, FixedCourseScenarios

    course, tasks = native_course(count=4)
    reset_rows(course, tasks)
    original_draw = tasks._draw_segments
    router = SimpleNamespace(step_table=torch.tensor([0., .02, .20]), kinds=torch.tensor([0, 3, 3]))
    native = SimpleNamespace(_terrain_course=SimpleNamespace(tasks=tasks, router=router),
                             _gimbal_course=GimbalCourse(GimbalCourseConfig(enabled=True), 4),
                             _task_sampler=LabTaskDrawSampler(course))
    adapter = SimpleNamespace(env=native, curriculum=course, num_envs=4, device=torch.device('cpu'))
    plan = FixedCourseScenarios(adapter, CourseEvaluationConfig(
        scenarios=('jump_standing_low', 'jump_moving_intermediate')))
    plan.install()
    try:
        birth = native._task_sampler(torch.arange(4))
        tasks.reset(torch.arange(4), birth, torch.zeros(4, dtype=torch.float64),
                    terrain_kind=birth.terrain_kind, terrain_step_height_m=birth.draw.step_up_m)
        expected = torch.tensor([plan.scenarios[name]['commands'][0] for name in plan.row_names])
        assert torch.equal(tasks.trajectory.end, expected)
        # A later fixed segment deliberately differs from the cached birth vx.
        for case in plan.scenarios.values():
            case['commands'] = (case['commands'][0], [-.6, 0., 0., .30])
        tasks.elapsed.fill_(plan.pool.config.segment_s[0])
        tasks.segment_end_s.zero_()
        tasks._resample_segments(None)
        assert tasks.trajectory.end[:, 0].eq(-.6).all()
    finally:
        plan.close()
    assert tasks._draw_segments is original_draw


def test_recorded_motion_matches_native_request_commands_without_resampling():
    course, tasks = native_course(count=4)
    reset_rows(course, tasks)
    gimbal = GimbalCourse(GimbalCourseConfig(enabled=True), 4)
    native = SimpleNamespace(_terrain_course=SimpleNamespace(tasks=tasks), _gimbal_course=gimbal)
    motion = MotionTask.decode(MotionTask.capture(native))
    queues = SensorQueues(torch.zeros(4, 3, 18), torch.zeros(4, 3, 5),
                          heading_sample(torch.zeros(4, 3)), torch.full((4,), 3.))
    readout = SensorReadout(torch.zeros(4, 4), torch.zeros(4, 1, dtype=torch.bool), torch.zeros(4, 18),
                            torch.zeros(4, 1), torch.zeros(4, 6))
    for tick in range(1, 33):
        admission, _, _, _ = tasks.step(torch.full((4,), tick * .02, dtype=torch.float64),
                                        None, torch.zeros(4), yaw_rad=torch.zeros(4))
        motion, command = motion.advance_admission(queues, readout, .02)
        torch.testing.assert_close(command, admission.admitted, rtol=0, atol=1e-6)
        torch.testing.assert_close(motion.encode(), MotionTask.capture(native), rtol=0, atol=1e-6)


def test_new_request_motion_bootstraps_before_unknown_segment_resampling():
    from support.world_v2.jump_imagination import fixture
    from supermimic_rl.world_model_v2.jump_task import JumpTaskState
    from supermimic_rl.world_model_v2.learning import lambda_returns

    learner, start, frames = fixture()
    course, tasks = native_course(count=2)
    reset_rows(course, tasks)
    native = SimpleNamespace(_terrain_course=SimpleNamespace(tasks=tasks,
                             router=SimpleNamespace(origins=torch.zeros(2, 3))),
                             _gimbal_course=GimbalCourse(GimbalCourseConfig(enabled=True), 2),
                             _clock=lambda: torch.zeros(2, dtype=torch.float64))
    start['motion_task'] = MotionTask.capture(native)
    start.update(JumpTaskState.capture(native).record())
    start['tof_segment_end'].fill_(.04)
    rollout = learner.imagination.rollout(start, frames, 16)
    assert rollout.lengths.tolist() == [1, 1]
    assert rollout.boundary.sum(0).tolist() == [1, 1]
    expected = rollout.rewards[0] + .99 * rollout.continuation[0] * rollout.values[1]
    torch.testing.assert_close(lambda_returns(rollout, .99, .95)[0], expected.detach())
