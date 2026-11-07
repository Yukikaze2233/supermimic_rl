"""Opt-in dynamic commands reach the actual native stream after repeated draws."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from transformer_rl.formal.lab_env.course_tasks import LabCourseTasks
from transformer_rl.formal.tasks.command_pool import CommandPool, SegmentKind
from transformer_rl.formal.tasks.curriculum import TaskFamily
from transformer_rl.formal.tasks.gimbal_course import GimbalCourse, GimbalCourseConfig
from transformer_rl.formal.tasks.trajectory import DriveEnvelope, TrajectoryKind
from supermimic_rl.course.curriculum import CourseCurriculumConfig
from supermimic_rl.course.gimbal_arbitration import install_request_gimbal_exclusion
from supermimic_rl.course.jump_first import JumpFirstCurriculum, install_jump_first_command_pool
from supermimic_rl.course.motion_trajectory import MotionTrajectoryOwner, install_motion_trajectory_owner
from supermimic_rl.world_model_v2.motion_task import MotionTask

from test_jump_command_mixture import reset_rows
from test_request_gimbal_arbitration import native_gimbal_boundary
from test_world_v2_motion_task import sensors


def native_course(count=256, *, enabled=True, mixed=False):
    options = {'stage_transition_budgets': (100,) * 8}
    if not mixed:
        weights = list(CourseCurriculumConfig().stage_weights)
        weights[1] = (0., 1., 0., 0., 0., 0.)
        options['stage_weights'] = weights
    curriculum = JumpFirstCurriculum(count, seed=17, config=CourseCurriculumConfig(**options))
    curriculum.set_progress(785 if mixed else 185, 0)
    tasks = LabCourseTasks(count, DriveEnvelope(.06, .4372992516),
        command_pool=CommandPool(), command_context=curriculum.command_context, policy_dt_s=.02)
    install_jump_first_command_pool(tasks, curriculum)
    native = SimpleNamespace(_terrain_course=SimpleNamespace(tasks=tasks),
        _gimbal_course=GimbalCourse(GimbalCourseConfig(enabled=True), count),
        num_envs=count, _device=torch.device('cpu'), step_dt=.02)
    install_request_gimbal_exclusion(native, enabled=True)
    receipt = install_motion_trajectory_owner(native, enabled=enabled)
    return curriculum, native, receipt


def advance(tasks, step):
    now = torch.full((tasks.num_envs,), step*.02, dtype=torch.float64)
    return tasks.step(now, None, torch.zeros(tasks.num_envs), yaw_rad=torch.zeros(tasks.num_envs))[0].admitted


def test_default_off_is_a_noop_and_keeps_existing_receipts():
    curriculum, native, receipt = native_course(enabled=False)
    tasks = native._terrain_course.tasks
    before = (tasks.command_pool.sample, tasks._begin_episode_segments,
              tasks._draw_segments, tasks._apply_segment, native._gimbal_course.advance)
    assert receipt is None
    assert install_motion_trajectory_owner(native) is None
    assert not hasattr(native, '_motion_trajectory_owner')
    assert before == (tasks.command_pool.sample, tasks._begin_episode_segments,
                      tasks._draw_segments, tasks._apply_segment, native._gimbal_course.advance)
    assert 'motion_trajectory_owner' not in install_jump_first_command_pool(tasks, curriculum)
    with pytest.raises(ValueError, match='boolean'):
        install_motion_trajectory_owner(native, enabled=1)


def test_motion_uses_native_signed_normal_draws_not_relabelled_spin_and_preserves_other_rows():
    course, native, _ = native_course(count=4096, mixed=True)
    legacy_course, legacy, _ = native_course(count=4096, enabled=False, mixed=True)
    tasks, original = native._terrain_course.tasks, legacy._terrain_course.tasks
    birth, before = reset_rows(course, tasks), reset_rows(legacy_course, original)
    assert torch.equal(birth.draw.family, before.draw.family)
    motion = tasks.family == int(TaskFamily.MOTION)
    assert motion.any() and (~motion).any()
    for _ in range(4):
        for owner in (tasks, original):
            owner.elapsed.fill_(10.)
            owner.segment_end_s.zero_()
            owner._resample_segments(None)
        assert not torch.isin(tasks.segment_kind[motion], torch.tensor([2, 3, 4])).any()
        moving = motion & (tasks.segment_kind == int(SegmentKind.NORMAL))
        vx = tasks.trajectory.end[moving, 0]
        assert (vx > 1.).any() and (vx < -1.).any()
        assert tasks.trajectory.end[motion,1:3].eq(0).all()
        for name in ('start', 'end', 'duration_s', 'kind', 'frequency_start_hz', 'frequency_end_hz'):
            assert torch.equal(getattr(tasks.trajectory,name)[~motion], getattr(original.trajectory,name)[~motion])
        for name in ('segment_kind', 'segment_end_s', 'heading_enabled', 'heading_offset_rad'):
            assert torch.equal(getattr(tasks,name)[~motion], getattr(original,name)[~motion])
        assert torch.equal(tasks.generator.get_state(), original.generator.get_state())
        dynamic = moving & (birth.draw.trajectory_kind != 0)
        assert torch.equal(tasks.trajectory.kind[dynamic], birth.draw.trajectory_kind[dynamic])
        assert ((tasks.trajectory.duration_s[motion] >= .5) & (tasks.trajectory.duration_s[motion] <= 3.)).all()


@pytest.mark.parametrize('waveform', list(TrajectoryKind))
def test_native_stream_matches_recorded_trajectory_each_tick_including_height_and_zero_yaw(waveform):
    course, native, _ = native_course(count=7)
    tasks = native._terrain_course.tasks
    reset_rows(course, tasks)
    owner = native._motion_trajectory_owner
    rows = torch.arange(7)
    owner.pending_kind.fill_(int(waveform)); owner.pending_duration.fill_(1.)
    tasks.stream.current[:] = torch.tensor([-.5, 0., 0., .28])
    command = torch.tensor([.7, 0., 0., .36]).expand(7,-1).clone()
    tasks._apply_segment(rows, command, torch.full((7,), int(SegmentKind.NORMAL)),
                         torch.full((7,), 10.), torch.ones(7,dtype=torch.bool), torch.ones(7))
    assert not tasks.heading_enabled.any()
    motion = MotionTask.decode(MotionTask.capture(native))
    seen = []
    for step in range(1, 121):
        queues, readout, _ = sensors(7, step*4)
        actual = advance(tasks, step)
        motion, recorded = motion.advance_admission(queues, readout, .02)
        torch.testing.assert_close(actual, recorded, atol=2e-6, rtol=0)
        assert actual[:,1:3].eq(0).all()
        assert ((actual[:,3] >= .28-1e-6) & (actual[:,3] <= .36+1e-6)).all()
        assert tasks.stream.envelope.project(actual)[3].all()
        seen.append(actual[0,0].item())
    if waveform != TrajectoryKind.HOLD:
        assert max(seen)-min(seen) > .2
    if waveform == TrajectoryKind.START_STOP:
        assert min(abs(v) for v in seen) == 0
        assert max(seen) > .69


def test_stand_is_zero_and_partial_reset_does_not_inherit_previous_episode_command():
    course, native, _ = native_course(count=64)
    tasks = native._terrain_course.tasks
    reset_rows(course,tasks)
    owner = native._motion_trajectory_owner
    owner.pending_kind.fill_(int(TrajectoryKind.SINE))
    rows = torch.arange(64)
    tasks._apply_segment(rows, torch.tensor([1.,0.,4.,.3]).repeat(64,1), torch.zeros(64,dtype=torch.long),
        torch.full((64,),10.), torch.ones(64,dtype=torch.bool), torch.ones(64))
    assert tasks.trajectory.kind.eq(0).all()
    assert advance(tasks,1)[:,:3].eq(0).all()
    tasks.stream.current[:] = torch.tensor([2.,0.,-3.,.39])
    before = tasks.stream.current.clone()
    reset_ids = torch.arange(0,64,2)
    reset_rows(course,tasks,reset_ids)
    other = torch.ones(64,dtype=torch.bool); other[reset_ids] = False
    assert torch.equal(tasks.stream.current[other],before[other])
    torch.testing.assert_close(tasks.stream.current[reset_ids],torch.tensor([0.,0.,0.,.32]).repeat(len(reset_ids),1))
    dynamic = reset_ids[(tasks.trajectory.kind[reset_ids] != int(TrajectoryKind.HOLD))
                       & (tasks.trajectory.kind[reset_ids] != int(TrajectoryKind.START_STOP))]
    assert len(dynamic)
    torch.testing.assert_close(tasks.trajectory.start[dynamic],tasks.stream.current[dynamic])


def test_gimbal_chassis_special_is_excluded_but_servo_and_request_ownership_remain():
    _, native, receipt = native_course(count=7)
    tasks, gimbal = native._terrain_course.tasks, native._gimbal_course
    tasks.family[:] = torch.tensor([1,1,4,2,0,3,5])
    tasks.base_mode.zero_(); tasks.elapsed.fill_(10.)
    tasks.segment_clock.zero_(); tasks.segment_end_s.fill_(20.)
    tasks.segment_kind.fill_(int(SegmentKind.NORMAL))
    tasks.physical_id[:] = torch.tensor([-1,-1,-1,-1,-1,-1,9])
    gimbal.special.fill_(True)
    gimbal.remaining_s.fill_(10.)
    gimbal.config = replace(gimbal.config, spin_translate=True, spin_fraction=1.)
    servo_calls = []
    native._gimbal = SimpleNamespace(controller=SimpleNamespace(set_command=lambda *a, **k: servo_calls.append(k)))
    native.gimbal_measurement = lambda: SimpleNamespace(world_angles=torch.zeros(7,2))
    native._curriculum_update = lambda: 10000
    native._command = torch.tensor([.5,0.,0.,.32]).repeat(7,1)
    native._command_mode = torch.zeros(7,dtype=torch.long)
    before = native._command.clone()
    native_gimbal_boundary()(native)
    assert servo_calls
    assert torch.isin(torch.tensor([0,1]), torch.cat([call['ids'] for call in servo_calls])).all()
    assert gimbal.target[:2].abs().sum() > 0  # Independent servo targets were sampled.
    assert not gimbal.special[[0,1,2,6]].any()
    assert gimbal.special[[3,4,5]].all()
    torch.testing.assert_close(native._command[[0,1,2,6]], before[[0,1,2,6]])
    assert receipt['implementation'] == 'motion_trajectory_owner_v1'
    with pytest.raises(ValueError,match='already installed'):
        install_motion_trajectory_owner(native,enabled=True)


def test_fixed_evaluation_temporarily_owns_motion_and_explicit_gimbal_translation():
    from transformer_rl.formal.lab_env.formal_curriculum import LabTaskDrawSampler
    from transformer_rl.formal.tasks.geometry import TerrainKind
    from supermimic_rl.course.evaluation import CourseEvaluationConfig, FixedCourseScenarios

    course, native, _ = native_course(count=12, mixed=True)
    native._terrain_course.router = SimpleNamespace(step_table=torch.tensor([0.,.02,.2]),
        kinds=torch.tensor([int(TerrainKind.FLAT),int(TerrainKind.STEP_UP),int(TerrainKind.STEP_UP)]))
    native._task_sampler = LabTaskDrawSampler(course)
    adapter = SimpleNamespace(env=native,curriculum=course,num_envs=12,device=torch.device('cpu'))
    plan = FixedCourseScenarios(adapter,CourseEvaluationConfig(
        scenarios=('forward_speed','spin_translation_positive','braking')))
    owner, tasks = native._motion_trajectory_owner, native._terrain_course.tasks
    assert owner.training_commands_enabled
    plan.install()
    assert not owner.training_commands_enabled
    ids = torch.arange(12)
    birth = native._task_sampler(ids)
    tasks.reset(ids,birth,torch.zeros(12,dtype=torch.float64),terrain_kind=birth.terrain_kind,
                terrain_step_height_m=birth.draw.step_up_m)
    tasks.elapsed.fill_(6.); tasks.segment_end_s.zero_()
    tasks._resample_segments(None)
    native._command = tasks.trajectory.end.clone()
    native._command_mode = torch.zeros(12,dtype=torch.long)
    native._gimbal = SimpleNamespace(controller=SimpleNamespace(set_command=lambda *a,**k:None))
    native.gimbal_measurement = lambda: SimpleNamespace(world_angles=torch.zeros(12,2))
    native._curriculum_update = lambda: 10000
    native_gimbal_boundary()(native)
    special = torch.tensor([name == 'spin_translation_positive' for name in plan.row_names])
    assert torch.equal(native._gimbal_course.special,special)
    assert tasks.trajectory.kind.eq(int(TrajectoryKind.HOLD)).all()
    plan.close()
    assert owner.training_commands_enabled
