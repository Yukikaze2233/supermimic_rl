"""In-place spin exposure follows the exact admitted-command reward contract."""
from dataclasses import fields, replace
from types import SimpleNamespace

import torch

from transformer_rl.formal.tasks.command_pool import CommandContext, CommandPool, SegmentKind
from transformer_rl.formal.tasks.curriculum import TaskDomain, TaskFamily
from transformer_rl.formal.tasks.gimbal_course import GimbalCourse, GimbalCourseConfig
from supermimic_rl.course.jump_first import JumpFirstCommandPool, install_jump_first_command_pool
from supermimic_rl.formal_observation import SensorReadout
from supermimic_rl.reward_kernel import RewardHistory, RewardParameters, physical_reward_densities
from supermimic_rl.world_model_v2.motion_task import MotionTask
from supermimic_rl.world_model_v2.sensing import SensorQueues, heading_sample

from test_jump_command_mixture import native_course, reset_rows
from test_reward_kernel import begin, sample, setup


def test_only_special_spin_forward_targets_change_and_random_stream_is_identical():
    count = 4096
    arguments = (torch.full((count,), int(TaskFamily.MOTION)), torch.zeros(count, dtype=torch.long),
                 torch.full((count,), 10.), torch.ones(count), CommandContext(TaskDomain(), 10000))
    left_rng, right_rng = torch.Generator().manual_seed(19), torch.Generator().manual_seed(19)
    old = CommandPool().sample(*arguments, left_rng)
    new = JumpFirstCommandPool().sample(*arguments, right_rng)
    spin = (old.kind == int(SegmentKind.SPIN_LOW)) | (old.kind == int(SegmentKind.SPIN_MID))
    assert (old.kind == int(SegmentKind.SPIN_LOW)).any() and (old.kind == int(SegmentKind.SPIN_MID)).any()
    assert old.command[spin, 0].ne(0).all()
    assert new.command[spin, :2].eq(0).all()
    assert torch.equal(new.command[spin, 2:], old.command[spin, 2:])
    assert torch.equal(new.command[~spin], old.command[~spin])
    normal_micro = (old.kind == int(SegmentKind.NORMAL)) & (old.command[:, 0].abs() < .1)
    assert normal_micro.any() and new.command[normal_micro, 0].ne(0).all()
    for field in fields(old):
        if field.name != 'command':
            assert torch.equal(getattr(old, field.name), getattr(new, field.name))
    assert torch.equal(left_rng.get_state(), right_rng.get_state())


def test_native_resample_step_recording_and_reward_receive_zero_translation_spin():
    count = 512
    course, tasks = native_course(stage=7, count=count, requests_only=False)
    reset_rows(course, tasks)
    tasks.elapsed.fill_(6.)
    tasks.segment_end_s.zero_()
    tasks._resample_segments(None)
    spin = (tasks.segment_kind == int(SegmentKind.SPIN_LOW)) | (tasks.segment_kind == int(SegmentKind.SPIN_MID))
    assert spin.sum() > 20
    assert tasks.trajectory.end[spin, :2].eq(0).all()
    assert tasks.trajectory.end[spin, 2].ne(0).all()
    assert not tasks.heading_enabled[spin].any()
    # The existing admission slew remains responsible for transitions. Start
    # this sample at its endpoint, then exercise the real step and recording.
    tasks.stream.current[spin] = tasks.trajectory.end[spin]
    gimbal = GimbalCourse(GimbalCourseConfig(enabled=True), count)
    native = SimpleNamespace(_terrain_course=SimpleNamespace(tasks=tasks), _gimbal_course=gimbal)
    motion = MotionTask.decode(MotionTask.capture(native))
    queues = SensorQueues(torch.zeros(count, 3, 18), torch.zeros(count, 3, 5),
                          heading_sample(torch.zeros(count, 3)), torch.full((count,), 3.))
    readout = SensorReadout(torch.zeros(count, 4), torch.zeros(count, 1, dtype=torch.bool),
                           torch.zeros(count, 18), torch.zeros(count, 1), torch.zeros(count, 6))
    for tick in range(1, 9):
        admission, _, _, _ = tasks.step(torch.full((count,), tick * .02, dtype=torch.float64),
                                        None, torch.zeros(count), yaw_rad=torch.zeros(count))
        motion, imagined_command = motion.advance_admission(queues, readout, .02)
        assert admission.admitted[spin, :2].eq(0).all()
        torch.testing.assert_close(imagined_command[spin], admission.admitted[spin], atol=1e-6, rtol=0)
    # A sampled native spin now activates the already existing kernel penalty,
    # without changing its coefficient or interpreting a micro-command as zero.
    composer, reference = setup('ordinary')
    reference = replace(reference, admitted_body_command=admission.admitted[spin][:2])
    begin(composer, reference, torch.zeros(2, 6))
    result = physical_reward_densities(sample(composer, 1), reference,
        RewardHistory(composer._window['dq'], torch.zeros(2), torch.zeros(2), torch.zeros(2)),
        RewardParameters.from_composer(composer))
    assert result.gates['spin_centering'].all()
    assert (result.cost['stand_world_xy'] > 0).all()
    assert not result.gates['strict_stationary_deadband'].any()


def test_pool_receipt_declares_sampling_change_and_keeps_request_contract():
    course, tasks = native_course(installed=False)
    receipt = install_jump_first_command_pool(tasks, course)
    assert receipt['spin_motion'] == 'zero_forward_spin_segments_v1'
    assert receipt['request_motion'] == 'birth_forward_held_v1'
    assert receipt['request_motion_resume'] == 'cold_physics_reset_repopulates_cache'
