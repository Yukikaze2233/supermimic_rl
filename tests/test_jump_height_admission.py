"""Caller heights use the existing native/imagined projection and slew path."""
from unittest.mock import patch

import pytest
import torch

from supermimic_rl.course.command_override import override_command_height
from supermimic_rl.course.jump_task_adapter import NativeJumpTaskAdapter
from supermimic_rl.course.jump_takeoff import install_jump_takeoff
from supermimic_rl.world_model_v2.jump_task import JumpTask, JumpTaskConfig, JumpTaskState
from supermimic_rl.world_model_v2.motion_task import MotionTask
from transformer_rl.formal.control.sensors import SensorModel
from transformer_rl.formal.lab_env.course_tasks import LabCourseTasks, RequestCommand
from transformer_rl.formal.tasks.gimbal_course import GimbalCourse, GimbalCourseConfig
from transformer_rl.formal.tasks.protocol import CompletedEstimate, RequestResponse
from transformer_rl.formal.tasks.trajectory import CommandRates, CommandStream
from support.world_v2.jump_task import native_jumps
from test_world_v2_motion_task import native_tasks, sensors


new_native = pytest.mark.skipif(not hasattr(LabCourseTasks, '_select_request'),
                               reason='Requires the private native request-selection entry point')


def test_height_override_only_replaces_requested_height_and_keeps_gradients():
    original = torch.tensor([[.5, .2, 1., .32], [1., 0., -.4, .3]], requires_grad=True)
    height = torch.tensor([.4, .5], requires_grad=True)
    mask = torch.tensor([True, False])
    before = original.detach().clone()
    assert override_command_height(original) is original
    result = override_command_height(original, height, mask)
    torch.testing.assert_close(result[:, :3], original[:, :3], rtol=0, atol=0)
    torch.testing.assert_close(result[:, 3], torch.tensor([.4, .3]), rtol=0, atol=0)
    result.sum().backward()
    assert original.grad.tolist() == [[1., 1., 1., 0.], [1., 1., 1., 1.]]
    assert height.grad.tolist() == [1., 0.]
    torch.testing.assert_close(original, before, rtol=0, atol=0)


@pytest.mark.parametrize('height,mask', [
    (torch.ones(2), None), (None, torch.ones(2, dtype=torch.bool)),
    (torch.ones(2, dtype=torch.float64), torch.ones(2, dtype=torch.bool)),
    (torch.ones(2), torch.ones(2)), (torch.ones(1), torch.ones(2, dtype=torch.bool)),
])
def test_override_requires_matching_value_and_mask(height, mask):
    with pytest.raises(ValueError, match='Height override'):
        override_command_height(torch.zeros(2, 4), height, mask)


@new_native
@pytest.mark.parametrize('rates', [None, CommandRates(.8, 3., .2)])
def test_native_and_world_override_before_coupled_slew_and_leave_other_rows_unchanged(rates):
    native = native_tasks(rates)
    owner = native._terrain_course.tasks
    NativeJumpTaskAdapter(owner)
    task = MotionTask.decode(MotionTask.capture(native))
    initial = task
    baseline = MotionTask.decode(task.encode().clone())
    before = task.encode().clone()
    mask = torch.tensor([True, False, True, False, True, False, True])
    height = torch.tensor([.40, .70, .36, .70, .42, .70, .37])
    for tick in range(1, 5):
        queues, readout, _ = sensors(7, tick * 4)
        sensor = type('Sensor', (), dict(lags=readout.ages.long(), count=queues.count.long(), capacity=3,
            physical_tick=2, _rows=torch.arange(7), uplink_delay_s=None))()
        yaw = SensorModel._delayed_angle(sensor, queues.heading.flip(1).transpose(0, 1), group=3)
        previous = task.current.clone()
        result = owner.step(torch.full((7,), tick * .02, dtype=torch.float64), None, torch.zeros(7),
                            yaw_rad=yaw, height_override=height, height_override_mask=mask)[0]
        task, admitted = task.advance_admission(queues, readout, .02,
                                                height_override=height, height_override_mask=mask)
        baseline, unmodified = baseline.advance_admission(queues, readout, .02)
        torch.testing.assert_close(admitted, result.admitted, rtol=0, atol=1e-6)
        torch.testing.assert_close(task.encode(), MotionTask.capture(native), rtol=0, atol=1e-6)
        torch.testing.assert_close(admitted[~mask], unmodified[~mask], rtol=0, atol=0)
        torch.testing.assert_close(result.requested[mask, 3], height[mask], rtol=0, atol=0)
        if rates is None:
            torch.testing.assert_close(admitted[mask, 3], height[mask], rtol=0, atol=0)
        else:
            assert ((admitted[:, 3] - previous[:, 3]).abs() <= rates.height_rate_m_s * .02 + 1e-7).all()
            assert (result.slew_fraction[mask] < 1.).all()
            assert (admitted[mask, 3] != height[mask]).all()
    # The original functional state remains a reusable imagination start.
    torch.testing.assert_close(initial.encode(), before, rtol=0, atol=0)


@new_native
def test_native_and_world_latch_the_slew_admitted_height_on_acceptance():
    native = native_jumps(3)
    native._gimbal_course = GimbalCourse(GimbalCourseConfig(enabled=False), 3)
    owner = native._terrain_course.tasks
    owner.stream = CommandStream(owner.stream.current, owner.stream.envelope, CommandRates(.8, 3., .2),
                                 policy_dt_s=.02)
    NativeJumpTaskAdapter(owner)
    install_jump_takeoff(native, {})
    state = JumpTaskState.capture(native)
    jump = JumpTask(JumpTaskConfig.capture(native))
    motion = MotionTask.decode(MotionTask.capture(native))
    queues, readout, _ = sensors(3, 4)
    height, mask = torch.full((3,), .4), torch.tensor([True, False, False])
    external = RequestCommand(torch.tensor([50, -1, -1]), torch.full((3,), .2))
    estimate = CompletedEstimate(torch.zeros(3), torch.full((3, 2), .9), torch.zeros(3),
        torch.zeros(3), native.time.clone(), torch.ones(3, dtype=torch.bool))
    elapsed, previous_height = owner.elapsed.clone(), owner.stream.current[:, 3].clone()
    native.time.fill_(.02)
    result = owner.step(native.time, estimate, torch.zeros(3), command=external,
                        height_override=height, height_override_mask=mask)
    following, admitted = motion.advance_admission(queues, readout, .02,
                                                   height_override=height, height_override_mask=mask)
    imagined = jump.advance(state, now_s=native.time, estimate=estimate, admitted=admitted,
        base_mode=owner.base_mode, task_family=owner.family, elapsed_s=following.timing[:, 0],
        ground_z=torch.zeros(3), blocked_by_stair=torch.zeros(3, dtype=torch.bool), packet=external,
        previous_command_height=previous_height, previous_elapsed_s=elapsed)
    assert imagined.response[0] == int(RequestResponse.ACCEPTED)
    assert owner.latched_height[0] == pytest.approx(.324, abs=1e-7)
    assert owner.latched_height[0] != height[0]
    torch.testing.assert_close(imagined.command, result[0].admitted, rtol=0, atol=0)
    actual = JumpTaskState.capture(native)
    for key, value in actual.record().items():
        torch.testing.assert_close(imagined.state.record()[key], value, rtol=0, atol=0)


@new_native
def test_native_height_context_is_cleared_after_failure_without_leaking_into_stream():
    owner = native_jumps(2)._terrain_course.tasks
    adapter = NativeJumpTaskAdapter(owner)
    height, mask = torch.full((2,), .4), torch.ones(2, dtype=torch.bool)
    with patch.object(adapter, '_native_step', side_effect=RuntimeError('synthetic native exception')):
        with pytest.raises(RuntimeError, match='synthetic native'):
            owner.step(torch.zeros(2, dtype=torch.float64), None, torch.zeros(2),
                       height_override=height, height_override_mask=mask)
    assert adapter._height_override is None and adapter._height_override_mask is None
    assert adapter._packet_owned is None
    target = owner.stream.current.clone()
    target[:, 3] = .36
    torch.testing.assert_close(owner.stream.step(target).requested, target, rtol=0, atol=0)


@pytest.mark.parametrize('rates', [None, CommandRates(.8, 3., .2)])
def test_world_override_admission_retains_finite_caller_gradients(rates):
    native = native_tasks(rates)
    task = MotionTask.decode(MotionTask.capture(native))
    queues, readout, _ = sensors(7, 4)
    height = torch.full((7,), .4, requires_grad=True)
    mask = torch.tensor([True, False, True, False, True, False, True])
    _, admitted = task.advance_admission(queues, readout, .02,
                                          height_override=height, height_override_mask=mask)
    admitted.sum().backward()
    assert torch.isfinite(height.grad).all()
    assert not height.grad[~mask].any()
    if rates is None:
        assert height.grad[mask].eq(1.).all()
