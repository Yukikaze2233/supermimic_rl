"""Task progression matches the pinned native command/sensor/gimbal implementation."""
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.formal_observation import SensorReadout, known_horizon
from supermimic_rl.world_model_v2.motion_task import MotionTask, MOTION_TASK_DIM
from supermimic_rl.world_model_v2.sensing import SensorQueues, heading_sample

from transformer_rl.formal.control.sensors import SensorModel
from transformer_rl.formal.lab_env.course_tasks import LabCourseTasks
from transformer_rl.formal.tasks.gimbal_course import GimbalCourse, GimbalCourseConfig
from transformer_rl.formal.tasks.trajectory import CommandRates, CommandStream, CommandTrajectory, DriveEnvelope


def native_tasks(rates):
    n = 7
    tasks = LabCourseTasks(n, DriveEnvelope(.08, .38, 90.), policy_dt_s=.02,
                           heading_gain=1.3, heading_yaw_limit_rad_s=2.5)
    tasks.elapsed.fill_(6.)
    tasks.segment_kind.fill_(1)
    start = torch.tensor([[.2, 0., .1, .30]]).repeat(n, 1)
    end = torch.tensor([[9., .4, 8., .38]]).repeat(n, 1)  # Exercise projection, not hardware targets.
    tasks.stream = CommandStream(start, tasks.stream.envelope, rates, policy_dt_s=.02)
    duration = torch.linspace(.28, .65, n)
    tasks.trajectory = CommandTrajectory(start, end, duration, torch.arange(7))
    tasks.heading_enabled[:] = torch.arange(n) % 2 == 1
    tasks.heading_offset_rad[:] = torch.linspace(-3., 3., n)
    course = GimbalCourse(GimbalCourseConfig(enabled=True, spin_translate=True), n)
    course.special[[0, 2, 4]] = True
    course.speed.fill_(.5)
    course.direction[:] = torch.linspace(-2.7, 2.7, n)
    course.chassis_rate.fill_(8.)
    course.remaining_s.fill_(10.)
    native = SimpleNamespace(_terrain_course=SimpleNamespace(tasks=tasks), _gimbal_course=course)
    return native


def sensors(n, tick, *, requires_grad=False):
    yaw = torch.arange(n)[:, None] * .7 + torch.tensor([tick, tick - 1, tick - 2])[None] * .03
    yaw.requires_grad_(requires_grad)
    queues = SensorQueues(torch.zeros(n, 3, 18), torch.zeros(n, 3, 5), heading_sample(yaw), torch.full((n,), 3.))
    lags = torch.tensor([[0., 1., 0., 2.]]).repeat(n, 1)
    readout = SensorReadout(lags, torch.zeros(n, 1, dtype=torch.bool), torch.zeros(n, 18),
                           torch.zeros(n, 1), torch.zeros(n, 6))
    return queues, readout, yaw


@pytest.mark.parametrize('rates', [None, CommandRates(.8, 3., .2)])
def test_all_seven_native_trajectories_heading_and_gimbal_modes_over_32_steps(rates):
    native = native_tasks(rates)
    task = MotionTask.decode(MotionTask.capture(native))
    assert task.encode().shape == (7, MOTION_TASK_DIM)
    goal = torch.zeros(7, 16)
    goal[:, 4] = goal[:, 11] = 1.
    for step in range(32):
        queues, readout, _ = sensors(7, (step + 1) * 4)
        sensor = SimpleNamespace(lags=readout.ages.long(), count=queues.count.long(), capacity=3,
                                 physical_tick=2, _rows=torch.arange(7), uplink_delay_s=None)
        yaw = SensorModel._delayed_angle(sensor, queues.heading.flip(1).transpose(0, 1), group=3)
        native_task, gimbal = native._terrain_course.tasks, native._gimbal_course
        admission, mode, _, _ = native_task.step(torch.full((7,), (step + 1) * .02, dtype=torch.float64),
                                                None, torch.zeros(7), yaw_rad=yaw)
        eligible = ((native_task.terrain_kind == 0) & (native_task.segment_kind == 1) & (mode == 0)
                    & (native_task.elapsed >= gimbal.config.spin_after_episode_s)
                    & (admission.admitted[:, :3] != 0).any(-1))
        gimbal.advance(.02, torch.zeros(7, 2), eligible, update=0,
                       segment_boundary=torch.zeros(7, dtype=torch.bool))
        command = gimbal.body_command(admission.admitted)
        expected = torch.cat((command[:, :3], command[:, 3:4] * 5., goal[:, 4:11],
                              gimbal.context(torch.zeros(7))[:, :5]), -1)
        task, goal = task.advance(queues, readout, goal, .02)
        torch.testing.assert_close(goal, expected, rtol=0, atol=1e-6)
        torch.testing.assert_close(task.current, admission.admitted, rtol=0, atol=1e-6)
        torch.testing.assert_close(task.heading[:, 2], native_task.heading_target_rad, rtol=0, atol=1e-6)
        torch.testing.assert_close(task.encode(), MotionTask.capture(native), rtol=0, atol=1e-6)


def test_captured_parameters_do_not_alias_later_native_mutations():
    native = native_tasks(None)
    recorded = MotionTask.capture(native)
    initial = recorded.clone()
    native._terrain_course.tasks.trajectory.start.add_(1.)
    native._terrain_course.tasks.stream.current.zero_()
    native._gimbal_course.direction.add_(1.)
    torch.testing.assert_close(recorded, initial)


def test_heading_gradient_follows_delayed_measurement_not_clean_yaw():
    native = native_tasks(None)
    native._terrain_course.tasks.heading_enabled.fill_(True)
    native._terrain_course.tasks.heading_valid.fill_(True)
    native._terrain_course.tasks.heading_target_rad.fill_(.7)
    native._gimbal_course.special.zero_()
    task = MotionTask.decode(MotionTask.capture(native))
    queues, readout, yaw = sensors(7, 4, requires_grad=True)
    goal = torch.zeros(7, 16)
    _, following = task.advance(queues, readout, goal, .02)
    following[:, 2].sum().backward()
    assert torch.isfinite(yaw.grad).all()
    assert (yaw.grad[:, :2] == 0).all()
    assert yaw.grad[:, 2].abs().max() > .5
    # A known dynamic segment admits multiple steps, but its next stochastic
    # boundary still limits the endpoint. Request rows retain their own route.
    assert known_horizon(torch.tensor([.35]), torch.zeros(1), goal[:1], .02).item() == 17
    goal[0, 8] = 1.
    assert known_horizon(torch.tensor([.35]), torch.zeros(1), goal[:1], .02).item() == 1


def test_constant_commands_have_finite_zero_slew_gradients():
    native = native_tasks(CommandRates(.8, 3., .2))
    task = MotionTask.decode(MotionTask.capture(native))
    target = task.current.detach().requires_grad_()
    task._admit(target, .02).sum().backward()
    assert torch.isfinite(target.grad).all()
