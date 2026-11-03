import pytest
import torch

from supermimic_rl.gimbal import GimbalConfig, IndependentGimbalController


def controller(**kwargs):
    return IndependentGimbalController(GimbalConfig(**kwargs), 3, "cpu", 42, [1, -1], [4.5, 4.5])


def test_servo_maps_pitch_and_clips_actual_torque():
    servo = controller(yaw_velocity_rad_s=(20, 20), pitch_position_rad=(1, 1))
    servo.reset(None)
    torque = servo.effort(torch.zeros(3, 2), torch.zeros(3, 2))
    torch.testing.assert_close(torque, torch.tensor([[4.5, -4.5]]).expand(3, -1))
    # At the native position/velocity matching the control command, no error.
    torque = servo.effort(torch.tensor([[0., -1.]]).expand(3, -1),
                         torch.tensor([[20., 0.]]).expand(3, -1))
    assert torch.equal(torque, torch.zeros_like(torque))


def test_partial_reset_preserves_other_environments_and_seed_is_reproducible():
    left, right = controller(), controller()
    left.reset(None)
    right.reset(None)
    assert torch.equal(left.command, right.command)
    previous = left.command.clone()
    left.last_effort.fill_(1.)
    left.reset(torch.tensor([1]))
    assert torch.equal(left.command[[0, 2]], previous[[0, 2]])
    assert torch.equal(left.last_effort[[0, 2]], torch.ones(2, 2))
    assert torch.equal(left.last_effort[1], torch.zeros(2))


def test_damping_opposes_motion_in_both_native_axes():
    servo = controller(yaw_velocity_rad_s=(0, 0), pitch_position_rad=(0, 0))
    servo.reset(None)
    velocity = torch.tensor([[2., -3.], [-2., 3.], [0., 0.]])
    effort = servo.effort(torch.zeros_like(velocity), velocity)
    assert (effort * velocity).sum(-1).le(0).all()


def test_invalid_command_range_rejected():
    with pytest.raises(ValueError):
        GimbalConfig(yaw_velocity_rad_s=(2, -2))
