"""Local-frame increments are an exact inverse pair and invariant to heading and position."""
import math

import torch

from supermimic_rl.world_model_v2 import so3
from supermimic_rl.world_model_v2.physical import GIMBAL_YAW_COLUMN, PhysicalState, apply, increments


def random_state(generator, n=64):
    r = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.float64)
    q = r(n, 6)
    q[:, GIMBAL_YAW_COLUMN] = so3.wrap(q[:, GIMBAL_YAW_COLUMN] * 3)
    return PhysicalState(r(n, 3), so3.exp(r(n, 3)), r(n, 3), r(n, 3), q, r(n, 8))


def test_apply_inverts_increments_including_gimbal_yaw_wrap():
    g = torch.Generator().manual_seed(0)
    start, end = random_state(g), random_state(g)
    # Force a gimbal yaw step across the +-pi cut.
    start = PhysicalState(start.position, start.rotation, start.velocity, start.angular_velocity,
                          torch.cat((start.joint_position[:, :4], torch.full((64, 1), math.pi - .05, dtype=torch.float64),
                                     start.joint_position[:, 5:]), -1), start.joint_velocity)
    end = PhysicalState(end.position, end.rotation, end.velocity, end.angular_velocity,
                        torch.cat((end.joint_position[:, :4], torch.full((64, 1), -math.pi + .05, dtype=torch.float64),
                                   end.joint_position[:, 5:]), -1), end.joint_velocity)
    delta = increments(start, end)
    assert torch.allclose(delta[:, 12 + GIMBAL_YAW_COLUMN], torch.full((64,), .1, dtype=torch.float64))
    rebuilt = apply(start, delta)
    for name in ("position", "rotation", "velocity", "angular_velocity", "joint_position", "joint_velocity"):
        torch.testing.assert_close(getattr(rebuilt, name), getattr(end, name))


def test_increments_are_invariant_to_world_heading_and_translation():
    g = torch.Generator().manual_seed(1)
    start, end = random_state(g), random_state(g)
    turn = so3.exp(torch.tensor([0., 0., .7], dtype=torch.float64)).expand(64, 3, 3)
    shift = torch.tensor([3., -2., 0.], dtype=torch.float64)

    def moved(state):
        rotate = lambda v: (turn @ v[..., None]).squeeze(-1)
        return PhysicalState(rotate(state.position) + shift, turn @ state.rotation, rotate(state.velocity),
                             rotate(state.angular_velocity), state.joint_position, state.joint_velocity)

    torch.testing.assert_close(increments(moved(start), moved(end)), increments(start, end))
