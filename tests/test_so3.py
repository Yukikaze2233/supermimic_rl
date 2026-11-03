"""SO(3) helpers: exp/log inverse, quaternion round trip and yaw frame."""
import math

import torch

from supermimic_rl.world_model_v2 import so3


def test_exp_log_round_trip_including_small_and_large_angles():
    torch.manual_seed(0)
    vectors = torch.randn(256, 3, dtype=torch.float64)
    vectors = vectors / vectors.norm(dim=-1, keepdim=True) * torch.linspace(0, math.pi - 1e-3, 256,
                                                                             dtype=torch.float64)[:, None]
    torch.testing.assert_close(so3.log(so3.exp(vectors)), vectors, atol=1e-7, rtol=0)


def test_quaternion_matrix_round_trip_in_xyzw():
    torch.manual_seed(1)
    q = torch.nn.functional.normalize(torch.randn(128, 4, dtype=torch.float64), dim=-1)
    q = q * torch.where(q[:, 3:] < 0, -1., 1.)
    torch.testing.assert_close(so3.matrix_to_quaternion(so3.quaternion_to_matrix(q)), q)
    # +90 deg about z maps x to y.
    quarter = torch.tensor([0., 0., math.sin(math.pi / 4), math.cos(math.pi / 4)], dtype=torch.float64)
    torch.testing.assert_close(so3.quaternion_to_matrix(quarter) @ torch.tensor([1., 0., 0.], dtype=torch.float64),
                               torch.tensor([0., 1., 0.], dtype=torch.float64))


def test_exp_has_finite_gradient_at_zero():
    vector = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    so3.exp(vector).sum().backward()
    assert torch.isfinite(vector.grad).all()


def test_yaw_frame_keeps_only_heading():
    rotation = so3.exp(torch.tensor([.3, -.2, 1.1], dtype=torch.float64))
    heading = so3.yaw_frame(rotation)
    assert abs(float(so3.yaw(heading) - so3.yaw(rotation))) < 1e-12
    torch.testing.assert_close(heading[2], torch.tensor([0., 0., 1.], dtype=torch.float64))
