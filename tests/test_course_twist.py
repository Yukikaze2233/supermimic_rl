"""The Newton root sample uses the root body's COM twist, never the free-joint origin twist."""
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from supermimic_rl.course.twist import make_root_twist_corrected_environment


@dataclass(frozen=True)
class Sample:
    root_com_velocity_world_m_s: torch.Tensor
    angular_velocity_world_rad_s: torch.Tensor


class Native:
    def __init__(self, newton):
        self._is_newton, self._base_body_index = newton, 1
        body = torch.zeros(2, 3, 6)
        body[:, 1, :3] = torch.tensor([1., 2., 3.])
        self.robot = SimpleNamespace(data=SimpleNamespace(body_com_vel_w=body))

    @staticmethod
    def _as_torch(value):
        return value

    def _sample(self):
        return Sample(torch.full((2, 3), 9.), torch.zeros(2, 3))


def test_newton_sample_takes_the_root_body_com_velocity():
    corrected = make_root_twist_corrected_environment(Native)(newton=True)._sample()
    torch.testing.assert_close(corrected.root_com_velocity_world_m_s, torch.tensor([[1., 2., 3.]] * 2))


def test_physx_sample_is_unchanged():
    sample = make_root_twist_corrected_environment(Native)(newton=False)._sample()
    assert sample.root_com_velocity_world_m_s.eq(9).all()
