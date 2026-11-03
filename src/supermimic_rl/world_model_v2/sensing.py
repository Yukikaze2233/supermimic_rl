"""Deployed sensor queues advanced by code during imagination.

The native sensor model pushes one encoder/IMU sample per physics tick into a
short ring and the actor reads it with a per-episode delay. Snapshots store
that ring newest-first (``sensor.values``, ``sensor.gimbal_history``,
``sensor.heading``), clamped to the oldest real sample after a reset. This
module reproduces the per-tick sample and the push, keeping gradients, so the
deployed 45D frame is reassembled by ``formal_observation.deployed_frame``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SensorTransforms:
    """Fixed per-episode measurement transforms from the requested randomization."""

    policy_encoder_bias_rad: Tensor   # [N, 6]; wheels zero
    gyro_bias_rad_s: Tensor           # [N, 3]
    imu_sensor_to_body: Tensor        # [N, 3, 3]
    gimbal_sign: Tensor               # [N, 2]
    gimbal_zero_rad: Tensor           # [N, 2]


def encoder_imu_sample(leg_position_rad: Tensor, policy_velocity_rad_s: Tensor,
                       angular_velocity_body_rad_s: Tensor, gravity_body: Tensor,
                       transforms: SensorTransforms) -> Tensor:
    """One 18D tick sample: q(4 legs, 2 zero wheel slots), dq(6), gyro(3), gravity(3).

    Matches the requested transform: the policy encoder sees true position plus
    its own bias; the IMU reading is rotated into the sensor frame and the gyro
    receives its bias. Noise is added only when the delayed frame is read.
    """
    inverse = transforms.imu_sensor_to_body.transpose(-1, -2)
    position = leg_position_rad + transforms.policy_encoder_bias_rad[..., :4]
    gyro = (inverse @ angular_velocity_body_rad_s[..., None]).squeeze(-1) + transforms.gyro_bias_rad_s
    gravity = (inverse @ gravity_body[..., None]).squeeze(-1)
    return torch.cat((position, torch.zeros_like(position[..., :2]), policy_velocity_rad_s, gyro, gravity), -1)


def gimbal_sample(position_rad: Tensor, velocity_rad_s: Tensor, transforms: SensorTransforms) -> Tensor:
    """sin/cos of controller yaw, controller pitch and both controller rates (5D)."""
    position = (position_rad - transforms.gimbal_zero_rad) * transforms.gimbal_sign
    velocity = velocity_rad_s * transforms.gimbal_sign
    return torch.cat((position[..., :1].sin(), position[..., :1].cos(), position[..., 1:2], velocity), -1)


def heading_sample(yaw_rad: Tensor) -> Tensor:
    return torch.stack((yaw_rad.cos(), yaw_rad.sin()), -1)


@dataclass(frozen=True)
class SensorQueues:
    """Newest-first rings ``[N, capacity, width]`` and the count of real samples."""

    values: Tensor
    gimbal: Tensor
    heading: Tensor
    count: Tensor

    @property
    def capacity(self) -> int:
        return self.values.shape[-2]

    def push(self, values: Tensor, gimbal: Tensor, heading: Tensor) -> SensorQueues:
        """Append one physics tick; functional so imagined samples keep their gradients."""
        def shifted(queue, sample):
            return torch.cat((sample[..., None, :], queue[..., :-1, :]), -2)

        return SensorQueues(shifted(self.values, values), shifted(self.gimbal, gimbal),
                            shifted(self.heading, heading), (self.count + 1).clamp_max(self.capacity))
