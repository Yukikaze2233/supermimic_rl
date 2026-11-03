"""Rotation helpers for v2 measurements and modules (quaternions are xyzw).

The smooth exp/log and the yaw/6D helpers follow supermimic_vibed@8bfc115
(MIT, rotations.py), converted from its wxyz convention.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def quaternion_to_matrix(quaternion_xyzw: Tensor) -> Tensor:
    q = F.normalize(quaternion_xyzw, dim=-1)
    x, y, z, w = q.unbind(-1)
    return torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ), -1).reshape(*q.shape[:-1], 3, 3)


def matrix_to_quaternion(matrix: Tensor) -> Tensor:
    """Robust matrix -> xyzw quaternion with w >= 0."""
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    qw = torch.sqrt(torch.clamp(1 + m00 + m11 + m22, min=1e-10)) / 2
    qx = torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=1e-10)) / 2
    qy = torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=1e-10)) / 2
    qz = torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=1e-10)) / 2
    candidates = torch.stack((
        torch.stack(((m21 - m12) / (4 * qw), (m02 - m20) / (4 * qw), (m10 - m01) / (4 * qw), qw), -1),
        torch.stack((qx, (m01 + m10) / (4 * qx), (m02 + m20) / (4 * qx), (m21 - m12) / (4 * qx)), -1),
        torch.stack(((m01 + m10) / (4 * qy), qy, (m12 + m21) / (4 * qy), (m02 - m20) / (4 * qy)), -1),
        torch.stack(((m02 + m20) / (4 * qz), (m12 + m21) / (4 * qz), qz, (m10 - m01) / (4 * qz)), -1),
    ), -2)
    index = torch.stack((qw, qx, qy, qz), -1).argmax(-1)
    q = torch.gather(candidates, -2, index[..., None, None].expand(*index.shape, 1, 4)).squeeze(-2)
    q = F.normalize(q, dim=-1)
    return q * torch.where(q[..., 3:] < 0, -1., 1.)


def skew(vector: Tensor) -> Tensor:
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*vector.shape[:-1], 3, 3)


def exp(rotation_vector: Tensor) -> Tensor:
    """Rotation vector -> matrix, smooth (and differentiable) at zero."""
    theta2 = (rotation_vector * rotation_vector).sum(-1, keepdim=True)
    theta = torch.sqrt(theta2 + 1e-12)
    small = theta2 < 1e-8
    a = torch.where(small, 1 - theta2 / 6, torch.sin(theta) / theta)
    b = torch.where(small, .5 - theta2 / 24, (1 - torch.cos(theta)) / (theta2 + 1e-12))
    k = skew(rotation_vector)
    eye = torch.eye(3, device=rotation_vector.device, dtype=rotation_vector.dtype).expand(k.shape)
    return eye + a[..., None] * k + b[..., None] * (k @ k)


def log(rotation: Tensor) -> Tensor:
    """Matrix -> rotation vector via the quaternion (stable near 0 and pi)."""
    q = matrix_to_quaternion(rotation)
    vector, scalar = q[..., :3], q[..., 3:]
    sin_half = torch.linalg.norm(vector, dim=-1, keepdim=True)
    half = torch.atan2(sin_half, scalar)
    scale = torch.where(sin_half < 1e-6, 2. / torch.clamp(scalar, min=1e-6), 2 * half / (sin_half + 1e-12))
    return vector * scale


def angle(rotation: Tensor) -> Tensor:
    """Rotation angle of a matrix; accurate away from pi, sufficient for residuals."""
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.acos(((trace - 1) / 2).clamp(-1, 1))


def yaw_frame(rotation: Tensor) -> Tensor:
    """Heading (yaw-only) rotation of ``rotation``, from its projected x axis."""
    fx, fy = rotation[..., 0, 0], rotation[..., 1, 0]
    norm = torch.sqrt(fx * fx + fy * fy + 1e-9)
    c, s = fx / norm, fy / norm
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, zero, s, c, zero, zero, zero, one), -1).reshape(*rotation.shape[:-2], 3, 3)


def yaw(rotation: Tensor) -> Tensor:
    return torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])


def to_6d(rotation: Tensor) -> Tensor:
    """First two columns, flattened (continuous rotation representation)."""
    return rotation[..., :, :2].reshape(*rotation.shape[:-2], 6)


def wrap(angle_rad: Tensor) -> Tensor:
    return torch.atan2(torch.sin(angle_rad), torch.cos(angle_rad))
