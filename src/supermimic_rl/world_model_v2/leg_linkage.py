"""Closed-chain leg geometry as a 1D function of the relative hip angle.

Each leg is driven by two COAXIAL hip motors (``L_joint1``/``LL_joint1`` and
``R_joint1``/``RR_joint1``) with the same axis direction. Rotating both by the
same angle rotates the whole leg rigidly about the hip axis without changing
any loop constraint. The leg shape is therefore a function of one angle,

    phi = q_secondary - q_primary   (LL_joint1 - L_joint1, RR_joint1 - R_joint1),

and the primary angle only rotates that shape about the base-fixed hip axis.
The table stores the closed-chain solution on a uniform phi grid inside its
assemblable domain (bounded by the knee stops) and evaluates it with C1 cubic
Hermite interpolation, which keeps phi gradients for BPTT. Queries outside
the domain are reported invalid instead of extrapolated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import torch
from torch import Tensor

FORMAT = "supermimic-leg-linkage-table-v1"


@dataclass(frozen=True)
class LinkageSide:
    """One leg; ``values``/``slopes`` are ``[knots, features]`` in float64."""

    name: str
    primary_joint: str
    secondary_joint: str
    phi_min: float
    phi_max: float
    values: Tensor
    slopes: Tensor
    reference_primary_rad: float
    hip_axis_base: Tensor
    hip_pivot_base_m: Tensor
    passive_joint_names: tuple[str, ...] = ()
    _device_buffers: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def buffers_for(self, reference: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Keep immutable table/axis constants resident on each queried device/dtype.

        Copying the whole table from CPU on every imagined tick introduces
        hundreds of synchronous transfers per BPTT rollout. These are asset
        constants, so caching changes neither values nor state derivatives.
        """
        key = (reference.device, reference.dtype)
        if key not in self._device_buffers:
            self._device_buffers[key] = tuple(value.to(device=reference.device, dtype=reference.dtype)
                for value in (self.values, self.slopes, self.hip_axis_base, self.hip_pivot_base_m))
        return self._device_buffers[key]

    @property
    def spacing(self) -> float:
        return (self.phi_max - self.phi_min) / (self.values.shape[0] - 1)

    def evaluate(self, phi: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(features[..., F], valid[...])`` at relative angles ``phi``."""
        values, slopes, _, _ = self.buffers_for(phi)
        valid = (phi >= self.phi_min) & (phi <= self.phi_max)
        # Clamping only invalid rows keeps them finite (callers must mask them)
        # without cutting the gradient at an in-domain endpoint.
        inside = torch.where(valid, phi, phi.clamp(self.phi_min, self.phi_max))
        position = (inside - self.phi_min) / self.spacing
        index = position.detach().floor().long().clamp(0, values.shape[0] - 2)
        t = (position - index)[..., None]
        h = self.spacing
        y0, y1 = values[index], values[index + 1]
        m0, m1 = slopes[index] * h, slopes[index + 1] * h
        t2, t3 = t * t, t * t * t
        result = ((2 * t3 - 3 * t2 + 1) * y0 + (t3 - 2 * t2 + t) * m0
                  + (-2 * t3 + 3 * t2) * y1 + (t3 - t2) * m1)
        return result, valid

    def derivative(self, phi: Tensor) -> Tensor:
        """Analytic derivative of the same Hermite table, for passive joint speeds."""
        values, slopes, _, _ = self.buffers_for(phi)
        position = (phi.clamp(self.phi_min, self.phi_max) - self.phi_min) / self.spacing
        index = position.detach().floor().long().clamp(0, values.shape[0] - 2)
        t = (position - index)[..., None]
        return ((6 * t.square() - 6 * t) * (values[index] - values[index + 1]) / self.spacing
                + (3 * t.square() - 4 * t + 1) * slopes[index]
                + (3 * t.square() - 2 * t) * slopes[index + 1])


class LegLinkage:
    """Both legs plus the feature layout shared by their tables."""

    def __init__(self, sides: dict[str, LinkageSide], features: tuple[str, ...],
                 feature_sizes: dict[str, int], provenance: dict):
        self.sides, self.features, self.provenance = sides, features, provenance
        self.feature_sizes = feature_sizes
        self._slices, offset = {}, 0
        for name in features:
            self._slices[name] = slice(offset, offset + feature_sizes[name])
            offset += feature_sizes[name]
        self.width = offset

    def unpack(self, packed: Tensor) -> dict[str, Tensor]:
        return {name: packed[..., index] for name, index in self._slices.items()}

    def evaluate(self, side: str, phi: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        packed, valid = self.sides[side].evaluate(phi)
        return self.unpack(packed), valid

    def wheel_center_base(self, side: str, primary_rad: Tensor, phi: Tensor) -> tuple[Tensor, Tensor]:
        """Wheel axle position in the base frame: the reference shape rotated about the hip axis."""
        features, valid = self.geometry_base(side, primary_rad, phi)
        return features['wheel_center_base_m'], valid

    def geometry_base(self, side: str, primary_rad: Tensor, phi: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        """One lookup supplies both linkage shape and its rotated wheel axle."""
        geometry = self.sides[side]
        features, valid = self.evaluate(side, phi)
        _, _, axis, pivot = geometry.buffers_for(phi)
        offset = features["wheel_center_reference_base_m"] - pivot
        angle = (primary_rad - geometry.reference_primary_rad)[..., None]
        # Rodrigues rotation of the offset about the fixed unit hip axis.
        cross = torch.linalg.cross(axis.expand_as(offset), offset, dim=-1)
        dot = (offset * axis).sum(-1, keepdim=True)
        rotated = offset * angle.cos() + cross * angle.sin() + axis * dot * (1 - angle.cos())
        return {**features, 'wheel_center_base_m': pivot + rotated}, valid

    @classmethod
    def load(cls, path: str | Path, *, expected_asset_manifest_sha256: str | None = None) -> LegLinkage:
        payload = json.loads(Path(path).read_text())
        if payload.get("format") != FORMAT:
            raise ValueError("Unknown leg linkage table format")
        provenance = payload["provenance"]
        if (expected_asset_manifest_sha256 is not None
                and provenance["asset_manifest_sha256"] != expected_asset_manifest_sha256):
            raise ValueError("Leg linkage table was built for a different asset")
        sides = {}
        for name, side in payload["sides"].items():
            values = torch.tensor(side["values"], dtype=torch.float64)
            slopes = torch.tensor(side["slopes"], dtype=torch.float64)
            if values.shape != slopes.shape or values.ndim != 2 or values.shape[0] < 2:
                raise ValueError("Leg linkage values and slopes must share a [knots, features] shape")
            sides[name] = LinkageSide(
                name, side["primary_joint"], side["secondary_joint"], side["phi_min"], side["phi_max"],
                values, slopes, side["reference_primary_rad"],
                torch.tensor(side["hip_axis_base"], dtype=torch.float64),
                torch.tensor(side["hip_pivot_base_m"], dtype=torch.float64),
                tuple(side["passive_joints"]))
        return cls(sides, tuple(payload["features"]), payload["feature_sizes"], provenance)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
