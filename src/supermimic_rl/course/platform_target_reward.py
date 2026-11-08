"""Continuous approach cost for an already qualified platform flight.

This is geometric shaping, not contact evidence. The existing landing observer
alone decides flight, first contact, and when both wheels have reached a target.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor

from .jump_physics import PlatformLandingPhase

PLATFORM_TARGET_REWARD_FORMAT = 'platform-flight-target-distance-v1'


@dataclass(frozen=True)
class PlatformTargetProgress:
    cost: Tensor
    eligible: Tensor
    available: Tensor
    horizontal_distance_m: Tensor
    vertical_shortfall_m: Tensor

    def validate(self, prototype):
        shape, device = prototype.shape, prototype.device
        if (self.cost.shape != shape or self.cost.device != device or not self.cost.is_floating_point()
                or any(value.shape != shape or value.device != device or value.dtype != torch.bool
                       for value in (self.eligible, self.available))
                or any(value.shape != (*shape, 2) or value.device != device or not value.is_floating_point()
                       for value in (self.horizontal_distance_m, self.vertical_shortfall_m))):
            raise ValueError('Platform target progress must match the reward rows and two wheels')

    @classmethod
    def cat(cls, items):
        items = tuple(items)
        return cls(**{field.name: torch.cat([getattr(item, field.name) for item in items]) for field in fields(cls)})

    def intervals(self, count):
        return type(self)(**{field.name: getattr(self, field.name).unflatten(0, (count, 4))
                             for field in fields(self)})

    def tick(self, index):
        return type(self)(**{field.name: getattr(self, field.name)[:, index] for field in fields(self)})


def platform_target_progress(geometry, *, centres_local_m, bottoms_local_z_m, radii_m,
                             terrain_tile, target_index, requested_rise_m, bound, landing_phase):
    """Return a dimensionless two-wheel distance, using only current geometry.

    All positions use the same tile-local metres as ``geometry``. Radii come
    from the bound source cylinders, and rise is the request's admitted value.
    Only eligible rows query the actual top-face union. A missing/occluded
    target is unavailable and costs the existing per-wheel upper bound one,
    never a fabricated zero-distance completion or a free escape from cost.
    """
    from transformer_rl.formal.tasks.skills import SkillEvidenceConfig

    n, device = len(centres_local_m), centres_local_m.device
    if (centres_local_m.shape != (n, 2, 3) or bottoms_local_z_m.shape != (n, 2)
            or radii_m.shape != (2,) or radii_m.device != device
            or any(value.shape != (n,) or value.device != device for value in
                   (terrain_tile, target_index, requested_rise_m, bound, landing_phase))
            or bound.dtype != torch.bool or landing_phase.dtype != torch.long):
        raise ValueError('Platform target cost requires current two-wheel geometry and matching task rows')
    eligible = bound & ((landing_phase == int(PlatformLandingPhase.FLIGHT))
                        | (landing_phase == int(PlatformLandingPhase.TARGET_SINGLE_WHEEL)))
    cost = centres_local_m.new_zeros(n)
    available = torch.zeros_like(eligible)
    distance = centres_local_m.new_full((n, 2), torch.nan)
    shortfall = torch.full_like(distance, torch.nan)
    rows = eligible.nonzero().flatten()
    if not rows.numel():
        return PlatformTargetProgress(cost, eligible, available, distance, shortfall)
    centres, bottoms, rise = centres_local_m[rows], bottoms_local_z_m[rows], requested_rise_m[rows]
    near = geometry.nearest(centres, terrain_tile[rows], target_index[rows])
    good = (near.source_valid & near.atlas_valid & near.visible
            & torch.isfinite(near.point_local).all(-1) & torch.isfinite(centres).all(-1)
            & torch.isfinite(bottoms) & torch.isfinite(rise)[:, None] & (rise > 0)[:, None])
    # Replace unavailable values before arithmetic, rather than masking a NaN
    # result whose backward pass could still contaminate other rows.
    point = torch.where(good[..., None], near.point_local, 0.)
    centre = torch.where(good[..., None], centres, 0.)
    bottom = torch.where(good, bottoms, 0.)
    scale = torch.maximum(torch.where(good, rise[:, None], 0.), radii_m)
    horizontal2 = (centre[..., :2] - point[..., :2]).square().sum(-1)
    deficit = (point[..., 2] - bottom - SkillEvidenceConfig().support_penetration_max_m).clamp_min(0)
    squared = (horizontal2 + deficit.square()) / scale.square()
    # Unknown/occluded geometry must not be cheaper than a visible target.
    # Each known wheel keeps its own continuous derivative even if its partner
    # is unavailable. The upper bound introduces no new coefficient or gate.
    value = torch.where(good, squared / (1 + squared), 1.).mean(-1)
    return PlatformTargetProgress(cost.index_copy(0, rows, value), eligible,
        available.index_copy(0, rows, good.all(-1)),
        distance.index_copy(0, rows, torch.where(good, horizontal2.sqrt(), torch.nan)),
        shortfall.index_copy(0, rows, torch.where(good, deficit, torch.nan)))
