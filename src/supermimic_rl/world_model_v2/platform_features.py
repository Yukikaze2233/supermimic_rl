"""Privileged platform value context over the unchanged chassis actor interface."""
from __future__ import annotations

import torch
from torch.nn.functional import one_hot

from ..course.jump_physics import PlatformLandingPhase
from ..course.platform_intent import PLATFORM_TARGET_IDS, PlatformIntentPhase
from .so3 import yaw


PLATFORM_FEATURE_VERSION = 1
PLATFORM_FEATURE_DIM = 22


def platform_critic_features(intent, landing, physical, motion, jump, geometry, tile, config):
    """Encode a causal caller and its actual target surface, never raw IDs.

    The target offset is in the current yaw frame and metres. Time and command
    scales come from the same request/reference configuration. Ordinary rows
    contribute exactly zero; actor observations and motion-network inputs are
    not changed by this value-only context.
    """
    selected = intent.target_index >= 0
    near = geometry.nearest(physical.position, tile.long(), intent.target_index.clamp_min(0))
    if (selected & ~near.source_valid).any():
        raise ValueError('Platform value context needs the selected source surface on its actual tile')
    # Inactive rows can belong to another tile. Remove undefined geometry before
    # multiplying by yaw so its NaN cannot poison inactive-row gradients.
    point = torch.where(selected[:, None], near.point_local, physical.position)
    delta = point - physical.position
    heading = yaw(physical.rotation)
    c, s = heading.cos(), heading.sin()
    offset = torch.stack((c * delta[:, 0] + s * delta[:, 1],
                          -s * delta[:, 0] + c * delta[:, 1], delta[:, 2]), -1)
    matching = ((intent.bound_request_id >= 0) & (intent.bound_request_id == jump.physical_id))
    values = torch.cat((
        one_hot(intent.phase, len(PlatformIntentPhase)),
        one_hot(intent.target_index.clamp_min(0), len(PLATFORM_TARGET_IDS)),
        ((intent.earliest_episode_s - motion.timing[:, 0].double()) / config.request.timeout_s)[:, None],
        (intent.rise_m / config.reference.jump_max_m)[:, None],
        (intent.support_height_m / config.reference.support_height_max_m)[:, None],
        offset, matching[:, None], one_hot(landing.phase, len(PlatformLandingPhase)),
        (landing.free_ticks.clamp_max(2) / 2.)[:, None]), -1).to(physical.position)
    return torch.where(selected[:, None], values, 0.)
