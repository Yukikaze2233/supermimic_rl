"""Deterministic platform plans selected from an already placed course birth.

This producer never draws terrain, changes a pose or samples a command. Its
input is the actual tile and task family after the native reset has finished.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .platform_intent import PLATFORM_TARGET_IDS, PlatformIntentState


PLATFORM_SAMPLER_FORMAT = 'actual-route-platform-plans-v1'


@dataclass(frozen=True)
class PlatformRouteRule:
    tile_id: int
    target_id: str
    task_families: tuple[str, ...]
    earliest_episode_s: float
    rise_m: float
    support_height_m: float

    def to_dict(self):
        return {'tile_id': self.tile_id, 'target_id': self.target_id,
                'task_families': list(self.task_families),
                'earliest_episode_s': self.earliest_episode_s,
                'rise_m': self.rise_m, 'support_height_m': self.support_height_m}


@dataclass(frozen=True)
class PlatformSamplerConfig:
    rules: tuple[PlatformRouteRule, ...]

    @classmethod
    def read(cls, value):
        from transformer_rl.formal.tasks.curriculum import TaskFamily

        if (not isinstance(value, dict) or set(value) != {'rules'}
                or not isinstance(value['rules'], list) or not value['rules']):
            raise ValueError('Platform sampler requires explicit nonempty rules')
        names = {'tile_id', 'target_id', 'task_families', 'earliest_episode_s',
                 'rise_m', 'support_height_m'}
        result, used = [], set()
        for raw in value['rules']:
            if (not isinstance(raw, dict) or set(raw) != names
                    or type(raw['tile_id']) is not int or raw['tile_id'] not in (101, 102)
                    or raw['target_id'] not in PLATFORM_TARGET_IDS
                    or raw['target_id'] != f"{raw['tile_id']}:profile:1"):
                raise ValueError('Platform sampler rule must identify its actual supported tile and target')
            families = raw['task_families']
            if (not isinstance(families, list) or not families
                    or any(not isinstance(name, str) or name not in TaskFamily.__members__ for name in families)
                    or len(set(families)) != len(families) or 'RECOVERY' in families):
                raise ValueError('Platform sampler requires explicit non-recovery task families')
            for name in ('earliest_episode_s', 'rise_m', 'support_height_m'):
                number = raw[name]
                if (type(number) not in (int, float) or not math.isfinite(number)
                        or number < 0 or name != 'earliest_episode_s' and number == 0):
                    raise ValueError('Platform sampler requires explicit valid ' + name)
            keys = {(raw['tile_id'], name) for name in families}
            if used & keys:
                raise ValueError('Platform sampler rules overlap on a tile and task family')
            used.update(keys)
            result.append(PlatformRouteRule(**{**raw, 'task_families': tuple(families)}))
        return cls(tuple(result))

    def to_dict(self):
        return {'rules': [rule.to_dict() for rule in self.rules]}

    def validate_catalog(self, catalog):
        targets = {target['target_id']: target for target in catalog['targets']}
        for rule in self.rules:
            if targets[rule.target_id]['tile_id'] != rule.tile_id:
                raise ValueError('Platform sampler tile differs from the pinned target catalog')

    def select(self, tile, family, mode):
        """Return only these reset rows; selection has no mutable or RNG state."""
        from transformer_rl.formal.control.observation import CommandMode
        from transformer_rl.formal.tasks.curriculum import TaskFamily

        if (tile.ndim != 1 or tile.dtype != torch.long
                or any(value.shape != tile.shape or value.dtype != torch.long
                       or value.device != tile.device for value in (family, mode))):
            raise ValueError('Platform sampler requires matching actual tile/family/mode rows')
        state = PlatformIntentState.inactive(len(tile), tile.device)
        targets = state.target_index
        earliest, rise, support = state.earliest_episode_s, state.rise_m, state.support_height_m
        for rule in self.rules:
            families = tile.new_tensor([int(TaskFamily[name]) for name in rule.task_families])
            selected = ((tile == rule.tile_id) & torch.isin(family, families)
                        & (mode != int(CommandMode.RECOVERY)))
            targets = torch.where(selected, PLATFORM_TARGET_IDS.index(rule.target_id), targets)
            earliest = torch.where(selected, rule.earliest_episode_s, earliest)
            rise = torch.where(selected, rule.rise_m, rise)
            support = torch.where(selected, rule.support_height_m, support)
        return PlatformIntentState.plan(target_index=targets, earliest_episode_s=earliest,
                                        rise_m=rise, support_height_m=support)
