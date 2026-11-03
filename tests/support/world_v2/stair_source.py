"""Ideal ToF stair source with a two-riser terrain query."""

import torch
from supermimic_rl.world_model_v2.stair_source import IdealStairSource
from supermimic_rl.world_model_v2.terrain import AtlasTerrain
from transformer_rl.formal.lab_env.stair_trigger import SimulatedTofTrigger, TofTriggerConfig


class TwoRiserQuery:
    tile_count, device, _ceiling_m = 1, torch.device('cpu'), 10.
    origins = torch.tensor([[100., 20., 3.]])

    def __call__(self, points, tiles):
        local = points - self.origins[tiles]
        height = .10 * (local[..., 0] >= 0) + .10 * (local[..., 0] >= .60)
        normal = torch.zeros_like(points)
        normal[..., 2] = 1.
        valid = (local[..., :2].abs() < 10).all(-1)
        return height + self.origins[tiles, 2], valid, normal, torch.full_like(height, .7)


def source_fixture(n):
    query = TwoRiserQuery()
    terrain = AtlasTerrain(query, query.origins, manifest_sha256='fixture')
    trigger = SimulatedTofTrigger(n, min_height_m=.04, max_height_m=.20,
                                  config=TofTriggerConfig(), seed=51)
    trigger.reset(torch.arange(n))
    source = IdealStairSource(trigger.config, min_height_m=.04, max_height_m=.20, terrain=terrain)
    return source, trigger
