"""A small terrain atlas, a two-surface query and physical states."""

import torch
from supermimic_rl.world_model_v2 import so3
from supermimic_rl.world_model_v2.physical import PhysicalState
from supermimic_rl.world_model_v2.terrain import AtlasTerrain


class TwoSurfaceQuery:
    """Two translated independent tiles: a sloped facet and a 200 mm step."""
    tile_count, device, _ceiling_m = 2, torch.device('cpu'), 10.
    origins = torch.tensor([[100., 20., 3.], [-20., 30., -2.]])

    @torch.no_grad()
    def __call__(self, points, tiles):
        assert (points[..., 2] == self._ceiling_m).all()
        local = points - self.origins[tiles]
        slope = torch.tensor([.2, -.1])
        height = torch.where(tiles == 0, (local[..., :2] * slope).sum(-1), (local[..., 0] >= 0) * .2)
        normal = torch.zeros_like(points)
        normal[..., 2] = 1.
        normal[..., :2] = torch.where((tiles == 0)[..., None], -slope, 0.)
        normal /= normal.norm(dim=-1, keepdim=True)
        valid = (local[..., :2].abs() < 5).all(-1)
        height = torch.where(valid, height + self.origins[tiles, 2], float('nan'))
        return height, valid, normal, torch.full_like(height, .7)


def state(position, yaw=None):
    batch = position.shape[:-1]
    angles = torch.zeros(*batch, 3, dtype=position.dtype)
    if yaw is not None:
        angles[..., 2] = yaw
    return PhysicalState(position, so3.exp(angles), torch.zeros_like(position), torch.zeros_like(position),
                         torch.zeros(*batch, 6, dtype=position.dtype), torch.zeros(*batch, 8, dtype=position.dtype))


def atlas():
    query = TwoSurfaceQuery()
    return AtlasTerrain(query, query.origins, manifest_sha256='test')
