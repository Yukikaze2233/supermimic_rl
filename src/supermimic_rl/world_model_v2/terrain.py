"""Same-triangle terrain queries at predicted poses, with local facet gradients.

The immutable native atlas owns the geometry and triangle selection. Autograd
uses the selected facet's exact slope inside that facet; no interpolation
smooths a step edge or bridges a hole. Tile IDs and triangle changes are
discrete. Absolute atlas coordinates never become motion-network inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch
from torch import Tensor

from . import so3


@dataclass(frozen=True)
class TerrainSample:
    height: Tensor           # Tile-local metres, NaN on a native query miss.
    valid: Tensor
    normal: Tensor           # World axes; tile frames differ only by translation.
    friction: Tensor


@dataclass(frozen=True)
class TerrainObservation:
    height_above_ground: Tensor
    scan_relative: Tensor    # 77 raw ground-minus-root heights, before critic scaling.
    valid: Tensor


class AtlasTerrain:
    """Query an immutable atlas shared with native collection, on the same device."""

    def __init__(self, query, origins: Tensor, *, manifest_sha256: str):
        if origins.shape != (query.tile_count, 3) or origins.device != query.device:
            raise ValueError('Terrain origins must cover every native tile on the query device')
        self.query, self.origins = query, origins.detach().float()
        self.manifest_sha256 = manifest_sha256
        # Match LabTerrainCourse's GridPatternCfg order: X changes fastest.
        x, y = torch.meshgrid(torch.linspace(-.5, .5, 11, device=query.device),
                              torch.linspace(-.3, .3, 7, device=query.device), indexing='xy')
        self.offsets = torch.stack((x.flatten(), y.flatten()), -1)

    @classmethod
    def load(cls, manifest_path, *, expected_sha256: str, device='cpu'):
        from transformer_rl.formal.tasks.terrain_query import StaticTerrainQuery

        path = Path(manifest_path)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError('Imagined terrain and native source use different atlas manifests')
        manifest = json.loads(raw)
        if manifest['format'] != 'terrain_course_atlas_v1':
            raise ValueError('Unsupported native terrain atlas')
        for name in ('geometry.npz', 'query_index.npz'):
            payload = (path.parent / name).read_bytes()
            pin = manifest['files'][name]
            if len(payload) != pin['bytes'] or hashlib.sha256(payload).hexdigest() != pin['sha256']:
                raise ValueError('Imagined terrain payload differs: ' + name)
        query = StaticTerrainQuery(path.parent / 'geometry.npz', path.parent / 'query_index.npz', device=device)
        origins = torch.tensor([tile['origin_m'] for tile in manifest['tiles']], device=query.device)
        return cls(query, origins, manifest_sha256=expected_sha256)

    def height_at(self, points: Tensor, tile: Tensor) -> TerrainSample:
        """Batch ``[B, ..., 3]`` tile-local points; one exact tile ID per row.

        Ground columns start at the native atlas ceiling, even for a fallen
        body below the surface. Ray altitude is not the robot's altitude.
        The forward height is the native query result; only its local XY
        derivative is supplied explicitly from the selected triangle normal.
        """
        if points.shape[-1] != 3 or points.ndim < 2 or tile.shape != points.shape[:1]:
            raise ValueError('Terrain query requires batched XYZ points and one tile per row')
        if points.device != self.query.device or tile.device != points.device:
            raise ValueError('Terrain query and predicted state must share a device')
        if not ((tile == tile.round()) & (tile >= 0) & (tile < len(self.origins))).all():
            raise ValueError('Terrain tile IDs must be recorded integer routes inside the pinned atlas')
        shape = (-1, *([1] * (points.ndim - 2)))
        origins = self.origins[tile.long()].reshape(*shape, 3)
        world = points.detach().float() + origins
        rays = torch.cat((world[..., :2], torch.full_like(world[..., 2:3], self.query._ceiling_m)), -1)
        tiles = tile.long().reshape(shape).expand(points.shape[:-1])
        height, valid, normal, friction = self.query(rays, tiles)
        valid = valid & torch.isfinite(points).all(-1)
        # Vertical faces are not ground columns and have no dz/dxy. A native
        # hit always has nz > 0; a miss stays invalid rather than gaining a floor.
        valid = valid & (normal[..., 2] > 0)
        denominator = torch.where(valid, normal[..., 2], 1.)
        slope = torch.where(valid[..., None], -normal[..., :2] / denominator[..., None], 0.).detach()
        xy = torch.where(torch.isfinite(points[..., :2]), points[..., :2], 0.)
        correction = ((xy - xy.detach()) * slope).sum(-1)
        local = (height - origins[..., 2]).to(points.dtype).detach() + correction
        local = torch.where(valid, local, torch.full_like(local, float('nan')))
        return TerrainSample(local, valid, normal, friction)

    def scan_points(self, state):
        yaw = so3.yaw(state.rotation)
        cosine, sine = yaw.cos()[..., None], yaw.sin()[..., None]
        dx, dy = self.offsets[:, 0], self.offsets[:, 1]
        delta = torch.stack((cosine * dx - sine * dy, sine * dx + cosine * dy,
                             torch.zeros_like(cosine * dx)), -1)
        return state.position[..., None, :] + delta

    def observe(self, state, tile: Tensor) -> TerrainObservation:
        points = torch.cat((state.position[..., None, :], self.scan_points(state)), -2)
        sample = self.height_at(points, tile)
        height = state.position[..., 2:3] - sample.height[..., :1]
        scan = sample.height[..., 1:] - state.position[..., 2:3]
        return TerrainObservation(height, scan, sample.valid.all(-1))

    def advance(self, height, scan, start, end, *, tile=None):
        if tile is None:
            raise ValueError('Atlas imagination requires a recorded terrain tile for every start')
        observed = self.observe(end, tile)
        return observed.height_above_ground, observed.scan_relative
