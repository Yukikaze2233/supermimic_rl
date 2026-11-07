"""Whole finite-cylinder vertical gaps against the immutable native atlas.

The existing active-set solver owns the cylinder mathematics. This adapter
only expands the atlas CSR and proves coverage of its true projected faces.
It does not certify target membership, contact, or collision cooking parity.
"""
from __future__ import annotations

import math

import numpy as np
import torch


class AtlasCylinderTerrainGap:
    """Use the same tile/triangles as ``AtlasTerrain``, for selected rows only.

    Inputs and returned witnesses are tile-local metres; axes are world axes
    (tiles differ only by translation). The signed WORLD-Z gap is the minimum
    over the solid finite cylinder and the highest vertically queryable atlas
    surface at each footprint point. It is not Euclidean wall clearance.

    Coverage is a once-per-load polygon difference, not a bounding-box floor.
    Residual holes remain unknown, including conservatively at their boundary.
    No ground is synthesized in a hole or outside the authored atlas domain.
    """

    def __init__(self, terrain, radii_m, lengths_m, *, tile_ids=(101, 102)):
        from transformer_rl.formal.control.wheel_clearance import CylinderGap, CylinderTerrainGap

        radii, lengths, tiles = tuple(radii_m), tuple(lengths_m), tuple(tile_ids)
        if (not radii or len(radii) != len(lengths)
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                       for v in (*radii, *lengths))
                or not tiles or len(set(tiles)) != len(tiles)
                or any(type(t) is not int or not 0 <= t < terrain.query.tile_count for t in tiles)):
            raise ValueError('Atlas cylinder geometry needs positive dimensions and distinct native tiles')
        self.terrain, self.device = terrain, terrain.query.device
        self.count, self.tile_ids = len(radii), tiles
        self._result, self._minimum = CylinderGap, CylinderTerrainGap._minimum
        self._r = torch.tensor(radii, dtype=torch.float64, device=self.device)
        self._h = torch.tensor(lengths, dtype=torch.float64, device=self.device) * .5
        self._buffers = terrain.query._buffers
        self._cell_size = terrain.query.cell_size_m
        vertices = self._buffers['vertices'].detach().cpu().double().numpy()
        faces = self._buffers['faces'].detach().cpu().numpy()
        owners = terrain.query._source_face_tiles
        if owners is None:
            if terrain.query.tile_count != 1:
                raise ValueError('Atlas cylinder coverage needs native face-to-tile provenance')
            owners = np.zeros(len(faces), dtype=np.int32)
        owners = np.asarray(owners)
        if owners.shape != (len(faces),):
            raise ValueError('Atlas face-to-tile provenance has inconsistent dimensions')
        self._face_tiles = torch.as_tensor(owners, device=self.device, dtype=torch.long)
        origins = terrain.origins.detach().cpu().double().numpy()
        self._bounds = torch.zeros((terrain.query.tile_count, 2, 2), dtype=torch.float64, device=self.device)
        self._supported = torch.zeros(terrain.query.tile_count, dtype=torch.bool, device=self.device)
        holes, hole_tiles, self.coverage = [], [], {}
        for tile in tiles:
            triangle = vertices[faces[owners == tile]] - origins[tile]
            projected_area = self._cross(triangle[:, 1, :2] - triangle[:, 0, :2],
                                         triangle[:, 2, :2] - triangle[:, 0, :2])
            # Same physical-query inclusion criterion as build_query_index.
            triangle = triangle[np.abs(projected_area) > 1e-14]
            if not len(triangle):
                raise ValueError('Atlas tile has no vertically queryable source faces')
            xy = triangle[:, :, :2]
            bounds = np.stack((xy.min((0, 1)), xy.max((0, 1))))
            missing = self._uncovered(xy, bounds)
            start = len(holes)
            for polygon in missing:
                for j in range(1, len(polygon) - 1):
                    tri = polygon[[0, j, j + 1]]
                    area = self._cross(tri[1] - tri[0], tri[2] - tri[0])
                    if area != 0:
                        if area < 0:
                            tri = tri[[0, 2, 1]]
                        holes.append(np.column_stack((tri, np.zeros(3))))
                        hole_tiles.append(tile)
            self._bounds[tile] = torch.as_tensor(bounds, device=self.device)
            self._supported[tile] = True
            self.coverage[tile] = {
                'method': 'rectangle_minus_union_of_actual_projected_triangles',
                'atlas_manifest_sha256': terrain.manifest_sha256,
                'bounds_local_xy_m': bounds.tolist(), 'queryable_face_count': len(triangle),
                'missing_convex_polygons': len(missing), 'missing_triangles': len(holes) - start,
                'missing_projected_area_m2': float(sum(self._area(p) for p in missing)),
                'domain_boundary': 'outside bounds and any residual hole intersection are unknown',
            }
        self._holes = torch.tensor(np.asarray(holes).reshape(-1, 3, 3), dtype=torch.float64, device=self.device)
        self._hole_tiles = torch.tensor(hole_tiles, dtype=torch.long, device=self.device)

    @staticmethod
    def _cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    @classmethod
    def _area(cls, polygon):
        # Subtracting a local origin avoids cancellation for thin residuals.
        p = polygon - polygon[:1]
        return abs(float(cls._cross(p, np.roll(p, -1, axis=0)).sum())) * .5

    @classmethod
    def _half_plane(cls, polygon, start, edge, inside):
        values = cls._cross(edge, polygon - start)
        if not inside:
            values = -values
        keep = values >= 0
        if keep.all():
            return polygon
        if not keep.any():
            return np.empty((0, 2))
        points = []
        for i, point in enumerate(polygon):
            previous = (i - 1) % len(polygon)
            if keep[i] != keep[previous]:
                a, b = values[previous], values[i]
                points.append(polygon[previous] + a / (a - b) * (point - polygon[previous]))
            if keep[i]:
                points.append(point)
        result = np.asarray(points).reshape(-1, 2)
        return result[np.any(result != np.roll(result, 1, axis=0), axis=1)]

    @classmethod
    def _uncovered(cls, triangles, bounds):
        """Exact planar construction in FP64; do not weld or fill small holes.

        Subtract one convex triangle by peeling its three outside half-planes.
        Every remaining piece is convex and can be fan-triangulated. Ordering
        large faces first reduces fragmentation without changing the union.
        """
        lo, hi = bounds
        remaining = [np.asarray([lo, [hi[0], lo[1]], hi, [lo[0], hi[1]]])]
        area = cls._cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        for index in np.argsort(-np.abs(area), kind='stable'):
            triangle = triangles[index] if area[index] > 0 else triangles[index, [0, 2, 1]]
            low, high = triangle.min(0), triangle.max(0)
            pieces = []
            for polygon in remaining:
                if (polygon.max(0) < low).any() or (polygon.min(0) > high).any():
                    pieces.append(polygon)
                    continue
                rest = polygon
                for i in range(3):
                    start, edge = triangle[i], triangle[(i + 1) % 3] - triangle[i]
                    outside = cls._half_plane(rest, start, edge, False)
                    if len(outside) >= 3 and cls._area(outside) > 0:
                        pieces.append(outside)
                    rest = cls._half_plane(rest, start, edge, True)
                    if len(rest) < 3 or cls._area(rest) == 0:
                        break
            remaining = pieces
            if not remaining:
                break
        return remaining

    def _candidate_pairs(self, centres, extents, tiles):
        """Flattened cylinder row / native face pairs from existing atlas CSR."""
        b = self._buffers
        origins, shapes = b['origins'][tiles].double(), b['shapes'][tiles].long()
        world = centres + self.terrain.origins[tiles].double()
        low = torch.floor((world[:, :2] - extents[:, :2] - origins) / self._cell_size).long()
        high = torch.floor((world[:, :2] + extents[:, :2] - origins) / self._cell_size).long()
        intersects = ((high >= 0) & (low < shapes)).all(-1)
        low = torch.maximum(low, torch.zeros_like(low))
        high = torch.minimum(high, shapes - 1)
        sizes = (high - low + 1).clamp_min(0)
        counts = torch.where(intersects, sizes.prod(-1), 0)
        row = torch.repeat_interleave(torch.arange(len(centres), device=self.device), counts)
        if not row.numel():
            return row, row
        rank = torch.arange(len(row), device=self.device) - torch.repeat_interleave(counts.cumsum(0) - counts, counts)
        x = low[row, 0] + rank // sizes[row, 1]
        y = low[row, 1] + rank % sizes[row, 1]
        cell = b['bases'][tiles[row]].long() + x * shapes[row, 1] + y
        begin, end = b['offsets'][cell].long(), b['offsets'][cell + 1].long()
        counts = end - begin
        cell_row = torch.repeat_interleave(torch.arange(len(row), device=self.device), counts)
        if not cell_row.numel():
            return cell_row, cell_row
        rank = torch.arange(len(cell_row), device=self.device) - torch.repeat_interleave(counts.cumsum(0) - counts, counts)
        face = b['candidates'][begin[cell_row] + rank].long()
        # One face may occur in many cells under one wheel. Evaluate it once.
        pairs = torch.unique(torch.stack((row[cell_row], face), -1), dim=0)
        if (self._face_tiles[pairs[:, 1]] != tiles[pairs[:, 0]]).any():
            raise ValueError('Atlas CSR face ownership disagrees with recorded tile provenance')
        return pairs[:, 0], pairs[:, 1]

    def _domain(self, centres, axes, extents, tiles, wheels):
        bounds = self._bounds[tiles]
        covered = ((centres[:, :2] - extents[:, :2] >= bounds[:, 0])
                   & (centres[:, :2] + extents[:, :2] <= bounds[:, 1])).all(-1)
        if not len(self._holes):
            return covered
        # Holes are a compact once-per-tile certificate, not all mesh faces.
        hole_low, hole_high = self._holes[:, :, :2].amin(1), self._holes[:, :, :2].amax(1)
        overlap = ((tiles[:, None] == self._hole_tiles[None]) & covered[:, None]
                   & (centres[:, None, :2] + extents[:, None, :2] >= hole_low).all(-1)
                   & (centres[:, None, :2] - extents[:, None, :2] <= hole_high).all(-1))
        row, hole = overlap.nonzero(as_tuple=True)
        if row.numel():
            value, _ = self._minimum(centres[row], axes[row], self._r[wheels[row]], self._h[wheels[row]],
                                     self._holes[hole])
            bad = torch.zeros_like(covered, dtype=torch.long)
            bad.scatter_reduce_(0, row, torch.isfinite(value).long(), reduce='amax', include_self=True)
            covered &= bad == 0
        return covered

    @torch.no_grad()
    def at(self, centre_local_m, axis_world, tile, selected=None):
        """Return native ``CylinderGap[B,W]``; unselected/unknown rows are invalid.

        Radius/length must come from the actual source wheel collision shapes.
        The caller obtains centres/axes from the same native tick and wraps the
        result in its existing ``WheelClearanceSample`` provenance contract.
        """
        shape = centre_local_m.shape
        if (len(shape) != 3 or shape[1:] != (self.count, 3) or axis_world.shape != shape
                or any(v.device != self.device or v.dtype not in (torch.float32, torch.float64)
                       for v in (centre_local_m, axis_world))
                or tile.shape != shape[:1] or tile.device != self.device or tile.dtype != torch.long):
            raise ValueError('Atlas cylinder queries need matching local wheel centres, axes and integer tile rows')
        if selected is None:
            selected = torch.ones(shape[0], dtype=torch.bool, device=self.device)
        if selected.shape != shape[:1] or selected.device != self.device or selected.dtype != torch.bool:
            raise ValueError('Atlas cylinder selection must be a boolean mask on the same device')
        size = shape[0] * self.count
        gap = torch.full((size,), torch.nan, dtype=torch.float32, device=self.device)
        witness = torch.full((size, 3), torch.nan, dtype=torch.float64, device=self.device)
        terrain_witness = witness.clone()
        face_out = torch.full((size,), -1, dtype=torch.long, device=self.device)
        coverage = torch.zeros(size, dtype=torch.bool, device=self.device)
        valid_out = coverage.clone()
        rows = selected.nonzero().flatten()
        if rows.numel():
            routes = tile[rows]
            if (routes < 0).any() or (routes >= len(self._supported)).any() or not self._supported[routes].all():
                raise ValueError('Selected atlas cylinders require a precomputed coverage tile')
            centres = centre_local_m[rows].double().flatten(0, 1)
            axes = axis_world[rows].double().flatten(0, 1)
            if (not torch.isfinite(centres).all() or not torch.isfinite(axes).all()
                    or not torch.allclose(axes.norm(dim=-1), torch.ones(len(axes), device=self.device,
                                                                     dtype=torch.float64), atol=1e-5, rtol=0)):
                raise ValueError('Selected cylinder centres must be finite and axes normalized')
            axes = axes / axes.norm(dim=-1, keepdim=True)
            wheels = torch.arange(self.count, device=self.device).repeat(len(rows))
            routes = routes.repeat_interleave(self.count)
            extents = (self._h[wheels, None] * axes.abs()
                       + self._r[wheels, None] * (1 - axes.square()).clamp_min(0).sqrt())
            covered = self._domain(centres, axes, extents, routes, wheels)
            targets = (rows[:, None] * self.count + torch.arange(self.count, device=self.device)).flatten()
            coverage[targets] = covered
            row, face = self._candidate_pairs(centres, extents, routes)
            if row.numel():
                triangle = self._buffers['vertices'][self._buffers['faces'][face].long()].double()
                triangle -= self.terrain.origins[routes[row]].double()[:, None]
                cross = torch.linalg.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])[:, 2]
                triangle = torch.where((cross < 0)[:, None, None], triangle[:, [0, 2, 1]], triangle)
                values, points = self._minimum(centres[row], axes[row], self._r[wheels[row]], self._h[wheels[row]], triangle)
                best = centres.new_full((len(centres),), torch.inf)
                best.scatter_reduce_(0, row, values, reduce='amin', include_self=True)
                eligible = torch.isfinite(values) & (values == best[row]) & covered[row]
                sentinel = len(self._buffers['faces'])
                winner = torch.full((len(centres),), sentinel, device=self.device, dtype=torch.long)
                winner.scatter_reduce_(0, row, torch.where(eligible, face, sentinel), reduce='amin', include_self=True)
                chosen = eligible & (face == winner[row])
                out, point = targets[row[chosen]], points[chosen]
                gap[out] = values[chosen].float()
                witness[out] = point
                terrain_witness[out] = point
                terrain_witness[out, 2] -= values[chosen]
                face_out[out] = face[chosen]
                valid_out[out] = True
        return self._result(gap.view(shape[:2]), witness.view(shape), terrain_witness.view(shape),
                            face_out.view(shape[:2]), coverage.view(shape[:2]), valid_out.view(shape[:2]))
