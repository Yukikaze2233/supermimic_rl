"""Pinned source-triangle targets, not rectangular platform approximations.

Only geometric surface membership is established here. Contact, wheel extent,
clearance, flight and landing stability belong to the task observer. Coordinates
are tile-local metres; no world position is added to the actor observation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


@dataclass(frozen=True)
class PlatformSurfaceSample:
    height: Tensor
    inside: Tensor
    edge_distance: Tensor
    in_footprint: Tensor
    target_visible: Tensor
    atlas_valid: Tensor


class PlatformGeometry:
    """Immutable target geometry on the same device as the physical atlas.

    Height is the target upper envelope (NaN outside its ROI/triangle union).
    ``inside`` additionally requires this envelope to be the atlas top surface.
    Signed edge distance is to the XY footprint boundary, positive inside and
    negative outside. Occlusion changes visibility, not footprint distance.
    All face/region choices are discrete; height and distance retain their local
    derivatives. A boundary or facet switch is not smoothed into a new surface.
    """

    DEFAULT_TARGETS = ('101:profile:1', '102:profile:1')

    @classmethod
    def load(cls, catalog_path, *, expected_sha256: str, terrain,
             target_ids=DEFAULT_TARGETS):
        raw = Path(catalog_path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError('Platform catalog pin differs')
        catalog = json.loads(raw)
        if catalog['format'] != 'authored-platform-triangle-targets-v1':
            raise ValueError('Unsupported platform target catalog')
        source = catalog['source']
        if source['atlas_manifest']['sha256'] != terrain.manifest_sha256:
            raise ValueError('Platform targets and physical atlas differ')
        targets = {target['target_id']: target for target in catalog['targets']}
        if len(targets) != len(catalog['targets']) or not target_ids or len(set(target_ids)) != len(target_ids):
            raise ValueError('Platform targets must have unique identities')
        selected = [targets[name] for name in target_ids]
        triangles, boundaries, tiles, rois, tolerances = [], [], [], [], []
        for target in selected:
            ids = target['triangle_ids']
            if not ids or len(set(ids)) != len(ids):
                raise ValueError('Platform target needs distinct source faces')
            faces = [catalog['face_catalog'][str(face)] for face in ids]
            if any(face['tile_id'] != target['tile_id'] or face['source_part_role'] != 'prepared_source'
                   for face in faces):
                raise ValueError('Platform target contains padding or another tile')
            triangle = np.asarray([face['tile_local_vertices_m'] for face in faces], dtype=np.float64)
            roi = np.asarray(target['source_roi_xy_m'], dtype=np.float64)
            if (triangle.shape != (len(ids), 3, 3) or not np.isfinite(triangle).all()
                    or roi.shape != (2, 2) or not np.isfinite(roi).all() or not (roi[0] < roi[1]).all()):
                raise ValueError('Platform geometry needs finite triangles and a nonempty source ROI')
            cross = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
            if (cross[:, 2] == 0).any():
                raise ValueError('Platform target contains a vertical or degenerate face')
            tolerance = float(target['sparse_query']['numerical_z_tolerance_m'])
            if not np.isfinite(tolerance) or tolerance <= 0:
                raise ValueError('Platform catalog lacks its numerical height tolerance')
            triangles.append(triangle)
            boundaries.append(cls._union_boundary(triangle[:, :, :2], roi))
            tiles.append(target['tile_id'])
            rois.append(roi)
            tolerances.append(tolerance)
        self = cls()
        self.terrain, self.target_ids, self.catalog_sha256 = terrain, tuple(target_ids), expected_sha256
        self.geometry_sha256 = source['atlas_geometry']['sha256']
        self.device = terrain.origins.device
        # Padding is masked, never a source triangle or a fill for a source hole.
        self.triangles, self.face_mask = cls._padded(triangles, (3, 3), self.device)
        self.boundaries, self.edge_mask = cls._padded(boundaries, (2, 2), self.device)
        self.tiles = torch.tensor(tiles, dtype=torch.long, device=self.device)
        self.rois = torch.as_tensor(np.asarray(rois), device=self.device, dtype=torch.float32)
        self.height_tolerance = torch.tensor(tolerances, device=self.device, dtype=torch.float32)
        return self

    @staticmethod
    def _padded(arrays, trailing_shape, device):
        width = max(map(len, arrays))
        data = np.zeros((len(arrays), width, *trailing_shape), dtype=np.float64)
        mask = np.zeros((len(arrays), width), dtype=bool)
        for row, values in enumerate(arrays):
            data[row, :len(values)], mask[row, :len(values)] = values, True
        return torch.as_tensor(data, device=device, dtype=torch.float32), torch.as_tensor(mask, device=device)

    @staticmethod
    def _clip_triangle(triangle, roi):
        polygon = triangle.copy()
        for axis, bound, sign in ((0, roi[0, 0], 1), (0, roi[1, 0], -1),
                                  (1, roi[0, 1], 1), (1, roi[1, 1], -1)):
            if not len(polygon):
                break
            result = []
            previous = polygon[-1]
            for point in polygon:
                left, right = sign * (previous[axis] - bound), sign * (point[axis] - bound)
                if (left >= 0) != (right >= 0):
                    result.append(previous + left / (left - right) * (point - previous))
                if right >= 0:
                    result.append(point)
                previous = point
            polygon = np.asarray(result).reshape(-1, 2)
        if len(polygon) >= 3:
            area = np.sum(_cross2(polygon, np.roll(polygon, -1, axis=0)))
            if area < 0:
                polygon = polygon[::-1]
        return polygon

    @classmethod
    def _union_boundary(cls, triangles, roi):
        """Split all polygon intersections, then keep only exposed subsegments.

        This once-per-load CPU construction retains holes, overlaps and T joints.
        The infinitesimal-side test needs no physical offset or welding radius.
        """
        polygons = [cls._clip_triangle(triangle, roi) for triangle in triangles]
        polygons = [p for p in polygons if len(p) >= 3]
        if not polygons:
            raise ValueError('No platform source face intersects its ROI')
        edges = np.concatenate([np.stack((p, np.roll(p, -1, axis=0)), axis=1) for p in polygons])
        edges = edges[np.linalg.norm(edges[:, 1] - edges[:, 0], axis=-1) > 0]
        scale = max(1., float(np.abs(edges).max()))
        epsilon = 64 * np.finfo(np.float64).eps * scale * scale
        a, direction = edges[:, 0], edges[:, 1] - edges[:, 0]
        pieces = []
        for start, delta in zip(a, direction):
            offset = a - start
            cross = _cross2(delta, direction)
            regular = np.abs(cross) > epsilon
            safe = np.where(regular, cross, 1.)
            t = _cross2(offset, direction) / safe
            u = _cross2(offset, delta) / safe
            crossing = regular & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
            collinear = ~regular & (np.abs(_cross2(offset, delta)) <= epsilon)
            projected = ((edges[collinear] - start) @ delta / (delta @ delta)).reshape(-1)
            cuts = np.unique(np.concatenate(([0., 1.], t[crossing], projected[(projected > 0) & (projected < 1)])))
            for lo, hi in zip(cuts[:-1], cuts[1:]):
                if hi - lo <= epsilon:
                    continue
                midpoint = start + (.5 * (lo + hi)) * delta
                normal = np.asarray([-delta[1], delta[0]])
                sides = [False, False]
                for polygon in polygons:
                    edge = np.roll(polygon, -1, axis=0) - polygon
                    values = _cross2(edge, midpoint - polygon)
                    if (values < -epsilon).any():
                        continue
                    on = np.abs(values) <= epsilon
                    derivative = _cross2(edge, normal)
                    sides[0] |= bool((derivative[on] >= -epsilon).all())
                    sides[1] |= bool((derivative[on] <= epsilon).all())
                    if all(sides):
                        break
                if sides[0] != sides[1]:
                    pieces.append([start + lo * delta, start + hi * delta])
        if not pieces:
            raise ValueError('Platform footprint has no exposed boundary')
        # Duplicate collinear pieces do not alter distance; exact dedup reduces work.
        segments = np.asarray(pieces)
        order = (segments[:, 0, 0] > segments[:, 1, 0]) | (
            (segments[:, 0, 0] == segments[:, 1, 0]) & (segments[:, 0, 1] > segments[:, 1, 1]))
        segments[order] = segments[order, ::-1]
        return np.unique(segments.reshape(-1, 4), axis=0).reshape(-1, 2, 2)

    def query(self, points: Tensor, tile: Tensor, target_index: Tensor) -> PlatformSurfaceSample:
        """Batch tile-local ``[B, ..., 3]`` points, with one tile/target per row.

        Target -1 is inactive. A different tile is not target membership. Point Z
        does not constrain a downward surface column, just as in AtlasTerrain.
        Numerical comparisons use the catalog's original float32 geometry scale;
        this is not a landing-height, contact-force or stability tolerance.
        """
        if (points.ndim < 2 or points.shape[-1] != 3 or tile.shape != points.shape[:1]
                or target_index.shape != tile.shape or not points.is_floating_point()):
            raise ValueError('Platform query requires XYZ points and one tile/target per row')
        if points.device != self.device or tile.device != self.device or target_index.device != self.device:
            raise ValueError('Platform points, routes and atlas must share a device')
        if not ((target_index == target_index.round()) & (target_index >= -1)
                & (target_index < len(self.target_ids))).all():
            raise ValueError('Platform target indices must be integer catalog routes or -1')
        sampled = self.terrain.height_at(points, tile)
        target = target_index.long().clamp_min(0)
        shape = (len(points), *([1] * (points.ndim - 2)))
        active = ((target_index >= 0) & (tile == self.tiles[target])).reshape(shape)
        finite = torch.isfinite(points).all(-1)
        xy = torch.where(torch.isfinite(points[..., :2]), points[..., :2], 0.)
        # Atlas visibility uses its float32 world-ray boundary. Membership and
        # edge distance use the actual local point; quantization must not grow
        # the target outside a source edge.
        origin = self.terrain.origins[tile.long()].reshape(*shape, 3)
        snapped = ((xy.detach().float() + origin[..., :2]) - origin[..., :2]).to(points.dtype)
        position = xy
        faces = self.triangles[target].reshape(*shape, -1, 3, 3)
        mask = self.face_mask[target].reshape(*shape, -1)
        base, e1, e2 = faces[..., 0, :], faces[..., 1, :] - faces[..., 0, :], faces[..., 2, :] - faces[..., 0, :]
        relative = position[..., None, :] - base[..., :2]
        denominator = e1[..., 0] * e2[..., 1] - e1[..., 1] * e2[..., 0]
        denominator = torch.where(mask, denominator, 1.)
        s = (relative[..., 0] * e2[..., 1] - relative[..., 1] * e2[..., 0]) / denominator
        t = (e1[..., 0] * relative[..., 1] - e1[..., 1] * relative[..., 0]) / denominator
        # Do not inherit the atlas query's geometric edge padding.
        covered = mask & (s >= 0) & (t >= 0) & (s + t <= 1)
        heights = base[..., 2] + s * e1[..., 2] + t * e2[..., 2]
        height = torch.where(covered, heights, -torch.inf).amax(-1)
        roi = self.rois[target].reshape(*shape, 2, 2)
        in_roi = ((position >= roi[..., 0, :]) & (position <= roi[..., 1, :])).all(-1)
        footprint = covered.any(-1) & in_roi & active & finite
        tolerance = self.height_tolerance[target].reshape(shape)
        slope_x = (e1[..., 2] * e2[..., 1] - e2[..., 2] * e1[..., 1]) / denominator
        slope_y = (e1[..., 0] * e2[..., 2] - e2[..., 0] * e1[..., 2]) / denominator
        rounding = snapped - xy.detach()
        comparison = heights.detach() + slope_x * rounding[..., 0, None] + slope_y * rounding[..., 1, None]
        comparison = torch.where(covered, comparison, -torch.inf).amax(-1)
        visible = footprint & sampled.valid & ((comparison - sampled.height).abs() <= tolerance)
        # Distance uses unsnapped local points: its continuous geometric gradient
        # is not the renderer/atlas's world-coordinate float32 quantization.
        edges = self.boundaries[target].reshape(*shape, -1, 2, 2)
        edge_mask = self.edge_mask[target].reshape(*shape, -1)
        start, delta = edges[..., 0, :], edges[..., 1, :] - edges[..., 0, :]
        squared_length = delta.square().sum(-1)
        projection = ((xy.to(points.dtype)[..., None, :] - start) * delta).sum(-1)
        ratio = (projection / torch.where(edge_mask & (squared_length > 0), squared_length, 1.)).clamp(0, 1)
        closest = start + ratio[..., None] * delta
        distance = torch.linalg.vector_norm(xy.to(points.dtype)[..., None, :] - closest, dim=-1)
        distance = torch.where(edge_mask, distance, torch.inf).amin(-1)
        distance = torch.where(footprint, distance, -distance)
        distance = torch.where(active & finite, distance, torch.nan).to(points.dtype)
        height = torch.where(footprint, height, torch.nan).to(points.dtype)
        return PlatformSurfaceSample(height, visible, distance, footprint, visible, sampled.valid)
