"""Bind whole-wheel clearance to actual native collider poses and asset bytes."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NativeWheelPose:
    centres_local_m: torch.Tensor
    bottoms_local_z_m: torch.Tensor
    radii_m: torch.Tensor
    terrain_tile: torch.Tensor


class NativePlatformClearance:
    """No-argument clearance capture for ``NativePlatformLanding``.

    The default query is AtlasCylinderTerrainGap; an explicit query factory is
    also accepted for isolated CPU tests. Only caller-selected rows are queried.
    Geometry comes from the selected URDF and actual native wheel body poses,
    never ideal closed-chain forward kinematics or a wheel-centre ground ray.

    This binder owns one immutable scene/asset epoch (zero). Episode resets are
    different identities within that scene, not model edits. SDK collision
    cooking equivalence is not certified by parsing the source cylinder.
    """

    def __init__(self, native, terrain, *, selected_capture, tile_ids=(101, 102), query_factory=None):
        from transformer_rl.formal.simulation.ground import WheelCollisionCentres

        self.native, self.terrain = native, terrain
        self._asset, self._binding = native._asset, native._binding
        self._router = native._terrain_course.router
        if (self._router.identity != terrain.manifest_sha256
                or not torch.equal(self._router.origins_table, terrain.origins)):
            raise ValueError('Native wheel clearance needs the same bound atlas and tile origins')
        if not callable(selected_capture):
            raise ValueError('Native clearance requires explicit selected platform rows')
        raw = self._asset.bound_bytes(self._asset.manifest['urdf'])
        self.wheels = WheelCollisionCentres.from_urdf(raw, self._binding)
        self.selected_capture = selected_capture
        device = terrain.origins.device
        self._indices = torch.tensor(self._binding.wheel_body_indices, device=device, dtype=torch.long)
        self._offsets = torch.tensor(self.wheels.local_xyz_m, device=device, dtype=torch.float64)
        self._axes = torch.tensor(self.wheels.local_axis_xyz, device=device, dtype=torch.float64)
        if query_factory is None:
            from .atlas_cylinder_gap import AtlasCylinderTerrainGap
            query_factory = AtlasCylinderTerrainGap
        self.query = query_factory(terrain, self.wheels.radius_m, self.wheels.length_m, tile_ids=tile_ids)
        self._radii = torch.tensor(self.wheels.radius_m, device=device, dtype=torch.float64)
        self._half_lengths = torch.tensor(self.wheels.length_m, device=device, dtype=torch.float64) / 2
        self._cache = None

    @torch.no_grad()
    def capture(self):
        from transformer_rl.formal.control.wheel_clearance import WheelClearanceSample
        from transformer_rl.formal.lab_env.terrain_support import TerrainSupportReference

        native, router, device = self.native, self._router, self.terrain.origins.device
        if (native._asset is not self._asset or native._binding is not self._binding
                or native._terrain_course.router is not router
                or router.identity != self.terrain.manifest_sha256):
            raise RuntimeError('Native clearance scene/asset binding changed; create a new binder')
        selected = self.selected_capture()
        n = native.num_envs
        if (not isinstance(selected, torch.Tensor) or selected.shape != (n,)
                or selected.dtype != torch.bool or selected.device != device):
            raise ValueError('Native clearance selection must be same-device boolean environment rows')
        body_pose = native._as_torch(native.robot.data.body_link_pose_w)
        if (body_pose.shape != (n, len(self._binding.native_body_names), 7)
                or body_pose.dtype != torch.float32 or body_pose.device != device):
            raise ValueError('Native clearance requires actual named body-link poses in world xyzw')
        pose = body_pose.index_select(1, self._indices)
        pose = torch.where(selected[:, None, None], pose, 0.)
        tile, episode, origins = router.tile, native._episode_id, router.origins
        if (tile.shape != (n,) or tile.dtype != torch.long or tile.device != device
                or episode.shape != (n,) or episode.dtype != torch.long or episode.device != device
                or origins.shape != (n, 3) or origins.device != device or origins.dtype != torch.float32):
            raise ValueError('Native clearance requires current int64 tile and episode identities')
        # On normal substeps the tick changes, so no cache-comparison host wait
        # occurs. Duplicate reads also account for same-tick resets/reassignment
        # and explicit pose writes, which do not necessarily advance physics.
        tick = native._physical_tick
        if self._cache is not None and self._cache['sample'].physical_tick == tick:
            values = {'tile': tile, 'episode': episode, 'selected': selected, 'pose': pose, 'origins': origins}
            if all(torch.equal(value, self._cache[name]) for name, value in values.items()):
                return self._cache['sample']
        relevant = pose[selected]
        norm = relevant[..., 3:].double().norm(dim=-1)
        valid_route = (tile >= 0) & (tile < len(self.terrain.origins))
        if (not torch.isfinite(relevant).all()
                or not torch.allclose(norm, torch.ones_like(norm), atol=1e-4, rtol=0)
                or (selected & ~valid_route).any()):
            raise ValueError('Selected native wheel poses must have finite normalized xyzw rotations')
        if not torch.equal(origins[selected], self.terrain.origins[tile[selected]]):
            raise ValueError('Selected native environment origins differ from their current atlas tiles')
        # Unselected rows may be outside this task's atlas routes, or invalid in
        # a different task. They remain explicitly unqueried, not fake planes.
        quaternion = pose[..., 3:].double()
        quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        rotate = TerrainSupportReference._rotate
        centre = pose[..., :3].double() + rotate(quaternion, self._offsets)
        axis = rotate(quaternion, self._axes)
        local = centre - origins[:, None].double()
        bottom = local[..., 2] - (self._radii * axis[..., :2].norm(dim=-1)
                                  + self._half_lengths * axis[..., 2].abs())
        cylinder = NativeWheelPose(local, bottom, self._radii, tile.clone())
        gap = self.query.at(local, axis, tile, selected)
        sample = WheelClearanceSample(
            gap, origins.clone(), native._clock().clone(), tick, 0,
            self.wheels.body_names, self.wheels.urdf_sha256,
            source_geometry_bound=True, native_collision_equivalence_verified=False)
        self._cache = {'sample': sample, 'tile': tile.clone(), 'episode': episode.clone(),
                       'selected': selected.clone(), 'pose': pose.clone(), 'origins': origins.clone(), 'cylinder': cylinder}
        return sample

    def capture_pose(self):
        """Reuse the actual cylinder pose captured by this tick's landing observer.

        No second SDK pose read or clearance query. A same-tick reset or route
        change must be followed by the observer's normal capture first.
        """
        cache = self._cache
        if (cache is None or cache['sample'].physical_tick != self.native._physical_tick
                or not torch.equal(cache['episode'], self.native._episode_id)
                or not torch.equal(cache['tile'], self._router.tile)
                or not torch.equal(cache['origins'], self._router.origins)):
            raise RuntimeError('Platform target reward needs the current observed cylinder pose')
        return cache['cylinder']
