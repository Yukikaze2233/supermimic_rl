"""Asset-derived leg lengths and same-column wheel clearance for jump rewards.

The native producer uses measured link poses. Imagination uses its predicted
kinematic tree. Both use the wheel-cylinder lower envelope and terrain at the
collision centre, not a whole-footprint collision-clearance qualification.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from ..reward_kernel import JumpFlightGeometry
from ..world_model_v2.geometry import RewardGeometry
from ..world_model_v2.leg_linkage import LegLinkage


class JumpRewardGeometry:
    """Cache two asset hip pivots and the assembled table's endpoint lengths."""

    def __init__(self, geometry: RewardGeometry):
        pivots, minimum, span = [], [], []
        digest = hashlib.sha256()
        layout = {'provenance': geometry.linkage.provenance, 'features': geometry.linkage.features,
                  'feature_sizes': geometry.linkage.feature_sizes}
        digest.update(json.dumps(layout, sort_keys=True, separators=(',', ':')).encode())
        for side in geometry.sides:
            table = geometry.linkage.sides[side.name]
            hip = side.hip_origin[:3, 3].double().cpu()
            if not torch.equal(hip, table.hip_pivot_base_m):
                raise ValueError('Jump leg-length pivot differs from the pinned linkage')
            endpoints = geometry.linkage.unpack(table.values[[0, -1]])['wheel_center_reference_base_m']
            lengths = (endpoints - hip).norm(dim=-1)
            low, high = lengths.amin(), lengths.amax()
            if not torch.isfinite(lengths).all() or not (0 < low < high):
                raise ValueError('Jump geometry requires a nonzero assembled leg-length range')
            pivots.append(hip)
            minimum.append(low)
            span.append(high - low)
            metadata = dict(name=side.name, primary=table.primary_joint, secondary=table.secondary_joint,
                            phi_min=table.phi_min, phi_max=table.phi_max,
                            passive_joints=table.passive_joint_names,
                            reference_primary_rad=table.reference_primary_rad,
                            hip_axis_base=table.hip_axis_base.tolist(), hip_pivot_base_m=hip.tolist())
            digest.update(json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode())
            for value in (table.values, table.slopes):
                digest.update(value.contiguous().cpu().numpy().tobytes())
        self.hip_pivot_base_m = torch.stack(pivots)
        self.minimum_leg_length_m = torch.stack(minimum)
        self.leg_length_range_m = torch.stack(span)
        self._buffers = {}
        self.receipt = {
            'format': 'jump-flight-geometry-v1',
            'asset_manifest_sha256': geometry.linkage.provenance['asset_manifest_sha256'],
            'linkage_data_sha256': digest.hexdigest(),
            'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'side_order': ['left', 'right'],
            'hip_pivot_base_m': self.hip_pivot_base_m.tolist(),
            'minimum_leg_length_m': self.minimum_leg_length_m.tolist(),
            'leg_length_range_m': self.leg_length_range_m.tolist(),
            'length_definition': 'norm(wheel_link_origin_in_base - primary_hip_pivot_in_base)',
            'length_bounds': 'min/max of the two assembled linkage-table endpoints; no extra mechanical margin',
            'clearance_definition': 'wheel_cylinder_bottom_world_z minus terrain column at collision centre',
            'scope': 'reward shaping only; not full-cylinder collision clearance or skill qualification',
        }

    @classmethod
    def load(cls, asset_directory, linkage_path=None, *, expected_linkage_sha256=None):
        asset = Path(asset_directory)
        asset_sha = hashlib.sha256((asset / 'manifest.json').read_bytes()).hexdigest()
        if linkage_path is None:
            linkage_path = Path(__file__).resolve().parents[3] / 'configs' / 'linkage' / f'leg_linkage_{asset_sha[:8]}.json'
        linkage_path = Path(linkage_path)
        if (expected_linkage_sha256 is not None
                and hashlib.sha256(linkage_path.read_bytes()).hexdigest() != expected_linkage_sha256):
            raise ValueError('Jump reward linkage file differs from its explicit pin')
        linkage = LegLinkage.load(linkage_path, expected_asset_manifest_sha256=asset_sha)
        return cls(RewardGeometry(linkage, asset))

    def sample(self, wheel_link_position_base_m, wheel_bottom_world_z_m, wheel_ground_world_z_m, valid):
        """Assemble finite differentiable quantities without treating invalid rows as evidence."""
        n = len(valid)
        if (wheel_link_position_base_m.shape != (n, 2, 3)
                or wheel_bottom_world_z_m.shape != (n, 2) or wheel_ground_world_z_m.shape != (n, 2)
                or valid.shape != (n,) or valid.dtype != torch.bool):
            raise ValueError('Jump flight geometry requires two ordered wheels per row')
        key = (wheel_link_position_base_m.device, wheel_link_position_base_m.dtype)
        if key not in self._buffers:
            self._buffers[key] = tuple(value.to(device=key[0], dtype=key[1]) for value in (
                self.hip_pivot_base_m, self.minimum_leg_length_m, self.leg_length_range_m))
        hip, minimum, span = self._buffers[key]
        valid = (valid & torch.isfinite(wheel_link_position_base_m).all(dim=(-2, -1))
                 & torch.isfinite(wheel_bottom_world_z_m).all(-1)
                 & torch.isfinite(wheel_ground_world_z_m).all(-1))
        # Sanitize before arithmetic, so an unused NaN terrain row cannot
        # poison a neighboring valid row's BPTT gradient through torch.where.
        link = torch.where(valid[:, None, None], wheel_link_position_base_m, hip)
        bottom = torch.where(valid[:, None], wheel_bottom_world_z_m, 0.)
        ground = torch.where(valid[:, None], wheel_ground_world_z_m, 0.)
        return JumpFlightGeometry(bottom - ground, (link - hip).norm(dim=-1),
                                  minimum.expand(n, -1), span.expand(n, -1), valid)


class NativeJumpRewardGeometry:
    """Read the support observer and terrain producer from this same native tick."""

    def __init__(self, geometry: JumpRewardGeometry, course):
        self.geometry, self.course = geometry, course
        self.receipt = geometry.receipt

    def __call__(self, sample):
        support = self.course._support_sample
        if support is None or sample.wheel_position_base_m is None:
            raise RuntimeError('Native jump reward needs the current wheel pose/support observation')
        return self.geometry.sample(sample.wheel_position_base_m, support.wheel_bottom_world_z_m,
                                    self.course.wheel_ground, support.valid & ~self.course.query_invalid)
