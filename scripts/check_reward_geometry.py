#!/usr/bin/env python3
"""Measure ideal closed-chain reward geometry against native substep truth.

The LUT's interpolation accuracy is not a bound on solver constraint error.
Report both; do not silently replace the latter with the former in imagination.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from supermimic_rl.world_model_v2 import so3
from supermimic_rl.world_model_v2.geometry import RewardGeometry
from supermimic_rl.world_model_v2.leg_linkage import LegLinkage
from supermimic_rl.world_model_v2.physical import PhysicalState, POSITION_JOINTS, VELOCITY_JOINTS
from supermimic_rl.world_model_v2.substep_view import SubstepView


def statistics(value):
    value = value.abs().flatten()
    return None if not value.numel() else {
        'samples': value.numel(), 'p50': float(value.quantile(.5)),
        'p99': float(value.quantile(.99)), 'max': float(value.max()),
        'rms': float(value.square().mean().sqrt())}


@torch.no_grad()
def check(replay, table, asset):
    view = SubstepView.from_replay_payload(torch.load(replay, map_location='cpu', weights_only=True, mmap=True))
    linkage = LegLinkage.load(table)
    geometry = RewardGeometry(linkage, asset)
    source = json.loads(view.schema.source)
    if source['asset_manifest_sha256'] != linkage.provenance['asset_manifest_sha256']:
        raise ValueError('Recorded native asset differs from the linkage table')
    get = lambda name: view.get(name)[view.valid].flatten(0, 1)
    q, dq = get('joint_position_continuous'), get('joint_velocity_native')
    rotation = so3.quaternion_to_matrix(get('root_link_orientation_xyzw'))
    state = PhysicalState(get('root_link_position_world_m'), rotation,
        (rotation @ get('root_origin_velocity_body_m_s')[..., None]).squeeze(-1),
        get('root_angular_velocity_world_rad_s'), q[:, view.joints(POSITION_JOINTS)],
        dq[:, view.joints(VELOCITY_JOINTS)])
    predicted = geometry(state)
    knee_columns = view.joints(tuple(linkage.sides[side.name].passive_joint_names[side.passive_knee]
                                    for side in geometry.sides))
    measured_tree = geometry(state, passive_knee_position=q[:, knee_columns],
                             passive_knee_velocity=dq[:, knee_columns])
    fresh = (~view.follows_previous())[view.valid, None].expand(-1, view.substeps).flatten()
    groups = {'all': torch.ones_like(fresh), 'after_first_interval': ~fresh,
              'both_wheels_supported': (get('wheel_support_load_n') > 5).all(-1)}
    has_poses = view.has('body_link_pose_world_xyzw')
    poses = get('body_link_pose_world_xyzw') if has_poses else None
    report = {'replay': str(replay), 'asset_manifest_sha256': source['asset_manifest_sha256'],
              'backend': source.get('backend'), 'physical_samples': len(q),
              'body_pose_available': has_poses, 'sides': {}}
    for index, side in enumerate(geometry.sides):
        lut = linkage.sides[side.name]
        primary, secondary = view.joints((lut.primary_joint, lut.secondary_joint))
        knee = view.joints((lut.passive_joint_names[side.passive_knee],))[0]
        phi = q[:, secondary] - q[:, primary]
        features, valid = linkage.evaluate(side.name, phi)
        derivative = linkage.unpack(lut.derivative(phi))
        errors = {'passive_knee_position_rad': features['passive_joint'][:, side.passive_knee] - q[:, knee],
                  'passive_knee_velocity_rad_s': derivative['passive_joint'][:, side.passive_knee]
                  * (dq[:, secondary] - dq[:, primary]) - dq[:, knee]}
        if has_poses:
            wheel = view.bodies(('L_link3' if side.name == 'left' else 'R_link3',))[0]
            link_world = poses[:, wheel, :3]
            link_base = (rotation.transpose(-1, -2) @ (link_world - state.position)[..., None]).squeeze(-1)
            wheel_rotation = so3.quaternion_to_matrix(poses[:, wheel, 3:])
            centre = link_world + (wheel_rotation @ side.collision_offset[:, None]).squeeze(-1)
            axis = (wheel_rotation @ side.collision_axis[:, None]).squeeze(-1)
            bottom = centre[:, 2] - side.radius * axis[:, :2].norm(dim=-1) - side.half_length * axis[:, 2].abs()
            errors.update(wheel_link_position_m=(predicted.link_position_base[:, index] - link_base).norm(dim=-1),
                          wheel_centre_position_m=(predicted.centre_world[:, index] - centre).norm(dim=-1),
                          wheel_bottom_z_m=predicted.bottom_world_z[:, index] - bottom,
                          measured_knee_tree_wheel_position_m=(measured_tree.link_position_base[:, index]
                                                                - link_base).norm(dim=-1),
                          measured_knee_tree_wheel_bottom_z_m=measured_tree.bottom_world_z[:, index] - bottom)
        report['sides'][side.name] = {
            'outside_domain_fraction': float((~valid).double().mean()),
            'in_domain': {group: {name: statistics(value[valid & mask]) for name, value in errors.items()}
                          for group, mask in groups.items()}}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('replay', type=Path)
    parser.add_argument('--table', type=Path, default=Path('configs/linkage/leg_linkage_e726d23c.json'))
    parser.add_argument('--asset', type=Path, default=Path('assets/full_vehicle_v2p0p1_source_audited_v2'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = check(args.replay, args.table, args.asset)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
