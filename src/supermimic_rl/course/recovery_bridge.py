"""Opt-in, geometry-placed recovery births; no actions or success decisions."""
from __future__ import annotations

from dataclasses import fields
import json
import math
from pathlib import Path

import torch

from transformer_rl.formal.tasks.recovery import RECOVERY_NAMES


BRIDGE_VARIANTS = tuple(f'{side}_side_{angle}' for angle in (45, 60, 75)
                        for side in ('left', 'right'))
RECOVERY_VARIANTS = (*RECOVERY_NAMES, *BRIDGE_VARIANTS)
SIMPLE_POSTURES = (0, 1, 2, 3, 6, 7)
BRIDGE_ROLL_RAD = tuple(math.radians(angle) * sign
                        for angle in (45, 60, 75) for sign in (-1, 1))
MIDPOINT_VARIANTS = ('left_side_82p5', 'right_side_82p5',
                     'pitch_forward_67p5', 'pitch_backward_67p5')
MIDPOINT_BASE_POSTURES = (4, 5, 8, 9)
# Midpoints of the existing evaluated passing/failing birth angles. These are
# curriculum poses, not new mechanical limits or qualified recovery skills.
MIDPOINT_ROLL_PITCH_RAD = (
    (-math.radians((75 + 90) / 2), 0.), (math.radians((75 + 90) / 2), 0.),
    (0., math.radians((45 + 90) / 2)), (0., -math.radians((45 + 90) / 2)),
)


def variant_names(*, midpoints=False):
    return (*RECOVERY_VARIANTS, *MIDPOINT_VARIANTS) if midpoints else RECOVERY_VARIANTS


def select_variants(ordinal, posture, pool, *, midpoints=False):
    """Replace selected easy slots only once the full 16-pose pool is open.

    The default two-cycle schedule is unchanged. Opt-in midpoint sampling uses
    three cycles, retaining all difficult slots and substituting four easy
    slots in the third cycle. The recovery-family ordinal is the only clock.
    """
    variant = posture.clone()
    if set(pool) != set(RECOVERY_NAMES):
        return posture, variant
    cycle = (ordinal // len(pool)) % (3 if midpoints else 2)
    alternate = cycle == 1
    result = posture.clone()
    for index, source in enumerate(SIMPLE_POSTURES):
        selected = alternate & (posture == source)
        variant[selected] = len(RECOVERY_NAMES) + index
        result[selected] = 4 + index % 2
    if midpoints:
        for index, source in enumerate(SIMPLE_POSTURES[:4]):
            selected = (cycle == 2) & (posture == source)
            variant[selected] = len(RECOVERY_VARIANTS) + index
            result[selected] = MIDPOINT_BASE_POSTURES[index]
    return result, variant


def receipt(*, midpoints=False):
    result = {
        'contract': 'recovery_roll_bridge_v1',
        'variants': list(RECOVERY_VARIANTS),
        'bridge_requested_roll_deg': [math.degrees(v) for v in BRIDGE_ROLL_RAD],
        'selection': 'alternate full-16 recovery cycles; easy slots 0,1,2,3,6,7 only',
        'cycle_births': {'bridge': 6, 'easy': 6, 'original_difficult': 20},
        'sampling_clock': 'persisted recovery-family birth ordinal; no additional RNG',
        'resume': 'curriculum/WM preserve ordinal; current PPO cold resume does not restore native sampler',
        'placement': 'native RecoveryPoseCompiler and AssetCollisionGeometry.place',
        'cache': 'six zero-yaw local flat poses; original 90-degree endpoints unchanged',
        'evaluation': 'fixed scenario evaluation uses original 16 poses only',
        'labels': 'recovery_posture ledger uses variants; native base IDs are not outcome labels',
        'scope': 'birth distribution only; actor, reward, lifecycle and actions unchanged',
    }
    if midpoints:
        result.update({
            'contract': 'recovery_midpoint_bridge_v1',
            'variants': list(variant_names(midpoints=True)),
            'midpoint_requested_roll_pitch_deg': [
                [math.degrees(roll), math.degrees(pitch)]
                for roll, pitch in MIDPOINT_ROLL_PITCH_RAD],
            'midpoint_native_base_posture_ids': list(MIDPOINT_BASE_POSTURES),
            'angle_source': 'midpoints of existing side 75/90 and pitch 45/90 birth endpoints',
            'selection': ('three full-16 recovery cycles: native; easy slots 0,1,2,3,6,7 '
                          'become old bridges; easy slots 0,1,2,3 become midpoints'),
            'cycle_births': {'bridge': 6, 'midpoint': 4, 'easy': 8, 'original_difficult': 30},
            'distribution_change': ('original difficult share stays 62.5%; old bridges '
                                    '18.75% to 12.5%, easy 18.75% to 16.6667%, midpoints 8.3333%'),
            'cache': 'ten zero-yaw local flat poses; original sixteen endpoints unchanged',
        })
    return result


class RecoveryRollBridge:
    """Reuse the native pose contract, caching each new local placement once."""

    def __init__(self, native, curriculum, log_directory, *, evaluation=False):
        self.native, self.curriculum = native, curriculum
        self.reset = native._recovery_reset
        self.original_pose = self.reset.pose
        self.evaluation = evaluation
        self.midpoints = curriculum.config.recovery_midpoint_bridge
        self.variants = variant_names(midpoints=self.midpoints)
        self.base_postures = tuple(4 + i % 2 for i in range(len(BRIDGE_VARIANTS)))
        if self.midpoints:
            self.base_postures += MIDPOINT_BASE_POSTURES
        # A fixed evaluation may explicitly name bridge poses. The native
        # reset still receives its original side/front/back 90-degree base IDs.
        self.fixed_variants = None
        self.trace_path = Path(log_directory) / 'recovery-births.jsonl'
        self.pending = {}
        self.cache = None
        if not evaluation:
            # These calls are startup-only. Native num_envs may be one in
            # CPU tests; each placement then uses its valid local row zero.
            parts = [self._compile(torch.tensor([0]), torch.tensor([base]),
                                   torch.zeros(1, dtype=torch.float64),
                                   torch.zeros(1, 3, dtype=torch.float64),
                                   torch.tensor([i])) for i, base in enumerate(self.base_postures)]
            self.cache = type(parts[0])(*(torch.cat([getattr(p, f.name) for p in parts])
                                          for f in fields(parts[0])))

    def _compile(self, ids, posture, yaw, origins, variant):
        from transformer_rl.formal.control.actuation import OutputFeedbackController
        from transformer_rl.formal.lab_env.recovery import RecoveryResetPose

        reset, n = self.reset, len(ids)
        controller = OutputFeedbackController(reset.kinematics.binding.nominal_policy_position_rad,
                                              n, pair_domain=None, device='cpu')
        offset = torch.zeros(n, 2, dtype=torch.float64)
        original = variant < len(BRIDGE_VARIANTS)
        # Keep the original six variants on their unchanged arithmetic path.
        requested = torch.tensor(BRIDGE_ROLL_RAD, dtype=torch.float64)[variant[original]]
        baseline = torch.tensor((-math.pi / 2, math.pi / 2), dtype=torch.float64)[posture[original] - 4]
        offset[original, 0] = requested - baseline
        if (~original).any():
            index = variant[~original] - len(BRIDGE_VARIANTS)
            requested = torch.tensor(MIDPOINT_ROLL_PITCH_RAD, dtype=torch.float64)[index]
            baseline = torch.tensor(((-math.pi / 2, 0.), (math.pi / 2, 0.),
                                     (0., math.pi / 2), (0., -math.pi / 2)), dtype=torch.float64)[index]
            offset[~original] = requested - baseline
        candidate = reset.compiler.compile(posture, yaw, controller, roll_pitch_offset_rad=offset)
        placement = reset.geometry.place(
            candidate.closure.tree_position, candidate.requested_root_orientation_xyzw,
            origins[:, :2], reset.flat_patch(ids, origins),
            clearance_m=reset.config.clearance_m + candidate.release_clearance_m,
            loop_tolerance_m=reset.config.loop_tolerance_m,
            wheel_support_tolerance_m=reset.config.wheel_support_tolerance_m)
        if not placement.geometry_ready.all():
            raise ValueError('Side-roll birth failed native geometric placement')
        native_order = [reset.kinematics.binding.tree_joint_names.index(name)
                        for name in reset.kinematics.binding.native_joint_names]
        joints = placement.tree_position[:, native_order].float().to(reset.device)
        if candidate.angular_velocity_body_rad_s.any():
            raise ValueError('Side-roll compiler must retain zero angular velocity')
        velocity = torch.zeros(n, 6, dtype=torch.float32, device=reset.device)
        rotation = reset.geometry._rotation(placement.root_pose_xyzw[:, 3:])
        velocity[:, :3] = (rotation @ candidate.origin_velocity_body_m_s[..., None]).squeeze(-1).to(velocity)
        return RecoveryResetPose(placement.root_pose_xyzw.float().to(reset.device), joints,
                                 velocity, torch.zeros_like(joints), posture.to(reset.device),
                                 placement.geometry_ready.to(reset.device))

    @staticmethod
    def roll(quaternion):
        x, y, z, w = quaternion.unbind(-1)
        return torch.atan2(2 * (w * x + y * z), 1 - 2 * (x.square() + y.square()))

    @staticmethod
    def pitch(quaternion):
        x, y, z, w = quaternion.unbind(-1)
        return torch.asin((2 * (w * y - z * x)).clamp(-1, 1))

    @torch.no_grad()
    def pose(self, ids, posture_id, yaw_rad, origins_world_m):
        ids = torch.as_tensor(ids, dtype=torch.long, device='cpu')
        posture = torch.as_tensor(posture_id, dtype=torch.long, device='cpu')
        yaw = torch.as_tensor(yaw_rad, dtype=torch.float64, device='cpu')
        origins = torch.as_tensor(origins_world_m, dtype=torch.float64, device='cpu')
        n = ids.numel()
        if (ids.ndim != 1 or not n or posture.shape != (n,) or yaw.shape != (n,)
                or origins.shape != (n, 3) or ((posture < 0) | (posture >= 16)).any()
                or ids.unique().numel() != n or (ids < 0).any() or (ids >= self.reset.num_envs).any()
                or not torch.isfinite(yaw).all() or not torch.isfinite(origins).all()):
            raise ValueError('Recovery birth IDs, postures, yaw and origins must match native layout')
        if self.evaluation and self.fixed_variants is not None:
            variant = self.fixed_variants[ids].clone()
        else:
            variant = (posture.clone() if self.evaluation else
                       self.curriculum.recovery_birth_variants(ids.to(self.curriculum.device)).cpu())
        if ((variant < 0) | (variant >= len(self.variants))).any():
            raise ValueError('Unknown fixed recovery birth variant')
        bridge = variant >= len(RECOVERY_NAMES)
        selected = bridge.nonzero().flatten()
        ordinary = (~bridge).nonzero().flatten()
        parts = []
        if len(ordinary):
            parts.append((ordinary, self.original_pose(ids[ordinary], posture[ordinary],
                                                      yaw[ordinary], origins[ordinary])))
        if len(selected):
            index = variant[selected] - len(RECOVERY_NAMES)
            if not torch.equal(posture[selected], torch.tensor(self.base_postures)[index]):
                raise ValueError('Side-roll birth variant and native posture differ')
            if (self.cache is not None and yaw[selected].eq(0).all()
                    and origins[selected, 2].eq(0).all()):
                index = index.to(self.reset.device)
                value = type(self.cache)(*(getattr(self.cache, f.name).index_select(0, index)
                                          for f in fields(self.cache)))
                value.root_pose_xyzw[:, :2] = origins[selected, :2].to(value.root_pose_xyzw)
            else:
                value = self._compile(ids[selected], posture[selected], yaw[selected], origins[selected], index)
            parts.append((selected, value))
        template = parts[0][1]
        values = {f.name: getattr(template, f.name).new_empty(len(ids), *getattr(template, f.name).shape[1:])
                  for f in fields(template)}
        for rows, part in parts:
            for name, value in values.items():
                value[rows.to(value.device)] = getattr(part, name)
        result = type(template)(**values)
        placed_roll = self.roll(result.root_pose_xyzw[:, 3:]).cpu().tolist()
        placed_z = result.root_pose_xyzw[:, 2].cpu().tolist()
        native_roll = {4: -math.pi / 2, 5: math.pi / 2, 15: math.pi}
        for row, pid, label, angle, z, origin_z in zip(
                ids.tolist(), posture.tolist(), variant.tolist(), placed_roll, placed_z, origins[:, 2].tolist()):
            self.pending[row] = {'env_id': row, 'native_base_posture_id': pid,
                                 'variant_id': label, 'variant': self.variants[label],
                                 'requested_roll_rad': (BRIDGE_ROLL_RAD[label - 16] if 16 <= label < 22
                                                        else native_roll.get(pid, 0.)),
                                 'placed_roll_rad': angle, 'placed_root_world_z_m': z,
                                 'origin_world_z_m': origin_z}
        if self.midpoints:
            placed_pitch = self.pitch(result.root_pose_xyzw[:, 3:]).cpu().tolist()
            native_pitch = {6: math.pi / 4, 7: -math.pi / 4, 8: math.pi / 2,
                            9: -math.pi / 2, 10: math.pi / 2, 11: math.pi / 2,
                            12: -math.pi / 2, 13: -math.pi / 2}
            for row, pid, label, angle in zip(ids.tolist(), posture.tolist(), variant.tolist(), placed_pitch):
                record = self.pending[row]
                record['requested_pitch_rad'] = native_pitch.get(pid, 0.)
                record['placed_pitch_rad'] = angle
                if label >= len(RECOVERY_VARIANTS):
                    roll, pitch = MIDPOINT_ROLL_PITCH_RAD[label - len(RECOVERY_VARIANTS)]
                    record.update(requested_roll_rad=roll, requested_pitch_rad=pitch)
        return result

    def record_births(self, episode_ids, timestamp, root_pose, base_com_world_z):
        """Bind the first post-reset readback to actual adapter episode IDs."""
        if not self.pending:
            return
        ids = list(self.pending)
        angles = self.roll(root_pose[ids, 3:]).detach().cpu().tolist()
        root_z = root_pose[ids, 2].detach().cpu().tolist()
        com_z = base_com_world_z[ids].detach().cpu().tolist()
        episodes, clocks = episode_ids[ids].cpu().tolist(), timestamp[ids].cpu().tolist()
        pitch = (dict(zip(ids, self.pitch(root_pose[ids, 3:]).detach().cpu().tolist()))
                 if self.midpoints else None)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open('a') as stream:
            for row, episode, clock, angle, z, com in zip(ids, episodes, clocks, angles, root_z, com_z):
                record = {**self.pending[row], 'episode_id': episode,
                          'first_snapshot_time_s': clock, 'actual_birth_roll_rad': angle,
                          'actual_root_world_z_m': z, 'actual_base_com_world_z_m': com}
                if pitch is not None:
                    record['actual_birth_pitch_rad'] = pitch[row]
                stream.write(json.dumps(record, allow_nan=False) + '\n')
        self.pending.clear()


def install_recovery_roll_bridge(native, curriculum, log_directory, *, evaluation=False):
    if not curriculum.config.recovery_roll_bridge:
        return None
    bridge = RecoveryRollBridge(native, curriculum, log_directory, evaluation=evaluation)
    native._recovery_reset.pose = bridge.pose
    native.supermimic_recovery_bridge = bridge
    return receipt(midpoints=curriculum.config.recovery_midpoint_bridge)
