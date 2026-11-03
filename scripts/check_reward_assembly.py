#!/usr/bin/env python3
"""Native replay through the entire four-tick analytic locomotion reward assembly.

Run both ideal-LUT and measured-knee geometry to separate physics closure error
from reward/control implementation error. With --terrain-atlas, query the real
root/wheel terrain at every recorded tick and audit the native termination height.
This evaluates recorded intervals, not future skill-request propagation.
With --recorded-jumps, include recorded jump requests and reconstruct their
references from task memory; this is not an independent four-reference audit.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from transformer_rl.formal.lab_env.terrain_support import TerrainSupportConfig
from transformer_rl.formal.tasks.reward import RewardPhase, RewardReference, RewardTuning
from supermimic_rl.compact_state import named_field
from supermimic_rl.reward_kernel import RewardParameters
from supermimic_rl.world_model_v2 import so3
from supermimic_rl.world_model_v2.execution import LegExecution, WheelExecution
from supermimic_rl.world_model_v2.geometry import RewardGeometry
from supermimic_rl.world_model_v2.layout import SnapshotLayout
from supermimic_rl.world_model_v2.leg_linkage import LegLinkage
from supermimic_rl.world_model_v2.physical import PhysicalState, POSITION_JOINTS, VELOCITY_JOINTS
from supermimic_rl.world_model_v2.records import CONTACT_BODIES, WHEEL_BODIES
from supermimic_rl.world_model_v2.reward_assembly import RewardAssembly, RewardMemory, RewardTick
from supermimic_rl.world_model_v2.substep_view import SubstepView
from supermimic_rl.world_model_v2.support import SupportState
from supermimic_rl.world_model_v2.imagined_reward import OrdinaryFlatReward, TerrainLocomotionReward
from supermimic_rl.world_model_v2.terrain import AtlasTerrain
from supermimic_rl.world_model_v2.ingest import load_recorded_replay
from supermimic_rl.world_model_v2.model import Ticks
from supermimic_rl.world_model_v2.records import decode_memory, decode_physical
from supermimic_rl.formal_observation import known_horizon
from supermimic_rl.world_model_v2.jump_task import (
    JUMP_RECORD_FIELDS, JUMP_TRIGGER_RECORD_FIELDS, JumpTask, JumpTaskConfig, JumpTaskState,
)
from supermimic_rl.world_model_v2.motion_task import MotionTask


def stats(value):
    x = value.abs().double().flatten()
    return {'max': float(x.max()), 'p50': float(x.quantile(.5)), 'p99': float(x.quantile(.99)),
            'rms': float(x.square().mean().sqrt())}


@torch.no_grad()
def check_compact(path, table, asset, bundle, *, terrain_path=None, device='cpu', recorded_jumps=False):
    """The exact producer called by BPTT, with native labels replacing network outputs."""
    replay, schema = load_recorded_replay(path, device, record_terrain=terrain_path is not None)
    data = replay.data
    jump_task = None
    if recorded_jumps:
        if terrain_path is None:
            raise ValueError('Recorded-jump reward audit requires the pinned terrain atlas')
        payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if payload.get('jump_task_config') is None:
            raise ValueError('Recorded-jump reward audit requires the captured native jump task configuration')
        jump_task = JumpTask(JumpTaskConfig.from_dict(payload['jump_task_config']))
        required = {*JUMP_RECORD_FIELDS, 'jump_record_valid', 'motion_task', 'task_record_valid',
                    'failure_conditions_valid'}
        if jump_task.config.trigger is not None:
            required.update(JUMP_TRIGGER_RECORD_FIELDS)
        if not required.issubset(data):
            raise ValueError('Recorded-jump reward audit lacks native fields: ' + repr(sorted(required - data.keys())))
    rows = (data['valid'] & data['policy_owned'] & data['end_policy_owned']
            & (data['reward_context_valid'][..., 0] > .5))
    if terrain_path is None:
        rows &= ((data['reward_route'] == 0).all(-1) & (data['horizon'][..., 1] == 0)
                 & (known_horizon(data['horizon'][..., 0], data['horizon'][..., 1], data['goal'], schema.policy_dt_s) > 0))
    else:
        if 'task_family' not in data:
            raise ValueError('Terrain reward audit needs recorded task families; re-encode packed data or recollect')
        if recorded_jumps:
            mode = data['reward_route'][..., 0]
            rows &= (((mode <= 2) | (mode == 4)) & (data['task_family'][..., 0] < 5)
                     & (data['goal'][..., 7] <= .5)
                     & (MotionTask.decode(data['motion_task']).route[..., 3] <= 2))
            for name in ('jump_record_valid', 'task_record_valid', 'failure_conditions_valid'):
                rows &= data[name][..., 0] > .5
            if jump_task.config.trigger is not None:
                rows &= data['jump_trigger_flags'][..., 0]
        else:
            rows &= ((data['reward_route'][..., 0] <= 2) & (data['task_family'][..., 0] < 4)
                     & (data['goal'][..., 7:9] <= .5).all(-1))
    if not rows.any():
        raise ValueError('No complete eligible locomotion intervals for compact reward audit')
    start = {name: value[rows] for name, value in data.items()}
    state = decode_physical(start['physical'])
    ticks = Ticks.decode(start['ticks'], state)
    source = json.loads(schema.source)
    geometry = RewardGeometry(LegLinkage.load(table), asset)
    wheels = WheelExecution.from_bundle(bundle, schema.policy_dt_s / 4)
    terrain = (None if terrain_path is None else AtlasTerrain.load(
        terrain_path, expected_sha256=source['terrain_sha256'], device=device))
    producer = (OrdinaryFlatReward(source, geometry, wheels, schema.policy_dt_s) if terrain is None else
                TerrainLocomotionReward(source, geometry, wheels, schema.policy_dt_s, terrain=terrain,
                                        jump_task=jump_task))
    context = producer.bind(start, state, known_interval=terrain is not None)
    memory = RewardMemory.decode(torch.cat((start['reward_memory'], start['support_memory']), -1))
    tick_valid = (None if terrain is None else terrain.observe(ticks.state, start['terrain_tile'][:, 0]).valid)
    jump = None if jump_task is None else JumpTaskState.decode(start)
    jump_arguments = ({} if jump is None else {'jump_state': jump,
        'base_mode': MotionTask.decode(start['motion_task']).route[:, 3].long()})
    result = context.integrate(state, SimpleNamespace(ticks=ticks, tick_terrain_valid=tick_valid), start['action'], memory,
                               decode_memory(start['memory']), start['goal'], 0, **jump_arguments)
    success = (torch.zeros_like(start['terminated']) if result.success_credited is None else result.success_credited)
    reward = result.finish(start['terminated'], success, context.assembly.parameters)
    report = {'intervals': len(reward), 'reward_error': stats(reward - start['reward'][:, 0]),
            'invalid_intervals': int(result.invalid.sum()),
            'scope': 'actual BPTT reward producer, actual compact tick labels, no learned motion'}
    if terrain is not None:
        report.update(atlas_sha256=terrain.manifest_sha256, device=str(terrain.query.device),
            intervals_by_terrain_kind=torch.bincount(start['reward_route'][:, 1].long(), minlength=8).tolist(),
            intervals_by_task_family=torch.bincount(start['task_family'][:, 0].long(), minlength=6).tolist(),
            scope='recorded locomotion intervals through actual atlas/reward/termination-height producer; no future task or policy qualification')
        payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if 'storage' in payload:
            view = SubstepView.from_replay_payload(payload, dtype=torch.float32)
            actual = view.get('termination_height_m')[rows.cpu()].to(device)
            report['termination_height_error_m'] = stats(result.termination_height - actual)
    if jump is not None:
        from transformer_rl.formal.tasks.jump import ReferenceStage

        active = start['reward_route'][:, 0] == 4
        jump_rows = active | (start['task_family'][:, 0] == 4) | (jump.active_id >= 0) | (jump.physical_id >= 0)
        hold_missed = (active & (jump_task.reference(jump).clock_stage == int(ReferenceStage.HOLD))
                       & (result.jump_state.ids[:, 8] == int(RewardPhase.TAKEOFF)))
        report['subsets'] = {}
        for name, mask in (('ordinary', ~jump_rows), ('jump', jump_rows), ('active_jump', active),
                           ('hold_without_takeoff', hold_missed)):
            report['subsets'][name] = {'intervals': int(mask.sum()),
                'reward_error': stats((reward - start['reward'][:, 0])[mask]) if mask.any() else None,
                'invalid_intervals': int((result.invalid & mask).sum()),
                'success_events': int((success & ~start['terminated'] & mask).sum())}
        report.update(recorded_jumps=True,
            scope='recorded ordinary/jump intervals through the actual four-tick reward producer; true physical ticks and wheel normal support loads, no learned motion or policy qualification',
            reference_scope='jump references reconstructed from recorded task memory; per-tick native references are not recorded, so this is not an independent four-reference comparison',
            subset_scope='jump includes request-family waiting and active/physical attempts; HOLD without takeoff means active JUMP mode, reference HOLD and physical TAKEOFF still present after the recorded interval')
    return report


@torch.no_grad()
def check(path, table, asset, bundle):
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    view = SubstepView.from_replay_payload(payload, dtype=torch.float32)
    layout, data, schema = SnapshotLayout(view.schema), view.storage, view.schema
    source = json.loads(schema.source)
    config = source['reward_config']
    field = lambda x, name, **kw: named_field(x, schema, name, **kw)
    rows = (view.valid & data['policy_owned'] & data['end_policy_owned']
            & (view.get('command_mode') == 0).all((-2, -1))
            & ~field(data['start_state'], 'course.gimbal.special').bool()
            & (field(data['start_state'], 'course.tasks.terrain_kind') == 0))
    if not rows.any():
        raise ValueError('No policy-owned ordinary flat intervals in the recording')
    state0, context = data['start_state'][rows], data['start_context'][rows]
    get = lambda name: view.get(name)[rows]
    current = lambda name: field(state0, name)
    ctx = lambda name: field(context, name, context=True)
    support_columns = []
    for f in fields(SupportState):
        value = current('course.support._' + f.name)
        support_columns.append(value if value.ndim == 2 else value[:, None])
    support = SupportState.decode(torch.cat(support_columns, -1))
    memory = RewardMemory(current('reward._previous'), current('reward._previous2'),
                          current('reward._action_count'), support)
    linkage = LegLinkage.load(table, expected_asset_manifest_sha256=source['asset_manifest_sha256'])
    geometry = RewardGeometry(linkage, asset)
    wheels = WheelExecution.from_bundle(bundle, config['physics_dt_s'])
    tuning = RewardTuning(config['robot_mass_kg'], gamma=config['gamma'], policy_dt_s=schema.policy_dt_s,
                          profile=config['profile'])
    wheel_indices = [CONTACT_BODIES.index(name) for name in WHEEL_BODIES]
    parameters = RewardParameters(tuning, config['physics_dt_s'], ctx('body_mass_kg').sum(-1), wheel_indices,
                                 [i for i in range(len(CONTACT_BODIES)) if i not in wheel_indices], 0)
    assembly = RewardAssembly(parameters, tuple(config['available_terms']), geometry.knee_limits,
                              TerrainSupportConfig(**source['terrain_support_config']), wheels)
    legs = LegExecution(*(ctx('controller.' + name) for name in ('nominal', 'stiffness', 'damping', 'effort_bounds')))
    start = layout.physical(state0)
    # Reward/world-Z references below use the same global coordinates as the native samples.
    start = PhysicalState(start.position + ctx('environment_origin_world_m'), start.rotation, start.velocity,
                           start.angular_velocity, start.joint_position, start.joint_velocity)
    q, dq = get('joint_position_continuous'), get('joint_velocity_native')
    rotation = so3.quaternion_to_matrix(get('root_link_orientation_xyzw'))
    motion = PhysicalState(get('root_link_position_world_m'), rotation,
        (rotation @ get('root_origin_velocity_body_m_s')[..., None]).squeeze(-1),
        get('root_angular_velocity_world_rad_s'), q[..., view.joints(POSITION_JOINTS)],
        dq[..., view.joints(VELOCITY_JOINTS)])
    knee_indices = view.joints(tuple(linkage.sides[side.name].passive_joint_names[side.passive_knee]
                                    for side in geometry.sides))
    report = {'replay': str(path), 'intervals': int(rows.sum()), 'ticks': int(rows.sum()) * view.substeps,
              'scope': 'ordinary flat, no special gimbal/skill request; includes failure events', 'geometry': {}}
    n = len(start.position)
    for measured in (False, True):
        ticks = []
        for index in range(view.substeps):
            state = motion.tick(index)
            kwargs = ({'passive_knee_position': q[:, index, knee_indices],
                       'passive_knee_velocity': dq[:, index, knee_indices]} if measured else {})
            geom = geometry(state, **kwargs)
            command = get('command_admitted_body')[:, index]
            reference = RewardReference(command, torch.zeros(n, dtype=torch.long),
                torch.full((n,), int(RewardPhase.SUPPORT)), torch.zeros(n), torch.zeros(n), torch.zeros(n, 2),
                intentional_base_contact=get('body_contact_allowed')[:, index, 0] > .5)
            contact = get('terrain_contact_force_world_n')[:, index]
            ticks.append(RewardTick(state, geom, get('ground_at_root_world_z_m')[:, index, 0],
                get('ground_at_wheel_world_z_m')[:, index],
                get('body_normal_load_n')[:, index, view.bodies(CONTACT_BODIES)],
                get('wheel_support_load_n')[:, index], contact[:, view.bodies((*WHEEL_BODIES, 'base_link'))].norm(dim=-1),
                get('contact_measurement_valid')[:, index, 0] > .5, torch.zeros(n, dtype=torch.bool), reference,
                torch.full((n,), (index + 1) * view.physics_dt_s, dtype=torch.float64),
                data['start_episode_id'][rows], data['start_control_epoch'][rows], torch.ones(n, dtype=torch.bool)))
        result = assembly.integrate(start, tuple(ticks), data['actions'][rows], memory,
            current('identified.wheel_torque_history_nm'), legs, ctx('requested.pd_encoder_bias_rad'),
            ctx('requested.motor_strength'))
        reward = result.finish(data['terminated'][rows], torch.zeros(n, dtype=torch.bool), parameters)
        continuing = ~view.ended[rows]
        end_columns = []
        for f in fields(SupportState):
            value = field(data['end_state'][rows], 'course.support._' + f.name)
            end_columns.append(value if value.ndim == 2 else value[:, None])
        expected_support = torch.cat(end_columns, -1)
        report['geometry']['measured_knees' if measured else 'ideal_linkage'] = {
            'reward_error': stats(reward - data['rewards'][rows]),
            'support_memory_error_nonterminal': stats((result.memory.support.encode() - expected_support)[continuing]),
            'invalid_intervals': int(result.invalid.sum()),
            'expected_reward': stats(data['rewards'][rows]),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('replay', type=Path)
    parser.add_argument('--table', type=Path, default=Path('configs/linkage/leg_linkage_e726d23c.json'))
    parser.add_argument('--asset', type=Path, default=Path('assets/full_vehicle_v2p0p1_source_audited_v2'))
    parser.add_argument('--bundle', type=Path, default=Path('configs/identified/20261031'))
    parser.add_argument('--terrain-atlas', type=Path)
    parser.add_argument('--recorded-jumps', action='store_true',
                        help='Audit recorded jump tasks including HOLD misses; requires --terrain-atlas')
    parser.add_argument('--device', default='cpu', help='Device for the compact/atlas audit')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.recorded_jumps and args.terrain_atlas is None:
        parser.error('--recorded-jumps requires --terrain-atlas')
    report = ({} if args.terrain_atlas is not None else check(args.replay, args.table, args.asset, args.bundle))
    report['compact_imagination_producer'] = check_compact(args.replay, args.table, args.asset, args.bundle,
        terrain_path=args.terrain_atlas, device=args.device, recorded_jumps=args.recorded_jumps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
