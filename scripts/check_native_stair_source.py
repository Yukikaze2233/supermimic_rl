#!/usr/bin/env python3
"""Replay true physics through the imagined stair/command boundary, without learning.

The recorded deployed estimate drives decisions. Native vector-step RNG states
reproduce external request draws before selecting rows, including inactive draws.
Rows crossing resets or unknown command resampling are counted and excluded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from transformer_rl.formal.lab_env.course_tasks import StairRequestConfig
from transformer_rl.formal.lab_env.skill_fsm import StairFSMConfig, StairHeightAssistConfig
from transformer_rl.formal.lab_env.stair_trigger import TofTriggerConfig
from supermimic_rl.world_model_v2.geometry import RewardGeometry
from supermimic_rl.world_model_v2.ingest import load_recorded_replay
from supermimic_rl.world_model_v2.leg_linkage import LegLinkage
from supermimic_rl.world_model_v2.model import Ticks
from supermimic_rl.world_model_v2.motion_task import MotionTask
from supermimic_rl.world_model_v2.records import decode_physical, decode_queues, decode_readout
from supermimic_rl.world_model_v2.stair_source import IdealStairSource, StairSourceState
from supermimic_rl.world_model_v2.stair_task import StairTask, StairTaskConfig, StairTaskState
from supermimic_rl.world_model_v2.terrain import AtlasTerrain


@torch.no_grad()
def check(path, *, atlas, table, asset, device):
    payload = torch.load(path, weights_only=True, map_location='cpu', mmap=True)
    metadata, rng = payload['stair_source_config'], payload['source_rng']
    if (metadata['format'] != 'supermimic-native-stair-source-v1'
            or rng['format'] != 'supermimic-vector-step-rng-v1'):
        raise ValueError('Unknown native source recording format')
    if torch.device(device).type != torch.device(rng['device']).type:
        raise ValueError('RNG replay requires the native generator device type')
    replay, schema = load_recorded_replay(path, device, record_terrain=True)
    n, data, sequence = replay.num_envs, replay.data, replay.sequence[:replay.size]
    if rng['num_envs'] != n or not torch.equal(sequence.cpu(), rng['sequence']):
        raise ValueError('Source RNG must cover every original vector step in order')
    config = metadata['task']
    task = StairTask(StairTaskConfig(StairRequestConfig(**config['request']), StairFSMConfig(**config['fsm']),
        None if config['height_assist'] is None else StairHeightAssistConfig(**config['height_assist']),
        config['estimate_max_age_s'], config['use_mode_labels']))
    pins = json.loads(schema.source)
    terrain = AtlasTerrain.load(atlas, expected_sha256=pins['terrain_sha256'], device=device)
    source = IdealStairSource(TofTriggerConfig(**metadata['tof']), terrain=terrain,
        min_height_m=metadata['min_height_m'], max_height_m=metadata['max_height_m'])
    geometry = RewardGeometry(LegLinkage.load(table, expected_asset_manifest_sha256=pins['asset_manifest_sha256']), asset)
    generator = torch.Generator(device=device)
    errors, failures, samples = {}, {}, {}
    counts = {'candidate_intervals': (replay.size - 1) * n, 'eligible_intervals': 0,
              'unknown_command_boundary': 0, 'reset_or_unowned': 0, 'unsupported_task': 0,
              'fired_requests': 0, 'active_intervals': 0, 'finished_requests': 0,
              'expired_but_stair_blocked': 0, 'terrain_kind': [0] * 8}
    for slot in range(replay.size - 1):
        start = {key: value[slot] for key, value in data.items()}
        following = {key: value[slot + 1] for key, value in data.items()}
        stair, recorded = StairTaskState.decode(start), StairSourceState.decode(start)
        motion = MotionTask.decode(start['motion_task'])
        owned = (start['valid'] & start['policy_owned'] & start['end_policy_owned']
                 & ~start['terminated'] & ~start['truncated'] & following['policy_owned']
                 & (start['start_episode_id'] == start['end_episode_id'])
                 & (start['end_episode_id'] == following['start_episode_id'])
                 & (start['start_control_epoch'] == start['end_control_epoch'])
                 & (start['end_control_epoch'] == following['start_control_epoch']))
        ordinary = (motion.route[:, 3] <= 2) & (start['goal'][:, 7:9] <= .5).all(-1)
        known = recorded.boundary_known(motion, stair, schema.policy_dt_s)
        counts['reset_or_unowned'] += int((~owned).sum())
        counts['unsupported_task'] += int((owned & ~ordinary).sum())
        counts['unknown_command_boundary'] += int((owned & ordinary & ~known).sum())
        selected = owned & ordinary & known
        if not selected.any():
            continue
        # CUDA and CPU generators consume vector lengths differently. Never
        # redraw only the selected rows or substitute a CPU RNG for CUDA.
        generator.set_state(rng['state'][slot])
        draws = source.draw(n, 1, device=device, generator=generator)
        before = {key: value[selected] for key, value in start.items()}
        after = {key: value[selected] for key, value in following.items()}
        state, end = decode_physical(before['physical']), decode_physical(before['end_physical'])
        ticks = Ticks.decode(before['ticks'], state)
        wheel = geometry(end, passive_knee_position=ticks.passive_knee_position[:, -1],
                         passive_knee_velocity=ticks.passive_knee_velocity[:, -1])
        stair, recorded = StairTaskState.decode(before), StairSourceState.decode(before)
        motion = MotionTask.decode(before['motion_task'])
        estimate = task.completed_estimate(before['frame'][:, None], before['task_estimate'], stair.clock[:, 0])
        result = source.advance(recorded, stair, task=task, motion=motion, physical=end,
            wheel_centres=wheel.centre_world, wheel_radius=end.position.new_tensor([side.radius for side in geometry.sides]),
            terrain_tile=before['terrain_tile'][:, 0], queues=decode_queues(after['queues']),
            readout=decode_readout(after['readout']), goal=before['goal'], estimate=estimate,
            physics_dt_s=schema.policy_dt_s / 4, latency=draws['tof_latency'][selected, 0],
            draws=draws['tof_uniform'][selected, 0])
        boundary = result.boundary
        actual = {**result.source.record(), **boundary.stair.state.record(),
                  'motion_task': boundary.motion.encode(), 'goal': boundary.goal,
                  'failure_conditions': boundary.failure_conditions}
        count = int(selected.sum())
        counts['eligible_intervals'] += count
        counts['fired_requests'] += int(result.fired.sum())
        counts['active_intervals'] += int((stair.ids[:, 1] >= 0).sum())
        counts['finished_requests'] += int((boundary.stair.finished_id >= 0).sum())
        counts['expired_but_stair_blocked'] += int(((motion.timing[:, 0] >= recorded.segment_end[:, 0])
                                                   & (stair.ids[:, 1] >= 0)).sum())
        kinds = torch.bincount(before['reward_route'][:, 1].long(), minlength=8).tolist()
        counts['terrain_kind'] = [a + b for a, b in zip(counts['terrain_kind'], kinds)]
        for name, value in actual.items():
            expected = after[name]
            if value.is_floating_point():
                equal = torch.isclose(value, expected, atol=2e-5, rtol=2e-6)
                error = torch.where(value == expected, 0., (value - expected).abs()).double()
            else:
                equal = value == expected
                error = (~equal).double()
            bad = ~equal.flatten(1).all(-1)
            failures[name] = failures.get(name, 0) + int(bad.sum())
            errors.setdefault(name, []).append(error.cpu())
            if bad.any() and name not in samples:
                index = int(bad.nonzero()[0, 0])
                samples[name] = {'sequence': int(sequence[slot]),
                    'env': int(selected.nonzero()[index, 0]),
                    'actual': value[index].tolist(), 'expected': expected[index].tolist()}
    if not counts['eligible_intervals']:
        raise ValueError('No complete eligible native source interval')
    fields = {}
    for name, parts in errors.items():
        error = torch.cat(parts).flatten()
        fields[name] = {'max_abs': float(error.max()), 'rms': float(error.square().mean().sqrt()),
                        'mismatching_intervals': failures[name]}
    return {'passed': not any(failures.values()), 'counts': counts, 'fields': fields,
            'first_mismatch': samples, 'schema_fingerprint': schema.fingerprint,
            'asset_manifest_sha256': pins['asset_manifest_sha256'], 'atlas_sha256': terrain.manifest_sha256,
            'float_comparison': {'atol': 2e-5, 'rtol': 2e-6},
            'scope': 'true native endpoints, measured passive knees and deployed estimates through source/task chain; no learned motion or skill qualification'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('replay', type=Path)
    parser.add_argument('--atlas', type=Path, required=True)
    parser.add_argument('--table', type=Path, default=Path('configs/linkage/leg_linkage_e726d23c.json'))
    parser.add_argument('--asset', type=Path, default=Path('assets/full_vehicle_v2p0p1_source_audited_v2'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = check(args.replay, atlas=args.atlas, table=args.table, asset=args.asset, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
