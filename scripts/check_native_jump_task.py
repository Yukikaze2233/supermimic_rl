#!/usr/bin/env python3
"""Audit recorded physical ticks through the native jump/command/stair chain.

No world model or policy is loaded. Adjacent same-episode records independently
check persistent physical evidence and the next protocol/visible command. The
compact recording does not contain per-tick native reward references, so their
reconstruction is explicitly not an independent substep-reference comparison.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from supermimic_rl.world_model_v2.geometry import RewardGeometry
from supermimic_rl.world_model_v2.ingest import load_recorded_replay
from supermimic_rl.world_model_v2.jump_task import (
    JUMP_RECORD_FIELDS, JUMP_VALUE_NAMES, JumpTask, JumpTaskConfig, JumpTaskState,
)
from supermimic_rl.world_model_v2.leg_linkage import LegLinkage
from supermimic_rl.world_model_v2.model import Ticks
from supermimic_rl.world_model_v2.motion_task import MotionTask
from supermimic_rl.world_model_v2.records import decode_physical, decode_queues, decode_readout, encode_physical
from supermimic_rl.world_model_v2.stair_source import StairScenario, StairSourceState
from supermimic_rl.world_model_v2.stair_task import StairTaskState
from supermimic_rl.world_model_v2.terrain import AtlasTerrain


@torch.no_grad()
def audit(replay, schema, *, jump_config, stair_scenario, source_rng, terrain, geometry):
    """Inspect original vector rows; never replace truth with a model prediction."""
    n, device = replay.num_envs, replay.device
    data, sequence = replay.data, replay.sequence[:replay.size]
    required = {*JUMP_RECORD_FIELDS, 'jump_record_valid', 'task_estimate', 'motion_task',
                'task_record_valid', 'stair_source_valid', 'ticks', 'end_physical', 'end_height',
                'sim_context', 'terrain_tile', 'task_family', 'frame', 'queues', 'readout'}
    if not required.issubset(data):
        raise ValueError('Jump audit requires recorded native fields: ' + repr(sorted(required - data.keys())))
    if (source_rng['format'] != 'supermimic-vector-step-rng-v1' or source_rng['num_envs'] != n
            or not torch.equal(sequence.cpu(), source_rng['sequence'])):
        raise ValueError('Stair source RNG must cover every original vector step in order')
    if torch.device(source_rng['device']).type != device.type:
        raise ValueError('Stair RNG replay requires the native generator device type')
    if schema.policy_dt_s != .02 or schema.substeps_per_step != 4:
        raise ValueError('Jump audit requires actual 200/50 Hz records')
    task = JumpTask(JumpTaskConfig.from_dict(jump_config))
    if task.config.trigger is not None:
        from supermimic_rl.world_model_v2.jump_task import JUMP_TRIGGER_RECORD_FIELDS
        if not set(JUMP_TRIGGER_RECORD_FIELDS).issubset(data):
            raise ValueError('Gated native audit needs recorded jump trigger history')
    stair_task, source = StairScenario.from_dict(stair_scenario).build(terrain)
    generator = torch.Generator(device=device)
    if (source_rng['state'].dtype != torch.uint8
            or source_rng['state'].shape != (replay.size, generator.get_state().numel())):
        raise ValueError('Recorded source RNG state layout differs from the native generator')
    counts = dict.fromkeys(('candidate_intervals', 'eligible_intervals', 'reset_invalid_or_unowned',
        'missing_task_record', 'unsupported_task', 'unknown_command_boundary', 'terrain_query_invalid',
        'nonconsecutive_vector_step', 'request_family_intervals', 'active_request_intervals',
        'physical_active_intervals', 'flight_intervals', 'landing_intervals', 'accepted_requests',
        'finished_requests', 'physical_success_events', 'recovered_auto_deadlines', 'stair_fired_requests',
        'trigger_deferred_rows'), 0)
    totals, first_mismatch = {}, {}
    atol, rtol = 2e-5, 2e-6

    def compare(name, actual, expected, rows, slot):
        if actual.shape != expected.shape:
            raise ValueError('Audit comparison shape differs: ' + name)
        same = (torch.isclose(actual, expected, atol=atol, rtol=rtol)
                if actual.is_floating_point() else actual == expected)
        error = (torch.where(actual == expected, 0., (actual - expected).abs()).double()
                 if actual.is_floating_point() else (~same).double())
        bad = ~same.reshape(len(rows), -1).all(-1)
        statistics = totals.setdefault(name, {'max_abs': 0., 'square_sum': 0., 'elements': 0,
                                              'mismatching_intervals': 0, 'compared_intervals': 0})
        statistics['max_abs'] = max(statistics['max_abs'], float(error.max()))
        statistics['square_sum'] += float(error.square().sum())
        statistics['elements'] += error.numel()
        statistics['mismatching_intervals'] += int(bad.sum())
        statistics['compared_intervals'] += len(rows)
        if bad.any() and name not in first_mismatch:
            index = int(bad.nonzero()[0, 0])
            first_mismatch[name] = {'sequence': int(sequence[slot]), 'env': int(rows[index]),
                'actual': actual[index].tolist(), 'expected': expected[index].tolist()}

    for slot in range(replay.size - 1):
        counts['candidate_intervals'] += n
        if sequence[slot + 1] != sequence[slot] + 1:
            counts['nonconsecutive_vector_step'] += n
            continue
        start = {name: value[slot] for name, value in data.items()}
        following = {name: value[slot + 1] for name, value in data.items()}
        continuous = (start['valid'] & following['valid'] & start['policy_owned'] & start['end_policy_owned']
            & following['policy_owned'] & ~start['terminated'] & ~start['truncated']
            & (start['start_episode_id'] == start['end_episode_id'])
            & (start['end_episode_id'] == following['start_episode_id'])
            & (start['start_control_epoch'] == start['end_control_epoch'])
            & (start['end_control_epoch'] == following['start_control_epoch']))
        recorded = torch.ones(n, dtype=torch.bool, device=device)
        for name in ('jump_record_valid', 'task_record_valid', 'stair_source_valid'):
            recorded &= (start[name][:, 0] > .5) & (following[name][:, 0] > .5)
        motion = MotionTask.decode(start['motion_task'])
        supported = (motion.route[:, 3] <= 2) & (start['task_family'][:, 0] < 5) & (start['goal'][:, 7] <= .5)
        jump, stair = JumpTaskState.decode(start), StairTaskState.decode(start)
        known = StairSourceState.decode(start).boundary_known(motion, stair, schema.policy_dt_s, jump)
        counts['reset_invalid_or_unowned'] += int((~continuous).sum())
        counts['missing_task_record'] += int((continuous & ~recorded).sum())
        counts['unsupported_task'] += int((continuous & recorded & ~supported).sum())
        counts['unknown_command_boundary'] += int((continuous & recorded & supported & ~known).sum())
        rows = (continuous & recorded & supported & known).nonzero().flatten()
        if not len(rows):
            continue
        before = {name: value[rows] for name, value in start.items()}
        after = {name: value[rows] for name, value in following.items()}
        state = decode_physical(before['physical'])
        ticks = Ticks.decode(before['ticks'], state)
        ground = terrain.height_at(ticks.state.position, before['terrain_tile'][:, 0])
        if not ground.valid.all():
            counts['terrain_query_invalid'] += int((~ground.valid.all(-1)).sum())
            rows = rows[ground.valid.all(-1)]
            if not len(rows):
                continue
            before = {name: value[rows] for name, value in start.items()}
            after = {name: value[rows] for name, value in following.items()}
            state = decode_physical(before['physical'])
            ticks = Ticks.decode(before['ticks'], state)
            ground = terrain.height_at(ticks.state.position, before['terrain_tile'][:, 0])
        jump, stair = JumpTaskState.decode(before), StairTaskState.decode(before)
        recorded = StairSourceState.decode(before)
        motion = MotionTask.decode(before['motion_task'])
        end = decode_physical(before['end_physical'])
        physical = task.observe_interval(jump, physical=ticks.state,
            wheel_support_load=ticks.wheel_load,
            base_com_offset=before['sim_context'][:, 41:44], base_mode=motion.route[:, 3].long(),
            command_height=before['goal'][:, 3] / 5., ground_z=ground.height)
        estimate = stair_task.completed_estimate(before['frame'][:, None], before['task_estimate'], jump.now_s)
        following_motion, admitted = motion.advance_admission(
            decode_queues(after['queues']), decode_readout(after['readout']), schema.policy_dt_s)
        expected_jump = JumpTaskState.decode(after)
        due = ((before['task_family'][:, 0] == 4) & (stair.ids[:, 1] < 0)
               & (following_motion.timing[:, 0] >= jump.next_request_s) & (jump.active_id < 0))
        changed = expected_jump.next_request_s != jump.next_request_s
        recovered = (expected_jump.next_request_s - following_motion.timing[:, 0] - 3.) / 5.
        uniform = torch.where(due, recovered, .5).clamp(0, 1)
        advanced = task.advance(physical.state, now_s=task.boundary_time(physical.state), estimate=estimate,
            admitted=admitted, base_mode=following_motion.route[:, 3].long(),
            task_family=before['task_family'][:, 0].long(), elapsed_s=following_motion.timing[:, 0],
            ground_z=end.position[:, 2] - before['end_height'][:, 0], blocked_by_stair=stair.ids[:, 1] >= 0,
            interval_uniform=uniform, previous_command_height=motion.current[:, 3],
            previous_elapsed_s=motion.timing[:, 0])
        due &= ~advanced.deferred
        counts['trigger_deferred_rows'] += int(advanced.deferred.sum())
        compare('automatic_request_due', due, changed & ~advanced.deferred, rows, slot)
        compare('automatic_interval_in_native_range', (~due | ((recovered >= -2e-5) & (recovered <= 1 + 2e-5))),
                torch.ones_like(due), rows, slot)
        following_motion = replace(following_motion, current=advanced.command)
        # Give the backend an owning byte tensor, not an offset view into the
        # stacked/mmap record (CPU generator state readers require this).
        generator.set_state(source_rng['state'][slot].clone())
        draws = source.draw(n, 1, device=device, generator=generator)
        wheel = geometry(end, passive_knee_position=ticks.passive_knee_position[:, -1],
                         passive_knee_velocity=ticks.passive_knee_velocity[:, -1])
        boundary = source.advance(recorded, stair, task=stair_task, motion=motion, physical=end,
            wheel_centres=wheel.centre_world, wheel_radius=end.position.new_tensor([side.radius for side in geometry.sides]),
            terrain_tile=before['terrain_tile'][:, 0], queues=decode_queues(after['queues']),
            readout=decode_readout(after['readout']), goal=before['goal'], estimate=estimate, physics_dt_s=.005,
            latency=draws['tof_latency'][rows, 0], draws=draws['tof_uniform'][rows, 0],
            admission=(following_motion, advanced.command), base_mode=advanced.mode,
            jump_active=advanced.state.active_id >= 0, request_height=advanced.request_height,
            request_clock=advanced.request_clock)
        result = boundary.boundary
        actual = {**advanced.state.record(), **boundary.source.record(), **result.stair.state.record(),
                  'motion_task': result.motion.encode(), 'goal': result.goal,
                  'failure_conditions': result.failure_conditions}
        for name, value in actual.items():
            expected = after[name]
            if name == 'jump_values':
                # The realized random deadline supplied above is an input,
                # not independent evidence that its RNG was reproduced.
                columns = [i for i in range(value.shape[-1]) if i != 3]
                value, expected = value[:, columns], expected[:, columns]
            compare(name, value, expected, rows, slot)
        compare('true_tick_endpoint', encode_physical(ticks.state.tick(-1)), before['end_physical'], rows, slot)
        compare('physical_phase_last_tick', physical.phase[:, -1], expected_jump.ids[:, 8], rows, slot)
        compare('physical_payment_event', physical.success_credited,
                expected_jump.ids[:, 6] > jump.ids[:, 6], rows, slot)
        expected_reference, actual_reference = task.reference(expected_jump), task.reference(advanced.state)
        for name in ('height_world_z_m', 'velocity_world_z_m_s'):
            compare('next_reference_from_recorded_state/' + name,
                    getattr(actual_reference, name), getattr(expected_reference, name), rows, slot)
        counts['eligible_intervals'] += len(rows)
        counts['request_family_intervals'] += int((before['task_family'][:, 0] == 4).sum())
        counts['active_request_intervals'] += int((jump.active_id >= 0).sum())
        counts['physical_active_intervals'] += int((jump.physical_id >= 0).sum())
        counts['flight_intervals'] += int((physical.phase == 2).any(-1).sum())
        counts['landing_intervals'] += int((physical.phase == 3).any(-1).sum())
        counts['accepted_requests'] += int(((advanced.state.active_id >= 0) & (advanced.state.active_id != jump.active_id)).sum())
        counts['finished_requests'] += int((advanced.finished_id >= 0).sum())
        counts['physical_success_events'] += int(physical.success_credited.sum())
        counts['recovered_auto_deadlines'] += int(due.sum())
        counts['stair_fired_requests'] += int(boundary.fired.sum())
    if not counts['eligible_intervals']:
        raise ValueError('No complete recorded same-episode jump-capable boundary to audit')
    for statistics in totals.values():
        statistics['rms'] = (statistics.pop('square_sum') / statistics.pop('elements')) ** .5
    return {'format': 'supermimic-native-jump-audit-v1',
        'passed': not any(value['mismatching_intervals'] for value in totals.values()),
        'counts': counts, 'fields': totals, 'first_mismatch': first_mismatch,
        'jump_values_compared_columns': [name for name in JUMP_VALUE_NAMES if name != 'next_request_s'],
        'schema_fingerprint': schema.fingerprint, 'float_comparison': {'atol': atol, 'rtol': rtol},
        'randomness': 'stair RNG replayed at original full-vector size/device; automatic jump deadline recovered from following record, its value excluded from independent equality checks; due/range/IDs checked',
        'reference_scope': 'next reference reconstructed from independently recorded task state; per-tick native phase/reference not recorded, so no independent four-reference equality claim',
        'boundary_scope': 'adjacent consecutive valid policy-owned same episode/control epoch; reset/terminal/truncated/recovery/unknown command-resampling boundaries excluded and counted',
        'scope': 'actual recorded four-tick physics, resultant forces and deployed estimates; no learned motion, no teacher jump qualification or robot jump-success claim'}


def check(path, *, atlas, table, asset, device):
    payload = torch.load(path, weights_only=True, map_location='cpu', mmap=True)
    replay, schema = load_recorded_replay(path, device, record_terrain=True)
    pins = json.loads(schema.source)
    terrain = AtlasTerrain.load(atlas, expected_sha256=pins['terrain_sha256'], device=device)
    geometry = RewardGeometry(LegLinkage.load(table,
        expected_asset_manifest_sha256=pins['asset_manifest_sha256']), asset)
    report = audit(replay, schema, jump_config=payload['jump_task_config'],
        stair_scenario=payload['stair_source_config'], source_rng=payload['source_rng'], terrain=terrain, geometry=geometry)
    return {**report, 'dataset': str(Path(path).resolve()), 'teacher_source': payload.get('collection', {}).get('teacher'),
            'collection_coordinate': payload.get('collection', {}).get('fixed_course_coordinate'),
            'asset_manifest_sha256': pins['asset_manifest_sha256'], 'atlas_sha256': terrain.manifest_sha256}


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
