#!/usr/bin/env python3
"""Audit a saved online world model and its paired held-out replay, without Isaac.

Saved artifacts remain unchanged. CPU audits use a documented
fresh sampling seed; they do not reinterpret a CUDA RNG as a CPU RNG or alter
the training run. Only original held-out rows are loaded. Optional policy
updates exercise a disposable copy after model validation, never a training run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import torch

from supermimic_rl.contracts import StateSchema
from supermimic_rl.world_model_v2.factory import make_learner
from supermimic_rl.world_model_v2.online import CHECKPOINT_FORMAT, OnlineConfig, policy_start_candidates
from supermimic_rl.world_model_v2.records import MODEL_CONTRACT
from supermimic_rl.world_model_v2.replay import CompactReplay
from supermimic_rl.world_model_v2.validation import validate
from supermimic_rl.world_model_v2.stair_source import StairScenario
from supermimic_rl.world_model_v2.terrain import AtlasTerrain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--factory-config', type=Path, required=True)
    parser.add_argument('--linkage', type=Path, default=Path('configs/linkage/leg_linkage_e726d23c.json'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--horizon', type=int, default=32)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--source-recorded-only', action='store_true',
                        help='Require actual stair-source metadata on every step of each sampled window')
    parser.add_argument('--task-family', type=int, choices=range(6), action='append',
                        help='Select window starts by recorded TaskFamily; repeat to form a union')
    parser.add_argument('--terrain-kind', type=int, choices=range(8), action='append',
                        help='Select window starts by native TerrainKind; repeat to form a union')
    parser.add_argument('--active-stair-request', action='store_true',
                        help='Select starts with an active recorded stair request; not a success label')
    parser.add_argument('--policy-updates', type=int, default=0,
                        help='Exercise BPTT on a disposable restored copy; no weights are persisted')
    parser.add_argument('--policy-batch', type=int, default=4096)
    parser.add_argument('--policy-microbatch', type=int, default=512)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('World-model audit requires a fresh report path')
    if args.policy_updates < 0 or args.policy_batch < 1 or args.policy_microbatch < 1:
        raise ValueError('Policy audit counts must be nonnegative, with positive batches')
    started = time.monotonic()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if saved.get('format') != CHECKPOINT_FORMAT or saved['model_contract'] != MODEL_CONTRACT:
        raise ValueError('Audit requires a complete current online checkpoint')
    schema = StateSchema.from_dict(saved['schema'])
    if schema.fingerprint != saved['schema_fingerprint']:
        raise ValueError('Checkpoint schema fingerprint differs')
    source, config = json.loads(schema.source), OnlineConfig.from_dict(saved['config'])
    factory = json.loads(args.factory_config.read_text())
    upstream = Path(factory['upstream_root']).resolve(strict=True)
    if (subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
            != factory['expected_commit'] or factory['expected_commit'] != source['upstream_commit']
            or subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip()):
        raise ValueError('Audit upstream must match the native recorded commit')
    sys.path.insert(0, str(upstream / 'src'))
    runtime = None
    for kind in ('runtime', 'course'):
        content = Path(factory[kind + '_json']).read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != factory[kind + '_sha256'] or digest != source[kind + '_sha256']:
            raise ValueError('Audit pin differs from the recorded source: ' + kind)
        if kind == 'runtime':
            runtime = json.loads(content)
    replay_path = args.checkpoint.parent / saved['replay_file']
    dataset = torch.load(replay_path, map_location='cpu', weights_only=True, mmap=True)
    if dataset['cycle'] != saved['cycle'] or dataset['schema_fingerprint'] != schema.fingerprint:
        raise ValueError('Audit checkpoint/replay pair differs')
    count = dataset['replay']['num_envs']
    held = max(1, round(count * config.validation_fraction))
    if held >= count:
        raise ValueError('Audit needs the original disjoint training/validation split')
    rows = list(range(count - held, count))
    replay = CompactReplay.from_state_dict(dataset['replay'], args.device, env_ids=rows)
    required = starts = None
    if args.source_recorded_only or args.task_family or args.active_stair_request:
        if 'stair_source_valid' not in replay.data:
            raise ValueError('This population requires recorded source validity, not migrated placeholder fields')
        required = replay.data['stair_source_valid'][..., 0] > .5
    if args.task_family or args.terrain_kind or args.active_stair_request:
        starts = torch.ones(replay.capacity, replay.num_envs, dtype=torch.bool, device=args.device)
        if args.task_family:
            starts &= torch.isin(replay.data['task_family'][..., 0],
                                  torch.tensor(args.task_family, device=args.device))
        if args.terrain_kind:
            starts &= torch.isin(replay.data['reward_route'][..., 1],
                                  torch.tensor(args.terrain_kind, device=args.device))
        if args.active_stair_request:
            starts &= replay.data['stair_ids'][..., 1] >= 0
    terrain = stair_task = stair_source = jump_task = None
    if config.record_stair_source:
        terrain = AtlasTerrain.load(runtime['terrain_atlas_pin']['path'],
            expected_sha256=source['terrain_sha256'], device=args.device)
        stair_task, stair_source = StairScenario.from_dict(saved['stair_scenario']).build(terrain)
    if config.record_jumps:
        from supermimic_rl.world_model_v2.jump_task import JumpTask, JumpTaskConfig
        jump_task = JumpTask(JumpTaskConfig.from_dict(saved['jump_scenario']))
    learner = make_learner(schema, runtime, config.learning, linkage_path=args.linkage,
        asset_directory=factory['asset_directory'], identified_bundle=factory['identified_bundle'],
        device=args.device, model_hidden=config.model_hidden, actor_hidden=config.actor_hidden,
        estimator_hidden=config.estimator_hidden, critic_hidden=config.critic_hidden, record_tasks=config.record_tasks,
        terrain=terrain, stair_task=stair_task, stair_source=stair_source, jump_task=jump_task,
        actor_frame_dim=config.actor_frame_dim)
    learner.model.load_state_dict(saved['learner']['model'])
    learner.model.eval().requires_grad_(False)
    report = validate(learner.fit, replay, list(range(held)), args.horizon, args.batch,
        torch.Generator(device=args.device).manual_seed(args.seed), args.device,
        contact_threshold_n=learner.imagination.termination.limits.forbidden_body_load_n,
        termination=learner.imagination.termination, required_rows=required, start_rows=starts)
    result = {'checkpoint': str(args.checkpoint.resolve()), 'replay': str(replay_path.resolve()),
        'cycle': saved['cycle'], 'updates': saved['learner']['updates'], 'validation': report,
        'original_validation_envs': [rows[0], rows[-1]], 'validation_env_count': held,
        'sampling_seed': args.seed, 'device': args.device, 'elapsed_s': time.monotonic() - started,
        'population': {'source_recorded_entire_window': required is not None,
            'start_task_family': args.task_family, 'start_terrain_kind': args.terrain_kind,
            'start_has_active_stair_request': args.active_stair_request},
        'scope': 'saved world-model accuracy only; no physics, training, or student capability claim'}
    if args.policy_updates:
        learner.load_state_dict(saved['learner'])
        torch.manual_seed(args.seed + 1)
        generator = torch.Generator(device=args.device).manual_seed(args.seed + 2)
        candidates = policy_start_candidates(replay, config, list(range(held)), schema.policy_dt_s)
        offset, env = candidates // replay.num_envs, candidates % replay.num_envs
        slots = (offset + replay.head - replay.size) % replay.capacity
        selected = torch.ones_like(candidates, dtype=torch.bool)
        for mask in (required, starts):
            if mask is not None:
                selected &= mask[slots, env]
        candidates = candidates[selected]
        if not candidates.numel():
            raise ValueError('No policy starts in this recorded population')

        def sample():
            return replay.sample_starts(args.policy_batch, learner.actor.history_length,
                                        generator=generator, candidates=candidates)

        first = saved.get('critic_norm_from')
        if first is not None:
            calibration, _ = sample()
            learner.calibrate_appended_critic_features(calibration, first)
        attempts = []
        for index in range(args.policy_updates):
            tick = time.monotonic()
            start, frames = sample()
            metrics = learner.update_policy(start, frames, microbatch_size=args.policy_microbatch)
            metrics = {key: value if math.isfinite(value) else None for key, value in metrics.items()}
            attempts.append({'attempt': index + 1, 'elapsed_s': time.monotonic() - tick, **metrics})
            print(json.dumps({'phase': 'disposable_policy_audit', **attempts[-1]}, allow_nan=False), flush=True)
        result['policy_audit'] = {'batch': args.policy_batch, 'microbatch': args.policy_microbatch,
            'eligible_starts': candidates.numel(), 'appended_norm_calibrated_from': first,
            'source_updates': saved['learner']['updates'], 'updated_copy_counts': learner.updates,
            'attempts': attempts, 'scope': 'disposable BPTT/optimizer check on held records; no retained policy or skill claim'}
        result['scope'] = 'world accuracy before disposable policy audit; saved artifacts and native training unchanged'
    result['elapsed_s'] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(args.output)
    print(json.dumps({'report': str(args.output), 'cycle': saved['cycle'], 'elapsed_s': result['elapsed_s']}))


if __name__ == '__main__':
    main()
