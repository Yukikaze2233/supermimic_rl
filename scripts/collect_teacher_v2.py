#!/usr/bin/env python3
"""Capture a pinned own teacher on the current native asset into compact V2 replay."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import traceback

import torch

from supermimic_rl.adapters.packed import PackedEnvironment
from supermimic_rl.cli import native_logs, _native_schema_source
from supermimic_rl.runtime import atomic_save
from supermimic_rl.world_model_v2.collection import Collector
from supermimic_rl.world_model_v2.records import MODEL_CONTRACT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--factory-config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--action-noise-std', type=float, default=0.)
    parser.add_argument('--curriculum-transition-cap', type=int, default=0)
    parser.add_argument('--course-coordinate', type=int,
                        help='Hold an existing course lookup coordinate from the first birth; not an exposure counter')
    parser.add_argument('--record-terrain', action='store_true', help='Store native tile IDs for atlas rollouts')
    parser.add_argument('--record-tasks', action='store_true', help='Store the native command trajectory and control state')
    parser.add_argument('--record-stairs', action='store_true', help='Store exact native stair protocol/FSM state; no future packets inferred')
    parser.add_argument('--record-stair-source', action='store_true',
        help='Record the existing ideal simulation request source, including its terrain/command/FSM dependencies')
    parser.add_argument('--record-jumps', action='store_true',
        help='Record native jump requests/evidence and all terrain, command and stair arbitration dependencies')
    parser.add_argument('--record-recovery', action='store_true',
        help='Record native recovery lifecycle and complete jump/stair arbitration dependencies')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.record_jumps = args.record_jumps or args.record_recovery
    args.record_stair_source = args.record_stair_source or args.record_jumps
    if args.course_coordinate is not None and args.course_coordinate < 0:
        parser.error('--course-coordinate must be nonnegative')
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Teacher datasets require a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.factory_config.read_text())
    if not config.get('record_substeps', False):
        raise ValueError('V2 teacher collection requires physical substep labels')
    config['native_log_directory'] = str(output / 'native')
    (output / 'factory-config.json').write_text(json.dumps(config, indent=2) + '\n')
    with native_logs(output / 'native'):
        from supermimic_rl.course.factory import create_course
        adapter = create_course(config)
        env = PackedEnvironment(adapter, source=_native_schema_source(adapter.provenance), policy_dt_s=adapter.policy_dt_s)
        try:
            from supermimic_rl.world_model_v2.teacher import load_native_teacher, native_teacher_decision
            if args.course_coordinate is not None:
                env.set_training_progress(real_transitions=args.course_coordinate, actor_updates=0)
            env.reset()
            teacher = load_native_teacher(args.checkpoint, args.checkpoint_sha256,
                                          device=env.device, native_provenance=adapter.provenance)
            def decision(history):
                return native_teacher_decision(teacher, env)

            collector = Collector(env, decision, capacity_steps=args.steps, seed=config.get('seed', 7),
                                  action_noise_std=args.action_noise_std,
                                  curriculum_transition_cap=args.curriculum_transition_cap,
                                  record_terrain=args.record_terrain or args.record_stair_source,
                                  record_tasks=args.record_tasks or args.record_stair_source,
                                  record_stairs=args.record_stairs or args.record_stair_source,
                                  record_stair_source=args.record_stair_source,
                                  record_jumps=args.record_jumps, record_recovery=args.record_recovery,
                                  course_coordinate=args.course_coordinate)
            started = time.monotonic()
            source_rng, source_config = [], None
            if args.record_stair_source:
                from supermimic_rl.world_model_v2.stair_source import StairScenario
                trigger = adapter.env._terrain_course.tof_trigger
                source_config = StairScenario.capture(adapter.env).to_dict()

            def progress(record):
                if record['steps'] % 24 == 0 or record['steps'] == args.steps:
                    with (output / 'progress.jsonl').open('a') as stream:
                        stream.write(json.dumps({**record, 'elapsed_s': time.monotonic() - started}) + '\n')

            report = collector.collect(args.steps, on_progress=progress,
                on_source_rng=((lambda sequence, state: source_rng.append((sequence, state.cpu())))
                               if args.record_stair_source else None))
            report.update(elapsed_s=time.monotonic() - started, teacher=teacher.source_metadata,
                          model_contract=MODEL_CONTRACT, action_noise_std=args.action_noise_std,
                          scope='native teacher data collection only; no world-model or policy update')
            payload = {'format': 'supermimic-world-v2-dataset-v1', 'model_contract': MODEL_CONTRACT,
                         'schema': env.schema.to_dict(), 'schema_fingerprint': env.schema.fingerprint,
                         'replay': collector.replay.state_dict(), 'collection': report}
            if source_config is not None:
                payload['stair_source_config'] = source_config
                payload['source_rng'] = {
                    'format': 'supermimic-vector-step-rng-v1', 'device': str(trigger.generator.device),
                    'num_envs': env.num_envs,
                    'sequence': torch.tensor([entry[0] for entry in source_rng], dtype=torch.int64),
                    'state': torch.stack([entry[1] for entry in source_rng]),
                }
                (output / 'stair-source-config.json').write_text(json.dumps(source_config, indent=2) + '\n')
            if args.record_jumps:
                from supermimic_rl.world_model_v2.jump_task import JumpTaskConfig
                payload['jump_task_config'] = asdict(JumpTaskConfig.capture(adapter.env))
                (output / 'jump-task-config.json').write_text(json.dumps(payload['jump_task_config'], indent=2) + '\n')
            if args.record_recovery:
                from supermimic_rl.world_model_v2.recovery_task import RecoveryTaskConfig
                payload['recovery_task_config'] = asdict(RecoveryTaskConfig.capture(adapter.env))
                (output / 'recovery-task-config.json').write_text(
                    json.dumps(payload['recovery_task_config'], indent=2) + '\n')
            atomic_save(payload, output / 'replay.pt')
            (output / 'collection.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report), flush=True)
        except BaseException:
            # Isaac shutdown can exit before Python prints the actual failure.
            (output / 'failure-traceback.txt').write_text(traceback.format_exc())
            raise
        finally:
            env.close()


if __name__ == '__main__':
    main()
