#!/usr/bin/env python3
"""Native V2 online training, cold resume, student evaluation and ONNX export."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import traceback

import torch


def read_actor(path, device):
    from supermimic_rl.world_model_v2.student_transfer import load_student_actor

    return load_student_actor(path, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('train', 'evaluate'):
        command = commands.add_parser(name)
        command.add_argument('--factory-config', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
    train = commands.choices['train']
    train.add_argument('--config', type=Path, required=True)
    train.add_argument('--teacher', type=Path, required=True)
    train.add_argument('--teacher-sha256', required=True)
    train.add_argument('--seed-dataset', type=Path)
    train.add_argument('--world-checkpoint', type=Path)
    train.add_argument('--resume', type=Path)
    train.add_argument('--linkage', type=Path, default=Path('configs/linkage/leg_linkage_e726d23c.json'))
    commands.choices['evaluate'].add_argument('--checkpoint', type=Path, required=True)
    commands.choices['evaluate'].add_argument('--steps', type=int, default=1000)
    export = commands.add_parser('export')
    export.add_argument('--checkpoint', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'export':
        from supermimic_rl.world_model_v2.policy import export_onnx
        actor, _ = read_actor(args.checkpoint, 'cpu')
        print(export_onnx(actor, args.output))
        return
    from supermimic_rl.adapters.packed import PackedEnvironment
    from supermimic_rl.cli import native_logs, _native_schema_source

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not getattr(args, 'resume', None):
        raise FileExistsError('Use a new output directory or explicit resume')
    output.mkdir(parents=True, exist_ok=True)
    factory = json.loads(args.factory_config.read_text())
    factory['record_substeps'] = True
    factory['native_log_directory'] = str(output / 'native')
    (output / 'factory-config.json').write_text(json.dumps(factory, indent=2) + '\n')
    with native_logs(output / 'native'):
        from supermimic_rl.course.factory import create_course
        adapter = create_course(factory)
        env = PackedEnvironment(adapter, source=_native_schema_source(adapter.provenance), policy_dt_s=adapter.policy_dt_s)
        try:
            env.reset()
            if args.command == 'train':
                from supermimic_rl.world_model_v2.online import OnlineConfig, OnlineTrainer
                from supermimic_rl.world_model_v2.teacher import load_native_teacher, native_teacher_decision
                teacher = load_native_teacher(args.teacher, args.teacher_sha256,
                                              device=env.device, native_provenance=adapter.provenance)

                def teacher_decision():
                    return native_teacher_decision(teacher, env)

                config = OnlineConfig.from_dict(json.loads(args.config.read_text()))
                trainer = OnlineTrainer(env, teacher_decision, config, output / 'training',
                    runtime=json.loads(Path(factory['runtime_json']).read_text()), linkage_path=args.linkage,
                    asset_directory=factory['asset_directory'], identified_bundle=factory['identified_bundle'],
                    seed_dataset=args.seed_dataset, world_checkpoint=args.world_checkpoint, resume=args.resume,
                    teacher_source=teacher.source_metadata)
                handoff = False

                def request_checkpoint(signum, frame):
                    nonlocal handoff
                    handoff = True

                # A handoff completes the current cycle and atomically saves
                # model/optimizers/replay before closing the native simulator.
                previous_handler = signal.signal(signal.SIGUSR1, request_checkpoint)
                try:
                    report = trainer.train(checkpoint_requested=lambda: handoff)
                finally:
                    signal.signal(signal.SIGUSR1, previous_handler)
            else:
                from supermimic_rl.world_model_v2.evaluation import evaluate_native
                actor, saved = read_actor(args.checkpoint, env.device)
                if saved['schema_fingerprint'] != env.schema.fingerprint:
                    raise ValueError('Student and native evaluation schema/source differ')

                def progress(record):
                    with (output / 'progress.jsonl').open('a') as stream:
                        stream.write(json.dumps(record) + '\n')

                report = evaluate_native(env, actor, args.steps, on_progress=progress)
                report['checkpoint'] = str(args.checkpoint.resolve())
            (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report), flush=True)
        except BaseException:
            # Isaac shutdown can terminate the interpreter before Python prints
            # the original error. Persist it before closing the simulation app.
            (output / 'failure-traceback.txt').write_text(traceback.format_exc())
            raise
        finally:
            env.close()


if __name__ == '__main__':
    main()
