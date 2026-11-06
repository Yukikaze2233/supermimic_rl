#!/usr/bin/env python3
"""Native V2 online training, cold resume, student evaluation and ONNX export."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import traceback

import torch


def read_actor(path, device):
    from supermimic_rl.world_model_v2.student_transfer import load_student_actor

    return load_student_actor(path, device)


def read_course_migrations(path, schema):
    """Read verified sampling receipts only from this native recovery contract.

    The model file is mapped read-only; loading compatibility never reads or
    copies the online replay. Full learner/replay restore still belongs to the
    trainer, including its exact task-signature checks.
    """
    if path is None:
        return ()
    from supermimic_rl.contracts import StateSchema
    from supermimic_rl.world_model_v2.migrations.course import (
        FORMAT, IDENTITY_KEYS, verify_course_sampling,
    )
    from supermimic_rl.world_model_v2.records import MODEL_CONTRACT

    saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if (not isinstance(saved, dict) or saved.get('format') != 'supermimic-online-v2'
            or saved.get('model_contract') != MODEL_CONTRACT
            or not isinstance(saved.get('schema'), dict)
            or not isinstance(saved.get('learner'), dict) or 'optimizers' not in saved['learner']
            or not isinstance(saved.get('environment'), dict) or not isinstance(saved.get('config'), dict)
            or not isinstance(saved.get('replay_file'), str) or not saved['replay_file']
            or type(saved.get('cycle')) is not int or saved['cycle'] < 0):
        raise ValueError('Course compatibility requires a complete online recovery checkpoint')
    recorded = StateSchema.from_dict(saved['schema'])
    if saved.get('schema_fingerprint') != recorded.fingerprint or recorded.fingerprint != schema.fingerprint:
        raise ValueError('Resume compatibility schema differs from the complete native contract')
    ledger = saved.get('resume_migrations', ())
    if not isinstance(ledger, (tuple, list)):
        raise ValueError('Online migration ledger must be a sequence')
    proofs = []
    for entry in ledger:
        if not isinstance(entry, dict) or not ('course_sampling' in entry or entry.get('migration') == FORMAT):
            continue
        proof = entry.get('course_sampling')
        if (entry.get('migration') != FORMAT or not isinstance(proof, dict)
                or type(entry.get('cycle')) is not int or not 0 <= entry['cycle'] <= saved['cycle']):
            raise ValueError('Unrecognized course sampling migration record')
        try:
            checked = verify_course_sampling(proof['before'], proof['after'],
                source_course=proof['source_course_utf8'].encode('utf-8'),
                target_course=proof['target_course_utf8'].encode('utf-8'))
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError('Incomplete course sampling migration receipt') from error
        if checked != proof:
            raise ValueError('Course sampling receipt differs from its authenticated bytes')
        proofs.append(proof)
    identity = json.loads(recorded.source)
    for proof in reversed(proofs):
        if any(identity.get(key) != proof['after'].get(key) for key in IDENTITY_KEYS):
            raise ValueError('Course sampling ledger does not reach the native course identity')
        identity.update(proof['before'])
    return tuple(proofs)


def exit_native_failure(output, error):
    """Preserve the original failure without entering Kit's default exit-zero close."""
    try:
        try:
            (output / 'failure-traceback.txt').write_text(''.join(traceback.format_exception(error)))
        except OSError as receipt_error:
            error.add_note(f'Could not write failure-traceback.txt: {receipt_error}')
        traceback.print_exception(error, file=sys.stderr)
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except OSError:
                pass
    finally:
        # Kit close/finalizers may call os._exit(0), even while propagating an
        # exception. The failed process releases native resources at exit;
        # successful runs retain the ordinary close path below.
        os._exit(1)


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
    commands.choices['evaluate'].add_argument('--record-dataset', action='store_true',
        help='Save this evaluation\'s complete compact V2 replay for world-model training')
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
                course_migrations = read_course_migrations(args.resume, env.schema)
                teacher = load_native_teacher(args.teacher, args.teacher_sha256,
                    device=env.device, native_provenance=adapter.provenance, course_migrations=course_migrations)

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
                from supermimic_rl.world_model_v2.teacher import file_sha256

                digest = file_sha256(args.checkpoint)
                actor, saved = read_actor(args.checkpoint, env.device)
                if saved['schema_fingerprint'] != env.schema.fingerprint:
                    raise ValueError('Student and native evaluation schema/source differ')

                def progress(record):
                    with (output / 'progress.jsonl').open('a') as stream:
                        stream.write(json.dumps(record) + '\n')

                report = evaluate_native(env, actor, args.steps, on_progress=progress,
                    record_dataset=output / 'replay.pt' if args.record_dataset else None,
                    recording_source={'kind': 'student_native_evaluation',
                        'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': digest,
                        'action_sampling': 'deterministic'})
                if file_sha256(args.checkpoint) != digest:
                    raise ValueError('Student checkpoint changed during evaluation')
                report['checkpoint'] = str(args.checkpoint.resolve())
                report['checkpoint_sha256'] = digest
            (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report), flush=True)
        except BaseException as error:
            exit_native_failure(output, error)
        else:
            env.close()


if __name__ == '__main__':
    main()
