#!/usr/bin/env python3
"""Fit only the V2 world model on a fixed recorded replay, without Isaac or an actor.

An online recovery pair can initialize the world model and its Adam moments.
An explicit model-only initialization can instead fit a new native dataset,
with its own disjoint training and validation environments. Neither updates a policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import time

import torch

from supermimic_rl.contracts import StateSchema
from supermimic_rl.runtime import atomic_save
from supermimic_rl.world_model_v2.execution import WheelExecution
from supermimic_rl.world_model_v2.features import MotionInputs
from supermimic_rl.world_model_v2.fitting import WorldModelFit
from supermimic_rl.world_model_v2.ingest import load_recorded_replay
from supermimic_rl.world_model_v2.leg_linkage import LegLinkage
from supermimic_rl.world_model_v2.model import WorldModelV2
from supermimic_rl.world_model_v2.records import MODEL_CONTRACT
from supermimic_rl.world_model_v2.replay import CompactReplay
from supermimic_rl.world_model_v2.validation import validate

FORMAT = 'supermimic-offline-world-v2'


def online_source(path, *, model_only=False):
    """Select WM/Adam from an online checkpoint, with its replay only when requested."""
    saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if saved.get('format') != 'supermimic-online-v2' or saved.get('model_contract') != MODEL_CONTRACT:
        raise ValueError('World continuation needs a complete current online checkpoint')
    schema = StateSchema.from_dict(saved['schema'])
    if schema.fingerprint != saved['schema_fingerprint']:
        raise ValueError('Online world continuation schema differs')
    replay_path = None if model_only else (path.parent / saved['replay_file']).resolve(strict=True)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    source = {'kind': 'online_model_initialization' if model_only else 'online_pair',
              'checkpoint': str(path.resolve()), 'checkpoint_sha256': digest, 'cycle': saved['cycle'],
              'initial_model_updates': saved['learner']['updates']['model'],
              'initial_model_skipped': saved['learner']['skipped']['model'],
              'schema_fingerprint': schema.fingerprint}
    if replay_path is not None:
        source['replay'] = str(replay_path)
    initial = {'model': saved['learner']['model'], 'optimizer': saved['learner']['optimizers']['model']}
    return source, schema, saved['config'], initial


def paired_replay(source, schema, device):
    dataset = torch.load(source['replay'], map_location='cpu', weights_only=True, mmap=True)
    if (dataset['cycle'] != source['cycle'] or dataset['schema_fingerprint'] != schema.fingerprint
            or schema.fingerprint != source['schema_fingerprint']):
        raise ValueError('World continuation checkpoint and replay generation differ')
    return CompactReplay.from_state_dict(dataset['replay'], device,
                                         read_only=torch.device(device).type == 'cpu')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('replay', type=Path, nargs='?', help='Packed or compact native dataset for fitting')
    parser.add_argument('--online-checkpoint', type=Path, help='Initialize only WM/Adam from its paired replay')
    parser.add_argument('--initial-online-checkpoint', type=Path,
                        help='Initialize WM/Adam on the positional native dataset, using that dataset\'s own split')
    parser.add_argument('--linkage', type=Path, default=Path('configs/linkage/leg_linkage_e726d23c.json'))
    parser.add_argument('--terrain-atlas', type=Path, help='Pinned native atlas; required for nonflat online data')
    parser.add_argument('--identified-bundle', type=Path, default=Path('configs/identified/20261031'))
    parser.add_argument('--hidden', type=int, nargs='+')
    parser.add_argument('--window', type=int)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--iterations', type=int, default=2000)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--validation-fraction', type=float)
    parser.add_argument('--contact-event-fraction', type=float, default=None,
                        help='Candidate fraction of fitting windows containing >=10 ms double-wheel-off; default 0')
    parser.add_argument('--wheel-contact-weight', type=float, default=None,
                        help='Extra separately averaged continuous wheel-load MSE; default 0, 1 gives equal group coefficient')
    parser.add_argument('--validation-horizon', type=int, default=16)
    parser.add_argument('--validation-batch', type=int, default=1024)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--replay-device', choices=('cpu', 'cuda'), default=None)
    parser.add_argument('--allow-tf32', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path, help='Resume the independently saved WM-only checkpoint')
    parser.add_argument('--checkpoint-interval', type=int, default=250)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.replay is not None and args.online_checkpoint is not None:
        parser.error('Choose a recorded dataset or an online checkpoint')
    if args.initial_online_checkpoint is not None and (
            args.replay is None or args.online_checkpoint is not None or args.resume is not None):
        parser.error('--initial-online-checkpoint needs a new recorded dataset; use --resume alone afterwards')
    if args.resume is None and args.replay is None and args.online_checkpoint is None:
        parser.error('A replay, --online-checkpoint or --resume is required')
    if min(args.iterations, args.checkpoint_interval, args.validation_horizon, args.validation_batch) < 1:
        parser.error('Iteration, checkpoint and validation settings must be positive')
    device = torch.device(args.device)
    resumed = torch.load(args.resume, map_location='cpu', weights_only=True, mmap=True) if args.resume else None
    if resumed is not None and (resumed.get('format') != FORMAT
                               or resumed['model_contract']['version'] != MODEL_CONTRACT):
        raise ValueError('Offline resume world-model contract differs')
    old_configuration = resumed['configuration'] if resumed is not None else {}
    initial, source_config = None, {}
    if resumed is not None and resumed.get('source', {}).get('kind') == 'online_pair':
        source = resumed['source']
        schema = StateSchema.from_dict(resumed['schema'])
        if args.online_checkpoint is not None and str(args.online_checkpoint.resolve()) != source['checkpoint']:
            raise ValueError('Resume online source differs')
        if args.replay is not None:
            raise ValueError('Online-pair resume cannot switch to a different dataset')
    elif args.online_checkpoint is not None:
        if resumed is not None:
            raise ValueError('Offline dataset resume cannot change its source to an online pair')
        source, schema, source_config, initial = online_source(args.online_checkpoint)
    else:
        path = args.replay or Path(old_configuration['replay'])
        source = {**(resumed.get('source', {}) if resumed is not None else {}),
                  'kind': 'recorded_dataset', 'replay': str(path.resolve())}
        schema = None
        if args.initial_online_checkpoint is not None:
            initialization, _, source_config, initial = online_source(args.initial_online_checkpoint, model_only=True)
            source['initialization'] = initialization
        if 'initialization' in source:
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if 'replay_sha256' in source and source['replay_sha256'] != digest:
                raise ValueError('External native dataset changed since initialization')
            source['replay_sha256'] = digest

    def setting(name, default):
        value = getattr(args, name)
        return old_configuration.get(name, default) if value is None else value

    hidden = list(args.hidden or (resumed['model_contract']['hidden'] if resumed is not None else
                                 source_config.get('model_hidden', [2048, 2048, 2048])))
    batch = setting('batch', source_config.get('model_batch_size', 2048))
    window = setting('window', source_config.get('fit_horizon', 8))
    learning = source_config.get('learning', {})
    learning_rate = setting('learning_rate', learning.get('model_learning_rate', 2e-4))
    fraction = setting('validation_fraction', source_config.get('validation_fraction', .125))
    event_fraction = setting('contact_event_fraction', 0.)
    wheel_contact_weight = setting('wheel_contact_weight', 0.)
    seed = setting('seed', 0)
    replay_device = torch.device(setting('replay_device', 'cpu'))
    tf32 = setting('allow_tf32', source_config.get('allow_tf32', False) if 'initialization' in source
                   else False if source['kind'] == 'online_pair' else True)
    clip = old_configuration.get('gradient_clip', learning.get('gradient_clip', 1.))
    if min(batch, window, *hidden) < 1 or not 0 < fraction < 1 or learning_rate <= 0:
        raise ValueError('Positive batch/window/network/rate and a disjoint validation fraction are required')
    if not math.isfinite(event_fraction) or not 0 <= event_fraction < 1:
        raise ValueError('Contact event fraction must be finite and in [0, 1)')
    if not math.isfinite(wheel_contact_weight) or wheel_contact_weight < 0:
        raise ValueError('Wheel contact weight must be finite and nonnegative')
    if source_config:
        if source['kind'] == 'online_pair' and fraction != source_config.get('validation_fraction', .125):
            raise ValueError('Online continuation must preserve the original held-out environment IDs')
        if hidden != list(source_config['model_hidden']) or learning_rate != learning.get('model_learning_rate', 2e-4):
            raise ValueError('Online initialization must preserve the WM architecture and Adam learning rate')
        if source_config.get('record_stair_source') and args.terrain_atlas is None:
            raise ValueError('Nonflat online world fitting requires --terrain-atlas')
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = tf32
    if source['kind'] == 'online_pair':
        replay = paired_replay(source, schema, replay_device)
    else:
        replay, schema = load_recorded_replay(Path(source['replay']), replay_device,
                                              record_terrain=args.terrain_atlas is not None)
    if ('initialization' in source
            and schema.fingerprint != source['initialization']['schema_fingerprint']):
        raise ValueError('External native dataset and initial world-model schemas differ')
    terrain = None
    if args.terrain_atlas is not None:
        from supermimic_rl.world_model_v2.terrain import AtlasTerrain
        terrain = AtlasTerrain.load(args.terrain_atlas, expected_sha256=json.loads(schema.source)['terrain_sha256'],
                                    device=device)
    if 'terrain_tile' in replay.data and terrain is None:
        raise ValueError('Recorded terrain routes require --terrain-atlas; start scans cannot stand in for geometry')
    if event_fraction and (schema.policy_dt_s != .02 or schema.substeps_per_step != 4):
        raise ValueError('Contact event sampling requires recorded 200 Hz physics and 50 Hz policy steps')
    held = max(1, round(replay.num_envs * fraction))
    if held >= replay.num_envs:
        raise ValueError('Replay needs distinct training and held-out environment rows')
    train_ids = list(range(replay.num_envs - held))
    valid_ids = list(range(replay.num_envs - held, replay.num_envs))
    data_source = json.loads(schema.source)
    model = WorldModelV2(MotionInputs(LegLinkage.load(args.linkage,
        expected_asset_manifest_sha256=data_source.get('asset_manifest_sha256'))),
        WheelExecution.from_bundle(args.identified_bundle, schema.policy_dt_s / schema.substeps_per_step),
        hidden=tuple(hidden), terrain=terrain,
        disturbances=any(field.name == 'velocity_push_world' for field in schema.substep_fields)).to(device)
    fit = WorldModelFit(model, wheel_contact_weight=wheel_contact_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    contract = {'version': MODEL_CONTRACT, 'input_dim': model.motion.input_norm.mean.numel(),
                'output_dim': model.motion.output_dim, 'ticks': 4, 'hidden': hidden,
                'schema_fingerprint': schema.fingerprint}
    configuration = {'replay': source['replay'], 'window': window, 'batch': batch,
        'learning_rate': learning_rate, 'validation_fraction': fraction, 'seed': seed,
        'replay_device': str(replay_device), 'allow_tf32': tf32, 'gradient_clip': clip,
        'contact_event_fraction': event_fraction,
        'wheel_contact_weight': wheel_contact_weight,
        'normalization': ('update_on_uniform_prefix_only' if event_fraction else 'update_on_real_training_windows'),
        'train_env_ids': train_ids, 'validation_env_ids': valid_ids}
    if terrain is not None:
        configuration['terrain_atlas_sha256'] = terrain.manifest_sha256
    generator = torch.Generator(device=replay_device).manual_seed(seed)
    history, skipped, first_iteration, elapsed_before = [], 0, 1, 0.
    resume_changes = []
    if resumed is not None:
        expected_configuration = dict(resumed['configuration'])
        # Existing checkpoints used the same uniform path and had no event setting.
        expected_configuration.setdefault('contact_event_fraction', 0.)
        expected_configuration.setdefault('wheel_contact_weight', 0.)
        old_batch = expected_configuration['batch']
        if args.batch is not None:
            expected_configuration['batch'] = batch
        if resumed['model_contract'] != contract or expected_configuration != configuration:
            raise ValueError('Offline resume model/data/training contract differs')
        resume_changes = list(resumed.get('resume_changes', []))
        if old_batch != batch:
            resume_changes.append({'change': 'world_model_batch_size', 'before': old_batch, 'after': batch,
                'after_iteration': resumed['iteration'], 'effective_iteration': resumed['iteration'] + 1,
                'preserved': ['model', 'Adam state', 'sampler RNG state', 'source data', 'held-out IDs', 'learning rate'],
                'scope': 'explicit new batch size changes samples per update and subsequent draws; not original-training equivalence'})
        initial = resumed
        generator.set_state(resumed['generator'].cpu())
        history, skipped = resumed['history'], resumed['skipped']
        first_iteration, elapsed_before = resumed['iteration'] + 1, resumed['elapsed_s']
        if args.iterations < first_iteration:
            raise ValueError('Resume budget must exceed the saved iteration')
    if initial is not None:
        model.load_state_dict(initial['model'])
        optimizer.load_state_dict(initial['optimizer'])
    del initial, resumed
    candidates = replay.window_candidates(window, train_ids)
    if not candidates.numel():
        raise ValueError(f'No complete eligible training windows of length {window}')
    events = replay.contact_event_window_candidates(window, train_ids) if event_fraction else None
    field_names = fit.RECORD_FIELDS + tuple(name for name in fit.OPTIONAL_RECORD_FIELDS if name in replay.data)
    staging = None
    if replay.device.type == 'cpu' and device.type == 'cuda':
        staging = {name: torch.empty(batch, window, *replay.data[name].shape[2:],
            dtype=replay.data[name].dtype, pin_memory=True) for name in field_names}

    def sample():
        recorded, uniform_count = replay.sample_contact_windows(batch, window, generator=generator,
            env_ids=train_ids, contact_event_fraction=event_fraction, field_names=field_names, out=staging)
        moved = {name: value.to(device, non_blocking=staging is not None) for name, value in recorded.items()}
        if staging is not None:
            # One wait after all copies instead of a wait per field. It also
            # makes sample_seconds include transfer and permits buffer reuse.
            torch.cuda.current_stream(device).synchronize()
        return moved, uniform_count

    started, stop_requested = time.perf_counter(), False
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    def request_checkpoint(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    previous_handler = signal.signal(signal.SIGUSR1, request_checkpoint)
    scope = 'WM-only continued fitting of fixed recorded data; no Isaac, actor, critic or new PPO-data ingestion'
    if 'initialization' in source:
        scope = ('WM-only warm-started fitting on a new fixed native dataset; no Isaac, actor or critic updates. '
                 'Validation rows belong to this dataset, not the original online replay.')
    initial_model_updates = source.get('initialization', source).get('initial_model_updates', 0)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({'phase': 'initialized', 'source': source, 'replay_device': str(replay_device),
        'model_device': str(device), 'window': window, 'batch': batch, 'eligible_training_windows': candidates.numel(),
        'train_envs': len(train_ids), 'validation_envs': len(valid_ids), 'scope': scope,
        'contact_event_fraction': event_fraction, 'eligible_contact_event_windows': 0 if events is None else events.numel(),
        'wheel_contact_weight': wheel_contact_weight,
        'wheel_contact_scope': 'continuous native wheel support N; same output std/error-floor scales; mean over two wheels, four ticks and the fitting horizon',
        'contact_event_scope': 'candidate oversampling of native wheel support <=2 N for two consecutive 5 ms ticks; not jump qualification',
        'normalization': configuration['normalization'],
        'resume_changes': resume_changes,
        'sample_fields': field_names, 'pinned_sample_buffers': staging is not None,
        'sample_bytes': sum(batch * window * replay.data[name][0, 0].numel()
                            * replay.data[name].element_size() for name in field_names)}), flush=True)
    try:
        for iteration in range(first_iteration, args.iterations + 1):
            tick = time.perf_counter()
            windows, uniform_count = sample()
            sampled = time.perf_counter()
            fit.observe({name: value[:uniform_count] for name, value in windows.items()})
            terms = fit.terms(windows)
            loss = terms.state + terms.ticks
            if wheel_contact_weight:
                loss = loss + wheel_contact_weight * terms.wheel_contact
            optimizer.zero_grad(set_to_none=True)
            updated = bool(torch.isfinite(loss))
            if updated:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                updated = bool(torch.isfinite(norm))
            if updated:
                optimizer.step()
            else:
                optimizer.zero_grad(set_to_none=True)
                skipped += 1
            save = iteration % args.checkpoint_interval == 0 or iteration == args.iterations or stop_requested
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            record = {'iteration': iteration, 'batch': batch, 'state': float(terms.state.detach()),
                'uniform_windows': uniform_count, 'contact_event_windows': batch - uniform_count,
                'ticks': float(terms.ticks.detach()),
                'wheel_contact': None if terms.wheel_contact is None else float(terms.wheel_contact.detach()),
                'loss': float(loss.detach()),
                'terrain_truncated_fraction': float(terms.terrain_truncated_fraction),
                'per_step_state': [value if math.isfinite(value) else None
                                   for value in terms.per_step_state.mean(0).tolist()], 'skipped': skipped,
                'model_updates': initial_model_updates + iteration - skipped,
                'seconds': elapsed_before + time.perf_counter() - started,
                'iteration_seconds': time.perf_counter() - tick, 'sample_seconds': sampled - tick,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0}
            record = {name: value if not isinstance(value, float) or math.isfinite(value) else None
                      for name, value in record.items()}
            history.append(record)
            print(json.dumps({key: value for key, value in record.items() if key != 'per_step_state'}), flush=True)
            if args.output and save:
                payload = {'format': FORMAT, 'model_contract': contract, 'schema': schema.to_dict(),
                    'configuration': configuration, 'source': source, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'generator': generator.get_state(),
                    'iteration': iteration, 'history': history, 'skipped': skipped, 'scope': scope,
                    'resume_changes': resume_changes,
                    'elapsed_s': elapsed_before + time.perf_counter() - started}
                atomic_save(payload, args.output.with_suffix('.checkpoint.pt'))
                args.output.write_text(json.dumps({'status': 'checkpointed' if stop_requested else 'fitting',
                    'iteration': iteration, 'model_contract': contract, 'source': source, 'scope': scope,
                    'configuration': configuration, 'history': history, 'resume_changes': resume_changes},
                    indent=2) + '\n')
            if stop_requested:
                break
    finally:
        signal.signal(signal.SIGUSR1, previous_handler)
    if stop_requested:
        return
    validation_generator = torch.Generator(device=replay_device).manual_seed(seed + 1)
    report = {'history': history, 'validation': validate(fit, replay, valid_ids, args.validation_horizon,
        args.validation_batch, validation_generator, device), 'train_envs': len(train_ids),
        'validation_envs': len(valid_ids), 'skipped_updates': skipped, 'model_contract': contract,
        'configuration': configuration, 'source': source, 'scope': scope, 'status': 'complete',
        'resume_changes': resume_changes}
    print(json.dumps(report['validation'], indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        atomic_save(model.state_dict(), args.output.with_suffix('.pt'))


if __name__ == '__main__':
    main()
