"""Explicit new policy branch removing gimbal inputs while preserving physical data."""
from copy import deepcopy

import torch

from ...contracts import StateSchema
from ..policy import CHASSIS_FRAME_DIM, ESTIMATE_DIM, FRAME_DIM
from ..records import MODEL_CONTRACT


def restrict_actor_to_chassis(saved, dataset, *, replay_device=None,
                             imagination_microbatch_size=None, sampling_seed=None):
    """Initialize 5x35 from per-frame chassis columns of the legacy 5x45 MLP.

    This changes the policy's function. Old skill results are ancestor results,
    not qualification of this branch. Neither the simulator nor world-model
    representation loses the independently controlled gimbal.
    """
    if (saved.get('format') != 'supermimic-online-v2' or saved['model_contract'] != MODEL_CONTRACT
            or saved['config'].get('actor_frame_dim', FRAME_DIM) != FRAME_DIM):
        raise ValueError('Chassis actor initialization requires a complete legacy formal45 checkpoint')
    if replay_device not in (None, 'learner', 'cpu'):
        raise ValueError('Replay device must be learner or cpu')
    if imagination_microbatch_size is not None and (
            type(imagination_microbatch_size) is not int or imagination_microbatch_size < 1):
        raise ValueError('Policy microbatch must be a positive integer')
    previous_storage = saved['config'].get('replay_device', 'learner')
    change_storage = replay_device is not None and replay_device != previous_storage
    if change_storage and replay_device != 'cpu':
        raise ValueError('This branch migration only supports an explicit move to CPU replay')
    if change_storage:
        if type(sampling_seed) is not int or not 0 <= sampling_seed < 2**63 - 1:
            raise ValueError('CPU replay migration requires an explicit sampling seed')
    elif sampling_seed is not None:
        raise ValueError('Sampling seed only applies to an explicit CPU replay migration')
    schema = StateSchema.from_dict(saved['schema'])
    if (schema.fingerprint != saved['schema_fingerprint'] or dataset['schema_fingerprint'] != schema.fingerprint
            or dataset['cycle'] != saved['cycle'] or schema.observation_dim != FRAME_DIM
            or dataset['replay']['storage']['frame'].shape[-1] != FRAME_DIM):
        raise ValueError('Chassis actor initialization requires the paired unchanged formal45 replay')
    learner, actor = saved['learner'], dict(saved['learner']['actor'])
    first = actor['estimator.0.weight']
    if first.ndim != 2 or first.shape[1] != 5 * FRAME_DIM:
        raise ValueError('Chassis actor initialization requires the existing five-frame MLP estimator')
    flat = first.shape[1]
    decoder = actor['decoder.0.weight']
    if decoder.ndim != 2 or decoder.shape[1] != flat + ESTIMATE_DIM:
        raise ValueError('Chassis actor decoder must append the same five estimator outputs')
    columns = torch.arange(flat, device=first.device).view(5, FRAME_DIM)[:, :CHASSIS_FRAME_DIM].flatten()
    actor['estimator.0.weight'] = first.index_select(1, columns)
    decoder_columns = torch.cat((columns.to(decoder.device),
                                 torch.arange(flat, flat + ESTIMATE_DIM, device=decoder.device)))
    actor['decoder.0.weight'] = decoder.index_select(1, decoder_columns)
    optimizers = dict(learner['optimizers'])
    for name in ('actor', 'estimator'):
        optimizers[name] = {**deepcopy(optimizers[name]), 'state': {}}
    receipt = {
        'migration': 'formal45-to-chassis35-actor-branch-v1', 'cycle': saved['cycle'],
        'actor_frame_before': FRAME_DIM, 'actor_frame_after': CHASSIS_FRAME_DIM, 'history_frames': 5,
        'initialization': 'copy the first 35 columns of EACH frame; retain decoder estimator columns',
        'optimizer_states_reset': ['actor', 'estimator'],
        'preserved': ['world model including gimbal', 'critic', 'model and critic optimizers',
                      'formal45 physical replay', 'RNGs', 'teacher schedule', 'cumulative ancestor update counts'],
        'ancestor_updates': dict(learner['updates']),
        'policy_equivalent': False,
        'scope': 'new chassis-only policy initialization; ancestor skill evaluation does not qualify this branch',
    }
    upgraded = {**saved, 'config': {**saved['config'], 'actor_frame_dim': CHASSIS_FRAME_DIM},
        'learner': {**learner, 'actor': actor, 'optimizers': optimizers},
        'resume_migrations': [*saved.get('resume_migrations', []), receipt]}
    if imagination_microbatch_size is not None:
        receipt['policy_microbatch'] = {
            'before': saved['config'].get('imagination_microbatch_size', 512),
            'after': imagination_microbatch_size,
            'total_batch_unchanged': saved['config'].get('imagination_batch_size', 4096)}
        upgraded['config']['imagination_microbatch_size'] = imagination_microbatch_size
    if replay_device is not None:
        upgraded['config']['replay_device'] = replay_device
    if change_storage:
        upgraded.update(
            sample_rng=torch.Generator(device='cpu').manual_seed(sampling_seed).get_state(),
            validation_rng=torch.Generator(device='cpu').manual_seed(sampling_seed + 1).get_state(),
            sample_rng_device='cpu', validation_rng_device='cpu')
        receipt['replay_storage'] = {
            'before': previous_storage, 'after': 'cpu',
            'sample_seed': sampling_seed, 'validation_seed': sampling_seed + 1,
            'replay_bytes_unchanged': True,
            'rng_migration': 'new independent CPU generators; CUDA states are not reinterpreted'}
        receipt['preserved'].remove('RNGs')
        receipt['preserved'].append('collection, mixing and global RNGs; sampling RNGs explicitly replaced')
    return upgraded, dataset, receipt


def inject_fitted_world(saved, dataset, world, *, source_checkpoint_sha256):
    """Join WM-only fitting back to its own online ancestor, as an explicit branch.

    Policy/critic learning and online random streams stay at the ancestor.
    The offline sampler is a separate stream, not an online continuation.
    """
    if (saved.get('format') != 'supermimic-online-v2' or saved.get('model_contract') != MODEL_CONTRACT
            or world.get('format') != 'supermimic-offline-world-v2'):
        raise ValueError('World injection requires full online and WM-only checkpoints')
    source, contract = world['source'], world['model_contract']
    learner, configuration = saved['learner'], world['configuration']
    schema = StateSchema.from_dict(saved['schema'])
    if (source.get('kind') != 'online_pair'
            or not source_checkpoint_sha256
            or source.get('checkpoint_sha256') != source_checkpoint_sha256
            or source['cycle'] != saved['cycle'] or dataset['cycle'] != saved['cycle']
            or source['initial_model_updates'] != learner['updates']['model']
            or source['initial_model_skipped'] != learner['skipped']['model']):
        raise ValueError('Fitted world must descend from this exact online checkpoint and model counters')
    if (schema.fingerprint != saved['schema_fingerprint']
            or any(value != schema.fingerprint for value in (
                source['schema_fingerprint'], contract['schema_fingerprint'], dataset['schema_fingerprint'],
                StateSchema.from_dict(world['schema']).fingerprint))
            or contract['version'] != MODEL_CONTRACT
            or tuple(contract['hidden']) != tuple(saved['config']['model_hidden'])
            or contract['ticks'] != schema.substeps_per_step
            or contract['input_dim'] != learner['model']['motion.input_norm.mean'].numel()
            or contract['output_dim'] != learner['model']['motion.output_norm.mean'].numel()
            or world['model'].keys() != learner['model'].keys()
            or any(value.shape != learner['model'][name].shape for name, value in world['model'].items())):
        raise ValueError('Fitted world physical schema or network contract differs')
    count = dataset['replay']['num_envs']
    fraction = saved['config'].get('validation_fraction', .125)
    held = max(1, round(count * fraction))
    if (not 0 < held < count or configuration['validation_fraction'] != fraction
            or configuration['train_env_ids'] != list(range(count - held))
            or configuration['validation_env_ids'] != list(range(count - held, count))
            or configuration['replay'] != source['replay']):
        raise ValueError('Fitted world must retain the paired replay and held-out environment IDs')
    learning_rate = saved['config'].get('learning', {}).get('model_learning_rate', 2e-4)
    if (configuration['window'] != saved['config'].get('fit_horizon', 8)
            or configuration['learning_rate'] != learning_rate
            or world['optimizer']['param_groups'] != learner['optimizers']['model']['param_groups']):
        raise ValueError('Fitted world horizon or Adam contract differs')
    iteration, skipped = world['iteration'], world['skipped']
    if type(iteration) is not int or type(skipped) is not int or not 0 <= skipped <= iteration:
        raise ValueError('Invalid fitted-world update counters')
    updates = {**learner['updates'], 'model': learner['updates']['model'] + iteration - skipped}
    skipped_updates = {**learner['skipped'], 'model': learner['skipped']['model'] + skipped}
    receipt = {
        'migration': 'same-source-fitted-world-online-branch-v1', 'cycle': saved['cycle'],
        'source': deepcopy(source), 'ancestor_updates': dict(learner['updates']),
        'ancestor_skipped': dict(learner['skipped']),
        'offline_iterations': iteration, 'offline_skipped': skipped,
        'result_updates': updates.copy(), 'result_skipped': skipped_updates.copy(),
        'offline_configuration': deepcopy(configuration),
        'offline_resume_changes': deepcopy(world.get('resume_changes', [])),
        'replaced': ['world model including normalization', 'world model Adam state', 'model update/skip counters'],
        'preserved': ['actor and estimator', 'critic and critic Adam', 'policy optimizer state',
                      'paired physical replay', 'held-out environment IDs', 'online RNGs',
                      'teacher schedule', 'cycle and collected transitions'],
        'rng': 'offline model sampler is not substituted for the online shared sampler',
        'scope': 'new online branch from fixed-replay WM fitting; no new PPO data or policy qualification',
    }
    upgraded = {**saved, 'learner': {**learner, 'model': world['model'],
        'optimizers': {**learner['optimizers'], 'model': world['optimizer']},
        'updates': updates, 'skipped': skipped_updates},
        'resume_migrations': [*saved.get('resume_migrations', []), receipt]}
    return upgraded, receipt
