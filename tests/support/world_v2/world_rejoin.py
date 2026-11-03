"""A matched online checkpoint and replay pair for world-model rejoin tests."""

import hashlib
import torch
from supermimic_rl.world_model_v2.records import MODEL_CONTRACT
from support.world_v2.chassis_actor import migrated_fixture
from support.world_v2.offline_continuation import fixture as world_fixture


def pair(tmp_path):
    checkpoint, model, _ = world_fixture(tmp_path)
    saved = torch.load(checkpoint, weights_only=True)
    legacy, _, _ = migrated_fixture()
    for name in ('actor', 'critic'):
        saved['learner'][name] = legacy['learner'][name]
    for name in ('actor', 'critic', 'estimator'):
        saved['learner']['optimizers'][name] = legacy['learner']['optimizers'][name]
    saved['learner']['updates'].update(actor=768, critic=15360, estimator=1000)
    saved['learner']['skipped'].update(actor=3, critic=2, estimator=1)
    saved['config'].update(actor_hidden=[16], estimator_hidden=[8])
    saved['sample_rng'] = torch.Generator().manual_seed(2).get_state()
    torch.save(saved, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    replay_path = tmp_path / saved['replay_file']
    dataset = torch.load(replay_path, weights_only=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    optimizer.load_state_dict(saved['learner']['optimizers']['model'])
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        sum(parameter.square().mean() for parameter in model.parameters()).backward()
        optimizer.step()
    model.motion.input_norm.update(torch.randn(3, model.motion.input_norm.mean.numel()))
    world = {
        'format': 'supermimic-offline-world-v2', 'model': model.state_dict(),
        'optimizer': optimizer.state_dict(), 'iteration': 5, 'skipped': 1,
        'schema': saved['schema'], 'generator': torch.Generator().manual_seed(999).get_state(),
        'model_contract': {'version': MODEL_CONTRACT, 'hidden': [16], 'ticks': 4,
            'input_dim': model.motion.input_norm.mean.numel(), 'output_dim': model.motion.output_dim,
            'schema_fingerprint': saved['schema_fingerprint']},
        'source': {'kind': 'online_pair', 'checkpoint': str(checkpoint), 'checkpoint_sha256': digest,
            'replay': str(replay_path), 'cycle': saved['cycle'], 'initial_model_updates': 42,
            'initial_model_skipped': 2, 'schema_fingerprint': saved['schema_fingerprint']},
        'configuration': {'replay': str(replay_path), 'window': 8, 'batch': 16,
            'learning_rate': 2e-4, 'validation_fraction': .5,
            'train_env_ids': [0], 'validation_env_ids': [1]},
        'resume_changes': [{'change': 'world_model_batch_size', 'before': 4, 'after': 16}],
    }
    return checkpoint, saved, dataset, world, model
