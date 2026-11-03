"""A foundation online checkpoint and its motion-task migration."""

import torch
from supermimic_rl.contracts import Field, StateSchema
from supermimic_rl.world_model_v2.imagination import Critic
from supermimic_rl.world_model_v2.records import MODEL_CONTRACT, RecordSpec
from supermimic_rl.world_model_v2.replay import CompactReplay


def fixture():
    schema = StateSchema((Field('fixture', (1,)),), (), goal_dim=16, observation_dim=45)
    base = Critic.feature_size(16, 65)
    critic = Critic(base, (16, 8))
    optimizer = torch.optim.Adam(critic.parameters(), lr=2e-4)
    x = torch.randn(32, base)
    critic.norm.update(x)
    critic(x).square().mean().backward()
    optimizer.step()
    replay = CompactReplay(RecordSpec(16, 65).fields(), 2, 8, 'cpu')
    record = {name: torch.zeros(2, *shape) for name, shape in RecordSpec(16, 65).fields().items()}
    record['physical'][:, 6] = record['end_physical'][:, 6] = 1.
    record['goal'][:, 4] = 1.
    record['horizon'][:, 0] = 1.
    record['reward_context_valid'][:] = record['failure_conditions_valid'][:] = 1.
    boundary = {key: torch.ones(2, dtype=torch.bool) for key in ('valid', 'policy_owned', 'end_policy_owned')}
    boundary.update({key: torch.zeros(2, dtype=torch.bool) for key in ('terminated', 'truncated')})
    boundary.update({key: torch.zeros(2, dtype=torch.long) for key in
        ('start_episode_id', 'end_episode_id', 'start_control_epoch', 'end_control_epoch')})
    for _ in range(3):
        replay.add(record, boundary)
    saved = {'format': 'supermimic-online-v2', 'model_contract': MODEL_CONTRACT, 'cycle': 32,
        'schema': schema.to_dict(), 'schema_fingerprint': schema.fingerprint, 'config': {'record_tasks': False},
        'learner': {'critic': critic.state_dict(), 'actor': {'fixture': torch.rand(3)},
                    'model': {'fixture': torch.rand(3)}, 'updates': {'actor': 768},
                    'optimizers': {'critic': optimizer.state_dict(), 'actor': {}, 'model': {}, 'estimator': {}}},
        'sample_rng': torch.Generator().get_state()}
    dataset = {'cycle': 32, 'schema_fingerprint': schema.fingerprint, 'replay': replay.state_dict()}
    return saved, dataset, critic, optimizer, x, record, boundary
