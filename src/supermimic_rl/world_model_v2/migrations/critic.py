"""Preserve a critic's function and Adam history when appending input features."""
from copy import deepcopy

from torch.nn.functional import pad


def append_critic_features(learner, *, expected_dim, extra_dim):
    critic = dict(learner['critic'])
    if (extra_dim < 1 or critic['net.0.weight'].shape[1] != expected_dim
            or critic['norm.mean'].shape != (expected_dim,)):
        raise ValueError('Critic feature expansion contract differs')
    critic['net.0.weight'] = pad(critic['net.0.weight'], (0, extra_dim))
    critic['norm.mean'] = pad(critic['norm.mean'], (0, extra_dim))
    critic['norm.var'] = pad(critic['norm.var'], (0, extra_dim), value=1.)
    optimizer = deepcopy(learner['optimizers']['critic'])
    if len(optimizer['param_groups']) != 1:
        raise ValueError('Expansion requires the existing single-group critic Adam')
    first = optimizer['param_groups'][0]['params'][0]
    for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
        if name in optimizer['state'].get(first, {}):
            moment = optimizer['state'][first][name]
            if moment.shape != learner['critic']['net.0.weight'].shape:
                raise ValueError('Critic optimizer moment shape differs')
            optimizer['state'][first][name] = pad(moment, (0, extra_dim))
    return {**learner, 'critic': critic, 'optimizers': {**learner['optimizers'], 'critic': optimizer}}
