"""An online pair migrated to the 5x35 chassis student."""

import torch
from supermimic_rl.world_model_v2.policy import EstimatorActor
from support.world_v2.motion_upgrade import fixture as checkpoint_fixture


def migrated_fixture():
    saved, dataset, _, _, _, _, _ = checkpoint_fixture()
    actor = EstimatorActor(estimator_hidden=(8,), actor_hidden=(16,))
    optimizers = {'actor': torch.optim.Adam(actor.decoder.parameters()),
                  'estimator': torch.optim.Adam(actor.estimator.parameters())}
    history = torch.randn(2, 5, 45)
    actor(history).square().sum().backward()
    optimizers['actor'].step()
    actor.estimator_loss(history, torch.zeros(2, 3), torch.zeros(2, 2)).backward()
    optimizers['estimator'].step()
    saved['learner']['actor'] = actor.state_dict()
    saved['learner']['optimizers'].update({name: value.state_dict() for name, value in optimizers.items()})
    saved['config'].update(estimator_hidden=[8], actor_hidden=[16])
    return saved, dataset, actor
