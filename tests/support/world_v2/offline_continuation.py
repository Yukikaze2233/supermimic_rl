"""An offline world-model fit and its continuation inputs."""

import json
import torch
from supermimic_rl.contracts import Field, StateSchema
from supermimic_rl.world_model_v2.execution import WheelExecution
from supermimic_rl.world_model_v2.features import MotionInputs
from supermimic_rl.world_model_v2.fitting import WorldModelFit
from supermimic_rl.world_model_v2.leg_linkage import LegLinkage
from supermimic_rl.world_model_v2.model import WorldModelV2
from supermimic_rl.world_model_v2.physical import PhysicalState
from supermimic_rl.world_model_v2.records import MODEL_CONTRACT, RecordSpec, encode_physical
from supermimic_rl.world_model_v2.replay import CompactReplay
from support import REPO_ROOT


ROOT = REPO_ROOT


def fixture(tmp_path):
    linkage = LegLinkage.load(ROOT / 'configs/linkage/leg_linkage_e726d23c.json')
    wheels = WheelExecution.from_bundle(ROOT / 'configs/identified/20261031', .005)
    model = WorldModelV2(MotionInputs(linkage), wheels, hidden=(16,))
    WorldModelFit(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    sum(parameter.square().mean() for parameter in model.parameters()).backward()
    optimizer.step()
    schema = StateSchema((Field('fixture', (1,)),), (), goal_dim=16, observation_dim=45,
        substep_fields=(Field('fixture_tick', (1,)),), substeps_per_step=4,
        source=json.dumps({'asset_manifest_sha256': linkage.provenance['asset_manifest_sha256']}))
    replay = CompactReplay(RecordSpec(16, 65).fields(), 2, 10, 'cpu')
    q = torch.zeros(2, 6)
    for name, primary, secondary in (('left', 0, 1), ('right', 2, 3)):
        side = linkage.sides[name]
        q[:, primary] = side.reference_primary_rad
        q[:, secondary] = side.reference_primary_rad + (side.phi_min + side.phi_max) / 2
    state = PhysicalState(torch.tensor([[0., 0., .35], [0., 0., .35]]), torch.eye(3).repeat(2, 1, 1),
                          torch.zeros(2, 3), torch.zeros(2, 3), q, torch.zeros(2, 8))
    record = {name: torch.zeros(2, *shape) for name, shape in RecordSpec(16, 65).fields().items()}
    record['physical'] = record['end_physical'] = encode_physical(state)
    record['height'].fill_(.35)
    record['end_height'].fill_(.35)
    record['scan'].fill_(-.35)
    record['motor_strength'].fill_(1.)
    record['action'][1, 0] = 17.  # Held-out rows are distinguishable at the fit interface.
    boundary = {name: torch.ones(2, dtype=torch.bool) for name in ('valid', 'policy_owned', 'end_policy_owned')}
    boundary.update({name: torch.zeros(2, dtype=torch.bool) for name in ('terminated', 'truncated')})
    boundary.update({name: torch.zeros(2, dtype=torch.long) for name in
        ('start_episode_id', 'end_episode_id', 'start_control_epoch', 'end_control_epoch')})
    for _ in range(10):
        replay.add(record, boundary)
    saved = {'format': 'supermimic-online-v2', 'model_contract': MODEL_CONTRACT, 'cycle': 78,
        'schema': schema.to_dict(), 'schema_fingerprint': schema.fingerprint, 'replay_file': 'replay-a.pt',
        'config': {'model_hidden': [16], 'model_batch_size': 4, 'fit_horizon': 8,
                   'validation_fraction': .5, 'learning': {'model_learning_rate': 2e-4}},
        'learner': {'model': model.state_dict(), 'optimizers': {'model': optimizer.state_dict()},
                    'updates': {'model': 42}, 'skipped': {'model': 2}}}
    checkpoint = tmp_path / 'online.pt'
    torch.save(saved, checkpoint)
    torch.save({'cycle': 78, 'schema_fingerprint': schema.fingerprint, 'replay': replay.state_dict()},
               tmp_path / 'replay-a.pt')
    return checkpoint, model, replay
