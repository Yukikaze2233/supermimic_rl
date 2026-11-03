"""Explicit motion-to-terrain/source migration; no fabricated task histories."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json

import torch

from ...contracts import StateSchema
from .critic import append_critic_features
from ..imagination import Critic
from ..motion_task import MOTION_TASK_DIM
from ..online import OnlineConfig
from ..records import MODEL_CONTRACT, RecordSpec
from ..stair_source import StairScenario, TOF_RECORD_DTYPES
from ..stair_task import STAIR_RECORD_DTYPES


def upgrade_stair_source(saved, dataset, reference, atlas_raw, *, course_coordinate, world_only_cycles,
                         source_runtime_raw, target_runtime_raw, microbatch_size=512):
    config = OnlineConfig.from_dict(saved['config'])
    if (saved.get('format') != 'supermimic-online-v2' or saved['model_contract'] != MODEL_CONTRACT
            or not config.record_tasks or config.record_stair_source or saved.get('task_norm_pending', False)):
        raise ValueError('Stair expansion requires the calibrated motion-task checkpoint')
    old_schema, target_schema = StateSchema.from_dict(saved['schema']), StateSchema.from_dict(reference['schema'])
    if (old_schema.fingerprint != saved['schema_fingerprint'] or dataset['cycle'] != saved['cycle']
            or dataset['schema_fingerprint'] != old_schema.fingerprint
            or reference.get('format') != 'supermimic-world-v2-dataset-v1'
            or reference['model_contract'] != MODEL_CONTRACT
            or reference['schema_fingerprint'] != target_schema.fingerprint):
        raise ValueError('Stair migration requires authenticated same-generation data contracts')
    old_fields, target_fields = old_schema.to_dict(), target_schema.to_dict()
    old_source, target_source = json.loads(old_fields.pop('source')), json.loads(target_fields.pop('source'))
    if old_fields != target_fields:
        raise ValueError('Stair migration cannot change the physical/observation tensor contract')
    for source, raw in ((old_source, source_runtime_raw), (target_source, target_runtime_raw)):
        if hashlib.sha256(raw).hexdigest() != source['runtime_sha256']:
            raise ValueError('Stair migration runtime pin differs')
    old_runtime, target_runtime = json.loads(source_runtime_raw), json.loads(target_runtime_raw)
    hint_changed = old_runtime != target_runtime
    if hint_changed:
        permitted = deepcopy(old_runtime)
        if permitted['stair_height_assist']['enabled'] is not False:
            raise ValueError('Only explicitly enabling the existing height hint is supported')
        permitted['stair_height_assist']['enabled'] = True
        if target_runtime != permitted:
            raise ValueError('Stair migration cannot alter physics, actuator or other runtime parameters')
    old_source.pop('runtime_sha256')
    target_source.pop('runtime_sha256')
    if old_source != target_source:
        raise ValueError('Stair migration cannot change another source contract')
    if hashlib.sha256(atlas_raw).hexdigest() != old_source['terrain_sha256']:
        raise ValueError('Stair migration atlas differs')
    atlas = json.loads(atlas_raw)
    flat_tiles = [index for index, tile in enumerate(atlas['tiles']) if tile['kind'] == 0]
    storage = dataset['replay']['storage']
    if (len(flat_tiles) != 1 or 'terrain_tile' in storage
            or (storage['reward_route'][..., :2] != 0).any()):
        raise ValueError('Legacy physical rows require provably unique ordinary flat placement')
    scenario = StairScenario.from_dict(reference['stair_source_config'])
    if asdict(scenario.task.height_assist) != target_runtime['stair_height_assist']:
        raise ValueError('Native reference does not implement the requested height hint')
    if type(world_only_cycles) is not int or world_only_cycles < 0:
        raise ValueError('World-only adaptation cycles must be nonnegative')
    next_config = OnlineConfig.from_dict({**asdict(config), 'record_stair_source': True,
        'course_coordinate': course_coordinate, 'policy_start_cycle': saved['cycle'] + world_only_cycles,
        'imagination_microbatch_size': microbatch_size})
    curriculum = saved['environment']['curriculum']
    if not curriculum['real_transitions'] <= course_coordinate < sum(curriculum['config']['stage_transition_budgets'][:5]):
        raise ValueError('This migration places ordinary terrain; jump/recovery need separate producers')
    base = Critic.feature_size(old_schema.goal_dim, storage['sim_context'].shape[-1], task_dim=MOTION_TASK_DIM)
    expanded = Critic.feature_size(old_schema.goal_dim, storage['sim_context'].shape[-1],
                                  task_dim=MOTION_TASK_DIM, terrain=True, stairs=True)
    learner = append_critic_features(saved['learner'], expected_dim=base, extra_dim=expanded - base)
    fields = RecordSpec(old_schema.goal_dim, storage['sim_context'].shape[-1]).fields(
        record_tasks=True, record_terrain=True, record_stairs=True, record_stair_source=True)
    leading = storage['frame'].shape[:2]
    dtypes = {**STAIR_RECORD_DTYPES, **TOF_RECORD_DTYPES}
    augmented = dict(storage)
    for name, shape in fields.items():
        if name not in augmented:
            augmented[name] = torch.zeros(*leading, *shape, dtype=dtypes.get(name, torch.float32))
    augmented['terrain_tile'].fill_(flat_tiles[0])
    # Zero values are storage placeholders only. The explicit validity bit
    # excludes every legacy row from source-aware imagination and calibration.
    augmented['stair_source_valid'].zero_()
    environment = {**saved['environment'], 'curriculum': {**curriculum, 'real_transitions': course_coordinate}}
    receipt = {'migration': 'motion-to-stair-source-v1', 'cycle': saved['cycle'],
        'critic_input_before': base, 'critic_input_after': expanded, 'course_coordinate': course_coordinate,
        'policy_start_cycle': next_config.policy_start_cycle, 'height_hint_enabled': hint_changed,
        'legacy_flat_tile': flat_tiles[0], 'legacy_source_records_valid': 0,
        'old_physical_transitions_retained': leading[0] * leading[1],
        'source_schema': old_schema.fingerprint, 'target_schema': target_schema.fingerprint,
        'preserved': ['model, actor and estimator', 'old critic columns and Adam moments', 'other optimizers',
                      'all update counters and RNG states', 'old physical records, sequence IDs and boundaries'],
        'scope': 'explicit terrain training placement and native hint transfer; not a capability promotion or hardware ToF qualification'}
    upgraded = {**saved, 'schema': target_schema.to_dict(), 'schema_fingerprint': target_schema.fingerprint,
        'config': asdict(next_config), 'learner': learner, 'environment': environment,
        'stair_scenario': scenario.to_dict(), 'critic_norm_from': base,
        'resume_migrations': [*saved.get('resume_migrations', []), receipt]}
    replay = {**dataset, 'schema_fingerprint': target_schema.fingerprint,
              'replay': {**dataset['replay'], 'storage': augmented}}
    return upgraded, replay, receipt
