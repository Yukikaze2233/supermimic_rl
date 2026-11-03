"""Explicit gate expansion; legacy dynamics survive, unknown task history does not."""
from dataclasses import asdict, replace
import json

import torch

from ...contracts import StateSchema
from ..imagination import Critic
from ..jump_task import (
    JUMP_TRIGGER_FEATURE_DIM, JUMP_TRIGGER_RECORD_DTYPES, JUMP_TRIGGER_RECORD_FIELDS,
    JUMP_TRIGGER_RECORD_VERSION, JumpTaskConfig,
)
from ..motion_task import MOTION_TASK_DIM
from ..online import OnlineConfig
from ..records import MODEL_CONTRACT
from .critic import append_critic_features


def upgrade_jump_trigger(saved, dataset, reference, *, world_only_cycles=0):
    """Append gate records/features using a real native target producer.

    An old episode-start flag does not prove a reset gate: native gate memory
    survives physical resets. No historical last-end clock is inferred. All
    old rows remain dynamics supervision, but fresh native rows are required
    before gated imagination and new critic-column calibration.
    """
    config = OnlineConfig.from_dict(saved['config'])
    if (saved.get('format') != 'supermimic-online-v2' or saved.get('model_contract') != MODEL_CONTRACT
            or not config.record_jumps or not config.record_stair_source
            or saved.get('task_norm_pending', False) or saved.get('critic_norm_from') is not None):
        raise ValueError('Jump gate migration needs a calibrated recorded-jump checkpoint')
    if type(world_only_cycles) is not int or world_only_cycles < 0:
        raise ValueError('World-only adaptation cycles must be nonnegative')
    schema, target_schema = (StateSchema.from_dict(item['schema']) for item in (saved, reference))
    if (schema.fingerprint != saved['schema_fingerprint']
            or dataset.get('cycle') != saved['cycle'] or dataset['schema_fingerprint'] != schema.fingerprint
            or reference.get('format') != 'supermimic-world-v2-dataset-v1'
            or reference.get('model_contract') != MODEL_CONTRACT
            or reference['schema_fingerprint'] != target_schema.fingerprint):
        raise ValueError('Jump gate migration needs a paired checkpoint and authenticated native reference')
    old_fields, new_fields = schema.to_dict(), target_schema.to_dict()
    source, target = (json.loads(item.pop('source')) for item in (old_fields, new_fields))
    if old_fields != new_fields or {k: v for k, v in source.items() if k != 'jump_takeoff'} != {
            k: v for k, v in target.items() if k != 'jump_takeoff'}:
        raise ValueError('Jump gate migration cannot change physics, reward, course, terrain or observation contracts')
    old = JumpTaskConfig.from_dict(saved['jump_scenario'])
    new = JumpTaskConfig.from_dict(reference['jump_task_config'])
    if old.trigger is not None or new.trigger is None:
        raise ValueError('Jump gate migration requires an ungated source and gated native target')
    expected_reference = replace(old.reference,
        compression_depth_max_m=new.trigger.compression_depth_max_m,
        compression_duration_s=new.trigger.compression_duration_s,
        nominal_extension_duration_s=new.trigger.nominal_extension_duration_s)
    if old.request != new.request or old.physics_dt_s != new.physics_dt_s or new.reference != expected_reference:
        raise ValueError('Gate migration may only apply the declared takeoff reference settings')
    receipt = target.get('jump_takeoff')
    if (not isinstance(receipt, dict) or any(receipt.get(k) != v for k, v in asdict(new.trigger).items())
            or receipt.get('reference') != asdict(new.reference)):
        raise ValueError('Native source receipt does not bind the target jump gate/reference')
    for saved_key, reference_key in (('stair_scenario', 'stair_source_config'),
                                     ('recovery_scenario', 'recovery_task_config')):
        if saved.get(saved_key) != reference.get(reference_key):
            raise ValueError('Gate migration cannot replace stair or recovery state machines')
    storage, recorded = dataset['replay']['storage'], reference['replay']['storage']
    if any(name in storage for name in JUMP_TRIGGER_RECORD_FIELDS):
        raise ValueError('Gate migration cannot overwrite existing trigger records')
    flag = recorded.get('jump_record_valid')
    if (flag is None or flag.ndim != 3 or flag.shape[-1] != 1
            or any(name not in recorded or recorded[name].shape != (*flag.shape[:2], *shape)
                   or recorded[name].dtype != JUMP_TRIGGER_RECORD_DTYPES[name]
                   for name, shape in JUMP_TRIGGER_RECORD_FIELDS.items())):
        raise ValueError('Gate reference needs complete typed native trigger records')
    known = recorded['jump_trigger_flags'][..., 0].clone()
    clocks = recorded['jump_trigger_clock'][..., 0]
    known &= torch.isfinite(clocks) | torch.isneginf(clocks)
    valid = recorded.get('valid')
    owned = recorded.get('policy_owned')
    if (valid is None or owned is None or valid.shape != flag.shape[:2] or owned.shape != flag.shape[:2]
            or not (known & (flag[..., 0] > .5) & valid & owned).any()):
        raise ValueError('Gate reference needs valid policy-owned native gate history')
    base = Critic.feature_size(schema.goal_dim, storage['sim_context'].shape[-1],
        task_dim=MOTION_TASK_DIM, terrain=True, stairs=True, jumps=True, recovery=config.record_recovery)
    learner = append_critic_features(saved['learner'], expected_dim=base, extra_dim=JUMP_TRIGGER_FEATURE_DIM)
    leading = storage['frame'].shape[:2]
    additions = {name: torch.zeros(*leading, *shape, device=storage['frame'].device,
                                  dtype=JUMP_TRIGGER_RECORD_DTYPES[name])
                 for name, shape in JUMP_TRIGGER_RECORD_FIELDS.items()}
    additions['jump_record_valid'] = torch.zeros_like(storage['jump_record_valid'])
    next_config = replace(config, policy_start_cycle=saved['cycle'] + world_only_cycles)
    report = {
        'migration': 'recorded-jumps-to-trigger-v1', 'trigger_record_version': JUMP_TRIGGER_RECORD_VERSION,
        'cycle': saved['cycle'], 'source_schema': schema.fingerprint, 'target_schema': target_schema.fingerprint,
        'changed_source_contracts': [key for key in sorted(source.keys() | target.keys())
                                     if source.get(key) != target.get(key)],
        'critic_input_before': base, 'critic_input_after': base + JUMP_TRIGGER_FEATURE_DIM,
        'critic_initialization': 'append zero weights and Adam moments; zero mean, unit variance; old columns unchanged',
        'critic_norm': 'calibrate appended columns from fresh valid gate records before policy updates',
        'old_dynamics_rows_retained': leading[0] * leading[1],
        'legacy_valid_task_rows_invalidated': int((storage['jump_record_valid'] > .5).sum()),
        'legacy_gate_histories_inferred': 0,
        'episode_boundary_policy': 'no inference: gate cooldown memory survives native physical resets',
        'new_native_state': 'explicit cold simulator/gate initialization; collector offsets episode IDs',
        'telemetry': 'native global deferred counter is not causal replay state; imagined deferred rows are reported per step',
        'policy_start_cycle': next_config.policy_start_cycle,
    }
    upgraded = {**saved, 'schema': target_schema.to_dict(), 'schema_fingerprint': target_schema.fingerprint,
        'config': asdict(next_config), 'learner': learner, 'jump_scenario': asdict(new), 'critic_norm_from': base,
        'resume_migrations': [*saved.get('resume_migrations', []), report]}
    replay = {**dataset, 'schema_fingerprint': target_schema.fingerprint,
              'replay': {**dataset['replay'], 'storage': {**storage, **additions}}}
    if 'schema' in dataset:
        replay['schema'] = target_schema.to_dict()
    return upgraded, replay, report
