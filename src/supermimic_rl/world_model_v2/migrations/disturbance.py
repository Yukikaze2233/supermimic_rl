"""Retain learned motion and optimizer history when recording external inputs."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json

import torch
from torch.nn.functional import pad

from ...contracts import StateSchema
from ...course.randomization import RequestedRandomizationConfig
from ..disturbance import DISTURBANCE_RECORD_FIELDS, DISTURBANCE_STATE_FIELDS
from ..model import INPUT_DIM
from ..records import MODEL_CONTRACT
from .critic import append_critic_features
from .teacher import rebind_teacher_source


AMPLITUDES = ('external_force_n', 'external_torque_nm', 'push_linear_m_s', 'push_yaw_rad_s')
SUBSTEP_FIELDS = ('velocity_push_world', 'external_wrench_body')


def zero_disturbance_contract(recorded, live, runtime_pairs):
    """Match unchanged contracts apart from explicitly audited external inputs.

    Arguments are schema dictionaries with decoded sources. Reward-only callers
    normalize their unused reward metadata first. A runtime hash is never simply
    discarded: its exact before/after pair must come from the byte-verified
    migration below, which also compares every other runtime setting.
    """
    before, after = deepcopy(recorded), deepcopy(live)
    source, target = before['source'], after['source']
    try:
        old = RequestedRandomizationConfig(**source['randomization']['configuration'])
        new = RequestedRandomizationConfig(**target['randomization']['configuration'])
        if old.pushes or old.wrenches or not source['randomization']['applied']:
            return False
        if not any(pair == {'before': source['runtime_sha256'], 'after': target['runtime_sha256']}
                   for pair in runtime_pairs):
            return False
        for group, names, shapes in (
                ('state_fields', DISTURBANCE_STATE_FIELDS, ((1,), (1,), (6,), (1,))),
                ('substep_fields', SUBSTEP_FIELDS, ((3,), (6,)))):
            if any(f['name'] in names for f in before.get(group, ())):
                return False
            additions = {f['name']: tuple(f['shape']) for f in after.get(group, ()) if f['name'] in names}
            if additions != dict(zip(names, shapes)):
                return False
            after[group] = tuple(f for f in after[group] if f['name'] not in names)
        old_config, new_config = asdict(old), asdict(new)
        for name in AMPLITUDES:
            new_config[name] = old_config[name]
        if old_config != new_config:
            return False
        target['randomization']['configuration'] = source['randomization']['configuration']
        target['randomization']['disturbances'] = source['randomization']['disturbances']
        target['runtime_sha256'] = source['runtime_sha256']
    except (KeyError, TypeError, ValueError):
        return False
    return before == after


def _runtime_pair(source_bytes, target_bytes, source, target):
    before, after = (json.loads(value) for value in (source_bytes, target_bytes))
    pair = {'before': hashlib.sha256(source_bytes).hexdigest(),
            'after': hashlib.sha256(target_bytes).hexdigest()}
    if pair != {'before': source['runtime_sha256'], 'after': target['runtime_sha256']}:
        raise ValueError('Disturbance runtime bytes differ from the native pins')
    configs = [RequestedRandomizationConfig(**value['requested_randomization']) for value in (before, after)]
    for runtime, config, native in zip((before, after), configs, (source, target)):
        if config != RequestedRandomizationConfig(**native['randomization']['configuration']):
            raise ValueError('Native randomization and runtime configuration differ')
        runtime['requested_randomization'] = asdict(config)
    for name in AMPLITUDES:
        after['requested_randomization'][name] = before['requested_randomization'][name]
    if before != after:
        raise ValueError('Disturbance migration cannot change other runtime settings')
    return pair, configs[1]


def _append_motion_conditions(learner, variance):
    model = dict(learner['model'])
    weight = model['motion.net.0.weight']
    if weight.shape[1] != INPUT_DIM or model['motion.input_norm.mean'].shape != (INPUT_DIM,):
        raise ValueError('Disturbance migration requires the unconditioned motion input')
    model['motion.net.0.weight'] = pad(weight, (0, 24))
    model['motion.input_norm.mean'] = pad(model['motion.input_norm.mean'], (0, 24))
    model['motion.input_norm.var'] = torch.cat((model['motion.input_norm.var'], variance.to(weight)))
    optimizer = deepcopy(learner['optimizers']['model'])
    if len(optimizer['param_groups']) != 1:
        raise ValueError('Motion expansion requires its existing single-group Adam')
    first = optimizer['param_groups'][0]['params'][0]
    for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
        if name in optimizer['state'].get(first, {}):
            moment = optimizer['state'][first][name]
            if moment.shape != weight.shape:
                raise ValueError('Motion optimizer moment shape differs')
            optimizer['state'][first][name] = pad(moment, (0, 24))
    return {**learner, 'model': model, 'optimizers': {**learner['optimizers'], 'model': optimizer}}


def upgrade_disturbances(saved, dataset, reference, *, source_runtime, target_runtime,
                         source_teacher_path, target_teacher_path):
    """Add recorded disturbances without resetting any learner or training clock."""
    old, new = (StateSchema.from_dict(item['schema']) for item in (saved, reference))
    if (saved.get('format') != 'supermimic-online-v2' or saved.get('model_contract') != MODEL_CONTRACT
            or reference.get('format') != 'supermimic-world-v2-dataset-v1'
            or reference.get('model_contract') != MODEL_CONTRACT
            or saved['schema_fingerprint'] != old.fingerprint or reference['schema_fingerprint'] != new.fingerprint
            or dataset['schema_fingerprint'] != old.fingerprint or dataset['cycle'] != saved['cycle']):
        raise ValueError('Disturbance migration needs a complete paired checkpoint and native reference')
    before, after = old.to_dict(), new.to_dict()
    source, target = json.loads(old.source), json.loads(new.source)
    before['source'], after['source'] = source, target
    pair, config = _runtime_pair(source_runtime, target_runtime, source, target)
    if not zero_disturbance_contract(before, after, (pair,)):
        raise ValueError('Only verified zero-disturbance history can migrate; other contracts must match')
    if saved.get('task_norm_pending') or saved.get('critic_norm_from') is not None:
        raise ValueError('Complete the existing feature migration before adding disturbances')
    for a, b in (('stair_scenario', 'stair_source_config'), ('jump_scenario', 'jump_task_config'),
                 ('recovery_scenario', 'recovery_task_config')):
        if saved.get(a) != reference.get(b):
            raise ValueError('Disturbance migration cannot change task protocols')
    storage, recorded = dataset['replay']['storage'], reference['replay']['storage']
    if set(storage) & DISTURBANCE_RECORD_FIELDS.keys():
        raise ValueError('Disturbance migration cannot reinterpret existing disturbance records')
    if (set(recorded) != set(storage) | DISTURBANCE_RECORD_FIELDS.keys()
            or any(value.shape[2:] != recorded[name].shape[2:] or value.dtype != recorded[name].dtype
                   for name, value in storage.items())):
        raise ValueError('Reference records must preserve the complete previous replay layout')
    leading = recorded['physical'].shape[:2]
    if any(recorded[name].shape != (*leading, *shape) or recorded[name].dtype != torch.float32
           for name, shape in DISTURBANCE_RECORD_FIELDS.items()):
        raise ValueError('Reference disturbance fields differ from the compact contract')
    known = recorded['valid'] & recorded['policy_owned'] & (recorded['disturbance_record_valid'][..., 0] > .5)
    if not known.any() or any(not torch.isfinite(recorded[name][known]).all() for name in DISTURBANCE_RECORD_FIELDS):
        raise ValueError('Reference needs finite policy-owned native disturbance records')
    # Use pinned uniform-force scales, not a cold reset batch with no due events.
    # Disabled constant channels use unit variance; their physical inputs are zero.
    wrench_variance = torch.tensor([config.external_force_n**2 / 3] * 3 + [config.external_torque_nm**2 / 3] * 3)
    wrench_variance = torch.where(wrench_variance > 0, wrench_variance, 1.)
    learner = _append_motion_conditions(saved['learner'], wrench_variance.repeat(4))
    critic_dim = learner['critic']['norm.mean'].numel()
    learner = append_critic_features(learner, expected_dim=critic_dim, extra_dim=9)
    critic_variance = torch.cat((torch.tensor([config.push_interval_s[1]**2, config.wrench_interval_s[1]**2]),
                                 wrench_variance, torch.ones(1)))
    learner['critic']['norm.var'][-9:] = critic_variance.to(learner['critic']['norm.var'])
    additions = {name: storage['physical'].new_zeros(*storage['physical'].shape[:2], *shape)
                 for name, shape in DISTURBANCE_RECORD_FIELDS.items()}
    teacher = rebind_teacher_source(saved, source, target, source_teacher_path, target_teacher_path)
    report = {'migration': 'recorded-disturbances-v1', 'cycle': saved['cycle'],
        'source_schema': old.fingerprint, 'target_schema': new.fingerprint,
        'disturbance_runtime_pair': pair, 'motion_input_before': INPUT_DIM, 'motion_input_after': INPUT_DIM + 24,
        'critic_input_before': critic_dim, 'critic_input_after': critic_dim + 9,
        'motion_appended_variance': wrench_variance.repeat(4).tolist(),
        'critic_appended_variance': critic_variance.tolist(),
        'normalization': 'old statistics/counts unchanged; appended zero means with pinned physical scales',
        'old_physical_transitions_retained': storage['physical'].shape[0] * storage['physical'].shape[1],
        'legacy_conditions': 'verified zero push/wrench; unavailable scheduler states excluded from imagination',
        'new_weights_and_adam_columns': 'zero; all previous columns and optimization states preserved',
        'additional_warmup_cycles': 0, 'physics_resume': 'explicit cold reset; new episode namespace'}
    upgraded = {**saved, 'schema': new.to_dict(), 'schema_fingerprint': new.fingerprint,
        'learner': learner, 'teacher_source': teacher,
        'resume_migrations': [*saved.get('resume_migrations', []), report]}
    replay = {**dataset, 'schema_fingerprint': new.fingerprint,
              'replay': {**dataset['replay'], 'storage': {**storage, **additions}}}
    if 'schema' in dataset:
        replay['schema'] = new.to_dict()
    return upgraded, replay, report
