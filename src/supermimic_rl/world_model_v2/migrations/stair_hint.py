"""Bind future stair hints to requests without rewriting historical task memory."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json

import torch

from ...contracts import StateSchema
from ..records import MODEL_CONTRACT
from ..stair_source import StairScenario
from .teacher import rebind_teacher_source


FORMAT = 'stair-request-hint-lifetime-v1'


def verify_stair_hint(source, target, source_scenario, target_scenario, *, runtime_raw):
    """Authenticate only the opt-in duration change against the unchanged runtime."""
    if (not isinstance(runtime_raw, bytes)
            or hashlib.sha256(runtime_raw).hexdigest() != source.get('runtime_sha256')
            or {k: v for k, v in source.items() if k != 'stair_label'}
            != {k: v for k, v in target.items() if k != 'stair_label'}):
        raise ValueError('Stair hint migration requires the unchanged pinned runtime and task source')
    before = StairScenario.from_dict(source_scenario)
    after = StairScenario.from_dict(target_scenario)
    runtime = json.loads(runtime_raw)
    assist = before.task.height_assist
    if (assist is None or not assist.enabled or not before.task.use_mode_labels
            or asdict(assist) != runtime.get('stair_height_assist')):
        raise ValueError('Stair hint migration requires the original enabled runtime height hint')
    expected = deepcopy(source_scenario)
    expected['task']['height_assist']['hold_s'] = before.task.request.timeout_s
    if target_scenario != expected or after.task.height_assist.hold_s == assist.hold_s:
        raise ValueError('Only binding stair hold_s to its existing request timeout is supported')
    label = source.get('stair_label')
    if (not isinstance(label, dict) or set(label) != {'stair_label'}
            or label['stair_label'] != 'STAIR set by the ToF stair state machine only; terrain-kind labels disabled'):
        raise ValueError('Stair hint migration requires the known ToF-only label contract')
    hint = {
        'contract': 'stair_request_lifetime_hint_v1',
        'runtime_hold_s': assist.hold_s,
        'effective_hold_s': before.task.request.timeout_s,
        'duration_source': 'StairRequestProtocol.config.timeout_s',
        'height_reference': 'native request-entry target, unchanged raise and maximum height',
        'release': 'native completion, cancellation, timeout or abort; replayed IDs do not refresh',
        'priority': 'native jump-active abort; no raised output in base JUMP or RECOVERY mode',
    }
    if target.get('stair_label') != {**label, 'height_hint': hint}:
        raise ValueError('Stair hint receipt does not describe the authenticated request duration')
    return {'format': FORMAT, 'runtime_sha256': source['runtime_sha256'],
            'runtime_utf8': runtime_raw.decode('utf-8'),
            'before': deepcopy(label), 'after': deepcopy(target['stair_label']),
            'source_scenario': deepcopy(source_scenario), 'target_scenario': deepcopy(target_scenario)}


def compatible_stair_hint_source(recorded, live, migrations):
    """Undo authenticated hint provenance for dynamics-only data compatibility.

    The caller still compares every other source field. This never qualifies
    historical task records as starts for the new request lifetime.
    """
    target = deepcopy(live)
    for proof in reversed(migrations):
        if recorded.get('stair_label') == target.get('stair_label'):
            break
        if (proof.get('format') != FORMAT or target.get('stair_label') != proof.get('after')
                or target.get('runtime_sha256') != proof.get('runtime_sha256')):
            continue
        common = {'runtime_sha256': proof['runtime_sha256']}
        authenticated = verify_stair_hint(
            {**common, 'stair_label': proof['before']}, {**common, 'stair_label': proof['after']},
            proof['source_scenario'], proof['target_scenario'],
            runtime_raw=proof['runtime_utf8'].encode('utf-8'))
        if authenticated != proof:
            raise ValueError('Stair hint compatibility receipt differs from its authenticated pair')
        target['stair_label'] = deepcopy(proof['before'])
    return target


def upgrade_stair_hint(saved, dataset, reference, *, runtime_raw, source_teacher_path,
                       target_teacher_path, jump_peak_completion=False):
    """Retain the complete learner; replace only future stair task semantics.

    Optional peak-completion reward changes use the same final native reference
    and teacher qualification, without inventing an intermediate evaluation.
    """
    old, new = (StateSchema.from_dict(item['schema']) for item in (saved, reference))
    if (saved.get('format') != 'supermimic-online-v2' or saved.get('model_contract') != MODEL_CONTRACT
            or reference.get('format') != 'supermimic-world-v2-dataset-v1'
            or reference.get('model_contract') != MODEL_CONTRACT
            or saved.get('schema_fingerprint') != old.fingerprint
            or reference.get('schema_fingerprint') != new.fingerprint
            or dataset.get('schema_fingerprint') != old.fingerprint
            or dataset.get('cycle') != saved.get('cycle')
            or 'optimizers' not in saved.get('learner', {})):
        raise ValueError('Stair hint migration requires a complete paired checkpoint and native reference')
    if 'schema' in dataset and StateSchema.from_dict(dataset['schema']) != old:
        raise ValueError('Replay schema differs from its checkpoint')
    old_fields, new_fields = old.to_dict(), new.to_dict()
    source, target = (json.loads(fields.pop('source')) for fields in (old_fields, new_fields))
    if old_fields != new_fields:
        raise ValueError('Stair hint migration cannot change physical or observation tensor contracts')
    proof_source = source
    if jump_peak_completion:
        from .reward_reference import verify_jump_peak_completion

        verify_jump_peak_completion(source['reward_kernel'], target['reward_kernel'])
        proof_source = {**source, 'reward_kernel': target['reward_kernel']}
    proof = verify_stair_hint(proof_source, target, saved['stair_scenario'],
                              reference['stair_source_config'], runtime_raw=runtime_raw)
    for stored, captured in (('jump_scenario', 'jump_task_config'),
                             ('recovery_scenario', 'recovery_task_config')):
        if saved.get(stored) != reference.get(captured):
            raise ValueError('Stair hint migration cannot change jump or recovery configuration')
    if not all(saved['config'].get(key) for key in ('record_tasks', 'record_stair_source')):
        raise ValueError('Stair hint migration requires the existing recorded stair source')
    storage, fresh = dataset['replay']['storage'], reference['replay']['storage']
    if (storage.keys() != fresh.keys()
            or any(value.shape[2:] != fresh[key].shape[2:] or value.dtype != fresh[key].dtype
                   for key, value in storage.items())):
        raise ValueError('Stair hint reference and replay record layouts differ')
    valid = storage.get('stair_source_valid')
    if valid is None or valid.shape[-1] != 1:
        raise ValueError('Stair hint migration needs explicit stair-source validity')
    teacher = rebind_teacher_source(saved, source, target, source_teacher_path, target_teacher_path)
    report = {
        'migration': FORMAT, 'cycle': saved['cycle'], 'stair_hint': proof,
        'source_schema': old.fingerprint, 'target_schema': new.fingerprint,
        'jump_peak_completion': jump_peak_completion,
        'legacy_stair_sequence_before': dataset['replay']['total_steps'],
        'legacy_stair_starts_excluded': int((valid > .5).sum()),
        'historical_labels': 'physical, reward, task parameters and remaining hint steps unchanged',
        'learner_optimizers_rng_and_curriculum': 'all preserved',
        'auxiliary_data': 'original source, split and RNG retained; dynamics-only compatibility',
        'teacher_qualification': teacher['qualified_for'], 'teacher_weights_identical': True,
        'additional_warmup_cycles': 0,
    }
    if jump_peak_completion:
        report['legacy_reward_sequence_before'] = dataset['replay']['total_steps']
    updated = {**saved, 'schema': new.to_dict(), 'schema_fingerprint': new.fingerprint,
               'teacher_source': teacher, 'stair_scenario': reference['stair_source_config'],
               'resume_migrations': [*saved.get('resume_migrations', ()), report]}
    replay = {**dataset, 'schema_fingerprint': new.fingerprint,
              'replay': {**dataset['replay'], 'storage': {**storage, 'stair_source_valid': torch.zeros_like(valid)}}}
    if 'schema' in replay:
        replay['schema'] = new.to_dict()
    return updated, replay, report
