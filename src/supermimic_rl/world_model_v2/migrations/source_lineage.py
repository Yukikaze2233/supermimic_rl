"""Authenticate source changes in ledger order without relabeling artifacts."""
from copy import deepcopy
import json

from ...contracts import StateSchema
from ..records import MODEL_CONTRACT
from .course import FORMAT as COURSE_FORMAT, IDENTITY_KEYS, verify_course_sampling
from .stair_hint import FORMAT as HINT_FORMAT, verify_stair_hint


_PROOF_KEYS = {
    COURSE_FORMAT: 'course_sampling',
    HINT_FORMAT: 'stair_hint',
    'stair-timeout-retry-v1': 'stair_timeout_retry',
    'recovery-roll-bridge-sampling-v1': 'recovery_roll_bridge_sampling',
}


def verify_source_proof(proof):
    """Dispatch only known byte-authenticated contracts; retain exact receipts."""
    if not isinstance(proof, dict):
        raise ValueError('Source compatibility requires a complete authenticated proof')
    kind = proof.get('format')
    try:
        if kind == COURSE_FORMAT:
            checked = verify_course_sampling(proof['before'], proof['after'],
                source_course=proof['source_course_utf8'].encode('utf-8'),
                target_course=proof['target_course_utf8'].encode('utf-8'))
        elif kind == HINT_FORMAT:
            common = {'runtime_sha256': proof['runtime_sha256']}
            checked = verify_stair_hint(
                {**common, 'stair_label': proof['before']},
                {**common, 'stair_label': proof['after']},
                proof['source_scenario'], proof['target_scenario'],
                runtime_raw=proof['runtime_utf8'].encode('utf-8'))
        elif kind == 'stair-timeout-retry-v1':
            from .stair_retry import verify_saved_proof

            checked = verify_saved_proof(proof)
        elif kind == 'recovery-roll-bridge-sampling-v1':
            from .recovery_bridge import verify_saved_proof

            checked = verify_saved_proof(proof)
        else:
            raise ValueError('Unknown source compatibility proof')
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('Incomplete source compatibility proof') from error
    if checked != proof:
        raise ValueError('Source compatibility proof differs from its authenticated bytes')
    return checked


def _undo(proof, target, recorded=None):
    """New proofs bind full sources; legacy receipts retain their narrow scope."""
    kind = proof['format']
    if kind == COURSE_FORMAT:
        # Older teacher exports can combine their original course with a
        # separately qualified hint/reward handoff. Do not undo matching
        # legacy fields merely because another source field still differs.
        if recorded is not None and all(recorded.get(k) == target.get(k) for k in IDENTITY_KEYS):
            return target
        if any(target.get(k) != proof['after'].get(k) for k in IDENTITY_KEYS):
            raise ValueError('Course sampling ledger does not reach the native course identity')
        return {**target, **deepcopy(proof['before'])}
    if kind == HINT_FORMAT:
        if recorded is not None and recorded.get('stair_label') == target.get('stair_label'):
            return target
        if (target.get('stair_label') != proof['after']
                or target.get('runtime_sha256') != proof['runtime_sha256']):
            raise ValueError('Stair hint ledger does not reach the native source identity')
        return {**target, 'stair_label': deepcopy(proof['before'])}
    if target != proof['after']:
        raise ValueError('Source lineage does not reach the complete after source')
    return deepcopy(proof['before'])


def source_lineage_from_saved(saved, schema):
    """Read proofs from a complete model checkpoint, never its large replay.

    Non-source training receipts remain ordinary metadata. Unknown entries
    cannot substitute for a known proof or authorize a source difference.
    """
    if (not isinstance(saved, dict) or saved.get('format') != 'supermimic-online-v2'
            or saved.get('model_contract') != MODEL_CONTRACT
            or not isinstance(saved.get('schema'), dict)
            or not isinstance(saved.get('learner'), dict) or 'optimizers' not in saved['learner']
            or not isinstance(saved.get('environment'), dict) or not isinstance(saved.get('config'), dict)
            or not isinstance(saved.get('replay_file'), str) or not saved['replay_file']
            or type(saved.get('cycle')) is not int or saved['cycle'] < 0):
        raise ValueError('Course compatibility requires a complete online recovery checkpoint')
    recorded = StateSchema.from_dict(saved['schema'])
    if saved.get('schema_fingerprint') != recorded.fingerprint or recorded.fingerprint != schema.fingerprint:
        raise ValueError('Resume compatibility schema differs from the complete native contract')
    ledger = saved.get('resume_migrations', ())
    if not isinstance(ledger, (tuple, list)):
        raise ValueError('Online migration ledger must be a sequence')
    proofs = []
    for entry in ledger:
        if not isinstance(entry, dict):
            continue
        keys = [key for key in _PROOF_KEYS.values() if key in entry]
        kind = entry.get('migration')
        if not keys and kind not in _PROOF_KEYS:
            continue
        if (kind not in _PROOF_KEYS or keys != [_PROOF_KEYS[kind]]
                or type(entry.get('cycle')) is not int or not 0 <= entry['cycle'] <= saved['cycle']):
            raise ValueError('Unrecognized source migration record')
        proof = verify_source_proof(entry[keys[0]])
        if proof['format'] != kind:
            raise ValueError('Source migration kind differs from its proof')
        proofs.append(proof)
    identity = json.loads(recorded.source)
    for proof in reversed(proofs):
        identity = _undo(proof, identity)
    return tuple(proofs)


def compatible_source_lineage(recorded, live, proofs):
    """Undo actual source handoffs to an artifact, retaining its own identity.

    The caller must compare the returned source in full, or use its existing
    dynamics-only reward/terrain/disturbance rules. No qualification, task
    labels, dataset split or original artifact is changed by this operation.
    """
    verified = tuple(verify_source_proof(proof) for proof in proofs)
    target = deepcopy(live)
    for proof in reversed(verified):
        if target == recorded:
            break
        target = _undo(proof, target, recorded)
    return target
