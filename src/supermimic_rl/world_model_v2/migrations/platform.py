"""In-memory building blocks for an explicitly authenticated platform handoff.

These functions neither authorize a source change nor relabel a checkpoint.
The handoff must establish caller absence from the original producer before
requesting known-inactive history. Unqualified external data remain unknown.
"""
from copy import deepcopy

import torch

from ..platform_contacts import PLATFORM_CONTACT_PRODUCER
from ..platform_features import PLATFORM_FEATURE_DIM
from ..platform_records import PLATFORM_RECORD_DTYPES, PLATFORM_RECORD_FIELDS
from .critic import append_critic_features


def append_platform_history(replay, *, caller_known_inactive: bool):
    """Allocate only new fields on the already restored training ring.

    No old motion, ownership, reward, boundary, sequence or tensor allocation
    changes. Knowing the caller was absent never creates physical tick labels.
    A false argument produces unknown caller placeholders, not negative data.
    Source authentication and the new schema ledger belong to the handoff.
    """
    if type(caller_known_inactive) is not bool:
        raise ValueError('Caller knowledge must be explicitly specified')
    if replay.read_only:
        raise ValueError('Platform migration cannot mutate a read-only replay view')
    if set(replay.data).intersection(PLATFORM_RECORD_FIELDS):
        raise ValueError('Existing or partial platform history cannot be overwritten')
    additions = {
        name: torch.zeros(replay.capacity, replay.num_envs, *shape,
                          dtype=PLATFORM_RECORD_DTYPES[name], device=replay.device)
        for name, shape in PLATFORM_RECORD_FIELDS.items()
    }
    additions['platform_intent_ids'][..., 0] = -1
    additions['platform_intent_ids'][..., 2] = -1
    additions['platform_landing_ids'][..., 0] = -1
    additions['platform_landing_state_ticks'][..., 0] = -1
    additions['platform_landing_world_height_ticks'].fill_(torch.nan)
    if caller_known_inactive:
        additions['platform_record_valid'][replay._chronological()] = True
    replay.data.update(additions)
    replay._cache.clear()
    return {
        'caller_history': 'known_inactive' if caller_known_inactive else 'unknown',
        'physical_tick_evidence': 'unknown',
        'populated_transitions': len(replay),
        'allocated_bytes': sum(t.numel() * t.element_size() for t in additions.values()),
        'old_storage_reallocated': False,
    }


def append_platform_learner(learner, head, *, expected_critic_dim):
    """Append value inputs and a head's complete Adam history, without RNG use.

    This preserves a candidate exactly; it does not certify its accuracy or
    establish source/asset compatibility. The head's offline RNG stays in its
    provenance instead of replacing the online learner or auxiliary RNG.
    """
    names = {'model', 'actor', 'critic', 'estimator'}
    if ('platform_contact' in learner
            or any(set(learner.get(key, {})) != names
                   for key in ('optimizers', 'updates', 'skipped'))):
        raise ValueError('Platform initialization requires the original four learner states')
    if (head.get('format') != 'supermimic-platform-contact-head-v1'
            or head.get('producer') != PLATFORM_CONTACT_PRODUCER
            or not isinstance(head.get('model'), dict)
            or not isinstance(head.get('optimizer'), dict)
            or any(type(head.get(key)) is not int or head[key] < 0
                   for key in ('updates', 'skipped'))):
        raise ValueError('Platform initialization needs a complete contact head and Adam state')
    result = append_critic_features(learner, expected_dim=expected_critic_dim,
                                    extra_dim=PLATFORM_FEATURE_DIM)
    return {
        **result,
        'platform_contact': deepcopy(head['model']),
        'optimizers': {**result['optimizers'], 'platform_contact': deepcopy(head['optimizer'])},
        'updates': {**result['updates'], 'platform_contact': head['updates']},
        'skipped': {**result['skipped'], 'platform_contact': head['skipped']},
    }
