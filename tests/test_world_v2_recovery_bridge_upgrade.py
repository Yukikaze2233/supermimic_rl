"""The bridge changes future births, preserving complete learner and replay truth."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json

import pytest
import torch

from supermimic_rl.contracts import StateSchema
from supermimic_rl.course.jump_first import make_course_curriculum
from supermimic_rl.course.recovery_bridge import receipt as bridge_receipt
from supermimic_rl.world_model_v2.migrations.recovery_bridge import (
    FORMAT, compatible_recovery_bridge_source, upgrade_recovery_roll_bridge,
    verify_recovery_bridge_sampling, verify_saved_proof,
)
from test_world_v2_course_sampling_upgrade import candidate as original_fixture, encoded


@pytest.fixture
def candidate():
    saved, data, _, _ = original_fixture.__wrapped__()
    source = json.loads(saved['schema']['source'])
    before = source['curriculum']
    after = deepcopy(before)
    after['base']['recovery_roll_bridge'] = True
    target = {**source, 'curriculum': after,
              'course_sha256': hashlib.sha256(encoded(after)).hexdigest(),
              'recovery_roll_bridge': bridge_receipt()}
    schema = replace(StateSchema.from_dict(saved['schema']), source=json.dumps(target))
    names = ('model', 'actor', 'critic', 'estimator')
    learner = saved['learner']
    learner['optimizers'] = {name: {'state': {'step': torch.tensor(123.), 'exp_avg': torch.arange(5.)},
                                   'param_groups': [{'lr': 2e-4}]} for name in names}
    learner['updates'] = {name: 123 for name in names}
    learner['skipped'] = {name: 2 for name in names}
    learner['model']['normalizer'] = torch.arange(7.)
    saved['cuda_rng'] = []
    saved['validation_rng'] = torch.Generator().manual_seed(42).get_state()
    saved['collection_rng'] = torch.Generator().manual_seed(43).get_state()
    saved['mixing_rng'] = torch.Generator().manual_seed(44).get_state()
    # Make a nonzero, non-cycle-aligned family ordinal and truthful old labels.
    old_course = make_course_curriculum(2, 'cpu', 19, before)
    old_course.load_state_dict(saved['environment']['curriculum'])
    for _ in range(53):
        old_course.sample(torch.tensor([0, 1]))
    saved['environment']['curriculum'] = old_course.state_dict()
    assert sum(old_course.sample_counts['recovery_posture']) > 0
    data['replay']['storage']['valid'][0, 0] = False
    data['replay']['storage']['stair_source_valid'] = torch.ones(4, 2, 1)
    data['replay']['storage']['stair_source_valid'][1, 0] = 0
    return saved, data, schema, {'source_course': encoded(before), 'target_course': encoded(after)}


def equal_nested(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for name in left:
            equal_nested(left[name], right[name])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal_nested(a, b)
    else:
        assert left == right


def test_complete_pair_and_old_truth_are_retained_without_new_random_draws(candidate):
    saved, data, target, options = candidate
    old_saved, old_data, rng = deepcopy(saved), deepcopy(data), torch.get_rng_state().clone()
    upgraded, replay, report = upgrade_recovery_roll_bridge(saved, data, target, **options)
    assert torch.equal(torch.get_rng_state(), rng)
    equal_nested(saved, old_saved)
    equal_nested(data, old_data)
    for name in saved.keys() - {'schema', 'schema_fingerprint', 'environment', 'resume_migrations'}:
        assert upgraded[name] is saved[name], name
    for name in saved['environment'].keys() - {'curriculum'}:
        assert upgraded['environment'][name] is saved['environment'][name], name
    assert replay['replay'] is data['replay']
    for name, value in data['replay']['storage'].items():
        assert replay['replay']['storage'][name] is value
    assert not replay['replay']['storage']['valid'][0, 0]
    assert replay['replay']['storage']['stair_source_valid'][1, 0] == 0
    before, after = saved['environment']['curriculum'], upgraded['environment']['curriculum']
    assert after['sample_counts']['recovery_posture'] == before['sample_counts']['recovery_posture'] + [0] * 6
    for name in before['sample_counts'].keys() - {'recovery_posture'}:
        assert after['sample_counts'][name] == before['sample_counts'][name]
    assert after['sampler'] is before['sampler']
    assert after['real_transitions'] == before['real_transitions']
    assert after['actor_updates'] == before['actor_updates']
    assert replay['schema_fingerprint'] == upgraded['schema_fingerprint'] == target.fingerprint
    assert report['additional_warmup_cycles'] == 0
    assert report['pending_integration']


def test_restored_sampler_retains_next_family_rng_and_bridge_phase(candidate):
    saved, data, target, options = candidate
    upgraded, _, _ = upgrade_recovery_roll_bridge(saved, data, target, **options)
    before = saved['environment']['curriculum']
    original = make_course_curriculum(2, 'cpu', 0, json.loads(options['source_course']))
    restored = make_course_curriculum(2, 'cpu', 999, json.loads(options['target_course']))
    original.load_state_dict(before)
    restored.load_state_dict(upgraded['environment']['curriculum'])
    # The legacy v5 fixture has not unlocked all postures yet. Advance both
    # courses identically; the historical birth ordinal must not be reset.
    for course in (original, restored):
        course.set_progress(sum(course.stage_budgets), before['actor_updates'])
    offset = before['sample_counts']['family'][5]
    seen = []
    for _ in range(180):
        ids = torch.tensor([0, 1])
        left, right = original.sample(ids), restored.sample(ids)
        assert torch.equal(left.family, right.family)
        assert torch.equal(original.sampler.generator.get_state(), restored.sampler.generator.get_state())
        mask = right.family == 5
        if mask.any():
            variants = restored.recovery_birth_variants(ids[mask])
            seen.extend(variants.tolist())
    pool = restored.recovery_posture_pool()
    simple = (0, 1, 2, 3, 6, 7)
    from transformer_rl.formal.tasks.recovery import RECOVERY_NAMES

    expected = []
    for ordinal in range(offset, offset + len(seen)):
        posture = RECOVERY_NAMES.index(pool[ordinal % len(pool)])
        expected.append(16 + simple.index(posture) if ordinal // len(pool) % 2 and posture in simple else posture)
    assert seen == expected and any(label >= 16 for label in seen)


def test_teacher_and_auxiliary_metadata_remain_original_and_scope_is_not_expanded(candidate):
    saved, data, target, options = candidate
    upgraded, _, report = upgrade_recovery_roll_bridge(saved, data, target, **options)
    assert upgraded['teacher_source'] is saved['teacher_source']
    assert upgraded['teacher_source']['qualification'] is saved['teacher_source']['qualification']
    assert upgraded['auxiliary_world'] is saved['auxiliary_world']
    old = upgraded['teacher_source']['native_source']
    live = json.loads(target.source)
    proof = report['recovery_roll_bridge_sampling']
    assert old != live
    assert compatible_recovery_bridge_source(old, live, (proof,)) == old
    assert upgraded['teacher_source']['qualified_for'] == ['original-data-domain']


def test_proof_authentication_leaves_unrelated_differences_visible(candidate):
    saved, data, target, options = candidate
    _, _, report = upgrade_recovery_roll_bridge(saved, data, target, **options)
    proof = report['recovery_roll_bridge_sampling']
    assert proof['format'] == FORMAT and verify_saved_proof(proof) == proof
    source, live = json.loads(saved['schema']['source']), json.loads(target.source)
    assert compatible_recovery_bridge_source(source, live, ()) == live
    different = {**live, 'runtime_sha256': 'not-the-same-runtime'}
    undone = compatible_recovery_bridge_source(source, different, (proof,))
    assert undone['runtime_sha256'] == 'not-the-same-runtime' and undone != source
    for modified in ({**proof, 'extra': True}, {**proof, 'source_course_utf8': proof['source_course_utf8'] + ' '}):
        with pytest.raises(ValueError):
            verify_saved_proof(modified)


@pytest.mark.parametrize('change', ('reward', 'physics', 'observer', 'task', 'receipt', 'receipt_scope',
                                   'mixture', 'other_course_flag', 'height_domain', 'disable', 'already_enabled'))
def test_only_exact_one_way_birth_change_is_authorized(candidate, change):
    saved, data, schema, options = candidate
    source, target = json.loads(saved['schema']['source']), json.loads(schema.source)
    if change in ('reward', 'physics', 'observer', 'task'):
        key = {'reward': 'reward_kernel', 'physics': 'runtime_sha256',
               'observer': 'observer_contract', 'task': 'stair_label'}[change]
        target[key] = 'changed'
    elif change == 'receipt':
        target.pop('recovery_roll_bridge')
    elif change == 'receipt_scope':
        target['recovery_roll_bridge']['cycle_births']['bridge'] = 8
    elif change == 'already_enabled':
        source = deepcopy(target)
        options = {**options, 'source_course': options['target_course']}
    else:
        course = deepcopy(target['curriculum'])
        if change == 'mixture':
            course['base']['stage_weights'][0] = [.4, .5, .1, 0, 0, 0]
        elif change == 'other_course_flag':
            course['base']['foundation_authored_rehearsal'] = True
        elif change == 'height_domain':
            course['base']['domain']['height_m'][1] = .41
        else:
            course['base']['recovery_roll_bridge'] = False
        raw = encoded(course)
        target.update(curriculum=course, course_sha256=hashlib.sha256(raw).hexdigest())
        options = {**options, 'target_course': raw}
    with pytest.raises(ValueError):
        verify_recovery_bridge_sampling(source, target, **options)


@pytest.mark.parametrize('damage', ('cycle', 'schema', 'layout', 'clock', 'optimizer', 'task_record',
                                   'ledger', 'rng', 'teacher'))
def test_incomplete_or_mismatched_pairs_are_not_migrated(candidate, damage):
    saved, data, target, options = candidate
    saved, data = deepcopy(saved), deepcopy(data)
    if damage == 'cycle':
        data['cycle'] -= 1
    elif damage == 'schema':
        data['schema_fingerprint'] = 'wrong'
    elif damage == 'layout':
        target = replace(target, observation_dim=45)
    elif damage == 'clock':
        saved['collection_steps'] += 1
    elif damage == 'optimizer':
        saved['learner']['optimizers'].pop('estimator')
    elif damage == 'task_record':
        saved['config']['record_recovery'] = False
    elif damage == 'ledger':
        saved['environment']['curriculum']['sample_counts']['recovery_posture'].extend([0] * 6)
    elif damage == 'rng':
        saved['environment']['curriculum']['sampler']['rng'] = torch.zeros(1, dtype=torch.uint8)
    else:
        saved.pop('teacher_source')
    with pytest.raises(ValueError):
        upgrade_recovery_roll_bridge(saved, data, target, **options)


def test_serialized_pair_contains_old_labels_with_only_appended_zero_counts(candidate):
    saved, data, target, options = candidate
    upgraded, replay, _ = upgrade_recovery_roll_bridge(saved, data, target, **options)
    output = io.BytesIO()
    torch.save((upgraded, replay), output)
    output.seek(0)
    reloaded, replay_loaded = torch.load(output, weights_only=True)
    equal_nested(reloaded, upgraded)
    equal_nested(replay_loaded, replay)
    equal_nested(replay_loaded['replay']['storage'], data['replay']['storage'])
