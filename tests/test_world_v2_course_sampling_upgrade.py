"""Sampling-only changes retain historical conditions and full recovery state."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import torch

from supermimic_rl.contracts import Field, StateSchema
from supermimic_rl.course.jump_first import make_course_curriculum
from supermimic_rl.world_model_v2.migrations.course import (
    compatible_course_source, upgrade_course_sampling, verify_course_sampling,
)
from supermimic_rl.world_model_v2.records import MODEL_CONTRACT
from supermimic_rl.world_model_v2.replay import CompactReplay


def encoded(course):
    return json.dumps(course, indent=2).encode('utf-8')


def course_source(course):
    return {
        'course_sha256': hashlib.sha256(encoded(course)).hexdigest(), 'curriculum': deepcopy(course),
        'course_schedule': {'schedule': course['schedule'], 'request_heading_override': False,
                            'native_handoff_preserved': True, 'request_yaw_rate': 0.,
                            'early_jump_domain': course['base']['initial_domain']['jump_m']},
        'runtime_sha256': 'same-runtime', 'terrain_sha256': 'same-atlas',
        'upstream_commit': 'same-native', 'asset_manifest_sha256': 'same-asset',
        'reward_kernel': {'implementation': 'shared_physical_reward_densities', 'kernel_sha256': 'same-kernel'},
        'jump_takeoff': {'trigger': 'same-trigger'}, 'stair_label': {'version': 'same-label'},
    }


@pytest.fixture
def candidate():
    path = Path(__file__).resolve().parents[1] / 'configs/gpufree/ppo-course-gated-v5.json'
    old_course = json.loads(path.read_bytes())
    new_course = deepcopy(old_course)
    new_course['base']['initial_domain']['jump_m'][1] = .2
    new_course['base']['jump_height_gates'][0][1] = .25
    new_course['base']['stage_weights'][3] = [.03, .20, .07, .30, .30, .10]
    new_course['base']['recovery_posture_stages']['left_side_90'] = 'active_jump'
    source, target = [course_source(course) for course in (old_course, new_course)]
    schema = StateSchema((Field('physical_fixture', (2,)),), (), source=json.dumps(source))
    target_schema = replace(schema, source=json.dumps(target))
    curriculum = make_course_curriculum(2, 'cpu', 19, old_course)
    coordinate = 1_076_428_800
    curriculum.set_progress(coordinate, 18458)
    # Exercise nonzero quota credits, RNG advancement and sampling counters.
    curriculum.sample(torch.tensor([0, 1]))
    replay = CompactReplay({'physical': (2,), 'jump_values': (17,), 'ticks': (4, 3)}, 2, 4, 'cpu')
    replay.size = replay.head = replay.total_steps = 4
    replay.sequence[:] = torch.arange(4)
    replay.data['valid'].fill_(True)
    replay.data['physical'].fill_(3.25)
    replay.data['jump_values'].fill_(.4)
    replay.data['ticks'].copy_(torch.arange(96).reshape(4, 2, 4, 3))
    teacher = {'native_source': deepcopy(source), 'teacher_artifact': {'sha256': 'teacher-original'},
               'qualified_for': ['original-data-domain'],
               'qualification': {'report': {'native_source': deepcopy(source), 'steps': 240}}}
    config = {'record_tasks': True, 'record_stair_source': True, 'record_jumps': True,
              'record_recovery': True, 'course_coordinate': coordinate,
              'course_progression': {'start_collection_step': 99, 'end_collection_step': 199,
                                    'start_coordinate': coordinate, 'end_coordinate': coordinate + 10000,
                                    'jump_height_progression': 'schedule'},
              'world_datasets': [{'path': 'student750.pt', 'sha256': 'original-data',
                                  'train_env_ids': list(range(18)), 'validation_env_ids': list(range(18, 24))}]}
    saved = {
        'format': 'supermimic-online-v2', 'model_contract': MODEL_CONTRACT, 'cycle': 1025,
        'replay_file': 'replay-a.pt',
        'schema': schema.to_dict(), 'schema_fingerprint': schema.fingerprint, 'config': config,
        'collection_steps': 100, 'learner': {
            'model': {'weights': torch.arange(7.)}, 'actor': {'weights': torch.arange(3.)},
            'critic': {'weights': torch.arange(5.)}, 'updates': {'actor': 23640},
            'optimizers': {'actor': {'exp_avg': torch.arange(3.), 'lr': 5e-5}}},
        'environment': {'format': 'supermimic_course_v1', 'curriculum': curriculum.state_dict(),
                        'course': {'tasks': {'pending_rise': torch.tensor([.4, .3])}},
                        'control_timing': {'clock': 999}, 'sensors': {'rng': torch.get_rng_state()}},
        'teacher_source': teacher, 'torch_rng': torch.get_rng_state(),
        'sample_rng': torch.Generator().manual_seed(23).get_state(),
        'auxiliary_world': {'sample_rng': torch.Generator().manual_seed(29).get_state(),
                            'sampled_windows': {'student750': 12345}},
        'course_progression': {'coordinate': coordinate, 'opened': [], 'history': [{'count': 12}]},
        'stair_scenario': {'unchanged': 'stair'}, 'jump_scenario': {'jump_max_m': .4},
        'recovery_scenario': {'unchanged': 'recovery'}, 'resume_migrations': [],
    }
    data = {'cycle': saved['cycle'], 'schema_fingerprint': schema.fingerprint,
            'schema': schema.to_dict(), 'replay': replay.state_dict()}
    options = {'source_course': encoded(old_course), 'target_course': encoded(new_course)}
    return saved, data, target_schema, options


def test_course_candidate_retains_complete_learning_and_historical_high_requests(candidate):
    saved, data, target, options = candidate
    upgraded, replay, report = upgrade_course_sampling(saved, data, target, **options)
    for name in saved.keys() - {'schema', 'schema_fingerprint', 'environment', 'resume_migrations'}:
        assert upgraded[name] is saved[name]
    assert replay['replay'] is data['replay']
    assert replay['replay']['storage']['valid'].all()
    assert replay['replay']['storage']['jump_values'].eq(.4).all()
    for name in saved['environment'].keys() - {'curriculum'}:
        assert upgraded['environment'][name] is saved['environment'][name]
    assert upgraded['teacher_source']['qualification'] is saved['teacher_source']['qualification']
    assert upgraded['schema_fingerprint'] == target.fingerprint == replay['schema_fingerprint']
    assert report['teacher_evaluation_scope_unchanged'] and report['sampler_credits_reset']
    assert report['course_coordinate'] == saved['environment']['curriculum']['real_transitions']
    assert report['pending_integration']
    assert saved['schema_fingerprint'] != target.fingerprint


def test_new_sampler_loads_its_own_configuration_and_preserves_rng_counts(candidate):
    saved, data, target, options = candidate
    upgraded, _, _ = upgrade_course_sampling(saved, data, target, **options)
    old, new = saved['environment']['curriculum'], upgraded['environment']['curriculum']
    assert new['config'] != old['config']
    assert new['sampler']['stage_weights'] != old['sampler']['stage_weights']
    assert torch.equal(new['sampler']['rng'], old['sampler']['rng'])
    assert new['sample_counts'] == old['sample_counts']
    assert new['actor_updates'] == old['actor_updates']
    restored = make_course_curriculum(2, 'cpu', 1, json.loads(options['target_course']))
    restored.load_state_dict(new)
    assert restored.real_transitions == old['real_transitions']
    assert restored._effective_domain().jump_m[1] == .4  # schedule mode remains explicit
    restored.set_opened_gates(())
    assert restored._effective_domain().jump_m[1] == .2


def test_compatibility_requires_registered_exact_identity_and_keeps_other_differences(candidate):
    saved, data, target, options = candidate
    _, _, report = upgrade_course_sampling(saved, data, target, **options)
    source, live = json.loads(saved['schema']['source']), json.loads(target.source)
    proof = report['course_sampling']
    assert compatible_course_source(source, live, ()) == live
    assert compatible_course_source(source, live, (proof,)) == source
    changed = {**live, 'runtime_sha256': 'other-runtime'}
    assert compatible_course_source(source, changed, (proof,)) != source
    assert compatible_course_source(source, changed, (proof,))['runtime_sha256'] == 'other-runtime'
    unmatched = {**live, 'course_sha256': 'unregistered'}
    assert compatible_course_source(source, unmatched, (proof,)) == unmatched
    forged = deepcopy(proof)
    forged['source_course_utf8'] += ' '
    with pytest.raises(ValueError, match='source pin'):
        compatible_course_source(source, live, (forged,))


@pytest.mark.parametrize('change', ('physics', 'reward', 'command_pool', 'budget', 'coordinates',
                                   'velocity_domain', 'full_jump_domain', 'gate_keys'))
def test_unrelated_or_unmigrated_semantics_are_rejected(candidate, change):
    saved, data, schema, options = candidate
    target = json.loads(schema.source)
    course = deepcopy(target['curriculum'])
    if change == 'physics':
        target['runtime_sha256'] = 'different-physics'
    elif change == 'reward':
        target['reward_kernel']['kernel_sha256'] = 'different-reward'
    elif change == 'command_pool':
        target['course_schedule']['request_yaw_rate'] = 1.
    else:
        if change == 'budget':
            course['base']['stage_transition_budgets'][0] += 1
        elif change == 'coordinates':
            course['base']['source_schedule_coordinates'][2] += 1
        elif change == 'velocity_domain':
            course['base']['initial_domain']['forward_m_s'][1] = .5
        elif change == 'full_jump_domain':
            course['base']['domain']['jump_m'][1] = .5
        else:
            course['base']['jump_height_gates'][0][0] = .2
        target = course_source(course)
        options = {**options, 'target_course': encoded(course)}
    with pytest.raises(ValueError):
        upgrade_course_sampling(saved, data, replace(schema, source=json.dumps(target)), **options)


def test_pin_layout_pair_and_clock_errors_are_not_hidden(candidate):
    saved, data, target, options = candidate
    with pytest.raises(ValueError, match='source pin'):
        upgrade_course_sampling(saved, data, target, **{**options, 'target_course': options['target_course'] + b' '})
    with pytest.raises(ValueError, match='tensor contract'):
        upgrade_course_sampling(saved, data, replace(target, observation_dim=45), **options)
    with pytest.raises(ValueError, match='paired'):
        upgrade_course_sampling(saved, {**data, 'cycle': saved['cycle'] - 1}, target, **options)
    with pytest.raises(ValueError, match='collection-clock'):
        upgrade_course_sampling({**saved, 'collection_steps': 101}, data, target, **options)


def test_teacher_qualification_cannot_be_relabelled_as_new_native_evidence(candidate):
    saved, data, target, options = candidate
    wrong = deepcopy(saved['teacher_source'])
    wrong['native_source']['runtime_sha256'] = 'unrelated-teacher'
    with pytest.raises(ValueError, match='teacher source'):
        upgrade_course_sampling({**saved, 'teacher_source': wrong}, data, target, **options)
    upgraded, _, report = upgrade_course_sampling(saved, data, target, **options)
    original_source = saved['teacher_source']['qualification']['report']['native_source']
    assert original_source != json.loads(target.source)
    assert upgraded['teacher_source']['qualification']['report']['native_source'] == original_source
    assert compatible_course_source(original_source, json.loads(target.source),
                                    (report['course_sampling'],)) == original_source


def test_registered_sampling_lineage_composes_without_losing_old_teacher(candidate):
    saved, data, target, options = candidate
    first, replay, receipt = upgrade_course_sampling(saved, data, target, **options)
    third_course = json.loads(options['target_course'])
    third_course['base']['stage_weights'][4] = [.03, .20, .07, .30, .30, .10]
    third = replace(target, source=json.dumps(course_source(third_course)))
    upgraded, final_replay, second_receipt = upgrade_course_sampling(
        first, replay, third, source_course=options['target_course'], target_course=encoded(third_course))
    original = json.loads(saved['schema']['source'])
    lineage = (receipt['course_sampling'], second_receipt['course_sampling'])
    assert compatible_course_source(original, json.loads(third.source), lineage) == original
    assert upgraded['teacher_source'] is saved['teacher_source']
    assert final_replay['replay'] is data['replay']


def test_source_verifier_does_not_mutate_raw_configuration(candidate):
    saved, _, target, options = candidate
    source, live = json.loads(saved['schema']['source']), json.loads(target.source)
    before = deepcopy((source, live))
    proof = verify_course_sampling(source, live, **options)
    assert (source, live) == before
    assert proof['source_course_utf8'].encode() == options['source_course']
    assert proof['target_course_utf8'].encode() == options['target_course']


def test_height_only_change_retains_existing_sampler_quota_debts(candidate):
    saved, data, schema, options = candidate
    course = json.loads(options['target_course'])
    course['base']['stage_weights'] = json.loads(options['source_course'])['base']['stage_weights']
    target = replace(schema, source=json.dumps(course_source(course)))
    upgraded, _, report = upgrade_course_sampling(
        saved, data, target, **{**options, 'target_course': encoded(course)})
    assert upgraded['environment']['curriculum']['sampler'] is saved['environment']['curriculum']['sampler']
    assert not report['sampler_credits_reset']


def test_unrecorded_protocols_and_missing_teacher_are_not_aliased(candidate):
    saved, data, target, options = candidate
    with pytest.raises(ValueError, match='task producer'):
        upgrade_course_sampling({**saved, 'config': {**saved['config'], 'record_jumps': False}},
                                data, target, **options)
    with pytest.raises(ValueError, match='task configuration signatures'):
        upgrade_course_sampling({**saved, 'jump_scenario': None}, data, target, **options)
    with pytest.raises(ValueError, match='teacher source'):
        upgrade_course_sampling({**saved, 'teacher_source': None}, data, target, **options)
