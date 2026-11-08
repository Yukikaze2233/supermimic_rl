"""A ToF source-policy migration preserves learned and historical truth."""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json

import pytest
import torch

from supermimic_rl.cli import _native_schema_source
from supermimic_rl.contracts import Field, StateSchema
from supermimic_rl.course.jump_first import make_course_curriculum
from supermimic_rl.course.stair_retry import timeout_retry_receipt
from supermimic_rl.world_model_v2.auxiliary_data import AuxiliaryWorldData, WorldDatasetConfig
from supermimic_rl.world_model_v2.migrations.stair_retry import (
    FORMAT, upgrade_stair_timeout_retry, verify_saved_proof, verify_stair_timeout_retry,
)
from supermimic_rl.world_model_v2.records import MODEL_CONTRACT, RecordSpec
from supermimic_rl.world_model_v2.replay import CompactReplay
from supermimic_rl.world_model_v2.stair_source import StairScenario
from support import REPO_ROOT
from support.world_v2.stair_source import source_fixture
from support.world_v2.stair_task import native_stairs


def assert_identical(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_identical(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for first, second in zip(actual, expected):
            assert_identical(first, second)
    else:
        assert actual == expected


@pytest.fixture
def candidate(tmp_path):
    runtime_raw = (REPO_ROOT / 'configs/gpufree/runtime-physx64-16-stair-hint.json').read_bytes()
    course_raw = (REPO_ROOT / 'configs/gpufree/ppo-course-gated-v7.json').read_bytes()
    course = json.loads(course_raw)
    source = {
        'runtime_sha256': hashlib.sha256(runtime_raw).hexdigest(),
        'course_sha256': hashlib.sha256(course_raw).hexdigest(), 'curriculum': course,
        'course_schedule': {'schedule': course['schedule'], 'request_heading_override': False,
                            'early_jump_domain': course['base']['initial_domain']['jump_m']},
        'asset_manifest_sha256': 'a' * 64, 'terrain_sha256': 'b' * 64, 'upstream_commit': 'c' * 40,
        'action_bounds': [3., 3., 3., 3., 9., 9.],
        'reward_kernel': {'implementation': 'shared_physical_reward_densities', 'kernel_sha256': 'd' * 64},
        'recovery_roll_bridge': {'contract': 'prior-unchanged-bridge-fixture'},
    }
    schema = StateSchema((Field('motion', (3,)),), (), goal_dim=16, observation_dim=45,
                         substep_fields=(Field('physical_tick', (3,)),), substeps_per_step=4,
                         source=_native_schema_source(source))
    target = replace(schema, source=_native_schema_source({**source, 'stair_timeout_retry': timeout_retry_receipt()}))
    native = native_stairs(4, labels=True, assist=True)
    _, native._terrain_course.tof_trigger = source_fixture(4)
    scenario = StairScenario.capture(native)
    fields = RecordSpec(16, 65).fields(record_tasks=True, record_stairs=True, record_stair_source=True,
                                      record_terrain=True, record_jumps=True, record_recovery=True)
    replay = CompactReplay(fields, 4, 8, 'cpu')
    boundary = {k: torch.zeros(4, dtype=torch.bool) for k in ('terminated', 'truncated')}
    boundary.update({k: torch.ones(4, dtype=torch.bool) for k in ('valid', 'policy_owned', 'end_policy_owned')})
    boundary.update({k: torch.zeros(4, dtype=torch.int64) for k in
        ('start_episode_id', 'end_episode_id', 'start_control_epoch', 'end_control_epoch')})
    for step in range(8):
        record = {key: torch.zeros(4, *shape, dtype=replay.data[key].dtype) for key, shape in fields.items()}
        record['physical'][:, 6] = record['end_physical'][:, 6] = 1.
        record['action'][:3] = step + 100
        record['action'][3] = step + 1000  # Independent validation row, never a training draw.
        record['stair_source_valid'].fill_(1.)
        record['tof_flags'].fill_(False)  # Old same-riser lock is historical truth.
        record['stair_ids'][:, 0] = 2**54 + step
        record['stair_ids'][:, 1] = 2**54 + step
        record['stair_clock'].fill_(.02 * step)
        record['stair_parameters'].fill_(.2)
        record['tof_parameters'].fill_(.3)
        record['jump_values'].fill_(.4)
        record['reward'].fill_(-.125)
        replay.add(record, boundary)
    data = {'cycle': 1109, 'schema': schema.to_dict(), 'schema_fingerprint': schema.fingerprint,
            'replay': replay.state_dict()}
    data['replay']['storage']['stair_source_valid'][0, 0] = 0.
    aux_path = tmp_path / 'old-independent.pt'
    torch.save({'format': 'supermimic-world-v2-dataset-v1', 'model_contract': MODEL_CONTRACT,
                'schema': schema.to_dict(), 'schema_fingerprint': schema.fingerprint,
                'replay': replay.state_dict()}, aux_path)
    aux_config = WorldDatasetConfig('student750', str(aux_path),
        hashlib.sha256(aux_path.read_bytes()).hexdigest(), .0610809673, (0, 1, 2), (3,))
    auxiliary = AuxiliaryWorldData((aux_config,), schema, replay, 2, seed=31)
    auxiliary.mix(replay.sample_windows(32, 2, env_ids=[0, 1, 2],
        generator=torch.Generator().manual_seed(17), field_names=auxiliary.fields))
    curriculum = make_course_curriculum(4, 'cpu', 19, course)
    curriculum.set_progress(2_100_000_000, 1109)
    curriculum.sample(torch.arange(4))  # Nonzero quotas and real sampler state.
    optimizers = {}
    for i, name in enumerate(('actor', 'critic', 'model', 'estimator')):
        parameter = torch.nn.Parameter(torch.tensor([float(i + 1)]))
        optimizer = torch.optim.Adam([parameter], lr=5e-5 * (i + 1))
        parameter.square().sum().backward()
        optimizer.step()
        optimizers[name] = optimizer.state_dict()
    saved = {'format': 'supermimic-online-v2', 'model_contract': MODEL_CONTRACT, 'cycle': 1109,
        'schema': schema.to_dict(), 'schema_fingerprint': schema.fingerprint, 'replay_file': 'replay-a.pt',
        'config': {'record_tasks': True, 'record_stair_source': True, 'record_jumps': True,
                   'record_recovery': True, 'world_datasets': [asdict(aux_config)]},
        'collection_steps': 26616,
        'learner': {'model': {'weight': torch.arange(7.), 'norm': torch.tensor([.25, 2.])},
                    'actor': {'weight': torch.arange(3.)}, 'critic': {'weight': torch.arange(5.)},
                    'estimator': {'weight': torch.arange(4.)}, 'optimizers': optimizers,
                    'updates': {'actor': 26616, 'model': 106464, 'critic': 532320, 'estimator': 26616},
                    'skipped': dict.fromkeys(optimizers, 0)},
        'environment': {'format': 'supermimic_course_v1', 'curriculum': curriculum.state_dict(),
                        'course': {'stair_next_id': torch.tensor([2**54 + 8] * 4),
                                   'tof_trigger': {'rng': native._terrain_course.tof_trigger.generator.get_state()}},
                        'control_timing': {'clock': 543}, 'sensors': {'rng': torch.get_rng_state()}},
        'course_progression': {'coordinate': 2_100_000_000, 'history': [{'opened': False, 'count': 31}]},
        'teacher_source': {'native_source': deepcopy(source), 'teacher_artifact': {'sha256': 'e' * 64},
                           'qualified_for': ['original-data-only'], 'qualification': {'report': {
                               'native_source': deepcopy(source), 'steps': 384}}},
        'stair_scenario': scenario.to_dict(), 'jump_scenario': {'max_height_m': .4},
        'recovery_scenario': {'postures': 'existing'},
        'torch_rng': torch.get_rng_state(), 'cuda_rng': [],
        'sample_rng': torch.Generator().manual_seed(23).get_state(),
        'validation_rng': torch.Generator().manual_seed(29).get_state(),
        'collection_rng': torch.Generator().manual_seed(37).get_state(),
        'mixing': {'rng': torch.Generator().manual_seed(41).get_state()},
        'auxiliary_world': auxiliary.state_dict(), 'resume_migrations': [{'older': 'retain verbatim'}]}
    options = {'target_scenario': replace(scenario, timeout_retry=True).to_dict(),
               'runtime_raw': runtime_raw, 'course_raw': course_raw}
    return saved, data, target, options


def test_full_pair_only_loses_old_source_start_eligibility(candidate):
    saved, data, target, options = candidate
    original = deepcopy((saved, data))
    rng = torch.get_rng_state().clone()
    updated, migrated, report = upgrade_stair_timeout_retry(saved, data, target, **options)
    assert torch.equal(torch.get_rng_state(), rng)
    changed = {'schema', 'schema_fingerprint', 'stair_scenario', 'resume_migrations'}
    for key in saved.keys() - changed:
        assert updated[key] is saved[key], key
    assert updated['resume_migrations'][:-1] == saved['resume_migrations']
    assert report is updated['resume_migrations'][-1]
    assert report['migration'] == FORMAT and report['additional_warmup_cycles'] == 0
    assert report['legacy_stair_starts_excluded'] == 31
    assert report['teacher_evaluation_scope_unchanged'] and report['pending_integration']
    assert report['target_schema_origin'].endswith('no new native evaluation')
    assert updated['schema_fingerprint'] == migrated['schema_fingerprint'] == target.fingerprint
    assert StateSchema.from_dict(migrated['schema']) == target
    for name, value in data['replay']['storage'].items():
        if name != 'stair_source_valid':
            assert migrated['replay']['storage'][name] is value, name
    assert migrated['replay']['sequence'] is data['replay']['sequence']
    assert not migrated['replay']['storage']['stair_source_valid'].any()
    assert data['replay']['storage']['stair_source_valid'].sum() == 31
    assert_identical((saved, data), original)
    assert verify_saved_proof(json.loads(json.dumps(report['stair_timeout_retry']))) == report['stair_timeout_retry']


def test_physical_windows_aux_split_next_draw_and_quota_clocks_are_preserved(candidate):
    saved, data, target, options = candidate
    updated, migrated, _ = upgrade_stair_timeout_retry(saved, data, target, **options)
    before = CompactReplay.from_state_dict(data['replay'], read_only=True)
    after = CompactReplay.from_state_dict(migrated['replay'], read_only=True)
    a = torch.Generator().set_state(saved['sample_rng'])
    b = torch.Generator().set_state(updated['sample_rng'])
    fields = ('physical', 'action', 'reward', 'ticks', 'tof_flags', 'stair_ids', 'stair_clock', 'jump_values')
    torch.testing.assert_close(before.sample_windows(32, 2, env_ids=[0, 1, 2], generator=a, field_names=fields),
                               after.sample_windows(32, 2, env_ids=[0, 1, 2], generator=b, field_names=fields),
                               rtol=0, atol=0)
    assert torch.equal(a.get_state(), b.get_state())
    schema = StateSchema.from_dict(saved['schema'])
    config = WorldDatasetConfig(**saved['config']['world_datasets'][0])
    auxiliaries = [AuxiliaryWorldData((config,), schema, replay, 2, seed=999) for replay in (before, after)]
    for auxiliary, checkpoint in zip(auxiliaries, (saved, updated)):
        auxiliary.load_state_dict(checkpoint['auxiliary_world'])
    start = before.sample_windows(64, 2, env_ids=[0, 1, 2], generator=a, field_names=auxiliaries[0].fields)
    torch.testing.assert_close(auxiliaries[0].mix(deepcopy(start)), auxiliaries[1].mix(deepcopy(start)), rtol=0, atol=0)
    torch.testing.assert_close(auxiliaries[0].state_dict(), auxiliaries[1].state_dict(), rtol=0, atol=0)
    assert config.validation_env_ids == (3,) and config.train_env_ids == (0, 1, 2)
    assert updated['environment']['curriculum'] is saved['environment']['curriculum']
    assert updated['course_progression'] is saved['course_progression']
    # The pure transform does not install a source alias or forge qualification.
    with pytest.raises(ValueError, match='contract differs'):
        AuxiliaryWorldData((config,), target, after, 2, seed=999)


@pytest.mark.parametrize('field', ['runtime_sha256', 'course_sha256', 'terrain_sha256',
                                  'asset_manifest_sha256', 'upstream_commit', 'reward_kernel',
                                  'action_bounds', 'course_schedule', 'recovery_roll_bridge'])
def test_source_proof_rejects_every_unrelated_change(candidate, field):
    saved, data, target, options = candidate
    changed = json.loads(target.source)
    changed[field] = {'unrelated': True}
    with pytest.raises(ValueError, match='Only the fixed ToF'):
        upgrade_stair_timeout_retry(saved, data, replace(target, source=json.dumps(changed)), **options)


@pytest.mark.parametrize('change', ['tof', 'task', 'scope', 'already_enabled', 'missing_format', 'no_enable'])
def test_scenario_proof_only_enables_the_existing_disabled_policy(candidate, change):
    saved, data, target, options = candidate
    options = deepcopy(options)
    if change == 'tof':
        options['target_scenario']['tof']['sample_spacing_m'] *= 2
    elif change == 'task':
        options['target_scenario']['task']['request']['timeout_s'] *= 2
    elif change == 'scope':
        options['target_scenario']['scope'] = 'a new evaluation'
    elif change == 'already_enabled':
        saved = {**saved, 'stair_scenario': options['target_scenario']}
    elif change == 'missing_format':
        options['target_scenario']['format'] = 'supermimic-native-stair-source-v1'
    else:
        options['target_scenario'] = saved['stair_scenario']
    with pytest.raises(ValueError):
        upgrade_stair_timeout_retry(saved, data, target, **options)


@pytest.mark.parametrize('raw', ['runtime_raw', 'course_raw'])
def test_original_input_bytes_must_match_their_pins(candidate, raw):
    saved, data, target, options = candidate
    with pytest.raises(ValueError, match='pinned'):
        upgrade_stair_timeout_retry(saved, data, target, **{**options, raw: options[raw] + b' '})


@pytest.mark.parametrize('change', ['cycle', 'replay_source', 'missing_adam', 'layout', 'validity', 'no_stairs'])
def test_incomplete_or_different_pair_is_not_migrated(candidate, change):
    saved, data, target, options = candidate
    saved, data = deepcopy(saved), deepcopy(data)
    if change == 'cycle':
        data['cycle'] -= 1
    elif change == 'replay_source':
        data['schema_fingerprint'] = 'different'
    elif change == 'missing_adam':
        del saved['learner']['optimizers']['critic']
    elif change == 'layout':
        target = replace(target, goal_dim=17)
    elif change == 'validity':
        del data['replay']['storage']['stair_source_valid']
    else:
        saved['config']['record_stair_source'] = False
    with pytest.raises(ValueError):
        upgrade_stair_timeout_retry(saved, data, target, **options)


@pytest.mark.parametrize('change', ['format', 'extra', 'receipt', 'course', 'before', 'missing'])
def test_saved_proof_cannot_hide_additions_or_forge_evidence(candidate, change):
    saved, _, target, options = candidate
    proof = verify_stair_timeout_retry(json.loads(saved['schema']['source']), json.loads(target.source),
        saved['stair_scenario'], **options)
    if change == 'format':
        proof['format'] = 'course-sampling-only-v1'
    elif change == 'extra':
        proof['native_evaluation_passed'] = True
    elif change == 'receipt':
        proof['after']['stair_timeout_retry']['next_request'] = 'reuse the old ID'
    elif change == 'course':
        proof['course_utf8'] += '\n'
    elif change == 'before':
        proof['before']['terrain_sha256'] = 'other'
    else:
        del proof['source_scenario']
    with pytest.raises(ValueError):
        verify_saved_proof(proof)
