"""Opt-in platform records and independently sampled native contact supervision."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.world_model_v2.online import OnlineConfig, OnlineTrainer, policy_start_candidates
from supermimic_rl.world_model_v2.platform_contacts import PLATFORM_CONTACT_PRODUCER
from supermimic_rl.world_model_v2.replay import CompactReplay
from test_world_v2_platform_learning import fixture, assert_equal


def config(**values):
    defaults = dict(record_tasks=True, record_stair_source=True, record_jumps=True,
        course_coordinate=0, record_platform=True, platform_contact_hidden=(8,),
        model_batch_size=8, replay_device='cpu')
    return OnlineConfig(**(defaults | values))


def ring(sample, steps=2):
    record = {name: value for name, value in sample.items()
              if name not in ('observation_noise', 'tof_latency', 'tof_uniform', 'jump_uniform')}
    record['frame'] = torch.zeros(2, 45)
    replay = CompactReplay({name: value.shape[1:] for name, value in record.items()}, 2, steps, 'cpu')
    boundary = {name: torch.ones(2, dtype=torch.bool) for name in ('valid', 'policy_owned', 'end_policy_owned')}
    boundary.update({name: torch.zeros(2, dtype=torch.bool) for name in ('terminated', 'truncated')})
    boundary.update({name: torch.zeros(2, dtype=torch.long) for name in
        ('start_episode_id', 'end_episode_id', 'start_control_epoch', 'end_control_epoch')})
    for _ in range(steps):
        replay.add(record, boundary)
    return replay


def trainer(tmp_path):
    from supermimic_rl.world_model_v2.learning import Learner
    from supermimic_rl.world_model_v2.platform_contacts import PlatformContactModel

    learner, sample, _ = fixture(tmp_path)
    previous = learner
    prior = previous.imagination
    contact = previous.platform_contact
    contact = PlatformContactModel(contact.wheel_geometry, contact.geometry, contact.clearance, hidden=(8,))
    learner = Learner(previous.model, previous.actor, previous.critic, prior.termination, previous.config,
        reward=prior.reward, stair_task=prior.stair_task, stair_source=prior.stair_source, jump_task=prior.jump_task,
        platform_task=prior.platform_task, platform_contact=contact, platform_geometry=prior.platform_geometry)
    sample['goal'][:, 3] = 2.  # The admitted command, independent of intent/anchor height.
    sample['jump_values'][:, 0] = 123.  # No command-height inference from task state.
    obj = object.__new__(OnlineTrainer)
    obj.config, obj.learner, obj.device = config(), learner, torch.device('cpu')
    obj.train_ids, obj.validation_ids = [0], [1]
    obj.collector = SimpleNamespace(replay=ring(sample))
    obj.platform_generator = torch.Generator().manual_seed(9)
    obj.generator = torch.Generator().manual_seed(10)
    obj.validation_generator = torch.Generator().manual_seed(11)
    return obj


def test_defaults_and_explicit_width_configuration():
    assert not OnlineConfig().record_platform
    assert OnlineConfig().platform_contact_hidden is None
    assert OnlineConfig.from_dict(asdict(config())) == config()
    for values in ({'record_platform': True}, {'platform_contact_hidden': (8,)},
                   {'record_platform': 1}, {'initial_platform_contact': 'unpaired.pt'}):
        with pytest.raises(ValueError):
            OnlineConfig(**values)
    with pytest.raises(ValueError, match='widths'):
        config(platform_contact_hidden=(0,))


def test_native_intervals_fit_only_the_head_with_independent_rng_and_no_aux(tmp_path):
    obj = trainer(tmp_path)
    replay = obj.collector.replay
    replay.data['ticks'][:, 1].fill_(float('nan'))  # Held-out truth must never enter fitting.
    replay.data['policy_owned'][:, 0] = False  # A real teacher transition is still a physics label.
    obj.auxiliary_world = SimpleNamespace(mix=lambda *_: pytest.fail('Contact labels came from old auxiliary data'))
    obj.sample_windows = lambda *_: pytest.fail('Contact fitting reused motion windows')
    before = deepcopy(obj.learner.state_dict())
    sample_rng = obj.generator.get_state().clone()
    replay_before = deepcopy(replay.state_dict())
    outcome = obj.fit_platform_contact()
    assert outcome['platform_contact/updated'] and outcome['platform_contact/labels'] == 32
    assert obj.learner.platform_contact.input_norm.count == 32
    assert_equal(sample_rng, obj.generator.get_state())
    assert_equal(replay_before, replay.state_dict())
    for name in ('actor', 'critic', 'model'):
        assert_equal(before[name], obj.learner.state_dict()[name])
    for name in ('actor', 'critic', 'model', 'estimator'):
        assert_equal(before['optimizers'][name], obj.learner.optimizers[name].state_dict())


@pytest.mark.parametrize('missing', ['platform_record_valid', 'contact_valid_ticks', 'selected'])
def test_unknown_or_inactive_intervals_do_not_draw_or_update(tmp_path, missing):
    obj = trainer(tmp_path)
    if missing == 'selected':
        obj.collector.replay.data['platform_evidence_ticks'][..., 0] = False
    else:
        obj.collector.replay.data[missing].zero_()
    before = deepcopy(obj.learner.state_dict())
    rng = obj.platform_generator.get_state().clone()
    assert not obj.fit_platform_contact()['platform_contact/updated']
    assert_equal(obj.learner.state_dict(), before)
    assert_equal(obj.platform_generator.get_state(), rng)


def test_policy_start_cache_separates_unknown_platform_records_from_legacy_mode(tmp_path):
    obj = trainer(tmp_path)
    replay = obj.collector.replay
    replay.data['platform_record_valid'][:, 1] = False
    legacy = replace(obj.config, record_platform=False, platform_contact_hidden=None)
    # This fixture has a known command segment and no future resampling.
    old = policy_start_candidates(replay, legacy, [0, 1], .02)
    new = policy_start_candidates(replay, obj.config, [0, 1], .02)
    assert old.numel() and (old % 2 == 1).any()
    assert new.numel() and (new % 2 == 0).all()
    assert_equal(policy_start_candidates(replay, legacy, [0, 1], .02), old)


def test_resume_restores_head_draws_without_consuming_motion_rng(tmp_path):
    first = trainer(tmp_path)
    first.fit_platform_contact()
    saved = {name + '_rng': generator.get_state() for name, generator in
             [('sample', first.generator), ('validation', first.validation_generator),
              ('platform_sample', first.platform_generator)]}
    saved.update({name + '_rng_device': 'cpu' for name in ('sample', 'validation', 'platform_sample')})
    child = tmp_path / 'other'
    child.mkdir()
    second = trainer(child)
    second.learner.load_state_dict(deepcopy(first.learner.state_dict()))
    second._restore_sampling_rng(saved)
    assert first.fit_platform_contact() == second.fit_platform_contact()
    assert_equal(first.learner.state_dict(), second.learner.state_dict())
    assert_equal(first.platform_generator.get_state(), second.platform_generator.get_state())
    missing = {key: value for key, value in saved.items() if key != 'platform_sample_rng'}
    with pytest.raises(KeyError):
        second._restore_sampling_rng(missing)


def test_owner_requires_exact_source_and_rejects_collection_only_future_commands():
    obj = object.__new__(OnlineTrainer)
    obj.device = torch.device('cpu')
    native = SimpleNamespace()
    receipt = {'version': 'fixture'}
    terrain = SimpleNamespace(manifest_sha256='atlas', origins=torch.zeros(1, 3))
    owner = SimpleNamespace(native=native, terrain=terrain, geometry=SimpleNamespace(terrain=terrain),
        receipt=lambda: receipt, recording_options=lambda: {'record_platform': True})
    native.supermimic_platform_course = owner
    source = {'platform_task': receipt, 'terrain_sha256': 'atlas'}
    obj.env = SimpleNamespace(adapter=SimpleNamespace(env=native), schema=SimpleNamespace(source=json.dumps(source)))
    assert obj._platform_owner() is owner
    source['contact_collection'] = {'joint_actor_starts_allowed': False}
    obj.env.schema.source = json.dumps(source)
    with pytest.raises(ValueError, match='future command'):
        obj._platform_owner()
    del source['contact_collection']
    source['platform_task'] = {'version': 'other'}
    obj.env.schema.source = json.dumps(source)
    with pytest.raises(ValueError, match='source-matched'):
        obj._platform_owner()


def test_optional_artifact_initializes_only_head_and_records_its_original_source(tmp_path):
    from supermimic_rl.world_model_v2.leg_linkage import file_sha256

    obj = trainer(tmp_path)
    table = tmp_path / 'table.json'
    table.write_text('same linkage bytes')
    pins = {'linkage_sha256': file_sha256(table), 'asset_manifest_sha256': 'asset',
            'terrain_sha256': 'terrain', 'catalog_sha256': 'catalog'}
    obj.env = SimpleNamespace(schema=SimpleNamespace(source=json.dumps({**pins,
                             'platform_task': {'catalog_sha256': 'catalog'}})))
    source = {'original_dataset': 'native-contact-only'}
    artifact = {'format': 'supermimic-platform-contact-head-v1', 'producer': PLATFORM_CONTACT_PRODUCER,
                'recipe': {'hidden': [8]}, 'geometry_pins': pins, 'model': obj.learner.platform_contact.state_dict(),
                'inputs': source, 'split_sha256': 'original-independent-split'}
    path = tmp_path / 'head.pt'
    torch.save(artifact, path)
    obj.config = replace(obj.config, initial_platform_contact=str(path), initial_platform_contact_sha256=file_sha256(path))
    before = deepcopy(obj.learner.state_dict())
    obj._initialize_platform_contact(table)
    assert_equal(obj.learner.state_dict(), before)
    assert obj.platform_contact_initialization['source_inputs'] == source
    artifact['geometry_pins']['asset_manifest_sha256'] = 'different-robot'
    torch.save(artifact, path)
    obj.config = replace(obj.config, initial_platform_contact_sha256=file_sha256(path))
    with pytest.raises(ValueError, match='identity'):
        obj._initialize_platform_contact(table)


def test_recipe_does_not_silently_upgrade_legacy_platform_history():
    from test_world_v2_horizon_resume import check_fixture

    obj, saved = check_fixture(16, 16)
    previous = replace(config(), record_platform=False, platform_contact_hidden=None)
    saved['config'], obj.config = asdict(previous), config()
    with pytest.raises(ValueError, match='training recipe'):
        obj._check_saved(saved)


def test_real_factory_reuses_reward_geometry_and_platform_atlas(tmp_path):
    from supermimic_rl.contracts import Field, StateSchema
    from supermimic_rl.world_model_v2.factory import make_learner
    from support import REPO_ROOT

    prior, _, _ = fixture(tmp_path)
    imagined = prior.imagination
    reward = imagined.reward
    source = {'asset_manifest_sha256': reward.geometry.linkage.provenance['asset_manifest_sha256'],
        'terrain_sha256': prior.model.terrain.manifest_sha256,
        'reward_config': {'robot_mass_kg': reward.tuning.robot_mass_kg, 'gamma': reward.tuning.gamma,
            'profile': reward.tuning.profile, 'physics_dt_s': .005, 'available_terms': list(reward.terms)},
        'terrain_support_config': asdict(reward.support_config)}
    schema = StateSchema((Field('fixture', (1,)),), (), goal_dim=16, observation_dim=35,
        substep_fields=(Field('fixture_tick', (1,)),), substeps_per_step=4, source=json.dumps(source))
    runtime = json.loads((REPO_ROOT / 'configs/gpufree/runtime-physx64-16-geometry-audit.json').read_text())
    learner = make_learner(schema, runtime, prior.config,
        linkage_path=REPO_ROOT / 'configs/linkage/leg_linkage_e726d23c.json',
        asset_directory=REPO_ROOT / 'assets/full_vehicle_v2p0p1_source_audited_v2',
        identified_bundle=REPO_ROOT / 'configs/identified/20261031', device='cpu',
        model_hidden=(8,), actor_hidden=(8,), estimator_hidden=(8,), critic_hidden=(8,),
        actor_frame_dim=35, record_tasks=True, terrain=prior.model.terrain,
        stair_task=imagined.stair_task, stair_source=imagined.stair_source, jump_task=imagined.jump_task,
        platform_task=imagined.platform_task, platform_geometry=imagined.platform_geometry,
        platform_contact_hidden=(8,))
    assert learner.platform_contact.wheel_geometry is learner.imagination.reward.geometry
    assert learner.platform_contact.geometry is imagined.platform_geometry
    assert learner.platform_contact.clearance.terrain is prior.model.terrain
    assert set(learner.optimizers) == {'model', 'actor', 'critic', 'estimator', 'platform_contact'}


def test_complete_checkpoint_carries_head_rng_and_collector_owner_without_reinitializing(tmp_path, monkeypatch):
    from supermimic_rl.contracts import Field, StateSchema
    from supermimic_rl.world_model_v2.jump_task import JumpTaskConfig
    from supermimic_rl.world_model_v2.records import MODEL_CONTRACT, RecordSpec
    from supermimic_rl.world_model_v2.online import CHECKPOINT_FORMAT

    base = trainer(tmp_path)
    base.fit_platform_contact()
    requested = config(cycles=10, replay_steps=8, validation_fraction=.5,
        initial_platform_contact='not-reloaded-on-resume.pt', initial_platform_contact_sha256='a' * 64)
    schema = StateSchema((Field('fixture', (1,)),), (), goal_dim=16, observation_dim=45,
        substep_fields=(Field('fixture_tick', (1,)),), substeps_per_step=4, source='{}')
    fields = RecordSpec(16, 65).fields(record_tasks=True, record_terrain=True, record_stairs=True,
        record_stair_source=True, record_jumps=True, record_jump_trigger=True, record_platform=True)
    fields.update({name: value.shape[2:] for name, value in base.collector.replay.data.items()})
    replay = CompactReplay(fields, 2, 8, 'cpu')
    # Complete real typed records are required; this fixture fills only unrelated
    # motion endpoints that the isolated head test never used.
    for name, value in base.collector.replay.data.items():
        replay.data[name][:2].copy_(value[:2])
    replay.sequence[:2] = torch.arange(2)
    replay.size = replay.head = replay.total_steps = 2
    jump_scenario = asdict(base.learner.imagination.jump_task.config)
    saved = {'format': CHECKPOINT_FORMAT, 'model_contract': MODEL_CONTRACT,
        'schema': schema.to_dict(), 'schema_fingerprint': schema.fingerprint, 'config': asdict(requested),
        'cycle': 2, 'replay_file': 'replay.pt', 'learner': deepcopy(base.learner.state_dict()),
        'environment': {'curriculum': {'real_transitions': 0}}, 'collection_steps': 2,
        'seed_source': {'source': 'current-complete-platform-seed'}, 'teacher_source': None,
        'stair_scenario': {'fixture': 'same'}, 'jump_scenario': jump_scenario, 'recovery_scenario': None,
        'platform_contact_initialization': {'method': 'original-artifact', 'sha256': 'retained'},
        'sample_rng': base.generator.get_state(), 'sample_rng_device': 'cpu',
        'validation_rng': base.validation_generator.get_state(), 'validation_rng_device': 'cpu',
        'platform_sample_rng': base.platform_generator.get_state(), 'platform_sample_rng_device': 'cpu',
        'collection_rng': torch.Generator().manual_seed(12).get_state(),
        'mixing': {'format': 'teacher-population-v1', 'priority': torch.tensor([1, 0]),
                   'rng': torch.Generator().manual_seed(13).get_state()},
        'torch_rng': torch.Generator().manual_seed(14).get_state(), 'cuda_rng': []}
    path = tmp_path / 'resume.pt'
    torch.save(saved, path)
    torch.save({'cycle': 2, 'schema_fingerprint': schema.fingerprint, 'replay': replay.state_dict()},
               tmp_path / 'replay.pt')
    owner = SimpleNamespace(terrain=base.learner.model.terrain,
        geometry=base.learner.platform_contact.geometry, landing=object())
    owner.recording_options = lambda: {'record_platform': True, 'platform_landing': owner.landing}
    obj_env = SimpleNamespace(device=torch.device('cpu'), num_envs=2, schema=schema,
        current=SimpleNamespace(actor_observation=torch.zeros(2, 45), episode_id=torch.zeros(2).long()),
        adapter=SimpleNamespace(env=object()), load_training_state_dict=lambda state: None,
        training_state_dict=lambda: saved['environment'])
    scenario = SimpleNamespace(to_dict=lambda: saved['stair_scenario'],
        build=lambda terrain: (base.learner.imagination.stair_task, base.learner.imagination.stair_source))
    monkeypatch.setattr('supermimic_rl.world_model_v2.online.StairScenario.capture', lambda _: scenario)
    monkeypatch.setattr(JumpTaskConfig, 'capture', lambda _: base.learner.imagination.jump_task.config)
    monkeypatch.setattr(OnlineTrainer, '_platform_owner', lambda self: owner)
    monkeypatch.setattr(OnlineTrainer, '_initialize_platform_contact',
                        lambda *_: pytest.fail('Complete resume reinitialized its contact artifact'))

    def build(*args, **kwargs):
        assert kwargs['terrain'] is owner.terrain and kwargs['platform_geometry'] is owner.geometry
        assert kwargs['platform_task'].jump_task is kwargs['jump_task']
        assert kwargs['platform_contact_hidden'] == (8,)
        return deepcopy(base.learner)

    def collect(environment, decision, **kwargs):
        assert kwargs['record_platform'] and kwargs['platform_landing'] is owner.landing
        return SimpleNamespace(replay=kwargs['replay'], generator=torch.Generator(), steps=0, episode_offset=0)

    monkeypatch.setattr('supermimic_rl.world_model_v2.online.make_learner', build)
    monkeypatch.setattr('supermimic_rl.world_model_v2.online.Collector', collect)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    restored = OnlineTrainer(obj_env, lambda: None, requested, tmp_path / 'complete', runtime={},
        linkage_path=None, asset_directory=None, identified_bundle=None, resume=path)
    assert_equal(restored.learner.state_dict(), saved['learner'])
    assert_equal(restored.platform_generator.get_state(), saved['platform_sample_rng'])
    assert_equal(torch.get_rng_state(), saved['torch_rng'])
    assert_equal(restored.collector.replay.state_dict(), replay.state_dict())
    assert restored.platform_contact_initialization == saved['platform_contact_initialization']
    restored.save(with_replay=True)
    pointer = json.loads((restored.output / 'resume.json').read_text())
    written = torch.load(restored.output / pointer['checkpoint'], weights_only=True)
    assert_equal(written['learner'], saved['learner'])
    assert_equal(written['platform_sample_rng'], saved['platform_sample_rng'])
    assert written['platform_sample_rng_device'] == 'cpu'
    assert written['platform_contact_initialization'] == saved['platform_contact_initialization']
