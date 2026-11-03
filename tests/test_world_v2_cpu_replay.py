"""CPU-resident online replay preserves data and uses explicit CPU sampling RNGs."""
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.world_model_v2.migrations.actor import restrict_actor_to_chassis
from supermimic_rl.world_model_v2.collection import Collector
from supermimic_rl.world_model_v2.online import OnlineConfig, OnlineTrainer, policy_start_candidates
from supermimic_rl.world_model_v2.replay import CompactReplay

from support.world_v2.chassis_actor import migrated_fixture
from support.world_v2.collection import Environment
from support.world_v2.motion_upgrade import fixture as replay_fixture


def test_legacy_config_keeps_replay_on_learner_and_storage_changes_require_migration():
    assert OnlineConfig.from_dict({}).replay_device == 'learner'
    saved, _, _, _, _, _, _ = replay_fixture()
    trainer = object.__new__(OnlineTrainer)
    trainer.env = SimpleNamespace(schema=SimpleNamespace(fingerprint=saved['schema_fingerprint']))
    trainer.teacher_source = trainer.stair_scenario = trainer.jump_scenario = trainer.recovery_scenario = None
    trainer.config = OnlineConfig(cycles=33)
    trainer._check_saved(saved)
    trainer.config = OnlineConfig(cycles=33, replay_device='cpu')
    with pytest.raises(ValueError, match='training recipe'):
        trainer._check_saved(saved)
    for invalid in ('cuda', '', None, False):
        with pytest.raises(ValueError, match='Replay device'):
            OnlineConfig(replay_device=invalid)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_only_sampled_windows_and_starts_move_to_the_learner(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required for cross-device transfer')
    _, dataset, _, _, _, _, _ = replay_fixture()
    replay = CompactReplay.from_state_dict(dataset['replay'], 'cpu')
    before = {name: value.clone() for name, value in replay.data.items()}
    trainer = object.__new__(OnlineTrainer)
    trainer.config = OnlineConfig(replay_device='cpu')
    trainer.device = torch.device(device)
    trainer.collector = SimpleNamespace(replay=replay)
    trainer.env = SimpleNamespace(policy_dt_s=.02)
    trainer.learner = SimpleNamespace(actor=SimpleNamespace(history_length=5))
    trainer.train_ids = [0, 1]
    trainer.generator = torch.Generator(device='cpu').manual_seed(57)
    expected_rng = torch.Generator(device='cpu').set_state(trainer.generator.get_state())
    candidates = policy_start_candidates(replay, trainer.config, trainer.train_ids, .02)
    expected, expected_frames = replay.sample_starts(16, 5, generator=expected_rng, candidates=candidates)
    actual, frames = trainer.sample_starts(16)
    for name, value in actual.items():
        assert value.device.type == device
        assert torch.equal(value.cpu(), expected[name])
    assert frames.device.type == device and torch.equal(frames.cpu(), expected_frames)
    expected_windows = replay.sample_windows(8, 2, generator=expected_rng, env_ids=trainer.train_ids)
    windows = trainer.sample_windows(8, 2)
    from supermimic_rl.world_model_v2.fitting import WorldModelFit
    assert set(windows) == set(WorldModelFit.RECORD_FIELDS) | (
        set(WorldModelFit.OPTIONAL_RECORD_FIELDS) & set(replay.data))
    for name, value in windows.items():
        assert value.device.type == device
        assert torch.equal(value.cpu(), expected_windows[name])
    assert torch.equal(trainer.generator.get_state(), expected_rng.get_state())
    if device == 'cuda':
        first_buffers = trainer._model_window_staging
        previous = {name: value.clone() for name, value in windows.items()}
        expected_next = replay.sample_windows(8, 2, generator=expected_rng, env_ids=trainer.train_ids)
        following = trainer.sample_windows(8, 2)
        assert trainer._model_window_staging is first_buffers
        for name in following:
            assert torch.equal(following[name].cpu(), expected_next[name])
            assert torch.equal(windows[name], previous[name])
        assert torch.equal(trainer.generator.get_state(), expected_rng.get_state())
    for name, value in replay.data.items():
        assert value.device.type == 'cpu' and torch.equal(value, before[name])


def test_explicit_cpu_branch_keeps_full_replay_and_reseeds_only_sampling():
    saved, dataset, _ = migrated_fixture()
    saved.update(validation_rng=torch.Generator().manual_seed(31).get_state(),
                 collection_rng=torch.Generator().manual_seed(32).get_state(),
                 mixing_rng=torch.Generator().manual_seed(33).get_state())
    upgraded, retained, receipt = restrict_actor_to_chassis(saved, dataset, replay_device='cpu',
        imagination_microbatch_size=4096, sampling_seed=731)
    assert retained is dataset
    assert upgraded['config']['replay_device'] == 'cpu'
    assert upgraded['config']['imagination_microbatch_size'] == 4096
    assert 'replay_device' not in saved['config']
    assert receipt['replay_storage']['replay_bytes_unchanged']
    assert receipt['policy_microbatch']['total_batch_unchanged'] == 4096
    for name, seed in (('sample', 731), ('validation', 732)):
        assert upgraded[name + '_rng_device'] == 'cpu'
        assert torch.equal(upgraded[name + '_rng'], torch.Generator().manual_seed(seed).get_state())
    for name in ('collection_rng', 'mixing_rng'):
        assert torch.equal(upgraded[name], saved[name])
    trainer = object.__new__(OnlineTrainer)
    trainer.config = OnlineConfig.from_dict(upgraded['config'])
    trainer.generator, trainer.validation_generator = torch.Generator(), torch.Generator()
    trainer._restore_sampling_rng(upgraded)
    assert torch.equal(trainer.generator.get_state(), upgraded['sample_rng'])
    damaged = {**upgraded, 'sample_rng_device': 'cuda:0'}
    with pytest.raises(ValueError, match='explicit storage migration'):
        trainer._restore_sampling_rng(damaged)
    damaged = {key: value for key, value in upgraded.items() if key != 'sample_rng_device'}
    with pytest.raises(ValueError, match='explicit storage migration'):
        trainer._restore_sampling_rng(damaged)


@pytest.mark.parametrize('options', [
    {'replay_device': 'cpu'}, {'replay_device': 'cpu', 'sampling_seed': -1},
    {'sampling_seed': 4}, {'imagination_microbatch_size': 0},
])
def test_storage_migration_never_reinterprets_an_old_rng_implicitly(options):
    saved, dataset, _ = migrated_fixture()
    with pytest.raises(ValueError):
        restrict_actor_to_chassis(saved, dataset, **options)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA collection transfer required')
def test_cuda_collection_appends_cpu_records_and_boundaries_without_joining_episodes(monkeypatch):
    monkeypatch.setattr('supermimic_rl.world_model_v2.collection.SnapshotLayout',
        lambda schema: SimpleNamespace(encode=lambda transition: {'frame': transition.start.actor_observation}))

    class CudaEnvironment(Environment):
        device = 'cuda:0'

        def snapshot(self):
            value = super().snapshot()
            return SimpleNamespace(**{name: tensor.cuda() for name, tensor in vars(value).items()})

        def step(self, action):
            value = super().step(action)
            for name, tensor in vars(value).items():
                if isinstance(tensor, torch.Tensor):
                    setattr(value, name, tensor.cuda())
            return value

    replay = CompactReplay({'frame': (45,)}, 2, 16, 'cpu')
    decision = lambda history: (history.new_zeros(2, 6), history.new_zeros(2, 5))
    collector = Collector(CudaEnvironment(), decision, capacity_steps=16, replay=replay)
    collector.collect(5)
    baseline = Collector(Environment(), decision, capacity_steps=16)
    baseline.collect(5)
    for name, value in replay.data.items():
        assert value.device.type == 'cpu'
        assert torch.equal(value, baseline.replay.data[name]), name
    restarted = Collector(CudaEnvironment(), decision, capacity_steps=16, replay=replay)
    restarted.collect(1)
    assert (replay.data['segment_age'][5] == 0).all()
    assert (replay.data['start_episode_id'][5] > replay.data['end_episode_id'][4]).all()
