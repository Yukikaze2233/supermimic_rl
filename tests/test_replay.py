"""Behavioral tests for replay chronology, ownership and persistence."""
from dataclasses import fields, replace
import os
from pathlib import Path
import sys

import pytest
import torch

from supermimic_rl.contracts import Field, Snapshot, StateSchema, Transition
from supermimic_rl.replay import ReplayBuffer


@pytest.fixture
def schema():
    return StateSchema((Field("position", (2,), "m", "world"),),
                       (Field("friction", (1,)),), source="replay-test")


def transition(step, num_envs=2, *, episode=None, epoch=None, owned=None):
    episode = (
        torch.zeros(num_envs, dtype=torch.int64)
        if episode is None else torch.tensor(episode)
    )
    epoch = torch.zeros(num_envs, dtype=torch.int64) if epoch is None else torch.tensor(epoch)
    owned = torch.ones(num_envs, dtype=torch.bool) if owned is None else torch.tensor(owned)
    identifiers = torch.arange(num_envs, dtype=torch.float32) * 1000

    def snapshot(time):
        return Snapshot(
            state=torch.stack((identifiers, torch.full((num_envs,), float(time))), dim=-1),
            observation=(identifiers + time).unsqueeze(-1).expand(-1, 35).clone(),
            goal=torch.full((num_envs, 11), float(time)),
            context=torch.full((num_envs, 1), float(time)),
            episode_id=episode.clone(), control_epoch=epoch.clone(),
            policy_owned=owned.clone(),
        )

    return Transition(
        start=snapshot(step), action=torch.full((num_envs, 6), step / 100),
        end=snapshot(step + 1), reward=torch.arange(num_envs, dtype=torch.float32) + step,
        terminated=torch.zeros(num_envs, dtype=torch.bool),
        truncated=torch.zeros(num_envs, dtype=torch.bool),
        valid=torch.ones(num_envs, dtype=torch.bool),
    )


def test_windows_preserve_environment_and_all_frame_fields(schema):
    replay = ReplayBuffer(schema, num_envs=2, capacity=12)
    for step in range(5):
        replay.add(transition(step))
    batch = replay.sample(32, 3, generator=torch.Generator().manual_seed(2))
    assert batch.states.shape == (32, 4, 2)
    assert batch.observations.shape == (32, 4, 35)
    assert batch.goals.shape == (32, 4, 11)
    assert batch.contexts.shape == (32, 4, 1)
    assert batch.actions.shape == (32, 3, 6)
    assert all(getattr(batch, name).shape == (32, 3) for name in (
        "rewards", "terminated", "truncated", "valid", "policy_owned",
    ))
    assert torch.all(batch.states[:, 1:, 1] - batch.states[:, :-1, 1] == 1)
    assert torch.all(batch.states[:, :, 0] == batch.states[:, :1, 0])
    assert torch.equal(batch.observations[:, :, 0], batch.states.sum(-1))
    assert torch.equal(batch.goals[:, :, 0], batch.states[:, :, 1])
    assert torch.equal(batch.contexts[:, :, 0], batch.states[:, :, 1])
    assert torch.allclose(batch.actions[:, :, 0] * 100, batch.states[:, :-1, 1])
    assert replay.num_candidates(3) == 6


def test_ring_capacity_and_wraparound(schema):
    replay = ReplayBuffer(schema, num_envs=2, capacity=7)
    for step in range(9):
        replay.add(transition(step))
    assert replay.effective_capacity == 6
    assert len(replay) == replay.num_samples == 6
    assert replay.total_added == 18
    assert replay.num_candidates(3) == 2
    batch = replay.sample(20, 3)
    assert torch.all(batch.states[:, :, 1] == torch.tensor([6., 7., 8., 9.]))
    assert torch.all(batch.states[:, :, 0] == batch.states[:, :1, 0])


@pytest.mark.parametrize("flag", ["terminated", "truncated"])
def test_terminal_is_allowed_only_at_window_end(schema, flag):
    replay = ReplayBuffer(schema, num_envs=2, capacity=10)
    for step in range(3):
        item = transition(step)
        if step == 1:
            getattr(item, flag)[:] = True
        replay.add(item)
    assert replay.num_candidates(2) == 2
    batch = replay.sample(8, 2)
    assert torch.all(batch.states[:, 0, 1] == 0)
    assert getattr(batch, flag)[:, -1].all()
    assert replay.num_candidates(3) == 0


@pytest.mark.parametrize("boundary", ["episode", "epoch"])
def test_windows_do_not_cross_metadata_boundaries(schema, boundary):
    replay = ReplayBuffer(schema, num_envs=2, capacity=12)
    for step in range(4):
        arguments = {boundary: [int(step >= 2), 0]}
        replay.add(transition(step, **arguments))
    assert replay.num_candidates(3) == 2
    batch = replay.sample(20, 3)
    assert torch.all(batch.states[:, :, 0] == 1000)


def test_invalid_data_breaks_windows_and_ownership_only_filters_actor(schema):
    replay = ReplayBuffer(schema, num_envs=2, capacity=12)
    for step in range(3):
        item = transition(step, owned=[False, True])
        if step == 1:
            item.valid[1] = False
        replay.add(item)
    assert replay.num_valid == 5
    assert replay.num_valid_starts == 2
    assert replay.num_candidates(3) == 1
    assert not replay.sample(8, 3).policy_owned.any()
    starts = replay.sample_starts(20)
    assert starts.policy_owned.all()
    assert starts.states.shape == (20, 2, 2)
    assert torch.all(starts.states[:, 0, 0] == 1000)
    assert torch.all(starts.states[:, 0, 1] != 1)


def test_real_preparation_commands_survive_cpu_replay_but_never_seed_actor(schema, tmp_path):
    upstream = Path(os.environ.get("SUPERMIMIC_UPSTREAM_ROOT",
        Path(__file__).resolve().parents[2] / "transformer_rl-worktrees/full-vehicle-gimbal")) / "src"
    if not (upstream / "transformer_rl/formal/control/actuation.py").exists():
        pytest.skip("Pinned upstream controller is required")
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    source = pytest.importorskip("transformer_rl.formal.control.actuation")
    controller = source.OutputFeedbackController([0.] * 6, 2,
        pair_domain=source.PairTargetDomain(((-1., 1.), (-1., 1.)), (1, -1)))
    physical_pose = torch.tensor([[1.25, 1.25, -1.25, -1.25, 0., 0.], [0.] * 6])
    hold = controller.preparation_hold(physical_pose, position_tolerance_rad=1e-6)
    assert hold.admitted.all()
    assert (hold.requested_action[0, :4].abs() > controller.action_bounds[:4]).all()
    replay = ReplayBuffer(schema, num_envs=2, capacity=8)
    assert replay.device.type == "cpu"
    for step in range(3):
        item = transition(step, owned=[False, True])
        item.action[0] = hold.requested_action[0]
        item.action[1] = float(step) / 100
        replay.add(item)
    assert replay.num_candidates(3) == 2
    script = replay.sample(8, 3, env_ids=[0])
    torch.testing.assert_close(script.actions,
                              hold.requested_action[0].expand(8, 3, 6))
    assert not script.policy_owned.any()
    assert replay.num_start_candidates(3, env_ids=[0]) == 0
    starts = replay.sample_starts(12, history_length=3)
    assert starts.policy_owned.all()
    assert torch.all(starts.states[:, :, 0] == 1000)
    path = tmp_path / "script-replay.pt"
    replay.save(path)
    restored = ReplayBuffer(schema, num_envs=2, capacity=8)
    restored.load(path)
    torch.testing.assert_close(restored.sample(4, 3, env_ids=[0]).actions,
                              hold.requested_action[0].expand(4, 3, 6))
    assert restored.num_start_candidates(3, env_ids=[0]) == 0


def test_epoch_change_inside_single_transition_is_never_sampled(schema):
    replay = ReplayBuffer(schema, num_envs=2, capacity=4)
    item = transition(0)
    item.end.control_epoch[0] = 1
    replay.add(item)
    assert replay.num_valid == replay.num_valid_starts == 1
    assert torch.all(replay.sample_starts(10).states[:, 0, 0] == 1000)


def test_actor_history_uses_real_frames_and_nonterminal_owned_endpoint(schema):
    replay = ReplayBuffer(schema, num_envs=2, capacity=12)
    for step in range(5):
        item = transition(step)
        if step == 3:
            item.terminated[0] = True
        if step == 4:
            item.end.policy_owned[1] = False
        replay.add(item)
    assert replay.num_start_candidates(5) == 1
    batch = replay.sample_starts(8, history_length=5)
    assert batch.observations.shape == (8, 5, 35)
    assert torch.all(batch.states[:, :, 0] == 1000)
    assert torch.all(batch.states[:, :, 1] == torch.arange(5))
    assert batch.policy_owned.all()
    assert not (batch.terminated | batch.truncated).any()
    # World-model windows may retain terminal endpoints and unowned future states.
    assert replay.num_candidates(4) == 3


@pytest.mark.parametrize("cause", ["invalid", "ownership", "truncated", "epoch"])
def test_actor_history_rejects_breaks_anywhere_in_history(schema, cause):
    replay = ReplayBuffer(schema, num_envs=2, capacity=12)
    for step in range(4):
        item = transition(step)
        if step == 1:
            if cause == "invalid":
                item.valid[:] = False
            elif cause == "ownership":
                item.start.policy_owned[:] = False
            elif cause == "truncated":
                item.truncated[:] = True
            else:
                item.end.control_epoch[:] = 1
        replay.add(item)
    assert replay.num_start_candidates(5) == 0
    with pytest.raises(ValueError, match="No eligible replay windows"):
        replay.sample_starts(1, history_length=5)


def test_single_frame_actor_start_may_precede_terminal_endpoint(schema):
    replay = ReplayBuffer(schema, num_envs=2, capacity=4)
    item = transition(0)
    item.terminated[:] = True
    item.end.policy_owned[:] = False
    replay.add(item)
    assert replay.num_start_candidates(1) == 2
    assert replay.sample_starts(4, history_length=1).terminated.all()
    assert replay.num_start_candidates(2) == 0
    with pytest.raises(ValueError, match="history_length"):
        replay.sample_starts(1, history_length=0)


def test_sampling_errors_and_cache_invalidation(schema):
    replay = ReplayBuffer(schema, num_envs=2, capacity=4)
    with pytest.raises(ValueError, match="No eligible replay windows"):
        replay.sample(1, 1)
    replay.add(transition(0, owned=[False, False]))
    assert replay.num_valid_starts == 0
    with pytest.raises(ValueError, match="No eligible replay windows"):
        replay.sample_starts(1)
    with pytest.raises(ValueError, match="No eligible replay windows"):
        replay.sample(1, 2)
    replay.add(transition(1))
    assert replay.num_valid_starts == 2
    assert replay.num_candidates(2) == 2
    with pytest.raises(ValueError, match="horizon"):
        replay.sample(1, 0)
    with pytest.raises(ValueError, match="batch_size"):
        replay.sample(0, 1)


def test_add_copies_inputs_and_failed_add_does_not_advance(schema):
    replay = ReplayBuffer(schema, num_envs=2, capacity=4)
    item = transition(0)
    item.start.state.requires_grad_()
    replay.add(item)
    item.start.state = torch.full_like(item.start.state, 99)
    item.end.state[:] = 99
    assert torch.all(replay.sample(5, 1).states[:, 1, 1] == 1)
    assert not replay.sample(5, 1).states.requires_grad
    invalid = transition(1)
    invalid.action[:, 0] = 4
    with pytest.raises(ValueError, match="bounded"):
        replay.add(invalid)
    assert replay.total_added == 2


@pytest.mark.parametrize("steps", [0, 2, 7])
def test_persistence_roundtrip_preserves_sampling_and_counters(schema, tmp_path, steps):
    replay = ReplayBuffer(schema, num_envs=2, capacity=8)
    for step in range(steps):
        replay.add(transition(step))
    path = tmp_path / "replay.pt"
    replay.save(path)
    restored = ReplayBuffer(schema, num_envs=2, capacity=8)
    restored.load(path)
    assert restored.total_added == replay.total_added
    assert restored.num_samples == replay.num_samples
    assert restored.num_valid == replay.num_valid
    if steps:
        original = replay.sample(10, 2, generator=torch.Generator().manual_seed(7))
        loaded = restored.sample(10, 2, generator=torch.Generator().manual_seed(7))
        # Optional fields absent from this schema (substeps) stay absent after loading.
        assert all(torch.equal(getattr(original, f.name), getattr(loaded, f.name))
                   if getattr(original, f.name) is not None else getattr(loaded, f.name) is None
                   for f in fields(original))
    replay.add(transition(steps))
    restored.add(transition(steps))
    assert restored.num_candidates(2) == replay.num_candidates(2)


def test_checkpoint_validation_is_atomic(schema):
    replay = ReplayBuffer(schema, num_envs=2, capacity=8)
    replay.add(transition(0))
    wrong_schema = StateSchema(schema.state_fields, schema.context_fields, source="other")
    other = ReplayBuffer(wrong_schema, num_envs=2, capacity=8)
    with pytest.raises(ValueError, match="fingerprint"):
        other.load_state_dict(replay.state_dict())
    damaged = replay.state_dict()
    damaged["storage"]["end_state"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        replay.load_state_dict(damaged)
    assert replay.total_added == 2
    assert torch.isfinite(replay.sample(1, 1).states).all()
    damaged = replay.state_dict()
    damaged["sequence"][0] = 5
    with pytest.raises(ValueError, match="sequence"):
        replay.load_state_dict(damaged)


def test_empty_context_and_training_device_transfer():
    schema = StateSchema((Field("state", (2,)),), ())
    replay = ReplayBuffer(schema, num_envs=2, capacity=4, device="cpu")
    item = transition(0)
    item.start.context = torch.empty(2, 0)
    item.end.context = torch.empty(2, 0)
    replay.add(item)
    batch = replay.sample(3, 1, device="cpu")
    assert batch.contexts.shape == (3, 2, 0)
    present = lambda batch: [getattr(batch, f.name) for f in fields(batch) if getattr(batch, f.name) is not None]
    assert all(value.device.type == "cpu" for value in present(batch))
    if torch.cuda.is_available():
        batch = replay.sample_starts(3, device="cuda")
        assert all(value.device.type == "cuda" for value in present(batch))


@pytest.mark.parametrize("num_envs,capacity", [(0, 4), (2, 1), (2, 2.5)])
def test_invalid_capacity_is_rejected(schema, num_envs, capacity):
    with pytest.raises(ValueError):
        ReplayBuffer(schema, num_envs=num_envs, capacity=capacity)


def test_environment_holdout_isolates_world_windows_and_actor_histories(schema):
    replay = ReplayBuffer(schema, num_envs=3, capacity=18)
    for step in range(8):
        replay.add(transition(step, num_envs=3))
    train_ids, validation_ids = [0, 2], torch.tensor([1], dtype=torch.int32)
    assert replay.num_candidates(3, env_ids=train_ids) == 8
    assert replay.num_candidates(3, env_ids=validation_ids) == 4
    assert replay.num_start_candidates(5, env_ids=train_ids) == 6
    assert replay.num_start_candidates(5, env_ids=validation_ids) == 3

    for sample in (
        lambda ids: replay.sample(64, 3, env_ids=ids),
        lambda ids: replay.sample_starts(64, 5, env_ids=ids),
    ):
        training, validation = sample(train_ids), sample(validation_ids)
        assert torch.all((training.states[:, :, 0] == 0)
                         | (training.states[:, :, 0] == 2000))
        assert torch.all(validation.states[:, :, 0] == 1000)
        assert torch.all(training.states[:, :, 0] == training.states[:, :1, 0])
        assert torch.all(validation.states[:, :, 0] == validation.states[:, :1, 0])
        assert torch.all(training.states[:, 1:, 1] - training.states[:, :-1, 1] == 1)
        assert torch.all(validation.states[:, 1:, 1] - validation.states[:, :-1, 1] == 1)


def test_environment_filter_counts_match_empty_sampling_and_do_not_change_cache(schema):
    replay = ReplayBuffer(schema, num_envs=2, capacity=10)
    for step in range(4):
        item = transition(step)
        item.valid[1] = False
        replay.add(item)
    assert replay.num_candidates(2, env_ids=[1]) == 0
    assert replay.num_start_candidates(3, env_ids=[1]) == 0
    with pytest.raises(ValueError, match="No eligible replay windows"):
        replay.sample(1, 2, env_ids=[1])
    with pytest.raises(ValueError, match="No eligible replay windows"):
        replay.sample_starts(1, 3, env_ids=[1])
    assert replay.num_candidates(2, env_ids=[0]) == 3
    assert replay.num_start_candidates(3, env_ids=[0]) == 3
    assert replay.num_candidates(2) == 3


@pytest.mark.parametrize("env_ids", [
    [], [0, 0], [-1], [2], [True], [0.5], "0",
    torch.tensor([]), torch.tensor([0., 1.]), torch.tensor([[0]]),
    torch.tensor([True]), torch.tensor([0, 0]),
])
def test_invalid_environment_selections_are_rejected_everywhere(schema, env_ids):
    replay = ReplayBuffer(schema, num_envs=2, capacity=6)
    replay.add(transition(0))
    replay.add(transition(1))
    for call in (
        lambda: replay.num_candidates(1, env_ids=env_ids),
        lambda: replay.num_start_candidates(2, env_ids=env_ids),
        lambda: replay.sample(1, 1, env_ids=env_ids),
        lambda: replay.sample_starts(1, 2, env_ids=env_ids),
    ):
        with pytest.raises(ValueError, match="env_ids"):
            call()


def test_actual_actor_frames_and_estimator_labels_follow_each_endpoint(schema):
    replay = ReplayBuffer(schema, num_envs=2, capacity=8)
    for step in range(3):
        item = transition(step, owned=[False, True])
        for snapshot in (item.start, item.end):
            snapshot.actor_observation = snapshot.observation + 17
            time = snapshot.state[:, 1]
            snapshot.estimator_target = torch.stack((
                time, -time, time * 2, time / 4, 1 - time / 4,
            ), dim=-1)
            snapshot.estimator_valid = snapshot.state[:, 0] == 1000
        if step == 2:
            item.terminated[0] = True
        replay.add(item)
        # Replay owns its copies, including the actual actor frame and labels.
        item.end.actor_observation[:] = -99
        item.end.estimator_target[:] = 0
        item.end.estimator_valid[:] = False
    for batch in (replay.sample(16, 3), replay.sample_starts(16, 4)):
        assert batch.actor_observations.shape == (16, 4, 35)
        assert batch.estimator_targets.shape == (16, 4, 5)
        assert batch.estimator_valid.shape == (16, 4)
        assert torch.equal(batch.actor_observations, batch.observations + 17)
        assert torch.equal(batch.estimator_targets[..., 0], batch.states[..., 1])
        assert torch.equal(batch.estimator_targets[..., 3], batch.states[..., 1] / 4)
        assert torch.equal(batch.estimator_valid, batch.states[..., 0] == 1000)
    actor = replay.sample_starts(16, 4)
    assert actor.policy_owned.all()
    assert torch.all(actor.states[..., 0] == 1000)
    world = replay.sample(4, 3, env_ids=[0])
    assert not world.policy_owned.any()
    assert world.terminated[:, -1].all()
    assert torch.equal(world.estimator_targets[:, -1, 0], torch.full((4,), 3.))
    assert torch.allclose(world.actions[..., 0] * 100, world.states[:, :-1, 1])


def test_legacy_checkpoint_migration_is_explicit_and_preserves_ring(schema):
    replay = ReplayBuffer(schema, num_envs=2, capacity=8)
    for step in range(7):
        replay.add(transition(step))
    payload = replay.state_dict()
    payload["format_version"] = 1
    for endpoint in ("start", "end"):
        for name in ("actor_observation", "estimator_target", "estimator_valid"):
            del payload["storage"][f"{endpoint}_{name}"]
    restored = ReplayBuffer(schema, num_envs=2, capacity=8)
    restored.load_state_dict(payload)
    assert restored.total_added == 14
    batch = restored.sample(8, 4)
    assert torch.equal(batch.actor_observations, batch.observations)
    assert not batch.estimator_valid.any()
    assert torch.count_nonzero(batch.estimator_targets) == 0
    assert torch.all(batch.states[..., 1] == torch.arange(3, 8))
    restored.add(transition(7))
    assert restored.total_added == 16
    assert restored.state_dict()["format_version"] == 2


def test_new_checkpoint_requires_all_auxiliary_storage_without_partial_load(schema):
    replay = ReplayBuffer(schema, num_envs=2, capacity=8)
    replay.add(transition(0))
    payload = replay.state_dict()
    del payload["storage"]["end_actor_observation"]
    payload["storage"]["start_state"][:] = 123
    with pytest.raises(ValueError, match="storage fields"):
        replay.load_state_dict(payload)
    assert torch.all(replay.sample(4, 1).states[:, 0, 1] == 0)


def test_formal_interface_requires_actual_actor_frame_but_allows_unlabelled_data(schema):
    formal = replace(schema, observation_dim=45)
    replay = ReplayBuffer(formal, num_envs=2, capacity=8)
    item = transition(0)
    for snapshot in (item.start, item.end):
        snapshot.observation = torch.zeros(2, 45)
        snapshot.actor_observation = torch.ones(2, 45)
    replay.add(item)
    batch = replay.sample(4, 1)
    assert torch.all(batch.observations == 0)
    assert torch.all(batch.actor_observations == 1)
    assert not batch.estimator_valid.any()
    item.end.actor_observation = None
    with pytest.raises(ValueError, match="actual deployment frame|explicit actor"):
        replay.add(item)
    assert replay.total_added == 2
    payload = replay.state_dict()
    payload["format_version"] = 1
    with pytest.raises(ValueError, match="35D"):
        replay.load_state_dict(payload)
