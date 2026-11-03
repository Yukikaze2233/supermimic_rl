"""V2 sequence boundaries, periodic supervision and explicit gimbal observations."""
import math

import pytest
import torch

from supermimic_rl.world_model_v2.fitting import tick_residual
from supermimic_rl.world_model_v2.model import Ticks
from supermimic_rl.world_model_v2.physical import PhysicalState
from supermimic_rl.world_model_v2 import so3
from supermimic_rl.world_model_v2.policy import EstimatorActor, export_onnx
from supermimic_rl.world_model_v2.replay import CompactReplay
from supermimic_rl.world_model_v2.records import CONTACT_BODIES, TICK_LABEL_DIM


def add(replay, step, *, sequence_id=None, **overrides):
    n = replay.num_envs
    boundary = {name: torch.ones(n, dtype=torch.bool) for name in
                ("valid", "policy_owned", "end_policy_owned")}
    boundary.update({name: torch.zeros(n, dtype=torch.bool) for name in ("terminated", "truncated")})
    boundary.update({name: torch.zeros(n, dtype=torch.int64) for name in
                     ("start_episode_id", "end_episode_id", "start_control_epoch", "end_control_epoch")})
    for name, value in overrides.items():
        boundary[name].fill_(value)
    replay.add({"frame": torch.full((n, 45), float(step))}, boundary, sequence_id=sequence_id)


def test_first_four_steps_are_available_with_reset_padding():
    replay = CompactReplay({"frame": (45,)}, 1, 8, "cpu")
    for step in range(4):
        add(replay, step)
    assert replay.start_candidates().tolist() == [0, 1, 2, 3]
    starts, history = replay.sample_starts(64, 5, generator=torch.Generator().manual_seed(1))
    step = starts["frame"][:, 0]
    expected = (step[:, None] - torch.arange(4, -1, -1)[None]).clamp_min(0)
    torch.testing.assert_close(history[..., 0], expected)


@pytest.mark.parametrize("boundary", [
    {"valid": False}, {"policy_owned": False}, {"end_policy_owned": False},
    {"terminated": True}, {"truncated": True}, {"end_control_epoch": 1},
    {"end_episode_id": 1},
])
def test_history_never_reaches_back_before_takeover_or_reset(boundary):
    replay = CompactReplay({"frame": (45,)}, 1, 4, "cpu")
    add(replay, 0)
    add(replay, 1, **boundary)
    add(replay, 2)
    assert replay.data["segment_age"][2, 0] == 0
    starts, history = replay.sample_starts(64, 5, generator=torch.Generator().manual_seed(5))
    after = starts["frame"][:, 0] == 2
    assert after.any()
    assert (history[after] == 2).all()


def test_recording_gap_breaks_windows_and_actor_history():
    replay = CompactReplay({"frame": (45,)}, 1, 4, "cpu")
    add(replay, 10, sequence_id=10)
    add(replay, 12, sequence_id=12)
    add(replay, 13, sequence_id=13)
    assert replay.window_candidates(2).tolist() == [1]
    assert replay.data["segment_age"][:3, 0].tolist() == [0, 0, 1]
    starts, history = replay.sample_starts(64, 5, generator=torch.Generator().manual_seed(0))
    for step in (12, 13):
        subset = history[starts["frame"][:, 0] == step]
        assert subset.numel()
        assert (subset >= 12).all()
    with pytest.raises(ValueError, match="strictly increase"):
        add(replay, 13, sequence_id=13)


def test_ring_overwrite_does_not_fabricate_old_history():
    replay = CompactReplay({"frame": (45,)}, 1, 3, "cpu")
    for step in range(8):
        add(replay, step)
    starts, history = replay.sample_starts(64, 5, generator=torch.Generator().manual_seed(0))
    expected = (starts["frame"][:, :1] - torch.arange(4, -1, -1)[None]).clamp_min(5)
    torch.testing.assert_close(history[..., 0], expected)
    assert replay.sample_windows(1, 3)["frame"][0, :, 0].tolist() == [5, 6, 7]


def test_compact_persistence_preserves_wrapped_ring_gaps_and_future_append(tmp_path):
    original = CompactReplay({'frame': (45,)}, 2, 5, 'cpu')
    for step in (0, 1, 3, 4, 5, 6, 7):
        add(original, step, sequence_id=step, terminated=step == 4)
    path = tmp_path / 'compact.pt'
    torch.save(original.state_dict(), path)
    restored = CompactReplay.from_state_dict(torch.load(path, weights_only=True, mmap=True))
    for ring in (original, restored):
        add(ring, 8, sequence_id=8)
    torch.testing.assert_close(original.window_candidates(2), restored.window_candidates(2))
    a, ah = original.sample_starts(64, 5, generator=torch.Generator().manual_seed(4))
    b, bh = restored.sample_starts(64, 5, generator=torch.Generator().manual_seed(4))
    for name in a:
        torch.testing.assert_close(a[name], b[name])
    torch.testing.assert_close(ah, bh)
    damaged = original.state_dict()
    damaged['sequence'][1] = damaged['sequence'][0]
    with pytest.raises(ValueError, match='chronology'):
        CompactReplay.from_state_dict(damaged)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA checkpoint peak-memory regression')
def test_checkpoint_does_not_gather_another_full_gpu_tick_tensor():
    replay = CompactReplay({'ticks': (4, TICK_LABEL_DIM)}, 256, 64, 'cuda:0')
    replay.size, replay.head, replay.total_steps = 64, 17, 81
    for slot in range(64):
        replay.data['ticks'][slot].fill_(slot)
    expected = replay.data['ticks'].cpu()[torch.arange(17, 81) % 64]
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    saved = replay.state_dict()
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - baseline
    tensor_bytes = replay.data['ticks'].numel() * replay.data['ticks'].element_size()
    assert extra < tensor_bytes // 4
    torch.testing.assert_close(saved['storage']['ticks'], expected, atol=0, rtol=0)


def test_tick_residual_wraps_base_and_gimbal_yaw_only():
    zeros = lambda width: torch.zeros(1, 4, width)
    def state(yaw, pitch):
        q = zeros(6)
        q[..., 4:] = torch.tensor([yaw, pitch])
        rotation = so3.exp(torch.tensor([0., 0., yaw])).expand(1, 4, 3, 3)
        return PhysicalState(zeros(3), rotation, zeros(3), zeros(3), q, zeros(8))
    actual = Ticks(state(math.pi - .01, -.1), zeros(2), zeros(21), zeros(3), zeros(2), zeros(2))
    predicted = Ticks(state(-math.pi + .01, .1), zeros(2), zeros(21), zeros(3), zeros(2), zeros(2))
    residual = tick_residual(predicted, actual)
    torch.testing.assert_close(residual[..., 5], torch.full((1, 4), .02))
    torch.testing.assert_close(residual[..., 16], torch.full((1, 4), .02))
    torch.testing.assert_close(residual[..., 17], torch.full((1, 4), .2))


def test_contact_labels_preserve_support_riser_and_leg_body_contacts_separately():
    zero = lambda width: torch.zeros(1, width)
    state = PhysicalState(zero(3), torch.eye(3)[None], zero(3), zero(3), zero(6), zero(8))
    labels = torch.zeros(1, 4, TICK_LABEL_DIM)
    labels[..., 26:28] = torch.tensor([10., 20.])
    labels[..., 28 + CONTACT_BODIES.index('L_link3')] = 150.
    labels[..., 28 + CONTACT_BODIES.index('L_link2')] = 7.
    labels[..., 28 + CONTACT_BODIES.index('M_link1')] = 4.
    labels[..., 49:52] = torch.tensor([140., 30., 8.])
    labels[..., 52:] = torch.tensor([.4, -.5, 2., -3.])
    ticks = Ticks.decode(labels, state)
    assert (ticks.wheel_load[..., 0] == 10).all()
    assert (ticks.wheel_body_load[..., 0] == 150).all()
    assert (ticks.wheel_contact_force[..., 0] == 140).all()
    assert (ticks.base_contact_force == 8).all()
    assert (ticks.body_normal_load[..., CONTACT_BODIES.index('L_link2')] == 7).all()
    assert (ticks.forbidden_load[..., 1] == 4).all()


def test_actor_uses_gimbal_channels_and_keeps_estimator_supervision_separate():
    torch.manual_seed(1)
    actor = EstimatorActor(estimator_hidden=(16,), actor_hidden=(32,))
    history = torch.randn(3, 5, 45, requires_grad=True)
    actor(history).square().sum().backward()
    assert torch.isfinite(history.grad).all()
    assert history.grad[..., 35:].abs().sum() > 0
    assert all(parameter.grad is None for parameter in actor.estimator.parameters())
    actor.estimator_loss(history.detach(), torch.zeros(3, 3), torch.zeros(3, 2)).backward()
    assert all(parameter.grad is not None for parameter in actor.estimator.parameters())
    with pytest.raises(ValueError, match="45"):
        actor(torch.zeros(3, 5, 35))


def test_estimator_input_gradient_agrees_with_finite_difference():
    torch.manual_seed(2)
    actor = EstimatorActor(estimator_hidden=(8,), actor_hidden=(16,)).double()
    # Isolate the path through estimated velocity/contact instead of allowing
    # the decoder's direct history input to hide an accidental detach.
    with torch.no_grad():
        actor.decoder[0].weight[:, :5 * 45].zero_()
    history = torch.randn(2, 5, 45, dtype=torch.float64, requires_grad=True)
    direction = torch.randn_like(history)
    value = actor(history).sum()
    gradient, = torch.autograd.grad(value, history)
    analytic = (gradient * direction).sum()
    with torch.no_grad():
        eps = 1e-5
        numerical = (actor(history + eps * direction).sum() - actor(history - eps * direction).sum()) / (2 * eps)
    assert analytic.abs() > 1e-7
    torch.testing.assert_close(analytic, numerical, atol=1e-9, rtol=1e-4)


def test_exported_formal45_actor_matches_training_and_inference_paths(tmp_path):
    pytest.importorskip("onnx")
    runtime = pytest.importorskip("onnxruntime")
    torch.manual_seed(3)
    actor = EstimatorActor(estimator_hidden=(16,), actor_hidden=(32,))
    path = export_onnx(actor, tmp_path / "actor45.onnx")
    session = runtime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for scale in (0., 1., 5.):
        history = torch.randn(1, 5, 45) * scale
        trained_path = actor(history).detach()
        with torch.no_grad():
            inference_path = actor(history)
        exported = torch.from_numpy(session.run(None, {"history": history.numpy()})[0])
        torch.testing.assert_close(inference_path, trained_path)
        torch.testing.assert_close(exported, trained_path, atol=1e-6, rtol=1e-5)
    assert actor.training  # Export only changes its private copy.


def test_replay_growth_and_environment_selection_preserve_chronology_without_truncation():
    original = CompactReplay({'frame': (45,)}, 3, 4, 'cpu')
    for step in range(6):
        add(original, step)
    original.data['frame'][:, 2] += 100
    payload = original.state_dict()
    grown = CompactReplay.from_state_dict(payload, capacity_steps=10, env_ids=[2, 0])
    assert grown.capacity == 10 and grown.num_envs == 2 and grown.size == 4
    torch.testing.assert_close(grown.sequence[:4], torch.tensor([2, 3, 4, 5]))
    torch.testing.assert_close(grown.data['frame'][:4, 0, 0], torch.tensor([102., 103., 104., 105.]))
    torch.testing.assert_close(grown.data['frame'][:4, 1, 0], torch.tensor([2., 3., 4., 5.]))
    with pytest.raises(ValueError, match='discard'):
        CompactReplay.from_state_dict(payload, capacity_steps=2)
    with pytest.raises(ValueError, match='distinct'):
        CompactReplay.from_state_dict(payload, env_ids=[1, 1])
