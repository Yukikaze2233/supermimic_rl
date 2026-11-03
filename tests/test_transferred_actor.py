"""Numeric transfer checks using the real optional upstream policy implementation."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import io
import json
import subprocess
import sys

import pytest
import torch

from supermimic_rl.contracts import ACTION_BOUNDS
from supermimic_rl.transferred_actor import TransferredActor


@pytest.fixture
def upstream():
    return pytest.importorskip(
        "transformer_rl.formal.policy",
        reason="Transferred actor tests need the pinned transformer_rl source on PYTHONPATH",
    )


@pytest.fixture
def small_config(upstream):
    return asdict(upstream.StructuredPolicyConfig(
        body_width=8, body_heads=2, body_ffn=16, temporal_width=16,
        temporal_heads=2, temporal_layers=1, temporal_ffn=32,
        state_hidden_dim=8, leg_hidden_dims=(16, 8), wheel_hidden_dims=(16, 8),
        prediction_hidden_dim=8, critic_hidden_dims=(16, 8, 4),
        mirror_canonical=True, mean_init_scale=0.1,
    ))


@pytest.fixture
def source_checkpoint(upstream, small_config):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(135)
        policy = upstream.StructuredPolicy(upstream.policy_config_from_dict(small_config)).eval()
    payload = {
        "format": "private_structured_learning_snapshot",
        "recipe": {"policy": small_config, "collection": {"action_bounds": list(ACTION_BOUNDS)},
                   "learner_source_sha256": "a" * 64, "asset_manifest_sha256": "b" * 64,
                   "task_contract_sha256": "c" * 64},
        "normalization": "fixed_manual35_185_no_running_statistics",
        "model": {"actor.policy." + name: value.clone() for name, value in policy.state_dict().items()},
        "actor_adam": {"must_not_be_imported": True},
        "critic_adam": {"must_not_be_imported": True},
        "trainer": {"update_index": 18026},
    }
    payload["model"].update({"actor.log_std": torch.full((6,), 123.),
                             "critic.unrelated": torch.full((7,), float("nan"))})
    return policy, payload


def save_checkpoint(tmp_path, payload):
    path = tmp_path / "teacher.pt"
    torch.save(payload, path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def history():
    generator = torch.Generator().manual_seed(42)
    frames = torch.randn((2, 5, 35), generator=generator) * 0.1
    frames[..., 3] = 1.6
    frames[..., 9] = -1
    frames[..., 14:16] = 0
    frames[..., 28] = 1
    return frames


def test_module_import_does_not_require_or_import_upstream():
    code = """
import sys
class RejectUpstream:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('transformer_rl'):
            raise AssertionError('upstream was imported eagerly')
sys.meta_path.insert(0, RejectUpstream())
import supermimic_rl.transferred_actor
assert not any(name.startswith('transformer_rl') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_transfer_matches_raw_mean_clipping_and_preserves_parameter_identity(
        tmp_path, small_config, source_checkpoint):
    reference, payload = source_checkpoint
    actor = TransferredActor(small_config)
    before = [id(value) for value in actor.parameters()]
    rng = torch.get_rng_state().clone()
    path, sha = save_checkpoint(tmp_path, payload)
    metadata = actor.load_initial_checkpoint(path, sha)
    assert torch.equal(torch.get_rng_state(), rng)
    assert before == [id(value) for value in actor.parameters()]
    frames = history()
    bounds = torch.tensor(ACTION_BOUNDS)
    torch.testing.assert_close(actor(frames), reference(frames).clamp(-bounds, bounds), rtol=0, atol=0)
    assert actor.policy.prediction_head is None
    action, estimate = actor.action_and_estimate(frames)
    mean, raw_state, _ = reference.outputs(frames)
    torch.testing.assert_close(action, mean.clamp(-bounds, bounds), rtol=0, atol=0)
    torch.testing.assert_close(estimate, torch.cat((raw_state[:, :3], raw_state[:, 6:8]), -1), rtol=0, atol=0)
    assert not any("critic" in name or "log_std" in name or "prediction_head" in name
                   for name in actor.state_dict())
    assert any("state_head" in name for name in actor.state_dict())
    assert metadata["initial_checkpoint"]["sha256"] == sha
    assert metadata["initial_checkpoint"]["learner_source_sha256"] == "a" * 64
    assert len(metadata["source_files"]) == 4
    json.dumps(metadata, allow_nan=False)
    metadata["policy_config"]["mirror_canonical"] = False
    assert actor.source_metadata["policy_config"]["mirror_canonical"] is True


def test_input_and_actor_gradients_survive_transfer_and_frozen_teacher(
        tmp_path, small_config, source_checkpoint):
    _, payload = source_checkpoint
    actor = TransferredActor(small_config)
    actor.load_initial_checkpoint(*save_checkpoint(tmp_path, payload))
    actor.double()
    frames = history().double().requires_grad_()
    weights = torch.arange(1., 7., dtype=torch.double)
    loss = (actor(frames) * weights).sum()
    loss.backward()
    assert torch.isfinite(frames.grad).all()
    assert (frames.grad.norm(dim=(0, 2)) > 0).all()
    assert frames.grad[..., 14:16].eq(0).all()
    assert actor.policy.command_projection.weight.grad.norm() > 0
    assert all(value.grad is not None and torch.isfinite(value.grad).all()
               for value in actor.parameters())
    direction = torch.randn(frames.shape, generator=torch.Generator().manual_seed(7), dtype=torch.double)
    epsilon = 1e-5
    with torch.no_grad():
        finite_difference = (((actor(frames + epsilon * direction)
                              - actor(frames - epsilon * direction)) * weights).sum() / (2 * epsilon))
    torch.testing.assert_close((frames.grad * direction).sum(), finite_difference, rtol=1e-5, atol=1e-7)
    actor.zero_grad(set_to_none=True)
    actor.eval().requires_grad_(False)
    frozen_gradient, = torch.autograd.grad((actor(frames) * weights).sum(), frames)
    torch.testing.assert_close(frozen_gradient, frames.grad)
    assert all(value.grad is None for value in actor.parameters())


def test_hard_clip_keeps_interior_mean_and_zeroes_only_saturated_gradient(small_config):
    actor = TransferredActor(small_config)
    with torch.no_grad():
        actor.policy.leg_decoder[-1].weight.zero_()
        actor.policy.leg_decoder[-1].bias.copy_(torch.tensor([5., -1.25]))
        actor.policy.wheel_decoder[-1].weight.zero_()
        actor.policy.wheel_decoder[-1].bias.fill_(12.)
    result = actor(history()[:1])
    torch.testing.assert_close(result, torch.tensor([[3., -1.25, -3., 1.25, 9., -9.]]), rtol=0, atol=0)
    (result * torch.tensor([1., 2., 4., 8., 16., 32.])).sum().backward()
    torch.testing.assert_close(actor.policy.leg_decoder[-1].bias.grad, torch.tensor([0., -6.]))
    torch.testing.assert_close(actor.policy.wheel_decoder[-1].bias.grad, torch.zeros(1))


def test_state_dict_roundtrip_restores_mean_buffers_and_gradients(tmp_path, small_config, source_checkpoint):
    _, payload = source_checkpoint
    actor = TransferredActor(small_config)
    actor.load_initial_checkpoint(*save_checkpoint(tmp_path, payload))
    stream = io.BytesIO()
    torch.save(actor.state_dict(), stream)
    stream.seek(0)
    restored = TransferredActor(small_config)
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    frames = history().requires_grad_()
    torch.testing.assert_close(restored(frames), actor(frames), rtol=0, atol=0)
    first, = torch.autograd.grad(actor(frames).square().sum(), frames)
    second, = torch.autograd.grad(restored(frames).square().sum(), frames)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.equal(restored.action_bounds, torch.tensor(ACTION_BOUNDS))


def test_bad_sha_is_rejected_before_deserialization(tmp_path, small_config, source_checkpoint, monkeypatch):
    _, payload = source_checkpoint
    actor = TransferredActor(small_config)
    path, _ = save_checkpoint(tmp_path, payload)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("unverified checkpoint was loaded"))
    with pytest.raises(ValueError, match="SHA256"):
        actor.load_initial_checkpoint(path, "0" * 64)
    assert actor.source_metadata["initial_checkpoint"] is None


def test_time_encoding_roundoff_is_preserved_without_replacing_saved_values(
        tmp_path, small_config, source_checkpoint):
    _, payload = source_checkpoint
    saved = payload["model"]["actor.policy.time_encoding"]
    saved[0, 0] += torch.finfo(saved.dtype).eps
    actor = TransferredActor(small_config)
    assert not torch.equal(actor.policy.time_encoding, saved)
    actor.load_initial_checkpoint(*save_checkpoint(tmp_path, payload))
    assert torch.equal(actor.policy.time_encoding, saved)


@pytest.mark.parametrize("damage", [
    "format", "normalization", "missing_normalization", "history", "config", "action_bounds",
    "missing_source_pin", "missing_weight", "extra_weight", "missing_prediction",
    "dtype", "nonfinite", "fixed_buffer", "time_encoding",
])
def test_invalid_transfer_is_atomic(tmp_path, small_config, source_checkpoint, damage):
    _, source = source_checkpoint
    payload = deepcopy(source)
    weight = "actor.policy.command_projection.weight"
    if damage == "format":
        payload["format"] = "other"
    elif damage == "normalization":
        payload["normalization"] = "running_statistics"
    elif damage == "missing_normalization":
        del payload["normalization"]
    elif damage == "history":
        payload["recipe"]["policy"]["history_length"] = 4
    elif damage == "config":
        payload["recipe"]["policy"]["mirror_canonical"] = False
    elif damage == "action_bounds":
        payload["recipe"]["collection"]["action_bounds"][0] = 1.
    elif damage == "missing_source_pin":
        del payload["recipe"]["learner_source_sha256"]
    elif damage == "missing_weight":
        del payload["model"][weight]
    elif damage == "extra_weight":
        payload["model"]["actor.policy.unexpected"] = torch.zeros(1)
    elif damage == "missing_prediction":
        del payload["model"]["actor.policy.prediction_head.0.weight"]
    elif damage == "dtype":
        payload["model"][weight] = payload["model"][weight].double()
    elif damage == "nonfinite":
        payload["model"][weight][0, 0] = float("nan")
    elif damage == "fixed_buffer":
        payload["model"]["actor.policy.command_indices"][0] = 1
    elif damage == "time_encoding":
        payload["model"]["actor.policy.time_encoding"][0, 0] += 0.01
    actor = TransferredActor(small_config)
    original = {name: value.clone() for name, value in actor.state_dict().items()}
    with pytest.raises(ValueError):
        actor.load_initial_checkpoint(*save_checkpoint(tmp_path, payload))
    assert actor.source_metadata["initial_checkpoint"] is None
    for name, value in actor.state_dict().items():
        assert torch.equal(value, original[name])


def test_history_contract_is_explicit(small_config):
    with pytest.raises(ValueError, match="history"):
        TransferredActor(small_config, history_length=4)
    actor = TransferredActor(small_config)
    with pytest.raises(ValueError, match="chronological"):
        actor(torch.zeros(2, 175))
