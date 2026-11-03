"""Differentiable transfer of the pinned upstream deterministic Transformer actor.

The caller installs the selected ``transformer_rl`` source before construction.
Importing this module does not import that optional dependency or start Isaac.
The original manual 35D observation scaling and clipped action means are retained.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib
import io
import json
from pathlib import Path
import re

import torch
from torch import Tensor, nn

from .contracts import ACTION_BOUNDS


class TransferredActor(nn.Module):
    """A trainable H5 Transformer mean with the upstream hard action bounds.

    ``eval()`` does not disable differentiation. Freeze a separate teacher with
    ``requires_grad_(False)``; keep the trainable actor outside inference mode.
    Initial transfer loads no PPO optimizer, critic, exploration std or counters.
    """

    def __init__(self, policy_config: dict, history_length: int = 5):
        super().__init__()
        from transformer_rl.formal.policy import (
            StructuredPolicy, StructuredPolicyConfig, policy_config_from_dict,
        )

        config = policy_config_from_dict(policy_config, complete=True)
        if not isinstance(config, StructuredPolicyConfig):
            raise ValueError("TransferredActor requires the upstream StructuredPolicy configuration")
        if (type(history_length) is not int or history_length < 1
                or config.history_length != history_length):
            raise ValueError("Transferred actor history length differs from its policy configuration")
        self.history_length = history_length
        self._config = config
        self.policy = StructuredPolicy(config).for_deployment()
        self.register_buffer("action_bounds", torch.tensor(ACTION_BOUNDS))
        source_files = {}
        for name in ("transformer_rl.formal.policy", "transformer_rl.formal.mlp_policy",
                     "transformer_rl.formal.critic", "transformer_rl.common.types"):
            module = importlib.import_module(name)
            path = Path(module.__file__).resolve(strict=True)
            source_files[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        source_hashes = {name: item["sha256"] for name, item in source_files.items()}
        self._source_metadata = {
            "format": "supermimic-transferred-actor-v1",
            "policy_config": asdict(config),
            "normalization": config.normalization_contract,
            "history_length": history_length,
            "observation_dim": 35,
            "action_bounds": list(ACTION_BOUNDS),
            "action_semantics": "upstream raw deterministic mean followed by per-axis hard clipping",
            "source_files": source_files,
            "source_sha256": hashlib.sha256(json.dumps(
                source_hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "initial_checkpoint": None,
        }

    @property
    def source_metadata(self) -> dict:
        """Return manifest evidence without exposing mutable internal metadata."""
        return deepcopy(self._source_metadata)

    def forward(self, history: Tensor) -> Tensor:
        if (not isinstance(history, Tensor) or history.ndim != 3 or history.shape[0] < 1
                or history.shape[1:] != (self.history_length, 35)):
            raise ValueError("TransferredActor requires chronological [batch, history, 35]")
        return self.policy(history).clamp(-self.action_bounds, self.action_bounds)

    def action_and_estimate(self, history: Tensor) -> tuple[Tensor, Tensor]:
        """Return clipped action and deployed velocity/contact logits in one pass.

        The course request controller consumes velocity and contact logits, not
        privileged simulator labels. The source head's log-variance is unused
        by that controller, so omit exactly its three middle entries here.
        """
        if history.ndim != 3 or history.shape[1:] != (self.history_length, 35):
            raise ValueError("TransferredActor requires chronological [batch, history, 35]")
        action, raw_state, _ = self.policy.outputs(history)
        estimate = torch.cat((raw_state[:, :3], raw_state[:, 6:8]), -1)
        return action.clamp(-self.action_bounds, self.action_bounds), estimate

    def load_initial_checkpoint(self, path, expected_sha256: str) -> dict:
        """Validate the pinned training checkpoint, then load only actor policy weights.

        Validate the complete upstream policy, including its fixed masks and its
        training-only prediction head, before dropping exactly that unused head.
        Loading preserves existing parameter identities, device and trainability.
        Source evidence belongs in the manifest; normal ``state_dict`` retains
        all actor parameters and persistent buffers for subsequent SM checkpoints.
        """
        from transformer_rl.formal.policy import StructuredPolicy, policy_config_from_dict

        if (not isinstance(expected_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None):
            raise ValueError("Initial actor checkpoint requires an explicit lowercase SHA256")
        path = Path(path).resolve(strict=True)
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected_sha256:
            raise ValueError("Initial actor checkpoint SHA256 differs from the expected pin")
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        if (not isinstance(payload, dict)
                or payload.get("format") != "private_structured_learning_snapshot"
                or not isinstance(payload.get("recipe"), dict)):
            raise ValueError("Initial actor checkpoint format or recipe differs")
        recipe = payload["recipe"]
        config = policy_config_from_dict(recipe.get("policy"), complete=True)
        if config != self._config or config.history_length != self.history_length:
            raise ValueError("Initial actor checkpoint policy configuration or history differs")
        if payload.get("normalization") != self._config.normalization_contract:
            raise ValueError("Initial actor checkpoint fixed normalization contract differs")
        for name in ("learner_source_sha256", "asset_manifest_sha256", "task_contract_sha256"):
            value = recipe.get(name)
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("Initial actor checkpoint recipe requires a source pin: " + name)
        collection = recipe.get("collection")
        if (not isinstance(collection, dict)
                or not isinstance(collection.get("action_bounds"), (list, tuple))
                or tuple(collection["action_bounds"]) != ACTION_BOUNDS):
            raise ValueError("Initial actor checkpoint action bounds differ")
        model = payload.get("model")
        if not isinstance(model, dict) or any(not isinstance(key, str) for key in model):
            raise ValueError("Initial actor checkpoint model state must be a named dictionary")
        prefix = "actor.policy."
        weights = {key[len(prefix):]: value for key, value in model.items() if key.startswith(prefix)}
        # Validation must not consume the learner's sampling RNG or mutate its
        # actor if any later tensor or fixed-buffer check fails.
        with torch.random.fork_rng(devices=[]):
            candidate = StructuredPolicy(self._config)
        expected = candidate.state_dict()
        if weights.keys() != expected.keys():
            raise ValueError("Initial actor checkpoint policy weight keys differ")
        buffers = dict(candidate.named_buffers())
        for name, reference in expected.items():
            value = weights[name]
            if (not isinstance(value, Tensor) or value.shape != reference.shape
                    or value.dtype != reference.dtype or not torch.isfinite(value).all()):
                raise ValueError("Initial actor checkpoint tensor differs: " + name)
            if name in buffers:
                if name == "time_encoding":
                    # CP18026's saved sin/cos table differs from this CPU's
                    # recomputation by at most one float32 epsilon. Preserve
                    # the saved table; tolerate only numerical roundoff here.
                    epsilon = 2 * torch.finfo(reference.dtype).eps
                    matches = torch.allclose(value, reference, atol=epsilon, rtol=epsilon)
                else:
                    matches = torch.equal(value, reference)
                if not matches:
                    raise ValueError("Initial actor checkpoint fixed policy buffer differs: " + name)
        candidate.load_state_dict(weights, strict=True)
        self.policy.load_state_dict(candidate.for_deployment().state_dict(), strict=True)
        self._source_metadata["initial_checkpoint"] = {
            "path": str(path), "sha256": actual,
            "learner_source_sha256": recipe["learner_source_sha256"],
            "asset_manifest_sha256": recipe["asset_manifest_sha256"],
            "task_contract_sha256": recipe["task_contract_sha256"],
            "normalization": payload["normalization"],
            "transfer_scope": "actor.policy only; no PPO optimizer, critic, std or counters",
        }
        return self.source_metadata
