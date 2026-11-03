"""Explicit tensor contracts at the simulator, replay and learner boundaries."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from functools import cached_property
import hashlib
import json
import math
from typing import Mapping

import torch
from torch import Tensor

ACTION_BOUNDS = (3.0, 3.0, 3.0, 3.0, 9.0, 9.0)


def action_bounds_from_source(source: Mapping) -> tuple[float, ...]:
    """Execution bounds pinned by a native schema source (legacy when absent).

    The native runtime's ``task.action_bounds`` owns this value; schema sources
    record it only when it differs from the legacy contract.
    """
    bounds = tuple(float(value) for value in source.get("action_bounds", ACTION_BOUNDS))
    if len(bounds) != 6 or not all(math.isfinite(value) and value > 0 for value in bounds):
        raise ValueError("Native action bounds must be six finite positive values")
    if len(set(bounds[:4])) != 1:
        raise ValueError("All four leg axes must share one action bound")
    return bounds
OBSERVATION_DIM = 35
GOAL_DIM = 11


class _TensorValidation:
    """Collect device flags and perform one host decision at a tensor boundary.

    Shape/dtype/device checks stay immediate. Failed value checks retain their
    labels; expensive per-field diagnostics are evaluated only on failure.
    Instances belong to one synchronous validation call, never to cached data.
    """

    def __init__(self):
        self._flags = []
        self._labels = []
        self._device = None

    def require(self, label, violated):
        if self._device is None:
            self._device = violated.device
        elif violated.device != self._device:
            raise ValueError("Validated tensors must share a device")
        self._flags.append(violated.any())
        self._labels.append(label)

    def finish(self):
        if not self._flags:
            return
        failures = torch.stack(self._flags)
        if bool(failures.any()):
            first = failures.nonzero(as_tuple=False)[0, 0].item()
            label = self._labels[first]
            raise ValueError(label() if callable(label) else label)


@dataclass(frozen=True)
class Field:
    name: str
    shape: tuple[int, ...]
    units: str = "source-defined"
    frame: str = "source-defined"

    def __post_init__(self):
        if not self.name or any(type(n) is not int or n <= 0 for n in self.shape):
            raise ValueError("A field needs a name and positive dimensions")

    @property
    def size(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class StateSchema:
    state_fields: tuple[Field, ...]
    context_fields: tuple[Field, ...]
    policy_dt_s: float = 0.02
    observation_dim: int = OBSERVATION_DIM
    action_dim: int = 6
    goal_dim: int = GOAL_DIM
    source: str = "unspecified"
    # Physics-tick truth inside one policy interval, one row per substep.
    substep_fields: tuple[Field, ...] = ()
    substeps_per_step: int = 0

    def __post_init__(self):
        if not self.state_fields or not math.isfinite(self.policy_dt_s) or self.policy_dt_s <= 0:
            raise ValueError("A schema requires state fields and a positive policy period")
        for group in (self.state_fields, self.context_fields, self.substep_fields):
            if len({f.name for f in group}) != len(group):
                raise ValueError("Duplicate field names")
        if (self.observation_dim not in (35, 45) or self.action_dim != 6
                or type(self.goal_dim) is not int or self.goal_dim < 11):
            raise ValueError("Wheel-leg schemas require 35/45 observations, six actions and at least 11 goals")
        if type(self.substeps_per_step) is not int or (
                (self.substeps_per_step > 0) != bool(self.substep_fields)) or self.substeps_per_step < 0:
            raise ValueError("Substep fields require a positive substep count, and only then")

    @property
    def state_dim(self) -> int:
        return sum(f.size for f in self.state_fields)

    @property
    def context_dim(self) -> int:
        return sum(f.size for f in self.context_fields)

    @property
    def substep_dim(self) -> int:
        return sum(f.size for f in self.substep_fields)

    @cached_property
    def action_bounds(self) -> tuple[float, ...]:
        """Executed-action contract of a native JSON source; legacy for other sources."""
        try:
            source = json.loads(self.source)
        except (TypeError, ValueError):
            return ACTION_BOUNDS
        return action_bounds_from_source(source) if isinstance(source, dict) else ACTION_BOUNDS

    def to_dict(self) -> dict:
        result = asdict(self)
        # Schemas without substeps keep their original serialization and hash.
        if not self.substep_fields:
            del result["substep_fields"], result["substeps_per_step"]
        return result

    @classmethod
    def from_dict(cls, payload: dict) -> StateSchema:
        payload = dict(payload)
        for name in ("state_fields", "context_fields", "substep_fields"):
            if name in payload:
                payload[name] = tuple(Field(**{**f, "shape": tuple(f["shape"])})
                                      for f in payload[name])
        return cls(**payload)

    @property
    def fingerprint(self) -> str:
        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(body.encode()).hexdigest()

    def pack(self, values: Mapping[str, Tensor], *, context: bool = False,
             _checks: _TensorValidation | None = None) -> Tensor:
        spec = self.context_fields if context else self.state_fields
        if set(values) != {f.name for f in spec}:
            raise ValueError("Snapshot fields do not match the frozen schema")
        if not spec:
            raise ValueError("Empty context must be constructed with its batch and device")
        if any(not isinstance(value, Tensor) or value.ndim < 1 or value.is_complex()
               for value in values.values()):
            raise ValueError("Snapshot fields require real batched tensors")
        batches = {value.shape[0] for value in values.values()}
        if len(batches) != 1:
            raise ValueError("Snapshot fields disagree on batch size")
        if len({value.device for value in values.values()}) != 1:
            raise ValueError("Snapshot fields must share a device")
        result = []
        for field in spec:
            value = values[field.name]
            if tuple(value.shape[1:]) != field.shape:
                raise ValueError(f"Shape changed for {field.name}")
            result.append(value.reshape(value.shape[0], -1).to(torch.float32))
        packed = torch.cat(result, dim=-1)

        def diagnose_nonfinite():
            offset = 0
            for field in spec:
                if not torch.isfinite(packed[:, offset:offset + field.size]).all():
                    return f"Nonfinite snapshot field: {field.name}"
                offset += field.size
            return "Nonfinite packed snapshot"

        checks = _TensorValidation() if _checks is None else _checks
        # Inspect the converted buffer as well: finite float64 can overflow
        # float32, and must not escape the learning-state boundary as infinity.
        checks.require(diagnose_nonfinite, ~torch.isfinite(packed))
        if _checks is None:
            checks.finish()
        return packed

    def pack_substeps(self, values: Mapping[str, Tensor], *,
                      _checks: _TensorValidation | None = None) -> Tensor:
        """Pack ``[batch, substeps, *shape]`` tensors into ``float32[batch, substeps, D]``."""
        if set(values) != {f.name for f in self.substep_fields}:
            raise ValueError("Substep fields do not match the frozen schema")
        if any(not isinstance(value, Tensor) or value.ndim < 2 or value.is_complex()
               for value in values.values()):
            raise ValueError("Substep fields require real [batch, substep, ...] tensors")
        leading = {tuple(value.shape[:2]) for value in values.values()}
        if len(leading) != 1 or next(iter(leading))[1] != self.substeps_per_step:
            raise ValueError("Substep fields disagree on batch or substep count")
        if len({value.device for value in values.values()}) != 1:
            raise ValueError("Substep fields must share a device")
        result = []
        for field in self.substep_fields:
            value = values[field.name]
            if tuple(value.shape[2:]) != field.shape:
                raise ValueError(f"Shape changed for substep field {field.name}")
            result.append(value.reshape(*value.shape[:2], -1).to(torch.float32))
        packed = torch.cat(result, dim=-1)
        checks = _TensorValidation() if _checks is None else _checks
        checks.require("Nonfinite substep field", ~torch.isfinite(packed))
        if _checks is None:
            checks.finish()
        return packed


@dataclass
class Snapshot:
    state: Tensor
    observation: Tensor
    goal: Tensor
    context: Tensor
    episode_id: Tensor
    control_epoch: Tensor
    policy_owned: Tensor
    actor_observation: Tensor | None = None
    estimator_target: Tensor | None = None
    estimator_valid: Tensor | None = None

    def validate(self, schema: StateSchema, *, _checks: _TensorValidation | None = None) -> None:
        if not isinstance(self.state, Tensor) or self.state.ndim != 2:
            raise ValueError("state must be a batched matrix")
        n = self.state.shape[0]
        checks = _TensorValidation() if _checks is None else _checks
        for name, width in (("state", schema.state_dim), ("context", schema.context_dim),
                            ("observation", schema.observation_dim), ("goal", schema.goal_dim)):
            value = getattr(self, name)
            if not isinstance(value, Tensor) or value.shape != (n, width) or value.dtype != torch.float32:
                raise ValueError(f"{name} must be float32[{n},{width}]")
            checks.require(f"Nonfinite {name}", ~torch.isfinite(value))
        for name in ("episode_id", "control_epoch", "policy_owned"):
            value = getattr(self, name)
            dtype = torch.bool if name == "policy_owned" else torch.int64
            if not isinstance(value, Tensor) or value.shape != (n,) or value.dtype != dtype:
                raise ValueError(f"Invalid {name}")
        if self.actor_observation is not None:
            if (not isinstance(self.actor_observation, Tensor)
                    or self.actor_observation.shape != (n, schema.observation_dim)
                    or self.actor_observation.dtype != torch.float32):
                raise ValueError("Invalid actor observation")
            checks.require("Invalid actor observation", ~torch.isfinite(self.actor_observation))
        elif schema.observation_dim == 45:
            raise ValueError("Formal observations require their actual deployment frame")
        if (self.estimator_target is None) != (self.estimator_valid is None):
            raise ValueError("Estimator labels require a matching validity mask")
        if self.estimator_target is not None:
            if (not isinstance(self.estimator_target, Tensor) or not isinstance(self.estimator_valid, Tensor)
                    or self.estimator_target.shape != (n, 5) or self.estimator_target.dtype != torch.float32
                    or self.estimator_valid.shape != (n,) or self.estimator_valid.dtype != torch.bool):
                raise ValueError("Invalid body-velocity/contact estimator labels")
            checks.require("Invalid body-velocity/contact estimator labels", ~torch.isfinite(self.estimator_target))
            checks.require("Invalid body-velocity/contact estimator labels",
                           (self.estimator_target[:, 3:] < 0) | (self.estimator_target[:, 3:] > 1))
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None and value.device != self.state.device:
                raise ValueError("Snapshot tensors must share a device")
        if _checks is None:
            checks.finish()

    def clone(self) -> Snapshot:
        return Snapshot(**{f.name: None if getattr(self, f.name) is None
                           else getattr(self, f.name).detach().clone() for f in fields(self)})


@dataclass
class Transition:
    start: Snapshot
    action: Tensor
    end: Snapshot
    reward: Tensor
    terminated: Tensor
    truncated: Tensor
    valid: Tensor
    # float32[batch, substeps, substep_dim]; the last row is the pre-reset end.
    substeps: Tensor | None = None

    def validate(self, schema: StateSchema, *, _checks: _TensorValidation | None = None) -> None:
        checks = _TensorValidation() if _checks is None else _checks
        self.start.validate(schema, _checks=checks)
        self.end.validate(schema, _checks=checks)
        n = self.start.state.shape[0]
        if self.end.state.shape[0] != n or self.action.shape != (n, 6):
            raise ValueError("Transition batch or action shape mismatch")
        if schema.substep_fields:
            expected = (n, schema.substeps_per_step, schema.substep_dim)
            if (not isinstance(self.substeps, Tensor) or self.substeps.shape != expected
                    or self.substeps.dtype != torch.float32
                    or self.substeps.device != self.start.state.device):
                raise ValueError(f"substeps must be float32{list(expected)}")
            checks.require("Nonfinite substep truth", ~torch.isfinite(self.substeps))
        elif self.substeps is not None:
            raise ValueError("The schema declares no substep truth")
        if self.action.dtype != torch.float32:
            raise ValueError("Invalid executed action")
        checks.require("Invalid executed action", ~torch.isfinite(self.action))
        bounds = self.action.new_tensor(schema.action_bounds)
        # Recovery preparation can issue exact script targets outside the
        # actor's box. Keep those executed commands for dynamics supervision;
        # the policy action contract belongs only to the interval's owner.
        checks.require("Store executed bounded actions, not pre-clip samples",
                       (self.action.abs() > bounds + 1e-6) & self.start.policy_owned[:, None])
        if self.reward.shape != (n,):
            raise ValueError("Invalid reward")
        checks.require("Invalid reward", ~torch.isfinite(self.reward))
        for name in ("terminated", "truncated", "valid"):
            value = getattr(self, name)
            if value.shape != (n,) or value.dtype != torch.bool:
                raise ValueError(f"Invalid {name}")
        for name in ("action", "reward", "terminated", "truncated", "valid"):
            if getattr(self, name).device != self.start.state.device:
                raise ValueError("Transition tensors must share a device")
        same_episode = self.start.episode_id == self.end.episode_id
        checks.require("Next state must be captured BEFORE auto-reset", ~same_episode & self.valid)
        if _checks is None:
            checks.finish()


@dataclass
class WindowBatch:
    states: Tensor
    observations: Tensor
    goals: Tensor
    contexts: Tensor
    actions: Tensor
    rewards: Tensor
    terminated: Tensor
    truncated: Tensor
    valid: Tensor
    policy_owned: Tensor
    actor_observations: Tensor | None = None
    estimator_targets: Tensor | None = None
    estimator_valid: Tensor | None = None
    # [batch, horizon, substeps, substep_dim], aligned with actions.
    substeps: Tensor | None = None

    def to(self, device: str | torch.device) -> WindowBatch:
        return WindowBatch(**{f.name: None if getattr(self, f.name) is None
                              else getattr(self, f.name).to(device) for f in fields(self)})


def encoded_goal(observation: Tensor) -> Tensor:
    """Commands use the existing actor's units/scales and context ordering."""
    return torch.cat((observation[..., :4], observation[..., 28:35]), dim=-1)
