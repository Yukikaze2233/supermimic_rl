"""Named, read-only access to recorded substep truth in replay storage.

The recorder (``adapters.substeps``) declares every field with its native
joint/body order in the field frame. This view resolves those names once, so
consumers (measurements, parity tests, interval-event labels) never hard-code
column offsets or rely on a second copy of the native order.
"""
from __future__ import annotations

from collections.abc import Mapping
import re

import torch
from torch import Tensor

from ..contracts import StateSchema


def _names_in(frame: str, kind: str) -> tuple[str, ...] | None:
    match = re.search(rf"native {kind} order: ([^;]+)", frame)
    return None if match is None else tuple(name.strip() for name in match.group(1).split(","))


class SubstepView:
    """Chronological ``[T, N, K, ...]`` substep fields plus transition metadata."""

    def __init__(self, schema: StateSchema, storage: Mapping[str, Tensor], *,
                 dtype: torch.dtype = torch.float64):
        if not schema.substep_fields or "substeps" not in storage:
            raise ValueError("Replay has no substep truth; collect with record_substeps=true")
        self.schema, self.storage = schema, storage
        self.values = storage["substeps"].to(dtype)
        self.steps, self.num_envs, self.substeps, _ = self.values.shape
        self.physics_dt_s = schema.policy_dt_s / self.substeps
        self._slices, offset = {}, 0
        for field in schema.substep_fields:
            self._slices[field.name] = (slice(offset, offset + field.size), field)
            offset += field.size
        self.joint_names = _names_in(self.field("joint_position_continuous").frame, "joint")
        self.body_names = _names_in(self.field("body_normal_load_n").frame, "body")
        if self.joint_names is None or self.body_names is None:
            raise ValueError("Substep fields must declare their native joint and body order")

    @classmethod
    def from_replay_payload(cls, payload: Mapping, **options) -> SubstepView:
        """Read a saved ``ReplayBuffer.state_dict``, restoring time order after ring wrap."""
        order = torch.argsort(payload["sequence"])
        storage = {name: value.index_select(0, order) for name, value in payload["storage"].items()}
        storage["sequence"] = payload["sequence"].index_select(0, order)
        return cls(StateSchema.from_dict(payload["schema"]), storage, **options)

    def field(self, name: str):
        return self._slices[name][1]

    def get(self, name: str) -> Tensor:
        index, field = self._slices[name]
        return self.values[..., index].reshape(self.steps, self.num_envs, self.substeps, *field.shape)

    def has(self, name: str) -> bool:
        return name in self._slices

    def joints(self, names) -> list[int]:
        return [self.joint_names.index(name) for name in names]

    def bodies(self, names) -> list[int]:
        return [self.body_names.index(name) for name in names]

    @property
    def valid(self) -> Tensor:
        return self.storage["valid"].bool()

    @property
    def ended(self) -> Tensor:
        return self.storage["terminated"].bool() | self.storage["truncated"].bool()

    def follows_previous(self) -> Tensor:
        """``[T, N]``: interval t starts exactly where valid interval t-1 ended."""
        data = self.storage
        result = torch.zeros(self.steps, self.num_envs, dtype=torch.bool)
        same = ((data["end_episode_id"][:-1] == data["start_episode_id"][1:])
                & (data["end_control_epoch"][:-1] == data["start_control_epoch"][1:]))
        sequence = data.get("sequence")
        if sequence is not None:
            same &= (sequence[1:] == sequence[:-1] + 1)[:, None]
        result[1:] = self.valid[:-1] & self.valid[1:] & ~self.ended[:-1] & same
        return result

    def robot_weight_n(self) -> Tensor | None:
        """``[T, N]`` total weight from the start context, when body masses are recorded."""
        fields = self.schema.context_fields
        for position, field in enumerate(fields):
            if field.name == "body_mass_kg":
                offset = sum(f.size for f in fields[:position])
                mass = self.storage["start_context"][..., offset:offset + field.size].double()
                return mass.sum(-1) * 9.81
        return None
