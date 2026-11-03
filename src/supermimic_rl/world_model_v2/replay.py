"""GPU ring of compact v2 records with the same boundary rules as the packed replay.

Fitting windows use ``replay.eligible_windows``: they never cross an episode,
control epoch, invalid step or non-consecutive vector step, and may end in a
terminal step. Imagination starts may be any valid, policy-owned step; their
actor history is padded with the segment's first frame, exactly like the
collector after a reset or takeover, so the first steps of an episode stay
eligible.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from ..replay import BOUNDARY_FIELDS, eligible_windows
from .stair_task import STAIR_RECORD_DTYPES
from .stair_source import TOF_RECORD_DTYPES
from .jump_task import JUMP_RECORD_DTYPES, JUMP_TRIGGER_RECORD_DTYPES
from .recovery_task import RECOVERY_RECORD_DTYPES

_NATIVE_RECORD_DTYPES = {**STAIR_RECORD_DTYPES, **TOF_RECORD_DTYPES, **JUMP_RECORD_DTYPES,
                         **JUMP_TRIGGER_RECORD_DTYPES, **RECOVERY_RECORD_DTYPES}

_BOOL_FIELDS = ("valid", "terminated", "truncated", "policy_owned", "end_policy_owned")
_ID_FIELDS = ("start_episode_id", "end_episode_id", "start_control_epoch", "end_control_epoch")


class CompactReplay:
    def __init__(self, fields: dict[str, tuple[int, ...]], num_envs: int, capacity_steps: int,
                 device: str | torch.device):
        if num_envs < 1 or capacity_steps < 1:
            raise ValueError("Replay requires positive environment count and capacity")
        self.num_envs, self.capacity, self.device = num_envs, capacity_steps, torch.device(device)
        self.data = {name: torch.zeros(capacity_steps, num_envs, *shape, device=self.device,
                                      dtype=_NATIVE_RECORD_DTYPES.get(name, torch.float32))
                     for name, shape in fields.items()}
        for name in _BOOL_FIELDS:
            self.data[name] = torch.zeros(capacity_steps, num_envs, dtype=torch.bool, device=self.device)
        for name in _ID_FIELDS:
            self.data[name] = torch.zeros(capacity_steps, num_envs, dtype=torch.int64, device=self.device)
        # Steps since the segment's first frame (episode or control epoch start).
        self.data["segment_age"] = torch.zeros(capacity_steps, num_envs, dtype=torch.int64, device=self.device)
        self.sequence = torch.full((capacity_steps,), -1, dtype=torch.int64, device=self.device)
        self.size = self.head = self.total_steps = 0
        self._cache: dict = {}

    def __len__(self) -> int:
        return self.size * self.num_envs

    @property
    def total_added(self) -> int:
        return self.total_steps * self.num_envs

    def state_dict(self) -> dict:
        """Persist only populated chronological rows, including their actual sequence IDs."""
        # A full advanced-index gather of the tick field creates another ~3.7 GB
        # GPU tensor at 4096 x 1000. Copy the ring's two contiguous views directly
        # into the CPU snapshot instead; checkpointing must not need that headroom.
        first_slot = (self.head - self.size) % self.capacity
        first_count = min(self.size, self.capacity - first_slot)

        def snapshot(value):
            output = torch.empty(self.size, *value.shape[1:], dtype=value.dtype, device='cpu')
            output[:first_count].copy_(value[first_slot:first_slot + first_count])
            output[first_count:].copy_(value[:self.size - first_count])
            return output

        return {'format': 'supermimic-compact-replay-v1', 'num_envs': self.num_envs,
                'capacity': self.capacity, 'total_steps': self.total_steps,
                'sequence': snapshot(self.sequence),
                'storage': {name: snapshot(value) for name, value in self.data.items()}}

    @classmethod
    def from_state_dict(cls, payload: dict, device='cpu', *, capacity_steps=None, env_ids=None):
        if payload.get('format') != 'supermimic-compact-replay-v1':
            raise ValueError('Unknown compact replay format')
        data, sequence = payload['storage'], payload['sequence']
        n, size, capacity = payload['num_envs'], len(sequence), payload['capacity']
        if (sequence.dtype != torch.int64 or sequence.ndim != 1 or size > capacity
                or (size and (sequence[0] < 0 or sequence[-1] >= payload['total_steps']))
                or (sequence[1:] <= sequence[:-1]).any()):
            raise ValueError('Compact replay chronology or capacity differs')
        reserved = set(_BOOL_FIELDS + _ID_FIELDS + ('segment_age',))
        if not reserved.issubset(data):
            raise ValueError('Compact replay lacks ownership, episode or history boundaries')
        for name, value in data.items():
            dtype = (torch.bool if name in _BOOL_FIELDS else torch.int64 if name in reserved
                     else _NATIVE_RECORD_DTYPES.get(name, torch.float32))
            if value.shape[:2] != (size, n) or value.dtype != dtype:
                raise ValueError('Invalid compact field shape/dtype: ' + name)
        capacity = capacity if capacity_steps is None else capacity_steps
        if capacity < size:
            raise ValueError('Restoring replay may not silently discard recorded history')
        if env_ids is not None:
            env_ids = torch.as_tensor(env_ids, dtype=torch.int64)
            if (env_ids.ndim != 1 or not len(env_ids) or env_ids.unique().numel() != len(env_ids)
                    or (env_ids < 0).any() or (env_ids >= n).any()):
                raise ValueError('Replay environment subset must contain distinct recorded rows')
            n = len(env_ids)
        replay = cls({name: tuple(value.shape[2:]) for name, value in data.items() if name not in reserved},
                     n, capacity, device)
        for name, value in data.items():
            if env_ids is not None:
                value = value.index_select(1, env_ids.to(value.device))
            replay.data[name][:size].copy_(value)
        replay.sequence[:size].copy_(sequence)
        replay.size, replay.head, replay.total_steps = size, size % capacity, payload['total_steps']
        return replay

    @torch.no_grad()
    def add(self, record: dict[str, Tensor], boundary: dict[str, Tensor], *, sequence_id: int | None = None) -> None:
        sequence_id = self.total_steps if sequence_id is None else sequence_id
        if sequence_id < self.total_steps:
            raise ValueError("Replay sequence IDs must strictly increase")
        for name, dtype in _NATIVE_RECORD_DTYPES.items():
            if name in record and record[name].dtype != dtype:
                raise ValueError('Task IDs/clocks must retain their native replay dtype: ' + name)
        slot = self.head
        if self.size:
            previous = (slot - 1) % self.capacity
            data = self.data
            new_segment = (data["terminated"][previous] | data["truncated"][previous]
                           | ~data["valid"][previous] | ~data["policy_owned"][previous]
                           | ~data["end_policy_owned"][previous]
                           | (sequence_id != self.sequence[previous] + 1)
                           | (data["start_episode_id"][previous] != data["end_episode_id"][previous])
                           | (data["start_control_epoch"][previous] != data["end_control_epoch"][previous])
                           | (boundary["start_episode_id"] != data["end_episode_id"][previous])
                           | (boundary["start_control_epoch"] != data["end_control_epoch"][previous]))
            age = torch.where(new_segment, 0, data["segment_age"][previous] + 1)
        else:
            age = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        for name, value in record.items():
            self.data[name][slot].copy_(value.detach())
        for name in _BOOL_FIELDS + _ID_FIELDS:
            self.data[name][slot].copy_(boundary[name])
        self.data["segment_age"][slot].copy_(age)
        self.sequence[slot] = sequence_id
        self.total_steps = sequence_id + 1
        self.size = min(self.size + 1, self.capacity)
        self.head = (slot + 1) % self.capacity
        self._cache.clear()

    def _chronological(self) -> Tensor:
        return (torch.arange(self.size, device=self.device) + self.head - self.size) % self.capacity

    def _allowed(self, candidates: Tensor, env_ids) -> Tensor:
        if env_ids is None:
            return candidates
        allowed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        allowed[torch.as_tensor(env_ids, device=self.device)] = True
        return candidates[allowed[candidates % self.num_envs]]

    def window_candidates(self, horizon: int, env_ids: Sequence[int] | None = None) -> Tensor:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        key = ("window", horizon)
        if key not in self._cache:
            if self.size < horizon:
                self._cache[key] = torch.empty(0, dtype=torch.int64, device=self.device)
            else:
                slots = self._chronological()
                chronological = {name: self.data[name][slots] for name in BOUNDARY_FIELDS}
                eligible = eligible_windows(chronological, self.sequence[slots], horizon)
                self._cache[key] = eligible.reshape(-1).nonzero().flatten()
        return self._allowed(self._cache[key], env_ids)

    def start_candidates(self, env_ids: Sequence[int] | None = None) -> Tensor:
        key = ("start",)
        if key not in self._cache:
            slots = self._chronological()
            owned = (self.data["valid"][slots] & self.data["policy_owned"][slots]
                     & self.data["end_policy_owned"][slots]
                     & (self.data["start_episode_id"][slots] == self.data["end_episode_id"][slots])
                     & (self.data["start_control_epoch"][slots] == self.data["end_control_epoch"][slots]))
            if "failure_memory" in self.data:
                owned &= self.data["failure_memory"][slots, :, 2] == 0
                owned &= self.data["failure_conditions_valid"][slots, :, 0] > .5
            self._cache[key] = owned.reshape(-1).nonzero().flatten()
        return self._allowed(self._cache[key], env_ids)

    def _choose(self, candidates: Tensor, batch: int, generator) -> tuple[Tensor, Tensor]:
        if batch < 1:
            raise ValueError("batch must be positive")
        if not candidates.numel():
            raise ValueError("No eligible replay rows")
        index = torch.randint(candidates.numel(), (batch,), generator=generator, device=self.device)
        selected = candidates[index]
        return selected // self.num_envs, selected % self.num_envs

    def sample_windows(self, batch: int, horizon: int, *, generator=None, env_ids=None,
                       candidates=None, field_names: Sequence[str] | None = None,
                       out: dict[str, Tensor] | None = None) -> dict[str, Tensor]:
        """Consecutive fields from one draw, optionally projected into reusable storage.

        Projection never changes eligible rows or the random draw. CPU callers
        can provide pinned output buffers to avoid allocating and staging every
        field afresh when only a model-training batch is needed on the GPU.
        """
        names = tuple(self.data) if field_names is None else tuple(field_names)
        if not names or len(set(names)) != len(names) or any(name not in self.data for name in names):
            raise ValueError('Replay projection requires distinct existing field names')
        if out is not None:
            if set(out) != set(names):
                raise ValueError('Replay output buffers must match the selected fields')
            for name in names:
                source, target = self.data[name], out[name]
                if (target.shape != (batch, horizon, *source.shape[2:]) or target.dtype != source.dtype
                        or target.device != self.device or not target.is_contiguous()):
                    raise ValueError('Replay output buffer shape, dtype, device or layout differs: ' + name)
        candidates = (self.window_candidates(horizon, env_ids) if candidates is None
                      else self._allowed(candidates, env_ids))
        offset, env = self._choose(candidates, batch, generator)
        steps = (offset[:, None] + torch.arange(horizon, device=self.device)) + self.head - self.size
        slots = steps % self.capacity
        if out is None:
            return {name: self.data[name][slots, env[:, None]] for name in names}
        flat = (slots * self.num_envs + env[:, None]).flatten()
        for name in names:
            torch.index_select(self.data[name].flatten(0, 1), 0, flat, out=out[name].flatten(0, 1))
        return out

    def sample_starts(self, batch: int, history: int, *, generator=None, env_ids=None, candidates=None):
        """Start-step fields ``[B, ...]`` and the actor's history frames ``[B, history, 45]``."""
        if history < 1:
            raise ValueError("history must be positive")
        candidates = self.start_candidates(env_ids) if candidates is None else self._allowed(candidates, env_ids)
        offset, env = self._choose(candidates, batch, generator)
        slot = (offset + self.head - self.size) % self.capacity
        start = {name: value[slot, env] for name, value in self.data.items()}
        # Older frames than the segment start, or than the buffer, repeat the first available frame.
        back = torch.arange(history - 1, -1, -1, device=self.device)[None]
        back = torch.minimum(back, torch.minimum(start["segment_age"], offset)[:, None])
        frames = self.data["frame"][(slot[:, None] - back) % self.capacity, env[:, None]]
        return start, frames
