"""Tensor replay with explicit episode and controller-ownership boundaries."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .contracts import StateSchema, Transition, WindowBatch


BOUNDARY_FIELDS = ("valid", "terminated", "truncated", "policy_owned", "end_policy_owned",
                   "start_episode_id", "end_episode_id", "start_control_epoch", "end_control_epoch")


def eligible_windows(data: dict[str, Tensor], sequence: Tensor, horizon: int, *,
                     policy_owned: bool = False, actor_history: bool = False) -> Tensor:
    """``[T - horizon + 1, N]`` mask of windows that never cross a boundary.

    ``data`` holds the chronological ``[T, N]`` BOUNDARY_FIELDS and ``sequence``
    the ``[T]`` vector-step ids. A window must be valid at every step, may end
    in a terminal step but never continue past one, and never joins different
    episodes, control epochs or non-consecutive steps. Actor histories also
    require policy ownership throughout and a policy-owned endpoint.
    """
    num_envs = data["valid"].shape[-1]
    device = data["valid"].device
    valid = data["valid"] & (data["start_episode_id"] == data["end_episode_id"]) & (
        data["start_control_epoch"] == data["end_control_epoch"])
    if actor_history:
        valid &= data["policy_owned"] & ~data["terminated"] & ~data["truncated"]
    # Prefix sums keep window checks linear in storage size, not horizon.
    invalid_prefix = torch.cat((torch.zeros(1, num_envs, dtype=torch.int64, device=device),
                                (~valid).to(torch.int64).cumsum(dim=0)))
    eligible = invalid_prefix[horizon:] == invalid_prefix[:-horizon]
    if horizon > 1:
        boundary = data["terminated"][:-1] | data["truncated"][:-1]
        for name in ("episode_id", "control_epoch"):
            boundary = boundary | (data[f"end_{name}"][:-1] != data[f"start_{name}"][1:])
        boundary = boundary | (sequence[1:] != sequence[:-1] + 1).unsqueeze(-1)
        boundary_prefix = torch.cat((torch.zeros(1, num_envs, dtype=torch.int64, device=device),
                                     boundary.to(torch.int64).cumsum(dim=0)))
        eligible &= boundary_prefix[horizon - 1:] == boundary_prefix[:-(horizon - 1)]
    if policy_owned:
        eligible &= data["policy_owned"][:eligible.shape[0]]
    if actor_history:
        eligible &= data["end_policy_owned"][horizon - 1:]
    return eligible


class ReplayBuffer:
    """Store synchronized environment batches in a fixed-size time ring.

    Capacity counts transitions, so storage rounds down to a whole vector step.
    Sampling uses replacement and never mixes environments within a window.
    The collector must submit every successive control step, including invalid
    steps, so omitted data cannot masquerade as consecutive simulation states.
    """

    FORMAT_VERSION = 2
    _ENDPOINT_AUXILIARY_FIELDS = (
        "actor_observation", "estimator_target", "estimator_valid",
    )

    def __init__(
        self,
        schema: StateSchema,
        num_envs: int,
        capacity: int,
        device: str | torch.device = "cpu",
    ) -> None:
        if type(num_envs) is not int or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if type(capacity) is not int or capacity < num_envs:
            raise ValueError("capacity must hold at least one vector step")
        self.schema = schema
        self.num_envs = num_envs
        self.capacity = capacity
        self.device = torch.device(device)
        self.capacity_steps = capacity // num_envs
        self.effective_capacity = self.capacity_steps * num_envs
        self._size_steps = 0
        self._write_index = 0
        self._total_steps = 0
        self._candidate_cache: dict[tuple[int, bool, bool], Tensor] = {}

        # Both endpoints are needed to retain terminal states before auto-reset.
        widths = {
            "state": schema.state_dim,
            "observation": schema.observation_dim,
            "actor_observation": schema.observation_dim,
            "estimator_target": 5,
            "goal": schema.goal_dim,
            "context": schema.context_dim,
        }
        self._storage: dict[str, Tensor] = {}
        for endpoint in ("start", "end"):
            for name, width in widths.items():
                self._storage[f"{endpoint}_{name}"] = torch.zeros(
                    self.capacity_steps, num_envs, width,
                    dtype=torch.float32, device=self.device,
                )
            for name in ("episode_id", "control_epoch"):
                self._storage[f"{endpoint}_{name}"] = torch.zeros(
                    self.capacity_steps, num_envs,
                    dtype=torch.int64, device=self.device,
                )
            self._storage[f"{endpoint}_estimator_valid"] = torch.zeros(
                self.capacity_steps, num_envs,
                dtype=torch.bool, device=self.device,
            )
        self._storage["actions"] = torch.zeros(
            self.capacity_steps, num_envs, schema.action_dim,
            dtype=torch.float32, device=self.device,
        )
        self._storage["rewards"] = torch.zeros(
            self.capacity_steps, num_envs, dtype=torch.float32, device=self.device,
        )
        if schema.substep_fields:
            self._storage["substeps"] = torch.zeros(
                self.capacity_steps, num_envs, schema.substeps_per_step, schema.substep_dim,
                dtype=torch.float32, device=self.device,
            )
        for name in (
            "terminated", "truncated", "valid", "policy_owned", "end_policy_owned",
        ):
            self._storage[name] = torch.zeros(
                self.capacity_steps, num_envs, dtype=torch.bool, device=self.device,
            )
        self._sequence = torch.full(
            (self.capacity_steps,), -1, dtype=torch.int64, device=self.device,
        )

    def __len__(self) -> int:
        return self._size_steps * self.num_envs

    @property
    def num_samples(self) -> int:
        """Number of stored transitions, including invalid transitions."""
        return len(self)

    @property
    def total_added(self) -> int:
        return self._total_steps * self.num_envs

    @property
    def num_valid(self) -> int:
        return self.num_candidates(1)

    @property
    def num_valid_starts(self) -> int:
        return self.num_candidates(1, policy_owned=True)

    @torch.no_grad()
    def add(self, transition: Transition) -> None:
        transition.validate(self.schema)
        if transition.start.state.shape[0] != self.num_envs:
            raise ValueError("Transition batch does not match replay num_envs")
        auxiliary = {}
        for endpoint in ("start", "end"):
            snapshot = getattr(transition, endpoint)
            values = {
                name: getattr(snapshot, name)
                for name in self._ENDPOINT_AUXILIARY_FIELDS
            }
            if self.schema.observation_dim != 35 and values["actor_observation"] is None:
                raise ValueError(
                    "Nonlegacy replay requires explicit actor observations at both endpoints",
                )
            if values["actor_observation"] is None:
                values["actor_observation"] = snapshot.observation
            # An absent legacy label has no supervision authority. Zero is a
            # storage placeholder, never a claim about measured body velocity.
            if values["estimator_target"] is None:
                if values["estimator_valid"] is not None and values["estimator_valid"].any():
                    raise ValueError("Valid estimator labels require explicit targets")
                values["estimator_target"] = snapshot.state.new_zeros(self.num_envs, 5)
            if values["estimator_valid"] is None:
                values["estimator_valid"] = torch.zeros(
                    self.num_envs, dtype=torch.bool, device=snapshot.state.device,
                )
            auxiliary[endpoint] = values
        slot = self._write_index
        for endpoint in ("start", "end"):
            snapshot = getattr(transition, endpoint)
            for name in (
                "state", "observation", "goal", "context",
                "episode_id", "control_epoch",
            ):
                self._storage[f"{endpoint}_{name}"][slot].copy_(
                    getattr(snapshot, name).detach(),
                )
            for name, value in auxiliary[endpoint].items():
                self._storage[f"{endpoint}_{name}"][slot].copy_(value.detach())
        for destination, value in (
            ("actions", transition.action),
            ("rewards", transition.reward),
            ("terminated", transition.terminated),
            ("truncated", transition.truncated),
            ("valid", transition.valid),
            ("policy_owned", transition.start.policy_owned),
            ("end_policy_owned", transition.end.policy_owned),
        ):
            self._storage[destination][slot].copy_(value.detach())
        if "substeps" in self._storage:
            self._storage["substeps"][slot].copy_(transition.substeps.detach())
        self._sequence[slot] = self._total_steps
        self._total_steps += 1
        self._size_steps = min(self._size_steps + 1, self.capacity_steps)
        self._write_index = (slot + 1) % self.capacity_steps
        self._candidate_cache.clear()

    def _chronological_slots(self) -> Tensor:
        oldest = (self._write_index - self._size_steps) % self.capacity_steps
        return (
            torch.arange(self._size_steps, device=self.device) + oldest
        ) % self.capacity_steps

    def _candidates(
        self,
        horizon: int,
        policy_owned: bool = False,
        actor_history: bool = False,
    ) -> Tensor:
        if type(horizon) is not int or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        key = (horizon, policy_owned, actor_history)
        if key in self._candidate_cache:
            return self._candidate_cache[key]
        if self._size_steps < horizon:
            return torch.empty(0, dtype=torch.int64, device=self.device)

        slots = self._chronological_slots()
        chronological = {name: self._storage[name][slots] for name in BOUNDARY_FIELDS}
        eligible = eligible_windows(chronological, self._sequence[slots], horizon,
                                    policy_owned=policy_owned, actor_history=actor_history)
        candidates = eligible.reshape(-1).nonzero(as_tuple=False).flatten()
        self._candidate_cache[key] = candidates
        return candidates

    def _filter_environments(
        self,
        candidates: Tensor,
        env_ids: Tensor | Sequence[int] | None,
    ) -> Tensor:
        if env_ids is None:
            return candidates
        if isinstance(env_ids, Tensor):
            integer_types = (
                torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
            )
            if env_ids.ndim != 1 or env_ids.dtype not in integer_types:
                raise ValueError("env_ids must be a one-dimensional integer tensor")
            selected = env_ids.to(device=self.device, dtype=torch.int64)
        else:
            if not isinstance(env_ids, Sequence) or any(
                type(index) is not int for index in env_ids
            ):
                raise ValueError("env_ids must be a sequence of integers")
            selected = torch.tensor(env_ids, dtype=torch.int64, device=self.device)
        if selected.numel() == 0:
            raise ValueError("env_ids must not be empty")
        if ((selected < 0) | (selected >= self.num_envs)).any():
            raise ValueError("env_ids contains an out-of-range environment")
        if selected.unique().numel() != selected.numel():
            raise ValueError("env_ids must not contain duplicates")
        allowed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        allowed[selected] = True
        return candidates[allowed[candidates % self.num_envs]]

    def num_candidates(
        self,
        horizon: int,
        *,
        policy_owned: bool = False,
        env_ids: Tensor | Sequence[int] | None = None,
    ) -> int:
        return self._filter_environments(
            self._candidates(horizon, policy_owned), env_ids,
        ).numel()

    def num_start_candidates(
        self,
        history_length: int = 1,
        *,
        env_ids: Tensor | Sequence[int] | None = None,
    ) -> int:
        if type(history_length) is not int or history_length <= 0:
            raise ValueError("history_length must be a positive integer")
        candidates = self._candidates(
            max(1, history_length - 1), True, history_length > 1,
        )
        return self._filter_environments(candidates, env_ids).numel()

    def _sample(
        self,
        batch_size: int,
        horizon: int,
        device: str | torch.device | None,
        generator: torch.Generator | None,
        policy_owned: bool,
        actor_history: bool = False,
        env_ids: Tensor | Sequence[int] | None = None,
    ) -> WindowBatch:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        candidates = self._filter_environments(
            self._candidates(horizon, policy_owned, actor_history), env_ids,
        )
        if candidates.numel() == 0:
            raise ValueError(
                f"No eligible replay windows for horizon={horizon}; "
                f"stored_steps={self._size_steps}, policy_owned={policy_owned}",
            )
        random_device = self.device if generator is None else generator.device
        choices = torch.randint(
            candidates.numel(), (batch_size,), generator=generator,
            device=random_device,
        ).to(self.device)
        selected = candidates[choices]
        starts = selected // self.num_envs
        environments = (selected % self.num_envs).unsqueeze(-1)
        oldest = (self._write_index - self._size_steps) % self.capacity_steps
        slots = (
            oldest + starts.unsqueeze(-1)
            + torch.arange(horizon, device=self.device)
        ) % self.capacity_steps
        data = self._storage
        frames = {}
        for name, output in (
            ("state", "states"), ("observation", "observations"),
            ("goal", "goals"), ("context", "contexts"),
            ("actor_observation", "actor_observations"),
            ("estimator_target", "estimator_targets"),
            ("estimator_valid", "estimator_valid"),
        ):
            frames[output] = torch.cat((
                data[f"start_{name}"][slots, environments],
                data[f"end_{name}"][slots[:, -1:], environments],
            ), dim=1)
        result = WindowBatch(**frames, **{
            name: data[name][slots, environments]
            for name in (
                "actions", "rewards", "terminated", "truncated",
                "valid", "policy_owned",
            )
        }, substeps=(data["substeps"][slots, environments]
                     if "substeps" in data else None))
        return result.to(self.device if device is None else device)

    def sample(
        self,
        batch_size: int,
        horizon: int,
        *,
        device: str | torch.device | None = None,
        generator: torch.Generator | None = None,
        env_ids: Tensor | Sequence[int] | None = None,
    ) -> WindowBatch:
        """Sample valid world-model windows, regardless of controller ownership."""
        return self._sample(
            batch_size, horizon, device, generator, False, env_ids=env_ids,
        )

    def sample_starts(
        self,
        batch_size: int,
        history_length: int = 1,
        *,
        device: str | torch.device | None = None,
        generator: torch.Generator | None = None,
        env_ids: Tensor | Sequence[int] | None = None,
    ) -> WindowBatch:
        """Sample real policy history ending at a valid imagination start.

        For history_length > 1, the learner starts at the window's final frame;
        every action in that history and its endpoint must belong to the policy.
        A one-frame history starts at the first frame of a one-step window.
        """
        if type(history_length) is not int or history_length <= 0:
            raise ValueError("history_length must be a positive integer")
        return self._sample(
            batch_size, max(1, history_length - 1), device, generator,
            True, history_length > 1, env_ids,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return a portable, detached snapshot without unused capacity slots."""
        return {
            "format_version": self.FORMAT_VERSION,
            "schema": self.schema.to_dict(),
            "schema_fingerprint": self.schema.fingerprint,
            "num_envs": self.num_envs,
            "capacity": self.capacity,
            "size_steps": self._size_steps,
            "write_index": self._write_index,
            "total_steps": self._total_steps,
            "sequence": self._sequence[:self._size_steps].detach().cpu().clone(),
            "storage": {
                name: value[:self._size_steps].detach().cpu().clone()
                for name, value in self._storage.items()
            },
        }

    @torch.no_grad()
    def load_state_dict(self, payload: dict[str, Any]) -> None:
        """Validate the complete snapshot before mutating this buffer."""
        version = payload.get("format_version")
        if version not in (1, self.FORMAT_VERSION):
            raise ValueError("Unsupported replay format version")
        if version == 1 and self.schema.observation_dim != 35:
            raise ValueError("Legacy replay format supports only the 35D interface")
        if payload.get("schema_fingerprint") != self.schema.fingerprint:
            raise ValueError("Replay schema fingerprint mismatch")
        if StateSchema.from_dict(payload["schema"]).fingerprint != self.schema.fingerprint:
            raise ValueError("Replay schema contents do not match its fingerprint")
        if (payload.get("num_envs"), payload.get("capacity")) != (
            self.num_envs, self.capacity,
        ):
            raise ValueError("Replay capacity or environment count mismatch")
        size, write, total = (
            payload.get(name) for name in ("size_steps", "write_index", "total_steps")
        )
        if any(type(value) is not int for value in (size, write, total)):
            raise ValueError("Replay counters must be integers")
        if (
            total < 0 or size != min(total, self.capacity_steps)
            or write != total % self.capacity_steps
        ):
            raise ValueError("Inconsistent replay counters")
        stored = payload.get("storage")
        if version == 1 and isinstance(stored, dict):
            auxiliary_fields = {
                f"{endpoint}_{name}"
                for endpoint in ("start", "end")
                for name in self._ENDPOINT_AUXILIARY_FIELDS
            }
            if set(stored) != set(self._storage) - auxiliary_fields:
                raise ValueError("Legacy replay storage fields do not match")
            stored = dict(stored)
            for endpoint in ("start", "end"):
                stored[f"{endpoint}_actor_observation"] = stored[f"{endpoint}_observation"]
                stored[f"{endpoint}_estimator_target"] = torch.zeros(
                    size, self.num_envs, 5, dtype=torch.float32,
                )
                stored[f"{endpoint}_estimator_valid"] = torch.zeros(
                    size, self.num_envs, dtype=torch.bool,
                )
        if not isinstance(stored, dict) or set(stored) != set(self._storage):
            raise ValueError("Replay storage fields do not match")
        for name, destination in self._storage.items():
            value = stored[name]
            if (
                not isinstance(value, Tensor)
                or value.shape != (size, *destination.shape[1:])
                or value.dtype != destination.dtype
            ):
                raise ValueError(f"Invalid replay tensor: {name}")
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite replay tensor: {name}")
        sequence = payload.get("sequence")
        if (
            not isinstance(sequence, Tensor) or sequence.shape != (size,)
            or sequence.dtype != torch.int64
        ):
            raise ValueError("Invalid replay sequence IDs")
        slots = (torch.arange(size) + write - size) % self.capacity_steps
        if not torch.equal(sequence.cpu()[slots], torch.arange(total - size, total)):
            raise ValueError("Replay sequence IDs disagree with its counters")
        for name, destination in self._storage.items():
            destination[:size].copy_(stored[name])
        self._sequence[:size].copy_(sequence)
        self._size_steps, self._write_index, self._total_steps = size, write, total
        self._candidate_cache.clear()

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), Path(path))

    def load(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        self.load_state_dict(payload)
