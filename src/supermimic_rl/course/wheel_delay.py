"""Episode-local aggregate wheel I/O uncertainty, without an extra delay chain.

The pinned identification reports 3–6 ms variation between training directions.
Uniform, independent wheel draws are a training prior within those bounds, not
an empirical jitter distribution. Existing nominal lanes and course ramp apply.
"""
from __future__ import annotations

import torch

from .identified_execution import IdentifiedExecution, MODEL_PINS, WHEEL_AGGREGATE_DELAY_BOUNDS_S
from .randomization import CourseRandomization


class WheelDelayRandomization(CourseRandomization):
    """Add one independent reset RNG without perturbing existing DR draws."""

    STATE_KEY = "wheel_aggregate_delay_randomization"
    RNG_XOR = 32452843

    def __init__(self, num_envs, seed=0, device="cpu", config=None, *, execution):
        super().__init__(num_envs, seed, device, config)
        if (not isinstance(execution, IdentifiedExecution)
                or not execution.wheel_delay_randomization
                or execution.count != num_envs or execution.device != self.device):
            raise ValueError("Episode wheel delay requires the matching opted-in execution")
        self.execution = execution
        self.delay_generator = torch.Generator(device=self.device).manual_seed(seed ^ self.RNG_XOR)
        self.delay_reset_rows_consumed = 0
        self.nominal_delay_s = execution.wheel_aggregate_delay_s.new_tensor(
            [row["aggregate_delay_s"] for row in execution.wheel_rows])

    def reset(self, mask, update):
        difficulty = super().reset(mask, update)
        rows = mask.nonzero().flatten()
        low, high = WHEEL_AGGREGATE_DELAY_BOUNDS_S
        sample = low + (high - low) * torch.rand(
            len(rows), 2, generator=self.delay_generator, device=self.device,
            dtype=self.execution.dtype)
        scale = difficulty * (~self.nominal[rows, None])
        delay = self.nominal_delay_s + scale * (sample - self.nominal_delay_s)
        self.execution.set_wheel_aggregate_delay(rows, delay)
        self.delay_reset_rows_consumed += len(rows)
        return difficulty

    def context_parameters(self):
        return {**super().context_parameters(),
                "wheel_aggregate_delay_s": self.execution.wheel_aggregate_delay_s.clone()}

    def receipt(self):
        return {
            "format": "wheel_aggregate_delay_randomization_v1",
            "model_sha256": MODEL_PINS["wheel_model.json"],
            "source": "wheel_model.json uncertainty.delay: variation between training directions",
            "bounds_s": list(WHEEL_AGGREGATE_DELAY_BOUNDS_S),
            "nominal_s": [row["aggregate_delay_s"] for row in self.execution.wheel_rows],
            "nominal_fraction": self.config.nominal_fraction,
            "nominal_mask": "shared fixed startup lanes from CourseRandomization",
            "ramp": "existing course smoothstep; nominal + difficulty * (uniform draw - nominal)",
            "full_at_coordinate": self.config.full_at_coordinate,
            "scope": "independent per-wheel uniform model-parameter prior, held per episode; not tick jitter",
            "owner": "replace IdentifiedExecution aggregate delay; no extra USB or report-chain delay",
            "context_field": "requested.wheel_aggregate_delay_s",
            "rng": "independent seed xor 32452843; preserved by component state_dict",
            "world_model_v2": "unsupported fixed-delay imagination; SnapshotLayout rejects this context",
        }

    def state_dict(self):
        return {**super().state_dict(), self.STATE_KEY: {
            "format": "wheel_aggregate_delay_randomization_v1",
            "rng": self.delay_generator.get_state().clone(),
            "reset_rows_consumed": self.delay_reset_rows_consumed,
        }}

    def restore_after_physical_reset(self, state):
        delay = state.get(self.STATE_KEY) if isinstance(state, dict) else None
        if (not isinstance(delay, dict) or set(delay) != {"format", "rng", "reset_rows_consumed"}
                or delay["format"] != "wheel_aggregate_delay_randomization_v1"
                or type(delay["reset_rows_consumed"]) is not int or delay["reset_rows_consumed"] < 0):
            raise ValueError("Wheel delay checkpoint identity differs; explicit source migration is required")
        generator = torch.Generator(device=self.device)
        generator.set_state(delay["rng"].cpu())
        super().restore_after_physical_reset(state)
        self.delay_generator = generator
        self.delay_reset_rows_consumed = delay["reset_rows_consumed"]
