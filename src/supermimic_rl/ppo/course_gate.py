"""Performance-gated course coordinate for PPO.

The native course maps a real-transition coordinate to its stage, command and
terrain envelopes and the SCUT special-mode unlocks. Without gates that
coordinate equals real exposure, so a stage starts whether or not the policy
learned the previous one; PPO2000 entered later stages while still crouched.
Here the coordinate still advances with real data inside a stage, but waits
just before a gated stage until the gate's criteria hold over a rolling window
of update records. A gate held longer than its timeout stops training for
diagnosis instead of consuming more data.

Criteria read the trainer's update record: either the rolling mean of one
metric (dotted paths reach nested dictionaries), or a rolling ratio of summed
per-update counters such as native verified jump successes per finished jump
request. The criterion values are candidates chosen per run, not hardware data.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math


class GateTimeout(RuntimeError):
    """A course gate stayed closed longer than its configured update budget."""


def _lookup(record: dict, path: str):
    value = record
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


@dataclass(frozen=True)
class GateCriterion:
    metric: str | None = None
    numerator: tuple[str, ...] = ()
    denominator: tuple[str, ...] = ()
    min: float | None = None
    max: float | None = None
    # Smallest summed denominator for which a ratio is defined.
    min_count: float = 1.

    def __post_init__(self):
        object.__setattr__(self, "numerator", tuple(self.numerator))
        object.__setattr__(self, "denominator", tuple(self.denominator))
        if (self.metric is None) == (not self.numerator or not self.denominator):
            raise ValueError("A criterion reads either one metric or a numerator/denominator ratio")
        if self.min is None and self.max is None:
            raise ValueError("A criterion needs a min or max")
        if not math.isfinite(self.min_count) or self.min_count <= 0:
            raise ValueError("min_count must be finite and positive")

    @property
    def name(self) -> str:
        return self.metric or "+".join(self.numerator) + "/" + "+".join(self.denominator)

    def value(self, records) -> float | None:
        if self.metric is not None:
            values = [value for value in (_lookup(record, self.metric) for record in records) if value is not None]
            return sum(values) / len(values) if values else None
        numerator = sum(_lookup(record, key) or 0. for record in records for key in self.numerator)
        denominator = sum(_lookup(record, key) or 0. for record in records for key in self.denominator)
        return numerator / denominator if denominator >= self.min_count else None

    def holds(self, value: float | None) -> bool:
        return (value is not None and (self.min is None or value >= self.min)
                and (self.max is None or value <= self.max))


@dataclass(frozen=True)
class StageGate:
    """Entering ``stage`` requires every criterion over the last ``window`` updates."""

    stage: str
    criteria: tuple[GateCriterion, ...]
    window: int = 20
    timeout_updates: int = 2000

    def __post_init__(self):
        criteria = tuple(item if isinstance(item, GateCriterion) else GateCriterion(**item) for item in self.criteria)
        object.__setattr__(self, "criteria", criteria)
        if not criteria:
            raise ValueError("A stage gate needs at least one criterion")
        for name in ("window", "timeout_updates"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + " must be a positive integer")


class CourseGates:
    def __init__(self, gates, stage_starts: dict[str, int], cap: int | None = None):
        gates = tuple(gate if isinstance(gate, StageGate) else StageGate(**gate) for gate in gates)
        unknown = [gate.stage for gate in gates if gate.stage not in stage_starts]
        if unknown:
            raise ValueError(f"Gated stages are not in the course: {unknown}")
        if len({gate.stage for gate in gates}) != len(gates) or any(stage_starts[gate.stage] < 1 for gate in gates):
            raise ValueError("Gates must name distinct stages that start after coordinate zero")
        self.gates = tuple(sorted(gates, key=lambda gate: stage_starts[gate.stage]))
        self.stage_starts, self.cap = dict(stage_starts), cap
        self.coordinate, self.opened, self.held_updates = 0, [], 0
        self.history = deque(maxlen=max(gate.window for gate in self.gates) if self.gates else 1)

    def _next(self) -> StageGate | None:
        return next((gate for gate in self.gates if gate.stage not in self.opened), None)

    def _limit(self, gate: StageGate) -> int:
        # The native course enters a stage at its first coordinate; wait one before.
        return self.stage_starts[gate.stage] - 1

    def advance(self, record: dict, transitions: int) -> dict:
        """Evaluate the waiting gate on records up to this one, then move the coordinate."""
        self.history.append(record)
        status = {"coordinate_before": self.coordinate}
        gate = self._next()
        if gate is not None and self.coordinate >= self._limit(gate):
            window = list(self.history)[-gate.window:]
            values = {criterion.name: criterion.value(window) for criterion in gate.criteria}
            passed = (len(window) >= gate.window
                      and all(criterion.holds(values[criterion.name]) for criterion in gate.criteria))
            status.update(gate=gate.stage, gate_values=values, gate_passed=passed)
            if passed:
                self.opened.append(gate.stage)
                self.held_updates = 0
            else:
                self.held_updates += 1
                status["held_updates"] = self.held_updates
                if self.held_updates > gate.timeout_updates:
                    raise GateTimeout(f"Gate before {gate.stage} held for {self.held_updates} updates: {values}")
        target = self.coordinate + transitions
        gate = self._next()
        if gate is not None:
            target = min(target, max(self.coordinate, self._limit(gate)))
        if self.cap is not None:
            target = min(target, self.cap)
        self.coordinate = target
        status.update(coordinate=self.coordinate, opened=list(self.opened),
                      waiting_for=None if gate is None else gate.stage)
        return status

    def rewind(self, stage: str) -> dict:
        """Return the coordinate to the first transition of ``stage`` to repeat it.

        Gates of later stages close again; the waiting clock and window
        restart, so admission is judged only on the repeated stage.
        """
        if stage not in self.stage_starts:
            raise ValueError("Rewind stage is not in the course: " + stage)
        start = self.stage_starts[stage]
        if start > self.coordinate:
            raise ValueError("A rewind cannot move the course forward")
        receipt = {"stage": stage, "coordinate_before": self.coordinate, "coordinate": start,
                   "opened_before": list(self.opened), "held_updates_before": self.held_updates}
        self.coordinate = start
        self.opened = [name for name in self.opened if self.stage_starts[name] <= start]
        self.held_updates = 0
        self.history.clear()
        return {**receipt, "opened": list(self.opened)}

    def state_dict(self) -> dict:
        return {"coordinate": self.coordinate, "opened": list(self.opened), "held_updates": self.held_updates,
                "history": list(self.history)}

    def load_state_dict(self, state: dict, *, restart_hold: bool = False) -> None:
        """Restore the coordinate and opened gates; optionally restart the waiting gate's clock.

        Restarting also drops the old window, so the gate is judged only on
        updates collected under the resumed configuration.
        """
        if not set(state["opened"]) <= {gate.stage for gate in self.gates}:
            raise ValueError("Saved course gates differ from the configured gates")
        self.coordinate, self.opened = int(state["coordinate"]), list(state["opened"])
        self.held_updates = 0 if restart_hold else int(state["held_updates"])
        self.history.clear()
        if not restart_hold:
            self.history.extend(state["history"])
