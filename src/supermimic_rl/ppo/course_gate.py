"""Performance-gated course coordinate for PPO.

The native course maps a real-transition coordinate to its stage, command and
terrain envelopes and the SCUT special-mode unlocks. Without gates that
coordinate equals real exposure, so a stage starts whether or not the policy
learned the previous one; PPO2000 entered later stages while still crouched.
In ``hold`` mode the coordinate still advances with real data inside a stage,
but waits just before a gated stage until the gate's criteria hold over a
rolling window of update records; a gate held longer than its timeout stops
training for diagnosis. One failing skill then freezes every later stage.

In ``report`` mode the coordinate never waits. Each reached gate is judged on
its window every update until it passes, and its waiting time is reported for
review instead of stopping training. Opened gates still drive the difficulty
that is tied to them (e.g. the jump height ladder), so a skill's difficulty
follows its own performance while the other skills keep their schedule.

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
    # Flat metric names may contain decimal points, e.g. rise_0.00-0.10.
    # Prefer the complete key before treating dots as nested-dictionary paths.
    value = record.get(path)
    if path not in record:
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
    """Entering ``stage`` (or a point inside it) requires every criterion over the last ``window`` updates.

    ``at_fraction`` > 0 places the gate inside ``stage`` at that fraction of
    its transitions. Holding the coordinate there also holds every schedule
    derived from it (e.g. the jump height cap and the assist level), so the
    difficulty inside a stage advances on performance, not on time.
    """

    stage: str
    criteria: tuple[GateCriterion, ...]
    window: int = 20
    timeout_updates: int = 2000
    at_fraction: float = 0.

    @property
    def key(self) -> str:
        return self.stage if not self.at_fraction else f"{self.stage}@{self.at_fraction:g}"

    def __post_init__(self):
        if type(self.at_fraction) not in (int, float) or not 0 <= self.at_fraction < 1:
            raise ValueError("at_fraction must lie in [0, 1)")
        criteria = tuple(item if isinstance(item, GateCriterion) else GateCriterion(**item) for item in self.criteria)
        object.__setattr__(self, "criteria", criteria)
        if not criteria:
            raise ValueError("A stage gate needs at least one criterion")
        for name in ("window", "timeout_updates"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + " must be a positive integer")


GATE_MODES = ("hold", "report")


class CourseGates:
    def __init__(self, gates, stage_starts: dict[str, int], cap: int | None = None, mode: str = "hold"):
        if mode not in GATE_MODES:
            raise ValueError("Course gate mode must be one of " + str(GATE_MODES))
        self.mode = mode
        gates = tuple(gate if isinstance(gate, StageGate) else StageGate(**gate) for gate in gates)
        unknown = [gate.stage for gate in gates if gate.stage not in stage_starts]
        if unknown:
            raise ValueError(f"Gated stages are not in the course: {unknown}")
        starts = sorted(stage_starts.values())
        self._ends = {name: next((value for value in starts if value > start), None)
                      for name, start in stage_starts.items()}
        if (len({gate.key for gate in gates}) != len(gates)
                or any(not gate.at_fraction and stage_starts[gate.stage] < 1 for gate in gates)
                or any(gate.at_fraction and self._ends[gate.stage] is None for gate in gates)):
            raise ValueError("Gates must be distinct, start after coordinate zero and lie inside a bounded stage")
        self.stage_starts, self.cap = dict(stage_starts), cap
        self.gates = tuple(sorted(gates, key=self._position))
        self._by_key = {gate.key: gate for gate in self.gates}
        self.coordinate, self.opened, self.held_updates = 0, [], 0
        self.waiting: dict[str, int] = {}   # report mode: updates since each reached gate first waited
        self.records = 0                     # advance() calls, to place gate openings in the history
        self.opened_at: dict[str, int] = {}  # report mode: record count when each gate opened
        self.history = deque(maxlen=max(gate.window for gate in self.gates) if self.gates else 1)

    def _position(self, gate: StageGate) -> int:
        start = self.stage_starts[gate.stage]
        if not gate.at_fraction:
            return start
        return start + int(gate.at_fraction * (self._ends[gate.stage] - start))

    def _next(self) -> StageGate | None:
        return next((gate for gate in self.gates if gate.key not in self.opened), None)

    def _limit(self, gate: StageGate) -> int:
        # The native course enters a stage at its first coordinate; wait one before.
        return self._position(gate) - 1

    def _judge(self, gate: StageGate) -> tuple[bool, dict]:
        window = list(self.history)[-gate.window:]
        values = {criterion.name: criterion.value(window) for criterion in gate.criteria}
        passed = (len(window) >= gate.window
                  and all(criterion.holds(values[criterion.name]) for criterion in gate.criteria))
        return passed, values

    def _advance_report(self, transitions: int) -> dict:
        status = {"coordinate_before": self.coordinate, "gates": {}}
        for gate in self.gates:
            if gate.key in self.opened or self.coordinate < self._limit(gate):
                continue
            # Inside one stage the ladder opens in order (e.g. jump heights), and
            # each step is judged only on a full window collected after the
            # previous step opened, i.e. at the difficulty it unlocked.
            earlier = [other.key for other in self.gates if other.stage == gate.stage
                       and self._position(other) < self._position(gate)]
            if not set(earlier) <= set(self.opened) or any(
                    self.records - self.opened_at.get(key, 0) < gate.window for key in earlier):
                continue
            passed, values = self._judge(gate)
            waited = self.waiting.get(gate.key, 0)
            status["gates"][gate.key] = {"values": values, "passed": passed, "waiting_updates": waited}
            if passed:
                self.opened.append(gate.key)
                self.opened_at[gate.key] = self.records
                self.waiting.pop(gate.key, None)
            else:
                self.waiting[gate.key] = waited + 1
        target = self.coordinate + transitions
        self.coordinate = target if self.cap is None else min(target, self.cap)
        status.update(coordinate=self.coordinate, opened=list(self.opened), waiting=dict(self.waiting))
        return status

    def advance(self, record: dict, transitions: int) -> dict:
        """Evaluate the waiting gate on records up to this one, then move the coordinate."""
        self.history.append(record)
        self.records += 1
        if self.mode == "report":
            return self._advance_report(transitions)
        status = {"coordinate_before": self.coordinate}
        gate = self._next()
        if gate is not None and self.coordinate >= self._limit(gate):
            passed, values = self._judge(gate)
            status.update(gate=gate.key, gate_values=values, gate_passed=passed)
            if passed:
                self.opened.append(gate.key)
                self.held_updates = 0
            else:
                self.held_updates += 1
                status["held_updates"] = self.held_updates
                if self.held_updates > gate.timeout_updates:
                    raise GateTimeout(f"Gate before {gate.key} held for {self.held_updates} updates: {values}")
        target = self.coordinate + transitions
        gate = self._next()
        if gate is not None:
            target = min(target, max(self.coordinate, self._limit(gate)))
        if self.cap is not None:
            target = min(target, self.cap)
        self.coordinate = target
        status.update(coordinate=self.coordinate, opened=list(self.opened),
                      waiting_for=None if gate is None else gate.key)
        return status

    def stage(self) -> str:
        """The course stage containing the current coordinate."""
        return max((start, name) for name, start in self.stage_starts.items() if start <= self.coordinate)[1]

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
        self.opened = [key for key in self.opened if self._position(self._by_key[key]) <= start]
        self.held_updates = 0
        self.waiting = {}
        self.opened_at = {key: 0 for key in self.opened}
        self.records = 0
        self.history.clear()
        return {**receipt, "opened": list(self.opened)}

    def state_dict(self) -> dict:
        return {"coordinate": self.coordinate, "opened": list(self.opened), "held_updates": self.held_updates,
                "waiting": dict(self.waiting), "history": list(self.history), "records": self.records,
                "opened_at": dict(self.opened_at), "mode": self.mode}

    def load_state_dict(self, state: dict, *, restart_hold: bool = False) -> None:
        """Restore the coordinate and opened gates; optionally restart the waiting gate's clock.

        Restarting also drops the old window, so the gate is judged only on
        updates collected under the resumed configuration.
        """
        if not set(state["opened"]) <= set(self._by_key):
            raise ValueError("Saved course gates differ from the configured gates")
        if state.get("mode", "hold") == "report" and self.mode == "hold":
            # A reported course may already be past closed gates; holding would freeze it there.
            raise ValueError("A report-mode course cannot resume in hold mode")
        self.coordinate, self.opened = int(state["coordinate"]), list(state["opened"])
        self.held_updates = 0 if restart_hold else int(state["held_updates"])
        self.waiting = {} if restart_hold else {key: int(value) for key, value in state.get("waiting", {}).items()}
        self.history.clear()
        if restart_hold:
            # The old window is dropped, so earlier openings count as complete windows ago.
            self.records, self.opened_at = 0, {key: -max(g.window for g in self.gates) for key in self.opened}
        else:
            self.history.extend(state["history"])
            self.records = int(state.get("records", len(state["history"])))
            self.opened_at = {key: int(value) for key, value in state.get("opened_at", {}).items()}
