"""Advance native sampling domains without treating exposure as competence."""
from dataclasses import dataclass, field

from ..course.metrics import course_progress
from ..ppo.course_gate import CourseGates, StageGate


@dataclass(frozen=True)
class CourseProgressionConfig:
    start_collection_step: int
    end_collection_step: int
    start_coordinate: int
    end_coordinate: int
    jump_height_gates: tuple[dict, ...] = field(default_factory=tuple)
    jump_height_progression: str = 'performance'

    def __post_init__(self):
        for name in ('start_collection_step', 'end_collection_step', 'start_coordinate', 'end_coordinate'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError('Course progression needs nonnegative integer clocks: ' + name)
        if (self.end_collection_step <= self.start_collection_step
                or self.end_coordinate <= self.start_coordinate):
            raise ValueError('Course progression requires increasing collection and course intervals')
        gates = tuple(dict(gate) for gate in self.jump_height_gates)
        for gate in gates:
            StageGate(**gate)
        object.__setattr__(self, 'jump_height_gates', gates)
        if self.jump_height_progression not in ('performance', 'schedule'):
            raise ValueError('jump_height_progression must be performance or schedule')

    def coordinate(self, collection_step: int) -> int:
        """Map a collected vector-step clock to a sampling coordinate, exactly."""
        if type(collection_step) is not int or collection_step < 0:
            raise ValueError('Collection step must be a nonnegative integer')
        span = self.end_collection_step - self.start_collection_step
        elapsed = min(max(collection_step - self.start_collection_step, 0), span)
        return self.start_coordinate + (self.end_coordinate - self.start_coordinate) * elapsed // span

    def validate_activation(self, saved, fixed_coordinate):
        """Start at this exact full checkpoint, without rewinding or new warmup."""
        current = saved['environment']['curriculum']['real_transitions']
        if (self.start_collection_step != saved['collection_steps']
                or self.start_coordinate != current or current != fixed_coordinate):
            raise ValueError('Course progression must start at the saved collection and sampling clocks')


class CourseProgression:
    """Sampling clock plus the existing report-only jump difficulty ladder.

    Native task records describe each issued command/terrain. The clock only
    affects future sampling; recorded histories and imagined task state retain
    their actual conditions. Skill counters always determine gate verdicts;
    the explicit progression option selects whether those verdicts cap sampling.
    """

    def __init__(self, config: CourseProgressionConfig, curriculum):
        self.config, self.curriculum = config, curriculum
        starts, total = {}, 0
        for name, budget in zip(curriculum.stage_names, curriculum.stage_budgets):
            starts[name], total = total, total + budget
        if config.end_coordinate > total:
            raise ValueError('Course endpoint exceeds the pinned native schedule')
        gates = tuple(StageGate(**gate) for gate in config.jump_height_gates)
        if {gate.key for gate in gates} != set(curriculum.ladder_gate_keys()):
            raise ValueError('Progression must contain exactly the native jump height ladder')
        self.gates = CourseGates(gates, starts, mode='report')
        self.gates.coordinate = config.start_coordinate

    def apply_difficulty(self):
        keys = self.gates.opened if self.config.jump_height_progression == 'performance' else None
        self.curriculum.set_opened_gates(keys)

    def snapshot_counters(self, native):
        return course_progress(native, self.curriculum)[0]

    def observe(self, native, collection, previous_counters):
        counters, gauges = course_progress(native, self.curriculum)
        delta = {name: value - previous_counters[name] for name, value in counters.items()}
        if any(value < 0 for value in delta.values()):
            raise ValueError('Native skill counters reset inside a collection')
        # The coordinate is supplied by the independent sampling clock, so a
        # gate can report waiting without holding any other task family back.
        self.gates.coordinate = collection['course_coordinate']
        record = {**collection, **delta, **gauges}
        result = self.gates.advance(record, 0)
        self.apply_difficulty()
        result.update(stage=self.gates.stage(), counters=delta, gauges=gauges,
                      jump_height_progression=self.config.jump_height_progression,
                      semantics='sampling schedule, not real transitions or demonstrated skills')
        return result

    def state_dict(self):
        return self.gates.state_dict()

    def load_state_dict(self, state):
        self.gates.load_state_dict(state)
        self.apply_difficulty()
