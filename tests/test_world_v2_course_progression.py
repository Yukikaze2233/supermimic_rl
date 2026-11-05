"""Sampling progression preserves data clocks, resume identity and skill difficulty."""
from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.world_model_v2.collection import Collector
from supermimic_rl.world_model_v2.course_progression import CourseProgression, CourseProgressionConfig
from supermimic_rl.world_model_v2.online import CHECKPOINT_FORMAT, OnlineConfig, OnlineTrainer
from supermimic_rl.world_model_v2.records import MODEL_CONTRACT
from supermimic_rl.course.metrics import install_jump_outcome_metrics
from support.world_v2.collection import Environment


def test_sampling_coordinate_is_exact_and_separate_from_actual_data():
    clock = CourseProgressionConfig(10, 17, 100, 10000000001)
    assert clock.coordinate(0) == clock.coordinate(10) == 100
    assert clock.coordinate(17) == clock.coordinate(100) == 10000000001
    assert clock.coordinate(13) == 100 + (10000000001 - 100) * 3 // 7
    values = [clock.coordinate(step) for step in range(25)]
    assert values == sorted(values)
    with pytest.raises(ValueError):
        clock.coordinate(-1)


def test_collector_uses_sampling_clock_without_inflating_transition_count(monkeypatch):
    monkeypatch.setattr('supermimic_rl.world_model_v2.collection.SnapshotLayout',
        lambda schema: SimpleNamespace(encode=lambda transition: {'frame': transition.start.actor_observation}))
    env = Environment()
    clock = CourseProgressionConfig(0, 10, 100, 1100)
    collector = Collector(env, lambda history: (torch.zeros(2, 6), torch.zeros(2, 5)),
        capacity_steps=8, course_coordinate=100, course_coordinate_at=clock.coordinate)
    report = collector.collect(5)
    assert [row['real_transitions'] for row in env.progress] == [100, 200, 300, 400, 500]
    assert report['transitions'] == 10 and report['stored_transitions'] == 10
    assert report['course_coordinate'] == 500 and report['fixed_course_coordinate'] is None
    collector.collect(1)
    assert env.progress[-1]['real_transitions'] == clock.coordinate(collector.steps - 1) == 600


def curriculum():
    result = SimpleNamespace(stage_names=('flat', 'jump', 'terrain', 'combined'),
        stage_budgets=(100, 100, 100, 100),
        ladder_gate_keys=lambda: ('jump@0.25', 'jump@0.5'), opened=())
    result.set_opened_gates = lambda keys: setattr(result, 'opened', tuple(keys))
    return result


def progression_config():
    criteria = [{'numerator': ['success'], 'denominator': ['ended'], 'min': .5, 'min_count': 2}]
    return CourseProgressionConfig(20, 100, 100, 400, tuple(
        {'stage': 'jump', 'at_fraction': fraction, 'window': 2, 'timeout_updates': 1, 'criteria': criteria}
        for fraction in (.25, .5)))


def test_failed_jump_does_not_hold_terrain_and_height_ladder_survives_resume(monkeypatch):
    counters = {'success': 0, 'ended': 0}
    monkeypatch.setattr('supermimic_rl.world_model_v2.course_progression.course_progress',
        lambda native, course: (dict(counters), {}))
    native, course = object(), curriculum()
    progress = CourseProgression(progression_config(), course)
    progress.apply_difficulty()
    for _ in range(5):
        before = progress.snapshot_counters(native)
        counters['ended'] += 1
        report = progress.observe(native, {'course_coordinate': 350}, before)
    assert report['stage'] == 'combined' and not course.opened
    assert report['waiting']['jump@0.25'] > 1  # The old timeout must not stop training.
    for _ in range(2):
        before = progress.snapshot_counters(native)
        counters['ended'] += 1
        counters['success'] += 1
        progress.observe(native, {'course_coordinate': 350}, before)
    assert course.opened == ('jump@0.25',)
    restored_course = curriculum()
    resumed = CourseProgression(progression_config(), restored_course)
    resumed.load_state_dict(deepcopy(progress.state_dict()))
    assert resumed.state_dict() == progress.state_dict()
    for _ in range(2):
        before = resumed.snapshot_counters(native)
        counters['ended'] += 1
        counters['success'] += 1
        resumed.observe(native, {'course_coordinate': 350}, before)
    assert restored_course.opened == ('jump@0.25', 'jump@0.5')


def resume_fixture():
    old = OnlineConfig(cycles=100, record_tasks=True, record_stair_source=True, record_jumps=True,
                       record_recovery=True, course_coordinate=100, policy_start_cycle=9)
    saved = {'format': CHECKPOINT_FORMAT, 'model_contract': MODEL_CONTRACT, 'schema_fingerprint': 'native',
             'config': asdict(old), 'cycle': 10, 'collection_steps': 20,
             'environment': {'curriculum': {'real_transitions': 100}},
             'teacher_source': None, 'stair_scenario': None, 'jump_scenario': None, 'recovery_scenario': None}
    trainer = object.__new__(OnlineTrainer)
    trainer.env = SimpleNamespace(schema=SimpleNamespace(fingerprint='native'))
    trainer.teacher_source = trainer.stair_scenario = trainer.jump_scenario = trainer.recovery_scenario = None
    trainer.config = replace(old, course_progression=progression_config())
    return saved, trainer


def test_activation_preserves_existing_training_recipe_and_absolute_warmup():
    saved, trainer = resume_fixture()
    original = deepcopy(saved)
    trainer._check_saved(saved)
    assert trainer.course_progression_activated
    assert saved == original
    assert trainer.config.policy_start_cycle < saved['cycle']
    assert trainer.config.learning == OnlineConfig.from_dict(saved['config']).learning
    assert OnlineConfig.from_dict(asdict(trainer.config)) == trainer.config


@pytest.mark.parametrize('damage', ['data_clock', 'course_clock', 'optimizer', 'replace_active', 'missing_ladder'])
def test_progression_cannot_hide_a_rewind_recipe_change_or_lost_difficulty(damage):
    saved, trainer = resume_fixture()
    if damage == 'data_clock':
        saved['collection_steps'] += 1
    elif damage == 'course_clock':
        saved['environment']['curriculum']['real_transitions'] += 1
    elif damage == 'optimizer':
        trainer.config = replace(trainer.config, learning=replace(trainer.config.learning, actor_learning_rate=.01))
    elif damage == 'replace_active':
        saved['config']['course_progression'] = asdict(replace(progression_config(), end_coordinate=399))
    else:
        with pytest.raises(ValueError, match='height ladder'):
            CourseProgression(replace(progression_config(), jump_height_gates=()), curriculum())
        return
    with pytest.raises(ValueError):
        trainer._check_saved(saved)


def test_progression_requires_all_recorded_task_producers():
    with pytest.raises(ValueError, match='full task producer'):
        OnlineConfig(record_tasks=True, record_stair_source=True, course_coordinate=100,
                     course_progression=progression_config())


def test_unassisted_outcomes_capture_substep_success_and_reset_without_actuation():
    tasks = SimpleNamespace(physical_id=torch.tensor([1, 2, 3, -1]),
        active_rise=torch.tensor([.05, .15, .25, 0.]), success_id=torch.full((4,), -1))

    def observe(succeed):
        if succeed:
            tasks.success_id[0] = tasks.physical_id[0]
            tasks.physical_id[:2] = -1  # One success and one timeout in the same tick.
        return 'native result'

    tasks.observe_physics = observe
    original_action = object()
    native = SimpleNamespace(_terrain_course=SimpleNamespace(tasks=tasks), _apply_action=original_action)
    install_jump_outcome_metrics(native)
    observer = tasks.observe_physics
    install_jump_outcome_metrics(native)
    assert observer is tasks.observe_physics and native._apply_action is original_action
    assert not hasattr(native, 'supermimic_jump_assist')
    rng = torch.get_rng_state().clone()
    assert tasks.observe_physics(True) == 'native result'
    # Simulate an environment reset erasing native success and request memory.
    tasks.success_id.fill_(-1)
    tasks.physical_id.fill_(-1)
    tasks.active_rise.zero_()
    tasks.observe_physics(False)
    tasks.observe_physics(False)  # No double counting idle ticks.
    values, _ = native.supermimic_jump_outcomes.metrics()
    assert values['course/jump_assist/unassisted_ended'] == 3
    assert values['course/jump_assist/unassisted_success'] == 1
    for name in ('0.00-0.10', '0.10-0.20', '0.20-0.30'):
        assert values['course/jump_assist/unassisted_ended_rise_' + name] == 1
    assert values['course/jump_assist/unassisted_success_rise_0.00-0.10'] == 1
    assert torch.equal(rng, torch.get_rng_state())
