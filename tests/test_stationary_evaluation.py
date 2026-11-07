"""Zero-command drift must not confuse oscillation, driving, tasks or resets."""
import pytest
import torch

from supermimic_rl.contracts import Field, Snapshot, StateSchema, Transition
from supermimic_rl.course.stationary_evaluation import StationaryEvaluation
from supermimic_rl.evaluation import NativeTrackingSample


SCHEMA = StateSchema(tuple(Field(name, (3,) if i == 0 else ())
                           for i, name in enumerate(StationaryEvaluation.FIELDS)), ())


def snapshot(xy, *, episode=0, epoch=0, owned=True, mode=0, recovery=0, gimbal=0, physical=0, protocol=0):
    xy = torch.as_tensor(xy, dtype=torch.float32).reshape(-1, 2)
    n = len(xy)
    state = torch.cat((xy, torch.zeros(n, 1), torch.tensor([mode, recovery, gimbal, physical, protocol])
                      .float().repeat(n, 1)), -1)
    return Snapshot(state, torch.zeros(n, 45), torch.zeros(n, 16), torch.empty(n, 0),
                    torch.full((n,), episode, dtype=torch.long), torch.full((n,), epoch, dtype=torch.long),
                    torch.full((n,), owned))


def observe(evidence, start, end, *, command=(0., 0., 0., .32), actual=(.1, .2, .32), valid=True,
            terminated=False, truncated=False):
    n = len(start.state)
    tracking = NativeTrackingSample(torch.tensor(actual).repeat(n, 1), torch.zeros(n, 3), course={
        'stationary_admitted_command': torch.tensor(command).repeat(n, 1)})
    transition = Transition(start, torch.zeros(n, 6), end, torch.zeros(n), torch.full((n,), terminated),
                            torch.full((n,), truncated), torch.full((n,), valid))
    evidence.observe_transition(transition, tracking, transition.valid)


def test_oscillation_path_and_mean_speed_are_not_net_drift():
    evidence = StationaryEvaluation(SCHEMA, ['stand'], normal_mode=0)
    observe(evidence, snapshot([0, 0]), snapshot([.01, 0]))
    observe(evidence, snapshot([.01, 0]), snapshot([0, 0]))
    record, = evidence.summary()['segments']
    assert record['net_displacement_m'] == 0.
    assert record['path_m'] == pytest.approx(.02)
    assert record['duration_s'] == pytest.approx(.04)
    assert record['mean_abs_forward_m_s'] == pytest.approx(.1)
    assert record['mean_abs_yaw_rad_s'] == pytest.approx(.2)
    assert record['end_reason'] == 'active_at_evaluation_end'
    assert evidence.summary() == evidence.summary()  # Reading does not close the segment.


@pytest.mark.parametrize('command', [(2.7, 0, 0, .32), (0, .1, 0, .32), (0, 0, .1, .32)])
def test_only_the_explicit_zero_part_of_braking_counts(command):
    evidence = StationaryEvaluation(SCHEMA, ['braking'], normal_mode=0)
    observe(evidence, snapshot([0, 0]), snapshot([1, 0]), command=command)
    observe(evidence, snapshot([1, 0]), snapshot([1.01, 0]))
    record, = evidence.summary()['segments']
    assert record['samples'] == 1 and record['path_m'] == pytest.approx(.01)
    assert evidence.summary()['excluded_intervals'] == {'nonzero_command': 1}


@pytest.mark.parametrize('field,reason', [('mode', 'other_mode'), ('recovery', 'recovery'),
    ('gimbal', 'gimbal_translation'), ('physical', 'active_jump'), ('protocol', 'active_jump')])
@pytest.mark.parametrize('endpoint', ['start', 'end'])
def test_special_tasks_at_either_endpoint_do_not_count(field, reason, endpoint):
    evidence = StationaryEvaluation(SCHEMA, ['stand'], normal_mode=0)
    start, end = snapshot([0, 0]), snapshot([.1, 0])
    changed = snapshot([0, 0] if endpoint == 'start' else [.1, 0], **{field: 1})
    observe(evidence, changed if endpoint == 'start' else start, changed if endpoint == 'end' else end)
    assert not evidence.summary()['segments']
    assert evidence.summary()['excluded_intervals'] == {reason: 1}


def test_terminal_pre_reset_motion_is_retained_and_new_episode_is_separate():
    evidence = StationaryEvaluation(SCHEMA, ['stand'], normal_mode=0)
    observe(evidence, snapshot([0, 0]), snapshot([.02, 0]), terminated=True)
    observe(evidence, snapshot([30, 20], episode=1), snapshot([30.01, 20], episode=1))
    first, second = evidence.summary()['segments']
    assert first['end_reason'] == 'terminated' and first['path_m'] == pytest.approx(.02)
    assert second['episode_id'] == 1 and second['path_m'] == pytest.approx(.01, abs=1e-6)
    assert second['net_displacement_m'] < .011


def test_invalid_interval_and_control_handoff_never_bridge_segments():
    evidence = StationaryEvaluation(SCHEMA, ['stand'], normal_mode=0)
    observe(evidence, snapshot([0, 0]), snapshot([.01, 0]))
    observe(evidence, snapshot([.01, 0]), snapshot([10, 0]), valid=False)
    observe(evidence, snapshot([10, 0]), snapshot([10.01, 0]))
    observe(evidence, snapshot([10.01, 0]), snapshot([11, 0], epoch=1))
    observe(evidence, snapshot([11, 0], epoch=1), snapshot([11.02, 0], epoch=1))
    records = evidence.summary()['segments']
    assert len(records) == 3
    assert [record['end_reason'] for record in records] == ['invalid_interval', 'control_boundary', 'active_at_evaluation_end']
    assert sum(record['path_m'] for record in records) == pytest.approx(.04, abs=1e-6)


def test_two_rows_have_independent_segments_and_missing_positions_are_not_invented():
    evidence = StationaryEvaluation(SCHEMA, ['stand', 'gimbal_sweep'], normal_mode=0)
    start, end = snapshot([[0, 0], [100, 100]]), snapshot([[.01, 0], [100, 100.02]])
    observe(evidence, start, end)
    end = snapshot([[.02, 0], [100, float('nan')]])
    observe(evidence, snapshot([[.01, 0], [100, 100.02]]), end)
    records = {record['env_id']: record for record in evidence.summary()['segments']}
    assert records[0]['samples'] == 2 and records[1]['samples'] == 1
    assert records[0]['net_displacement_m'] == pytest.approx(.02)
    assert records[1]['end_reason'] == 'nonfinite_measurement'


def test_unowned_or_cross_episode_interval_is_excluded():
    for end, reason in [(snapshot([10, 0], episode=1), 'episode_boundary'),
                        (snapshot([10, 0], owned=False), 'unowned')]:
        evidence = StationaryEvaluation(SCHEMA, ['stand'], normal_mode=0)
        observe(evidence, snapshot([0, 0]), end)
        assert evidence.summary()['excluded_intervals'] == {reason: 1}
        assert not evidence.summary()['segments']


def test_course_report_copies_admitted_command_before_action_without_terrain_pollution(monkeypatch):
    from types import SimpleNamespace
    from supermimic_rl.course import evaluation

    class Plan:
        def __init__(self, adapter, config):
            self.names = self.row_names = ('stand',)
            self.command_source = torch.ones(1, dtype=torch.long)

        def install(self):
            pass

        def close(self):
            pass

        def description(self):
            return {}

    counters = {'physical_request_outcomes': {}, 'request_end_counts': {},
                'recovery': {'counts': {}}, 'authored_stair_completion': {}}
    router = SimpleNamespace(tile=torch.zeros(1, dtype=torch.long), step_table=torch.zeros(1),
        slope_table=torch.zeros(1), roughness_table=torch.zeros(1))
    command = torch.tensor([[0., 0., 0., .32]])
    native = SimpleNamespace(_terrain_course=SimpleNamespace(router=router, metrics=lambda: counters),
        _commands=lambda: SimpleNamespace(admitted_body_command=command))
    adapter = SimpleNamespace(env=native, reset=lambda: None)
    monkeypatch.setattr(evaluation, 'FixedCourseScenarios', Plan)
    evaluator = evaluation.CourseEvaluation(adapter)
    evaluator.begin(SCHEMA)
    try:
        context = evaluator.sample()
        command[0, 0] = 2.7  # A later command draw must not reclassify the preceding interval.
        start, end = snapshot([0, 0]), snapshot([.01, 0])
        transition = Transition(start, torch.zeros(1, 6), end, torch.zeros(1), torch.zeros(1, dtype=torch.bool),
                                torch.zeros(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool))
        evaluator.step(transition, NativeTrackingSample(torch.tensor([[.1, .2, .32]]),
            torch.tensor([[0., 0., .32]]), course=context))
        report = evaluator.summary()
        assert report['stationary_diagnostics']['segments'][0]['path_m'] == pytest.approx(.01)
        assert 'stationary_admitted_command' not in report['buckets']['stand']['routed_terrain_ranges']
    finally:
        evaluator.close()
