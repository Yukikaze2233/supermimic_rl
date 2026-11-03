"""Collector ownership/history and deployed-estimator wiring, independent of Isaac."""
from types import SimpleNamespace

import torch

from supermimic_rl.world_model_v2.collection import Collector
from support.world_v2.collection import Environment


def test_collection_uses_decision_estimate_and_resets_history_at_takeover_and_episode(monkeypatch):
    monkeypatch.setattr('supermimic_rl.world_model_v2.collection.SnapshotLayout',
                        lambda schema: SimpleNamespace(encode=lambda transition: {'frame': transition.start.actor_observation}))
    env, seen = Environment(), []

    def decide(history):
        seen.append(history.clone())
        return torch.zeros(2, 6), torch.tensor([[1., 2., 3., 4., 5.]]).repeat(2, 1)

    collector = Collector(env, decide, capacity_steps=8, curriculum_transition_cap=0)
    report = collector.collect(5)
    assert report['transitions'] == 10 and report['terminations'] == 1
    assert len(collector.replay) == 10
    assert all(record['real_transitions'] == 0 for record in env.progress)
    assert all(torch.equal(value[0], torch.tensor([1., 2., 3., 0., 0., 0., 4., 5.])) for value in env.estimates)
    assert (seen[2][0] == 2).all()  # First policy-owned row, no script-era history.
    assert (seen[3][1] == 3).all()  # New physical episode, no pre-reset frame.
    torch.testing.assert_close(seen[4][0, :, 0], torch.tensor([2., 2., 2., 3., 4.]))


def test_cold_collector_cannot_join_seed_episode_even_when_native_ids_restart(monkeypatch):
    monkeypatch.setattr('supermimic_rl.world_model_v2.collection.SnapshotLayout',
        lambda schema: SimpleNamespace(encode=lambda transition: {'frame': transition.start.actor_observation}))
    decision = lambda history: (torch.zeros(2, 6), torch.zeros(2, 5))
    first = Collector(Environment(), decision, capacity_steps=16)
    first.collect(5)
    second = Collector(Environment(), decision, capacity_steps=16, replay=first.replay)
    assert second.episode_offset > int(first.replay.data['end_episode_id'].max())
    second.collect(1)
    assert (second.replay.data['segment_age'][5] == 0).all()
    assert (second.replay.data['start_episode_id'][5] > second.replay.data['end_episode_id'][4]).all()


def test_task_record_belongs_to_start_even_when_native_step_mutates_the_task(monkeypatch):
    monkeypatch.setattr('supermimic_rl.world_model_v2.collection.SnapshotLayout',
        lambda schema: SimpleNamespace(encode=lambda transition: {'frame': transition.start.actor_observation}))
    monkeypatch.setattr('supermimic_rl.world_model_v2.collection.MotionTask.capture',
                        lambda native: native.task_record.clone())
    env = Environment()
    native = SimpleNamespace(task_record=torch.zeros(2, 40))
    env.adapter = SimpleNamespace(env=native)
    original_step = env.step

    def step(action):
        native.task_record.add_(1.)
        return original_step(action)

    env.step = step
    collector = Collector(env, lambda history: (torch.zeros(2, 6), torch.zeros(2, 5)),
                          capacity_steps=4, record_tasks=True)
    collector.collect(2)
    assert (collector.replay.data['motion_task'][0] == 0).all()
    assert (collector.replay.data['motion_task'][1] == 1).all()


def test_teacher_mixing_is_episode_wise_and_uses_the_selected_deployed_estimator():
    from supermimic_rl.world_model_v2.mixing import MixedDecision
    class Actor:
        def __call__(self, history):
            return torch.ones(2, 6)
        def estimate(self, history):
            return torch.full((2, 5), 2.)
    env = Environment()
    teacher = lambda: (torch.full((2, 6), 3.), torch.full((2, 5), 4.))
    mix = MixedDecision(env, Actor(), teacher, seed=3, initial_fraction=1., hold_updates=1, fade_updates=1)
    history = torch.zeros(2, 5, 45)
    action, estimate = mix(history)
    assert (action == 3).all() and (estimate == 4).all()
    mix.actor_updates = 2
    action, _ = mix(history)
    assert (action == 3).all()  # The running teacher episode retains its controller.
    env.current.episode_id[0] += 1
    action, estimate = mix(history)
    assert (action[0] == 1).all() and (estimate[0] == 2).all()
    assert (action[1] == 3).all() and (estimate[1] == 4).all()


def test_stair_record_is_owned_before_step_and_retains_native_dtypes(monkeypatch):
    from supermimic_rl.world_model_v2.stair_task import StairTaskState

    monkeypatch.setattr('supermimic_rl.world_model_v2.collection.SnapshotLayout',
        lambda schema: SimpleNamespace(encode=lambda transition: {'frame': transition.start.actor_observation}))
    env = Environment()
    native = SimpleNamespace(stair=StairTaskState(
        torch.full((2, 4), 2**54 + 17, dtype=torch.int64),
        torch.full((2, 3), 1_000_000.01, dtype=torch.float64), torch.zeros(2, 6), torch.zeros(2, 16)))
    env.adapter = SimpleNamespace(env=native)
    monkeypatch.setattr('supermimic_rl.world_model_v2.collection.StairTaskState.capture',
        lambda native: StairTaskState.decode({k: v.clone() for k, v in native.stair.record().items()}))
    original_step = env.step

    def step(action):
        native.stair.ids.add_(1)
        native.stair.clock.add_(.02)
        return original_step(action)

    env.step = step
    collector = Collector(env, lambda history: (torch.zeros(2, 6), torch.zeros(2, 5)),
                          capacity_steps=4, record_stairs=True)
    collector.collect(2)
    replay = collector.replay
    assert replay.data['stair_ids'].dtype == torch.int64
    assert replay.data['stair_clock'].dtype == torch.float64
    assert (replay.data['stair_ids'][0] == 2**54 + 17).all()
    assert (replay.data['stair_ids'][1] == 2**54 + 18).all()
    assert (replay.data['stair_clock'][0] == 1_000_000.01).all()


def test_frequent_student_failures_do_not_turn_every_environment_into_a_teacher():
    from supermimic_rl.world_model_v2.mixing import MixedDecision
    class Actor:
        def __call__(self, history):
            return torch.zeros(2, 6)
        def estimate(self, history):
            return torch.zeros(2, 5)
    env = Environment()
    teacher = lambda: (torch.ones(2, 6), torch.ones(2, 5))
    mix = MixedDecision(env, Actor(), teacher, seed=3, initial_fraction=.5, hold_updates=10, fade_updates=10)
    for _ in range(100):
        mix(torch.zeros(2, 5, 45))
        assert int(mix.teacher_rows.sum()) == 1
        env.current.episode_id[~mix.teacher_rows] += 1  # Student fails every step; teacher survives.
    assert mix.teacher_steps == mix.student_steps == 100
    restored = MixedDecision(Environment(), Actor(), teacher, seed=71,
                             initial_fraction=.5, hold_updates=10, fade_updates=10)
    restored.load_state_dict(mix.state_dict())
    restored(torch.zeros(2, 5, 45))
    torch.testing.assert_close(restored.teacher_rows, mix.teacher_rows)
    mix.actor_updates = 20
    mix(torch.zeros(2, 5, 45))
    assert mix.teacher_rows.any()  # No mid-episode handoff when the fraction fades to zero.
    env.current.episode_id += 1
    mix(torch.zeros(2, 5, 45))
    assert not mix.teacher_rows.any()


def test_fixed_course_coordinate_does_not_claim_uncollected_exposure(monkeypatch):
    monkeypatch.setattr('supermimic_rl.world_model_v2.collection.SnapshotLayout',
        lambda schema: SimpleNamespace(encode=lambda transition: {'frame': transition.start.actor_observation}))
    env = Environment()
    collector = Collector(env, lambda history: (torch.zeros(2, 6), torch.zeros(2, 5)),
                          capacity_steps=4, course_coordinate=220200960, curriculum_transition_cap=0)
    report = collector.collect(3, actor_updates=17)
    assert env.progress == [{'real_transitions': 220200960, 'actor_updates': 17}] * 3
    assert report['transitions'] == report['stored_transitions'] == 6
    assert report['course_coordinate'] == report['fixed_course_coordinate'] == 220200960


def test_source_rng_and_estimate_are_captured_before_the_native_step(monkeypatch):
    from supermimic_rl.world_model_v2.stair_source import StairSourceState
    from supermimic_rl.world_model_v2.stair_task import StairTaskState

    monkeypatch.setattr('supermimic_rl.world_model_v2.collection.SnapshotLayout',
        lambda schema, **kwargs: SimpleNamespace(encode=lambda transition: {
            'frame': transition.start.actor_observation, 'terrain_tile': torch.zeros(2, 1),
            'task_family': torch.zeros(2, 1)}))
    monkeypatch.setattr('supermimic_rl.world_model_v2.collection.MotionTask.capture',
                        lambda native: torch.zeros(2, 40))
    monkeypatch.setattr('supermimic_rl.world_model_v2.collection.StairTaskState.capture',
        lambda native: StairTaskState(torch.zeros(2, 4, dtype=torch.int64),
            torch.zeros(2, 3, dtype=torch.float64), torch.zeros(2, 6), torch.zeros(2, 16)))
    monkeypatch.setattr('supermimic_rl.world_model_v2.collection.StairSourceState.capture',
        lambda native: StairSourceState(torch.zeros(2, 4, dtype=torch.int64),
            torch.zeros(2, 4, dtype=torch.bool), torch.zeros(2, 13), torch.zeros(2, 15), torch.ones(2, 1)))
    env, generator = Environment(), torch.Generator().manual_seed(51)
    env.adapter = SimpleNamespace(env=SimpleNamespace(
        _terrain_course=SimpleNamespace(tof_trigger=SimpleNamespace(generator=generator))))
    estimate = torch.full((2, 5), 4.)
    expected_states, recorded = [], []
    original_step = env.step

    def step(action):
        expected_states.append(generator.get_state().clone())
        torch.rand(14, generator=generator)
        estimate.add_(1.)  # Native buffers may be reused during physics.
        return original_step(action)

    env.step = step
    collector = Collector(env, lambda history: (torch.zeros(2, 6), estimate), capacity_steps=4,
                          record_terrain=True, record_tasks=True, record_stairs=True, record_stair_source=True)
    collector.collect(2, on_source_rng=lambda sequence, state: recorded.append((sequence, state)))
    assert [item[0] for item in recorded] == [0, 1]
    for (_, actual), expected in zip(recorded, expected_states):
        assert torch.equal(actual, expected)
    assert (collector.replay.data['task_estimate'][0] == 4.).all()
    assert (collector.replay.data['task_estimate'][1] == 5.).all()
    assert (env.current.estimator_target == 999.).all()  # Never used as a decision estimate.


def test_mixed_collection_reports_student_and_teacher_returns_separately(monkeypatch):
    monkeypatch.setattr('supermimic_rl.world_model_v2.collection.SnapshotLayout',
        lambda schema: SimpleNamespace(encode=lambda transition: {'frame': transition.start.actor_observation}))

    class Mixed:
        teacher_rows = torch.tensor([True, False])

        def __call__(self, history):
            return torch.zeros(2, 6), torch.zeros(2, 5)

    env = Environment()
    original = env.step
    env.step = lambda action: SimpleNamespace(**{**vars(original(action)), 'reward': torch.tensor([2., -1.])})
    report = Collector(env, Mixed(), capacity_steps=8).collect(4)
    assert report['teacher_reward_mean'] == 2. and report['student_reward_mean'] == -1.
    assert report['teacher_transitions'] == report['student_transitions'] == 4
    assert report['teacher_terminations'] == 0 and report['student_terminations'] == 1
    plain = Collector(Environment(), lambda history: (torch.zeros(2, 6), torch.zeros(2, 5)), capacity_steps=8)
    assert 'student_reward_mean' not in plain.collect(1)
