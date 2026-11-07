"""Student evaluation uses the same valid-sample accounting as PPO."""
from types import SimpleNamespace

import torch

from supermimic_rl.world_model_v2 import evaluation


def test_native_evaluation_excludes_invalid_physics_from_tracking_and_failures(monkeypatch):
    owner = SimpleNamespace(env=SimpleNamespace(_numerical_invalid=torch.tensor([False, True])))
    environment = SimpleNamespace(
        num_envs=2, device='cpu', policy_dt_s=.02, adapter=owner,
        current=SimpleNamespace(state=torch.zeros(2, 1)),
        schema=SimpleNamespace(state_fields=[SimpleNamespace(name='failure.failure_reasons', size=1)]))
    tracking = SimpleNamespace(actual=torch.tensor([[1., 2., 3.], [float('nan'), 100., 100.]]),
                               target=torch.zeros(2, 3))
    monkeypatch.setattr(evaluation, 'native_tracking_sample', lambda env: tracking)

    class Actor:
        def eval(self):
            return self

        def __call__(self, history):
            return torch.zeros(2, 6)

        def estimate(self, history):
            return torch.zeros(2, 5)

    class Collector:
        def __init__(self, env, decision, **kwargs):
            self.decision = decision
            self.layout = SimpleNamespace(physical=lambda state: SimpleNamespace(
                gravity_body=torch.tensor([[0., 0., -1.], [0., 0., -1.]])))

        def collect(self, steps, **kwargs):
            self.decision(torch.zeros(2, 5, 35))
            kwargs['on_transition'](SimpleNamespace(
                reward=torch.tensor([1., float('nan')]), terminated=torch.tensor([True, True]),
                truncated=torch.zeros(2, dtype=torch.bool), valid=torch.tensor([True, False]),
                end=SimpleNamespace(state=torch.tensor([[1.], [float('nan')]]))))

    monkeypatch.setattr(evaluation, 'Collector', Collector)
    report = evaluation.evaluate_native(environment, Actor(), 1)
    assert report['terminations'] == report['invalid_transitions'] == 1
    assert report['numerical_invalid_transitions'] == report['valid_tracking_samples'] == 1
    assert report['termination_masks'] == {'1': 1}
    assert report['valid_simulated_seconds'] == .02
    assert report['failures_per_valid_env_minute'] == 3000.
    assert report['rms/forward_error_m_s'] == 1.
    assert report['rms/yaw_rate_error_rad_s'] == 2.
    assert report['rms/height_error_m'] == 3.


def platform_evaluation_fixture(monkeypatch, *, enabled=True, fixed=True, summary=True):
    calls = {}
    platform = SimpleNamespace(platform_rows=torch.tensor([False, True]), landing=object())
    platform.recording_options = lambda: {'record_platform': True, 'platform_landing': platform.landing}
    if summary:
        platform.summary = lambda: {'scope': 'actual final state only', 'phase': [0, 3], 'success_rate': None}
    native = SimpleNamespace()
    if enabled:
        native.supermimic_platform_course = platform
    current = SimpleNamespace(state=torch.zeros(2, 1))
    original_suite = {'jump_diagnostics': {'requests': [{'env_id': 1}], 'totals': {'success': 0}},
                      'success_semantics': 'Native pulses', 'buckets': {'stand': {'samples': 1}}}
    owner = SimpleNamespace(env=native, current=current)
    if fixed:
        owner._evaluation_configuration = object()
        owner.evaluation_begin = lambda schema: calls.update(begin=True)
        owner.evaluation_step = lambda *args: calls.update(measured=True)
        owner.evaluation_end = lambda: calls.update(end=True)
        owner.evaluation_summary = lambda: original_suite
    env = SimpleNamespace(num_envs=2, device='cpu', policy_dt_s=.02, adapter=owner, current=current,
        _pack=lambda value: value, schema=SimpleNamespace(state_fields=[]))
    monkeypatch.setattr(evaluation, 'native_tracking_sample', lambda _: SimpleNamespace(
        actual=torch.zeros(2, 3), target=torch.zeros(2, 3)))
    monkeypatch.setattr(evaluation, 'terminal_failure_masks', lambda *args: {})
    monkeypatch.setattr('supermimic_rl.world_model_v2.stair_source.StairScenario.capture',
        lambda _: SimpleNamespace(to_dict=lambda: {'scenario': 'actual'}))

    def save(collector, path, report, **kwargs):
        calls['saved_report'], calls['saved_source'] = report, kwargs
        return {'path': str(path)}

    monkeypatch.setattr('supermimic_rl.world_model_v2.dataset.save_native_dataset', save)

    class Collector:
        def __init__(self, environment, decision, **kwargs):
            calls['collector_options'] = kwargs
            self.decision = decision
            self.layout = SimpleNamespace(physical=lambda _: SimpleNamespace(
                gravity_body=torch.tensor([[0., 0., -1.]] * 2)))

        def collect(self, steps, **kwargs):
            self.decision(torch.zeros(2, 5, 35))
            kwargs['on_transition'](SimpleNamespace(reward=torch.zeros(2), valid=torch.ones(2, dtype=torch.bool),
                terminated=torch.zeros(2, dtype=torch.bool), truncated=torch.zeros(2, dtype=torch.bool),
                end=SimpleNamespace(state=torch.zeros(2, 1))))

    monkeypatch.setattr(evaluation, 'Collector', Collector)

    class Actor:
        def eval(self):
            pass

        def __call__(self, frames):
            return torch.zeros(2, 6)

        def estimate(self, frames):
            return torch.zeros(2, 5)

    return env, Actor(), platform, calls, original_suite


def test_platform_evaluation_records_single_owner_and_labels_flat_diagnostics(monkeypatch, tmp_path):
    env, actor, platform, calls, original = platform_evaluation_fixture(monkeypatch)
    report = evaluation.evaluate_native(env, actor, 1, record_dataset=tmp_path / 'dataset.pt')
    assert calls['collector_options']['record_platform'] is True
    assert calls['collector_options']['platform_landing'] is platform.landing
    assert calls['collector_options']['record_jumps'] and calls['collector_options']['record_recovery']
    assert calls['collector_options']['advance_training_progress'] is False
    assert report['platform_diagnostics'] == platform.summary()
    assert report['platform_diagnostics']['success_rate'] is None
    assert report['platform_evaluation_scope']['planned_env_ids'] == [1]
    suite = report['task_suite']
    assert 'jump_diagnostics' not in suite
    assert 'jump_diagnostics' in original  # Do not mutate the native diagnostic owner.
    flat = suite['flat_anchor_jump_diagnostics']
    assert flat['requests'] == original['jump_diagnostics']['requests']
    assert flat['totals'] == original['jump_diagnostics']['totals']
    assert flat['platform_interpretation']['includes_platform_rows'] is True
    assert calls['saved_report']['platform_diagnostics'] == report['platform_diagnostics']
    assert calls['begin'] and calls['end'] and calls['measured']


def test_platform_evaluation_without_dataset_does_not_record_ticks(monkeypatch):
    env, actor, _, calls, _ = platform_evaluation_fixture(monkeypatch, fixed=False, summary=False)
    report = evaluation.evaluate_native(env, actor, 1)
    assert 'record_platform' not in calls['collector_options']
    assert 'platform_landing' not in calls['collector_options']
    assert report['platform_diagnostics']['status'] == 'unknown'
    assert report['platform_evaluation_scope']['recorded_physics_ticks'] is False
    assert 'task_suite' not in report


def test_legacy_evaluation_retains_original_summary_and_record_options(monkeypatch, tmp_path):
    env, actor, _, calls, original = platform_evaluation_fixture(monkeypatch, enabled=False)
    report = evaluation.evaluate_native(env, actor, 1, record_dataset=tmp_path / 'dataset.pt')
    assert report['task_suite'] is original
    assert 'platform_diagnostics' not in report and 'platform_evaluation_scope' not in report
    assert 'record_platform' not in calls['collector_options']
    assert 'platform_landing' not in calls['collector_options']
    assert calls['collector_options']['record_jumps'] is True


def test_mixed_platform_rows_follow_begin_reset_but_do_not_claim_full_run_membership(monkeypatch):
    env, actor, platform, calls, original = platform_evaluation_fixture(monkeypatch)
    platform.plan = SimpleNamespace(sampler=object())

    def begin(schema):
        calls['begin'] = True
        platform.platform_rows = torch.tensor([True, False])

    def another_episode(*args):
        # A later reset changes the same physical row's task assignment.
        platform.platform_rows = torch.tensor([False, True])

    env.adapter.evaluation_begin = begin
    env.adapter.evaluation_step = another_episode
    report = evaluation.evaluate_native(env, actor, 1)
    scope = report['platform_evaluation_scope']
    assert scope['planned_env_ids'] == [0]
    assert 'initial episode' in scope['planned_env_ids_scope']
    assert platform.platform_rows.tolist() == [False, True]
    interpretation = report['task_suite']['flat_anchor_jump_diagnostics']['platform_interpretation']
    assert interpretation['planned_env_ids'] == [0]
    assert interpretation['includes_platform_rows'] is None
    assert interpretation['initial_includes_platform_rows'] is True
    assert 'not a full-run' in interpretation['planned_env_ids_scope']
    assert report['task_suite']['flat_anchor_jump_diagnostics']['totals'] == original['jump_diagnostics']['totals']
