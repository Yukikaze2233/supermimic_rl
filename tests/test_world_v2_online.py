"""A process handoff must finish the current cycle and save complete recovery state."""
import json
from types import SimpleNamespace

import torch
import pytest

from supermimic_rl.world_model_v2.online import OnlineConfig, OnlineTrainer


@pytest.mark.parametrize('world_only', [False, True])
@pytest.mark.parametrize('audit_fails', [False, True])
def test_checkpoint_request_during_model_fit_finishes_policy_updates_before_full_save(
        tmp_path, monkeypatch, world_only, audit_fails):
    requested, events = [False], []
    learner = SimpleNamespace(updates={'actor': 1}, skipped={'actor': 0}, fit=None)

    def fit_model(windows):
        events.append('model')
        requested[0] = True
        return {'model/updated': 1.}

    def update_policy(start, frames, **kwargs):
        events.append('policy')
        learner.updates['actor'] += 1
        return {'actor/updated': 1.}

    def fit_estimator(start, frames):
        events.append('estimator')
        return {'estimator/updated': 1.}

    learner.fit_model, learner.update_policy, learner.fit_estimator = fit_model, update_policy, fit_estimator
    learner.imagination = SimpleNamespace(termination=SimpleNamespace(limits=SimpleNamespace(forbidden_body_load_n=1.)))

    class Replay:
        def __len__(self):
            return 24
        def sample_windows(self, *args, **kwargs):
            return {}

    trainer = object.__new__(OnlineTrainer)
    trainer.guard = None
    trainer.config = OnlineConfig(cycles=9, model_updates=2, actor_updates=3,
        record_tasks=world_only, record_stair_source=world_only, course_coordinate=0 if world_only else None,
        policy_start_cycle=3 if world_only else 0)
    trainer.cycle, trainer.output = 1, tmp_path
    trainer.env = SimpleNamespace(num_envs=2)
    trainer.device = torch.device('cpu')
    trainer.generator = trainer.validation_generator = None
    trainer.validation_ids = trainer.train_ids = [0]
    trainer.learner = learner
    trainer.mixing = SimpleNamespace(actor_updates=1, fraction=.5, teacher_steps=0, student_steps=0)
    trainer.collector = SimpleNamespace(steps=12, episode_offset=1, replay=Replay())

    def collect(*args, **kwargs):
        events.append('collect')
        trainer.collector.steps += 24
        return {'terminations': 0}

    trainer.collector.collect = collect
    trainer.sample_windows = lambda batch, horizon: {}
    trainer.sample_starts = lambda batch: ({}, torch.empty(2, 5, 45))
    def validate(*args, **kwargs):
        if audit_fails:
            raise ValueError('Nonfinite validation result')
        return {}

    monkeypatch.setattr('supermimic_rl.world_model_v2.online.validate', validate)
    trainer.save = lambda *, with_replay: events.append(('save', with_replay, trainer.cycle))
    if audit_fails:
        with pytest.raises(ValueError, match='Nonfinite validation'):
            trainer.train(checkpoint_requested=lambda: requested[0])
    else:
        report = trainer.train(checkpoint_requested=lambda: requested[0])
        assert report['cycle'] == 2 and 'budget remains' in report['scope']
        assert (tmp_path / 'world-validation-000002.json').is_file()
    phase = 'estimator' if world_only else 'policy'
    assert events == ['collect', 'model', 'model', phase, phase, phase, ('save', True, 2)]
    assert learner.updates['actor'] == (1 if world_only else 4)



def _stub_trainer(tmp_path, monkeypatch, calls, **config):
    learner = SimpleNamespace(updates={'actor': 0}, skipped={'actor': 0}, fit=None)
    learner.fit_model = lambda windows: {'model/updated': 1.}
    learner.fit_estimator = lambda start, frames: calls.append('estimator') or {'estimator/updated': 1.}

    def update_policy(start, frames, **kwargs):
        calls.append('actor' if kwargs['train_actor'] else 'critic')
        return {'actor/updated': float(kwargs['train_actor'])}

    learner.update_policy = update_policy
    learner.imagination = SimpleNamespace(termination=SimpleNamespace(limits=SimpleNamespace(forbidden_body_load_n=1.)),
                                          contact_floor=None)

    class Replay:
        def __len__(self):
            return 24

    trainer = object.__new__(OnlineTrainer)
    trainer.guard = None
    trainer.config = OnlineConfig(**{'cycles': 3, 'model_updates': 1, 'actor_updates': 2, 'record_tasks': True,
                                     'record_stair_source': True, 'course_coordinate': 0, 'policy_start_cycle': 2,
                                     **config})
    trainer.cycle, trainer.output = 0, tmp_path
    trainer.env = SimpleNamespace(num_envs=2)
    trainer.device = torch.device('cpu')
    trainer.generator = trainer.validation_generator = None
    trainer.validation_ids = trainer.train_ids = [0]
    trainer.learner = learner
    trainer.mixing = SimpleNamespace(actor_updates=0, fraction=.5, teacher_steps=0, student_steps=0)
    trainer.collector = SimpleNamespace(steps=0, episode_offset=1, replay=Replay())
    trainer.collector.collect = lambda *args, **kwargs: {'terminations': 0}
    trainer.sample_windows = lambda batch, horizon: {}
    trainer.sample_starts = lambda batch: ({}, torch.empty(2, 5, 45))
    trainer.save = lambda **kwargs: None
    monkeypatch.setattr('supermimic_rl.world_model_v2.online.validate', lambda *args, **kwargs: {})
    return trainer


def test_critic_warmup_cycles_fit_the_critic_before_any_actor_step(tmp_path, monkeypatch):
    calls = []
    _stub_trainer(tmp_path, monkeypatch, calls, critic_warmup=True).train()
    # The separate supervised estimator warmup runs first; cycles then fit the critic before the actor.
    cycles = calls[calls.index('critic'):]
    assert cycles == ['critic', 'critic', 'critic', 'critic', 'actor', 'actor']


def test_contact_floor_is_refit_after_each_world_fit_before_imagination(tmp_path, monkeypatch):
    calls = []
    floor = torch.full((21,), 2.), torch.tensor(3.)
    report = {'channels': {'base_link': {'floor_n': 2., 'false_positive_raw': 9, 'false_positive_floored': 0,
                                         'true_positive_raw': 4, 'true_positive_floored': 4, 'negatives': 90}}}

    def calibrate(fit, replay, env_ids, batch, generator, device, *, threshold_n):
        assert env_ids == [0] and threshold_n == 1.
        calls.append('floor')
        return {'body_floor': floor[0], 'base_floor': floor[1], 'report': report}

    monkeypatch.setattr('supermimic_rl.world_model_v2.online.calibrate_contact_floor', calibrate)
    trainer = _stub_trainer(tmp_path, monkeypatch, calls, critic_warmup=True, calibrate_contact_floor=True)
    trainer.train()
    cycles = calls[calls.index('floor'):]
    assert cycles == ['floor', 'critic', 'critic', 'floor', 'critic', 'critic', 'floor', 'actor', 'actor']
    assert trainer.learner.imagination.contact_floor == floor
    metrics = [json.loads(line) for line in (tmp_path / 'metrics.jsonl').read_text().splitlines()]
    assert metrics[-1]['contact_floor']['base_link'] == {key: report['channels']['base_link'][key] for key in
        ('floor_n', 'false_positive_raw', 'false_positive_floored', 'true_positive_raw', 'true_positive_floored')}
    assert json.loads((tmp_path / 'world-validation-000003.json').read_text())['contact_floor'] == report


def test_guard_judges_only_policy_cycles_and_restores_after_a_sustained_drop(tmp_path, monkeypatch):
    from supermimic_rl.world_model_v2.guard import NativeRegressionGuard

    rewards = iter([.02, .09, .09, .09, .05, .04])
    trainer = _stub_trainer(tmp_path, monkeypatch, [], cycles=6, guard_drop_fraction=.1, guard_patience=2)
    learner = trainer.learner
    learner.actor = torch.nn.Linear(2, 2)
    learner.optimizers = {'actor': torch.optim.Adam(learner.actor.parameters(), lr=5e-5)}
    original = {name: value.clone() for name, value in learner.actor.state_dict().items()}

    def update_policy(start, frames, **kwargs):
        if kwargs['train_actor']:
            with torch.no_grad():
                learner.actor.weight.add_(1.)
        return {'actor/updated': float(kwargs['train_actor'])}

    learner.update_policy = update_policy
    trainer.collector.collect = lambda *args, **kwargs: {'terminations': 0, 'student_reward_mean': next(rewards)}
    trainer.guard = NativeRegressionGuard(.1, 2, 5e-5)
    trainer.train()
    metrics = [json.loads(line) for line in (tmp_path / 'metrics.jsonl').read_text().splitlines()]
    assert ['guard' in row for row in metrics] == [False, False, True, True, True, True]
    assert metrics[2]['guard']['best_student_reward'] == .09 and not metrics[4]['guard']['restored']
    assert metrics[5]['guard']['restored'] and metrics[5]['guard']['actor_learning_rate'] == 2.5e-5
    # Restored to the actor that collected the best (baseline) return, then that cycle's two updates.
    torch.testing.assert_close(learner.actor.weight, original['weight'] + 2.)
