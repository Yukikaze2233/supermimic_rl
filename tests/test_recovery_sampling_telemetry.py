"""Actual reset draws remain distinct from geometry readback and recovery outcomes."""
from copy import deepcopy
import io
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('transformer_rl.formal.tasks.recovery')
from transformer_rl.formal.tasks.recovery import RECOVERY_NAMES

from supermimic_rl.course.curriculum import CourseCurriculumConfig
from supermimic_rl.course.jump_first import JumpFirstCurriculum
from supermimic_rl.course.metrics import course_progress
from supermimic_rl.course.recovery_bridge import RECOVERY_VARIANTS
from supermimic_rl.ppo.environment import CourseEnvironment
from supermimic_rl.ppo.trainer import PPOTrainer
from supermimic_rl.world_model_v2.course_progression import CourseProgression, CourseProgressionConfig
from support.ppo import TargetEnvironment, config


PREFIX = 'course/recovery/birth_samples/'


def curriculum(bridge=True, n=32):
    weights = list(CourseCurriculumConfig().stage_weights)
    weights[0] = (0., 0., 0., 0., 0., 1.)
    return JumpFirstCurriculum(n, seed=7, config=CourseCurriculumConfig(
        recovery_postures=RECOVERY_NAMES, stage_weights=tuple(weights), recovery_roll_bridge=bridge))


def native_metrics(course):
    def metrics():
        return {'physical_request_outcomes': {}, 'stair_closed_loop': {},
                'authored_stair_completion': {'ended_by_tile': [], 'ended_complete_by_tile': []},
                'recovery': {'counts': {'births': sum(course.sample_counts['recovery_posture'])}},
                'episode_birth_family_counts': {'recovery': sum(course.sample_counts['recovery_posture'])},
                'level_histogram': []}
    return SimpleNamespace(_terrain_course=SimpleNamespace(metrics=metrics))


@pytest.mark.parametrize('bridge', [False, True])
def test_existing_reset_draws_keep_original_and_bridge_labels_and_detached_values(bridge):
    course = curriculum(bridge)
    course.sample(torch.arange(32))
    before, rng = deepcopy(course.sample_counts), course.sampler.generator.get_state().clone()
    values = course.recovery_birth_samples()
    names = RECOVERY_VARIANTS if bridge else RECOVERY_NAMES
    assert tuple(values) == names and sum(values.values()) == 32
    assert values['left_side_90'] == values['right_side_90'] == 2
    if bridge:
        assert [values[name] for name in RECOVERY_VARIANTS[16:]] == [1] * 6
    else:
        assert set(values.values()) == {2}
    counters, _ = course_progress(native_metrics(course), course)
    assert {name.removeprefix(PREFIX): count for name, count in counters.items()
            if name.startswith(PREFIX)} == values
    assert counters['course/recovery/births'] == sum(values.values())
    values[names[0]] = -1
    assert course.sample_counts == before
    assert torch.equal(course.sampler.generator.get_state(), rng)


def test_ppo_and_world_use_existing_counter_deltas_without_repeating_previous_draws():
    course = curriculum()
    native = native_metrics(course)
    course.sample(torch.arange(32))  # Cold setup is an observation baseline.
    environment = object.__new__(CourseEnvironment)
    environment.env, environment.adapter = native, SimpleNamespace(curriculum=course)
    trainer = object.__new__(PPOTrainer)
    trainer.environment, trainer._course_counters = environment, environment.course_progress()[0]
    progress = CourseProgression(CourseProgressionConfig(0, 10, 0, 1), course)
    before = progress.snapshot_counters(native)
    course.sample(torch.arange(7))
    ppo = trainer._course_progress()
    world = progress.observe(native, {'course_coordinate': 0}, before)['counters']
    assert ppo == world | {'course/terrain/level_fraction': 0., 'course/jump/height_cap_m': course.command_context().domain.jump_m[1]}
    assert sum(value for name, value in ppo.items() if name.startswith(PREFIX)) == 7
    assert not any(value for name, value in trainer._course_progress().items() if name.startswith(PREFIX))


class SamplingEnvironment(TargetEnvironment):
    """The native-like draw ledger changes on reset, never on observation reads."""

    def __init__(self):
        super().__init__(num_envs=4)
        self.curriculum = curriculum(n=4)
        self.native = native_metrics(self.curriculum)
        self.resets = 0

    def reset(self):
        self.curriculum.sample(torch.arange(self.num_envs))
        self.resets += 1
        return super().reset()

    def course_progress(self):
        return course_progress(self.native, self.curriculum)

    def recovery_birth_samples(self):
        return self.curriculum.recovery_birth_samples()

    def course_stage_starts(self):
        return {'flat': 0, 'jump': 100}

    def check_course_gates(self, keys):
        pass

    def set_opened_course_gates(self, keys):
        pass


def report_config():
    return config(updates=1, course_gate_mode='report', course_gates=[
        {'stage': 'jump', 'criteria': [{'metric': 'rms/forward_error_m_s', 'max': 1.}]}])


def test_final_cold_setup_is_excluded_and_checkpoint_reads_current_host_values(tmp_path):
    env = SamplingEnvironment()
    trainer = PPOTrainer(env, report_config(), tmp_path)
    assert env.resets == 2
    assert not any(value for name, value in trainer._course_progress().items() if name.startswith(PREFIX))
    env.curriculum.sample(torch.arange(3))
    counts = deepcopy(env.curriculum.sample_counts)
    rng = env.curriculum.sampler.generator.get_state().clone()
    torch_rng = torch.get_rng_state().clone()
    # A full immutable snapshot precedes _course_progress in the actual loop.
    snapshot = torch.load(io.BytesIO(trainer._checkpoint_snapshot()), weights_only=True)
    light = trainer.state_dict(light=True)
    telemetry = snapshot['recovery_sampling_telemetry']
    assert telemetry == light['recovery_sampling_telemetry']
    assert sum(telemetry['setup_baseline'].values()) == 8
    assert sum(telemetry['cumulative'].values()) == 11
    assert sum(telemetry['after_setup'].values()) == 3
    assert telemetry['restored_on_resume'] is False
    assert 'not placement readback or skill success' in telemetry['scope']
    assert env.curriculum.sample_counts == counts
    assert torch.equal(env.curriculum.sampler.generator.get_state(), rng)
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert sum(value for name, value in trainer._course_progress().items() if name.startswith(PREFIX)) == 3
    env.curriculum.sample(torch.arange(4))
    assert sum(telemetry['cumulative'].values()) == 11
    telemetry['cumulative']['left_side_90'] = -100
    assert env.curriculum.recovery_birth_samples()['left_side_90'] >= 0


def test_cold_resume_does_not_restore_old_sampling_telemetry_or_sampler(tmp_path):
    old = SamplingEnvironment()
    trainer = PPOTrainer(old, report_config(), tmp_path / 'old')
    for _ in range(12):
        old.curriculum.sample(torch.arange(4))
    path = tmp_path / 'resume.pt'
    trainer.save(path)
    saved = torch.load(path, weights_only=True)
    assert sum(saved['recovery_sampling_telemetry']['cumulative'].values()) == 56
    new = SamplingEnvironment()
    restored = PPOTrainer(new, report_config(), tmp_path / 'new', resume=path)
    assert new.resets == 2
    expected = curriculum(n=4)
    expected.sample(torch.arange(4))
    expected.sample(torch.arange(4))
    assert new.curriculum.sample_counts == expected.sample_counts
    assert torch.equal(new.curriculum.sampler.generator.get_state(), expected.sampler.generator.get_state())
    telemetry = restored.state_dict()['recovery_sampling_telemetry']
    assert sum(telemetry['cumulative'].values()) == 8
    assert sum(telemetry['after_setup'].values()) == 0
    assert not any(value for name, value in restored._course_progress().items() if name.startswith(PREFIX))


def test_old_mock_or_missing_environment_has_no_new_checkpoint_requirement(tmp_path):
    trainer = PPOTrainer(TargetEnvironment(num_envs=4), config(updates=1), tmp_path / 'old')
    assert 'recovery_sampling_telemetry' not in trainer.state_dict()
    trainer.save(tmp_path / 'old.pt')
    restored = PPOTrainer(TargetEnvironment(num_envs=4), config(updates=1), tmp_path / 'new',
                          resume=tmp_path / 'old.pt')
    assert 'recovery_sampling_telemetry' not in restored.state_dict(light=True)
    del restored.environment
    assert 'recovery_sampling_telemetry' not in restored.state_dict()
