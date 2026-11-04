"""Fixed requests reuse real source task mapping and command admission on CPU."""
from types import SimpleNamespace
from dataclasses import replace
import math
import os
from pathlib import Path
import sys

import pytest
import torch

_UPSTREAM = Path(os.environ.get("SUPERMIMIC_UPSTREAM_ROOT",
    Path(__file__).resolve().parents[2] / "transformer_rl-worktrees/full-vehicle-gimbal"))
if (_UPSTREAM / "src").is_dir():
    sys.path.insert(0, str(_UPSTREAM / "src"))

from supermimic_rl.course.evaluation import CourseEvaluation, CourseEvaluationConfig, FixedCourseScenarios
from supermimic_rl.contracts import Field, Snapshot, StateSchema, Transition
from supermimic_rl.evaluation import NativeTrackingSample


@pytest.fixture
def adapter():
    mapping = pytest.importorskip("transformer_rl.formal.lab_env.formal_curriculum")
    from transformer_rl.formal.lab_env.course_tasks import LabCourseTasks
    from transformer_rl.formal.tasks.command_pool import CommandPool
    from transformer_rl.formal.tasks.gimbal_course import GimbalCourse, GimbalCourseConfig
    from transformer_rl.formal.tasks.geometry import TerrainKind
    from transformer_rl.formal.tasks.trajectory import DriveEnvelope
    from supermimic_rl.course.curriculum import SuperMimicCurriculum

    n = 32
    curriculum = SuperMimicCurriculum(n)
    pool = CommandPool()
    tasks = LabCourseTasks(n, DriveEnvelope(.06, .34), command_pool=pool,
                           command_context=curriculum.command_context, policy_dt_s=.02,
                           terrain_mode_labels=False)
    router = SimpleNamespace(step_table=torch.tensor([0., .02, .20]),
                              kinds=torch.tensor([int(TerrainKind.FLAT), int(TerrainKind.STEP_UP),
                                                  int(TerrainKind.STEP_UP)]))
    gimbal = GimbalCourse(GimbalCourseConfig(enabled=True, spin_translate=True), n)
    env = SimpleNamespace(_terrain_course=SimpleNamespace(tasks=tasks, router=router),
                          _gimbal_course=gimbal, _task_sampler=mapping.LabTaskDrawSampler(curriculum))
    return SimpleNamespace(env=env, curriculum=curriculum, num_envs=n, device=torch.device("cpu"))


def test_fixed_suite_preserves_ids_and_maps_real_request_and_recovery_modes(adapter):
    from transformer_rl.formal.control.observation import CommandMode
    from transformer_rl.formal.tasks.curriculum import TaskFamily
    from transformer_rl.formal.tasks.geometry import TerrainKind

    original = adapter.curriculum
    plan = FixedCourseScenarios(adapter, CourseEvaluationConfig())
    plan.install()
    ids = torch.arange(adapter.num_envs)
    sample = adapter.env._task_sampler(ids)
    for row, name in enumerate(plan.row_names):
        case = plan.scenarios[name]
        assert sample.draw.family[row] == int(TaskFamily[case["family"]])
        assert sample.terrain_kind[row] == int(TerrainKind[case["terrain"]])
        if case["family"] == "REQUEST":
            assert sample.mode[row] == int(CommandMode.JUMP)
        if case["family"] == "RECOVERY":
            assert sample.mode[row] == int(CommandMode.RECOVERY)
    selection = torch.tensor([7, 3, 0])
    again = adapter.env._task_sampler(selection)
    torch.testing.assert_close(again.draw.requested_command, sample.draw.requested_command[selection])
    torch.testing.assert_close(again.draw.recovery_posture, sample.draw.recovery_posture[selection])
    assert plan.scenarios["authored_single_step"]["step_up_m"] == pytest.approx(.2)
    assert plan.scenarios["low_obstacle"]["step_up_m"] == pytest.approx(.02)
    assert original.real_transitions == 0  # Evaluation does not charge training.
    resolved = plan.description()["resolved_curriculum"]
    assert resolved["stage"] == 7 and resolved["stage_name"] == "combined_dr"
    assert resolved["source_schedule_coordinate"] == 34999
    assert len(plan.names) == 27 and not plan.description()["unassigned_scenarios"]
    assert plan.curriculum.terrain_step_up_cap_m() == pytest.approx(.2)
    for motion in ("standing", "moving"):
        assert [plan.scenarios[f"jump_{motion}_{level}"]["jump_height_m"]
                for level in ("low", "intermediate", "high")] == pytest.approx([.03, .10, .40])
    for row, name in enumerate(plan.row_names):
        assert sample.draw.jump_height_m[row] == pytest.approx(plan.scenarios[name]["jump_height_m"])


def test_source_task_reset_uses_fixed_braking_segments_and_gimbal_contract(adapter):
    from transformer_rl.formal.control.gimbal import GimbalMode

    plan = FixedCourseScenarios(adapter, CourseEvaluationConfig(
        scenarios=("braking", "spin_translation_positive", "gimbal_yaw_sweep_positive",
                   "spin_translation_negative", "gimbal_yaw_sweep_negative")))
    plan.install()
    ids = torch.arange(adapter.num_envs)
    sample = adapter.env._task_sampler(ids)
    tasks = adapter.env._terrain_course.tasks
    tasks.reset(ids, sample, torch.zeros(adapter.num_envs, dtype=torch.float64),
                terrain_kind=sample.terrain_kind, terrain_step_height_m=sample.draw.step_up_m)
    torch.testing.assert_close(tasks.trajectory.end[0], torch.tensor(plan.scenarios["braking"]["commands"][0]))
    tasks.elapsed.fill_(plan.pool.config.segment_s[0])
    later = tasks._draw_segments(torch.tensor([0, 1, 2]))
    assert later.command[0, :3].eq(0).all()
    assert later.command[1, 0] > 0  # Native eligibility needs a nonzero ordinary command.
    plan.draw_gimbal(torch.tensor([1, 2, 3, 4]), torch.zeros(adapter.num_envs, 2),
                     torch.ones(adapter.num_envs, dtype=torch.bool), update=0)
    course = adapter.env._gimbal_course
    assert course.special[1] and not course.special[2]
    assert course.mode[1] == int(GimbalMode.WORLD_HEADING)
    assert course.mode[2] == int(GimbalMode.YAW_RATE)
    assert course.rate[2] == pytest.approx(course.config.yaw_rate_rad_s[1])
    assert course.target[2, 1] == pytest.approx(course.config.pitch_target_rad[1])
    assert course.rate[4] == pytest.approx(course.config.yaw_rate_rad_s[0])
    assert course.chassis_rate[1] == pytest.approx(course.config.chassis_spin_rad_s[0])
    assert course.chassis_rate[3] == pytest.approx(-course.config.chassis_spin_rad_s[0])
    tasks.elapsed.zero_()
    plan.draw_gimbal(torch.arange(5), torch.zeros(adapter.num_envs, 2),
                     torch.ones(adapter.num_envs, dtype=torch.bool), update=0)
    # Both pitch targets exist at birth; a first-episode failure cannot hide one.
    assert course.target[:5, 1].tolist() == pytest.approx(
        [course.config.pitch_target_rad[row % 2] for row in range(5)])


def test_family_selection_and_unassigned_coverage_are_explicit(adapter):
    plan = FixedCourseScenarios(adapter, CourseEvaluationConfig(family="RECOVERY"))
    assert len(plan.names) == 4
    assert all(case["family"] == "RECOVERY" for case in plan.scenarios.values())
    adapter.num_envs = 1
    plan = FixedCourseScenarios(adapter, CourseEvaluationConfig())
    report = plan.description()
    assert report["assigned_rows"][plan.names[0]] == 1
    assert report["unassigned_scenarios"] == list(plan.names[1:])
    with pytest.raises(ValueError, match="Unknown fixed scenario"):
        FixedCourseScenarios(adapter, CourseEvaluationConfig(scenarios=("imagined_success",)))


def test_fixed_jump_evaluation_uses_the_selected_schedule_order_and_domain(adapter):
    from supermimic_rl.course.jump_first import JumpFirstCurriculum

    adapter.curriculum = JumpFirstCurriculum(adapter.num_envs)
    plan = FixedCourseScenarios(adapter, CourseEvaluationConfig(
        evaluation_stage='active_jump', scenarios=('jump_standing_high', 'jump_moving_high')))
    assert isinstance(plan.curriculum, JumpFirstCurriculum)
    assert plan.curriculum.stage == 2
    assert plan.domain.jump_m == adapter.curriculum.initial_domain.jump_m
    assert plan.domain.forward_m_s == adapter.curriculum.initial_domain.forward_m_s
    assert plan.scenarios['jump_standing_high']['jump_height_m'] == pytest.approx(.10)


def test_command_source_preserves_native_handoff_rng_and_restores_hooks(adapter):
    tasks = adapter.env._terrain_course.tasks
    original_curriculum, original_tracking = adapter.curriculum, tasks.begin_tracking
    plan = FixedCourseScenarios(adapter, CourseEvaluationConfig(scenarios=('stand',)))
    plan.install()
    assert 'step' not in tasks.stream.__dict__
    assert 'recovery_qualification' not in plan.description()['selection']
    assert 'recovery_qualification' not in plan.description()
    ids = torch.arange(adapter.num_envs)
    sampled = adapter.env._task_sampler(ids)
    tasks.reset(ids, sampled, torch.zeros(adapter.num_envs, dtype=torch.float64),
                terrain_kind=sampled.terrain_kind, terrain_step_height_m=sampled.draw.step_up_m)
    assert plan.command_source.eq(1).all()
    expected_rng = torch.Generator().set_state(tasks.generator.get_state())
    expected_vx, expected_wz = tasks.command_pool.handoff(2, expected_rng, 'cpu')
    mask = ids < 2
    tasks.begin_tracking(mask)
    torch.testing.assert_close(tasks.trajectory.end[:2, 0], expected_vx)
    torch.testing.assert_close(tasks.trajectory.end[:2, 2], expected_wz)
    assert torch.equal(tasks.generator.get_state(), expected_rng.get_state())
    assert plan.command_source[:3].tolist() == [2, 2, 1]
    tasks._draw_segments(torch.tensor([1]))
    assert plan.command_source[:3].tolist() == [2, 1, 1]
    plan.close()
    assert adapter.curriculum is original_curriculum
    assert tasks.begin_tracking == original_tracking
    assert not {'_draw_segments', 'begin_tracking'} & tasks.__dict__.keys()


def test_recovery_qualification_overrides_request_before_native_admission_and_preserves_handoff(adapter):
    from transformer_rl.formal.control.observation import CommandMode
    from transformer_rl.formal.lab_env.recovery_lifecycle import RecoveryLifecycle
    from transformer_rl.formal.tasks.trajectory import CommandRates, CommandStream

    # A nondefault height must still pass through native initialization and slew.
    config = adapter.curriculum.config
    adapter.curriculum = type(adapter.curriculum)(adapter.num_envs, config=replace(
        config, domain=replace(config.domain, nominal_height_m=.36)))
    native = RecoveryLifecycle(adapter.num_envs)
    adapter.env._terrain_course.recovery = native
    tasks = adapter.env._terrain_course.tasks
    tasks.stream = CommandStream(tasks.stream.current, tasks.stream.envelope,
        CommandRates(1., 2., .1), policy_dt_s=tasks.policy_dt_s)
    original_step = tasks.stream.step
    name = 'recovery_' + config.recovery_postures[0]
    plan = FixedCourseScenarios(adapter, CourseEvaluationConfig(
        scenarios=(name, 'forward_speed'), recovery_qualification=True))
    plan.install()
    try:
        ids = torch.arange(adapter.num_envs)
        sample = adapter.env._task_sampler(ids)
        tasks.reset(ids, sample, torch.zeros(adapter.num_envs, dtype=torch.float64),
                    terrain_kind=sample.terrain_kind, terrain_step_height_m=sample.draw.step_up_m)
        hold = plan.qualification_rows
        torch.testing.assert_close(tasks.stream.current[:, 3], torch.full((adapter.num_envs,), .32))
        assert plan.qualification_commands[hold, 3].tolist() == pytest.approx([.36] * int(hold.sum()))
        expected_rng = torch.Generator().set_state(tasks.generator.get_state())
        vx, wz = tasks.command_pool.handoff(int(hold.sum()), expected_rng, 'cpu')
        # Native changes mode before begin_tracking, so scenario identity is
        # essential: a mode==RECOVERY hook would miss this exact transition.
        tasks.base_mode[hold] = int(CommandMode.NORMAL)
        tasks.begin_tracking(hold)
        trajectory = tasks.trajectory.end.clone()
        torch.testing.assert_close(trajectory[hold, 0], vx)
        torch.testing.assert_close(trajectory[hold, 2], wz)
        assert torch.equal(tasks.generator.get_state(), expected_rng.get_state())
        assert (trajectory[hold, :3] != 0).any()
        modes = tasks.base_mode.clone()
        reference = CommandStream(tasks.stream.current, tasks.stream.envelope,
                                  tasks.stream.rates, policy_dt_s=tasks.policy_dt_s)
        for clock in (0., native.config.timeout_s - native.config.physics_dt_s,
                      native.config.timeout_s):
            plan.qualification_clock.fill_(clock)
            wanted = trajectory.clone()
            if clock < native.config.timeout_s:
                wanted[hold] = plan.qualification_commands[hold]
            expected = reference.step(wanted)
            admitted, _, _, _ = tasks.step(torch.full((adapter.num_envs,), .02, dtype=torch.float64),
                                            None, torch.zeros(adapter.num_envs))
            # Native automatic-request timing draws once even when no row is due.
            torch.rand(adapter.num_envs, generator=expected_rng)
            for field in ('requested', 'wheel_projected', 'admitted', 'slew_fraction'):
                torch.testing.assert_close(getattr(admitted, field), getattr(expected, field))
            torch.testing.assert_close(admitted.requested[~hold], trajectory[~hold])
            torch.testing.assert_close(tasks.trajectory.end, trajectory)
            torch.testing.assert_close(tasks.base_mode, modes)
            assert torch.equal(tasks.generator.get_state(), expected_rng.get_state())
            if clock == 0.:
                assert admitted.admitted[hold, 3].tolist() == pytest.approx([.322] * int(hold.sum()))
                assert plan.command_source[hold].eq(1).all()
        assert plan.description()['selection']['recovery_qualification'] is True
    finally:
        plan.close()
    assert tasks.stream.step == original_step and 'step' not in tasks.stream.__dict__


def test_qualification_request_budget_uses_the_installed_physics_clock(adapter):
    """Owned physics ticks, including those after handoff, govern request expiry."""
    from transformer_rl.formal.control.observation import CommandMode
    from transformer_rl.formal.lab_env.recovery_lifecycle import RecoveryLifecycle, RecoveryLifecycleConfig
    from transformer_rl.formal.tasks.protocol import CompletedEstimate

    n, env = adapter.num_envs, adapter.env
    tasks = env._terrain_course.tasks
    owner = RecoveryLifecycle(n, config=RecoveryLifecycleConfig(policy_dt_s=.02, stable_s=.02, timeout_s=.08))
    env._terrain_course.recovery = owner
    env._terrain_course.metrics = lambda: {}
    env._episode_id = torch.zeros(n, dtype=torch.long)
    ids = torch.arange(n)

    def reset():
        sample = env._task_sampler(ids)
        tasks.reset(ids, sample, torch.zeros(n, dtype=torch.float64),
                    terrain_kind=sample.terrain_kind, terrain_step_height_m=sample.draw.step_up_m)
        owner.reset(ids, sample.mode == int(CommandMode.RECOVERY))
        env._episode_id += 1

    adapter.reset = reset
    name = 'recovery_' + adapter.curriculum.config.recovery_postures[0]
    evaluator = CourseEvaluation(adapter, CourseEvaluationConfig(
        scenarios=(name, 'forward_speed'), recovery_qualification=True))
    evaluator.begin(StateSchema((Field('unused', ()),), ()))
    try:
        plan = evaluator.plan
        rows = plan.qualification_rows
        assert evaluator.recovery_evidence.qualification['clock'] is plan.qualification_clock
        requested = torch.tensor([[1., 0., 0., .32]]).expand(n, -1).clone()
        values = dict(height_m=torch.full((n,), .32), command_height_m=torch.full((n,), .32),
            tilt_rad=torch.zeros(n), angular_speed_rad_s=torch.zeros(n), velocity_z_m_s=torch.zeros(n),
            wheel_load_n=torch.full((n, 2), 20.), base_load_n=torch.zeros(n),
            policy_owned=torch.zeros(n, dtype=torch.bool))
        owner.observe_physics(**values)
        assert plan.qualification_clock.eq(0).all()  # Release does not spend policy-owned time.
        values['policy_owned'].fill_(True)
        for interval in range(4):
            admitted = tasks.stream.step(requested)
            torch.testing.assert_close(admitted.requested[rows], plan.qualification_commands[rows])
            torch.testing.assert_close(admitted.requested[~rows], requested[~rows])
            for _ in range(4):
                owner.observe_physics(**values)
            if interval == 1:
                values['policy_owned'].fill_(False)
                owner.observe_physics(**values)
                values['policy_owned'].fill_(True)
            owned_time = torch.full((n,), .02 * (interval + 1), dtype=torch.float64)
            now = owned_time + (.005 if interval >= 1 else 0.)
            estimate = CompletedEstimate(torch.zeros(n), torch.ones(n, 2), torch.zeros(n), torch.zeros(n),
                now - .01, torch.ones(n, dtype=torch.bool))
            result = owner.finish_interval(now, estimate, policy_owned=values['policy_owned'],
                                           failed=torch.zeros(n, dtype=torch.bool))
            if interval == 0:
                assert result.handoff[rows].all()
                assert not evaluator.recovery_evidence.qualification['open'][rows].any()
            torch.testing.assert_close(plan.qualification_clock[rows], owned_time[rows])
            if interval >= 1:
                assert not torch.allclose(plan.qualification_clock[rows], now[rows])
            assert plan.qualification_clock[~rows].eq(0).all()
        # Physical success/handoff did not release the request early; only
        # the shared owned-time budget now permits the native request again.
        torch.testing.assert_close(tasks.stream.step(requested).requested, requested)
        owner.reset(ids, rows)
        assert plan.qualification_clock.eq(0).all()
        torch.testing.assert_close(tasks.stream.step(requested).requested[rows], plan.qualification_commands[rows])
    finally:
        evaluator.close()
    assert 'step' not in tasks.stream.__dict__
    assert not {'observe_physics', 'finish_interval', 'reset'} & owner.__dict__.keys()


def test_fixed_physical_tiles_ignore_training_proficiency_and_ceiling_exploration(adapter):
    from transformer_rl.formal.tasks.terrain_courses import TerrainCourseRouter

    path = Path(__file__).resolve().parents[1] / 'assets/rmuc2027_course_atlas/manifest.json'
    router = TerrainCourseRouter(path, adapter.num_envs, seed=7)
    adapter.env._terrain_course.router = router
    original_reset = router.reset
    plan = FixedCourseScenarios(adapter, CourseEvaluationConfig(
        scenarios=('low_obstacle', 'authored_single_step', 'authored_two_platforms', 'slope', 'rough')))
    plan.install()
    ids = torch.arange(adapter.num_envs)
    sampled = adapter.env._task_sampler(ids)
    requested = {name: getattr(sampled.draw, name) for name in
                 ('slope_rad', 'roughness_m', 'step_up_m', 'step_down_m')}
    chosen = torch.tensor([plan.fixed_tiles[name] for name in plan.row_names])
    for _ in range(3):
        router.level.random_(0, router.manifest['levels'])
        origins, _ = router.reset(ids, sampled.terrain_kind, stage=2,
            command_forward=sampled.command[:, 0], requested=requested,
            step_up_cap_m=.2, level_cap=router.manifest['levels'] - 1)
        assert torch.equal(router.tile, chosen)
        torch.testing.assert_close(origins, router.origins_table[chosen])
    assert router.stair_coverage_counts.eq(0).all()
    tiles = router.manifest['tiles']
    assert tiles[plan.fixed_tiles['low_obstacle']]['name'] == 'step_up'
    assert tiles[plan.fixed_tiles['authored_single_step']]['name'] == 'road_step'
    assert tiles[plan.fixed_tiles['authored_two_platforms']]['parameters']['profile_m'] == [0., .2, .35, .3]
    assert plan.scenarios['authored_single_step']['step_up_m'] == pytest.approx(.2)
    plan.close()
    assert router.reset == original_reset and 'reset' not in router.__dict__


def test_phase_partition_and_terrain_metadata_belong_to_the_pre_reset_interval():
    n = 4
    schema = StateSchema((Field('course.terrain_tile', ()),), ())
    router = SimpleNamespace(tile=torch.arange(n), step_table=torch.tensor([0., .003, .2, .2]),
        slope_table=torch.zeros(n), roughness_table=torch.zeros(n), manifest={'tiles': [
            {'name': 'fixture', 'kind': 3, 'level': 0,
             'parameters': {'step_height_m': .2, 'profile_m': [0., .2, .35, .3]}}] * n})
    counters = {'physical_request_outcomes': {}, 'request_end_counts': {},
                'recovery': {'counts': {}}, 'authored_stair_completion': {}}
    evaluator = CourseEvaluation(SimpleNamespace(env=SimpleNamespace(
        _terrain_course=SimpleNamespace(router=router, metrics=lambda: counters))))
    evaluator.schema, evaluator.before, evaluator._active = schema, counters, True
    evaluator.plan = SimpleNamespace(names=('stand',), row_names=('stand',) * n,
        command_source=torch.tensor([1, 2, 1, 0]), description=lambda: {'fixture': True})
    context = evaluator.sample()
    evaluator.plan.command_source.fill_(2)
    router.tile.zero_()  # An automatic reset must not rewrite the completed interval's metadata.
    owned = torch.tensor([False, True, True, True])
    snapshot = Snapshot(torch.arange(n).float()[:, None], torch.zeros(n, 35), torch.zeros(n, 11),
        torch.empty(n, 0), torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long), owned)
    transition = Transition(snapshot, torch.zeros(n, 6), snapshot, torch.ones(n),
        torch.tensor([True, False, True, False]), torch.tensor([False, True, False, False]),
        torch.ones(n, dtype=torch.bool))
    evaluator.step(transition, NativeTrackingSample(torch.ones(n, 3), torch.zeros(n, 3), course=context))
    report = evaluator.summary()
    bucket = report['buckets']['stand']
    phases = bucket['command_phases']
    assert set(phases) == {'script_owned', 'native_handoff', 'scheduled_segment', 'unclassified'}
    for key in ('samples', 'terminated', 'truncated', 'policy_owned_samples', 'reward_sum'):
        assert sum(phase[key] for phase in phases.values()) == bucket[key]
    assert phases['native_handoff']['truncated'] == 1
    assert phases['scheduled_segment']['terminated'] == 1
    assert phases['scheduled_segment']['routed_terrain_ranges']['routed_step_height_m']['min'] == pytest.approx([.2])
    assert report['routed_terrain_catalog']['2']['profile_peak_above_first_m'] == .35
    evaluator.close()
    assert evaluator.sample() is None


def test_recovery_phase_uses_native_pre_action_active_even_after_new_command_draw():
    from transformer_rl.formal.lab_env.recovery_lifecycle import RecoveryLifecycle
    from supermimic_rl.course.recovery_evaluation import RecoveryEvaluation

    n = 3
    recovery = RecoveryLifecycle(n)
    recovery.reset(torch.arange(n), torch.tensor([True, False, False]))
    router = SimpleNamespace(tile=torch.zeros(n, dtype=torch.long), step_table=torch.zeros(1),
        slope_table=torch.zeros(1), roughness_table=torch.zeros(1))
    counters = {'physical_request_outcomes': {}, 'request_end_counts': {},
                'recovery': {'counts': {}}, 'authored_stair_completion': {}}
    env = SimpleNamespace(_episode_id=torch.ones(n, dtype=torch.long),
        _terrain_course=SimpleNamespace(recovery=recovery, router=router, metrics=lambda: counters))
    evaluator = CourseEvaluation(SimpleNamespace(env=env))
    evaluator.schema = StateSchema((Field('unused', ()),), ())
    evaluator.before, evaluator._active = counters, True
    evaluator.plan = SimpleNamespace(names=('recovery_fixture',), row_names=('recovery_fixture',) * n,
        command_source=torch.ones(n, dtype=torch.long), description=lambda: {})
    evaluator.recovery_evidence = RecoveryEvaluation(env, evaluator.plan.row_names)
    evaluator.recovery_evidence.install()
    evaluator.recovery_evidence.after_handoff[1] = True
    context = evaluator.sample()
    # Handoff/reset can change all live values before the completed interval is recorded.
    recovery.active.zero_()
    evaluator.recovery_evidence.after_handoff.zero_()
    evaluator.plan.command_source.fill_(2)
    snapshot = Snapshot(torch.zeros(n, 1), torch.zeros(n, 35), torch.zeros(n, 11), torch.empty(n, 0),
        torch.ones(n, dtype=torch.long), torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.bool))
    transition = Transition(snapshot, torch.zeros(n, 6), snapshot, torch.tensor([1., 2., 3.]),
        torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool), torch.ones(n, dtype=torch.bool))
    try:
        evaluator.step(transition, NativeTrackingSample(torch.ones(n, 3), torch.zeros(n, 3), course=context))
        report = evaluator.summary()
        bucket = report['buckets']['recovery_fixture']
        phases = bucket['recovery_phases']
        assert set(phases) == {'recovery', 'post_handoff_tracking', 'other'}
        assert phases['recovery']['reward_sum'] == 1
        assert phases['post_handoff_tracking']['reward_sum'] == 2
        assert bucket['command_phases']['scheduled_segment']['samples'] == 3
        assert sum(value['samples'] for value in phases.values()) == bucket['samples']
        assert 'recovery_active' not in bucket['routed_terrain_ranges']
        assert report['recovery_diagnostics']['by_scenario']['recovery_fixture']['births'] == 1
    finally:
        evaluator.close()
    assert 'finish_interval' not in recovery.__dict__


def test_begin_cleans_up_partially_installed_recovery_hooks(monkeypatch):
    from transformer_rl.formal.lab_env.recovery_lifecycle import RecoveryLifecycle
    from supermimic_rl.course import evaluation

    class Plan:
        def __init__(self, adapter, config):
            self.row_names = ('recovery_fixture',)
            self.installed = False

        def install(self):
            self.installed = True

        def close(self):
            self.installed = False

    owner = RecoveryLifecycle(1)
    owner.reset(torch.tensor([0]), torch.tensor([True]))
    env = SimpleNamespace(_episode_id=torch.ones(1, dtype=torch.long),
        _terrain_course=SimpleNamespace(recovery=owner, metrics=lambda: {}))
    evaluator = CourseEvaluation(SimpleNamespace(env=env, reset=lambda: None))
    original_install = evaluation.RecoveryEvaluation.install

    def fail_after_install(self):
        original_install(self)
        raise RuntimeError('incomplete install fixture')

    monkeypatch.setattr(evaluation, 'FixedCourseScenarios', Plan)
    monkeypatch.setattr(evaluation.RecoveryEvaluation, 'install', fail_after_install)
    with pytest.raises(RuntimeError, match='incomplete install fixture'):
        evaluator.begin(StateSchema((Field('unused', ()),), ()))
    assert not evaluator.plan.installed and not evaluator._active
    for name in ('observe_physics', 'finish_interval', 'reset'):
        assert name not in owner.__dict__


def test_native_jump_request_reaches_course_report_and_hooks_restore(monkeypatch):
    from supermimic_rl.course import evaluation
    from support.world_v2.jump_task import native_jumps
    from transformer_rl.formal.lab_env.course_tasks import RequestCommand
    from transformer_rl.formal.tasks.protocol import CompletedEstimate

    class Plan:
        def __init__(self, adapter, config):
            self.names = self.row_names = ('jump_fixture',)

        def install(self):
            pass

        def close(self):
            pass

        def description(self):
            return {}

    env = native_jumps()
    env._episode_id = torch.zeros(1, dtype=torch.long)
    env._interval_policy_owned = torch.ones(1, dtype=torch.bool)
    counters = {'physical_request_outcomes': {}, 'request_end_counts': {},
                'recovery': {'counts': {}}, 'authored_stair_completion': {}}
    env._terrain_course.metrics = lambda: counters
    evaluator = CourseEvaluation(SimpleNamespace(env=env, reset=lambda: None))
    monkeypatch.setattr(evaluation, 'FixedCourseScenarios', Plan)
    evaluator.begin(StateSchema((Field('unused', ()),), ()))
    tasks = env._terrain_course.tasks
    try:
        now = torch.tensor([.02], dtype=torch.float64)
        estimate = CompletedEstimate(torch.zeros(1), torch.ones(1, 2), torch.zeros(1), torch.zeros(1),
                                     now - .01, torch.ones(1, dtype=torch.bool))
        tasks.step(now, estimate, torch.zeros(1),
            command=RequestCommand(torch.tensor([7]), torch.tensor([.06]), torch.tensor([-1])))
        snapshot = Snapshot(torch.zeros(1, 1), torch.zeros(1, 35), torch.zeros(1, 11), torch.empty(1, 0),
            torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long), torch.ones(1, dtype=torch.bool))
        evaluator.step(Transition(snapshot, torch.zeros(1, 6), snapshot, torch.ones(1),
            torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool)), None)
        request = evaluator.summary()['jump_diagnostics']['requests'][0]
        assert request['request_id'] == 7 and request['observed_intervals'] == 1
        assert request['physical_end_reason'] == 'active_at_evaluation_end'
        assert not request['native_verified_physical_success']
    finally:
        evaluator.close()
    assert all(name not in tasks.__dict__ for name in ('step', 'observe_physics', 'reset'))
    assert 'step' not in tasks.protocol.__dict__


def test_course_metrics_use_transition_endpoint_and_subtract_old_training_counters():
    schema = StateSchema(tuple(Field(name, ()) for name in (
        "course.tasks.family", "course.tasks.terrain_kind", "course.terrain_tile", "course.gimbal.mode",
        "course.tasks.success_id.valid", "course.recovery._verified")), ())

    def snapshot(values):
        return Snapshot(torch.tensor(values, dtype=torch.float32), torch.zeros(2, 35),
                        torch.zeros(2, 11), torch.empty(2, 0), torch.zeros(2, dtype=torch.long),
                        torch.zeros(2, dtype=torch.long), torch.tensor([True, False]))

    before = {"physical_request_outcomes": {"verified_success": 11},
              "request_end_counts": {"timeout": 3}, "recovery": {"counts": {"verified_physical_successes": 9}},
              "authored_stair_completion": {"ended_by_tile": [7, 4], "ended_complete_by_tile": [2, 1]}}
    after = {"physical_request_outcomes": {"verified_success": 12},
             "request_end_counts": {"timeout": 3}, "recovery": {"counts": {"verified_physical_successes": 10}},
             "authored_stair_completion": {"ended_by_tile": [8, 4], "ended_complete_by_tile": [3, 1]}}
    evaluator = CourseEvaluation(SimpleNamespace(env=SimpleNamespace(
        _terrain_course=SimpleNamespace(metrics=lambda: after))))
    evaluator.schema, evaluator.before = schema, before
    evaluator.plan = SimpleNamespace(names=("jump", "righting"), row_names=("jump", "righting"),
                                     scenarios={name: {"gimbal": "hold"} for name in ("jump", "righting")},
                                     description=lambda: {"fixture": True})
    transition = Transition(
        snapshot([[4, 0, 8, 0, 0, 0], [5, 0, 9, 0, 0, 0]]), torch.zeros(2, 6),
        snapshot([[4, 0, 8, 0, 1, 0], [5, 0, 9, 0, 0, 1]]), torch.tensor([2., 3.]),
        torch.tensor([True, False]), torch.tensor([False, True]), torch.ones(2, dtype=torch.bool))
    evaluator.step(transition, NativeTrackingSample(
        torch.tensor([[1., 2., 3.], [-1., 0., 2.]]), torch.zeros(2, 3), {
            "translation_actual_m_s": torch.tensor([[1., 3.], [100., 100.]]),
            "translation_target_m_s": torch.tensor([[0., 1.], [0., 0.]]),
            "translation_active": torch.tensor([True, False])}))
    report = evaluator.summary()
    assert report["buckets"]["jump"]["jump_physical_success_events"] == 1
    assert report["buckets"]["righting"]["recovery_physical_success_events"] == 1
    assert report["buckets"]["jump"]["actual"]["tile"] == {"8": 1}
    assert report["buckets"]["righting"]["policy_owned_samples"] == 0
    assert report["buckets"]["jump"]["tracking"]["rmse"] == [1., 2., 3.]
    assert report["buckets"]["jump"]["gimbal"]["translation_tracking"]["mae"] == [1., 2.]
    assert report["buckets"]["jump"]["gimbal"]["translation_tracking"]["rmse"] == [1., 2.]
    assert report["buckets"]["righting"]["gimbal"]["translation_tracking"]["samples"] == 0
    assert report["buckets"]["righting"]["gimbal"]["translation_tracking"]["rmse"] is None
    delta = report["physical_counter_deltas_after_initial_reset"]
    assert delta["physical_request_outcomes"]["verified_success"] == 1
    assert delta["authored_stair_completion"]["ended_complete_by_tile"] == [1, 0]


def test_gimbal_ranges_and_tracking_only_count_active_rows():
    evaluator = CourseEvaluation(SimpleNamespace())
    bucket = {}
    values = {
        "native_position_rad": torch.tensor([[.2, -.5], [9., 8.]]),
        "native_velocity_rad_s": torch.tensor([[2., 3.], [-4., 5.]]),
        "pitch_command_rad": torch.tensor([[-.6], [.5]]),
        "translation_active": torch.tensor([True, False]),
        "translation_actual_m_s": torch.tensor([[1., 3.], [100., 100.]]),
        "translation_target_m_s": torch.tensor([[0., 1.], [0., 0.]]),
    }
    evaluator._record_gimbal(bucket, values, 0)
    evaluator._record_gimbal(bucket, values, 1)
    stats = bucket["gimbal"]
    assert stats["samples"] == 2 and stats["translation_samples"] == 1
    assert stats["translation_absolute_sum"] == [1., 2.]
    assert stats["translation_square_sum"] == [1., 4.]
    assert stats["ranges"]["native_velocity_rad_s"] == {"min": [-4., 3.], "max": [2., 5.]}
    assert stats["ranges"]["translation_actual_m_s"] == {"min": [1., 3.], "max": [1., 3.]}


def test_invalid_course_rows_do_not_become_failures_successes_or_tracking_evidence():
    schema = StateSchema(tuple(Field(name, ()) for name in (
        'course.tasks.family', 'course.terrain_tile', 'course.tasks.success_id.valid',
        'course.recovery._verified')), ())
    n = 5
    fields = torch.tensor([[4., 0., 0., 0.]]).expand(n, -1).clone()
    snapshot = Snapshot(fields, torch.zeros(n, 35), torch.zeros(n, 11), torch.empty(n, 0),
        torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.bool))
    endpoint = Snapshot(fields.clone(), snapshot.actor_observation, snapshot.goal, snapshot.context,
                        snapshot.episode_id, snapshot.control_epoch, snapshot.policy_owned)
    endpoint.state[:, 2:] = 1.
    counters = {'physical_request_outcomes': {'invalid_raw_example': 9}, 'request_end_counts': {},
                'recovery': {'counts': {}}, 'authored_stair_completion': {}}
    evaluator = CourseEvaluation(SimpleNamespace(env=SimpleNamespace(
        _terrain_course=SimpleNamespace(metrics=lambda: counters))))
    evaluator.schema, evaluator.before = schema, counters
    evaluator.plan = SimpleNamespace(names=('jump',), row_names=('jump',) * n,
                                     description=lambda: {'fixture': True})
    # Bad finite placeholder, nonfinite endpoint, and unflagged nonfinite
    # measurement are all excluded; the valid terminal sample still counts.
    transition = Transition(snapshot, torch.zeros(n, 6), endpoint,
        torch.tensor([1., 2., 100., float('nan'), 3.]),
        torch.tensor([False, True, True, True, True]), torch.zeros(n, dtype=torch.bool),
        torch.tensor([True, True, False, False, True]))
    evaluator._authored_pending = torch.ones(n, 5, dtype=torch.long)
    error = torch.tensor([[.2, .1, .02], [-3., -2., -.3], [99., 99., 99.],
                          [float('nan'), 0., 0.], [0., float('nan'), 0.]])
    context = {'command_source': torch.ones(n, dtype=torch.long),
               'routed_step_height_m': torch.tensor([.1, .2, 99., float('nan'), 99.])}
    evaluator.step(transition, NativeTrackingSample(error, torch.zeros(n, 3), {
        'translation_active': torch.ones(n, dtype=torch.bool),
        'translation_actual_m_s': error[:, :2], 'translation_target_m_s': torch.zeros(n, 2)}, course=context))
    report = evaluator.summary()
    bucket = report['buckets']['jump']
    assert report['format'] == 'supermimic-fixed-course-evaluation-v3'
    assert bucket['samples'] == 5 and bucket['valid_samples'] == 2 and bucket['invalid_count'] == 3
    assert bucket['nonfinite_measurement_count'] == 2
    assert bucket['terminated'] == 1 and bucket['reward_sum'] == 3.
    assert bucket['jump_physical_success_events'] == bucket['recovery_physical_success_events'] == 2
    assert bucket['actual']['tile'] == {'0': 2}
    assert bucket['tracking']['samples'] == 2
    assert bucket['tracking']['mae'] == pytest.approx([1.6, 1.05, .16])
    assert bucket['tracking']['rmse'] == pytest.approx([math.sqrt(4.52), math.sqrt(2.005), math.sqrt(.0452)])
    assert bucket['tracking']['clipped_mae'] == pytest.approx([.6, .45, .085])
    assert bucket['routed_terrain_ranges']['routed_step_height_m']['max'] == pytest.approx([.2])
    assert bucket['gimbal']['translation_tracking']['samples'] == 2
    assert bucket['authored_whole_course']['ended'] == 2
    assert not evaluator._authored_pending.any()
    assert bucket['command_phases']['scheduled_segment']['invalid_count'] == 3
    assert 'raw upstream' in report['native_counter_semantics']


def test_authored_events_keep_row_identity_through_finish_and_reset():
    source = pytest.importorskip("transformer_rl.formal.lab_env.stair_evidence")
    evidence = source.AuthoredStairEvidence(
        2, profiles_table=torch.tensor([[0., .2], [0., .2]]), profile_lengths=torch.tensor([2, 2]),
        last_edge_x=torch.zeros(2), roi_xy=torch.tensor([[[-1., -1.], [2., 1.]]] * 2),
        wheel_radius_m=torch.tensor([.05, .05]))
    ids = torch.arange(2)
    evidence.reset(ids, ids, torch.zeros(2, 3))
    evaluator = CourseEvaluation(SimpleNamespace(env=SimpleNamespace(_terrain_course=SimpleNamespace(stair_evidence=evidence))))
    evaluator._install_authored_evidence()
    original_observe = source.AuthoredStairEvidence.observe

    def observe(height):
        return evidence.observe(
            torch.tensor([[.4, 0., height + .32]] * 2),
            torch.tensor([[[.2, -.1, height + .05], [.6, .1, height + .05]]] * 2),
            torch.full((2,), height), torch.full((2, 2), height),
            torch.tensor([[10., 10.], [0., 0.]]), torch.zeros(2), torch.full((2,), .32),
            torch.full((2,), .32), torch.zeros(2, dtype=torch.bool), .2)

    try:
        assert not observe(0.).any()
        assert observe(.2).tolist() == [True, False]
        assert not observe(.2).any()  # Completion is a pulse, not repeated occupancy.
        evidence._finish(torch.tensor([True, True]))
        evidence.reset(torch.tensor([0]), torch.tensor([0]), torch.zeros(1, 3))
        assert evaluator._authored_pending.tolist() == [[2, 1, 1, 1, 0], [1, 1, 0, 0, 1]]
        evaluator.schema = StateSchema((Field("fixture", (1,)),), ())
        evaluator.plan = SimpleNamespace(row_names=("single", "double"))
        state = Snapshot(torch.zeros(2, 1), torch.zeros(2, 35), torch.zeros(2, 11), torch.empty(2, 0),
                         torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.bool))
        transition = Transition(state, torch.zeros(2, 6), state, torch.zeros(2), torch.zeros(2, dtype=torch.bool),
                                torch.zeros(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool))
        evaluator.step(transition, NativeTrackingSample(torch.zeros(2, 3), torch.zeros(2, 3)))
        assert evaluator.buckets["single"]["authored_whole_course"]["complete"] == 1
        assert evaluator.buckets["double"]["authored_whole_course"]["complete"] == 0
        assert evaluator.buckets["double"]["authored_whole_course"]["incomplete"] == 1
        assert evaluator._authored_pending.eq(0).all()
    finally:
        evaluator.close()
    assert evidence.observe.__func__ is original_observe
    assert not {"observe", "reset", "_finish"} & evidence.__dict__.keys()
