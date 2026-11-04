"""Native jump-first exposure order, domain limits and exact resume semantics."""
from dataclasses import fields
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

UPSTREAM = Path(os.environ.get('SUPERMIMIC_UPSTREAM_ROOT',
    str(Path(__file__).resolve().parents[2] / 'transformer_rl-worktrees/full-vehicle-gimbal'))) / 'src'
if not (UPSTREAM / 'transformer_rl/formal/lab_env/formal_curriculum.py').is_file():
    pytest.skip('Pinned native curriculum required', allow_module_level=True)
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from transformer_rl.formal.lab_env.formal_curriculum import LabTaskDrawSampler
from transformer_rl.formal.tasks.command_pool import CommandPool
from transformer_rl.formal.tasks.curriculum import TaskFamily
from supermimic_rl.course.curriculum import CourseCurriculumConfig, SuperMimicCurriculum
from supermimic_rl.course.jump_first import (
    JUMP_FIRST_STAGE_NAMES, SCHEDULE, JumpFirstCurriculum, make_course_curriculum,
    JumpFirstCommandPool, install_jump_first_command_pool,
)


def course_at(stage, consumed=50, count=256):
    course = JumpFirstCurriculum(count, seed=17,
        config=CourseCurriculumConfig(stage_transition_budgets=(100,) * 8))
    course.set_progress(stage * 100 + consumed, 0)
    return course


def test_jump_precedes_dynamic_and_terrain_with_existing_budget_preserved():
    course = JumpFirstCurriculum(1)
    assert course.stage_names == JUMP_FIRST_STAGE_NAMES
    assert course.stage_names.index('active_jump') == 2
    assert sum(course.stage_budgets) == sum(course.config.stage_transition_budgets)
    assert course.stage_budgets[2] == course.config.stage_transition_budgets[5]
    for stage in range(4):
        current = course_at(stage)
        draw = current.sample(torch.arange(current.num_envs))
        assert not (draw.family == int(TaskFamily.TERRAIN)).any()
        assert not (draw.family == int(TaskFamily.RECOVERY)).any()
        assert current.terrain_level_cap(10) == 0
        assert current.terrain_step_up_cap_m() == 0


def test_early_jump_keeps_initial_height_and_speed_domain_and_has_two_motion_phases():
    standing = course_at(2, 20)
    moving = course_at(2, 85)
    for course in (standing, moving):
        domain = course._effective_domain()
        assert domain.jump_m == course.initial_domain.jump_m
        assert domain.forward_m_s == course.initial_domain.forward_m_s
        assert domain.yaw_rad_s == course.initial_domain.yaw_rad_s
        assert domain.height_m == course.domain.height_m
        assert course.command_context().update == course.config.source_schedule_coordinates[2]
        draw = course.sample(torch.arange(course.num_envs))
        request = draw.family == int(TaskFamily.REQUEST)
        assert request.any()
        assert draw.jump_height_m[request].max() <= domain.jump_m[1] + 1e-7
        assert draw.requested_command[request, 0].abs().max() <= .5
        if course is standing:
            assert draw.requested_command[request, :3].count_nonzero() == 0
        else:
            assert all(count > 0 for count in course.sample_counts['jump_motion'])
    assert standing.command_context().request_motion_scale == 0
    assert moving.command_context().request_motion_scale == 1


def test_native_within_episode_pool_obeys_early_jump_motion_domain():
    for consumed in (20, 85):
        course = course_at(2, consumed)
        n = course.num_envs
        pool = JumpFirstCommandPool()
        draw = pool.sample(torch.full((n,), int(TaskFamily.REQUEST), dtype=torch.long),
            torch.zeros(n, dtype=torch.long), torch.full((n,), 20.), torch.ones(n),
            course.command_context(), torch.Generator().manual_seed(3))
        assert draw.command[:, 0].abs().max() <= .5
        assert draw.command[:, 2].abs().max() <= 1.
        assert not draw.heading_enabled.any()
        if consumed == 20:
            assert draw.command[:, :3].count_nonzero() == 0


def test_request_heading_wrapper_preserves_other_native_commands_rng_and_handoff():
    course = course_at(2, 20)
    family = torch.tensor([int(TaskFamily.REQUEST), int(TaskFamily.MOTION)] * 128)
    native, modified = CommandPool(), JumpFirstCommandPool()
    left_rng, right_rng = torch.Generator().manual_seed(81), torch.Generator().manual_seed(81)
    arguments = (family, torch.zeros(256, dtype=torch.long), torch.full((256,), 20.),
                 torch.ones(256), course.command_context())
    left, right = native.sample(*arguments, left_rng), modified.sample(*arguments, right_rng)
    for item in fields(left):
        if item.name != 'heading_enabled':
            assert torch.equal(getattr(left, item.name), getattr(right, item.name))
    request = family == int(TaskFamily.REQUEST)
    assert torch.equal(left.heading_enabled[~request], right.heading_enabled[~request])
    assert not right.heading_enabled[request].any()
    assert torch.equal(left_rng.get_state(), right_rng.get_state())
    for left_value, right_value in zip(native.handoff(32, left_rng, 'cpu'),
                                       modified.handoff(32, right_rng, 'cpu')):
        assert torch.equal(left_value, right_value)
    tasks = SimpleNamespace(command_pool=native)
    assert install_jump_first_command_pool(tasks, SuperMimicCurriculum(2)) is None
    assert tasks.command_pool is native
    receipt = install_jump_first_command_pool(tasks, course)
    assert isinstance(tasks.command_pool, JumpFirstCommandPool)
    assert tasks.command_pool.config == native.config
    assert receipt['request_heading_override'] is False


def test_native_task_step_does_not_restore_rotation_to_a_standing_jump():
    from transformer_rl.formal.lab_env.course_tasks import LabCourseTasks
    from transformer_rl.formal.tasks.trajectory import DriveEnvelope

    course = course_at(2, 20)
    n = course.num_envs
    tasks = LabCourseTasks(n, DriveEnvelope(.06, .34), command_pool=JumpFirstCommandPool(),
        command_context=course.command_context, policy_dt_s=.02, terrain_mode_labels=False)
    ids = torch.arange(n)
    task = LabTaskDrawSampler(course)(ids)
    tasks.reset(ids, task, torch.zeros(n, dtype=torch.float64), terrain_kind=task.terrain_kind,
                terrain_step_height_m=task.draw.step_up_m)
    request = task.draw.family == int(TaskFamily.REQUEST)
    assert request.any()
    assert not tasks.heading_enabled[request].any()
    admitted, _, _, _ = tasks.step(torch.full((n,), .02, dtype=torch.float64), None,
                                  torch.zeros(n), yaw_rad=torch.ones(n))
    assert admitted.requested[request, :3].count_nonzero() == 0


def test_native_mapper_and_complete_later_skill_coverage():
    moving = course_at(2, 85)
    mapped = LabTaskDrawSampler(moving)(torch.arange(moving.num_envs))
    assert mapped.command.shape == (moving.num_envs, 4)
    terrain = course_at(4, 50)
    assert 0 < terrain.terrain_level_cap(10) < 9
    assert .05 < terrain.terrain_step_up_cap_m() < .20
    final = course_at(7, 100)
    draw = final.sample(torch.arange(final.num_envs))
    assert final._effective_domain() == final.domain
    assert final.status == 'training_budget_complete'
    assert final.terrain_level_cap(10) == 9
    assert set(draw.family.tolist()) == set(range(len(TaskFamily)))


def test_resume_preserves_reordered_schedule_rng_and_rejects_legacy_identity():
    original = course_at(2, 85, count=16)
    original.sample(torch.arange(16))
    state = original.state_dict()
    restored = JumpFirstCurriculum(16, seed=999, config=original.config)
    restored.load_state_dict(state)
    assert restored.stage_name == 'active_jump'
    assert restored.budget.total_updates == original.budget.total_updates
    left, right = original.sample([1, 5, 9]), restored.sample([1, 5, 9])
    for item in fields(left):
        assert torch.equal(getattr(left, item.name), getattr(right, item.name)), item.name
    with pytest.raises(ValueError, match='identity'):
        SuperMimicCurriculum(16, config=original.config).load_state_dict(state)


def test_explicit_factory_selection_and_pinned_configuration():
    base = CourseCurriculumConfig().to_dict()
    assert type(make_course_curriculum(2, 'cpu', 7, base)) is SuperMimicCurriculum
    wrapper = {'schedule': SCHEDULE, 'base': base}
    assert type(make_course_curriculum(2, 'cpu', 7, wrapper)) is JumpFirstCurriculum
    with pytest.raises(ValueError, match='Unknown'):
        make_course_curriculum(2, 'cpu', 7, {**wrapper, 'schedule': 'not_implemented'})
    path = Path(__file__).resolve().parents[1] / 'configs/gpufree/ppo-course-jump-first.json'
    configured = make_course_curriculum(2, 'cpu', 7, json.loads(path.read_text()))
    original = json.loads(path.with_name('ppo-course.json').read_text())
    assert configured.config.to_dict() == {**original, 'jump_height_stage': 'combined_dr',
                                           'jump_height_gates': [], 'recovery_posture_stages': {}}


def test_jump_height_range_widens_inside_the_configured_stage():
    import json
    from supermimic_rl.course.curriculum import CourseCurriculumConfig
    from supermimic_rl.course.jump_first import JumpFirstCurriculum
    from support import REPO_ROOT

    base = json.loads((REPO_ROOT / 'configs/gpufree/ppo-course-gated-v4.json').read_text())['base']
    uppers = {}
    for stage in ('combined_dr', 'active_jump'):
        curriculum = JumpFirstCurriculum(4, 'cpu', 7, CourseCurriculumConfig.from_dict({**base, 'jump_height_stage': stage}))
        start = sum(curriculum.stage_budgets[:curriculum.stage_names.index('active_jump')])
        values = []
        for fraction in (0., .5, 1.):
            curriculum.set_progress(start + int(fraction * curriculum.stage_budgets[2]), 0)
            values.append(curriculum._effective_domain().jump_m[1])
        uppers[stage] = values
    assert uppers['combined_dr'] == [pytest.approx(.1)] * 3
    low, mid, high = uppers['active_jump']
    assert low == pytest.approx(.1) and .1 < mid < .4 and high == pytest.approx(.4)
    with pytest.raises(ValueError, match='jump_height_stage'):
        CourseCurriculumConfig.from_dict({**base, 'jump_height_stage': 'nope'})


def test_v5_height_caps_wait_for_unassisted_bin_gates_at_exact_integer_boundaries():
    from supermimic_rl.ppo.course_gate import CourseGates
    from support import REPO_ROOT

    root = REPO_ROOT / 'configs/gpufree'
    curriculum = make_course_curriculum(4, 'cpu', 7, json.loads((root / 'ppo-course-gated-v5.json').read_text()))
    starts, total = {}, 0
    for name, budget in zip(curriculum.stage_names, curriculum.stage_budgets):
        starts[name] = total
        total += budget
    gates = CourseGates(json.loads((root / 'ppo-gated5-fullturn-resume-r6.json').read_text())['course_gates'], starts)
    gates.coordinate = starts['active_jump']
    gates.opened.append('active_jump')
    curriculum.set_progress(gates.coordinate, 0)
    assert curriculum._effective_domain().jump_m == (.03, .10)
    for fraction, expected_before, expected_after in ((.25, .10, .20), (.50, .20, .30), (.75, .30, .40)):
        gate = gates._next()
        assert gate.at_fraction == fraction
        # Reach the hold with arbitrarily many transitions. Assisted success
        # cannot open a gate, and elapsed training cannot widen the height cap.
        good = {'terminations': 0}
        for criterion in gate.criteria:
            for name in criterion.numerator:
                assert '/unassisted_success_rise_' in name
                good[name] = 100
            for name in criterion.denominator:
                assert '/unassisted_ended_rise_' in name
                good[name] = 200
        assisted = {key.replace('/unassisted_', '/'): value for key, value in good.items()}
        gates.advance(assisted, 10**9)
        for _ in range(gate.window):
            status = gates.advance(assisted, 10**9)
            assert not status['gate_passed']
        curriculum.set_progress(gates.coordinate, 0)
        assert gates.coordinate == gates._position(gate) - 1
        assert curriculum._effective_domain().jump_m[1] == expected_before
        # Force a full clean evidence window, then cross by one transition.
        gates.history.clear()
        for _ in range(gate.window):
            status = gates.advance(good, 1)
        assert status['gate_passed'] and gates.coordinate == gates._position(gate)
        curriculum.set_progress(gates.coordinate, 0)
        assert curriculum._effective_domain().jump_m[1] == expected_after


def test_reported_height_ladder_follows_opened_gates_while_the_assist_fades_with_the_stage():
    from supermimic_rl.course.jump_assist import JumpAssist, JumpAssistConfig
    from support import REPO_ROOT

    root = REPO_ROOT / 'configs/gpufree'
    curriculum = make_course_curriculum(4, 'cpu', 7, json.loads((root / 'ppo-course-gated-v5.json').read_text()))
    assert curriculum.jump_height_level() is None
    assist = object.__new__(JumpAssist)
    assist.curriculum, assist.config = curriculum, JumpAssistConfig(probability=1., decay='force')
    index = curriculum.stage_names.index('active_jump')
    assist.stage_start, assist.stage_budget = sum(curriculum.stage_budgets[:index]), curriculum.stage_budgets[index]
    # Far past every height fraction and the assist stage: with reported gates
    # the cap follows the opened ladder, while the assist has faded with the stage.
    curriculum.set_progress(sum(curriculum.stage_budgets[:6]), 0)
    curriculum.set_opened_gates(['active_jump'])
    assert curriculum.jump_height_level() == 0 and curriculum._effective_domain().jump_m == (.03, .10)
    assert assist.level() == 0.
    curriculum.set_opened_gates(['active_jump', 'active_jump@0.25', 'active_jump@0.75'])  # 0.5 still closed
    assert curriculum.jump_height_level() == 1 and curriculum._effective_domain().jump_m[1] == .20
    curriculum.set_opened_gates(['active_jump@0.25', 'active_jump@0.5', 'active_jump@0.75'])
    assert curriculum._effective_domain().jump_m[1] == .40
    assert curriculum.ladder_gate_keys() == ('active_jump@0.25', 'active_jump@0.5', 'active_jump@0.75')
    assist.curriculum = make_course_curriculum(4, 'cpu', 7, json.loads((root / 'ppo-course-gated-v5.json').read_text()))
    assist.curriculum.set_progress(assist.stage_start + assist.stage_budget // 4, 0)   # a quarter into the stage
    assist.curriculum.set_opened_gates([])
    assert assist.level() == pytest.approx(2 / 3)


@pytest.mark.parametrize('height_gates', [[(.25, .20), (.20, .40)], [(.25, .05), (.50, .40)],
                                         [(.25, .20)], [(1., .40)], [(.25, float('nan'))]])
def test_invalid_discrete_height_schedule_is_rejected(height_gates):
    with pytest.raises(ValueError, match='jump_height_gates'):
        CourseCurriculumConfig(jump_height_gates=height_gates)


def test_recovery_postures_enter_by_stage_and_mix_into_the_jump_stage():
    import json
    from supermimic_rl.course.curriculum import CourseCurriculumConfig
    from supermimic_rl.course.jump_first import JumpFirstCurriculum
    from support import REPO_ROOT

    base = json.loads((REPO_ROOT / 'configs/gpufree/ppo-course-gated-v4.json').read_text())['base']
    weights = [list(row) for row in base['stage_weights']]
    weights[5] = [.05, .25, .05, .2, .4, .05]          # original active_jump row: 5% recovery
    schedule = {name: 'active_jump' for name in ('upright_release', 'crouched_release', 'pitch_forward_45')}
    schedule.update({'left_side_90': 'dynamic_commands', 'right_side_90': 'dynamic_commands'})
    config = CourseCurriculumConfig.from_dict({**base, 'stage_weights': weights,
                                               'recovery_postures': ['upright_release', 'crouched_release',
                                                                     'pitch_forward_45', 'left_side_90',
                                                                     'right_side_90', 'prone_belly_legs_above',
                                                                     'prone_back_legs_above'],
                                               'recovery_posture_stages': {
                                                   **schedule, 'prone_belly_legs_above': 'full_recovery',
                                                   'prone_back_legs_above': 'full_recovery'}})
    curriculum = JumpFirstCurriculum(4, 'cpu', 7, config)
    jump = curriculum.stage_names.index('active_jump')
    assert curriculum.stage_weights[jump][-1] > 0 and curriculum.stage_weights[jump][3] == 0.
    for stage, expected in (('active_jump', 3), ('dynamic_commands', 5), ('full_recovery', 7)):
        curriculum.set_progress(sum(curriculum.stage_budgets[:curriculum.stage_names.index(stage)]), 0)
        assert len(curriculum.recovery_posture_pool()) == expected
    with pytest.raises(ValueError, match='recovery_posture_stages'):
        CourseCurriculumConfig.from_dict({**base, 'recovery_posture_stages': {'left_side_90': 'nope'}})
