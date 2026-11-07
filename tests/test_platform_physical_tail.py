"""Selected-platform physical tails retain the existing jump objective on terrain."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from transformer_rl.formal.control.observation import CommandMode
from transformer_rl.formal.tasks.gimbal_course import GimbalRewardReference
from transformer_rl.formal.tasks.reward import TerrainRewardReference, RoughDashRewardReference
from supermimic_rl.course.jump_physics import PlatformLandingEvidence, PlatformLandingState, observe_jump_physics
from supermimic_rl.course.reward import install_shared_reward
from supermimic_rl.reward_kernel import JumpProgress, RewardHistory, RewardParameters
from supermimic_rl.world_model_v2.jump_task import JumpTask, JumpTaskConfig, JumpTaskState
from support.world_v2.jump_task import native_jumps
from test_jump_physical_tail import compare_components, evaluate, fixture
from test_reward_kernel import begin, finish, sample
from test_world_v2_jump_task import physical_ticks


def terrain_reference(reference, kind):
    yes, no = torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)
    if kind == 'rough':
        return replace(reference, rough_dash=RoughDashRewardReference(yes,
            reference.admitted_body_command[:, 3], torch.full((2, 2), .04), torch.zeros(2)))
    return replace(reference, terrain=TerrainRewardReference(yes,
        reference.admitted_body_command[:, 3], yes if kind == 'airborne' else no, no))


@pytest.mark.parametrize('kind', ['terrain', 'airborne', 'rough'])
def test_bound_platform_keeps_existing_peak_missed_and_landing_terms_on_terrain(kind):
    _, reference, physics, history, parameters, progress = fixture()
    reference = terrain_reference(reference, kind)
    if kind == 'rough':
        parameters = replace(parameters, tuning=replace(parameters.tuning, profile='rough_dash'))
    progress = replace(progress, platform_bound=torch.tensor([True, False]), landed_s=torch.full((2,), .5))
    actual = evaluate(reference, physics, history, parameters, progress)
    original = evaluate(reference, physics, history, parameters, replace(progress, platform_bound=None))
    changed = {'positive.forward', 'positive.yaw', 'positive.height', 'positive.jump_settle',
               'cost.jump_missed', 'cost.jump_landing_rate'}
    for name, value in actual.components.items():
        torch.testing.assert_close(value[1], original.components[name][1], rtol=0, atol=0)
        if name not in changed:
            torch.testing.assert_close(value, original.components[name], rtol=0, atol=0)
    for name in ('forward', 'yaw', 'height'):
        torch.testing.assert_close(actual.positive[name][0], original.positive[name][0] * .25)
    torch.testing.assert_close(actual.cost['jump_missed'], torch.tensor([.75, 0.]))
    torch.testing.assert_close(actual.positive['jump_settle'], torch.tensor([.875, 0.]))
    torch.testing.assert_close(actual.cost['jump_landing_rate'], torch.tensor([.16, 0.]), atol=1e-7, rtol=0)
    for name in ('jump_flight', 'jump_lift', 'jump_push', 'jump_vertical'):
        assert not actual.positive[name].any()


@pytest.mark.parametrize('scope', ['default', 'inactive', 'stair', 'recovery', 'gimbal', 'disabled'])
def test_unbound_ended_or_other_scopes_keep_original_components(scope):
    _, reference, physics, history, parameters, progress = fixture()
    reference = terrain_reference(reference, 'terrain')
    bound = torch.ones(2, dtype=torch.bool)
    if scope == 'default':
        bound.zero_()
    elif scope == 'inactive':
        progress = replace(progress, physical_active=torch.zeros(2, dtype=torch.bool))
    elif scope in ('stair', 'recovery'):
        reference = replace(reference, mode=torch.full((2,), int(getattr(CommandMode, scope.upper()))))
    elif scope == 'gimbal':
        reference = replace(reference, gimbal=GimbalRewardReference(bound, torch.zeros(2), torch.zeros(2), torch.zeros(2)))
    else:
        parameters = replace(parameters, extensions=replace(parameters.extensions, jump_physical_tail=False))
    compare_components(evaluate(reference, physics, history, parameters, replace(progress, platform_bound=bound)),
                       evaluate(reference, physics, history, parameters, progress))


def test_bound_platform_peak_gradient_retains_the_existing_completion_formula():
    _, reference, physics, history, parameters, progress = fixture()
    reference = terrain_reference(reference, 'terrain')
    progress = replace(progress, platform_bound=torch.tensor([True, False]))
    peak = torch.tensor([.05, .1], dtype=torch.float64, requires_grad=True)
    def score(value):
        return evaluate(reference, physics, history, parameters, replace(progress, peak_rise=value)).total.sum()
    gradient, = torch.autograd.grad(score(peak), peak)
    assert gradient[0] > 0 and gradient[1] == 0
    direction = torch.tensor([.3, -.2], dtype=torch.float64)
    numeric = (score(peak + 1e-5 * direction) - score(peak - 1e-5 * direction)) / 2e-5
    torch.testing.assert_close(gradient.dot(direction), numeric, rtol=1e-7, atol=1e-7)
    doubled = JumpProgress.cat([progress, progress])
    assert doubled.platform_bound.tolist() == [True, False, True, False]
    with pytest.raises(ValueError, match='coverage changes'):
        JumpProgress.cat([progress, replace(progress, platform_bound=None)])
    with pytest.raises(ValueError, match='boolean latches'):
        replace(progress, platform_bound=progress.platform_bound.float()).validate(reference.mode)


def test_native_and_wm_platform_tail_end_at_the_actual_payment_tick():
    native = native_jumps(2)
    tasks = native._terrain_course.tasks
    tasks.physical_id[:] = torch.tensor([7, 8])
    tasks.active_id.fill_(-1)
    tasks.active_rise.fill_(.06)
    tasks.peak_rise.fill_(.06)
    tasks.latched_height.fill_(.32)
    tasks.ground_anchor.zero_()
    for name in ('flight_seen', 'ascent_seen', 'supported_start'):
        getattr(tasks, name).fill_(True)
    tasks.landed_s[:] = torch.tensor([1.9925, 1.9875])
    yes, no = torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)
    evidence = PlatformLandingEvidence(yes, yes, no, yes[:, None].expand(-1, 2),
        yes[:, None].expand(-1, 2), no, yes, torch.full((2,), .38))
    initial_landing = PlatformLandingState(tasks.physical_id.clone(), torch.zeros(2, dtype=torch.long),
                                         torch.full((2,), 3, dtype=torch.long))
    landing = SimpleNamespace(tasks=tasks, state=initial_landing, last_evidence=evidence)
    def observe(*args, **kwargs):
        landing.state = observe_jump_physics(tasks, *args, **kwargs,
            platform_state=landing.state, platform_evidence=landing.last_evidence)
        return landing.state
    tasks.observe_physics = observe
    composer, reference, _, _, parameters, _ = fixture()
    reference = terrain_reference(reference, 'terrain')
    install_shared_reward(composer, extensions=parameters.extensions, jump_tasks=tasks, platform_landing=landing)
    tasks.supermimic_flight_latch.lifted.fill_(True)
    tasks.supermimic_flight_latch.closed.fill_(True)
    jump = JumpTaskState.capture(native)
    task = JumpTask(JumpTaskConfig.capture(native))
    interval_evidence = PlatformLandingEvidence(**{
        name: value[:, None].expand(-1, 4, *value.shape[1:]) for name, value in vars(evidence).items()})
    imagined = task.observe_interval(jump, physical=physical_ticks(2, z=.38),
        wheel_support_load=torch.full((2, 4, 2), 20.), base_com_offset=torch.zeros(2, 3),
        base_mode=torch.zeros(2, dtype=torch.long), command_height=torch.full((2,), .32),
        ground_z=torch.zeros(2, 4), retain_peak_gradient=True,
        platform_state=initial_landing, platform_evidence=interval_evidence)
    assert imagined.platform_bound.tolist() == [[True, False, False, False], [True, True, False, False]]
    assert imagined.success_credited.tolist() == [True, True]
    composer._window = None
    begin(composer, reference, torch.zeros(2, 6))
    contributions = {}
    for tick in range(4):
        tasks.observe_physics(torch.full((2, 2), 20.), torch.zeros(2), torch.zeros(2), torch.full((2,), .38),
                              angular_speed_rad_s=torch.zeros(2), dt=.005)
        physics = sample(composer, tick + 1)
        progress = JumpProgress(imagined.reference_stage[:, tick], imagined.ascent_seen[:, tick],
            imagined.flight_closed[:, tick], imagined.landed_s[:, tick], imagined.flight_started[:, tick],
            imagined.peak_rise[:, tick], imagined.requested_rise[:, tick], imagined.physical_active[:, tick],
            imagined.platform_bound[:, tick])
        history = RewardHistory(composer._window['dq'], composer._window['action_cost'],
                                composer._window['action2_cost'], composer._action_count)
        expected = evaluate(reference, physics, history, RewardParameters.from_composer(composer), progress)
        composer.observe(physics, reference)
        for name, value in expected.components.items():
            contributions.setdefault(name, []).append(value.double())
    actual = finish(composer)
    for name, values in contributions.items():
        torch.testing.assert_close(actual.component_densities[name], torch.stack(values).mean(0), rtol=1e-7, atol=1e-7)
    settle = torch.stack(contributions['positive.jump_settle'], -1)
    assert settle[0, 0] > 0 and not settle[0, 1:].any()
    assert (settle[1, :2] > 0).all() and not settle[1, 2:].any()


def test_native_progress_requires_the_same_current_request_and_selected_evidence(monkeypatch):
    from supermimic_rl.course import reward
    for case in ('matching', 'wrong_id', 'ended', 'unselected'):
        native = native_jumps(2)
        tasks = native._terrain_course.tasks
        tasks.physical_id.fill_(-1 if case == 'ended' else 7)
        tasks.active_rise.fill_(.2)
        landing = SimpleNamespace(tasks=tasks,
            state=PlatformLandingState(torch.full((2,), 99 if case == 'wrong_id' else 7),
                                       torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long)),
            last_evidence=SimpleNamespace(selected=torch.full((2,), case != 'unselected')))
        composer, reference, _, _, parameters, _ = fixture()
        install_shared_reward(composer, extensions=parameters.extensions, jump_tasks=tasks, platform_landing=landing)
        captured, original = [], reward.physical_reward_densities
        def capture(*args, **kwargs):
            captured.append(kwargs['jump_progress'].platform_bound.clone())
            return original(*args, **kwargs)
        with monkeypatch.context() as scope:
            scope.setattr(reward, 'physical_reward_densities', capture)
            composer.observe(sample(composer, 1), reference)
        assert captured[0].tolist() == [case == 'matching'] * 2


def test_integrated_terrain_reward_receives_platform_bound_ticks():
    from test_world_v2_platform_reward import inputs

    context, kwargs, _, platform, evidence = inputs()
    _, _, _, _, parameters, _ = fixture()
    context.assembly.parameters = replace(context.assembly.parameters, extensions=parameters.extensions)
    terrain = SimpleNamespace(height_at=lambda points, tile: SimpleNamespace(
        height=torch.zeros(points.shape[:-1]), valid=torch.ones(points.shape[:-1], dtype=torch.bool)))
    context = replace(context, terrain=terrain, terrain_tile=torch.ones(2),
                      terrain_kind=torch.full((2,), 3), task_family=torch.full((2,), 3))
    jump = kwargs['jump_state']
    jump.ids[:, 1] = -1  # Protocol released; the physical request remains live.
    kwargs['jump_state'] = replace(jump, flight_latch=torch.tensor([[3, 1, 1], [3, 1, 1]]))
    kwargs['goal'][:, 8], kwargs['goal'][:, 4] = 0., 1.
    result = context.integrate(**kwargs, platform_state=platform, platform_evidence=evidence)
    ordinary = context.integrate(**kwargs)
    assert result.components['positive.jump_settle'][0] > 0
    assert result.components['positive.jump_settle'][1] == 0  # Missed first landing stays missed.
    assert not ordinary.components['positive.jump_settle'].any()
    assert result.success_credited.tolist() == [True, False]
