"""One geometric landing cost, with unchanged qualification and legacy rewards."""
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.course.jump_physics import PlatformLandingPhase, PlatformLandingState
from supermimic_rl.course.native_platform_landing import NativePlatformLanding
from supermimic_rl.course.platform_intent import PlatformIntentState
from supermimic_rl.course.platform_target_reward import PlatformTargetProgress, platform_target_progress
from supermimic_rl.reward_kernel import JumpProgress
from test_jump_physical_tail import evaluate, fixture as reward_fixture
from test_platform_geometry import Terrain, fixture, rectangle


def target_inputs(tmp_path, *, z=.12, phase=1, dtype=torch.float64):
    geometry = fixture(tmp_path, rectangle(0, -2, 2, 2))
    centres = torch.tensor([[[-.1, .3, z + .06], [-.2, -.3, z + .06]]], dtype=dtype).repeat(2, 1, 1)
    return geometry, dict(centres_local_m=centres, bottoms_local_z_m=centres[..., 2] - .06,
        radii_m=centres.new_tensor([.06, .06]), terrain_tile=torch.zeros(2, dtype=torch.long),
        target_index=torch.zeros(2, dtype=torch.long), requested_rise_m=centres.new_full((2,), .2),
        bound=torch.ones(2, dtype=torch.bool), landing_phase=torch.full((2,), phase, dtype=torch.long))


def test_distance_formula_uses_source_dimensions_and_existing_penetration_allowance(tmp_path):
    from transformer_rl.formal.tasks.skills import SkillEvidenceConfig
    geometry, kwargs = target_inputs(tmp_path)
    result = platform_target_progress(geometry, **kwargs)
    delta = .2 - .12 - SkillEvidenceConfig().support_penetration_max_m
    squared = torch.tensor([.1**2 + delta**2, .2**2 + delta**2], dtype=torch.float64) / .2**2
    torch.testing.assert_close(result.cost, (squared / (1 + squared)).mean().expand(2))
    assert result.eligible.all() and result.available.all()
    torch.testing.assert_close(result.horizontal_distance_m, torch.tensor([[.1, .2]]).double().expand(2, -1))
    smaller = platform_target_progress(geometry, **{**kwargs, 'requested_rise_m': torch.full((2,), .01)})
    squared = squared * .2**2 / .06**2
    torch.testing.assert_close(smaller.cost, (squared / (1 + squared)).mean().expand(2))


def test_horizontal_and_upward_gradients_match_finite_differences(tmp_path):
    geometry, kwargs = target_inputs(tmp_path)
    centres = kwargs['centres_local_m'].clone().requires_grad_()
    def score(value):
        return -platform_target_progress(geometry, **{**kwargs,
            'centres_local_m': value, 'bottoms_local_z_m': value[..., 2] - .06}).cost.sum()
    grad, = torch.autograd.grad(score(centres), centres)
    assert (grad[..., 0] > 0).all() and (grad[..., 2] > 0).all()
    assert not grad[..., 1].any() and torch.isfinite(grad).all()
    direction = torch.tensor([[[.3, .4, -.2], [.1, -.7, .5]]], dtype=torch.float64).expand_as(centres)
    numeric = (score(centres + 1e-5 * direction) - score(centres - 1e-5 * direction)) / 2e-5
    torch.testing.assert_close((grad * direction).sum(), numeric, atol=1e-8, rtol=1e-7)


def test_airborne_above_platform_has_no_downward_height_gradient(tmp_path):
    geometry, kwargs = target_inputs(tmp_path, z=.4)
    centres = kwargs['centres_local_m'].clone().requires_grad_()
    result = platform_target_progress(geometry, **{**kwargs, 'centres_local_m': centres,
                                                'bottoms_local_z_m': centres[..., 2] - .06})
    grad, = torch.autograd.grad(-result.cost.sum(), centres)
    assert (grad[..., 0] > 0).all() and not grad[..., 2].any()
    assert not result.vertical_shortfall_m.any()


@pytest.mark.parametrize('phase', [0, 3, 4, 5])
def test_other_phases_do_not_query_or_pull_back_to_target(tmp_path, monkeypatch, phase):
    geometry, kwargs = target_inputs(tmp_path, phase=phase)
    def forbidden(*args, **kwargs):
        raise AssertionError('Non-approach phase queried target geometry')
    monkeypatch.setattr(geometry, 'nearest', forbidden)
    result = platform_target_progress(geometry, **kwargs)
    assert not result.cost.any() and not result.eligible.any() and not result.available.any()
    assert torch.isnan(result.horizontal_distance_m).all()


def test_unknown_target_not_a_zero_distance_success_and_mixed_gradients_finite(tmp_path):
    geometry, kwargs = target_inputs(tmp_path)
    centres = kwargs['centres_local_m'].clone().requires_grad_()
    # Tile 1 is not the selected target's tile. Only row 0 has target geometry.
    result = platform_target_progress(geometry, **{**kwargs, 'centres_local_m': centres,
        'bottoms_local_z_m': centres[..., 2] - .06, 'terrain_tile': torch.tensor([0, 1])})
    assert result.eligible.all() and result.available.tolist() == [True, False]
    assert result.cost[0] > 0 and result.cost[1] == 1
    assert torch.isnan(result.horizontal_distance_m[1]).all()
    grad, = torch.autograd.grad(result.cost.sum(), centres)
    assert torch.isfinite(grad).all() and not grad[1].any()
    hidden = fixture(tmp_path, rectangle(0, -2, 2, 2), terrain=Terrain(.4))
    obscured = platform_target_progress(hidden, **kwargs)
    visible = platform_target_progress(geometry, **kwargs)
    assert not obscured.available.any() and (obscured.cost == 1).all()
    assert (obscured.cost >= visible.cost).all()
    _, ref, physics, history, parameters, progress = reward_fixture()
    density = evaluate(ref, physics, history, parameters, replace(progress,
        platform_target=obscured, platform_bound=torch.ones(2, dtype=torch.bool)))
    assert (density.cost['jump_target'] == parameters.extensions.jump_missed).all()


def test_exact_hole_geometry_has_nonzero_distance_inside_bounding_rectangle(tmp_path):
    geometry = fixture(tmp_path, rectangle(0, 0, 3, 1) + rectangle(0, 2, 3, 3)
        + rectangle(0, 1, 1, 2) + rectangle(2, 1, 3, 2))
    centres = torch.tensor([[[1.4, 1.5, .3], [1.5, 1.4, .3]]])
    result = platform_target_progress(geometry, centres_local_m=centres,
        bottoms_local_z_m=torch.full((1, 2), .24), radii_m=torch.tensor([.06, .06]),
        terrain_tile=torch.zeros(1, dtype=torch.long), target_index=torch.zeros(1, dtype=torch.long),
        requested_rise_m=torch.tensor([.2]), bound=torch.ones(1, dtype=torch.bool), landing_phase=torch.tensor([1]))
    assert result.available.all() and result.cost.item() > .7
    torch.testing.assert_close(result.horizontal_distance_m, torch.full((1, 2), .4))


def test_kernel_keeps_low_peak_and_lateral_gradient_independent(tmp_path):
    geometry, kwargs = target_inputs(tmp_path, z=.3)
    centres = kwargs['centres_local_m'].clone().requires_grad_()
    target = platform_target_progress(geometry, **{**kwargs, 'centres_local_m': centres,
                                                 'bottoms_local_z_m': centres[..., 2] - .06})
    _, reference, physics, history, parameters, progress = reward_fixture()
    progress = replace(progress, platform_bound=torch.ones(2, dtype=torch.bool),
                       peak_rise=torch.zeros(2), platform_target=target)
    result = evaluate(reference, physics, history, parameters, progress)
    original = evaluate(reference, physics, history, parameters, replace(progress, platform_target=None))
    for name, value in original.components.items():
        torch.testing.assert_close(result.components[name], value, rtol=0, atol=0)
    torch.testing.assert_close(result.cost['jump_target'], target.cost * parameters.extensions.jump_missed)
    assert (result.cost['jump_missed'] == parameters.extensions.jump_missed).all()
    assert ((result.cost['jump_missed'] + result.cost['jump_target']) < 2 * parameters.extensions.jump_missed).all()
    grad, = torch.autograd.grad(result.total.sum(), centres)
    assert (grad[..., 0] > 0).all() and torch.isfinite(grad).all()
    inactive = evaluate(reference, physics, history, parameters,
                        replace(progress, platform_bound=torch.zeros(2, dtype=torch.bool)))
    assert not inactive.cost['jump_target'].any()
    ended = evaluate(reference, physics, history, parameters,
                     replace(progress, physical_active=torch.zeros(2, dtype=torch.bool)))
    assert not ended.cost['jump_target'].any()
    joined = JumpProgress.cat([progress, progress])
    torch.testing.assert_close(joined.platform_target.cost, target.cost.repeat(2))
    with pytest.raises(ValueError, match='coverage changes'):
        JumpProgress.cat([progress, replace(progress, platform_target=None)])


def test_native_actual_pose_cache_and_wm_same_pose_have_identical_target_density(tmp_path, monkeypatch):
    from test_native_platform_clearance_integration import integrated_step
    native, capture, builder, selected, _ = integrated_step(tmp_path)
    capture.capture()
    def forbidden(*args, **kwargs):
        raise AssertionError('Reward must not read SDK pose or query whole-cylinder gaps again')
    monkeypatch.setattr(native, '_as_torch', forbidden)
    monkeypatch.setattr(capture.query, 'at', forbidden)
    pose = capture.capture_pose()
    intent = replace(PlatformIntentState.inactive(3), target_index=torch.tensor([0, 0, -1]),
                     bound_request_id=torch.tensor([7, 7, -1]))
    observer = object.__new__(NativePlatformLanding)
    observer.geometry, observer.wheel_pose_capture = builder.geometry, capture.capture_pose
    observer.intent_capture = lambda: intent
    observer.tasks = SimpleNamespace(active_rise=torch.full((3,), .2))
    observer.state = PlatformLandingState(torch.tensor([7, 7, -1]), torch.zeros(3, dtype=torch.long),
                                         torch.tensor([1, 2, 0]))
    actual = observer.target_progress(selected)
    predicted = platform_target_progress(builder.geometry, centres_local_m=pose.centres_local_m,
        bottoms_local_z_m=pose.bottoms_local_z_m, radii_m=pose.radii_m, terrain_tile=pose.terrain_tile,
        target_index=intent.target_index, requested_rise_m=observer.tasks.active_rise,
        bound=selected, landing_phase=observer.state.phase)
    for field in fields(actual):
        torch.testing.assert_close(getattr(actual, field.name), getattr(predicted, field.name),
                                   rtol=0, atol=0, equal_nan=True)
    _, ref, physics, history, parameters, progress = reward_fixture()
    def first_two(item):
        return PlatformTargetProgress(**{field.name: getattr(item, field.name)[:2] for field in fields(item)})
    native_density = evaluate(ref, physics, history, parameters, replace(progress,
        platform_bound=torch.ones(2, dtype=torch.bool), platform_target=first_two(actual)))
    wm_density = evaluate(ref, physics, history, parameters, replace(progress,
        platform_bound=torch.ones(2, dtype=torch.bool), platform_target=first_two(predicted)))
    torch.testing.assert_close(native_density.cost['jump_target'], wm_density.cost['jump_target'], rtol=0, atol=0)
    assert native_density.cost['jump_target'].gt(0).all()  # Rim touch is not centre membership.
    native._episode_id[0] += 1
    with pytest.raises(RuntimeError, match='current observed cylinder'):
        capture.capture_pose()


def test_four_tick_flight_single_both_path_stops_exactly_at_both_and_keeps_native_parity(tmp_path, monkeypatch):
    from supermimic_rl.course.jump_physics import PlatformLandingEvidence, observe_jump_physics
    from test_world_v2_platform_reward import inputs

    context, kwargs, native, landing, _ = inputs()
    geometry = fixture(tmp_path, rectangle(0, -2, 2, 2), terrain=Terrain(.2))
    _, _, _, _, parameters, _ = reward_fixture()
    context.assembly.parameters = replace(context.assembly.parameters, extensions=parameters.extensions)
    context = replace(context, terrain=geometry.terrain, terrain_tile=torch.zeros(2),
                      terrain_kind=torch.ones(2), task_family=torch.full((2,), 3))
    ticks = kwargs['prediction'].ticks
    wheel = context.geometry(ticks.state, passive_knee_position=ticks.passive_knee_position,
                             passive_knee_velocity=ticks.passive_knee_velocity)
    pos = ticks.state.position.clone()
    pos[..., 0] += -.1 - wheel.centre_world[..., 0, 0]
    ticks = replace(ticks, state=replace(ticks.state, position=pos), wheel_load=torch.tensor(
        [[[0., 0.], [20., 0.], [20., 20.], [0., 0.]]]).expand(2, -1, -1))
    kwargs['prediction'] = SimpleNamespace(ticks=ticks)
    kwargs['jump_state'] = replace(kwargs['jump_state'], flight_latch=torch.tensor([[3, 1, 0], [3, 1, 0]]))
    landing = replace(landing, phase=torch.ones(2, dtype=torch.long))
    yes = torch.ones(2, 4, dtype=torch.bool)
    no = torch.zeros_like(yes)
    touch = ticks.wheel_load > 2
    supported = (ticks.wheel_load >= 5).all(-1)
    evidence = PlatformLandingEvidence(yes, yes, ~touch.any(-1), touch, touch, no, supported,
                                      torch.full((2, 4), .52))
    intent = replace(PlatformIntentState.inactive(2), target_index=torch.zeros(2, dtype=torch.long),
                     bound_request_id=torch.full((2,), 3, dtype=torch.long))
    phases, progresses = [], []
    observe = context.jump_task.observe_interval
    def capture_phase(*args, **options):
        result = observe(*args, **options)
        phases.append(result.platform_phase)
        return result
    integrate = context.assembly.integrate
    def capture_progress(start, samples, *args, **options):
        progresses.extend(tick.jump_progress for tick in samples)
        return integrate(start, samples, *args, **options)
    monkeypatch.setattr(context.jump_task, 'observe_interval', capture_phase)
    monkeypatch.setattr(context.assembly, 'integrate', capture_progress)
    result = context.integrate(**kwargs, platform_state=landing, platform_evidence=evidence,
                               platform_intent=intent, platform_geometry=geometry)
    assert phases[0].tolist() == [[1, 2, 3, 3]] * 2
    assert [p.platform_target.eligible.tolist() for p in progresses] == [[True, True]] * 2 + [[False, False]] * 2
    assert all(p.platform_target.cost.gt(0).all() for p in progresses[:2])
    assert all(not p.platform_target.cost.any() for p in progresses[2:])
    assert result.components['cost.jump_target'].lt(0).all()
    expected = -sum(p.platform_target.cost for p in progresses).double() * .005 * parameters.extensions.jump_missed
    torch.testing.assert_close(result.components['cost.jump_target'], expected, atol=1e-9, rtol=1e-6)

    # Same explicit wheel poses and real physical evidence, but native uses
    # world Z and its unique updater. Compare each after-observe target cost.
    owner = native._terrain_course.tasks
    pose = context.geometry(ticks.state, passive_knee_position=ticks.passive_knee_position,
                            passive_knee_velocity=ticks.passive_knee_velocity)
    native_landing = landing
    for tick in range(4):
        actual_evidence = evidence.tick(tick)
        actual_evidence = replace(actual_evidence, landing_world_height=actual_evidence.landing_world_height + 1.125)
        native_landing = observe_jump_physics(owner, ticks.wheel_load[:, tick], torch.zeros(2), torch.zeros(2),
            ticks.state.position[:, tick, 2] + 1.125, angular_speed_rad_s=torch.zeros(2),
            platform_state=native_landing, platform_evidence=actual_evidence)
        torch.testing.assert_close(native_landing.phase, phases[0][:, tick], rtol=0, atol=0)
        current = platform_target_progress(geometry, centres_local_m=pose.centre_world[:, tick],
            bottoms_local_z_m=pose.bottom_world_z[:, tick], radii_m=pos.new_tensor([s.radius for s in context.geometry.sides]),
            terrain_tile=torch.zeros(2, dtype=torch.long), target_index=intent.target_index,
            requested_rise_m=owner.active_rise,
            bound=(owner.physical_id >= 0) & (native_landing.request_id == owner.physical_id),
            landing_phase=native_landing.phase)
        torch.testing.assert_close(current.cost, progresses[tick].platform_target.cost, rtol=0, atol=0)


def test_one_occluded_wheel_keeps_other_wheel_gradient_and_cannot_lower_cost(tmp_path):
    # Left wheel y>0 is occluded by a higher actual surface; right stays visible.
    atlas = Terrain(lambda points: torch.where(points[..., 1] > 0, .4, .2))
    geometry = fixture(tmp_path, rectangle(0, -2, 2, 2), terrain=atlas)
    centres = torch.tensor([[[-.1, .3, .4], [-.1, -.3, .4]]], dtype=torch.float64, requires_grad=True)
    kwargs = dict(centres_local_m=centres, bottoms_local_z_m=centres[..., 2] - .06,
        radii_m=centres.new_tensor([.06, .06]), terrain_tile=torch.tensor([0]), target_index=torch.tensor([0]),
        requested_rise_m=centres.new_tensor([.2]), bound=torch.tensor([True]), landing_phase=torch.tensor([1]))
    hidden = platform_target_progress(geometry, **kwargs)
    assert not hidden.available.any()
    assert torch.isnan(hidden.horizontal_distance_m[0, 0]) and torch.isfinite(hidden.horizontal_distance_m[0, 1])
    torch.testing.assert_close(hidden.cost, centres.new_tensor([.6]))  # mean(1, 0.2)
    grad, = torch.autograd.grad(-hidden.cost.sum(), centres)
    assert torch.isfinite(grad).all() and not grad[0, 0].any() and grad[0, 1, 0] > 0
    atlas.height = .2
    visible = platform_target_progress(geometry, **kwargs)
    assert visible.available.all() and (hidden.cost >= visible.cost).all()


def test_native_composer_integrates_the_same_per_tick_target_cost(tmp_path):
    from supermimic_rl.course.reward import install_shared_reward
    from support.world_v2.jump_task import native_jumps
    from test_reward_kernel import sample, finish

    geometry, kwargs = target_inputs(tmp_path)
    composer, reference, _, _, parameters, _ = reward_fixture()
    tasks = native_jumps(2)._terrain_course.tasks
    tasks.physical_id.fill_(7)
    tasks.active_rise.fill_(.2)
    observer = object.__new__(NativePlatformLanding)
    observer.tasks, observer.geometry = tasks, geometry
    observer.last_evidence = SimpleNamespace(selected=torch.ones(2, dtype=torch.bool))
    observer.intent_capture = lambda: replace(PlatformIntentState.inactive(2),
        target_index=torch.zeros(2, dtype=torch.long), bound_request_id=torch.full((2,), 7, dtype=torch.long))
    observer.wheel_pose_capture = lambda: SimpleNamespace(
        centres_local_m=kwargs['centres_local_m'], bottoms_local_z_m=kwargs['bottoms_local_z_m'],
        radii_m=kwargs['radii_m'], terrain_tile=kwargs['terrain_tile'])
    install_shared_reward(composer, extensions=parameters.extensions, jump_tasks=tasks, platform_landing=observer)
    expected = []
    for tick, phase in enumerate((1, 2, 3, 3)):
        observer.state = PlatformLandingState(torch.full((2,), 7), torch.zeros(2, dtype=torch.long),
                                             torch.full((2,), phase, dtype=torch.long))
        expected.append(platform_target_progress(geometry, **{**kwargs,
            'landing_phase': observer.state.phase}).cost * parameters.extensions.jump_missed)
        composer.observe(sample(composer, tick + 1), reference)
    result = finish(composer)
    torch.testing.assert_close(result.component_densities['cost.jump_target'],
                               -torch.stack(expected).mean(0), atol=1e-7, rtol=1e-7)
