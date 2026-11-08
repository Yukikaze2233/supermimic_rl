"""Release downward height tracking only during known pre-touchdown extension."""
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('transformer_rl.formal.tasks.reward')
from transformer_rl.formal.tasks.jump import ReferenceStage
from transformer_rl.formal.tasks.reward import CommandMode, RewardPhase, RewardReference

from supermimic_rl.course.reward import install_shared_reward
from supermimic_rl.reward_kernel import (
    JumpProgress, RewardExtensions, RewardHistory, RewardParameters,
    physical_reward_densities,
)
from test_reward_kernel import begin, sample, setup


def fixture(case='jump_takeoff'):
    composer, reference = setup(case)
    begin(composer, reference, torch.zeros(2, 6))
    physics = replace(sample(composer, 1), root_world_z_m=reference.jump_height_world_z_m + .06)
    history = RewardHistory(composer._window['dq'], torch.zeros(2), torch.zeros(2), torch.zeros(2))
    progress = JumpProgress(torch.full((2,), int(ReferenceStage.EXTENSION)), torch.ones(2, dtype=torch.bool),
                            torch.zeros(2, dtype=torch.bool))
    return physics, reference, history, RewardParameters.from_composer(composer), progress


def evaluate(inputs, *, enabled=True, progress=True):
    physics, reference, history, parameters, recorded = inputs
    parameters = replace(parameters, extensions=replace(parameters.extensions, jump_extension_height_floor=enabled))
    return physical_reward_densities(physics, reference, history, parameters,
                                    jump_progress=recorded if progress else None)


def assert_equal(actual, expected):
    for group in ('positive', 'cost', 'gates', 'errors'):
        left, right = getattr(actual, group), getattr(expected, group)
        assert left.keys() == right.keys()
        for name in right:
            torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
    torch.testing.assert_close(actual.total, expected.total, rtol=0, atol=0)
    assert torch.equal(actual.invalid, expected.invalid)


def test_default_and_explicit_false_preserve_exact_old_result_without_progress():
    assert not RewardExtensions().jump_extension_height_floor
    assert not RewardExtensions().active and not RewardExtensions().needs_jump_progress
    inputs = fixture()
    physics, reference, history, parameters, _ = inputs
    original = physical_reward_densities(physics, reference, history, parameters)
    assert_equal(evaluate(inputs, enabled=False, progress=False), original)
    assert_equal(evaluate(inputs, enabled=False), original)


@pytest.mark.parametrize('flag', [None, 0, 1, 'true'])
def test_flag_is_strict_boolean(flag):
    with pytest.raises(ValueError, match='boolean flag'):
        RewardExtensions(jump_extension_height_floor=flag)


@pytest.mark.parametrize('case', ['ordinary', 'terrain', 'gimbal', 'recovery', 'air_flight', 'air_landing'])
def test_non_jump_tasks_need_no_progress_and_keep_exact_reward(case):
    inputs = fixture(case)
    assert_equal(evaluate(inputs, progress=False), evaluate(inputs, enabled=False, progress=False))


@pytest.mark.parametrize('mode', [CommandMode.SLOPE, CommandMode.STAIR])
def test_slope_and_stair_keep_their_signed_height_objective(mode):
    physics, reference, history, parameters, progress = fixture('ordinary')
    reference = replace(reference, mode=torch.full((2,), int(mode)))
    inputs = physics, reference, history, parameters, progress
    assert_equal(evaluate(inputs, progress=False), evaluate(inputs, enabled=False, progress=False))


@pytest.mark.parametrize('stage', list(ReferenceStage))
@pytest.mark.parametrize('closed', [False, True])
def test_clock_stage_and_first_touchdown_bound_the_change(stage, closed):
    physics, reference, history, parameters, progress = fixture()
    progress = replace(progress, reference_stage=torch.full((2,), int(stage)),
                       flight_closed=torch.full((2,), closed))
    inputs = physics, reference, history, parameters, progress
    actual, baseline = evaluate(inputs), evaluate(inputs, enabled=False)
    if stage != ReferenceStage.EXTENSION or closed:
        assert_equal(actual, baseline)
    else:
        torch.testing.assert_close(actual.positive['height'], torch.ones(2), rtol=0, atol=0)
        assert not actual.cost['height'].any()
        for group in ('positive', 'cost'):
            for name, value in getattr(baseline, group).items():
                if name != 'height':
                    torch.testing.assert_close(getattr(actual, group)[name], value, rtol=0, atol=0)
        for name, value in baseline.errors.items():
            torch.testing.assert_close(actual.errors[name], value, rtol=0, atol=0)


def test_floor_is_row_local_and_does_not_infer_touchdown_from_phase():
    physics, reference, history, parameters, progress = fixture('jump_landing')
    # A temporary physical landing phase is not proof of a first real flight.
    progress = replace(progress, flight_closed=torch.tensor([False, True]))
    inputs = physics, reference, history, parameters, progress
    actual, baseline = evaluate(inputs), evaluate(inputs, enabled=False)
    assert actual.positive['height'][0] == 1 and actual.cost['height'][0] == 0
    assert actual.positive['height'][1] == baseline.positive['height'][1]
    assert actual.cost['height'][1] == baseline.cost['height'][1]
    reference = replace(reference, mode=torch.tensor([int(CommandMode.JUMP), int(CommandMode.NORMAL)]),
                        phase=torch.tensor([int(RewardPhase.TAKEOFF), int(RewardPhase.SUPPORT)]))
    inputs = physics, reference, history, parameters, replace(progress, flight_closed=torch.zeros(2, dtype=torch.bool))
    actual, baseline = evaluate(inputs), evaluate(inputs, enabled=False)
    assert actual.positive['height'][1] == baseline.positive['height'][1]
    assert actual.cost['height'][1] == baseline.cost['height'][1]


def test_unknown_or_malformed_progress_does_not_release_height():
    inputs = fixture()
    with pytest.raises(ValueError, match='explicit first-flight closure'):
        evaluate(inputs, progress=False)
    for progress in (replace(inputs[-1], flight_closed=None),
                     replace(inputs[-1], flight_closed=torch.zeros(2)),
                     replace(inputs[-1], reference_stage=torch.ones(1, dtype=torch.long))):
        with pytest.raises(ValueError, match='first-flight closure|matching int64'):
            evaluate((*inputs[:-1], progress))


@pytest.mark.parametrize('height', [float('nan'), float('inf'), -float('inf')])
def test_floor_does_not_hide_nonfinite_physical_height(height):
    physics, reference, history, parameters, progress = fixture()
    inputs = replace(physics, root_world_z_m=torch.full((2,), height)), reference, history, parameters, progress
    assert evaluate(inputs, enabled=False).invalid.all()
    assert evaluate(inputs).invalid.all()


def test_height_gradient_stops_pulling_down_but_preserves_upward_error_and_vz_gradient():
    physics, reference, history, parameters, progress = fixture()
    height = reference.jump_height_world_z_m.double() + torch.tensor([.06, -.06], dtype=torch.float64)
    height.requires_grad_()
    velocity = torch.tensor([[0., 0., -.2], [0., 0., -.3]], dtype=torch.float64, requires_grad=True)
    reference = replace(reference, jump_velocity_world_z_m_s=torch.tensor([.8, .7], dtype=torch.float64))

    def score(z, enabled):
        result = evaluate((replace(physics, root_world_z_m=z, velocity_origin_world_m_s=velocity),
                           reference, history, parameters, progress), enabled=enabled)
        return result.positive['height'] - result.cost['height'], result

    old, before = score(height, False)
    new, after = score(height, True)
    old_gradient, = torch.autograd.grad(old.sum(), height)
    new_gradient, = torch.autograd.grad(new.sum(), height)
    assert old_gradient[0] < 0 and new_gradient[0] == 0
    assert new_gradient[1] > 0 and new_gradient[1] == old_gradient[1]
    assert new[1] == old[1]
    direction = torch.tensor([.2, -.3], dtype=torch.float64)
    epsilon = 1e-6
    numeric = (score(height + epsilon * direction, True)[0].sum()
               - score(height - epsilon * direction, True)[0].sum()) / (2 * epsilon)
    torch.testing.assert_close((new_gradient * direction).sum(), numeric, rtol=1e-7, atol=1e-7)
    gradients = [torch.autograd.grad((value.positive['jump_vertical'] - value.cost['jump_vertical']).sum(),
                                    velocity)[0] for value in (before, after)]
    torch.testing.assert_close(*gradients, rtol=0, atol=0)
    assert (gradients[0][:, 2] > 0).all()
    boundary = reference.jump_height_world_z_m.double().requires_grad_()
    at_floor, _ = score(boundary, True)
    boundary_gradient, = torch.autograd.grad(at_floor.sum(), boundary)
    assert torch.isfinite(at_floor).all() and torch.isfinite(boundary_gradient).all()
    assert not boundary_gradient.any()


def task_owner(stage):
    return SimpleNamespace(jump_reference=lambda: SimpleNamespace(clock_stage=stage),
        ascent_seen=torch.ones(2, dtype=torch.bool), physical_id=torch.zeros(2, dtype=torch.long),
        air_ticks=torch.zeros(2, dtype=torch.long), landed_s=torch.zeros(2),
        observe_physics=lambda *args, **kwargs: None)


def test_native_installation_requests_latch_and_records_only_enabled_opt_in():
    enabled = RewardExtensions(jump_extension_height_floor=True)
    assert enabled.active and enabled.needs_jump_progress
    with pytest.raises(ValueError, match='native jump task owner'):
        install_shared_reward(setup('ordinary')[0], extensions=enabled)
    owner = task_owner(torch.full((2,), int(ReferenceStage.EXTENSION)))
    receipt = install_shared_reward(setup('jump_takeoff')[0], extensions=enabled, jump_tasks=owner)
    assert receipt['extensions']['jump_extension_height_floor'] is True
    assert not owner.supermimic_flight_latch.closed.any()
    for extension in (RewardExtensions(), RewardExtensions(jump_extension_height_floor=False)):
        assert 'extensions' not in install_shared_reward(setup('ordinary')[0], extensions=extension)
    for extension in (RewardExtensions(success_event=5.),
                      RewardExtensions(success_event=5., jump_extension_height_floor=False)):
        receipt = install_shared_reward(setup('ordinary')[0], extensions=extension)
        expected = asdict(extension)
        for key in ('jump_peak_completion', 'stationary_quiet', 'jump_physical_tail',
                    'jump_extension_height_floor', 'jump_flight_clearance', 'jump_tuck',
                    'stair_support_transfer', 'jump_takeoff_extension', 'jump_physical_reference',
                    'jump_tuck_endpoint_normalized'):
            expected.pop(key)
        assert receipt['extensions'] == expected


def test_four_tick_native_and_world_assembly_agree_across_first_touchdown():
    from support.world_v2.reward_assembly import setup as assembly_setup
    from supermimic_rl.world_model_v2.reward_assembly import RewardTick

    native, assembly, geometry, start, legs, memory = assembly_setup('legacy')
    extension = RewardExtensions(jump_extension_height_floor=True)
    stage = torch.full((2,), int(ReferenceStage.EXTENSION))
    owner = task_owner(stage)
    install_shared_reward(native, extensions=extension, jump_tasks=owner)
    assembly.parameters = replace(assembly.parameters, extensions=extension)
    ids, owned = torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.bool)
    mode, command = torch.full_like(ids, int(CommandMode.JUMP)), torch.tensor([[0., 0., 0., .32]]).repeat(2, 1)
    action = torch.zeros(2, 6)
    native.begin(action, start.joint_velocity[:, :6], now_s=native._time, episode_id=ids,
                 control_epoch=ids, policy_owned=owned, admitted_body_command=command, command_mode=mode)
    ticks = []
    for index in range(4):
        closed = torch.full((2,), index >= 2)
        phase = RewardPhase.FLIGHT if index < 2 else RewardPhase.LANDING
        reference = RewardReference(command, mode, torch.full_like(ids, int(phase)), start.position[:, 2] - .06,
                                    torch.ones(2), torch.zeros(2, 2))
        ticks.append(RewardTick(start, geometry(start), torch.zeros(2), torch.zeros(2, 2),
            torch.zeros(2, 21), torch.zeros(2, 2), torch.zeros(2, 3), owned, ~owned, reference,
            native._time + (index + 1) * .005, ids, ids, owned,
            jump_progress=JumpProgress(stage, owned, closed)))
    queue = torch.zeros(2, 3, 2)
    result = assembly.integrate(start, tuple(ticks), action, memory, queue, legs,
                                torch.zeros(2, 6), torch.ones(2, 6))
    target, correction = legs.decode_target(action, start.joint_position[:, :4])
    support = memory.support
    for index, tick in enumerate(ticks):
        # Exercise the native first-flight latch, rather than rewriting its output.
        owner.air_ticks.fill_(2 if index < 2 else 0)
        owner.observe_physics()
        assert owner.supermimic_flight_latch.closed.tolist() == [index >= 2] * 2
        speed = start.joint_velocity[:, 4:6]
        queue = assembly.wheels.push(queue, assembly.wheels.motor_input(
            assembly.wheels.target(action), speed, torch.ones(2, 2)))
        effort = torch.cat((legs.motor_effort(target, start.joint_position, start.joint_velocity,
                                              torch.ones(2, 6)), assembly.wheels.applied(queue, speed)), -1)
        reference, support = assembly.advance_reference(tick, support)
        native.observe(assembly.physics(tick, effort, correction), reference)
    expected = native.finish(terminated=~owned, truncated=~owned,
        physical_success_request_id=torch.full_like(ids, -1), entire_skill_policy_owned=owned)
    for name, value in expected.component_densities.items():
        torch.testing.assert_close(result.components[name], value * .02, atol=1e-12, rtol=1e-12)
    # The first two ticks receive unit height density; touchdown restores tracking.
    torch.testing.assert_close(result.components['positive.height'],
                               (1 + torch.exp(-torch.full((2,), .06).square() / .001)).double() * .01,
                               atol=1e-8, rtol=1e-6)
    assert not result.invalid.any()


@pytest.mark.parametrize('enabled', [False, True])
def test_native_peak_overshoot_still_cannot_qualify_as_physical_success(enabled):
    from support.world_v2.jump_task import native_jumps

    owner = native_jumps(2)._terrain_course.tasks
    owner.physical_id[:] = torch.tensor([7, 8])
    owner.active_rise.fill_(.2)
    owner.peak_rise[:] = torch.tensor([.2, .3])
    owner.latched_height.fill_(.32)
    owner.ground_anchor.zero_()
    owner.flight_seen.fill_(True)
    owner.ascent_seen.fill_(True)
    owner.supported_start.fill_(True)
    owner.landed_s.fill_(1.9975)
    install_shared_reward(setup('jump_landing')[0],
                          extensions=RewardExtensions(jump_extension_height_floor=enabled), jump_tasks=owner)
    owner.observe_physics(torch.full((2, 2), 20.), torch.zeros(2), torch.zeros(2), torch.full((2,), .32),
                          angular_speed_rad_s=torch.zeros(2), dt=.005)
    assert owner.success_id.tolist() == [7, -1]
    assert owner.physical_id.tolist() == [-1, 8]
    torch.testing.assert_close(owner.peak_rise, torch.tensor([.2, .3]), rtol=0, atol=0)
