"""Supported zero-command settling remains local and shares native/imagined truth."""
from dataclasses import replace

import pytest
import torch

from transformer_rl.formal.control.observation import CommandMode
from transformer_rl.formal.tasks.gimbal_course import GimbalRewardReference
from transformer_rl.formal.tasks.reward import RewardPhase, RoughDashRewardReference, TerrainRewardReference
from supermimic_rl.course.reward import install_shared_reward
from supermimic_rl.reward_kernel import (
    JumpProgress, RewardExtensions, RewardHistory, RewardParameters, physical_reward_densities,
)
from supermimic_rl.world_model_v2.jump_task import JumpTask, JumpTaskConfig, JumpTaskState
from support.world_v2.jump_task import native_jumps
from test_reward_kernel import setup, begin, sample, finish
from test_world_v2_jump_task import physical_ticks


def stationary():
    composer, reference = setup('ordinary')
    command = reference.admitted_body_command.clone()
    command[:, :3] = 0
    reference = replace(reference, admitted_body_command=command)
    begin(composer, reference, torch.zeros(2, 6))
    physics = replace(sample(composer, 1), height_above_ground_m=command[:, 3].clone(),
        gravity_body=torch.tensor([[0., 0., -1.]]).repeat(2, 1),
        angular_velocity_body_rad_s=torch.zeros(2, 3),
        body_normal_load_n=torch.tensor([[0., 5., 5., 0., 0.]]).repeat(2, 1))
    history = RewardHistory(composer._window['dq'], torch.zeros(2), torch.zeros(2), torch.ones(2))
    parameters = replace(RewardParameters.from_composer(composer), extensions=RewardExtensions(stationary_quiet=1.))
    progress = JumpProgress(torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.bool),
                            physical_active=torch.zeros(2, dtype=torch.bool))
    return composer, reference, physics, history, parameters, progress


def reward(reference, physics, history, parameters, progress):
    return physical_reward_densities(physics, reference, history, parameters, jump_progress=progress)


@pytest.mark.parametrize('case', ['moving', 'tiny_velocity', 'spin', 'lateral', 'slope', 'stair',
                                 'recovery', 'jump', 'gimbal', 'terrain', 'airborne', 'rough_landing',
                                 'one_wheel', 'unsupported', 'base_contact', 'other_contact',
                                 'contact_unavailable', 'physical_jump_tail'])
def test_only_explicit_supported_normal_standstill_earns_quiet(case):
    _, reference, physics, history, parameters, progress = stationary()
    command = reference.admitted_body_command.clone()
    if case in ('moving', 'tiny_velocity', 'spin', 'lateral'):
        axis = {'moving': 0, 'tiny_velocity': 0, 'spin': 2, 'lateral': 1}[case]
        command[:, axis] = 1e-8 if case == 'tiny_velocity' else .2
        reference = replace(reference, admitted_body_command=command)
    elif case in ('slope', 'stair', 'recovery', 'jump'):
        mode = {'slope': CommandMode.SLOPE, 'stair': CommandMode.STAIR,
                'recovery': CommandMode.RECOVERY, 'jump': CommandMode.JUMP}[case]
        phase = {'recovery': RewardPhase.RECOVERY, 'jump': RewardPhase.LANDING}.get(case, RewardPhase.SUPPORT)
        reference = replace(reference, mode=torch.full((2,), int(mode)), phase=torch.full((2,), int(phase)))
    elif case == 'gimbal':
        reference = replace(reference, gimbal=GimbalRewardReference(
            torch.ones(2, dtype=torch.bool), torch.zeros(2), torch.zeros(2), torch.zeros(2)))
    elif case in ('terrain', 'airborne'):
        reference = replace(reference, terrain=TerrainRewardReference(
            torch.full((2,), case == 'terrain'), command[:, 3], torch.full((2,), case == 'airborne'),
            torch.zeros(2, dtype=torch.bool)))
    elif case == 'rough_landing':
        parameters = replace(parameters, tuning=replace(parameters.tuning, profile='rough_dash'))
        reference = replace(reference, rough_dash=RoughDashRewardReference(
            torch.ones(2, dtype=torch.bool), command[:, 3], torch.full((2, 2), .04), torch.zeros(2)))
    elif case == 'contact_unavailable':
        physics = replace(physics, available_terms=('height', 'attitude'))
    elif case == 'physical_jump_tail':
        # Visible NORMAL does not certify that the physical jump has finished.
        progress = replace(progress, physical_active=torch.ones(2, dtype=torch.bool))
    else:
        loads = physics.body_normal_load_n.clone()
        if case == 'one_wheel':
            loads[:, 1] = 4.999
        elif case == 'unsupported':
            loads[:, 1:3] = 0
        else:
            loads[:, 0 if case == 'base_contact' else 3] = 3.001
        physics = replace(physics, body_normal_load_n=loads)
    baseline = reward(reference, physics, history,
                      replace(parameters, extensions=RewardExtensions()), progress)
    result = reward(reference, physics, history, parameters, progress)
    assert not result.invalid.any()
    assert torch.equal(result.positive['stationary_quiet'], torch.zeros(2))
    for name, value in baseline.components.items():
        torch.testing.assert_close(result.components[name], value, rtol=0, atol=0)


def test_quiet_matches_recovery_soft_height_upright_and_all_axis_scale():
    _, reference, physics, history, parameters, progress = stationary()
    assert parameters.extensions.active and parameters.extensions.needs_jump_progress
    result = reward(reference, physics, history, parameters, progress)
    torch.testing.assert_close(result.positive['stationary_quiet'], torch.ones(2))
    for axis in range(3):
        angular = torch.zeros(2, 3)
        angular[:, axis] = .5
        measured = replace(physics, angular_velocity_body_rad_s=angular)
        result = reward(reference, measured, history, parameters, progress)
        torch.testing.assert_close(result.positive['stationary_quiet'], torch.full((2,), torch.exp(torch.tensor(-1.)).item()))
        recovery_reference = replace(reference, mode=torch.full((2,), int(CommandMode.RECOVERY)),
                                     phase=torch.full((2,), int(RewardPhase.RECOVERY)))
        previous = reward(recovery_reference, measured, history,
            replace(parameters, extensions=RewardExtensions(recovery_quiet=1.)), progress)
        torch.testing.assert_close(result.positive['stationary_quiet'], previous.positive['recovery_quiet'])
    fallen = replace(physics, gravity_body=torch.tensor([[0., 0., 1.]]).repeat(2, 1))
    assert not reward(reference, fallen, history, parameters, progress).positive['stationary_quiet'].any()
    low = replace(physics, height_above_ground_m=physics.height_above_ground_m - .15)
    assert (reward(reference, low, history, parameters, progress).positive['stationary_quiet'] < 1e-9).all()
    # Reuse the existing strict harmful-contact predicate, not a new bound.
    loads = physics.body_normal_load_n.clone()
    loads[:, 0] = 3.
    torch.testing.assert_close(reward(reference, replace(physics, body_normal_load_n=loads), history,
        parameters, progress).positive['stationary_quiet'], torch.ones(2))


def test_activity_proof_is_required_and_batched_with_every_physical_tick():
    _, reference, physics, history, parameters, progress = stationary()
    for missing in (None, replace(progress, physical_active=None)):
        with pytest.raises(ValueError, match='explicit per-tick physical jump activity'):
            reward(reference, physics, history, parameters, missing)
    for invalid in (torch.zeros(2), torch.zeros(2, 1, dtype=torch.bool)):
        with pytest.raises(ValueError, match='boolean latches'):
            reward(reference, physics, history, parameters, replace(progress, physical_active=invalid))
    live = replace(progress, physical_active=torch.ones(2, dtype=torch.bool))
    combined = JumpProgress.cat((live, progress))
    assert combined.physical_active.tolist() == [True, True, False, False]
    with pytest.raises(ValueError, match='coverage changes'):
        JumpProgress.cat((progress, replace(progress, physical_active=None)))
    assert JumpProgress.cat((replace(progress, physical_active=None),) * 2).physical_active is None


def test_default_and_explicit_zero_keep_old_values_receipt_and_require_no_new_state():
    for case in ('ordinary', 'gimbal', 'terrain', 'recovery', 'jump_takeoff', 'jump_landing', 'air_landing'):
        composer, reference = setup(case)
        begin(composer, reference, torch.zeros(2, 6))
        physics, parameters = sample(composer, 1), RewardParameters.from_composer(composer)
        history = RewardHistory(composer._window['dq'], torch.zeros(2), torch.zeros(2), torch.ones(2))
        baseline = reward(reference, physics, history, parameters, None)
        explicit = reward(reference, physics, history,
                          replace(parameters, extensions=RewardExtensions(stationary_quiet=0.)), None)
        assert baseline.components.keys() == explicit.components.keys()
        for name, value in baseline.components.items():
            torch.testing.assert_close(explicit.components[name], value, rtol=0, atol=0)
        assert 'positive.stationary_quiet' not in explicit.components
    old = {'recovery_quiet': 1., 'ordinary_leg_acceleration': 1e-7}
    missing = install_shared_reward(setup('ordinary')[0], extensions=RewardExtensions(**old))
    zero = install_shared_reward(setup('ordinary')[0], extensions=RewardExtensions(**old, stationary_quiet=0.))
    assert missing == zero and 'stationary_quiet' not in missing['extensions']
    with pytest.raises(ValueError, match='native jump task owner'):
        install_shared_reward(setup('ordinary')[0], extensions=RewardExtensions(stationary_quiet=1.))
    for invalid in (-1., float('inf'), float('nan'), True):
        with pytest.raises(ValueError, match='finite and nonnegative'):
            RewardExtensions(stationary_quiet=invalid)


def test_quiet_angular_and_height_gradients_match_finite_differences():
    _, reference, physics, history, parameters, progress = stationary()
    angular = torch.tensor([[.1, -.2, .3], [0., 0., 0.]], dtype=torch.float64, requires_grad=True)
    height_error = torch.tensor([.01, -.02], dtype=torch.float64, requires_grad=True)

    def score(w, error):
        measured = replace(physics, angular_velocity_body_rad_s=w,
                           height_above_ground_m=reference.admitted_body_command[:, 3].double() + error)
        return reward(reference, measured, history, parameters, progress).positive['stationary_quiet'].sum()

    gw, gh = torch.autograd.grad(score(angular, height_error), (angular, height_error))
    assert torch.isfinite(gw).all() and (gw[0] * angular[0] < 0).all() and not gw[1].any()
    assert (gh * height_error < 0).all()
    dw, dh = torch.tensor([[.2, -.1, .3], [.1, .2, .3]], dtype=torch.float64), torch.tensor([.3, -.2], dtype=torch.float64)
    epsilon = 1e-6
    numeric = (score(angular + epsilon * dw, height_error + epsilon * dh)
               - score(angular - epsilon * dw, height_error - epsilon * dh)) / (2 * epsilon)
    torch.testing.assert_close((gw * dw).sum() + (gh * dh).sum(), numeric, rtol=1e-7, atol=1e-7)


def test_native_and_imagined_activity_agree_across_success_and_reset():
    native = native_jumps(2)
    tasks = native._terrain_course.tasks
    tasks.physical_id[:] = torch.tensor([7, -1])
    tasks.active_id.fill_(-1)  # NORMAL estimator-completed tail with physical evidence still active.
    tasks.base_mode.fill_(int(CommandMode.NORMAL))
    tasks.active_rise.fill_(.06)  # The inactive row retains its old rise: it cannot prove activity.
    tasks.peak_rise.fill_(.06)
    tasks.latched_height.fill_(.32)
    tasks.ground_anchor.zero_()
    tasks.flight_seen.fill_(True)
    tasks.ascent_seen.fill_(True)
    tasks.supported_start.fill_(True)
    tasks.landed_s[:] = torch.tensor([1.9925, 0.])
    composer, reference = setup('ordinary')
    command = reference.admitted_body_command.clone()
    command[:] = torch.tensor([0., 0., 0., .32])
    reference = replace(reference, admitted_body_command=command)
    extension = RewardExtensions(stationary_quiet=1.)
    receipt = install_shared_reward(composer, extensions=extension, jump_tasks=tasks)
    assert receipt['extensions']['stationary_quiet'] == 1.
    state = JumpTaskState.capture(native)
    task = JumpTask(JumpTaskConfig.capture(native))
    expected = []
    begin(composer, reference, torch.zeros(2, 6))
    for tick in range(1, 5):
        tasks.observe_physics(torch.full((2, 2), 5.), torch.zeros(2), torch.zeros(2), torch.full((2,), .32),
                              angular_speed_rad_s=torch.zeros(2), dt=.005)
        expected.append((tasks.physical_id >= 0).clone())
        physics = replace(sample(composer, tick), height_above_ground_m=torch.full((2,), .32),
            gravity_body=torch.tensor([[0., 0., -1.]]).repeat(2, 1), angular_velocity_body_rad_s=torch.zeros(2, 3),
            body_normal_load_n=torch.tensor([[0., 5., 5., 0., 0.]]).repeat(2, 1))
        composer.observe(physics, reference)
    native_reward = finish(composer).component_densities['positive.stationary_quiet']
    observed = task.observe_interval(state, physical=physical_ticks(2),
        wheel_support_load=torch.full((2, 4, 2), 5.), base_com_offset=torch.zeros(2, 3),
        base_mode=torch.zeros(2, dtype=torch.long), command_height=torch.full((2,), .32), ground_z=torch.zeros(2, 4))
    torch.testing.assert_close(observed.physical_active, torch.stack(expected, -1), rtol=0, atol=0)
    assert observed.physical_active.tolist() == [[True, False, False, False], [False, False, False, False]]
    torch.testing.assert_close(native_reward, torch.tensor([.75, 1.], dtype=torch.float64), rtol=0, atol=1e-7)
    assert not observed.physical_active.requires_grad
    # Native reset is the authority: a reset record needs no guessed request
    # activity from reference height/stage or a newly added replay column.
    from supermimic_rl.course.curriculum import SuperMimicCurriculum
    from transformer_rl.formal.lab_env.formal_curriculum import LabTaskDrawSampler

    tasks.physical_id[0] = 8
    ids = torch.arange(2)
    draw = LabTaskDrawSampler(SuperMimicCurriculum(2))(ids)
    tasks.reset(ids, draw, native.time, terrain_kind=draw.terrain_kind)
    reset_state = JumpTaskState.capture(native)
    after_reset = task.observe_interval(reset_state, physical=physical_ticks(2),
        wheel_support_load=torch.full((2, 4, 2), 5.), base_com_offset=torch.zeros(2, 3),
        base_mode=torch.zeros(2, dtype=torch.long), command_height=torch.full((2,), .32), ground_z=torch.zeros(2, 4))
    assert not after_reset.physical_active.any()
    assert reset_state.record().keys() == state.record().keys()


def test_imagined_reward_uses_each_tick_activity_and_keeps_motion_gradient():
    from supermimic_rl.world_model_v2.records import decode_memory, decode_physical
    from supermimic_rl.world_model_v2.reward_assembly import RewardMemory
    from support.world_v2.jump_imagination import fixture

    learner, start, _ = fixture()
    learner.imagination.reward.extensions = RewardExtensions(stationary_quiet=1.)
    state, memory = decode_physical(start['physical']), decode_memory(start['memory'])
    jump = JumpTaskState.decode(start)
    jump.ids[:, 1] = -1
    jump.ids[:, 4] = torch.tensor([7, -1])
    jump.values[:, [1, 9]] = 0.
    jump.values[:, [2, 10, 11, 12, 13]] = 1.
    jump.values[:, 5] = jump.values[:, 15] = .25
    jump.values[:, 6] = .35
    jump.values[:, 7] = 0.
    jump.values[:, 16] = 1.9925
    with torch.no_grad():
        learner.model.motion.net[-1].weight.zero_()
    action = torch.zeros(2, 6)
    prediction = learner.model.step(state, start['height'], start['scan'], memory,
        start['sim_context'], start['motor_strength'][:, 4:6], action,
        terrain_tile=start['terrain_tile'][:, 0])
    context = learner.imagination.reward.bind(start, state)
    angular = torch.zeros(2, 4, 3)
    angular[..., 1] = .2
    angular.requires_grad_()
    motion = replace(prediction.ticks.state, angular_velocity=angular)
    loads = torch.zeros_like(prediction.ticks.body_normal_load)
    loads[..., context.assembly.parameters.wheel_indices] = 20.
    prediction = replace(prediction, ticks=replace(prediction.ticks, state=motion,
        wheel_load=torch.full_like(prediction.ticks.wheel_load, 20.), body_normal_load=loads))
    reward_memory = RewardMemory.decode(torch.cat((start['reward_memory'], start['support_memory']), -1))
    result = context.integrate(state, prediction, action, reward_memory, memory, start['goal'], 0,
        jump_state=jump, base_mode=torch.zeros(2, dtype=torch.long))
    quiet = result.components['positive.stationary_quiet']
    assert quiet[1] > 0
    torch.testing.assert_close(quiet[0], .75 * quiet[1])
    assert result.success_credited.tolist() == [True, False]
    gradient, = torch.autograd.grad(quiet.sum(), angular)
    assert not gradient[0, 0].any()
    assert (gradient[0, 1:, 1] < 0).all() and (gradient[1, :, 1] < 0).all()
    assert not gradient[..., [0, 2]].any()
