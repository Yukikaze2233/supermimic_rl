"""Shared reward parity against the frozen native composer and real gradient checks."""
from copy import deepcopy
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('transformer_rl.formal.tasks.reward')
from transformer_rl.formal.control.observation import MechanicalBinding, CommandMode
from transformer_rl.formal.tasks.reward import (
    PhysicalRewardComposer, RewardPhysics, RewardReference, RewardTuning, RewardPhase,
    RoughDashRewardReference, TerrainRewardReference,
)
from transformer_rl.formal.tasks.gimbal_course import GimbalRewardReference
from supermimic_rl.course.reward import install_shared_reward
from supermimic_rl.reward_kernel import RewardHistory, RewardParameters, physical_reward_densities


def setup(case):
    joints = ('L_joint1', 'LL_joint1', 'R_joint1', 'RR_joint1', 'L_joint3', 'R_joint3')
    binding = MechanicalBinding(joints, ('base_link', 'L_link3', 'R_link3', 'M_link1', 'M_link2'),
                                joints, tuple(range(6)), tuple(range(6)), (1, 2), (4, 5), (0.,) * 6)
    tuning = RewardTuning(21.44, policy_dt_s=.02, profile='rough_dash' if case.startswith('air') else 'legacy')
    composer = PhysicalRewardComposer(binding, 2, tuning)
    composer.reset(torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.float64),
                   torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long))
    mode = CommandMode.RECOVERY if case == 'recovery' else CommandMode.JUMP if case.startswith('jump') else CommandMode.NORMAL
    phase = {'jump_takeoff': RewardPhase.TAKEOFF, 'jump_flight': RewardPhase.FLIGHT,
             'jump_landing': RewardPhase.LANDING, 'recovery': RewardPhase.RECOVERY}.get(case, RewardPhase.SUPPORT)
    reference = RewardReference(torch.tensor([[0., 0., 0., .32], [.5, 0., -.5, .3]]),
                                torch.full((2,), int(mode)), torch.full((2,), int(phase)),
                                torch.tensor([.45, .38]), torch.tensor([.2, -.1]), torch.zeros(2, 2))
    if case == 'gimbal':
        reference = replace(reference, gimbal=GimbalRewardReference(
            torch.tensor([True, False]), torch.tensor([.7, -.4]), torch.tensor([.5, 0.]), torch.tensor([1., 0.])))
    if case == 'terrain':
        reference = replace(reference, terrain=TerrainRewardReference(
            torch.tensor([True, False]), torch.tensor([.31, .29]), torch.tensor([True, False]),
            torch.zeros(2, dtype=torch.bool)), intentional_base_contact=torch.tensor([True, False]))
    if case.startswith('air'):
        reference = replace(reference, rough_dash=RoughDashRewardReference(
            torch.tensor([True, False]), torch.tensor([.32, .3]),
            torch.full((2, 2), .03 if case == 'air_landing' else 0.), torch.zeros(2)))
    return composer, reference


def begin(composer, reference, action):
    composer.begin(action, composer._last_dq, now_s=composer._time,
                   episode_id=composer._episode, control_epoch=composer._control,
                   policy_owned=torch.ones(2, dtype=torch.bool),
                   admitted_body_command=reference.admitted_body_command, command_mode=reference.mode)


def sample(composer, tick):
    generator = torch.Generator().manual_seed(tick)
    rand = lambda *shape: torch.rand(*shape, generator=generator)
    window = composer._window
    gravity = torch.tensor([[.15, .1, -1.], [-.2, -.1, -1.]])
    gravity /= gravity.norm(dim=-1, keepdim=True)
    return RewardPhysics(
        window['start'] + tick * .005, window['episode'], window['control'], window['owned'],
        rand(2, 3), rand(2, 3), rand(2) + .3, rand(2) * .1 + .26,
        rand(2, 2), gravity, rand(2, 3), rand(2, 6), rand(2, 6),
        torch.tensor([[4., 150., 200., 0., 0.], [0., 30., 40., 4., 0.]]),
        rand(2, 2), rand(2, 6) * .02,
        wheel_position_base_m=torch.tensor([[[.1, .2, -.26], [0., -.2, -.24]]]).repeat(2, 1, 1),
        wheel_forward_velocity_rad_s=rand(2, 2) * 12, wheel_contact_force_n=rand(2, 2) * 300,
        passive_knee_position_rad=rand(2, 2), passive_knee_velocity_rad_s=rand(2, 2) * 15,
        passive_knee_limits_rad=torch.tensor([[0., 1.], [0., 1.]]))


def finish(composer, failed=False):
    return composer.finish(terminated=torch.tensor([failed, False]), truncated=torch.zeros(2, dtype=torch.bool),
                           physical_success_request_id=torch.full((2,), -1),
                           entire_skill_policy_owned=torch.ones(2, dtype=torch.bool))


@pytest.mark.parametrize('case', ['ordinary', 'terrain', 'gimbal', 'jump_takeoff', 'jump_flight',
                                 'jump_landing', 'recovery', 'air_flight', 'air_landing'])
def test_shared_composer_matches_native_terms_events_and_telemetry(case):
    original, reference = setup(case)
    shared = deepcopy(original)
    install_shared_reward(shared)
    for interval in range(3):
        action = torch.full((2, 6), .1 * interval)
        begin(original, reference, action)
        begin(shared, reference, action)
        for tick in range(1, 5):
            physics = sample(original, tick)
            original.observe(physics, reference)
            shared.observe(physics, reference)
        expected, actual = finish(original, interval == 2), finish(shared, interval == 2)
        for field in fields(expected):
            left, right = getattr(expected, field.name), getattr(actual, field.name)
            if isinstance(left, dict):
                assert left.keys() == right.keys()
                for name in left:
                    torch.testing.assert_close(right[name], left[name], atol=0, rtol=0)
            else:
                torch.testing.assert_close(right, left, atol=0, rtol=0)


def test_native_shadow_audit_tracks_each_physical_contribution():
    composer, reference = setup('ordinary')
    receipt = install_shared_reward(composer, verify_native=True)
    begin(composer, reference, torch.zeros(2, 6))
    for tick in range(1, 5):
        composer.observe(sample(composer, tick), reference)
    report = composer.shared_kernel_audit.summary()
    assert report['physical_samples'] == 8 and report['max_abs'] < 1e-12
    assert receipt['implementation'] == 'native_with_shared_density_verification'


@pytest.mark.parametrize('case', ['ordinary', 'gimbal', 'jump_flight', 'recovery', 'air_landing'])
def test_reward_gradients_are_finite_and_match_directional_difference(case):
    composer, reference = setup(case)
    begin(composer, reference, torch.zeros(2, 6))
    physics = sample(composer, 1)
    # A stopped row exercises the undefined atan2(0,0) path of the native gimbal diagnostics.
    velocity = torch.tensor([[0., 0., .1], [.4, -.2, .3]], dtype=torch.float64, requires_grad=True)
    effort = torch.full((2, 6), .4, dtype=torch.float64, requires_grad=True)
    parameters = RewardParameters.from_composer(composer)
    history = RewardHistory(composer._window['dq'], torch.zeros(2), torch.zeros(2), torch.ones(2))

    def score(v, torque):
        value = replace(physics, velocity_origin_body_m_s=v, velocity_origin_world_m_s=v,
                        active_explicit_effort_nm=torque)
        return physical_reward_densities(value, reference, history, parameters).total.sum()

    gradients = torch.autograd.grad(score(velocity, effort), (velocity, effort))
    assert all(torch.isfinite(g).all() for g in gradients)
    # Avoid a finite-difference jump from zero speed into a valid heading objective.
    direction_v = torch.tensor([[0., 0., .1], [.1, .2, .3]], dtype=torch.float64)
    direction_t = torch.full_like(effort, .2)
    analytic = (gradients[0] * direction_v).sum() + (gradients[1] * direction_t).sum()
    epsilon = 1e-5
    numeric = (score(velocity + epsilon * direction_v, effort + epsilon * direction_t)
               - score(velocity - epsilon * direction_v, effort - epsilon * direction_t)) / (2 * epsilon)
    torch.testing.assert_close(analytic, numeric, atol=1e-6, rtol=1e-5)


def _jump_densities(case, extensions, progress=None):
    composer, reference = setup(case)
    install_shared_reward(composer, extensions=extensions, jump_tasks=_task_owner(lambda: None))
    begin(composer, reference, torch.zeros(2, 6))
    window = composer._window
    history = RewardHistory(window['dq'], window['action_cost'], window['action2_cost'], composer._action_count)
    physics = sample(composer, 1)
    physics = replace(physics, velocity_origin_world_m_s=torch.tensor([[0., 0., 1.5], [0., 0., -.5]]))
    return composer, physical_reward_densities(physics, reference, history, RewardParameters.from_composer(composer),
                                               jump_progress=progress)


def _progress(stage_name, ascent, closed=(False, False)):
    from transformer_rl.formal.tasks.jump import ReferenceStage
    from supermimic_rl.reward_kernel import JumpProgress

    return JumpProgress(torch.full((2,), int(getattr(ReferenceStage, stage_name))), torch.tensor(ascent),
                        None if closed is None else torch.tensor(closed))


def test_jump_extensions_pay_only_the_first_real_flight_and_are_off_by_default():
    from supermimic_rl.reward_kernel import RewardExtensions

    extensions = RewardExtensions(jump_flight=4., jump_lift=2., success_event=5.)
    _, flight = _jump_densities('jump_flight', extensions, _progress('BALLISTIC', [True, True]))
    torch.testing.assert_close(flight.positive['jump_flight'], torch.full((2,), 4.))
    torch.testing.assert_close(flight.positive['jump_lift'], torch.tensor([3., 0.]))
    assert not flight.invalid.any()
    # A real jump later than the reference still counts (SCUT: physical takeoff opens the window).
    _, late = _jump_densities('jump_flight', extensions, _progress('HOLD', [True, True]))
    torch.testing.assert_close(late.positive['jump_flight'], torch.full((2,), 4.))
    # Wheel unloading without real ascent earns nothing.
    _, unrisen = _jump_densities('jump_flight', extensions, _progress('COMPRESSION', [False, True]))
    torch.testing.assert_close(unrisen.positive['jump_flight'], torch.tensor([0., 4.]))
    # After the first touchdown, bouncing earns nothing.
    _, bounce = _jump_densities('jump_flight', extensions, _progress('HOLD', [True, True], (True, False)))
    torch.testing.assert_close(bounce.positive['jump_flight'], torch.tensor([0., 4.]))
    torch.testing.assert_close(bounce.positive['jump_lift'], torch.tensor([0., 0.]))
    _, takeoff = _jump_densities('jump_takeoff', extensions, _progress('EXTENSION', [True, True]))
    assert takeoff.positive['jump_flight'].abs().sum() == 0 and takeoff.positive['jump_lift'].abs().sum() == 0
    with pytest.raises(ValueError, match='request progress'):
        _jump_densities('jump_flight', extensions)
    with pytest.raises(ValueError, match='first-flight latch'):
        _jump_densities('jump_flight', extensions, _progress('BALLISTIC', [True, True], None))
    composer, default = _jump_densities('jump_flight', RewardExtensions())
    assert 'jump_flight' not in default.positive and composer.tuning.success_event == 1.


def test_first_flight_latch_needs_real_ascent_and_closes_at_touchdown_per_request():
    from supermimic_rl.course.reward import FirstFlightLatch

    tasks = SimpleNamespace(physical_id=torch.tensor([3, 3]), ascent_seen=torch.tensor([False, True]),
                            air_ticks=torch.tensor([2, 1]))
    latch = FirstFlightLatch(tasks)
    latch.observe(tasks)  # row 0 unloaded without ascent; row 1 not yet two airborne ticks
    assert not latch.lifted.any() and not latch.closed.any()
    tasks.air_ticks = torch.tensor([3, 2])
    latch.observe(tasks)
    assert latch.lifted.tolist() == [False, True] and not latch.closed.any()
    tasks.air_ticks = torch.tensor([0, 0])
    latch.observe(tasks)
    assert latch.closed.tolist() == [False, True]
    tasks.physical_id = torch.tensor([3, 4])  # a new request clears its latch
    tasks.ascent_seen = torch.tensor([False, False])
    latch.observe(tasks)
    assert not latch.lifted.any() and not latch.closed.any()


def test_success_event_extension_is_paid_and_recorded():
    from supermimic_rl.reward_kernel import RewardExtensions

    composer, reference = setup('jump_flight')
    receipt = install_shared_reward(composer, extensions=RewardExtensions(success_event=5.))
    assert composer.tuning.success_event == 5. and receipt['extensions']['success_event'] == 5.
    assert 'extensions' not in install_shared_reward(setup('ordinary')[0])
    with pytest.raises(ValueError, match='unextended'):
        install_shared_reward(setup('ordinary')[0], verify_native=True, extensions=RewardExtensions(jump_flight=1.))


@pytest.mark.parametrize('case', ['ordinary', 'terrain', 'gimbal', 'jump_takeoff', 'jump_flight', 'jump_landing'])
def test_recovery_quiet_does_not_pay_other_tasks(case):
    from supermimic_rl.reward_kernel import RewardExtensions

    composer, reference = setup(case)
    begin(composer, reference, torch.zeros(2, 6))
    parameters = RewardParameters.from_composer(composer)
    physics = sample(composer, 1)
    history = RewardHistory(composer._window['dq'], torch.zeros(2), torch.zeros(2), torch.ones(2))
    baseline = physical_reward_densities(physics, reference, history, parameters)
    result = physical_reward_densities(physics, reference, history,
        replace(parameters, extensions=RewardExtensions(recovery_quiet=1.)))
    assert not result.positive['recovery_quiet'].any()
    for name, expected in baseline.components.items():
        torch.testing.assert_close(result.components[name], expected, atol=0, rtol=0)


def test_recovery_quiet_uses_height_and_upright_and_penalizes_all_angular_axes():
    from supermimic_rl.reward_kernel import RewardExtensions

    extensions = RewardExtensions(recovery_quiet=1.)
    assert extensions.active and not extensions.needs_jump_progress
    composer, reference = setup('recovery')
    receipt = install_shared_reward(composer, extensions=extensions)
    assert receipt['extensions']['recovery_quiet'] == 1.
    begin(composer, reference, torch.zeros(2, 6))
    parameters = RewardParameters.from_composer(composer)
    history = RewardHistory(composer._window['dq'], torch.zeros(2), torch.zeros(2), torch.ones(2))
    physics = replace(sample(composer, 1),
        height_above_ground_m=reference.admitted_body_command[:, 3],
        gravity_body=torch.tensor([[0., 0., -1.], [0., 0., -1.]]),
        angular_velocity_body_rad_s=torch.zeros(2, 3))

    def quiet(value):
        return physical_reward_densities(value, reference, history, parameters).positive['recovery_quiet']

    torch.testing.assert_close(quiet(physics), torch.ones(2))
    for axis in range(3):
        angular = torch.zeros(2, 3)
        angular[:, axis] = .5
        torch.testing.assert_close(quiet(replace(physics, angular_velocity_body_rad_s=angular)),
                                   torch.full((2,), torch.exp(torch.tensor(-1.)).item()))
    fallen = replace(physics, gravity_body=torch.tensor([[0., 0., 1.], [0., 0., 1.]]))
    assert not quiet(fallen).any()
    collapsed = replace(physics, height_above_ground_m=physics.height_above_ground_m - .15)
    assert (quiet(collapsed) < 1e-9).all()
    with pytest.raises(ValueError, match='finite and nonnegative'):
        RewardExtensions(recovery_quiet=-1.)


def test_recovery_quiet_gradient_points_toward_rest_and_matches_finite_difference():
    from supermimic_rl.reward_kernel import RewardExtensions

    composer, reference = setup('recovery')
    begin(composer, reference, torch.zeros(2, 6))
    parameters = replace(RewardParameters.from_composer(composer),
                         extensions=RewardExtensions(recovery_quiet=1.))
    physics = sample(composer, 1)
    history = RewardHistory(composer._window['dq'], torch.zeros(2), torch.zeros(2), torch.ones(2))
    angular = torch.tensor([[.1, -.2, .3], [0., 0., 0.]], dtype=torch.float64, requires_grad=True)

    def score(value):
        result = physical_reward_densities(replace(physics, angular_velocity_body_rad_s=value),
                                          reference, history, parameters)
        return result.positive['recovery_quiet'].sum()

    gradient, = torch.autograd.grad(score(angular), (angular,))
    assert torch.isfinite(gradient).all() and (gradient[0] * angular[0] < 0).all()
    assert not gradient[1].any()
    direction = torch.tensor([[.2, -.1, .3], [.1, .2, .3]], dtype=angular.dtype)
    epsilon = 1e-5
    numeric = (score(angular + epsilon * direction) - score(angular - epsilon * direction)) / (2 * epsilon)
    torch.testing.assert_close((gradient * direction).sum(), numeric, atol=1e-8, rtol=1e-6)


def test_jump_push_pays_reference_upward_speed_only_before_the_first_real_landing():
    from supermimic_rl.reward_kernel import RewardExtensions

    extensions = RewardExtensions(jump_push=2.)
    fresh = _progress('EXTENSION', [False, False])
    # Reference vertical speeds are [0.2, -0.1] m/s; measured [1.5, -0.5] m/s.
    _, takeoff = _jump_densities('jump_takeoff', extensions, fresh)
    torch.testing.assert_close(takeoff.positive['jump_push'], torch.tensor([2., 0.]))
    assert not takeoff.invalid.any()
    composer, reference = setup('jump_takeoff')
    install_shared_reward(composer, extensions=extensions, jump_tasks=_task_owner(lambda: None))
    begin(composer, reference, torch.zeros(2, 6))
    window = composer._window
    history = RewardHistory(window['dq'], window['action_cost'], window['action2_cost'], composer._action_count)
    slow = replace(sample(composer, 1), velocity_origin_world_m_s=torch.tensor([[0., 0., .1], [0., 0., .3]]))
    partial = physical_reward_densities(slow, reference, history, RewardParameters.from_composer(composer),
                                        jump_progress=fresh)
    torch.testing.assert_close(partial.positive['jump_push'], torch.tensor([1., 0.]))
    _, flight = _jump_densities('jump_flight', extensions, fresh)
    assert flight.positive['jump_push'].abs().sum() == 0
    # Early wheel unloading moves the native phase to LANDING; the push is still paid.
    _, unloaded = _jump_densities('jump_landing', extensions, fresh)
    torch.testing.assert_close(unloaded.positive['jump_push'], torch.tensor([2., 0.]))
    # After the first real flight touched down, a second push is not paid.
    _, landed = _jump_densities('jump_landing', extensions, _progress('BALLISTIC', [True, True], (True, False)))
    torch.testing.assert_close(landed.positive['jump_push'], torch.tensor([0., 0.]))
    with pytest.raises(ValueError, match='first-flight latch'):
        _jump_densities('jump_takeoff', extensions, _progress('EXTENSION', [False, False], None))
    _, ordinary = _jump_densities('ordinary', extensions)
    assert ordinary.positive['jump_push'].abs().sum() == 0


@pytest.mark.parametrize('case', ['jump_takeoff', 'jump_flight', 'jump_landing', 'ordinary', 'recovery'])
@pytest.mark.parametrize('stage_name', ['COMPRESSION', 'EXTENSION', 'BALLISTIC', 'HOLD'])
@pytest.mark.parametrize('motion', ['none', 'supported_ascent', 'real_flight'])
def test_missed_jump_removes_tracking_only_after_reference_has_finished(case, stage_name, motion):
    from transformer_rl.formal.tasks.jump import ReferenceStage
    from supermimic_rl.reward_kernel import JumpProgress, RewardExtensions

    composer, reference = setup(case)
    begin(composer, reference, torch.zeros(2, 6))
    physics = sample(composer, 1)
    reference = replace(reference, jump_height_world_z_m=physics.root_world_z_m,
                        jump_velocity_world_z_m_s=torch.zeros(2))
    physics = replace(physics, velocity_origin_world_m_s=torch.zeros(2, 3))
    history = RewardHistory(composer._window['dq'], torch.zeros(2), torch.zeros(2), torch.zeros(2))
    base = RewardParameters.from_composer(composer)
    parameters = replace(base, extensions=RewardExtensions(jump_missed=1.))
    # Zero vertical speed also occurs before takeoff; it is not a missed jump.
    stages = torch.tensor([int(getattr(ReferenceStage, stage_name)), int(ReferenceStage.COMPRESSION)])
    original = physical_reward_densities(physics, reference, history, base)
    progress = JumpProgress(stages, torch.tensor([motion != 'none', False]),
                            flight_started=torch.tensor([motion == 'real_flight', False]))
    result = physical_reward_densities(physics, reference, history, parameters, jump_progress=progress)
    # Any jump phase counts, including native FLIGHT/LANDING from wheel unloading without rising.
    missed = torch.tensor([case.startswith('jump') and stage_name == 'HOLD' and motion != 'real_flight', False])
    torch.testing.assert_close(result.gates['jump_missed'], missed)
    torch.testing.assert_close(result.cost['jump_missed'], missed.float())
    for name, value in original.positive.items():
        expected = torch.where(missed, 0., value) if name in ('forward', 'yaw', 'height', 'jump_vertical') else value
        torch.testing.assert_close(result.positive[name], expected, atol=0, rtol=0)
    for name, value in original.cost.items():
        torch.testing.assert_close(result.cost[name], value, atol=0, rtol=0)
    if case == 'jump_takeoff':
        assert original.positive['jump_vertical'][0] == 1.
        with pytest.raises(ValueError, match='request progress'):
            physical_reward_densities(physics, reference, history, parameters)
    assert not result.invalid.any()


@pytest.mark.parametrize('ascended', [False, True])
def test_missed_jump_stable_hold_retains_attitude_without_tracking_profit(ascended):
    from transformer_rl.formal.tasks.jump import ReferenceStage
    from supermimic_rl.reward_kernel import JumpProgress, RewardExtensions

    composer, reference = setup('jump_takeoff')
    reference = replace(reference, admitted_body_command=torch.tensor([[0., 0., 0., .32]]).repeat(2, 1),
                        jump_height_world_z_m=torch.full((2,), .32),
                        jump_velocity_world_z_m_s=torch.zeros(2))
    begin(composer, reference, torch.zeros(2, 6))
    physics = replace(sample(composer, 1),
        velocity_origin_body_m_s=torch.zeros(2, 3), velocity_origin_world_m_s=torch.zeros(2, 3),
        root_world_z_m=torch.full((2,), .32), height_above_ground_m=torch.full((2,), .32),
        roll_pitch_rad=torch.zeros(2, 2), gravity_body=torch.tensor([[0., 0., -1.]]).repeat(2, 1),
        angular_velocity_body_rad_s=torch.zeros(2, 3), policy_velocity_rad_s=torch.zeros(2, 6),
        active_explicit_effort_nm=torch.zeros(2, 6),
        body_normal_load_n=torch.tensor([[0., 105., 105., 0., 0.]]).repeat(2, 1),
        wheel_contact_tangent_speed_m_s=torch.zeros(2, 2), joint_boundary_shortfall_rad=torch.zeros(2, 6),
        wheel_position_base_m=torch.tensor([[[0., .2, -.26], [0., -.2, -.26]]]).repeat(2, 1, 1))
    history = RewardHistory(torch.zeros(2, 6), torch.zeros(2), torch.zeros(2), torch.zeros(2))
    base = RewardParameters.from_composer(composer)
    original = physical_reward_densities(physics, reference, history, base)
    result = physical_reward_densities(physics, reference, history,
        replace(base, extensions=RewardExtensions(jump_missed=1.)),
        jump_progress=JumpProgress(torch.full((2,), int(ReferenceStage.HOLD)),
            torch.full((2,), ascended), flight_started=torch.zeros(2, dtype=torch.bool)))

    assert (original.positive['forward'] > 0).all() and (original.positive['yaw'] > 0).all()
    for name in ('forward', 'yaw', 'height', 'jump_vertical'):
        assert not result.positive[name].any()
    torch.testing.assert_close(result.positive['jump_attitude'], torch.ones(2))
    torch.testing.assert_close(result.cost['jump_missed'], torch.ones(2))
    torch.testing.assert_close(result.total, torch.zeros_like(result.total), atol=0, rtol=0)
    assert not result.invalid.any()

    # A completed first flight still earns tracking during landing/HOLD.
    after_flight = physical_reward_densities(physics, reference, history,
        replace(base, extensions=RewardExtensions(jump_missed=1.)),
        jump_progress=JumpProgress(torch.full((2,), int(ReferenceStage.HOLD)),
            torch.ones(2, dtype=torch.bool), flight_closed=torch.ones(2, dtype=torch.bool),
            flight_started=torch.ones(2, dtype=torch.bool)))
    assert not after_flight.cost['jump_missed'].any()
    torch.testing.assert_close(after_flight.total, original.total, atol=0, rtol=0)

    with pytest.raises(ValueError, match='first-real-flight latch'):
        physical_reward_densities(physics, reference, history,
            replace(base, extensions=RewardExtensions(jump_missed=1.)),
            jump_progress=JumpProgress(torch.full((2,), int(ReferenceStage.HOLD)),
                                       torch.ones(2, dtype=torch.bool)))


def _task_owner(jump_reference):
    """The native jump task interface the shared reward reads (2 rows)."""
    return SimpleNamespace(jump_reference=jump_reference, ascent_seen=torch.zeros(2, dtype=torch.bool),
                           physical_id=torch.zeros(2, dtype=torch.long), air_ticks=torch.zeros(2, dtype=torch.long),
                           landed_s=torch.zeros(2), observe_physics=lambda *args, **kwargs: None)


def test_missed_jump_native_clock_is_sampled_once_per_interval_and_liftoff_per_tick():
    from transformer_rl.formal.tasks.jump import ReferenceStage
    from supermimic_rl.reward_kernel import RewardExtensions

    composer, reference = setup('jump_takeoff')
    reads = []

    def jump_reference():
        reads.append(1)
        return SimpleNamespace(clock_stage=torch.full((2,), int(ReferenceStage.HOLD)))

    tasks = _task_owner(jump_reference)
    install_shared_reward(composer, extensions=RewardExtensions(jump_missed=1.), jump_tasks=tasks)
    for interval in range(2):
        tasks.physical_id.fill_(interval)
        begin(composer, reference, torch.zeros(2, 6))
        for tick in range(1, 5):
            # Both rows rise, but only row 1 actually leaves support on tick 3.
            tasks.ascent_seen = torch.tensor([True, tick >= 3])
            tasks.air_ticks = torch.tensor([0, 2 if tick >= 3 else 0])
            tasks.observe_physics()
            composer.observe(sample(composer, tick), reference)
        result = finish(composer)
        torch.testing.assert_close(result.component_densities['cost.jump_missed'],
                                   torch.tensor([-1., -.5], dtype=torch.float64))
        assert len(reads) == interval + 1


def test_settle_reward_grows_with_the_native_stable_landing_streak():
    from supermimic_rl.reward_kernel import JumpProgress, RewardExtensions
    from transformer_rl.formal.tasks.jump import ReferenceStage

    extensions = RewardExtensions(jump_settle=2.)
    stage, ascent = torch.full((2,), int(ReferenceStage.HOLD)), torch.ones(2, dtype=torch.bool)
    closed = torch.ones(2, dtype=torch.bool)
    _, start = _jump_densities('jump_landing', extensions, JumpProgress(stage, ascent, closed, torch.tensor([.005, 0.])))
    torch.testing.assert_close(start.positive['jump_settle'], torch.tensor([2. * (.25 + .75 * .0025), 0.]))
    _, long = _jump_densities('jump_landing', extensions, JumpProgress(stage, ascent, closed, torch.tensor([1., 3.])))
    torch.testing.assert_close(long.positive['jump_settle'], torch.tensor([2. * .625, 2.]))
    assert not long.invalid.any()
    _, ordinary = _jump_densities('ordinary', extensions)
    assert ordinary.positive['jump_settle'].abs().sum() == 0
    with pytest.raises(ValueError, match='stable-landing time'):
        _jump_densities('jump_landing', extensions, JumpProgress(stage, ascent, closed))


def test_landing_rate_cost_applies_only_after_the_first_real_touchdown():
    from supermimic_rl.reward_kernel import JumpProgress, RewardExtensions
    from transformer_rl.formal.tasks.jump import ReferenceStage

    extensions = RewardExtensions(jump_landing_rate=2.)
    stage, ascent = torch.full((2,), int(ReferenceStage.HOLD)), torch.ones(2, dtype=torch.bool)
    composer, reference = setup('jump_landing')
    install_shared_reward(composer, extensions=extensions, jump_tasks=_task_owner(lambda: None))
    begin(composer, reference, torch.zeros(2, 6))
    window = composer._window
    history = RewardHistory(window['dq'], window['action_cost'], window['action2_cost'], composer._action_count)
    physics = replace(sample(composer, 1), angular_velocity_body_rad_s=torch.tensor([[0., .6, .8], [.3, 0., 0.]]))
    progress = JumpProgress(stage, ascent, torch.tensor([True, True]))
    result = physical_reward_densities(physics, reference, history, RewardParameters.from_composer(composer),
                                       jump_progress=progress)
    # Row 0: angular speed 1.0 rad/s (pitch and yaw, as the native quiet test) -> 2 * (1.0 - 0.4)^2;
    # row 1 is inside the free band.
    torch.testing.assert_close(result.cost['jump_landing_rate'], torch.tensor([2. * .36, 0.]))
    before = physical_reward_densities(physics, reference, history, RewardParameters.from_composer(composer),
                                       jump_progress=JumpProgress(stage, ascent, torch.tensor([False, False])))
    assert before.cost['jump_landing_rate'].abs().sum() == 0
