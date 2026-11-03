"""Shared reward parity against the frozen native composer and real gradient checks."""
from copy import deepcopy
from dataclasses import fields, replace

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


def _jump_densities(case, extensions):
    from supermimic_rl.reward_kernel import RewardExtensions

    composer, reference = setup(case)
    install_shared_reward(composer, extensions=extensions)
    begin(composer, reference, torch.zeros(2, 6))
    window = composer._window
    history = RewardHistory(window['dq'], window['action_cost'], window['action2_cost'], composer._action_count)
    physics = sample(composer, 1)
    physics = replace(physics, velocity_origin_world_m_s=torch.tensor([[0., 0., 1.5], [0., 0., -.5]]))
    return composer, physical_reward_densities(physics, reference, history, RewardParameters.from_composer(composer))


def test_jump_extensions_pay_only_actual_flight_and_are_off_by_default():
    from supermimic_rl.reward_kernel import RewardExtensions

    extensions = RewardExtensions(jump_flight=4., jump_lift=2., success_event=5.)
    _, flight = _jump_densities('jump_flight', extensions)
    torch.testing.assert_close(flight.positive['jump_flight'], torch.full((2,), 4.))
    torch.testing.assert_close(flight.positive['jump_lift'], torch.tensor([3., 0.]))
    assert not flight.invalid.any()
    _, takeoff = _jump_densities('jump_takeoff', extensions)
    assert takeoff.positive['jump_flight'].abs().sum() == 0 and takeoff.positive['jump_lift'].abs().sum() == 0
    composer, default = _jump_densities('jump_flight', RewardExtensions())
    assert 'jump_flight' not in default.positive and composer.tuning.success_event == 1.


def test_success_event_extension_is_paid_and_recorded():
    from supermimic_rl.reward_kernel import RewardExtensions

    composer, reference = setup('jump_flight')
    receipt = install_shared_reward(composer, extensions=RewardExtensions(success_event=5.))
    assert composer.tuning.success_event == 5. and receipt['extensions']['success_event'] == 5.
    assert 'extensions' not in install_shared_reward(setup('ordinary')[0])
    with pytest.raises(ValueError, match='unextended'):
        install_shared_reward(setup('ordinary')[0], verify_native=True, extensions=RewardExtensions(jump_flight=1.))


def test_jump_push_pays_reference_upward_speed_only_before_flight_in_the_push_window():
    from supermimic_rl.reward_kernel import RewardExtensions

    extensions = RewardExtensions(jump_push=2.)
    # Reference vertical speeds are [0.2, -0.1] m/s; measured [1.5, -0.5] m/s.
    _, takeoff = _jump_densities('jump_takeoff', extensions)
    torch.testing.assert_close(takeoff.positive['jump_push'], torch.tensor([2., 0.]))
    assert not takeoff.invalid.any()
    composer, reference = setup('jump_takeoff')
    install_shared_reward(composer, extensions=extensions)
    begin(composer, reference, torch.zeros(2, 6))
    window = composer._window
    history = RewardHistory(window['dq'], window['action_cost'], window['action2_cost'], composer._action_count)
    slow = replace(sample(composer, 1), velocity_origin_world_m_s=torch.tensor([[0., 0., .1], [0., 0., .3]]))
    partial = physical_reward_densities(slow, reference, history, RewardParameters.from_composer(composer))
    torch.testing.assert_close(partial.positive['jump_push'], torch.tensor([1., 0.]))
    _, flight = _jump_densities('jump_flight', extensions)
    assert flight.positive['jump_push'].abs().sum() == 0
    _, ordinary = _jump_densities('ordinary', extensions)
    assert ordinary.positive['jump_push'].abs().sum() == 0
