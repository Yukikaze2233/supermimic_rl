"""Physical jump qualification keeps its reward after protocol release."""
from dataclasses import replace

import pytest
import torch

from transformer_rl.formal.control.observation import CommandMode
from transformer_rl.formal.tasks.gimbal_course import GimbalRewardReference
from transformer_rl.formal.tasks.jump import ReferenceStage
from transformer_rl.formal.tasks.reward import RewardPhase, RoughDashRewardReference, TerrainRewardReference

from supermimic_rl.course.reward import install_shared_reward
from supermimic_rl.reward_kernel import (
    JumpProgress, RewardExtensions, RewardHistory, RewardParameters, physical_reward_densities,
)
from supermimic_rl.world_model_v2.jump_task import JumpTask, JumpTaskConfig, JumpTaskState
from support.world_v2.jump_task import native_jumps
from test_reward_kernel import begin, finish, sample, setup
from test_world_v2_jump_task import physical_ticks


def fixture(case='ordinary'):
    composer, reference = setup(case)
    command = reference.admitted_body_command.clone()
    command[:, :3] = 0.
    reference = replace(reference, admitted_body_command=command)
    begin(composer, reference, torch.zeros(2, 6))
    physics = replace(sample(composer, 1), height_above_ground_m=command[:, 3],
        velocity_origin_body_m_s=torch.zeros(2, 3), velocity_origin_world_m_s=torch.zeros(2, 3),
        angular_velocity_body_rad_s=torch.tensor([[0., .6, 0.], [0., 0., .8]]))
    history = RewardHistory(composer._window['dq'], torch.zeros(2), torch.zeros(2), torch.ones(2))
    extension = RewardExtensions(jump_flight=4., jump_lift=2., jump_push=2., jump_missed=1.,
        jump_settle=2., jump_landing_rate=4., jump_peak_completion=True, jump_physical_tail=True)
    parameters = replace(RewardParameters.from_composer(composer), extensions=extension)
    progress = JumpProgress(torch.full((2,), int(ReferenceStage.HOLD)), torch.ones(2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool), torch.zeros(2), torch.ones(2, dtype=torch.bool),
        torch.tensor([.0425, .085]), torch.full((2,), .2), torch.ones(2, dtype=torch.bool))
    return composer, reference, physics, history, parameters, progress


def evaluate(reference, physics, history, parameters, progress):
    return physical_reward_densities(physics, reference, history, parameters, jump_progress=progress)


def compare_components(actual, expected):
    assert actual.components.keys() == expected.components.keys()
    for name, value in expected.components.items():
        torch.testing.assert_close(actual.components[name], value, rtol=0, atol=0)
    torch.testing.assert_close(actual.total, expected.total, rtol=0, atol=0)
    assert torch.equal(actual.invalid, expected.invalid)


def test_normal_tail_retains_only_peak_completion_missed_and_landing_terms():
    _, reference, physics, history, parameters, progress = fixture()
    baseline = evaluate(reference, physics, history,
        replace(parameters, extensions=replace(parameters.extensions, jump_physical_tail=False)), progress)
    actual = evaluate(reference, physics, history, parameters, progress)
    completion = torch.tensor([.25, .5])
    for key, value in baseline.positive.items():
        expected = value * completion if key in ('forward', 'yaw', 'height') else value
        torch.testing.assert_close(actual.positive[key], expected)
    for key, value in baseline.cost.items():
        expected = (1-completion if key == 'jump_missed' else
                    4*(physics.angular_velocity_body_rad_s.norm(dim=-1)-.4).square()
                    if key == 'jump_landing_rate' else value)
        torch.testing.assert_close(actual.cost[key], expected)
    for key in ('jump_flight', 'jump_lift', 'jump_push', 'jump_vertical', 'jump_settle'):
        assert not actual.positive[key].any()
    assert actual.gates['jump_missed'].all() and not actual.invalid.any()


def test_tail_does_not_depend_on_released_protocol_clock_or_invent_flight():
    _, reference, physics, history, parameters, progress = fixture()
    no_flight = replace(progress, reference_stage=torch.full((2,), int(ReferenceStage.COMPRESSION)),
                        flight_started=torch.zeros(2, dtype=torch.bool), flight_closed=torch.zeros(2, dtype=torch.bool))
    actual = evaluate(reference, physics, history, parameters, no_flight)
    for key in ('forward', 'yaw', 'height', 'jump_flight', 'jump_lift', 'jump_push', 'jump_vertical', 'jump_settle'):
        assert not actual.positive[key].any()
    torch.testing.assert_close(actual.cost['jump_missed'], torch.ones(2))
    assert not actual.cost['jump_landing_rate'].any()


def test_settling_keeps_native_streak_and_original_angular_deadband():
    _, reference, physics, history, parameters, progress = fixture()
    progress = replace(progress, peak_rise=torch.full((2,), .2), landed_s=torch.tensor([.5, 1.]))
    physics = replace(physics, angular_velocity_body_rad_s=torch.tensor([[.4, 0., 0.], [0., .45, 0.]]))
    result = evaluate(reference, physics, history, parameters, progress)
    torch.testing.assert_close(result.positive['jump_settle'], torch.tensor([.875, 1.25]))
    torch.testing.assert_close(result.cost['jump_landing_rate'], torch.tensor([0., .01]), atol=1e-7, rtol=0)
    assert not result.cost['jump_missed'].any()


@pytest.mark.parametrize('excluded', ['inactive', 'slope', 'stair', 'recovery', 'jump',
                                      'terrain', 'airborne', 'rough', 'gimbal'])
def test_other_scopes_keep_original_components_and_do_not_inherit_physical_tail(excluded):
    _, reference, physics, history, parameters, progress = fixture()
    if excluded == 'inactive':
        progress = replace(progress, physical_active=torch.zeros(2, dtype=torch.bool))
    elif excluded in ('slope', 'stair', 'recovery', 'jump'):
        mode = getattr(CommandMode, excluded.upper())
        phase = {'recovery': RewardPhase.RECOVERY, 'jump': RewardPhase.LANDING}.get(excluded, RewardPhase.SUPPORT)
        reference = replace(reference, mode=torch.full((2,), int(mode)), phase=torch.full((2,), int(phase)))
    elif excluded in ('terrain', 'airborne'):
        reference = replace(reference, terrain=TerrainRewardReference(torch.full((2,), excluded=='terrain'),
            reference.admitted_body_command[:, 3], torch.full((2,), excluded=='airborne'), torch.zeros(2, dtype=torch.bool)))
    elif excluded == 'rough':
        parameters = replace(parameters, tuning=replace(parameters.tuning, profile='rough_dash'))
        reference = replace(reference, rough_dash=RoughDashRewardReference(torch.ones(2, dtype=torch.bool),
            reference.admitted_body_command[:, 3], torch.full((2, 2), .04), torch.zeros(2)))
    else:
        reference = replace(reference, gimbal=GimbalRewardReference(torch.ones(2, dtype=torch.bool),
            torch.zeros(2), torch.zeros(2), torch.zeros(2)))
    actual = evaluate(reference, physics, history, parameters, progress)
    expected = evaluate(reference, physics, history,
        replace(parameters, extensions=replace(parameters.extensions, jump_physical_tail=False)), progress)
    compare_components(actual, expected)


def test_default_receipt_and_old_optional_activity_contract_are_unchanged():
    assert RewardExtensions().jump_physical_tail is False
    _, reference, physics, history, parameters, progress = fixture()
    old = replace(parameters, extensions=replace(parameters.extensions, jump_physical_tail=False))
    compare_components(evaluate(reference, physics, history, old, progress),
                       evaluate(reference, physics, history, old, replace(progress, physical_active=None)))
    tasks = native_jumps(2)._terrain_course.tasks
    receipt = install_shared_reward(setup('ordinary')[0], extensions=old.extensions, jump_tasks=tasks)
    assert 'jump_physical_tail' not in receipt['extensions']
    enabled = install_shared_reward(setup('ordinary')[0], extensions=parameters.extensions, jump_tasks=tasks)
    assert enabled['extensions']['jump_physical_tail'] is True
    for invalid in (1, 0, 'yes', None):
        with pytest.raises(ValueError, match='boolean flag'):
            RewardExtensions(jump_missed=1., jump_peak_completion=True, jump_physical_tail=invalid)
    with pytest.raises(ValueError, match='enabled jump_peak_completion'):
        RewardExtensions(jump_missed=1., jump_physical_tail=True)
    for absent in (None, replace(progress, physical_active=None)):
        with pytest.raises(ValueError, match='explicit per-tick physical jump activity'):
            evaluate(reference, physics, history, parameters, absent)


def test_peak_gradient_matches_finite_difference_and_stops_at_completion_or_inactive_request():
    _, reference, physics, history, parameters, progress = fixture()

    def score(peak, *, active=True):
        return evaluate(reference, physics, history, parameters, replace(progress, peak_rise=peak,
            physical_active=torch.full((2,), active))).total.sum()

    peak = torch.tensor([.05, .1], dtype=torch.float64, requires_grad=True)
    gradient, = torch.autograd.grad(score(peak), peak)
    assert (gradient > 0).all()
    direction = torch.tensor([.3, -.2], dtype=torch.float64)
    eps = 1e-5
    numeric = (score(peak+eps*direction)-score(peak-eps*direction))/(2*eps)
    torch.testing.assert_close((gradient*direction).sum(), numeric, rtol=1e-7, atol=1e-7)
    inactive, = torch.autograd.grad(score(peak, active=False), peak)
    saturated = torch.tensor([.2, .4], dtype=torch.float64, requires_grad=True)
    completed, = torch.autograd.grad(score(saturated), saturated)
    assert not inactive.any() and not completed.any()


def test_native_and_imagined_tail_rewards_stop_on_exact_success_substep():
    native = native_jumps(2)
    owner = native._terrain_course.tasks
    owner.physical_id[:] = torch.tensor([7, 8])
    owner.active_id.fill_(-1)
    owner.base_mode.fill_(int(CommandMode.NORMAL))
    owner.active_rise.fill_(.06)
    owner.peak_rise.fill_(.06)
    owner.latched_height.fill_(.32)
    owner.ground_anchor.zero_()
    owner.flight_seen.fill_(True)
    owner.ascent_seen.fill_(True)
    owner.supported_start.fill_(True)
    owner.landed_s[:] = torch.tensor([1.9925, 1.9875])
    composer, reference, _, _, parameters, _ = fixture()
    command = reference.admitted_body_command.clone()
    command[:, 3] = .32
    reference = replace(reference, admitted_body_command=command)
    install_shared_reward(composer, extensions=parameters.extensions, jump_tasks=owner)
    latch = owner.supermimic_flight_latch
    latch.lifted.fill_(True); latch.closed.fill_(True)
    state = JumpTaskState.capture(native)
    task = JumpTask(JumpTaskConfig.capture(native))
    observed = task.observe_interval(state, physical=physical_ticks(2), wheel_support_load=torch.full((2, 4, 2), 20.),
        base_com_offset=torch.zeros(2, 3), base_mode=torch.zeros(2, dtype=torch.long),
        command_height=torch.full((2,), .32), ground_z=torch.zeros(2, 4), retain_peak_gradient=True)
    assert observed.physical_active.tolist() == [[True, False, False, False], [True, True, False, False]]
    # Re-open the fixture's policy interval now that the actual producer is installed.
    composer._window = None
    begin(composer, reference, torch.zeros(2, 6))
    expected = {}
    for tick in range(4):
        owner.observe_physics(torch.full((2, 2), 20.), torch.zeros(2), torch.zeros(2), torch.full((2,), .32),
                              angular_speed_rad_s=torch.zeros(2), dt=.005)
        progress = JumpProgress(observed.reference_stage[:, tick], observed.ascent_seen[:, tick],
            observed.flight_closed[:, tick], observed.landed_s[:, tick], observed.flight_started[:, tick],
            observed.peak_rise[:, tick], observed.requested_rise[:, tick], observed.physical_active[:, tick])
        physics = sample(composer, tick+1)
        history = RewardHistory(composer._window['dq'], composer._window['action_cost'],
                                composer._window['action2_cost'], composer._action_count)
        result = evaluate(reference, physics, history, RewardParameters.from_composer(composer), progress)
        composer.observe(physics, reference)
        for key, value in result.components.items():
            expected.setdefault(key, []).append(value.double())
    actual = finish(composer)
    for key, values in expected.items():
        torch.testing.assert_close(actual.component_densities[key], torch.stack(values).mean(0), rtol=1e-7, atol=1e-7)
    settled = torch.stack(expected['positive.jump_settle'], -1)
    assert (settled[0, :1] > 0).all() and not settled[0, 1:].any()
    assert (settled[1, :2] > 0).all() and not settled[1, 2:].any()
    assert observed.state.physical_id.tolist() == [-1, -1]


def test_native_protocol_release_timeout_new_request_and_reset_do_not_reuse_tail_evidence():
    from transformer_rl.formal.lab_env.course_tasks import RequestCommand
    from transformer_rl.formal.tasks.protocol import RequestEnd, RequestResponse
    from supermimic_rl.course.curriculum import SuperMimicCurriculum
    from transformer_rl.formal.lab_env.formal_curriculum import LabTaskDrawSampler
    from test_world_v2_jump_task import advance, estimate_at

    native = native_jumps(2)
    owner = native._terrain_course.tasks
    owner.rise.fill_(.2)
    task = JumpTask(JumpTaskConfig.capture(native))
    state = advance(native, task, JumpTaskState.capture(native), estimate_at(native.time, 2)).state
    owner.protocol.flight_seen.fill_(True)
    owner.protocol.landing_since_s.fill_(.02)
    owner.protocol.last_time_s.fill_(2.)
    native.time.fill_(2.)
    released = advance(native, task, JumpTaskState.capture(native), estimate_at(native.time, 2))
    assert released.end_reason.tolist() == [int(RequestEnd.ESTIMATED_COMPLETE)] * 2
    assert released.mode.tolist() == [int(CommandMode.NORMAL)] * 2
    assert released.state.active_id.tolist() == [-1, -1]
    assert released.state.physical_id.tolist() == [0, 0]
    # Only the estimator saw flight; no physical flight or invented completion.
    assert not released.state.values[:, 10].any()
    owner.peak_rise.fill_(.08)
    empty = RequestCommand(torch.full((2,), -1), torch.zeros(2))
    packet = RequestCommand(torch.tensor([1, -1]), torch.full((2,), .2))
    new = advance(native, task, JumpTaskState.capture(native), estimate_at(native.time, 2), packet=packet)
    assert new.response[0] == int(RequestResponse.ACCEPTED)
    assert new.state.physical_id.tolist() == [1, 0]
    torch.testing.assert_close(new.state.values[:, 15], torch.tensor([0., .08]), rtol=0, atol=0)
    # At the old deadline row 1 expires; the newly accepted row 0 continues.
    native.time.fill_(6.02)
    timed = advance(native, task, JumpTaskState.capture(native), estimate_at(native.time, 2), packet=empty)
    assert timed.state.physical_id.tolist() == [1, -1]
    assert timed.mode[1] == int(CommandMode.NORMAL)
    observed = task.observe_interval(timed.state, physical=physical_ticks(2),
        wheel_support_load=torch.full((2, 4, 2), 20.), base_com_offset=torch.zeros(2, 3),
        base_mode=torch.zeros(2, dtype=torch.long), command_height=torch.full((2,), .32), ground_z=torch.zeros(2, 4))
    assert observed.physical_active.tolist() == [[True]*4, [False]*4]
    ids = torch.arange(2)
    draw = LabTaskDrawSampler(SuperMimicCurriculum(2))(ids)
    owner.reset(ids, draw, native.time, terrain_kind=draw.terrain_kind)
    reset = JumpTaskState.capture(native)
    assert reset.physical_id.tolist() == [-1, -1]
    assert not reset.values[:, 15:17].any()
    assert reset.record().keys() == state.record().keys()


@pytest.mark.parametrize('enabled', [False, True])
def test_full_normal_imagination_only_retains_prior_peak_gradient_when_tail_enabled(monkeypatch, enabled):
    from support.world_v2.jump_imagination import fixture as learner_fixture

    learner, start, frames = learner_fixture()
    learner.imagination.reward.extensions = RewardExtensions(jump_missed=1., jump_peak_completion=True,
                                                             jump_physical_tail=enabled)
    jump = JumpTaskState.decode(start)
    jump.ids[:, 0] = jump.ids[:, 4] = 7
    jump.ids[:, 1] = -1
    jump.clock[:, 0] = jump.clock[:, 2] = 3.
    jump.clock[:, 1] = jump.clock[:, 4] = 0.
    jump.values[:, 3] = 100.  # No new automatic request during this same-request test.
    jump.values[:, 5] = .2
    jump.values[:, 6] = .35
    jump.values[:, 8] = 0.  # Released protocol clock, not a fabricated HOLD.
    jump.values[:, [2, 10, 11, 12, 13]] = 1.
    jump = replace(jump, flight_latch=torch.tensor([[7, 1, 1], [7, 1, 1]]))
    start.update(jump.record())
    assert start['goal'][:, 4:9].argmax(-1).tolist() == [int(CommandMode.NORMAL)] * 2
    with torch.no_grad():
        learner.model.motion.net[-1].weight.zero_()
    recorded, original = [], learner.model.step

    def step(*args, **kwargs):
        prediction = original(*args, **kwargs)
        if not recorded:
            delta = torch.zeros_like(prediction.ticks.state.position)
            delta[:, 0, 2] = .08
            state = replace(prediction.ticks.state, position=prediction.ticks.state.position + delta)
            prediction = replace(prediction, ticks=replace(prediction.ticks, state=state))
        recorded.append(prediction.ticks.state.position)
        return prediction

    monkeypatch.setattr(learner.model, 'step', step)
    rollout = learner.imagination.rollout(start, frames, 3)
    assert rollout.lengths.tolist() == [3, 3] and not rollout.invalid.any()
    gradient, = torch.autograd.grad(rollout.rewards[-1].sum(), recorded[0])
    if enabled:
        assert (gradient[:, 0, 2] > 0).all()
    else:
        assert not gradient[:, 0].any()
    assert not gradient[:, 1:].any()
