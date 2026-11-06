"""Peak shaping retains real request history without differentiating the FSM."""
from dataclasses import replace

import pytest
import torch

pytest.importorskip('transformer_rl.formal.tasks.reward')
from transformer_rl.formal.control.observation import CommandMode
from transformer_rl.formal.tasks.curriculum import TaskFamily
from transformer_rl.formal.tasks.protocol import RequestResponse
from transformer_rl.formal.lab_env.course_tasks import RequestCommand

from supermimic_rl.course.reward import install_flight_latch
from supermimic_rl.world_model_v2.jump_task import JumpTask, JumpTaskConfig, JumpTaskState
from support.world_v2.jump_task import native_jumps
from test_world_v2_jump_task import advance, estimate_at, physical_ticks, same_state


def _active(n=2):
    native = native_jumps(n)
    native._terrain_course.tasks.rise.fill_(.2)
    install_flight_latch(native._terrain_course.tasks)
    task = JumpTask(JumpTaskConfig.capture(native))
    state = advance(native, task, JumpTaskState.capture(native), estimate_at(native.time, n)).state
    return native, task, state


def _observe(task, state, physical, *, force=None, retain=True):
    n = len(state.ids)
    return task.observe_interval(state, physical=physical,
        wheel_support_load=torch.zeros(n, 4, 2) if force is None else force,
        base_com_offset=torch.zeros(n, 3), base_mode=torch.zeros(n, dtype=torch.long),
        command_height=torch.full((n,), .32), ground_z=torch.zeros(n, 4),
        retain_peak_gradient=retain)


def _boundary(task, state, *, packet=None, retain=True):
    n = len(state.ids)
    now = task.boundary_time(state)
    admitted = torch.zeros(n, 4)
    admitted[:, 3] = .32
    return task.advance(state, now_s=now, estimate=estimate_at(state.now_s, n), admitted=admitted,
        base_mode=torch.full((n,), int(CommandMode.NORMAL)),
        task_family=torch.full((n,), int(TaskFamily.REQUEST)), elapsed_s=now.float(),
        ground_z=torch.zeros(n), blocked_by_stair=torch.zeros(n, dtype=torch.bool),
        interval_uniform=torch.ones(n), packet=packet, retain_peak_gradient=retain)


def test_native_peak_values_match_each_tick_and_survive_a_boundary_with_gradient():
    native, task, state = _active()
    owner = native._terrain_course.tasks
    physical = physical_ticks(2, vz=.4)
    physical.position[..., 2] = torch.tensor([[.34, .36, .40, .38], [.35, .39, .37, .36]])
    physical.position.requires_grad_()
    expected = []
    for tick in range(4):
        owner.observe_physics(torch.zeros(2, 2), physical.velocity[:, tick, 2], torch.zeros(2),
                              physical.position[:, tick, 2].detach(), angular_speed_rad_s=torch.zeros(2))
        expected.append(owner.peak_rise.clone())
    result = _observe(task, state, physical)
    torch.testing.assert_close(result.peak_rise, torch.stack(expected, -1), atol=0, rtol=0)
    same_state(result.state, native)
    assert not result.phase.requires_grad and not result.reference_stage.requires_grad
    after_boundary = _boundary(task, result.state).state
    next_physical = physical_ticks(2, z=.33, requires_grad=True)
    landed = _observe(task, after_boundary, next_physical, force=torch.full((2, 4, 2), 20.))
    gradient, next_gradient = torch.autograd.grad(landed.peak_rise.sum(),
                                                 (physical.position, next_physical.position))
    expected_gradient = torch.zeros_like(gradient)
    expected_gradient[0, 2, 2] = expected_gradient[1, 1, 2] = 4.
    torch.testing.assert_close(gradient, expected_gradient, atol=0, rtol=0)
    assert not next_gradient.any()


def test_multistep_peak_gradient_matches_finite_difference_without_cross_row_history():
    _, task, state = _active()

    def score(heights):
        physical = physical_ticks(2, vz=.4)
        physical.position[..., 2] = heights
        first = _observe(task, state, physical)
        second = _observe(task, _boundary(task, first.state).state, physical_ticks(2, z=.33))
        return second.peak_rise[:, -1]

    heights = torch.tensor([[.34, .36, .40, .38], [.35, .39, .37, .36]], requires_grad=True)
    values = score(heights)
    jacobian = torch.stack([torch.autograd.grad(value, heights, retain_graph=True)[0] for value in values])
    assert not jacobian[0, 1].any() and not jacobian[1, 0].any()
    direction = torch.tensor([[.2, .3, -.4, .1], [.1, -.2, .3, .1]])
    epsilon = 1e-3
    finite_difference = (score(heights + epsilon * direction) - score(heights - epsilon * direction)) / (2 * epsilon)
    torch.testing.assert_close((jacobian * direction).sum((-1, -2)), finite_difference, atol=1e-5, rtol=1e-4)


def test_new_accepted_request_resets_only_its_own_peak_and_gradient():
    _, task, state = _active()
    physical = physical_ticks(2, z=.40, vz=.4, requires_grad=True)
    first = _observe(task, state, physical)
    ids = first.state.ids.clone()
    # The deployment protocol has finished row 1, but its previous physical
    # request is still waiting for certification when the next request arrives.
    ids[1, 1] = -1
    pending = replace(first.state, ids=ids)
    packet = RequestCommand(torch.tensor([-1, 1]), torch.full((2,), .2))
    advanced = _boundary(task, pending, packet=packet)
    assert advanced.response[1] == int(RequestResponse.ACCEPTED)
    assert advanced.state.physical_id.tolist() == [0, 1]
    torch.testing.assert_close(advanced.state.values[:, 15], torch.tensor([.08, 0.]), atol=3e-8, rtol=0)
    after = _observe(task, advanced.state, physical_ticks(2, z=.33))
    gradient, = torch.autograd.grad(after.peak_rise.sum(), physical.position)
    assert gradient[0, :, 2].sum() == 4.
    assert not gradient[1].any()


def test_native_success_inside_interval_stops_peak_updates_and_cuts_completed_history():
    native, task, _ = _active(1)
    owner = native._terrain_course.tasks
    owner.peak_rise.fill_(.2)
    owner.flight_seen.fill_(True)
    owner.ascent_seen.fill_(True)
    owner.landed_s.fill_(1.995)
    state = JumpTaskState.capture(native)
    prior = torch.tensor([.2], requires_grad=True)
    state = replace(state, values=torch.cat((state.values[:, :15], prior[:, None], state.values[:, 16:]), -1))
    physical = physical_ticks(1)
    # Later samples must not contribute to a request already completed at tick 0.
    physical.position[:, 1:, 2] = .9
    physical.position.requires_grad_()
    force = torch.full((1, 4, 2), 20.)
    expected = []
    for tick in range(4):
        owner.observe_physics(force[:, tick], torch.zeros(1), torch.zeros(1),
                              physical.position[:, tick, 2].detach(), angular_speed_rad_s=torch.zeros(1))
        expected.append(owner.peak_rise.clone())
    actual = _observe(task, state, physical, force=force)
    torch.testing.assert_close(actual.peak_rise, torch.stack(expected, -1), atol=0, rtol=0)
    assert actual.state.physical_id.item() == -1
    gradient, = torch.autograd.grad(actual.state.values[:, 15].sum(), prior)
    assert not gradient.any()
    same_state(actual.state, native)


def test_disabled_extension_keeps_detached_old_interval_values():
    _, task, state = _active()
    physical = physical_ticks(2, z=.4, vz=.4, requires_grad=True)
    baseline = _observe(task, state, physical, retain=False)
    enhanced = _observe(task, state, physical)
    for key, value in baseline.state.record().items():
        torch.testing.assert_close(value, enhanced.state.record()[key], atol=0, rtol=0)
    assert baseline.peak_rise is None and baseline.requested_rise is None
    assert not baseline.state.values.requires_grad
    assert enhanced.state.values.requires_grad


def test_full_imagination_passes_peak_gradient_from_first_tick_to_later_hold_reward(monkeypatch):
    from supermimic_rl.reward_kernel import RewardExtensions
    from support.world_v2.jump_imagination import fixture

    learner, start, frames = fixture()
    learner.imagination.reward.extensions = RewardExtensions(jump_missed=1., jump_peak_completion=True)
    jump = JumpTaskState.decode(start)
    jump.ids[:, 0] = jump.ids[:, 1] = jump.ids[:, 4] = 7
    jump.clock[:, 0] = jump.clock[:, 2] = 3.
    jump.clock[:, 1] = jump.clock[:, 4] = 0.
    jump.values[:, 5] = .2
    jump.values[:, 6] = .35
    jump.values[:, 8] = 3.
    jump.values[:, [2, 10, 11, 12, 13]] = 1.
    jump = replace(jump, flight_latch=torch.tensor([[7, 1, 1], [7, 1, 1]]))
    start.update(jump.record())
    start['goal'][:, 4] = 0.
    start['goal'][:, 8] = 1.
    start['goal'][:, 9] = 1.
    start['goal'][:, 10] = 3.
    start['reward_route'][:, 0] = 4.
    # The spike returns to baseline within the first 20 ms. Its influence on
    # a later HOLD reward must pass through request peak history, not end pose.
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
    assert (gradient[:, 0, 2] > 0).all()
    assert not gradient[:, 1:].any()
