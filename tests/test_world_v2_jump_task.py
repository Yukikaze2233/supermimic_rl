"""Jump task state matches the pinned native request and physics producers."""
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.world_model_v2.jump_task import (
    JUMP_FEATURE_DIM, JUMP_RECORD_DTYPES, JUMP_RECORD_FIELDS, JumpTask, JumpTaskConfig, JumpTaskState,
)
from supermimic_rl.world_model_v2.physical import PhysicalState
from supermimic_rl.world_model_v2 import so3
from transformer_rl.formal.control.observation import CommandMode
from transformer_rl.formal.lab_env.course_tasks import RequestCommand
from transformer_rl.formal.tasks.curriculum import TaskFamily
from transformer_rl.formal.tasks.protocol import CompletedEstimate, RequestEnd, RequestResponse
from transformer_rl.formal.tasks.reward import RewardPhase
from support.world_v2.jump_task import native_jumps


def estimate_at(now, n, contact=.9):
    return CompletedEstimate(torch.zeros(n), torch.full((n, 2), contact), torch.zeros(n),
                             torch.zeros(n), now.clone(), torch.ones(n, dtype=torch.bool))


def physical_ticks(n, *, z=.32, vz=0., requires_grad=False):
    position = torch.zeros(n, 4, 3)
    position[..., 2] = z
    velocity = torch.zeros_like(position)
    velocity[..., 2] = vz
    return PhysicalState(position.requires_grad_(requires_grad), torch.eye(3).expand(n, 4, 3, 3).clone(),
        velocity.requires_grad_(requires_grad), torch.zeros_like(velocity),
        torch.zeros(n, 4, 6), torch.zeros(n, 4, 8))


def same_state(actual, native):
    expected = JumpTaskState.capture(native)
    for key, value in expected.record().items():
        torch.testing.assert_close(actual.record()[key], value, rtol=0, atol=0)


def advance(native, task, state, estimate, *, packet=None, blocked=None):
    owner, n = native._terrain_course.tasks, len(state.ids)
    blocked = torch.zeros(n, dtype=torch.bool) if blocked is None else blocked
    rng = torch.Generator().set_state(owner.generator.get_state())
    draw = torch.rand(n, generator=rng)
    native.time = task.boundary_time(state)
    # Hold commands in these task tests; command trajectory fidelity has its
    # own native MotionTask tests. This is the admission before jump latching.
    admitted = owner.trajectory.end.clone()
    previous_height, previous_elapsed = owner.stream.current[:, 3].clone(), owner.elapsed.clone()
    native_result = owner.step(native.time, estimate, native._terrain_course.router.origins[:, 2],
                               command=packet, blocked_by_stair=blocked)
    result = task.advance(state, now_s=native.time, estimate=estimate, admitted=admitted,
        base_mode=owner.base_mode, task_family=owner.family, elapsed_s=owner.elapsed,
        ground_z=torch.zeros(n), blocked_by_stair=blocked, interval_uniform=draw, packet=packet,
        previous_command_height=previous_height, previous_elapsed_s=previous_elapsed)
    torch.testing.assert_close(result.command, native_result[0].admitted, rtol=0, atol=0)
    torch.testing.assert_close(result.mode, native_result[1], rtol=0, atol=0)
    torch.testing.assert_close(result.request_height, native_result[2], rtol=0, atol=0)
    torch.testing.assert_close(result.request_clock, native_result[3], rtol=0, atol=0)
    same_state(result.state, native)
    return result


def test_optional_record_keeps_native_dtypes_coordinates_and_captured_values():
    native = native_jumps(2, origin_z=2.)
    owner = native._terrain_course.tasks
    owner.protocol.max_seen_id.fill_(2**54 + 17)
    native.time.fill_(1_000_000.015)
    owner.ground_anchor.fill_(2.)
    state = JumpTaskState.capture(native)
    assert {key: tuple(value.shape[1:]) for key, value in state.record().items()} == JUMP_RECORD_FIELDS
    assert all(state.record()[key].dtype == dtype for key, dtype in JUMP_RECORD_DTYPES.items())
    assert state.ids[0, 0].item() == 2**54 + 17
    assert state.now_s[0].item() == 1_000_000.015
    assert state.values[:, 7].count_nonzero() == 0
    torch.testing.assert_close(state.values[:, 14], torch.full((2,), .32), rtol=0, atol=1e-7)
    saved = {name: value.clone() for name, value in state.record().items()}
    owner.protocol.max_seen_id.zero_()
    owner.latched_height.zero_()
    for name, value in JumpTaskState.decode(saved).record().items():
        torch.testing.assert_close(state.record()[name], value, rtol=0, atol=0)
    config = JumpTaskConfig.capture(native)
    assert JumpTaskConfig.from_dict(asdict(config)) == config
    native._terrain_course._pending_requests['jump'] = RequestCommand(torch.ones(2, dtype=torch.long),
                                                                     torch.full((2,), .06))
    with pytest.raises(ValueError, match='pending external'):
        JumpTaskState.capture(native)


def test_automatic_requests_and_four_tick_flight_landing_success_match_native():
    n, native = 5, native_jumps(5)
    owner = native._terrain_course.tasks
    owner.family[2] = int(TaskFamily.HEIGHT)
    owner.family[4], owner.base_mode[4] = int(TaskFamily.RECOVERY), int(CommandMode.RECOVERY)
    blocked = torch.tensor([False, False, False, True, False])
    task, state = JumpTask(JumpTaskConfig.capture(native)), JumpTaskState.capture(native)
    captured_state = state
    captured = {key: value.clone() for key, value in state.record().items()}
    completed = estimate_at(native.time, n)
    success_count = torch.zeros(n, dtype=torch.long)
    saw_flight = False
    for step in range(135):
        physical = physical_ticks(n)
        force = torch.full((n, 4, 2), 20.)
        if 11 <= step < 18:
            force[0] = 0.
            physical.position[0, :, 2] = .38
            physical.velocity[0, :, 2] = .4
        expected_phase, expected_height, expected_velocity, expected_stage = [], [], [], []
        for tick in range(4):
            owner.observe_physics(force[:, tick], physical.velocity[:, tick, 2], torch.zeros(n),
                physical.position[:, tick, 2], angular_speed_rad_s=torch.zeros(n), dt=.005)
            curve = owner.jump_reference()
            landing = ((owner.phase == int(RewardPhase.LANDING))
                       | (owner.phase == int(RewardPhase.SETTLING)))
            expected_phase.append(owner.phase.clone())
            expected_height.append(torch.where(landing, owner.stream.current[:, 3], curve.height_world_z_m))
            expected_velocity.append(torch.where(landing, 0., curve.velocity_world_z_m_s))
            expected_stage.append(curve.clock_stage.clone())
        result = task.observe_interval(state, physical=physical, wheel_support_load=force,
            base_com_offset=torch.zeros(n, 3), base_mode=owner.base_mode,
            command_height=owner.stream.current[:, 3], ground_z=torch.zeros(n, 4))
        same_state(result.state, native)
        torch.testing.assert_close(result.phase, torch.stack(expected_phase, -1), rtol=0, atol=0)
        torch.testing.assert_close(result.height_world_z_m, torch.stack(expected_height, -1), rtol=0, atol=0)
        torch.testing.assert_close(result.velocity_world_z_m_s, torch.stack(expected_velocity, -1), rtol=0, atol=0)
        torch.testing.assert_close(result.reference_stage, torch.stack(expected_stage, -1), rtol=0, atol=0)
        success_count += result.success_credited.long()
        saw_flight |= bool((result.phase == int(RewardPhase.FLIGHT)).any())
        state = advance(native, task, result.state, completed, blocked=blocked).state
        completed = estimate_at(native.time, n)
        completed.wheel_contact_probability[0] = .1 if 11 <= step < 18 else .9
    assert saw_flight
    assert success_count.tolist() == [1, 0, 0, 0, 0]  # Standing still never qualifies as a jump.
    assert owner.paid_id[0] == 0
    for key, value in captured.items():
        # All protocol/physics methods own their state; the original start is reusable.
        torch.testing.assert_close(value, captured_state.record()[key], rtol=0, atol=0)


def test_manual_requests_replay_staleness_cancel_abort_and_timeout_match_native():
    n, native = 7, native_jumps(7)
    owner = native._terrain_course.tasks
    owner.protocol.config = replace(owner.protocol.config, timeout_s=2.2)
    task, state = JumpTask(JumpTaskConfig.capture(native)), JumpTaskState.capture(native)
    first_id = 2**54 + 17
    blocked = torch.zeros(n, dtype=torch.bool)
    for step in range(113):
        estimate = estimate_at(native.time, n)
        packet = RequestCommand(torch.full((n,), -1, dtype=torch.long), torch.full((n,), .06),
                                torch.full((n,), -1, dtype=torch.long), torch.zeros(n, dtype=torch.bool))
        if step == 0:
            packet.request_id.fill_(first_id)
            estimate.available_at_s[1] -= 1.
            blocked[2] = True
            packet.height_m[3] = .9
        elif step == 1:
            packet.request_id.fill_(first_id)
            packet.cancel_id[0] = first_id
        elif step == 2:
            packet.request_id[4] = first_id + 1
            packet.safety_abort[5] = True
            blocked.zero_()
        result = advance(native, task, state, estimate, packet=packet, blocked=blocked)
        if step == 0:
            assert result.response.tolist() == [1, 5, 5, 4, 1, 1, 1]
        elif step == 1:
            assert (result.response == int(RequestResponse.REPLAY)).all()
            assert result.end_reason[0] == int(RequestEnd.CANCELLED)
        elif step == 2:
            assert result.response[4] == int(RequestResponse.BUSY)
            assert result.end_reason[5] == int(RequestEnd.SAFETY_ABORT)
        elif step == 110:
            assert result.end_reason[6] == int(RequestEnd.TIMEOUT)
        state = result.state


def test_com_velocity_and_resultant_contact_semantics_not_origin_or_normal_load():
    native = native_jumps()
    task, state = JumpTask(JumpTaskConfig.capture(native)), JumpTaskState.capture(native)
    state = advance(native, task, state, estimate_at(native.time, 1)).state
    physical = physical_ticks(1, z=.34)
    physical.angular_velocity[..., 1] = -.3
    com_offset = torch.tensor([[.6, 0., 0.]])  # Synthetic geometry makes the semantic difference visible.
    torch.testing.assert_close(task.com_velocity(physical, com_offset)[..., 2], torch.full((1, 4), .18))
    airborne = task.observe_interval(state, physical=physical, wheel_support_load=torch.zeros(1, 4, 2),
        base_com_offset=com_offset, base_mode=torch.zeros(1, dtype=torch.long),
        command_height=torch.full((1,), .32), ground_z=torch.zeros(1, 4))
    assert airborne.state.values[0, 11] == 1  # ascent_seen depends on COM vz > .15.
    assert airborne.state.values[0, 10] == 1
    # Nonvertical contact can have zero normal load but a nonzero resultant.
    supported = task.observe_interval(state, physical=physical, wheel_support_load=torch.full((1, 4, 2), 6.),
        base_com_offset=com_offset, base_mode=torch.zeros(1, dtype=torch.long),
        command_height=torch.full((1,), .32), ground_z=torch.zeros(1, 4))
    assert supported.state.values[0, 10] == 0
    assert (supported.phase == int(RewardPhase.TAKEOFF)).all()


def test_reference_uses_one_visible_clock_and_event_state_stops_gradient():
    native = native_jumps()
    task, state = JumpTask(JumpTaskConfig.capture(native)), JumpTaskState.capture(native)
    state = advance(native, task, state, estimate_at(native.time, 1)).state
    values = state.values.clone()
    values[:, 8] = .15
    state = replace(state, values=values)
    physical = physical_ticks(1, z=.36, vz=.3, requires_grad=True)
    force = torch.zeros(1, 4, 2, requires_grad=True)
    result = task.observe_interval(state, physical=physical, wheel_support_load=force,
        base_com_offset=torch.zeros(1, 3), base_mode=torch.zeros(1, dtype=torch.long),
        command_height=torch.full((1,), .32), ground_z=torch.zeros(1, 4))
    assert result.phase.tolist() == [[int(RewardPhase.TAKEOFF), *([int(RewardPhase.FLIGHT)] * 3)]]
    assert result.height_world_z_m.unique().numel() == 1
    assert not result.state.values.requires_grad
    assert not result.height_world_z_m.requires_grad
    # Continuous tracking cost still differentiates through predicted motion.
    loss = (physical.position[..., 2] - result.height_world_z_m).square().sum()
    loss.backward()
    assert physical.position.grad[..., 2].abs().sum() > 0
    assert force.grad is None
    ground = torch.full((1, 4), .02, requires_grad=True)
    height = torch.tensor([.32], requires_grad=True)
    landing = task.observe_interval(result.state, physical=physical_ticks(1),
        wheel_support_load=torch.full((1, 4, 2), 20.), base_com_offset=torch.zeros(1, 3),
        base_mode=torch.zeros(1, dtype=torch.long), command_height=height, ground_z=ground)
    landing.height_world_z_m.sum().backward()
    torch.testing.assert_close(ground.grad, torch.ones_like(ground))
    torch.testing.assert_close(height.grad, torch.tensor([4.]))


def test_native_atan2_tilt_threshold_and_com_rotation_are_preserved():
    native = native_jumps(2)
    task, state = JumpTask(JumpTaskConfig.capture(native)), JumpTaskState.capture(native)
    physical = physical_ticks(2)
    rotation_vector = torch.zeros(2, 4, 3)
    rotation_vector[0, :, 1] = .249999
    rotation_vector[1, :, 1] = .250001
    physical = replace(physical, rotation=so3.exp(rotation_vector))
    result = task.observe_interval(state, physical=physical, wheel_support_load=torch.full((2, 4, 2), 20.),
        base_com_offset=torch.zeros(2, 3), base_mode=torch.zeros(2, dtype=torch.long),
        command_height=torch.full((2,), .32), ground_z=torch.zeros(2, 4))
    assert result.state.values[:, 13].tolist() == [1., 0.]
    # A body-local x offset rotated 90 degrees about y lies along world -z.
    physical = physical_ticks(1)
    rotation_vector = torch.zeros(1, 4, 3)
    rotation_vector[..., 1] = torch.pi / 2
    physical = replace(physical, rotation=so3.exp(rotation_vector))
    physical.angular_velocity[..., 0] = 2.
    velocity = task.com_velocity(physical, torch.tensor([[.1, 0., 0.]]))
    torch.testing.assert_close(velocity, torch.tensor([[[0., .2, 0.]]]).expand(1, 4, 3), rtol=0, atol=2e-8)


def test_estimated_completion_does_not_erase_pending_physical_outcome():
    native = native_jumps()
    task, state = JumpTask(JumpTaskConfig.capture(native)), JumpTaskState.capture(native)
    state = advance(native, task, state, estimate_at(native.time, 1)).state
    owner = native._terrain_course.tasks
    owner.protocol.flight_seen.fill_(True)
    owner.protocol.landing_since_s.fill_(.02)
    owner.protocol.last_time_s.fill_(2.)
    native.time.fill_(2.)
    state = JumpTaskState.capture(native)
    result = advance(native, task, state, estimate_at(native.time, 1))
    assert result.end_reason.item() == int(RequestEnd.ESTIMATED_COMPLETE)
    assert result.state.active_id.item() == -1
    assert result.state.physical_id.item() == 0


def test_critic_features_keep_causal_latches_without_absolute_ids_time_or_height():
    native = native_jumps()
    config = JumpTaskConfig.capture(native)
    task, state = JumpTask(config), JumpTaskState.capture(native)
    state = advance(native, task, state, estimate_at(native.time, 1)).state
    physical = physical_ticks(1).tick(0)
    motion = SimpleNamespace(timing=torch.tensor([[.02, 0., 0., 0., 0.]]))
    features = state.critic_features(physical, motion, config)
    assert features.shape == (1, JUMP_FEATURE_DIM) and torch.isfinite(features).all()
    values, ids, clock = state.values.clone(), state.ids.clone(), state.clock.clone()
    values[:, 3] += 8.
    values[:, [7, 14]] += 2.
    ids[:, [0, 1, 3, 4]] += 2**54
    clock += 8.
    shifted = JumpTaskState(ids, clock, values)
    shifted_motion = SimpleNamespace(timing=motion.timing + torch.tensor([[8., 0., 0., 0., 0.]]))
    shifted_physical = replace(physical, position=physical.position + torch.tensor([[0., 0., 2.]]))
    torch.testing.assert_close(shifted.critic_features(shifted_physical, shifted_motion, config), features,
                               rtol=0, atol=1e-6)
    changed = values.clone()
    changed[:, 15:17] = torch.tensor([[.05, .8]])
    assert not torch.equal(JumpTaskState(ids, clock, changed).critic_features(
        shifted_physical, shifted_motion, config), features)


def test_gated_automatic_requests_match_native_pre_admission_order():
    from supermimic_rl.course.jump_takeoff import install_jump_takeoff

    n, native = 7, native_jumps(7)
    install_jump_takeoff(native, dict(min_episode_s=.06, cooldown_s=.04))
    owner, gate = native._terrain_course.tasks, native.supermimic_jump_trigger
    native.time.fill_(.04)
    owner.protocol.last_time_s.fill_(.04)
    owner.elapsed.fill_(.04)
    owner._last_supported[1] = False
    owner._last_world_height[2] += .10
    gate.last_end_s[3] = .03
    owner.family[4] = int(TaskFamily.HEIGHT)
    blocked = torch.zeros(n, dtype=torch.bool)
    blocked[5] = True
    # Gate sees the prior .32 m stream; the next admitted target becomes .40 m.
    owner.trajectory.start[6, 3] = owner.trajectory.end[6, 3] = .40
    config, state = JumpTaskConfig.capture(native), JumpTaskState.capture(native)
    assert config.trigger is not None and JumpTaskConfig.from_dict(asdict(config)) == config
    result = advance(native, JumpTask(config), state, estimate_at(native.time, n), blocked=blocked)
    assert result.deferred.tolist() == [False, True, True, True, False, False, False]
    assert gate.deferred == result.deferred.sum()
    assert result.state.active_id.ge(0).tolist() == [True, False, False, False, False, False, True]
    assert result.command[6, 3] == pytest.approx(.40)
    assert result.state.trigger_flags[:, 0].all()
    torch.testing.assert_close(result.state.trigger_flags[:, 1], result.state.active_id >= 0)


def test_gate_manual_bypass_active_request_and_cooldown_match_native():
    from supermimic_rl.course.jump_takeoff import install_jump_takeoff

    native = native_jumps()
    install_jump_takeoff(native, dict(min_episode_s=.06, cooldown_s=.06))
    owner, gate = native._terrain_course.tasks, native.supermimic_jump_trigger
    owner._last_supported.zero_()
    task, state = JumpTask(JumpTaskConfig.capture(native)), JumpTaskState.capture(native)
    packet = RequestCommand(torch.tensor([8]), torch.tensor([.06]))
    state = advance(native, task, state, estimate_at(native.time, 1), packet=packet).state
    assert state.active_id.item() == 8 and gate.deferred.item() == 0
    result = advance(native, task, state, estimate_at(native.time, 1))
    assert not result.deferred.any() and not owner._physical_cancelled.any()
    cancel = RequestCommand(torch.tensor([-1]), torch.zeros(1), cancel_id=torch.tensor([8]))
    state = advance(native, task, result.state, estimate_at(native.time, 1), packet=cancel).state
    assert state.trigger_clock.item() == pytest.approx(.06)
    owner._last_supported.fill_(True)
    owner._last_world_height.fill_(.32)
    state = JumpTaskState.capture(native)
    result = advance(native, task, state, estimate_at(native.time, 1))
    assert result.deferred.item() and result.state.active_id.item() == -1
    result = advance(native, task, result.state, estimate_at(native.time, 1))
    assert result.deferred.item()
    result = advance(native, task, result.state, estimate_at(native.time, 1))
    assert not result.deferred.item()
    result = advance(native, task, result.state, estimate_at(native.time, 1))
    assert result.state.active_id.item() == 9 and not result.deferred.item()
    assert gate.deferred.item() == 2


def test_gate_critic_carries_relative_cooldown_and_episode_readiness():
    from supermimic_rl.course.jump_takeoff import install_jump_takeoff

    native = native_jumps()
    install_jump_takeoff(native, {})
    native.time.fill_(2.)
    native.supermimic_jump_trigger.last_end_s.fill_(1.8)
    config, state = JumpTaskConfig.capture(native), JumpTaskState.capture(native)
    motion = SimpleNamespace(timing=torch.tensor([[.4, 0., 0., 0., 0.]]))
    features = state.trigger_critic_features(motion, config)
    torch.testing.assert_close(features, torch.tensor([[.8, .6, 0.]]))
    shifted = replace(state, clock=state.clock + 100., trigger_clock=state.trigger_clock + 100.)
    torch.testing.assert_close(shifted.trigger_critic_features(motion, config), features)
    with pytest.raises(ValueError, match='known native clock/state'):
        replace(state, trigger_flags=torch.zeros_like(state.trigger_flags)).trigger_critic_features(motion, config)
    legacy = JumpTaskState(state.ids, state.clock, state.values)
    with pytest.raises(ValueError, match='known native clock/state'):
        JumpTask(config).advance(legacy, now_s=native.time + .02, estimate=None,
            admitted=torch.zeros(1, 4), base_mode=torch.zeros(1, dtype=torch.long),
            task_family=torch.full((1,), 4), elapsed_s=torch.ones(1), ground_z=torch.zeros(1),
            blocked_by_stair=torch.zeros(1, dtype=torch.bool), interval_uniform=torch.zeros(1))
