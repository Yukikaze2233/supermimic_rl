"""Recorded requests reproduce the pinned LabTerrainCourse deployment path."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.world_model_v2.replay import CompactReplay
from supermimic_rl.world_model_v2.motion_task import MotionTask
from supermimic_rl.world_model_v2.sensing import SensorQueues, heading_sample
from supermimic_rl.formal_observation import SensorReadout
from supermimic_rl.world_model_v2.stair_task import (
    STAIR_RECORD_FIELDS, StairPacket, StairTask, StairTaskConfig, StairTaskState,
)
from transformer_rl.formal.lab_env.course_tasks import LabCourseTasks, RequestCommand
from transformer_rl.formal.control.sensors import SensorModel
from transformer_rl.formal.tasks.protocol import RequestResponse
from transformer_rl.formal.tasks.gimbal_course import GimbalCourse, GimbalCourseConfig
from transformer_rl.formal.tasks.trajectory import CommandRates, CommandStream, CommandTrajectory, DriveEnvelope
from support.world_v2.stair_task import estimate_at, native_stairs


@pytest.mark.parametrize('labels,assist', [(False, False), (False, True), (True, False), (True, True)])
def test_request_timeline_matches_native_terrain_course(labels, assist):
    n, first_id = 10, 2**54 + 17
    native = native_stairs(n, labels=labels, assist=assist)
    task = StairTask(StairTaskConfig.capture(native))
    state = StairTaskState.capture(native)
    captured_start = state
    original = {key: value.clone() for key, value in state.record().items()}
    for step in range(18):
        previous_time = native.time.clone()
        native.time = torch.full((n,), (step + 1) * .02, dtype=torch.float64)
        native.admitted = torch.tensor([.8, 0., .2, .26 + .002 * step]).repeat(n, 1)
        native.base_mode = torch.zeros(n, dtype=torch.int64)
        native.base_mode[9] = 3  # Recovery keeps its own command and contact semantics.
        native._completed_estimate = estimate_at(previous_time, n)
        packet = StairPacket.empty(n, 'cpu')
        if step == 0:
            packet.request_id.fill_(first_id)
            packet.height_m[:] = torch.tensor([.08, .12, .20, .25, .10, .12, .20, .20, .08, .20])
            native._completed_estimate.available_at_s[4] -= 1.  # Stale estimate consumes/rejects request.
            native._terrain_course.tasks.active_id[5] = 1     # Jump has arbitration priority.
        elif step == 1:
            packet.request_id.fill_(first_id)
            packet.height_m.fill_(.10)  # Both accepted and rejected IDs are replays.
            native._terrain_course.tasks.active_id.fill_(-1)
        elif step == 2:
            packet.request_id[0] = first_id + 1  # New ID while active is BUSY, but consumed.
            packet.height_m[0] = .08
        elif step == 3:
            packet.safety_abort[6] = True
        elif step == 4:
            native._completed_estimate.tilt_rad[7] = .9
        elif step == 8:
            packet.cancel_id[2] = first_id
        elif step == 10:
            packet.completed_id[1] = first_id  # Completion beats timeout at the same boundary.
            packet.completed_id[8] = packet.cancel_id[8] = first_id
        elif step == 12:
            packet.request_id.fill_(first_id + 2)
            packet.height_m.fill_(.20)
            packet.completed_id.fill_(first_id + 2)  # Cannot complete a just-accepted request.
        elif step == 13:
            packet.completed_id.fill_(first_id + 2)
        result = task.advance(state, packet, now_s=native.time, estimate=native._completed_estimate,
                              admitted=native.admitted, base_mode=native.base_mode,
                              jump_active=native._terrain_course.tasks.active_id >= 0)
        native._terrain_course._pending_requests['stair'] = RequestCommand(
            packet.request_id, packet.height_m, packet.cancel_id, packet.safety_abort,
            completed_id=packet.completed_id)
        native._terrain_course.advance_skill()
        torch.testing.assert_close(result.command, native._command, rtol=0, atol=0)
        torch.testing.assert_close(result.mode, native._command_mode, rtol=0, atol=0)
        torch.testing.assert_close(result.body_contact_allowed, native._terrain_course.body_contact_allowed)
        torch.testing.assert_close(result.response, native._terrain_course._last_stair_response)
        for key, value in StairTaskState.capture(native).record().items():
            torch.testing.assert_close(result.state.record()[key], value, rtol=0, atol=0)
        if step == 0:
            assert result.response[:7].tolist() == [1, 1, 1, 4, 5, 3, 1]
        if step == 1:
            assert (result.response == int(RequestResponse.REPLAY)).all()
        if step == 2:
            assert result.response[0] == int(RequestResponse.BUSY)
        if step == 10:
            assert result.end_reason[[0, 1, 8]].tolist() == [4, 1, 2]
        if step == 12:
            assert (result.state.ids[:, 1] == first_id + 2).all()
        state = result.state
    # Native changes and imagined progression never mutate the captured start.
    for key, value in original.items():
        torch.testing.assert_close(captured_start.record()[key], value, rtol=0, atol=0)


def test_estimator_decoding_uses_latest_sensor_frame_and_si_velocities():
    history = torch.zeros(2, 5, 45)
    history[:, -1, 7:10] = torch.tensor([0., -.6, -.8])
    history[:, -1, 4] = .25  # Gyro is normalized by .5 in the deployed frame.
    estimate = torch.tensor([[1., 2., 3., -2., 2.]]).repeat(2, 1)
    decoded = StairTask.completed_estimate(history, estimate, torch.tensor([10., 20.], dtype=torch.float64))
    torch.testing.assert_close(decoded.velocity_z_m_s, torch.full((2,), 3.6))
    torch.testing.assert_close(decoded.tilt_rad, torch.full((2,), torch.acos(torch.tensor(.8))))
    torch.testing.assert_close(decoded.angular_speed_rad_s, torch.full((2,), .5))
    torch.testing.assert_close(decoded.wheel_contact_probability, estimate[:, 3:].sigmoid())
    assert decoded.valid.all()
    history[0, -1, 7:10] = 0.
    assert not StairTask.completed_estimate(history, estimate, decoded.available_at_s).valid[0]


def test_height_hint_retains_latched_command_gradient_and_stops_at_exact_tick():
    native = native_stairs(1, assist=True)
    task, state = StairTask(StairTaskConfig.capture(native)), StairTaskState.capture(native)
    first_height = torch.tensor([.26], requires_grad=True)
    second_height = torch.tensor([.28], requires_grad=True)
    for step in range(4):
        packet = StairPacket.empty(1, 'cpu')
        if step == 0:
            packet.request_id.fill_(1)
            packet.height_m.fill_(.20)
        admitted = torch.cat((torch.zeros(1, 3), (first_height if step == 0 else second_height)[:, None]), -1)
        now = torch.tensor([(step + 1) * .02], dtype=torch.float64)
        result = task.advance(state, packet, now_s=now, estimate=estimate_at(now - .02, 1),
            admitted=admitted, base_mode=torch.zeros(1, dtype=torch.int64), jump_active=torch.zeros(1, dtype=torch.bool))
        assert result.command[0, 3].item() == pytest.approx(.33 if step < 3 else .28)
        if step == 1:
            grad = torch.autograd.grad(result.command[:, 3].sum(), (first_height, second_height), retain_graph=True)
            assert grad[0].item() == 1. and grad[1].item() == 0.
        state = result.state


def test_empty_stair_reset_is_skipped_and_only_unreleased_finished_rows_reset(monkeypatch):
    from transformer_rl.formal.lab_env.skill_fsm import StairFSM, StairState
    from transformer_rl.formal.tasks.protocol import RequestEnd

    native = native_stairs(4, labels=True, assist=True)
    task = StairTask(StairTaskConfig.capture(native))
    state = StairTaskState.capture(native)
    captured_start = state
    original = {key: value.clone() for key, value in state.record().items()}
    reset_calls = []
    reset = StairFSM.reset

    def recorded_reset(fsm, ids=None):
        reset_calls.append(ids.clone())
        reset(fsm, ids)
        assert (fsm.state[ids] == int(StairState.NORMAL)).all()
        assert (fsm.state_time[ids] == 0).all()
        assert (fsm.target_height[ids] == 0).all()

    monkeypatch.setattr(StairFSM, 'reset', recorded_reset)
    for step in range(4):
        packet = StairPacket.empty(4, 'cpu')
        now = torch.full((4,), (.02, .04, .06, .26)[step], dtype=torch.float64)
        if step == 1:
            packet.request_id.fill_(7)
            packet.height_m.fill_(.20)
        elif step == 2:
            packet.cancel_id[0] = 7
            packet.completed_id[1] = 7
            packet.safety_abort[2] = True
        result = task.advance(state, packet, now_s=now, estimate=estimate_at(now - .02, 4),
            admitted=torch.tensor([[.5, 0., 0., .26]]).repeat(4, 1),
            base_mode=torch.zeros(4, dtype=torch.long), jump_active=torch.zeros(4, dtype=torch.bool))
        if step <= 1:
            assert reset_calls == []  # Idle and accepted requests have nothing to reset.
        elif step == 2:
            assert len(reset_calls) == 1
            torch.testing.assert_close(reset_calls[0], torch.tensor([0, 2]), atol=0, rtol=0)
            assert result.end_reason.tolist() == [int(RequestEnd.CANCELLED),
                int(RequestEnd.ESTIMATED_COMPLETE), int(RequestEnd.SAFETY_ABORT), int(RequestEnd.NONE)]
            assert result.state.ids[:, 1].tolist() == [-1, -1, -1, 7]
        else:
            assert len(reset_calls) == 2
            torch.testing.assert_close(reset_calls[1], torch.tensor([3]), atol=0, rtol=0)
            assert result.end_reason[3] == int(RequestEnd.TIMEOUT)
        state = result.state
    for key, value in original.items():
        torch.testing.assert_close(captured_start.record()[key], value, atol=0, rtol=0)


def test_compact_replay_keeps_large_request_ids_and_precise_clocks(tmp_path):
    native = native_stairs(2)
    native.time.fill_(1_000_000.01)
    native._terrain_course.stair_requests.max_seen_id.fill_(2**54 + 17)
    captured = StairTaskState.capture(native)
    replay = CompactReplay({**STAIR_RECORD_FIELDS, 'frame': (45,)}, 2, 3, 'cpu')
    boundary = {name: torch.ones(2, dtype=torch.bool) for name in ('valid', 'policy_owned', 'end_policy_owned')}
    boundary.update({name: torch.zeros(2, dtype=torch.bool) for name in ('terminated', 'truncated')})
    boundary.update({name: torch.zeros(2, dtype=torch.int64) for name in
        ('start_episode_id', 'end_episode_id', 'start_control_epoch', 'end_control_epoch')})
    record = {**captured.record(), 'frame': torch.zeros(2, 45)}
    replay.add(record, boundary)
    path = tmp_path / 'replay.pt'
    torch.save(replay.state_dict(), path)
    restored = CompactReplay.from_state_dict(torch.load(path, weights_only=True), env_ids=[1, 0])
    sampled, _ = restored.sample_starts(8, 5)
    assert sampled['stair_ids'].dtype == torch.int64
    assert sampled['stair_clock'].dtype == torch.float64
    assert (sampled['stair_ids'][:, 0] == 2**54 + 17).all()
    assert (sampled['stair_clock'][:, 0] == native.time[0]).all()
    before = replay.data['stair_ids'].clone()
    with pytest.raises(ValueError, match='native replay dtype'):
        replay.add({**record, 'stair_ids': record['stair_ids'].float()}, boundary)
    assert replay.total_steps == 1 and torch.equal(before, replay.data['stair_ids'])
    damaged = replay.state_dict()
    damaged['storage']['stair_clock'] = damaged['storage']['stair_clock'].float()
    with pytest.raises(ValueError, match='shape/dtype'):
        CompactReplay.from_state_dict(damaged)


def test_recorded_configuration_is_not_silently_reinterpreted():
    native = native_stairs(1)
    task = StairTask(replace(StairTaskConfig.capture(native), use_mode_labels=True))
    with pytest.raises(ValueError, match='parameters differ'):
        task.advance(StairTaskState.capture(native), StairPacket.empty(1, 'cpu'),
            now_s=native.time + .02, estimate=None, admitted=torch.zeros(1, 4),
            base_mode=torch.zeros(1, dtype=torch.int64), jump_active=torch.zeros(1, dtype=torch.bool))


@pytest.mark.parametrize('readiness', ['missing', 'current_tick', 'invalid_contacts', 'invalid_tilt'])
def test_request_cannot_use_missing_or_current_physics_truth(readiness):
    native = native_stairs(1)
    task, state = StairTask(StairTaskConfig.capture(native)), StairTaskState.capture(native)
    now = native.time + .02
    estimate = estimate_at(native.time, 1)
    if readiness == 'missing':
        estimate = None
    elif readiness == 'current_tick':
        estimate = replace(estimate, available_at_s=now)
    elif readiness == 'invalid_contacts':
        estimate.wheel_contact_probability.fill_(float('nan'))
    else:
        estimate.tilt_rad.fill_(float('nan'))
    packet = StairPacket.empty(1, 'cpu')
    packet.request_id.fill_(7)
    packet.height_m.fill_(.20)
    result = task.advance(state, packet, now_s=now, estimate=estimate,
        admitted=torch.tensor([[.5, 0., 0., .30]]), base_mode=torch.zeros(1, dtype=torch.int64),
        jump_active=torch.zeros(1, dtype=torch.bool))
    assert result.response.item() == int(RequestResponse.NOT_READY)
    assert result.state.ids[0, :2].tolist() == [7, -1]
    assert torch.isfinite(result.command).all()


@pytest.mark.parametrize('labels', [False, True])
def test_motion_stair_gimbal_boundary_matches_native_order_and_unassisted_stream(labels):
    n, dt, physics_dt = 4, .02, .005
    native = native_stairs(n, labels=labels, assist=True)
    tasks = LabCourseTasks(n, DriveEnvelope(.08, .38, 90.), policy_dt_s=dt)
    native._terrain_course.tasks = tasks
    tasks.elapsed.fill_(6.)
    tasks.segment_kind.fill_(1)
    start = torch.tensor([[.8, 0., .2, .26]]).repeat(n, 1)
    tasks.stream = CommandStream(start, tasks.stream.envelope, CommandRates(1., 3., .05), policy_dt_s=dt)
    tasks.trajectory = CommandTrajectory(start, start + torch.tensor([.3, 0., .4, .01]),
                                         torch.full((n,), 2.), torch.arange(n))
    tasks.heading_enabled[1] = True
    tasks.heading_offset_rad[1] = .1
    gimbal = GimbalCourse(GimbalCourseConfig(enabled=True, spin_translate=True), n)
    native._gimbal_course = gimbal
    gimbal.special[:3] = True
    gimbal.speed.fill_(.4)
    gimbal.direction[:] = torch.linspace(-1., 1., n)
    gimbal.chassis_rate.fill_(.8)
    gimbal.remaining_s.fill_(10.)
    native.time.fill_(6.)
    task, state = StairTask(StairTaskConfig.capture(native)), StairTaskState.capture(native)
    motion = MotionTask.decode(MotionTask.capture(native))
    goal = torch.zeros(n, 16)
    goal[:, 4] = 1.
    for step in range(16):
        previous = native.time.clone()
        native.time = torch.full((n,), (1200 + 4 * (step + 1)) * physics_dt, dtype=torch.float64)
        estimate = estimate_at(previous, n)
        native._completed_estimate = estimate
        yaw = torch.arange(n)[:, None] * .4 + torch.tensor([step, step - 1, step - 2])[None] * .05
        queues = SensorQueues(torch.zeros(n, 3, 18), torch.zeros(n, 3, 5), heading_sample(yaw), torch.full((n,), 3.))
        readout = SensorReadout(torch.tensor([[0., 1., 0., 2.]]).repeat(n, 1),
            torch.zeros(n, 1, dtype=torch.bool), torch.zeros(n, 18), torch.zeros(n, 1), torch.zeros(n, 6))
        sensor = SimpleNamespace(lags=readout.ages.long(), count=queues.count.long(), capacity=3,
                                 physical_tick=2, _rows=torch.arange(n), uplink_delay_s=None)
        native.heading_estimate = lambda: SensorModel._delayed_angle(
            sensor, queues.heading.flip(1).transpose(0, 1), group=3)
        packet = StairPacket.empty(n, 'cpu')
        if step == 0:
            packet.request_id[:2] = 7
            packet.height_m[:2] = .20
        elif step == 4:
            packet.request_id[2] = 8
            packet.height_m[2] = .12
        elif step == 6:
            packet.completed_id[:2] = 7
        boundary = task.advance_motion(state, packet, motion=motion, queues=queues, readout=readout,
            goal=goal, estimate=estimate, physics_dt_s=physics_dt)
        native._terrain_course._pending_requests['stair'] = RequestCommand(
            packet.request_id, packet.height_m, packet.cancel_id, packet.safety_abort,
            completed_id=packet.completed_id)
        native._terrain_course.advance_skill()
        eligible = ((tasks.terrain_kind == 0) & (tasks.segment_kind == 1) & (native._command_mode == 0)
                    & (tasks.elapsed >= gimbal.config.spin_after_episode_s)
                    & (native._command[:, :3] != 0).any(-1))
        gimbal.advance(dt, torch.zeros(n, 2), eligible, update=0, segment_boundary=tasks.segment_clock == 0.)
        native._command.copy_(gimbal.body_command(native._command))
        expected = torch.cat((native._command[:, :3], native._command[:, 3:4] * 5.,
            torch.nn.functional.one_hot(native._command_mode, 6).float(), torch.zeros(n, 1),
            gimbal.context(torch.zeros(n))[:, :5]), -1)
        torch.testing.assert_close(boundary.goal, expected, rtol=0, atol=1e-6)
        torch.testing.assert_close(boundary.motion.encode(), MotionTask.capture(native), rtol=0, atol=1e-6)
        torch.testing.assert_close(boundary.motion.current, tasks.stream.current, rtol=0, atol=1e-6)
        torch.testing.assert_close(boundary.failure_conditions[:, 1],
                                   native._terrain_course.body_contact_allowed.float())
        torch.testing.assert_close(boundary.stair.state.clock, StairTaskState.capture(native).clock, rtol=0, atol=0)
        if step == 0:
            assert boundary.goal[0, 3] > boundary.motion.current[0, 3] * 5.
            assert boundary.goal[0, 12].item() == float(not labels)
        state, motion, goal = boundary.stair.state, boundary.motion, boundary.goal
