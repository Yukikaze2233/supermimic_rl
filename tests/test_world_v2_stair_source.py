"""Ideal request timelines match native sensing/protocol/FSM before imagined use."""
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.formal_observation import SensorReadout
from supermimic_rl.course.stair_label import install_tof_stair_label
from supermimic_rl.world_model_v2.motion_task import MotionTask
from supermimic_rl.world_model_v2.stair_source import StairSourceState
from supermimic_rl.world_model_v2.stair_task import StairPacket, StairTask, StairTaskConfig, StairTaskState
from supermimic_rl.world_model_v2.sensing import SensorQueues
from transformer_rl.formal.lab_env.course_tasks import LabCourseTasks, RequestCommand, StairRequestConfig, StairRequestProtocol
from transformer_rl.formal.control.observation import CommandMode
from transformer_rl.formal.tasks.gimbal_course import GimbalCourse, GimbalCourseConfig
from transformer_rl.formal.tasks.trajectory import CommandTrajectory, DriveEnvelope

from support.world_v2.stair_task import native_stairs, estimate_at
from support.world_v2.terrain import state as physical_state
from support.world_v2.stair_source import source_fixture


def test_presampled_native_rng_order_and_all_inactive_draws_match_native_trigger():
    source, native = source_fixture(8)
    n, horizon = 8, 40
    original = StairSourceState.from_trigger(native, torch.arange(n, dtype=torch.int64) + 2**54)
    imagined = source.restore(original)
    generator = torch.Generator().set_state(native.generator.get_state())
    random = source.draw(n, horizon, device='cpu', generator=generator)
    fired_total = stale_total = 0
    for step in range(horizon):
        position = torch.tensor([-.813 + .037 * step, 0., .4]).expand(n, 3).clone()
        goal = torch.zeros(n, 16)
        goal[:, 0] = .8
        goal[1, 0] = -.8 if step % 5 == 0 else .8
        goal[2, 0] = 0.
        goal[7, 0] = -.8 if int(native.countdown[7]) == 0 else .8
        beam = source.profile(physical_state(position), torch.zeros(n), goal)
        idle = torch.arange(n) % 2 == (step % 2)
        expected_fire, expected_height = native.step(beam, idle=idle)
        fire, height = source.step(imagined, beam, idle=idle,
            latency=random['tof_latency'][:, step], draws=random['tof_uniform'][:, step])
        torch.testing.assert_close(fire, expected_fire)
        torch.testing.assert_close(height, expected_height, atol=0, rtol=0)
        rearm = fire & (torch.arange(n) % 3 == step % 3)
        native.arm(rearm)
        source.arm(imagined, rearm, random['tof_uniform'][:, step, 4:7])
        expected = StairSourceState.from_trigger(native, original.ids[:, 0])
        actual = StairSourceState.from_trigger(imagined, original.ids[:, 0])
        for key, value in expected.record().items():
            torch.testing.assert_close(actual.record()[key], value, atol=0, rtol=0)
        fired_total += int(fire.sum())
        stale_total = int(native.counts[1])
    assert fired_total > 0 and stale_total > 0
    assert torch.equal(generator.get_state(), native.generator.get_state())
    assert (original.ids[:, 0] == torch.arange(n) + 2**54).all()
    assert not original.flags[:, 3].any()


@pytest.mark.parametrize('labels', [False, True])
def test_two_riser_source_completion_and_command_chain_match_native_course(labels):
    n, steps = 4, 72
    source, trigger = source_fixture(n)
    native = native_stairs(n, labels=labels, assist=True)
    course = native._terrain_course
    course.stair_requests = StairRequestProtocol(n, config=StairRequestConfig(policy_dt_s=.02, timeout_s=6.))
    course.stair_requests.reset(torch.arange(n), native.time)
    course.tof_trigger = trigger
    course.query = source.terrain.query
    course.router = SimpleNamespace(tile=torch.zeros(n, dtype=torch.int64),
                                    origins=source.terrain.origins.expand(n, -1))
    course._wheel_radius = .08
    # Remove fixture overrides: exercise the actual native profile/source/completion.
    del course._tof_command, course._tof_rear_clearance
    tasks = LabCourseTasks(n, DriveEnvelope(.08, .38, 90.), policy_dt_s=.02)
    course.tasks = tasks
    if labels:
        install_tof_stair_label(native)
    tasks.stream.current[:] = torch.tensor([.8, 0., 0., .30])
    tasks.trajectory = CommandTrajectory(tasks.stream.current.clone(), tasks.stream.current.clone(),
                                         torch.full((n,), 5.), torch.zeros(n, dtype=torch.int64))
    tasks.segment_end_s.fill_(5.)
    tasks.segment_kind.fill_(1)
    tasks.terrain_kind.fill_(3)
    native._gimbal_course = GimbalCourse(GimbalCourseConfig(enabled=True), n)
    native._gimbal_course.remaining_s.fill_(10.)
    native._command.copy_(tasks.stream.current)
    native._root_link_position = lambda: native.position + course.router.origins
    native._root_link_quat = lambda: torch.tensor([0., 0., 0., 1.]).expand(n, -1)
    course._wheel_centres = lambda: native._root_link_position()[:, None] + torch.tensor([[[0., -.2, -.3], [0., .2, -.3]]])
    source_state = StairSourceState.capture(native)
    captured_state = source_state
    captured = {key: value.clone() for key, value in source_state.record().items()}
    stair_state, task = StairTaskState.capture(native), StairTask(StairTaskConfig.capture(native))
    motion = MotionTask.decode(MotionTask.capture(native))
    goal = torch.zeros(n, 16)
    goal[:, :3], goal[:, 3], goal[:, 4], goal[:, 11] = native._command[:, :3], 1.5, 1., 1.
    queues = SensorQueues(torch.zeros(n, 3, 18), torch.zeros(n, 3, 5),
                          torch.tensor([1., 0.]).expand(n, 3, 2), torch.full((n,), 3.))
    readout = SensorReadout(torch.zeros(n, 4), torch.zeros(n, 1, dtype=torch.bool),
                            torch.zeros(n, 18), torch.zeros(n, 1), torch.zeros(n, 6))
    random = source.draw(n, steps, device='cpu', generator=torch.Generator().set_state(trigger.generator.get_state()))
    completions = requests = 0
    for step in range(steps):
        native.position = torch.tensor([-.813 + .037 * step, .17, .4]).expand(n, 3).clone()
        previous = native.time.clone()
        native.time = torch.full((n,), (step + 1) * 4 * .005, dtype=torch.float64)
        estimate = estimate_at(previous, n)
        native._completed_estimate = estimate
        external = StairPacket.empty(n, 'cpu')
        if step == 4:
            external.request_id[3], external.height_m[3] = 17, .12
        if step == 15:
            external.cancel_id[3] = 17
        if step == 23:
            external.completed_id[0] = 999  # A stale packet cannot consume a valid simulated release.
        result = source.advance(source_state, stair_state, task=task, motion=motion,
            physical=physical_state(native.position), wheel_centres=course._wheel_centres() - course.router.origins[:, None],
            wheel_radius=.08, terrain_tile=course.router.tile, queues=queues, readout=readout,
            goal=goal, estimate=estimate, physics_dt_s=.005,
            latency=random['tof_latency'][:, step], draws=random['tof_uniform'][:, step], external=external)
        course._pending_requests['stair'] = RequestCommand(external.request_id, external.height_m,
            external.cancel_id, external.safety_abort, completed_id=external.completed_id)
        course.advance_skill()
        if labels:
            torch.testing.assert_close(
                native._command_mode == int(CommandMode.STAIR), native._skill_fsm.stair.active)
        boundary = result.boundary
        for key, value in StairTaskState.capture(native).record().items():
            torch.testing.assert_close(boundary.stair.state.record()[key], value, rtol=0, atol=1e-6)
        expected = StairSourceState.capture(native)
        for key, value in expected.record().items():
            torch.testing.assert_close(result.source.record()[key], value, rtol=0, atol=2e-5)
        torch.testing.assert_close(boundary.motion.encode(), MotionTask.capture(native), atol=1e-6, rtol=0)
        torch.testing.assert_close(boundary.goal[:, :3], native._command[:, :3], atol=1e-6, rtol=0)
        torch.testing.assert_close(boundary.goal[:, 3] / 5., native._command[:, 3], atol=1e-6, rtol=0)
        torch.testing.assert_close(boundary.goal[:, 4:10].argmax(-1), native._command_mode)
        torch.testing.assert_close(boundary.failure_conditions[:, 1].bool(), course.body_contact_allowed)
        requests += int(result.fired.sum())
        completions += int((result.packet.completed_id >= 0).sum())
        source_state, stair_state = result.source, boundary.stair.state
        motion, goal = boundary.motion, boundary.goal
    assert requests >= 6 and completions >= 6  # Both risers in the automatic rows.
    for key, value in captured.items():
        assert torch.equal(captured_state.record()[key], value)


def test_pending_external_packet_is_not_silently_replaced_by_ideal_source():
    native = native_stairs(1)
    native._terrain_course._pending_requests['stair'] = object()
    with pytest.raises(ValueError, match='pending external'):
        StairSourceState.capture(native)


def test_source_replay_preserves_flags_and_large_ids(tmp_path):
    from supermimic_rl.world_model_v2.replay import CompactReplay
    from supermimic_rl.world_model_v2.stair_source import TOF_RECORD_FIELDS

    source, native = source_fixture(2)
    original = StairSourceState.from_trigger(native, torch.tensor([2**54 + 17, 2**54 + 18]))
    replay = CompactReplay({**TOF_RECORD_FIELDS, 'frame': (45,)}, 2, 2, 'cpu')
    boundary = {key: torch.ones(2, dtype=torch.bool) for key in ('valid', 'policy_owned', 'end_policy_owned')}
    boundary.update({key: torch.zeros(2, dtype=torch.bool) for key in ('terminated', 'truncated')})
    boundary.update({key: torch.zeros(2, dtype=torch.int64) for key in
                     ('start_episode_id', 'end_episode_id', 'start_control_epoch', 'end_control_epoch')})
    replay.add({**original.record(), 'frame': torch.zeros(2, 45)}, boundary)
    path = tmp_path/'replay.pt'
    torch.save(replay.state_dict(), path)
    restored = CompactReplay.from_state_dict(torch.load(path, weights_only=True))
    record, _ = restored.sample_starts(8, 5)
    captured = StairSourceState.decode(record)
    assert captured.flags.dtype == torch.bool and captured.ids.dtype == torch.int64
    assert (captured.ids[:, 0] >= 2**54 + 17).all()
    assert torch.isfinite(captured.values).all()
    trigger = source.restore(captured)
    assert torch.isinf(trigger.last_riser_xy).all()
