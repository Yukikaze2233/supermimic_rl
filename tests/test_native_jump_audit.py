"""Audit prescribed physical truth through actual upstream tasks; detect damaged records."""
from dataclasses import asdict, replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.formal_observation import SensorReadout, deployed_frame
from supermimic_rl.world_model_v2.jump_task import JumpTaskConfig, JumpTaskState
from supermimic_rl.world_model_v2.motion_task import MotionTask
from supermimic_rl.world_model_v2.physical import increments
from supermimic_rl.world_model_v2.records import encode_physical, encode_queues
from supermimic_rl.world_model_v2.replay import CompactReplay
from supermimic_rl.world_model_v2.sensing import SensorQueues
from supermimic_rl.world_model_v2.stair_source import StairScenario, StairSourceState
from supermimic_rl.world_model_v2.stair_task import StairTask, StairTaskState
from transformer_rl.formal.lab_env.course_tasks import StairRequestConfig, StairRequestProtocol
from transformer_rl.formal.tasks.gimbal_course import GimbalCourse, GimbalCourseConfig
from transformer_rl.formal.tasks.trajectory import CommandTrajectory

from support.world_v2.jump_task import native_jumps
from support.world_v2.reward_assembly import setup
from support.world_v2.stair_source import source_fixture
from support.world_v2.stair_task import native_stairs


def module():
    path = Path(__file__).resolve().parents[1] / 'scripts/check_native_jump_task.py'
    spec = importlib.util.spec_from_file_location('native_jump_audit', path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def records(steps=110, *, gated=False):
    n = 2
    _, _, geometry, physical, _, _ = setup('rough_dash')
    physical = replace(physical, position=torch.tensor([[-2., 0., .35]]).repeat(n, 1))
    native, jumps = native_stairs(n, assist=True), native_jumps(n)
    source, trigger = source_fixture(n)
    native.physics_dt = .005
    course, tasks = native._terrain_course, jumps._terrain_course.tasks
    course.tasks = tasks
    if gated:
        from supermimic_rl.course.jump_takeoff import install_jump_takeoff
        install_jump_takeoff(native, {'min_episode_s': .10, 'cooldown_s': .10})
    course.stair_requests = StairRequestProtocol(n, config=StairRequestConfig(policy_dt_s=.02, timeout_s=6.))
    course.stair_requests.reset(torch.arange(n), native.time)
    course.tof_trigger, course.query = trigger, source.terrain.query
    course.router = SimpleNamespace(tile=torch.zeros(n, dtype=torch.int64), origins=source.terrain.origins.expand(n, -1))
    course._wheel_radius = .08
    course._stair_course_contact_allowed.zero_()
    del course._tof_command, course._tof_rear_clearance
    native._gimbal_course = GimbalCourse(GimbalCourseConfig(enabled=True), n)
    native._gimbal_course.remaining_s.fill_(100.)
    tasks.stream.current[:] = torch.tensor([0., 0., 0., .35])
    tasks.trajectory = CommandTrajectory(tasks.stream.current.clone(), tasks.stream.current.clone(),
        torch.full((n,), 10.), torch.zeros(n, dtype=torch.int64))
    tasks.segment_end_s.fill_(10.)
    tasks._last_world_height.fill_(3.35)
    native._command.copy_(tasks.stream.current)
    native._command_mode.zero_()
    native._ground_at_root.fill_(3.)
    native.physical = physical
    native._root_link_position = lambda: native.physical.position + course.router.origins
    native._root_link_quat = lambda: torch.tensor([0., 0., 0., 1.]).expand(n, -1)
    native.heading_estimate = lambda: torch.zeros(n)
    course._wheel_centres = lambda: geometry(native.physical).centre_world + course.router.origins[:, None]
    raw = torch.zeros(n, 3, 18)
    raw[:, :, 17] = -1.
    queues = SensorQueues(raw, torch.zeros(n, 3, 5), torch.tensor([1., 0.]).expand(n, 3, 2), torch.full((n,), 3.))
    readout = SensorReadout(torch.zeros(n, 4), torch.zeros(n, 1, dtype=torch.bool),
                           torch.zeros(n, 18), torch.zeros(n, 1), torch.zeros(n, 6))
    encoded_readout = torch.zeros(n, 31)
    encoded_readout[:, 23] = 1.  # Nominal, no sensor noise in this boundary fixture.
    estimate = torch.tensor([[0., 0., 0., 2., 2.]]).repeat(n, 1)
    schema = SimpleNamespace(policy_dt_s=.02, substeps_per_step=4, fingerprint='prescribed-native-physics-fixture')
    rng, replay = [], None
    for step in range(steps + 1):
        goal = torch.cat((native._command[:, :3], native._command[:, 3:4] * 5,
            torch.nn.functional.one_hot(native._command_mode, 5).float(), native._request_height[:, None] * 5,
            tasks.request_clock[:, None].clamp_max(5), native._gimbal_course.context(torch.zeros(n))[:, :5]), -1)
        frame = deployed_frame(queues.values, queues.count, queues.gimbal, torch.zeros(n, 6), goal, readout, noise=False)
        before = {**JumpTaskState.capture(native).record(), **StairTaskState.capture(native).record(),
            **StairSourceState.capture(native).record(), 'motion_task': MotionTask.capture(native),
            'physical': encode_physical(native.physical), 'goal': goal, 'frame': frame,
            'queues': encode_queues(queues), 'readout': encoded_readout, 'task_estimate': estimate,
            'terrain_tile': torch.zeros(n, 1), 'task_family': tasks.family[:, None].float(),
            'sim_context': torch.zeros(n, 65), 'jump_record_valid': torch.ones(n, 1),
            'task_record_valid': torch.ones(n, 1), 'stair_source_valid': torch.ones(n, 1),
            'failure_conditions': torch.stack((native._command_mode == 3, course.body_contact_allowed), -1).float()}
        airborne = step in (1, 2)
        height, velocity = (.38, .3) if step == 1 else (.41, .2) if step == 2 else (.35, 0.)
        position = native.physical.position[:, None].expand(-1, 4, -1).clone()
        position[:, :, 2] = height
        tick_state = replace(native.physical.with_time_axis(), position=position,
            rotation=native.physical.rotation[:, None].expand(-1, 4, -1, -1),
            velocity=torch.tensor([0., 0., velocity]).expand(n, 4, 3),
            angular_velocity=torch.zeros(n, 4, 3),
            joint_position=native.physical.joint_position[:, None].expand(-1, 4, -1),
            joint_velocity=torch.zeros(n, 4, 8))
        loads = torch.full((n, 4, 2), 0. if airborne else 80.)
        knee = geometry(tick_state)
        before['ticks'] = torch.cat((increments(native.physical.with_time_axis(), tick_state), loads,
            torch.zeros(n, 4, 21), torch.cat((loads, torch.zeros(n, 4, 1)), -1),
            knee.knee_angle_rad, knee.knee_velocity_rad_s), -1)
        before['end_physical'] = encode_physical(tick_state.tick(-1))
        before['end_height'] = position[:, -1, 2:3]
        for tick in range(4):
            tasks.observe_physics(loads[:, tick], tick_state.velocity[:, tick, 2], torch.zeros(n),
                position[:, tick, 2] + 3., angular_speed_rad_s=torch.zeros(n), dt=.005)
        rng.append(trigger.generator.get_state().clone())
        native._completed_estimate = StairTask.completed_estimate(frame[:, None], estimate, native.time)
        native.time = torch.full((n,), (step + 1) * 4 * .005, dtype=torch.float64)
        native.physical = tick_state.tick(-1)
        course.advance_skill()
        eligible = torch.zeros(n, dtype=torch.bool)
        native._gimbal_course.advance(.02, torch.zeros(n, 2), eligible, update=0,
                                     segment_boundary=torch.zeros(n, dtype=torch.bool))
        native._command.copy_(native._gimbal_course.body_command(native._command))
        if replay is None:
            replay = CompactReplay({name: value.shape[1:] for name, value in before.items()}, n, steps + 1, 'cpu')
        boundary = {name: torch.ones(n, dtype=torch.bool) for name in ('valid', 'policy_owned', 'end_policy_owned')}
        boundary.update({name: torch.zeros(n, dtype=torch.bool) for name in ('terminated', 'truncated')})
        boundary.update({name: torch.zeros(n, dtype=torch.int64) for name in
                         ('start_episode_id', 'end_episode_id', 'start_control_epoch', 'end_control_epoch')})
        replay.add(before, boundary)
    return replay, schema, {'jump_config': asdict(JumpTaskConfig.capture(native)),
        'stair_scenario': StairScenario.capture(native).to_dict(), 'terrain': source.terrain, 'geometry': geometry,
        'source_rng': {'format': 'supermimic-vector-step-rng-v1', 'device': 'cpu', 'num_envs': n,
                      'sequence': replay.sequence.clone(), 'state': torch.stack(rng)}}


def test_native_jump_audit_replays_real_producers_and_reports_limits():
    replay, schema, arguments = records()
    report = module().audit(replay, schema, **arguments)
    assert report['passed'], report['first_mismatch']
    assert report['counts']['eligible_intervals'] == 220
    assert report['counts']['accepted_requests'] == 2
    assert report['counts']['flight_intervals'] > 0 and report['counts']['landing_intervals'] > 0
    assert report['counts']['physical_success_events'] == 2
    assert report['fields']['jump_ids']['mismatching_intervals'] == 0
    assert 'excluded from independent' in report['randomness']
    assert 'not recorded' in report['reference_scope']


def test_native_jump_audit_covers_cold_gated_records_and_deferred_deadlines():
    replay, schema, arguments = records(steps=15, gated=True)
    report = module().audit(replay, schema, **arguments)
    assert report['passed'], report['first_mismatch']
    assert report['counts']['eligible_intervals'] == 30
    assert report['counts']['accepted_requests'] == 2
    assert report['fields']['jump_trigger_clock']['mismatching_intervals'] == 0
    assert report['fields']['jump_trigger_flags']['mismatching_intervals'] == 0
    assert report['counts']['trigger_deferred_rows'] > 0


def test_native_jump_audit_detects_modified_peak_and_excludes_invalid_boundaries():
    replay, schema, arguments = records(8)
    replay.data['jump_values'][4, 0, 15] += .1
    report = module().audit(replay, schema, **arguments)
    assert not report['passed'] and report['fields']['jump_values']['mismatching_intervals'] > 0
    assert report['first_mismatch']['jump_values']['env'] == 0
    replay.data['valid'][4, 0] = False
    report = module().audit(replay, schema, **arguments)
    assert report['passed'], report['first_mismatch']
    assert report['counts']['reset_invalid_or_unowned'] == 2
    assert report['counts']['eligible_intervals'] == 14
    arguments['source_rng'] = {**arguments['source_rng'], 'device': 'cuda:0'}
    with pytest.raises(ValueError, match='native generator device'):
        module().audit(replay, schema, **arguments)
