"""Only completed timeouts unlock a fresh request at the same ideal ToF riser."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.cli import _native_schema_source
from supermimic_rl.contracts import Field, StateSchema
from supermimic_rl.course.factory import create_course
from supermimic_rl.course.stair_label import install_tof_stair_label
from supermimic_rl.course.stair_retry import install_stair_timeout_retry, timeout_retry_receipt
from supermimic_rl.formal_observation import SensorReadout
from supermimic_rl.world_model_v2.migrations.stair_hint import verify_stair_hint
from supermimic_rl.world_model_v2.auxiliary_data import AuxiliaryWorldData
from supermimic_rl.world_model_v2.motion_task import MotionTask
from supermimic_rl.world_model_v2.online import CHECKPOINT_FORMAT, OnlineConfig, OnlineTrainer
from supermimic_rl.world_model_v2.records import MODEL_CONTRACT
from supermimic_rl.world_model_v2.sensing import SensorQueues
from supermimic_rl.world_model_v2.stair_source import StairScenario, StairSourceState, TOF_RECORD_FIELDS
from supermimic_rl.world_model_v2.stair_task import StairPacket, StairTaskState
from transformer_rl.formal.lab_env.course_tasks import LabCourseTasks, RequestCommand
from transformer_rl.formal.tasks.gimbal_course import GimbalCourse, GimbalCourseConfig
from transformer_rl.formal.tasks.protocol import RequestEnd, RequestResponse
from transformer_rl.formal.tasks.trajectory import CommandTrajectory, DriveEnvelope
from support.world_v2.stair_source import source_fixture
from support.world_v2.stair_task import estimate_at, native_stairs
from support.world_v2.terrain import state as physical_state


def timeline(*, retry):
    n = 4
    source, trigger = source_fixture(n)
    native = native_stairs(n, labels=True, assist=True)
    course = native._terrain_course
    course.tof_trigger = trigger
    course.query = source.terrain.query
    course.router = SimpleNamespace(tile=torch.zeros(n, dtype=torch.int64),
                                    origins=source.terrain.origins.expand(n, -1))
    course._wheel_radius = .08
    # Use the frozen native implementation, including its real end counters.
    del course._tof_command, course._tof_rear_clearance, course._record_stair_end
    course.stair_end_counts = torch.zeros(len(RequestEnd), dtype=torch.int64)
    tasks = LabCourseTasks(n, DriveEnvelope(.08, .38, 90.), policy_dt_s=.02)
    course.tasks = tasks
    tasks.stream.current[:] = torch.tensor([.8, 0., 0., .30])
    tasks.trajectory = CommandTrajectory(tasks.stream.current.clone(), tasks.stream.current.clone(),
                                         torch.full((n,), 5.), torch.zeros(n, dtype=torch.int64))
    tasks.segment_end_s.fill_(5.)
    tasks.segment_kind.fill_(1)
    tasks.terrain_kind.fill_(3)
    native._gimbal_course = GimbalCourse(GimbalCourseConfig(enabled=True), n)
    native._gimbal_course.remaining_s.fill_(10.)
    native._command.copy_(tasks.stream.current)
    native.position = torch.tensor([-.15, .17, .4]).expand(n, 3).clone()
    native._root_link_position = lambda: native.position + course.router.origins
    native._root_link_quat = lambda: torch.tensor([0., 0., 0., 1.]).expand(n, -1)
    course._wheel_centres = lambda: native._root_link_position()[:, None] + torch.tensor(
        [[[0., -.2, -.3], [0., .2, -.3]]])
    install_tof_stair_label(native, height_hint_lifetime='request')
    install_stair_timeout_retry(native, enabled=retry)
    task, source = StairScenario.capture(native).build(source.terrain)
    goal = torch.zeros(n, 16)
    goal[:, :3], goal[:, 3], goal[:, 4], goal[:, 11] = native._command[:, :3], 1.5, 1., 1.
    queues = SensorQueues(torch.zeros(n, 3, 18), torch.zeros(n, 3, 5),
                          torch.tensor([1., 0.]).expand(n, 3, 2), torch.full((n,), 3.))
    readout = SensorReadout(torch.zeros(n, 4), torch.zeros(n, 1, dtype=torch.bool),
                            torch.zeros(n, 18), torch.zeros(n, 1), torch.zeros(n, 6))
    return native, task, source, goal, queues, readout


@pytest.mark.parametrize('retry', [False, True])
def test_native_and_imagined_retry_use_new_ids_original_latency_and_release_old_hint(retry):
    native, task, source, goal, queues, readout = timeline(retry=retry)
    course, n, steps = native._terrain_course, native.num_envs, 55
    source_state, stair_state = StairSourceState.capture(native), StairTaskState.capture(native)
    captured = deepcopy(source_state.record())
    original_source = source_state
    motion = MotionTask.decode(MotionTask.capture(native))
    generator = torch.Generator().set_state(course.tof_trigger.generator.get_state())
    random = source.draw(n, steps, device='cpu', generator=generator)
    fires = [[] for _ in range(n)]
    timeouts, ended = [], []
    previous_timeout = None
    for step in range(steps):
        previous = native.time.clone()
        native.time = torch.full((n,), (step + 1) * .02, dtype=torch.float64)
        native._completed_estimate = estimate_at(previous, n)
        external = StairPacket.empty(n, 'cpu')
        if step == 6:
            assert (course.stair_requests.active_id >= 0).all()
            external.cancel_id[1] = course.stair_requests.active_id[1]
            external.completed_id[2] = course.stair_requests.active_id[2]
            external.safety_abort[3] = True
        before = stair_state
        result = source.advance(source_state, stair_state, task=task, motion=motion,
            physical=physical_state(native.position),
            wheel_centres=course._wheel_centres() - course.router.origins[:, None],
            wheel_radius=.08, terrain_tile=course.router.tile, queues=queues, readout=readout,
            goal=goal, estimate=native._completed_estimate, physics_dt_s=.005,
            latency=random['tof_latency'][:, step], draws=random['tof_uniform'][:, step], external=external)
        course._pending_requests['stair'] = RequestCommand(external.request_id, external.height_m,
            external.cancel_id, external.safety_abort, completed_id=external.completed_id)
        course.advance_skill()
        boundary = result.boundary
        for name, expected in StairTaskState.capture(native).record().items():
            torch.testing.assert_close(boundary.stair.state.record()[name], expected, rtol=0, atol=1e-6)
        for name, expected in StairSourceState.capture(native).record().items():
            torch.testing.assert_close(result.source.record()[name], expected, rtol=0, atol=2e-5)
        torch.testing.assert_close(boundary.motion.encode(), MotionTask.capture(native), rtol=0, atol=1e-6)
        torch.testing.assert_close(boundary.goal[:, 3] / 5., native._command[:, 3], rtol=0, atol=1e-6)
        for row in result.fired.nonzero().flatten().tolist():
            fires[row].append((step, int(result.packet.request_id[row])))
            assert before.ids[row, 1] < 0  # Never replaces a request still in progress.
            assert boundary.stair.response[row] == int(RequestResponse.ACCEPTED)
            if row == 0 and previous_timeout is not None:
                detection_step = previous_timeout + 1
                expected_fire_step = detection_step + int(random['tof_latency'][row, detection_step])
                assert step == expected_fire_step
        if boundary.stair.finished_id[0] >= 0:
            assert boundary.stair.end_reason[0] == int(RequestEnd.TIMEOUT)
            assert not result.fired[0]
            assert boundary.stair.state.ids[0, 1] == -1
            assert boundary.stair.state.ids[0, 3] == 0  # Old hint was reset, not extended.
            assert native._command[0, 3].item() == pytest.approx(.3)
            assert result.source.flags[0, 2].item() is retry
            timeouts.append(int(boundary.stair.finished_id[0]))
            previous_timeout = step
        ended.extend(boundary.stair.end_reason[boundary.stair.finished_id >= 0].tolist())
        source_state, stair_state = result.source, boundary.stair.state
        motion, goal = boundary.motion, boundary.goal
        if step == 25:
            # Existing recorded fields reconstruct the next source; no hidden
            # timeout clock or request counter is introduced by the policy.
            task, source = StairScenario.from_dict(json.loads(json.dumps(
                StairScenario.capture(native).to_dict()))).build(source.terrain)
            source_state = StairSourceState.decode(deepcopy(source_state.record()))
    assert (len(fires[0]) >= 3 if retry else len(fires[0]) == 1)
    assert all(len(row) == 1 for row in fires[1:])
    assert [request for _, request in fires[0]] == list(range(len(fires[0])))
    assert (len(timeouts) >= 3 if retry else len(timeouts) == 1)
    for reason in (RequestEnd.CANCELLED, RequestEnd.ESTIMATED_COMPLETE, RequestEnd.SAFETY_ABORT):
        assert ended.count(int(reason)) == 1
    assert course.stair_end_counts[int(RequestEnd.TIMEOUT)] == len(timeouts)
    assert int(course.stair_outcomes[1]) == len(ended)
    assert course.stair_outcomes[2] == 0  # A retry did not turn the old failure into a climb.
    assert torch.equal(generator.get_state(), course.tof_trigger.generator.get_state())
    for name, value in original_source.record().items():
        torch.testing.assert_close(value, captured[name], rtol=0, atol=0)


def test_default_off_preserves_scenario_bytes_and_uses_existing_record_shapes():
    native, _, source, _, _, _ = timeline(retry=False)
    scenario = StairScenario.capture(native)
    legacy_fields = asdict(scenario)
    legacy_fields.pop('timeout_retry')
    old = {'format': 'supermimic-native-stair-source-v1', **legacy_fields,
           'scope': 'existing ideal position-assisted simulator source; not calibrated hardware ToF'}
    assert json.dumps(scenario.to_dict()) == json.dumps(old)
    callback = native._terrain_course._record_stair_end
    rng = native._terrain_course.tof_trigger.generator.get_state().clone()
    assert install_stair_timeout_retry(native) is None
    assert native._terrain_course._record_stair_end == callback
    assert not hasattr(native._terrain_course, 'supermimic_stair_timeout_retry')
    assert StairScenario.from_dict(old) == scenario
    receipt = install_stair_timeout_retry(native, enabled=True)
    installed = native._terrain_course._record_stair_end
    assert install_stair_timeout_retry(native, enabled=True) == receipt
    assert native._terrain_course._record_stair_end is installed
    enabled = StairScenario.capture(native)
    assert enabled.timeout_retry and enabled.to_dict()['format'] != old['format']
    assert StairScenario.from_dict(enabled.to_dict()) == enabled
    assert enabled.build(source.terrain)[1].timeout_retry
    assert TOF_RECORD_FIELDS == {'tof_ids': (4,), 'tof_flags': (4,), 'tof_values': (13,),
                                  'tof_parameters': (15,), 'tof_segment_end': (1,)}
    torch.testing.assert_close(native._terrain_course.tof_trigger.generator.get_state(), rng, rtol=0, atol=0)


def test_enabled_policy_changes_source_identity_without_borrowing_hint_migration():
    native, _, _, _, _, _ = timeline(retry=False)
    old_scenario = StairScenario.capture(native).to_dict()
    runtime = b'{}'
    old = {'runtime_sha256': hashlib.sha256(runtime).hexdigest(), 'stair_label': {'old': 'unchanged'}}
    old['reward_kernel'] = {'implementation': 'shared_physical_reward_densities'}
    receipt = install_stair_timeout_retry(native, enabled=True)
    new = {**old, 'stair_timeout_retry': receipt}
    assert receipt == timeout_retry_receipt()
    assert _native_schema_source(new) != _native_schema_source(old)
    with pytest.raises(ValueError, match='unchanged pinned runtime and task source'):
        verify_stair_hint(old, new, old_scenario, StairScenario.capture(native).to_dict(), runtime_raw=runtime)
    old_schema = StateSchema((Field('fixture', (1,)),), (), source=_native_schema_source(old))
    new_schema = StateSchema((Field('fixture', (1,)),), (), source=_native_schema_source(new))
    assert AuxiliaryWorldData._same_dynamics_contract(old_schema, old_schema)
    assert not AuxiliaryWorldData._same_dynamics_contract(old_schema, new_schema)
    trainer = object.__new__(OnlineTrainer)
    trainer.config = OnlineConfig(cycles=100)
    trainer.env = SimpleNamespace(schema=old_schema)
    trainer.teacher_source = None
    trainer.stair_scenario = StairScenario.capture(native).to_dict()
    trainer.jump_scenario = trainer.recovery_scenario = None
    saved = {'format': CHECKPOINT_FORMAT, 'model_contract': MODEL_CONTRACT,
             'schema_fingerprint': old_schema.fingerprint, 'config': asdict(trainer.config),
             'cycle': 10, 'teacher_source': None, 'stair_scenario': old_scenario,
             'jump_scenario': None, 'recovery_scenario': None}
    with pytest.raises(ValueError, match='stair protocol/source'):
        trainer._check_saved(saved)
    # The same newly recorded semantics may resume normally; this is not a
    # blanket ban on new data, and does not alias an old one-shot checkpoint.
    saved['stair_scenario'] = trainer.stair_scenario
    trainer._check_saved(saved)


def test_timeout_unlock_occurs_after_original_accounting_and_ignores_other_reasons():
    native = native_stairs(len(RequestEnd) + 1)
    _, native._terrain_course.tof_trigger = source_fixture(native.num_envs)
    trigger = native._terrain_course.tof_trigger
    trigger.riser_cleared.zero_()
    observed = []
    native._terrain_course._record_stair_end = lambda ended, reason: observed.append(
        (ended.clone(), reason.clone(), trigger.riser_cleared.clone()))
    install_stair_timeout_retry(native, enabled=True)
    reasons = torch.tensor([int(reason) for reason in RequestEnd] + [int(RequestEnd.TIMEOUT)])
    ended = torch.ones(native.num_envs, dtype=torch.bool)
    ended[-1] = False
    native._terrain_course._record_stair_end(ended, reasons)
    assert len(observed) == 1 and not observed[0][2].any()
    assert trigger.riser_cleared.nonzero().flatten().tolist() == [int(RequestEnd.TIMEOUT)]


@pytest.mark.parametrize('value', [None, 0, 1, 'true'])
def test_invalid_opt_in_is_rejected_before_constructing_the_simulator(value):
    with pytest.raises(ValueError, match='boolean'):
        create_course({'stair_timeout_retry': value})
    with pytest.raises(ValueError, match='boolean'):
        install_stair_timeout_retry(SimpleNamespace(), enabled=value)


def test_new_semantics_cannot_be_hidden_in_the_legacy_scenario_format():
    native, _, _, _, _, _ = timeline(retry=False)
    old = StairScenario.capture(native).to_dict()
    with pytest.raises(ValueError, match='scenario format'):
        StairScenario.from_dict({**old, 'timeout_retry': True})
    with pytest.raises(ValueError, match='scenario format'):
        StairScenario.from_dict({**old, 'format': 'supermimic-native-stair-source-timeout-retry-v1'})


def test_evaluation_links_raw_timeout_retries_without_changing_native_states_or_rng():
    from supermimic_rl.course.stair_evaluation import StairRequestEvaluation

    plain, *_ = timeline(retry=True)
    observed, *_ = timeline(retry=True)
    for native in (plain, observed):
        native._episode_id = torch.zeros(native.num_envs, dtype=torch.long)
    course = observed._terrain_course
    original = (course._tof_command, course._record_stair_end, course.stair_requests.step)
    evidence = StairRequestEvaluation(observed, ['timeout', 'cancel', 'complete', 'abort'])
    evidence.install()
    for step in range(55):
        for native in (plain, observed):
            owner = native._terrain_course
            previous = native.time.clone()
            native.time = torch.full((native.num_envs,), (step + 1) * .02, dtype=torch.float64)
            native._completed_estimate = estimate_at(previous, native.num_envs)
            packet = StairPacket.empty(native.num_envs, 'cpu')
            if step == 6:
                packet.cancel_id[1] = owner.stair_requests.active_id[1]
                packet.completed_id[2] = owner.stair_requests.active_id[2]
                packet.safety_abort[3] = True
            owner._pending_requests['stair'] = RequestCommand(packet.request_id, packet.height_m,
                packet.cancel_id, packet.safety_abort, completed_id=packet.completed_id)
            owner.advance_skill()
        for capture in (StairTaskState.capture, StairSourceState.capture):
            left, right = capture(plain).record(), capture(observed).record()
            for key in left:
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        assert torch.equal(plain._terrain_course.tof_trigger.generator.get_state(),
                           course.tof_trigger.generator.get_state())
        torch.testing.assert_close(plain._terrain_course.stair_outcomes, course.stair_outcomes)
    result = evidence.summary()
    retry = [r for r in result['requests'] if r['env_id'] == 0]
    assert len(retry) >= 3
    assert retry[0]['retry_of_timed_out_request_id'] is None
    assert all(r['retry_of_timed_out_request_id'] == old['request_id']
               for old, r in zip(retry, retry[1:]))
    assert all(r['previous_riser_distance_m'] == 0 for r in retry[1:])
    for row, reason in enumerate(('cancelled', 'estimated_complete', 'safety_abort'), start=1):
        records = [r for r in result['requests'] if r['env_id'] == row]
        assert len(records) == 1 and records[0]['end_reason'] == reason
        assert records[0]['retry_of_timed_out_request_id'] is None
    assert result == evidence.summary()
    # Native reset accounting precedes the episode counter increment. Even
    # identical geometry in the next episode must not inherit a retry link.
    ended = torch.ones(observed.num_envs, dtype=torch.bool)
    course._record_stair_end(ended, torch.full((observed.num_envs,), int(RequestEnd.RESET)))
    course.stair_requests.reset(torch.arange(observed.num_envs), observed.time)
    observed._episode_id.add_(1)
    observed.time += .02
    course._completed_estimate = estimate_at(observed.time - .02, observed.num_envs)
    evidence.fired.fill_(True)
    course.stair_requests.step(now_s=observed.time,
        request_id=course.stair_requests.max_seen_id + 1,
        height_m=torch.full((observed.num_envs,), .1),
        cancel_id=torch.full((observed.num_envs,), -1),
        jump_active=torch.zeros(observed.num_envs, dtype=torch.bool))
    newborn = evidence.summary()['requests'][-observed.num_envs:]
    assert all(r['episode_id'] == 1 and r['retry_of_timed_out_request_id'] is None for r in newborn)
    evidence.close()
    assert (course._tof_command, course._record_stair_end, course.stair_requests.step) == original
