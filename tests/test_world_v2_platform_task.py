"""Platform boundaries against actual native task/source owners, without Isaac."""
from dataclasses import fields, is_dataclass, replace
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.formal_observation import SensorReadout
from supermimic_rl.course.platform_intent import PlatformIntentPhase, PlatformIntentState
from supermimic_rl.world_model_v2.jump_task import JumpTask, JumpTaskConfig, JumpTaskState
from supermimic_rl.world_model_v2.motion_task import MotionTask
from supermimic_rl.world_model_v2.platform_task import PlatformTask
from supermimic_rl.world_model_v2.sensing import SensorQueues
from supermimic_rl.world_model_v2.stair_source import StairSourceState
from supermimic_rl.world_model_v2.stair_task import StairTask, StairTaskConfig, StairTaskState
from support.world_v2.stair_source import source_fixture
from support.world_v2.stair_task import estimate_at
from support.world_v2.terrain import state as physical_state
from test_native_platform_intent import activate_stair, setup
from transformer_rl.formal.lab_env.course_tasks import LabCourseTasks
from transformer_rl.formal.tasks.gimbal_course import GimbalCourse, GimbalCourseConfig
from transformer_rl.formal.tasks.protocol import RequestEnd
from transformer_rl.formal.tasks.trajectory import CommandRates


pytestmark = pytest.mark.skipif(not hasattr(LabCourseTasks, '_select_request'),
    reason='Requires the private native request-selection entry point')


def same(actual, expected, *, atol=0.):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=0., equal_nan=True)
    elif is_dataclass(expected):
        for field in fields(expected):
            same(getattr(actual, field.name), getattr(expected, field.name), atol=atol)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for name in expected:
            same(actual[name], expected[name], atol=atol)
    else:
        assert actual == expected


class BoundaryHarness:
    def __init__(self, target=(0,), *, height=.32, earliest=None, rates=None, x=-3.):
        n = len(target)
        native, caller, _, _ = setup(target, height=height, earliest=earliest,
                                      rates=rates, install=False)
        source, trigger = source_fixture(n)
        self.native, self.caller, self.source = native, caller, source
        course, tasks = native._terrain_course, native._terrain_course.tasks
        native.physics_dt = .005
        course.tof_trigger, course.query = trigger, source.terrain.query
        course.router = SimpleNamespace(tile=torch.zeros(n, dtype=torch.int64),
                                       origins=source.terrain.origins.expand(n, -1))
        course._wheel_radius = .08
        del course._tof_command, course._tof_rear_clearance
        native.position = torch.tensor([x, 0., .32]).expand(n, -1).clone()
        native._ground_at_root.copy_(course.router.origins[:, 2])
        tasks._last_world_height.add_(course.router.origins[:, 2])
        native._root_link_position = lambda: native.position + course.router.origins
        native._root_link_quat = lambda: torch.tensor([0., 0., 0., 1.]).expand(n, -1)
        course._wheel_centres = lambda: native._root_link_position()[:, None] + torch.tensor(
            [[[0., -.2, -.24], [0., .2, -.24]]])
        native._gimbal_course = GimbalCourse(GimbalCourseConfig(enabled=True), n)
        native._gimbal_course.remaining_s.fill_(10.)
        tasks.trajectory.start[:, 0] = tasks.trajectory.end[:, 0] = tasks.stream.current[:, 0] = .8
        tasks.segment_end_s.fill_(10.)
        native._command.copy_(tasks.stream.current)
        self.goal = torch.zeros(n, 16)
        self.goal[:, :4] = native._command * torch.tensor([1., 1., 1., 5.])
        self.goal[:, 4] = self.goal[:, 11] = 1.
        self.queues = SensorQueues(torch.zeros(n, 3, 18), torch.zeros(n, 3, 5),
            torch.tensor([1., 0.]).expand(n, 3, 2).clone(), torch.full((n,), 3.))
        self.readout = SensorReadout(torch.zeros(n, 4), torch.zeros(n, 1, dtype=torch.bool),
            torch.zeros(n, 18), torch.zeros(n, 1), torch.zeros(n, 6))
        caller.install()
        self.task = PlatformTask(JumpTask(JumpTaskConfig.capture(native)),
                                 StairTask(StairTaskConfig.capture(native)), source)
        self.capture()

    def capture(self):
        self.intent = PlatformIntentState.decode(self.caller.capture())
        self.jump = JumpTaskState.capture(self.native)
        self.motion = MotionTask.decode(MotionTask.capture(self.native))
        self.stair = StairTaskState.capture(self.native)
        self.source_state = StairSourceState.capture(self.native)

    def inputs(self, *, invalid=False):
        native, n = self.native, self.native.num_envs
        course, tasks = native._terrain_course, native._terrain_course.tasks
        jgen = torch.Generator().set_state(tasks.generator.get_state())
        tgen = torch.Generator().set_state(course.tof_trigger.generator.get_state())
        uniform = torch.rand(n, generator=jgen)
        random = self.source.draw(n, 1, device='cpu', generator=tgen)
        estimate = estimate_at(native.time, n)
        if invalid:
            estimate = replace(estimate, valid=torch.zeros(n, dtype=torch.bool))
        kwargs = dict(motion=self.motion, stair=self.stair, source=self.source_state,
            physical=physical_state(native.position),
            wheel_centres=course._wheel_centres() - course.router.origins[:, None],
            wheel_radius=.08, terrain_tile=course.router.tile, queues=self.queues,
            readout=self.readout, goal=self.goal, estimate=estimate, task_family=tasks.family,
            ground_z=native._ground_at_root - course.router.origins[:, 2],
            interval_uniform=uniform, latency=random['tof_latency'][:, 0], draws=random['tof_uniform'][:, 0])
        return kwargs, jgen, tgen

    def step(self, *, invalid=False):
        kwargs, jgen, tgen = self.inputs(invalid=invalid)
        before = torch.random.get_rng_state().clone()
        result = self.task.advance(self.intent, self.jump, **kwargs)
        assert torch.equal(before, torch.random.get_rng_state())
        self.native.time = self.task.jump_task.boundary_time(self.jump)
        self.native._completed_estimate = kwargs['estimate']
        self.native._terrain_course.advance_skill()
        same(result.intent.record(), self.caller.capture())
        same(result.jump.state, JumpTaskState.capture(self.native), atol=1e-6)
        same(result.boundary.stair.state, StairTaskState.capture(self.native), atol=1e-6)
        same(result.source.source, StairSourceState.capture(self.native), atol=2e-5)
        same(result.boundary.motion.encode(), MotionTask.capture(self.native), atol=1e-6)
        same(result.boundary.goal[:, :3], self.native._command[:, :3], atol=1e-6)
        same(result.boundary.goal[:, 3] / 5., self.native._command[:, 3], atol=1e-6)
        same(result.boundary.goal[:, 4:10].argmax(-1), self.native._command_mode)
        assert torch.equal(jgen.get_state(), self.native._terrain_course.tasks.generator.get_state())
        assert torch.equal(tgen.get_state(), self.native._terrain_course.tof_trigger.generator.get_state())
        self.intent, self.jump = result.intent, result.jump.state
        self.motion, self.stair = result.boundary.motion, result.boundary.stair.state
        self.source_state, self.goal = result.source.source, result.boundary.goal
        return result


def test_cancel_stair_then_fresh_jump_next_boundary_matches_native():
    h = BoundaryHarness((0, 1, -1))
    activate_stair(h.native, torch.tensor([0, 1]))
    h.capture()
    first = h.step()
    assert first.intent.phase.tolist() == [PlatformIntentPhase.PENDING] * 2 + [0]
    assert first.boundary.stair.finished_id.tolist() == [7, 7, -1]
    assert first.jump.state.active_id.tolist() == [-1] * 3
    second = h.step()
    assert second.jump.state.active_id.tolist() == [0, 0, -1]
    assert second.intent.phase.tolist() == [PlatformIntentPhase.ACCEPTED] * 2 + [0]


def test_owned_and_automatic_rows_keep_original_full_batch_rng_and_gate():
    h = BoundaryHarness((0, -1, -1))
    tasks = h.native._terrain_course.tasks
    tasks.next_request_s.zero_()
    tasks._last_supported[2] = False
    h.capture()
    result = h.step()
    assert result.jump.state.active_id.tolist() == [0, 0, -1]
    assert result.jump.deferred.tolist() == [False, False, True]
    assert result.intent.bound_request_id.tolist() == [0, -1, -1]


def test_height_override_does_not_accept_until_preceding_admitted_height_and_support_ready():
    h = BoundaryHarness((0,), height=.36, rates=CommandRates(.8, 3., .2))
    for _ in range(10):
        result = h.step()
        assert result.jump.state.active_id.item() == -1
        assert result.intent.phase.item() == PlatformIntentPhase.PENDING
    assert h.motion.current[0, 3] == torch.tensor(.36)
    # The admitted command is ready, but the old physical stance still is not.
    assert h.step().jump.state.active_id.item() == -1
    h.native._terrain_course.tasks._last_world_height[:] = h.native._ground_at_root + .36
    h.jump = JumpTaskState.capture(h.native)
    assert h.step().jump.state.active_id.item() == 0


def test_rejected_id_consumed_then_physical_tail_survives_protocol_completion():
    h = BoundaryHarness()
    first = h.step(invalid=True)
    assert first.jump.state.ids[0, 0] == 0
    assert first.intent.phase.item() == PlatformIntentPhase.PENDING
    second = h.step()
    assert second.jump.state.physical_id.item() == 1
    tasks = h.native._terrain_course.tasks
    tasks.protocol.flight_seen.fill_(True)
    tasks.protocol.landing_since_s.copy_(h.native.time - tasks.protocol.config.stable_landing_s)
    h.jump = JumpTaskState.capture(h.native)
    tail = h.step()
    assert tail.jump.end_reason.item() == int(RequestEnd.ESTIMATED_COMPLETE)
    assert tail.intent.phase.item() == PlatformIntentPhase.PHYSICAL_TAIL
    assert tail.intent.reservation.item()
    tasks.physical_id.fill_(-1)
    h.jump = JumpTaskState.capture(h.native)
    ended = h.step()
    assert ended.intent.phase.item() == PlatformIntentPhase.DONE
    assert not ended.intent.reservation.any()


def test_waiting_plan_keeps_tof_sensing_then_pending_reservation_cancels_it():
    h = BoundaryHarness((0, -1), earliest=(2.12, 0.), x=-.113)
    h.native._terrain_course.tasks._last_supported[0] = False
    h.capture()
    fired = torch.zeros(2, dtype=torch.bool)
    for _ in range(8):
        result = h.step()
        fired |= result.source.fired
    assert fired.all()
    assert result.intent.phase[0] == PlatformIntentPhase.PENDING
    assert result.boundary.stair.state.ids[0, 1] == -1
    assert result.boundary.stair.state.ids[1, 1] >= 0


def test_pending_does_not_make_unknown_future_velocity_resampling_known():
    h = BoundaryHarness((0,))
    h.native._terrain_course.tasks._last_supported[:] = False
    h.capture()
    result = h.step()
    motion = replace(result.boundary.motion, route=result.boundary.motion.route.clone())
    motion.route[:, 4] = 1.
    source = replace(result.source.source, segment_end=torch.zeros(1, 1))
    assert result.intent.owned.item()
    assert not source.boundary_known(motion, result.boundary.stair.state, .02, result.jump.state).item()


def test_all_inactive_matches_existing_rollout_chain_without_changing_input_tensors():
    h = BoundaryHarness((-1, -1))
    h.native._terrain_course.tasks.next_request_s.zero_()
    h.capture()
    kwargs, _, _ = h.inputs()
    before = {name: value.clone() for name, value in h.jump.record().items()}
    original_intent = h.intent.record()
    result = h.task.advance(h.intent, h.jump, **kwargs)
    following, admitted = h.motion.advance_admission(h.queues, h.readout, .02)
    jump = h.task.jump_task.advance(h.jump, now_s=h.task.jump_task.boundary_time(h.jump),
        estimate=kwargs['estimate'], admitted=admitted, base_mode=following.route[:, 3].long(),
        task_family=kwargs['task_family'], elapsed_s=following.timing[:, 0], ground_z=kwargs['ground_z'],
        blocked_by_stair=h.stair.ids[:, 1] >= 0, interval_uniform=kwargs['interval_uniform'],
        previous_command_height=h.motion.current[:, 3], previous_elapsed_s=h.motion.timing[:, 0])
    source_kwargs = {name: kwargs[name] for name in ('physical', 'wheel_centres', 'wheel_radius',
        'terrain_tile', 'queues', 'readout', 'goal', 'estimate', 'latency', 'draws')}
    old = h.source.advance(h.source_state, h.stair, task=h.task.stair_task, motion=h.motion,
        physics_dt_s=.005, admission=(replace(following, current=jump.command), jump.command),
        base_mode=jump.mode, jump_active=jump.state.active_id >= 0,
        request_height=jump.request_height, request_clock=jump.request_clock, **source_kwargs)
    same(result.jump, jump)
    same(result.source, old)
    same(h.jump.record(), before)
    same(h.intent.record(), original_intent)


def test_boundary_keeps_current_continuous_command_derivatives():
    h = BoundaryHarness((-1,))
    kwargs, _, _ = h.inputs()
    motion = replace(h.motion, current=h.motion.current.clone().requires_grad_(),
                     rates=torch.tensor([[1., .8, 3., .2]]),
                     start=h.motion.start + .1, end=h.motion.end + .1)
    kwargs['motion'] = motion
    result = h.task.advance(h.intent, h.jump, **kwargs)
    gradient = torch.autograd.grad(result.boundary.motion.current.sum(), motion.current)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_continuing_request_retains_peak_derivative_across_the_boundary():
    h = BoundaryHarness()
    h.step()
    peak = torch.tensor([[.1]], requires_grad=True)
    jump = replace(h.jump, values=torch.cat((h.jump.values[:, :15], peak, h.jump.values[:, 16:]), -1))
    kwargs, _, _ = h.inputs()
    result = h.task.advance(h.intent, jump, retain_peak_gradient=True, **kwargs)
    assert result.jump.state.physical_id.item() == h.jump.physical_id.item()
    gradient = torch.autograd.grad(result.jump.state.values[:, 15].sum(), peak)[0]
    same(gradient, torch.ones_like(peak))


def test_missing_native_trigger_is_not_silently_replaced_with_new_gate_parameters():
    h = BoundaryHarness()
    without = JumpTask(replace(h.task.jump_task.config, trigger=None))
    with pytest.raises(ValueError, match='recorded native take-off gate'):
        PlatformTask(without, h.task.stair_task, h.source)
