"""Real native request bookkeeping under an explicit platform caller."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('transformer_rl.formal.lab_env.course_tasks')
from transformer_rl.formal.lab_env.course_tasks import (
    LabCourseTasks, RequestCommand, StairRequestConfig, StairRequestProtocol,
)
from transformer_rl.formal.lab_env.skill_fsm import StairState
from transformer_rl.formal.tasks.curriculum import TaskFamily
from transformer_rl.formal.tasks.protocol import RequestEnd, RequestResponse
from transformer_rl.formal.tasks.trajectory import CommandRates, CommandStream

from supermimic_rl.course.jump_takeoff import install_jump_takeoff
from supermimic_rl.course.jump_task_adapter import NativeJumpTaskAdapter
from supermimic_rl.course.native_platform_intent import NativePlatformIntent
from supermimic_rl.course.platform_intent import PlatformIntentPhase, PlatformIntentState
from supermimic_rl.course.reward import install_flight_latch
from tests.support.world_v2.jump_task import native_jumps
from tests.support.world_v2.stair_task import estimate_at, native_stairs

pytestmark = pytest.mark.skipif(not hasattr(LabCourseTasks, '_select_request'),
    reason='Requires the private native request-selection entry point')


def plan(target, *, earliest=None, height=.32):
    target = torch.tensor(target, dtype=torch.long)
    selected = target >= 0
    return PlatformIntentState.plan(target_index=target,
        earliest_episode_s=torch.zeros(len(target), dtype=torch.float64) if earliest is None
        else torch.tensor(earliest, dtype=torch.float64),
        rise_m=torch.where(selected, .2, 0.), support_height_m=torch.where(selected, height, 0.))


def setup(target=(0,), *, earliest=None, height=.32, rates=None, install=True):
    """Real LabCourseTasks/selector/gate and real course advance, without Isaac."""
    n = len(target)
    native = native_stairs(n)
    course = native._terrain_course
    course.tasks = native_jumps(n)._terrain_course.tasks
    tasks = course.tasks
    native.time.fill_(2.)
    tasks.elapsed.fill_(2.)
    tasks.next_request_s.fill_(100.)
    tasks.protocol.last_time_s.copy_(native.time)
    tasks.trajectory.start[:, 3] = .32
    tasks.trajectory.end[:, 3] = .32
    tasks.stream.current[:, 3] = .32
    if rates is not None:
        tasks.stream = CommandStream(tasks.stream.current, tasks.stream.envelope, rates, policy_dt_s=.02)
    course.stair_requests = StairRequestProtocol(n, config=StairRequestConfig(policy_dt_s=.02))
    course.stair_requests.reset(torch.arange(n), native.time)
    adapter = NativeJumpTaskAdapter(tasks)
    latch = install_flight_latch(tasks)
    install_jump_takeoff(native, {})
    native.tof_ready = []
    native.tof_ticks = 0
    native.tof_countdown = torch.full((n,), 2, dtype=torch.long)
    native.tof_fire = torch.zeros(n, dtype=torch.bool)

    def tof(request, rise, cancel, abort, ready, completed_id=None):
        native.tof_ready.append(ready.clone())
        native.tof_ticks += 1
        native.tof_countdown.sub_(1).clamp_min_(0)
        fire = (native.tof_fire & ready & (native.tof_countdown == 0)
                & (request == -1) & (cancel == -1) & ~abort
                & (tasks.active_id < 0) & (course.stair_requests.active_id < 0))
        ids = torch.maximum(course.stair_next_id, course.stair_requests.max_seen_id + 1)
        return torch.where(fire, ids, request), torch.where(fire, .2, rise), fire

    course._tof_command = tof

    def reset(ids, task):
        tasks.reset(ids, task, native.time)
        course.stair_requests.reset(ids, native.time)
        native._skill_fsm.stair.reset(ids)

    course.reset_commands = reset
    caller = NativePlatformIntent(native, task_adapter=adapter,
                                  state=plan(target, earliest=earliest, height=height))
    if install:
        caller.install()
    return native, caller, adapter, latch


def boundary(native):
    native.time.add_(.02)
    native._completed_estimate = estimate_at(native.time-.02, native.num_envs)
    return native._terrain_course.advance_skill()


def packet(ids, *, cancel=None, abort=None):
    ids = torch.tensor(ids, dtype=torch.long)
    return RequestCommand(ids, torch.full((len(ids),), .2),
        cancel_id=None if cancel is None else torch.tensor(cancel, dtype=torch.long),
        safety_abort=None if abort is None else torch.tensor(abort, dtype=torch.bool))


def activate_stair(native, rows):
    course = native._terrain_course
    course.stair_requests.active_id[rows] = 7
    course.stair_requests.max_seen_id[rows] = 7
    course.stair_requests.height_m[rows] = .2
    course.stair_requests.accepted_at_s[rows] = native.time[rows]
    native._skill_fsm.stair.state[rows] = int(StairState.STAIR)


def test_real_cancel_then_next_boundary_accept_preserves_other_manual_stair_rows():
    native, caller, _, latch = setup((0, -1, -1))
    course, tasks = native._terrain_course, native._terrain_course.tasks
    activate_stair(native, torch.tensor([0, 2]))
    course._pending_requests['stair'] = packet([-1, 81, -1], cancel=[-1, -1, 7])
    physics = tasks.observe_physics
    boundary(native)
    assert course.stair_requests.active_id.tolist() == [-1, 81, -1]
    assert tasks.physical_id.tolist() == [-1, -1, -1]
    assert caller.state.phase.tolist() == [PlatformIntentPhase.PENDING, 0, 0]
    assert native.tof_ready[-1].tolist() == [False, True, True]
    assert caller._pending_cancel is None
    boundary(native)
    assert tasks.active_id.tolist() == [0, -1, -1]
    assert tasks.physical_id.tolist() == [0, -1, -1]
    assert caller.state.phase[0] == PlatformIntentPhase.ACCEPTED
    assert caller.state.bound_request_id[0] == tasks.physical_id[0]
    assert tasks.observe_physics is physics and tasks.supermimic_flight_latch is latch


def test_missing_packet_keeps_unowned_automatic_schedule_and_rng():
    native, caller, _, _ = setup((0, -1, -1))
    tasks = native._terrain_course.tasks
    tasks.next_request_s.zero_()
    tasks._last_supported[2] = False
    expected = torch.Generator().set_state(tasks.generator.get_state())
    torch.rand(3, generator=expected)
    boundary(native)
    assert tasks.active_id.tolist() == [0, 0, -1]
    assert tasks.external_request_counters[RequestResponse.ACCEPTED] == 1
    assert tasks.request_counters[RequestResponse.ACCEPTED] == 2
    assert native.supermimic_jump_trigger.deferred == 1
    assert caller.state.bound_request_id.tolist() == [0, -1, -1]
    assert torch.equal(tasks.generator.get_state(), expected.get_state())


def test_existing_full_packet_retains_external_rows_without_automatic_rng_draw():
    native, caller, _, _ = setup((0, -1, -1))
    course, tasks = native._terrain_course, native._terrain_course.tasks
    tasks.next_request_s.zero_()
    before = tasks.generator.get_state().clone()
    course._pending_requests['jump'] = packet([91, 92, -1])
    boundary(native)
    assert tasks.active_id.tolist() == [0, 92, -1]
    assert tasks.external_request_counters[RequestResponse.ACCEPTED] == 2
    assert torch.equal(tasks.generator.get_state(), before)
    assert caller.state.bound_request_id.tolist() == [0, -1, -1]


def test_explicit_packet_ownership_does_not_promote_unowned_cancel_or_abort_fields():
    native, caller, _, _ = setup((0, -1, -1))
    tasks = native._terrain_course.tasks
    tasks.next_request_s.zero_()
    native.time.add_(.02)
    tasks.step(native.time, estimate_at(native.time-.02, 3), native._ground_at_root,
        command=packet([100, 101, 102], cancel=[0, -1, -1], abort=[True, False, False]),
        packet_owned=torch.tensor([False, True, False]))
    assert tasks.active_id.tolist() == [0, 101, 0]
    assert caller.state.phase[0] == PlatformIntentPhase.ACCEPTED


@pytest.mark.parametrize('kind', ['abort', 'matching_cancel'])
def test_pre_accept_external_stop_ends_caller_without_inventing_protocol_finish(kind):
    native, caller, _, _ = setup()
    course, tasks = native._terrain_course, native._terrain_course.tasks
    course._pending_requests['jump'] = packet([90],
        abort=[True] if kind == 'abort' else None,
        cancel=[0] if kind == 'matching_cancel' else None)
    boundary(native)
    assert tasks.active_id.item() == tasks.protocol.max_seen_id.item() == -1
    assert caller.state.phase.item() == PlatformIntentPhase.DONE
    assert tasks.request_end_counts.sum() == 0
    for _ in range(3):
        boundary(native)
    assert tasks.protocol.max_seen_id.item() == -1
    assert caller.state.phase.item() == PlatformIntentPhase.DONE


def test_unrelated_cancel_suppresses_current_submission_without_killing_the_plan():
    native, caller, _, _ = setup()
    tasks = native._terrain_course.tasks
    native._terrain_course._pending_requests['jump'] = packet([90], cancel=[123])
    boundary(native)
    assert tasks.active_id.item() == -1
    assert caller.state.phase.item() == PlatformIntentPhase.PENDING
    boundary(native)
    assert tasks.active_id.item() == 0
    assert caller.state.phase.item() == PlatformIntentPhase.ACCEPTED


@pytest.mark.parametrize('kind', ['abort', 'cancel'])
def test_bound_external_stop_uses_the_real_protocol_end_and_physical_clear(kind):
    native, caller, _, _ = setup()
    boundary(native)
    tasks = native._terrain_course.tasks
    native._terrain_course._pending_requests['jump'] = packet([99],
        abort=[True] if kind == 'abort' else None, cancel=[0] if kind == 'cancel' else None)
    boundary(native)
    assert tasks.physical_id.item() == tasks.active_id.item() == -1
    assert tasks.protocol.max_seen_id.item() == 0
    assert caller.state.phase.item() == PlatformIntentPhase.DONE
    reason = RequestEnd.SAFETY_ABORT if kind == 'abort' else RequestEnd.CANCELLED
    assert tasks.request_end_counts[reason] == 1


def test_tof_keeps_ticking_while_reserved_and_unowned_rows_can_submit():
    native, caller, _, _ = setup((0, -1))
    native._terrain_course.tasks._last_supported[0] = False
    native.tof_fire[:] = True
    for _ in range(2):
        boundary(native)
    assert native.tof_ticks == 2 and native.tof_countdown.tolist() == [0, 0]
    assert native.tof_ready[-1].tolist() == [False, True]
    assert native._terrain_course.stair_requests.active_id.tolist() == [-1, 0]
    assert caller.state.phase[0] == PlatformIntentPhase.PENDING
    caller.set_plan(plan((-1,)), rows=torch.tensor([0]))
    boundary(native)
    assert native._terrain_course.stair_requests.active_id.tolist() == [0, 0]


def test_waiting_primes_height_through_admission_and_does_not_reserve_tof():
    native, caller, _, _ = setup((0, -1), earliest=(4., 0.), height=.36,
                                 rates=CommandRates(.8, 3., .2))
    tasks = native._terrain_course.tasks
    native.time.add_(.02)
    result = tasks.step(native.time, estimate_at(native.time-.02, 2), native._ground_at_root,
        height_override=torch.tensor([.2, .34]), height_override_mask=torch.tensor([True, True]))
    assert result[0].requested[:, 3].tolist() == pytest.approx([.36, .34])
    assert result[0].admitted[:, 3].tolist() == pytest.approx([.324, .324])
    assert not caller.state.reservation.any()
    assert tasks.active_id.tolist() == [-1, -1]


def test_rejection_fresh_id_then_estimated_completion_retains_physical_tail():
    native, caller, _, _ = setup()
    tasks = native._terrain_course.tasks
    native.time.add_(.02)
    estimate = replace(estimate_at(native.time-.02, 1), valid=torch.tensor([False]))
    tasks.step(native.time, estimate, native._ground_at_root)
    assert tasks.protocol.max_seen_id.item() == 0
    assert caller.state.phase.item() == PlatformIntentPhase.PENDING
    boundary(native)
    assert tasks.physical_id.item() == 1 and caller.state.bound_request_id.item() == 1
    tasks.protocol.flight_seen.fill_(True)
    tasks.protocol.landing_since_s.copy_(native.time - tasks.protocol.config.stable_landing_s)
    boundary(native)
    assert tasks.active_id.item() == -1 and tasks.physical_id.item() == 1
    assert caller.state.phase.item() == PlatformIntentPhase.PHYSICAL_TAIL
    assert caller.state.reservation.item()
    tasks.physical_id.fill_(-1)  # The physical updater owns success, not the caller.
    boundary(native)
    assert caller.state.phase.item() == PlatformIntentPhase.DONE


def test_real_task_reset_rearms_same_plan_without_restoring_native_or_id_watermarks():
    native, caller, _, _ = setup((0, 1))
    boundary(native)
    tasks = native._terrain_course.tasks
    saved = caller.state_dict()
    captured = caller.capture()
    reset_task = SimpleNamespace(mode=torch.zeros(1, dtype=torch.long),
        draw=SimpleNamespace(family=torch.tensor([int(TaskFamily.REQUEST)]), nominal=torch.tensor([False]),
                             trajectory_kind=torch.zeros(1, dtype=torch.long)),
        terrain_kind=torch.zeros(1, dtype=torch.long), request_height_m=torch.tensor([.2]),
        command=torch.tensor([[0., 0., 0., .32]]))
    native._terrain_course.reset_commands(torch.tensor([0]), reset_task)
    assert caller.state.phase.tolist() == [PlatformIntentPhase.WAITING, PlatformIntentPhase.ACCEPTED]
    assert caller.state.bound_request_id.tolist() == [-1, 0]
    assert tasks.elapsed[0] == 0 and tasks.physical_id[0] == -1
    assert tasks.protocol.max_seen_id.tolist() == [0, 0]
    assert captured['platform_intent_ids'][:, 1].tolist() == [PlatformIntentPhase.ACCEPTED] * 2
    rng = tasks.generator.get_state().clone()
    caller.load_state_dict(saved)
    assert caller.state.bound_request_id.tolist() == [0, 0]
    assert tasks.physical_id[0] == -1 and tasks.elapsed[0] == 0
    assert torch.equal(tasks.generator.get_state(), rng)
    caller.set_plan(plan((1,), earliest=(3.,)), rows=torch.tensor([0]))
    assert caller.state.target_index.tolist() == [1, 1]
    saved['platform_intent_values'].zero_()
    assert caller.state.rise_m.tolist() == pytest.approx([.2, .2])


@pytest.mark.parametrize('where', ['protocol', 'after_protocol'])
def test_exception_restores_temporary_context_without_claiming_native_rollback(where):
    native, caller, adapter, _ = setup(install=False)
    tasks = native._terrain_course.tasks
    old_protocol = tasks.protocol.step
    calls = []

    def raising_protocol(**kwargs):
        calls.append(1)
        if where == 'after_protocol':
            old_protocol(**kwargs)
        raise RuntimeError('synthetic native failure')

    tasks.protocol.step = raising_protocol
    caller.install()
    before = caller.state_dict()
    with pytest.raises(RuntimeError, match='synthetic native failure'):
        boundary(native)
    assert tasks.protocol.step is raising_protocol
    assert adapter._packet_owned is adapter._height_override is adapter._height_override_mask is None
    assert caller._pending_cancel is None
    for name in caller.capture():
        assert torch.equal(caller.capture()[name], before[name])
    assert tasks.elapsed.item() > 2.  # No fictitious rollback of native progress.
    assert calls == [1]


@pytest.mark.parametrize('where', ['tof', 'stair'])
def test_post_task_boundary_exception_clears_only_ephemeral_cancellation(where):
    native, caller, _, _ = setup(install=False)
    course = native._terrain_course
    activate_stair(native, torch.tensor([0]))

    def fail(*args, **kwargs):
        raise RuntimeError('synthetic boundary failure')

    if where == 'tof':
        course._tof_command = fail
    else:
        course.stair_requests.step = fail
    caller.install()
    with pytest.raises(RuntimeError, match='synthetic boundary failure'):
        boundary(native)
    assert caller._pending_cancel is None
    assert caller.state.phase.item() == PlatformIntentPhase.PENDING


def test_install_close_restores_methods_and_leaves_physics_wrappers_alone():
    native, caller, adapter, latch = setup(install=False)
    owners = ((caller.tasks, 'step'), (caller.course, 'advance_skill'),
              (caller.course, 'reset_commands'), (caller.course, '_tof_command'),
              (caller.course.stair_requests, 'step'))
    before = [(owner, name, name in vars(owner), getattr(owner, name)) for owner, name in owners]
    physics = caller.tasks.observe_physics
    caller.install()
    with pytest.raises(ValueError, match='already installed'):
        caller.install()
    with pytest.raises(ValueError, match='already installed'):
        NativePlatformIntent(native, task_adapter=adapter, state=plan((0,))).install()
    caller.close()
    caller.close()
    assert not hasattr(native, 'supermimic_platform_intent')
    for owner, name, present, callback in before:
        assert (name in vars(owner)) == present and getattr(owner, name) == callback
    assert caller.tasks.observe_physics is physics and caller.tasks.supermimic_flight_latch is latch


def test_bad_adapter_gate_and_metadata_are_rejected_without_partial_installation():
    native, caller, adapter, _ = setup(install=False)
    other = NativeJumpTaskAdapter(native_jumps(1)._terrain_course.tasks)
    with pytest.raises(ValueError, match='matching native'):
        NativePlatformIntent(native, task_adapter=other, state=plan((0,)))
    native.supermimic_jump_trigger.tasks = other.tasks
    with pytest.raises(ValueError, match='take-off gate'):
        NativePlatformIntent(native, task_adapter=adapter, state=plan((0,)))
    with pytest.raises(ValueError, match='format'):
        caller.load_state_dict({})
    with pytest.raises(ValueError, match='rows and device'):
        caller.set_plan(plan((0, 1)))
    assert not hasattr(native, 'supermimic_platform_intent')


def test_inactive_caller_preserves_native_values_counters_and_random_draws():
    baseline, _, _, _ = setup((-1, -1), install=False)
    native, _, _, _ = setup((-1, -1))
    for env in (baseline, native):
        env._terrain_course.tasks.next_request_s.zero_()
    for _ in range(3):
        boundary(baseline)
        boundary(native)
        for before, after in ((baseline._terrain_course.tasks, native._terrain_course.tasks),
                              (baseline._terrain_course.tasks.protocol, native._terrain_course.tasks.protocol)):
            for name, value in vars(before).items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(getattr(after, name), value, rtol=0, atol=0)
        assert torch.equal(baseline._terrain_course.tasks.generator.get_state(),
                           native._terrain_course.tasks.generator.get_state())
        torch.testing.assert_close(native._command, baseline._command, rtol=0, atol=0)


def test_abort_before_acceptance_does_not_cancel_a_different_stair_request():
    native, caller, _, _ = setup()
    activate_stair(native, torch.tensor([0]))
    native._terrain_course._pending_requests['jump'] = packet([-1], abort=[True])
    boundary(native)
    assert caller.state.phase.item() == PlatformIntentPhase.DONE
    assert native._terrain_course.stair_requests.active_id.item() == 7
    assert native._terrain_course.tasks.request_end_counts.sum() == 0


def test_close_does_not_remove_later_outer_observers():
    native, caller, _, _ = setup()
    old = caller.tasks.step
    caller.tasks.step = lambda *args, **kwargs: old(*args, **kwargs)
    with pytest.raises(RuntimeError, match='outer native observers'):
        caller.close()
    assert native.supermimic_platform_intent is caller
    caller.tasks.step = old
    caller.close()


def test_protocol_timeout_and_physical_bookkeeping_end_the_bound_intent():
    native, caller, _, _ = setup()
    boundary(native)
    tasks = native._terrain_course.tasks
    native.time.add_(tasks.protocol.config.timeout_s)
    tasks.protocol.last_time_s.copy_(native.time-.02)
    tasks.step(native.time, estimate_at(native.time-.02, 1), native._ground_at_root)
    assert tasks.physical_id.item() == -1
    assert tasks.request_end_counts[RequestEnd.TIMEOUT] == 1
    assert caller.state.phase.item() == PlatformIntentPhase.DONE
