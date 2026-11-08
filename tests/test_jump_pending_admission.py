"""Automatic intent survives rejection at the real native admission boundary."""
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('transformer_rl.formal.lab_env.course_tasks')
from transformer_rl.formal.lab_env.course_tasks import LabCourseTasks, RequestCommand
from transformer_rl.formal.tasks.command_pool import CommandContext, CommandPoolConfig
from transformer_rl.formal.tasks.curriculum import TaskDomain
from transformer_rl.formal.tasks.protocol import CompletedEstimate, RequestResponse
from transformer_rl.formal.tasks.trajectory import CommandRates, CommandStream
from supermimic_rl.course.jump_first import JumpFirstCommandPool
from supermimic_rl.course.jump_takeoff import JumpTakeoffConfig, install_jump_takeoff
from supermimic_rl.course.metrics import course_progress
from supermimic_rl.world_model_v2.jump_task import JumpTask, JumpTaskConfig, JumpTaskState
from support.world_v2.jump_task import native_jumps


def make(n=1, *, pending=True, pool=False, rates=None, adapter=False):
    native = native_jumps(n)
    t = native._terrain_course.tasks
    if rates is not None:
        t.stream = CommandStream(t.stream.current, t.stream.envelope, rates, policy_dt_s=.02)
    if pool:
        t.command_pool = JumpFirstCommandPool(CommandPoolConfig())
        t.command_context = lambda: CommandContext(TaskDomain(height_m=(.23, .4)), 35000)
        t.command_pool.bind_episode_targets(t)
        t._begin_episode_segments(torch.arange(n), SimpleNamespace(
            command=torch.tensor([[.5, 0., 0., .32]]).repeat(n, 1), mode=torch.zeros(n, dtype=torch.long)))
    if adapter:
        from supermimic_rl.course.jump_task_adapter import NativeJumpTaskAdapter

        NativeJumpTaskAdapter(t)
    receipt = install_jump_takeoff(native, {'automatic_pending_admission': pending})
    t.elapsed.fill_(3.)
    native.time.fill_(3.)
    t.protocol.last_time_s.fill_(3.)
    t.next_request_s.fill_(3.01)
    t.rise.fill_(.4)
    t.trajectory.start[:, 3] = .32
    t.trajectory.end[:, 3] = .32
    return native, t, receipt


def estimate(now, contact=.9):
    n = len(now)
    return CompletedEstimate(torch.zeros(n), torch.full((n, 2), contact), torch.zeros(n),
                             torch.zeros(n), now - .02, torch.ones(n, dtype=torch.bool))


def tick(native, contact=.9, **kwargs):
    native.time += .02
    return native._terrain_course.tasks.step(native.time, estimate(native.time, contact),
                                            torch.zeros(len(native.time)), **kwargs)


def test_temporary_rejection_retries_new_id_next_tick_without_extra_random_draws():
    native, t, _ = make(3)
    rng = torch.Generator().set_state(t.generator.get_state())
    torch.rand(3, generator=rng)
    tick(native, .79)
    assert t.protocol.max_seen_id.eq(0).all() and t.protocol.active_id.eq(-1).all()
    torch.testing.assert_close(t.next_request_s, t.elapsed + .02, rtol=0, atol=0)
    assert torch.equal(t.generator.get_state(), rng.get_state())
    assert int(t.request_counters[RequestResponse.NOT_READY]) == 3
    draw = torch.rand(3, generator=rng)
    tick(native)
    assert t.protocol.active_id.eq(1).all() and t.supported_start.all()
    torch.testing.assert_close(t.next_request_s, t.elapsed + (3 + 5 * draw), rtol=0, atol=0)
    assert torch.equal(t.generator.get_state(), rng.get_state())


def test_segment_resample_uses_new_height_for_gate_and_supported_start():
    native, t, _ = make(16, pool=True)
    t.segment_end_s.fill_(3.01)
    admitted = tick(native)[0].admitted
    changed = (admitted[:, 3] - .32).abs() > .015
    assert changed.any()
    assert t.protocol.active_id[changed].eq(-1).all()
    assert t.protocol.max_seen_id[changed].eq(-1).all()
    assert int(native.supermimic_jump_trigger.deferred) == int(changed.sum())
    # Once the robot reaches the same admitted target, the pending request can
    # enter with its original ID and a truthful supported-start certificate.
    t._last_world_height.copy_(admitted[:, 3])
    tick(native)
    assert t.protocol.active_id.eq(0).all() and t.supported_start.all()
    assert t.stream.current[:, 0].eq(.5).all()


@pytest.mark.parametrize('rate,accepted', [(.5, True), (1., False)])
def test_gate_uses_slew_admitted_height_not_requested_height(rate, accepted):
    native, t, _ = make(rates=CommandRates(10., 10., rate))
    t.trajectory.start[:, 3] = .4
    t.trajectory.end[:, 3] = .4
    admitted = tick(native)[0].admitted
    assert admitted[0, 3] == pytest.approx(.32 + rate * .02)
    assert bool(t.protocol.active_id[0] >= 0) is accepted
    assert bool(t.supported_start[0]) is accepted


def test_minimum_episode_is_not_advanced_twice_after_admission():
    native, t, _ = make()
    t.elapsed.fill_(.97)
    t.next_request_s.zero_()
    tick(native)
    assert t.elapsed.item() == pytest.approx(.99) and t.protocol.max_seen_id.item() == -1
    tick(native)
    assert t.protocol.active_id.item() == 0


def test_external_rejections_do_not_retry_or_draw_and_active_requests_are_not_cancelled():
    native, t, _ = make()
    t._last_supported.zero_()
    due, rng = t.next_request_s.clone(), t.generator.get_state().clone()
    packet = RequestCommand(torch.tensor([70]), torch.tensor([.4]))
    tick(native, .79, command=packet)
    assert t.protocol.max_seen_id.item() == 70 and t.protocol.active_id.item() == -1
    torch.testing.assert_close(t.next_request_s, due, rtol=0, atol=0)
    assert torch.equal(t.generator.get_state(), rng)
    packet = RequestCommand(torch.tensor([71]), torch.tensor([.4]))
    tick(native, command=packet)
    assert t.protocol.active_id.item() == 71  # External admission still uses its original protocol.
    assert not t.supported_start.item()
    t.next_request_s.zero_()
    tick(native, .1)
    assert t.protocol.active_id.item() == 71 and not t.protocol.cancelled.item()
    assert t.next_request_s.item() == 0. and int(native.supermimic_jump_trigger.deferred) == 0


def test_blocked_and_nonrequest_rows_never_become_pending_requests():
    native, t, _ = make(3)
    t.family[1] = 0
    t._last_supported.zero_()
    before = t.next_request_s.clone()
    tick(native, blocked_by_stair=torch.tensor([True, False, False]))
    torch.testing.assert_close(t.next_request_s[:2], before[:2], rtol=0, atol=0)
    assert t.protocol.max_seen_id.eq(-1).all()
    assert int(native.supermimic_jump_trigger.deferred) == 1


def test_hooks_have_no_stale_context_after_success_or_exception():
    native, t, _ = make()
    wrapper = t.step.__self__
    tick(native, .79)
    assert wrapper._tick is None
    due = t.next_request_s.clone()
    t._last_supported.zero_()
    t.stream.step(t.stream.current.clone())
    torch.testing.assert_close(t.next_request_s, due, rtol=0, atol=0)
    original = wrapper._native_step

    def fail(*args, **kwargs):
        assert wrapper._tick is not None
        raise RuntimeError('constructed native error')

    wrapper._native_step = fail
    with pytest.raises(RuntimeError, match='constructed native error'):
        tick(native)
    assert wrapper._tick is None
    wrapper._native_step = original


def test_partial_reset_discards_old_pending_time_but_preserves_id_watermark():
    native, t, _ = make(2)
    tick(native, .79)
    pending = t.next_request_s[1].clone()
    task = SimpleNamespace(mode=torch.zeros(1, dtype=torch.long), request_height_m=torch.tensor([.4]),
        terrain_kind=torch.zeros(1, dtype=torch.long), command=torch.tensor([[.5, 0., 0., .32]]),
        draw=SimpleNamespace(family=t.family[:1].clone(), nominal=torch.zeros(1, dtype=torch.bool),
                             trajectory_kind=torch.zeros(1, dtype=torch.long)))
    t.reset(torch.tensor([0]), task, native.time)
    assert 3 <= t.next_request_s[0] <= 8 and t.elapsed[0] == 0
    assert t.next_request_s[1] == pending and t.protocol.max_seen_id.tolist() == [0, 0]
    tick(native)
    assert t.protocol.active_id.tolist() == [-1, 1]
    assert t.step.__self__._tick is None


def test_default_false_keeps_legacy_receipt_serialized_config_and_retry_interval():
    native, t, explicit = make(pending=False)
    other = native_jumps()
    implicit = install_jump_takeoff(other, {})
    assert explicit == implicit
    assert set(asdict(JumpTakeoffConfig())) == {
        'min_episode_s', 'cooldown_s', 'height_tolerance_m', 'compression_depth_max_m',
        'compression_duration_s', 'nominal_extension_duration_s'}
    config = JumpTaskConfig.capture(native)
    assert 'automatic_pending_admission' not in asdict(config)['trigger']
    assert JumpTaskConfig.from_dict(asdict(config)) == config
    rng = torch.Generator().set_state(t.generator.get_state())
    draw = torch.rand(1, generator=rng)
    tick(native, .79)
    torch.testing.assert_close(t.next_request_s, t.elapsed + (3 + 5 * draw), rtol=0, atol=0)
    assert torch.equal(t.generator.get_state(), rng.get_state())
    tick(native)
    assert t.protocol.active_id.item() == -1


@pytest.mark.parametrize('value', [0, 1, None, 'true'])
def test_opt_in_requires_explicit_boolean(value):
    with pytest.raises(ValueError, match='must be a bool'):
        JumpTakeoffConfig.from_dict({'automatic_pending_admission': value})


@pytest.mark.parametrize('changed_height,external', [(False, False), (True, False), (True, True)])
def test_world_matches_native_pending_retry_and_admitted_height_without_record_change(changed_height, external):
    native, t, receipt = make(2)
    config = JumpTaskConfig.capture(native)
    assert JumpTaskConfig.from_dict(asdict(config)) == config
    assert receipt['automatic_scheduling']['version'] == 'admitted_height_pending_retry_v1'
    task = JumpTask(config)
    old_keys = set(JumpTaskState.capture(native).record())
    if changed_height:
        t.trajectory.start[:, 3] = .36
        t.trajectory.end[:, 3] = .36
    for index, contact in enumerate((.79, .9, .9)):
        state = JumpTaskState.capture(native)
        prior_elapsed, prior_height = t.elapsed.clone(), t.stream.current[:, 3].clone()
        rng = torch.Generator().set_state(t.generator.get_state())
        uniform = None if external else torch.rand(2, generator=rng)
        native.time += .02
        completed = estimate(native.time, contact)
        packet = RequestCommand(torch.full((2,), 70 + index), torch.full((2,), .4)) if external else None
        native_result = t.step(native.time, completed, torch.zeros(2), command=packet)
        result = task.advance(state, now_s=native.time, estimate=completed,
            admitted=t.trajectory.end.clone(), base_mode=t.base_mode, task_family=t.family,
            elapsed_s=t.elapsed, ground_z=torch.zeros(2), blocked_by_stair=torch.zeros(2, dtype=torch.bool),
            interval_uniform=uniform, packet=packet, previous_command_height=prior_height,
            previous_elapsed_s=prior_elapsed)
        assert set(result.state.record()) == old_keys
        for name, value in JumpTaskState.capture(native).record().items():
            torch.testing.assert_close(result.state.record()[name], value, rtol=0, atol=0)
        torch.testing.assert_close(result.command, native_result[0].admitted, rtol=0, atol=0)
        assert torch.equal(t.generator.get_state(), rng.get_state())
        t._last_world_height.copy_(t.stream.current[:, 3])


@pytest.mark.skipif(not hasattr(LabCourseTasks, '_select_request'), reason='New native mixed ownership entry point')
def test_mixed_owned_empty_external_and_automatic_rejections_preserve_ownership_and_rng():
    native, t, _ = make(4, adapter=True)
    state = JumpTaskState.capture(native)
    task = JumpTask(JumpTaskConfig.capture(native))
    packet = RequestCommand(torch.tensor([70, -1, 900, 900]), torch.full((4,), .4))
    owned = torch.tensor([True, True, False, False])
    rng = torch.Generator().set_state(t.generator.get_state())
    uniform = torch.rand(4, generator=rng)
    prior_height, prior_elapsed = t.stream.current[:, 3].clone(), t.elapsed.clone()
    before = t.next_request_s.clone()
    tick(native, .79, command=packet, packet_owned=owned)
    assert t.protocol.max_seen_id.tolist() == [70, -1, 0, 0]
    torch.testing.assert_close(t.next_request_s[:2], before[:2], rtol=0, atol=0)
    torch.testing.assert_close(t.next_request_s[2:], t.elapsed[2:] + .02, rtol=0, atol=0)
    assert torch.equal(t.generator.get_state(), rng.get_state())
    result = task.advance(state, now_s=native.time, estimate=estimate(native.time, .79),
        admitted=t.trajectory.end.clone(), base_mode=t.base_mode, task_family=t.family,
        elapsed_s=t.elapsed, ground_z=torch.zeros(4), blocked_by_stair=torch.zeros(4, dtype=torch.bool),
        interval_uniform=uniform, packet=packet, packet_owned=owned,
        previous_command_height=prior_height, previous_elapsed_s=prior_elapsed)
    for name, value in JumpTaskState.capture(native).record().items():
        torch.testing.assert_close(result.state.record()[name], value, rtol=0, atol=0)
    tick(native, command=packet, packet_owned=owned)
    assert t.protocol.active_id.tolist() == [-1, -1, 1, 1]


def test_existing_response_end_and_deferred_counters_are_exported_without_sampling():
    native, t, _ = make()
    tick(native, .79)
    tick(native)
    native.supermimic_jump_trigger.deferred.fill_(7)
    report = {'physical_request_outcomes': {}, 'stair_closed_loop': {},
              'authored_stair_completion': {'ended_by_tile': [], 'ended_complete_by_tile': []},
              'recovery': {'counts': {}}, 'episode_birth_family_counts': {}, 'level_histogram': [],
              'request_responses': t.request_counters.tolist(),
              'external_request_responses': t.external_request_counters.tolist(),
              'request_end_counts': {'estimated_complete': 4, 'timeout': 2}}
    native._terrain_course.metrics = lambda: report
    curriculum = SimpleNamespace(command_context=lambda: SimpleNamespace(domain=TaskDomain()))
    rng = t.generator.get_state().clone()
    before = JumpTaskState.capture(native)
    counters, _ = course_progress(native, curriculum)
    assert counters['course/jump/request_response/not_ready'] == 1
    assert counters['course/jump/request_response/accepted'] == 1
    assert counters['course/jump/external_request_response/not_ready'] == 0
    assert counters['course/jump/request_end/estimated_complete'] == 4
    assert counters['course/jump/gate_deferred_row_ticks'] == 7
    assert torch.equal(rng, t.generator.get_state())
    for key, value in before.record().items():
        torch.testing.assert_close(JumpTaskState.capture(native).record()[key], value, rtol=0, atol=0)
