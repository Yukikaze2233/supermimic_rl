"""Mixed request rows keep the existing native protocol and automatic schedule."""
from dataclasses import fields, replace

import pytest
import torch

from supermimic_rl.course.jump_takeoff import install_jump_takeoff
from supermimic_rl.course.request_selection import select_jump_request
from supermimic_rl.world_model_v2.jump_task import (
    JUMP_CLOCK_NAMES, JUMP_ID_NAMES, JUMP_VALUE_NAMES, JumpTask, JumpTaskConfig, JumpTaskState,
)
from transformer_rl.formal.control.observation import CommandMode
from transformer_rl.formal.lab_env.course_tasks import RequestCommand
from transformer_rl.formal.tasks.curriculum import TaskFamily
from transformer_rl.formal.tasks.protocol import CompletedEstimate, RequestEnd, RequestResponse
from support.world_v2.jump_task import native_jumps


def selection_inputs(n=3):
    return dict(task_family=torch.full((n,), int(TaskFamily.REQUEST)),
        base_mode=torch.zeros(n, dtype=torch.long), blocked_by_stair=torch.zeros(n, dtype=torch.bool),
        elapsed_s=torch.full((n,), 2.), active_id=torch.full((n,), -1, dtype=torch.long),
        next_id=torch.arange(n, dtype=torch.long), next_request_s=torch.zeros(n), rise_m=torch.full((n,), .06))


def packet(n):
    return RequestCommand(torch.arange(n, dtype=torch.long) + 100, torch.full((n,), .2))


def estimate_at(now):
    n = len(now)
    return CompletedEstimate(torch.zeros(n), torch.full((n, 2), .9), torch.zeros(n),
        torch.zeros(n), now.clone(), torch.ones(n, dtype=torch.bool))


def test_selector_does_not_mutate_inputs_or_consume_rng_and_default_draw_contract_is_preserved():
    inputs = selection_inputs()
    originals = {name: value.clone() for name, value in inputs.items()}
    rng = torch.get_rng_state().clone()
    result = select_jump_request(**inputs, interval_uniform=torch.tensor([0., .5, 1.]))
    assert result.request_id.tolist() == [0, 1, 2]
    assert result.next_id.tolist() == [1, 2, 3]
    assert result.next_request_s.tolist() == [5., 7.5, 10.]
    external = select_jump_request(**inputs, packet=packet(3))
    assert external.request_id.tolist() == [100, 101, 102]
    assert external.next_id is inputs['next_id'] and external.next_request_s is inputs['next_request_s']
    assert not external.automatic_due.any()
    for name, expected in originals.items():
        torch.testing.assert_close(inputs[name], expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)
    with pytest.raises(ValueError, match='uniform draw'):
        select_jump_request(**inputs)
    # Explicit ownership has a fixed draw requirement, independent of mask values.
    with pytest.raises(ValueError, match='uniform draw'):
        select_jump_request(**inputs, packet=packet(3), packet_owned=torch.ones(3, dtype=torch.bool))


@pytest.mark.parametrize('owned,with_packet', [
    (torch.ones(3), True), (torch.ones(2, dtype=torch.bool), True),
    (torch.zeros(3, dtype=torch.bool), False),
])
def test_invalid_ownership_is_rejected_without_data_dependent_checks(owned, with_packet):
    with pytest.raises(ValueError, match='packet_owned'):
        select_jump_request(**selection_inputs(), interval_uniform=torch.zeros(3),
                            packet=packet(3) if with_packet else None, packet_owned=owned)


def test_owned_empty_packets_suppress_only_their_own_automatic_schedule_and_gradients():
    inputs = selection_inputs()
    inputs['rise_m'] = inputs['rise_m'].requires_grad_()
    external_height = torch.full((3,), .2, requires_grad=True)
    external = RequestCommand(torch.tensor([-1, 44, 88]), external_height,
        cancel_id=torch.tensor([-1, 10, 9]), safety_abort=torch.tensor([False, True, True]))
    result = select_jump_request(**inputs, interval_uniform=torch.zeros(3), packet=external,
                                 packet_owned=torch.tensor([True, False, True]))
    assert result.request_id.tolist() == [-1, 1, 88]
    assert result.next_id.tolist() == [0, 2, 2]
    assert result.next_request_s.tolist() == [0., 5., 0.]
    assert result.cancel_id.tolist() == [-1, -1, 9]
    assert result.safety_abort.tolist() == [False, False, True]
    result.requested_height_m.sum().backward()
    assert external_height.grad.tolist() == [1., 0., 1.]
    assert inputs['rise_m'].grad.tolist() == [0., 1., 0.]


def test_unowned_packet_is_identical_to_the_original_automatic_selection():
    inputs = selection_inputs()
    draw = torch.tensor([.1, .3, .7])
    expected = select_jump_request(**inputs, interval_uniform=draw)
    actual = select_jump_request(**inputs, interval_uniform=draw, packet=packet(3),
                                 packet_owned=torch.zeros(3, dtype=torch.bool))
    for field in fields(expected):
        torch.testing.assert_close(getattr(actual, field.name), getattr(expected, field.name), rtol=0, atol=0)


def configure_batch(*, gated=False):
    native = native_jumps(9)
    if gated:
        install_jump_takeoff(native, {})
    owner = native._terrain_course.tasks
    native.time.fill_(2.)
    owner.elapsed.fill_(2.)
    owner.protocol.last_time_s.fill_(2.)
    owner.family[:] = torch.tensor([int(TaskFamily.REQUEST), int(TaskFamily.REQUEST), int(TaskFamily.HEIGHT),
        int(TaskFamily.HEIGHT), int(TaskFamily.RECOVERY), *([int(TaskFamily.REQUEST)] * 4)])
    owner.base_mode[4] = int(CommandMode.RECOVERY)
    owner.next_id.copy_(torch.arange(10, 19))
    owner.next_request_s[1] = 100.
    owner.protocol.active_id[6:] = torch.tensor([7, 8, -1])
    owner.protocol.max_seen_id[6:] = torch.tensor([7, 8, -1])
    owner.protocol.accepted_at_s[6:8] = 1.99
    owner.protocol.height_m[6:8] = .06
    owner.physical_id[6:8] = torch.tensor([7, 8])
    owner._physical_accepted_at_s[6:8] = 1.99
    owner.active_rise[6:8] = .06
    owner.latched_height.fill_(.32)
    owned = torch.tensor([False, False, False, True, True, False, True, False, True])
    blocked = torch.tensor([False, False, False, False, False, True, False, False, True])
    external = RequestCommand(torch.tensor([100, 101, 102, 103, 104, 105, -1, 107, 108]),
        torch.full((9,), .2), cancel_id=torch.tensor([-1, -1, -1, -1, -1, -1, 7, -1, -1]),
        safety_abort=torch.tensor([True, True, True, False, False, True, False, True, False]))
    if gated:
        owner._last_supported[0] = False
        owner._last_supported[3] = False  # Caller-owned packets retain native external semantics.
    return native, external, owned, blocked


def single_row_native_oracles(native, external, owned, blocked, *, gated):
    """Run the unchanged native step separately with each row's original path."""
    owner = native._terrain_course.tasks
    rows, draws, expected = [], [], []
    for index in range(len(owned)):
        row = native_jumps(1)
        if gated:
            install_jump_takeoff(row, {})
        target = row._terrain_course.tasks
        for name in (*JUMP_ID_NAMES, *JUMP_CLOCK_NAMES, *JUMP_VALUE_NAMES):
            if name == 'now_s':
                continue
            before, after = (owner.protocol, target.protocol) if name.startswith('protocol.') else (owner, target)
            key = name.split('.')[-1]
            getattr(after, key).copy_(getattr(before, key)[index:index + 1])
        for name in ('family', 'base_mode', 'elapsed'):
            getattr(target, name).copy_(getattr(owner, name)[index:index + 1])
        target.stream.current.copy_(owner.stream.current[index:index + 1])
        target.generator.manual_seed(800 + index)
        rng = torch.Generator().set_state(target.generator.get_state())
        draws.append(torch.rand(1, generator=rng)[0])
        if gated:
            row.supermimic_jump_trigger.last_end_s.copy_(native.supermimic_jump_trigger.last_end_s[index:index + 1])
            row.supermimic_jump_trigger.was_active.copy_(native.supermimic_jump_trigger.was_active[index:index + 1])
        row.time.fill_(2.02)
        command = None
        if owned[index]:
            command = replace(external, **{field.name: getattr(external, field.name)[index:index + 1]
                for field in fields(external) if isinstance(getattr(external, field.name), torch.Tensor)})
        generator_before = target.generator.get_state().clone()
        result = target.step(row.time, estimate_at(torch.tensor([2.], dtype=torch.float64)), torch.zeros(1),
            command=command, blocked_by_stair=blocked[index:index + 1])
        assert torch.equal(target.generator.get_state(), generator_before if command is not None else rng.get_state())
        expected.append((row, result, int(target.request_counters.argmax()),
                         int(target.request_end_counts.argmax()) if target.request_end_counts.sum() else int(RequestEnd.NONE)))
        rows.append(row)
    return torch.stack(draws), expected


@pytest.mark.parametrize('gated', [False, True])
def test_mixed_world_batch_matches_independent_unchanged_native_rows(gated):
    native, external, owned, blocked = configure_batch(gated=gated)
    owner = native._terrain_course.tasks
    state = JumpTaskState.capture(native)
    captured = {name: value.clone() for name, value in state.record().items()}
    task = JumpTask(JumpTaskConfig.capture(native))
    draws, expected = single_row_native_oracles(native, external, owned, blocked, gated=gated)
    rng = torch.get_rng_state().clone()
    result = task.advance(state, now_s=torch.full((9,), 2.02, dtype=torch.float64),
        estimate=estimate_at(native.time), admitted=owner.trajectory.end.clone(), base_mode=owner.base_mode,
        task_family=owner.family, elapsed_s=owner.elapsed + .02, ground_z=torch.zeros(9),
        blocked_by_stair=blocked, interval_uniform=draws, packet=external, packet_owned=owned,
        previous_command_height=owner.stream.current[:, 3].clone(), previous_elapsed_s=owner.elapsed)
    assert torch.equal(torch.get_rng_state(), rng)
    expected_records = [JumpTaskState.capture(row).record() for row, *_ in expected]
    for name, actual in result.state.record().items():
        torch.testing.assert_close(actual, torch.cat([record[name] for record in expected_records]), rtol=0, atol=0)
        torch.testing.assert_close(state.record()[name], captured[name], rtol=0, atol=0)
    for field, index in (('command', 0), ('mode', 1), ('request_height', 2), ('request_clock', 3)):
        values = [output[index].admitted if index == 0 else output[index] for _, output, *_ in expected]
        torch.testing.assert_close(getattr(result, field), torch.cat(values), rtol=0, atol=0)
    assert result.response.tolist() == [row[2] for row in expected]
    assert result.end_reason.tolist() == [row[3] for row in expected]
    assert result.response[3] == int(RequestResponse.ACCEPTED)  # external HEIGHT row is still allowed
    assert result.response[4] == int(RequestResponse.NOT_READY)  # original RECOVERY exclusion
    assert result.end_reason[6] == int(RequestEnd.CANCELLED)
    assert result.state.active_id[7] == 8  # unowned safety_abort packet was ignored
    assert result.state.ids[3, 3] == 104  # protocol watermark still advances past external IDs
    assert result.state.values[3, 3] == state.values[3, 3]  # owned automatic clock stays dormant
    assert result.deferred[0].item() == gated
    assert not result.deferred[3]  # no extra gate on owned external rows


def test_explicit_all_owned_mask_matches_legacy_external_batch():
    native, external, _, blocked = configure_batch()
    owner = native._terrain_course.tasks
    task, state = JumpTask(JumpTaskConfig.capture(native)), JumpTaskState.capture(native)
    args = dict(now_s=torch.full((9,), 2.02, dtype=torch.float64), estimate=estimate_at(native.time),
        admitted=owner.trajectory.end.clone(), base_mode=owner.base_mode, task_family=owner.family,
        elapsed_s=owner.elapsed + .02, ground_z=torch.zeros(9), blocked_by_stair=blocked, packet=external)
    legacy = task.advance(state, **args)
    masked = task.advance(state, **args, packet_owned=torch.ones(9, dtype=torch.bool), interval_uniform=torch.zeros(9))
    for name, value in legacy.state.record().items():
        torch.testing.assert_close(masked.state.record()[name], value, rtol=0, atol=0)
    torch.testing.assert_close(masked.response, legacy.response, rtol=0, atol=0)


def test_external_consumed_id_is_followed_by_a_fresh_automatic_id_after_ownership_releases():
    native = native_jumps(2)
    owner = native._terrain_course.tasks
    task, state = JumpTask(JumpTaskConfig.capture(native)), JumpTaskState.capture(native)
    args = dict(admitted=owner.trajectory.end.clone(), base_mode=owner.base_mode,
        task_family=owner.family, ground_z=torch.zeros(2), interval_uniform=torch.zeros(2))
    # Row 0 consumes a blocked external ID while row 1 follows its normal schedule.
    first = task.advance(state, **args, now_s=torch.full((2,), .02, dtype=torch.float64),
        estimate=estimate_at(native.time), elapsed_s=owner.elapsed + .02,
        blocked_by_stair=torch.tensor([True, False]), packet=packet(2), packet_owned=torch.tensor([True, False]))
    assert first.response.tolist() == [int(RequestResponse.NOT_READY), int(RequestResponse.ACCEPTED)]
    assert first.state.ids[:, 3].tolist() == [101, 1]
    second = task.advance(first.state, **args, now_s=torch.full((2,), .04, dtype=torch.float64),
        estimate=estimate_at(torch.full((2,), .02, dtype=torch.float64)), elapsed_s=owner.elapsed + .04,
        blocked_by_stair=torch.zeros(2, dtype=torch.bool))
    assert second.state.active_id.tolist() == [101, 0]
    assert second.response[0] == int(RequestResponse.ACCEPTED)
