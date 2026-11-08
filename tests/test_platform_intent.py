"""Caller boundary decisions, actual protocol responses and restartable records."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.course.platform_intent import (
    PLATFORM_INTENT_RECORD_FIELDS, PlatformIntentPhase, PlatformIntentState,
    consume_platform_intent, prepare_platform_intent,
)
from supermimic_rl.course.request_selection import select_jump_request
from transformer_rl.formal.control.observation import CommandMode
from transformer_rl.formal.tasks.curriculum import TaskFamily
from transformer_rl.formal.tasks.protocol import (
    CompletedEstimate, RequestEnd, RequestProtocol, RequestProtocolConfig, RequestResponse,
)


def plan(target=(0,), earliest=None):
    target = torch.tensor(target, dtype=torch.int64)
    selected = target >= 0
    n = len(target)
    return PlatformIntentState.plan(target_index=target,
        earliest_episode_s=torch.zeros(n, dtype=torch.float64) if earliest is None
        else torch.tensor(earliest, dtype=torch.float64),
        rise_m=torch.where(selected, .2, 0.), support_height_m=torch.where(selected, .4, 0.))


def prepare(state, **overrides):
    n = len(state.phase)
    values = dict(now_s=torch.full((n,), .02, dtype=torch.float64), elapsed_s=torch.zeros(n),
        policy_dt_s=.02, timeout_s=6., physical_id=torch.full((n,), -1, dtype=torch.int64),
        physical_accepted_at_s=torch.zeros(n, dtype=torch.float64),
        stair_active_id=torch.full((n,), -1, dtype=torch.int64), stair_idle=torch.ones(n, dtype=torch.bool),
        native_ready=torch.ones(n, dtype=torch.bool), admitted_height=torch.full((n,), .4),
        max_seen_id=torch.full((n,), -1, dtype=torch.int64), next_id=torch.zeros(n, dtype=torch.int64))
    values.update(overrides)
    return prepare_platform_intent(state, **values)


def response(n=1, *, response=RequestResponse.NONE, active=-1, finished=-1, end=RequestEnd.NONE):
    return SimpleNamespace(response=torch.full((n,), int(response)), active_id=torch.full((n,), active),
                           finished_id=torch.full((n,), finished), end_reason=torch.full((n,), int(end)))


def accept(prepared, ids=None):
    ids = prepared.jump_packet.request_id if ids is None else ids
    condition = response(len(ids), response=RequestResponse.ACCEPTED)
    condition.active_id = ids
    return consume_platform_intent(prepared, condition, physical_id=ids)


def assert_state_equal(a, b):
    for name, value in a.record().items():
        torch.testing.assert_close(value, b.record()[name], atol=0, rtol=0)


def test_waiting_owns_only_planned_rows_and_primes_height_without_early_stair_reservation():
    state = plan((0, -1, 1), earliest=(2., 0., 3.))
    prepared = prepare(state)
    assert prepared.packet_owned.tolist() == [True, False, True]
    assert prepared.height_override_mask.tolist() == [True, False, True]
    assert not prepared.reservation.any()
    assert prepared.jump_packet.request_id.tolist() == [-1, -1, -1]
    assert prepared.stair_packet.cancel_id.tolist() == [-1, -1, -1]
    assert_state_equal(prepared.state, state)


@pytest.mark.parametrize('earliest,cancel_now,cancel_episode,request_now,request_episode', [
    (2., 2.02, 2.019998688697815, 2.04, 2.039998788833618),
    (3., 3.02, 3.019997854232788, 3.04, 3.0399978351593018),
])
def test_matches_recorded_manual_pilot_cancel_and_next_boundary_submission(
        earliest, cancel_now, cancel_episode, request_now, request_episode):
    # Representative boundaries copied from the real four-row native pilot:
    # manual-platform-jump.jsonl SHA256
    # b9dbac0fbe930cfc73746195900a4510752edbfede5538d4607ddb9d3770d53d.
    # Its elapsed clock is float32; preparation adds policy_dt in float64.
    state = plan((0, 1), earliest=(earliest, earliest))
    cancelled = prepare(state, now_s=torch.full((2,), cancel_now, dtype=torch.float64),
        elapsed_s=torch.full((2,), cancel_episode-.02), stair_active_id=torch.zeros(2, dtype=torch.int64),
        stair_idle=torch.zeros(2, dtype=torch.bool))
    assert cancelled.stair_packet.cancel_id.tolist() == [0, 0]
    assert cancelled.jump_packet.request_id.tolist() == [-1, -1]
    assert cancelled.reservation.tolist() == [True, True]
    submitted = prepare(cancelled.state, now_s=torch.full((2,), request_now, dtype=torch.float64),
        elapsed_s=torch.full((2,), request_episode-.02))
    assert submitted.jump_packet.request_id.tolist() == [0, 0]
    assert submitted.stair_packet.cancel_id.tolist() == [-1, -1]
    accepted = accept(submitted)
    assert accepted.phase.tolist() == [int(PlatformIntentPhase.ACCEPTED)] * 2
    following = prepare(accepted, now_s=torch.full((2,), request_now+.02, dtype=torch.float64),
        elapsed_s=torch.full((2,), request_episode), physical_id=torch.zeros(2, dtype=torch.int64),
        physical_accepted_at_s=torch.full((2,), request_now, dtype=torch.float64))
    assert following.jump_packet.request_id.tolist() == [-1, -1]
    assert following.reservation.all()


@pytest.mark.parametrize('reason', ['stair_not_idle', 'height_not_admitted', 'gate_not_ready'])
def test_pending_reuses_existing_admission_evidence_without_bypassing_it(reason):
    kwargs = {'stair_not_idle': {'stair_idle': torch.tensor([False])},
              'height_not_admitted': {'admitted_height': torch.tensor([.399])},
              'gate_not_ready': {'native_ready': torch.tensor([False])}}[reason]
    prepared = prepare(plan(), **kwargs)
    assert prepared.state.phase.item() == PlatformIntentPhase.PENDING
    assert prepared.jump_packet.request_id.item() == -1
    assert prepared.packet_owned.item() and prepared.reservation.item()


def test_real_protocol_rejection_consumes_id_and_next_submission_is_fresh():
    protocol = RequestProtocol(1, config=RequestProtocolConfig(policy_dt_s=.02))
    state = plan()
    for tick, valid in enumerate((False, True), 1):
        now = torch.full((1,), tick*.02, dtype=torch.float64)
        prepared = prepare(state, now_s=now, elapsed_s=torch.tensor([(tick-1)*.02]),
                           max_seen_id=protocol.max_seen_id,
                           next_id=torch.maximum(protocol.max_seen_id+1, torch.zeros(1, dtype=torch.int64)))
        assert prepared.jump_packet.request_id.item() == tick-1
        estimate = CompletedEstimate(torch.zeros(1), torch.ones(1, 2), torch.zeros(1), torch.zeros(1),
                                     now-.02, torch.tensor([valid]))
        condition = protocol.step(now_s=now, estimate=estimate,
            request_id=prepared.jump_packet.request_id, requested_height_m=prepared.jump_packet.height_m,
            cancel_id=torch.full((1,), -1, dtype=torch.int64), allow_new=torch.tensor([True]),
            safety_abort=torch.tensor([False]))
        state = consume_platform_intent(prepared, condition, physical_id=protocol.active_id)
        assert protocol.max_seen_id.item() == tick-1
        assert state.phase.item() == (PlatformIntentPhase.ACCEPTED if valid else PlatformIntentPhase.PENDING)
    assert state.bound_request_id.item() == 1


def test_high_watermark_controls_ids_even_after_episode_reset():
    first = prepare(plan(), max_seen_id=torch.tensor([2**54]), next_id=torch.tensor([3]))
    assert first.jump_packet.request_id.item() == 2**54+1
    accepted = accept(first)
    reset = accepted.reset(torch.tensor([True]))
    assert reset.phase.item() == PlatformIntentPhase.WAITING and reset.bound_request_id.item() == -1
    following = prepare(reset, max_seen_id=first.jump_packet.request_id, next_id=torch.tensor([7]))
    assert following.jump_packet.request_id.item() == 2**54+2
    assert accepted.bound_request_id.item() == 2**54+1


def test_estimated_completion_retains_physical_tail_and_height_until_actual_end():
    accepted = accept(prepare(plan()))
    prepared = prepare(accepted, physical_id=torch.tensor([0]), physical_accepted_at_s=torch.tensor([.02], dtype=torch.float64))
    tail = consume_platform_intent(prepared, response(finished=0, end=RequestEnd.ESTIMATED_COMPLETE),
                                   physical_id=torch.tensor([0]))
    assert tail.phase.item() == PlatformIntentPhase.PHYSICAL_TAIL
    next_boundary = prepare(tail, physical_id=torch.tensor([0]))
    assert next_boundary.reservation.item() and next_boundary.height_override_mask.item()
    assert next_boundary.jump_packet.request_id.item() == -1
    done = consume_platform_intent(next_boundary, response(), physical_id=torch.tensor([-1]))
    assert done.phase.item() == PlatformIntentPhase.DONE
    released = prepare(done)
    assert not released.reservation.any() and not released.packet_owned.any()
    assert not released.height_override_mask.any()


@pytest.mark.parametrize('end', [RequestEnd.CANCELLED, RequestEnd.SAFETY_ABORT, RequestEnd.TIMEOUT,
                                RequestEnd.RESET, RequestEnd.COLD_RESTORE])
def test_protocol_nonestimated_end_closes_only_the_bound_request(end):
    accepted = accept(prepare(plan()))
    prepared = prepare(accepted, physical_id=torch.tensor([0]))
    unrelated = consume_platform_intent(prepared, response(finished=7, end=end), physical_id=torch.tensor([0]))
    assert unrelated.phase.item() == PlatformIntentPhase.ACCEPTED
    done = consume_platform_intent(prepared, response(finished=0, end=end), physical_id=torch.tensor([-1]))
    assert done.phase.item() == PlatformIntentPhase.DONE and done.bound_request_id.item() == 0


def test_pending_and_accepted_windows_reuse_the_original_two_timeout_origins():
    pending = plan(earliest=(2.,))
    before = prepare(pending, now_s=torch.tensor([100.], dtype=torch.float64), elapsed_s=torch.tensor([7.97]))
    assert before.reservation.item()  # Pending window uses episode time, not absolute time.
    expired = prepare(before.state, elapsed_s=torch.tensor([7.98]))
    assert expired.state.phase.item() == PlatformIntentPhase.DONE
    accepted = accept(prepare(pending, elapsed_s=torch.tensor([2.]), now_s=torch.tensor([100.], dtype=torch.float64)))
    before = prepare(accepted, now_s=torch.tensor([105.98], dtype=torch.float64), elapsed_s=torch.tensor([200.]),
                     physical_id=torch.tensor([0]), physical_accepted_at_s=torch.tensor([100.], dtype=torch.float64))
    assert before.state.phase.item() == PlatformIntentPhase.ACCEPTED
    expired = prepare(accepted, now_s=torch.tensor([106.], dtype=torch.float64),
                      physical_id=torch.tensor([0]), physical_accepted_at_s=torch.tensor([100.], dtype=torch.float64))
    assert expired.state.phase.item() == PlatformIntentPhase.DONE
    assert not expired.reservation.item() and expired.stair_packet.cancel_id.item() == -1


def test_mixed_rows_keep_ordinary_automatic_request_draws_and_scheduler_updates():
    state = plan((0, -1, 1), earliest=(2., 0., 0.))
    prepared = prepare(state)
    selection = select_jump_request(task_family=torch.full((3,), int(TaskFamily.REQUEST)),
        base_mode=torch.full((3,), int(CommandMode.NORMAL)), blocked_by_stair=torch.zeros(3, dtype=torch.bool),
        elapsed_s=torch.ones(3), active_id=torch.full((3,), -1, dtype=torch.int64),
        next_id=torch.tensor([4, 5, 6]), next_request_s=torch.zeros(3), rise_m=torch.full((3,), .1),
        interval_uniform=torch.tensor([.2, .3, .4]), packet=prepared.jump_packet,
        packet_owned=prepared.packet_owned)
    assert selection.request_id.tolist() == [-1, 5, 0]
    assert selection.automatic_due.tolist() == [False, True, False]
    assert selection.next_id.tolist() == [4, 6, 6]
    torch.testing.assert_close(selection.next_request_s, torch.tensor([0., 5.5, 0.]))


def test_record_roundtrip_is_independent_and_resumes_identical_next_packets():
    state = accept(prepare(plan((0, 1))))
    state = replace(state, phase=torch.tensor([int(PlatformIntentPhase.PHYSICAL_TAIL), int(PlatformIntentPhase.DONE)]))
    saved = state.state_dict()
    restored = PlatformIntentState.from_state_dict(saved)
    assert_state_equal(state, restored)
    kwargs = {'physical_id': torch.tensor([0, -1]), 'physical_accepted_at_s': torch.tensor([.02, .02], dtype=torch.float64)}
    before, after = prepare(state, **kwargs), prepare(restored, **kwargs)
    assert_state_equal(before.state, after.state)
    for name in ('packet_owned', 'reservation', 'height_override_mask', 'height_override_m'):
        assert torch.equal(getattr(before, name), getattr(after, name))
    assert torch.equal(before.jump_packet.request_id, after.jump_packet.request_id)
    saved['platform_intent_ids'].fill_(-1)
    assert state.bound_request_id.tolist() == [0, 0]
    assert restored.bound_request_id.tolist() == [0, 0]


def test_legacy_inactive_state_is_explicit_and_never_owns_or_overrides_anything():
    old = PlatformIntentState.inactive(3)
    restored = PlatformIntentState.decode(old.record())
    actual = prepare(restored)
    assert not actual.packet_owned.any() and not actual.reservation.any()
    assert not actual.height_override_mask.any()
    assert actual.jump_packet.request_id.tolist() == [-1, -1, -1]
    assert actual.stair_packet.cancel_id.tolist() == [-1, -1, -1]
    with pytest.raises(ValueError, match='legacy rows'):
        PlatformIntentState.decode({})


@pytest.mark.parametrize('corruption', ['format', 'version', 'dtype', 'unknown_target', 'accepted_without_id', 'missing'])
def test_restore_rejects_unknown_or_incomplete_intent_contract(corruption):
    saved = plan().state_dict()
    if corruption == 'format':
        saved['format'] = 'future-v2'
    elif corruption == 'version':
        with pytest.raises(ValueError, match='version'):
            PlatformIntentState.decode(saved, version=2)
        return
    elif corruption == 'dtype':
        saved['platform_intent_clock'] = saved['platform_intent_clock'].float()
    elif corruption == 'unknown_target':
        saved['platform_intent_ids'][0, 0] = 4
    elif corruption == 'accepted_without_id':
        saved['platform_intent_ids'][0, 1] = int(PlatformIntentPhase.ACCEPTED)
    else:
        del saved[next(iter(PLATFORM_INTENT_RECORD_FIELDS))]
    with pytest.raises(ValueError):
        PlatformIntentState.from_state_dict(saved)
