"""Request-owned height hints retain native/imagined protocol parity."""
from dataclasses import asdict, replace
import json

import pytest
import torch

from supermimic_rl.course.factory import create_course
from supermimic_rl.course.stair_label import install_tof_stair_label
from supermimic_rl.world_model_v2.stair_task import (
    StairPacket, StairTask, StairTaskConfig, StairTaskState,
)
from transformer_rl.formal.lab_env.course_tasks import RequestCommand
from transformer_rl.formal.tasks.protocol import RequestEnd, RequestResponse
from support.world_v2.stair_task import estimate_at, native_stairs


def advance_both(native, task, state, step, packet, *, base_mode=None, jump_active=None):
    """Compare the complete existing native boundary with its WM counterpart."""
    previous_time = native.time
    native.time = torch.full((native.num_envs,), (step + 1) * .02, dtype=torch.float64)
    native.admitted = torch.tensor([.8, 0., .2, .26 if step == 0 else .28]).repeat(native.num_envs, 1)
    native.base_mode = (torch.zeros(native.num_envs, dtype=torch.long)
                        if base_mode is None else base_mode)
    if jump_active is None:
        jump_active = torch.zeros(native.num_envs, dtype=torch.bool)
    native._terrain_course.tasks.active_id.copy_(torch.where(jump_active, 1, -1))
    native._completed_estimate = estimate_at(previous_time, native.num_envs)
    result = task.advance(state, packet, now_s=native.time, estimate=native._completed_estimate,
        admitted=native.admitted, base_mode=native.base_mode, jump_active=jump_active)
    native._terrain_course._pending_requests['stair'] = RequestCommand(
        packet.request_id, packet.height_m, packet.cancel_id, packet.safety_abort,
        completed_id=packet.completed_id)
    native._terrain_course.advance_skill()
    torch.testing.assert_close(result.command, native._command, rtol=0, atol=0)
    torch.testing.assert_close(result.mode, native._command_mode, rtol=0, atol=0)
    torch.testing.assert_close(result.body_contact_allowed, native._terrain_course.body_contact_allowed,
                               rtol=0, atol=0)
    torch.testing.assert_close(result.response, native._terrain_course._last_stair_response,
                               rtol=0, atol=0)
    for key, value in StairTaskState.capture(native).record().items():
        torch.testing.assert_close(result.state.record()[key], value, rtol=0, atol=0)
    return result


@pytest.mark.parametrize('lifetime', ['fixed', 'request'])
def test_default_is_unchanged_and_candidate_outlives_old_hint_until_exact_request_expiry(lifetime):
    native = native_stairs(1, labels=True, assist=True)
    native.cfg.stair_height_assist_spec = replace(native.cfg.stair_height_assist_spec, hold_s=2.)
    protocol = native._terrain_course.stair_requests
    protocol.config = replace(protocol.config, timeout_s=6.)
    original_assist = native.cfg.stair_height_assist_spec
    receipt = install_tof_stair_label(native, height_hint_lifetime=lifetime)
    if lifetime == 'fixed':
        assert native.cfg.stair_height_assist_spec is original_assist
        assert receipt == {'stair_label': 'STAIR set by the ToF stair state machine only; terrain-kind labels disabled'}
    else:
        assert receipt['height_hint']['runtime_hold_s'] == 2.
        assert receipt['height_hint']['effective_hold_s'] == 6.
        assert native.cfg.stair_height_assist_spec == replace(original_assist, hold_s=6.)
        assert original_assist.hold_s == 2.  # The immutable runtime object was not rewritten.
    task, state = StairTask(StairTaskConfig.capture(native)), StairTaskState.capture(native)
    for step in range(302):
        packet = StairPacket.empty(1, 'cpu')
        if step in (0, 75, 300, 301):
            packet.request_id.fill_(10)
            packet.height_m.fill_(.20)
        result = advance_both(native, task, state, step, packet)
        expected = .33 if step < (100 if lifetime == 'fixed' else 300) else .28
        assert result.command[0, 3].item() == pytest.approx(expected)
        if step < 300:
            assert result.state.ids[0, 1] == 10
            assert result.state.clock[0, 1] == .02  # Replayed packets never refresh acceptance.
        else:
            assert result.state.ids[0, 1] == -1
            assert result.state.ids[0, 3] == 0
            assert result.mode[0] == 0
            assert result.response[0] == int(RequestResponse.REPLAY)
        if step == 300:
            assert result.end_reason[0] == int(RequestEnd.TIMEOUT)
        state = result.state


def test_completion_cancel_abort_jump_recovery_and_busy_id_use_existing_protocol():
    native = native_stairs(7, labels=True, assist=True)
    install_tof_stair_label(native, height_hint_lifetime='request')
    task, state = StairTask(StairTaskConfig.capture(native)), StairTaskState.capture(native)
    for step in range(12):
        packet = StairPacket.empty(7, 'cpu')
        mode = torch.zeros(7, dtype=torch.long)
        jump = torch.zeros(7, dtype=torch.bool)
        if step == 0:
            packet.request_id.fill_(100)
            packet.height_m.fill_(.20)
        elif step == 3:
            packet.request_id[6] = 101  # A consumed BUSY packet must not restart the hint.
            packet.height_m[6] = .08
        elif step == 4:
            packet.completed_id[0] = 100
            packet.cancel_id[1] = 100
            packet.safety_abort[2] = True
            jump[3], mode[3] = True, 2
        if step in (3, 4):
            mode[4] = 3  # Native recovery masks raised output, but does not cancel this request.
        result = advance_both(native, task, state, step, packet, base_mode=mode, jump_active=jump)
        if step == 3:
            assert result.response[6] == int(RequestResponse.BUSY)
            assert result.state.clock[6, 1] == .02
        if step == 4:
            assert result.end_reason[:4].tolist() == [int(RequestEnd.ESTIMATED_COMPLETE),
                int(RequestEnd.CANCELLED), int(RequestEnd.SAFETY_ABORT), int(RequestEnd.SAFETY_ABORT)]
            assert result.state.ids[:4, 1].tolist() == [-1] * 4
            assert result.state.ids[:4, 3].tolist() == [0] * 4
            torch.testing.assert_close(result.command[:4, 3], torch.full((4,), .28), rtol=0, atol=0)
        if step in (3, 4):
            assert result.command[4, 3].item() == pytest.approx(.28)
            assert result.state.ids[4, 1] == 100
        if 5 <= step < 10:
            assert result.command[4, 3].item() == pytest.approx(.33)
            assert result.command[6, 3].item() == pytest.approx(.33)
            assert result.state.clock[4, 1] == result.state.clock[6, 1] == .02
        if step == 10:
            assert result.end_reason[4:].tolist() == [int(RequestEnd.TIMEOUT)] * 3
        if step >= 10:
            assert result.state.ids[:, 1].tolist() == [-1] * 7
            torch.testing.assert_close(result.command[:, 3], torch.full((7,), .28), rtol=0, atol=0)
        state = result.state


def test_candidate_keeps_latched_height_gradient_without_mutating_start():
    native = native_stairs(1, labels=True, assist=True)
    install_tof_stair_label(native, height_hint_lifetime='request')
    task, state = StairTask(StairTaskConfig.capture(native)), StairTaskState.capture(native)
    original = {name: value.clone() for name, value in state.record().items()}
    captured = state
    first, following = torch.tensor([.26], requires_grad=True), torch.tensor([.28], requires_grad=True)
    for step in range(5):  # Beyond the legacy three-interval hint.
        packet = StairPacket.empty(1, 'cpu')
        if step == 0:
            packet.request_id.fill_(7)
            packet.height_m.fill_(.2)
        now = torch.tensor([(step + 1) * .02], dtype=torch.float64)
        command = torch.cat((torch.zeros(1, 3), (first if step == 0 else following)[:, None]), -1)
        result = task.advance(state, packet, now_s=now, estimate=estimate_at(now - .02, 1),
            admitted=command, base_mode=torch.zeros(1, dtype=torch.long), jump_active=torch.zeros(1, dtype=torch.bool))
        state = result.state
    grad = torch.autograd.grad(result.command[:, 3].sum(), (first, following))
    assert [value.item() for value in grad] == [1., 0.]
    for name, value in captured.record().items():
        torch.testing.assert_close(value, original[name], atol=0, rtol=0)


def test_candidate_parameters_reject_historical_hint_timing_and_roundtrip_config():
    native = native_stairs(1, labels=True, assist=True)
    old = StairTaskState.capture(native)
    install_tof_stair_label(native, height_hint_lifetime='request')
    config = StairTaskConfig.capture(native)
    restored = StairTaskConfig.from_dict(json.loads(json.dumps(asdict(config))))
    assert restored == config
    with pytest.raises(ValueError, match='parameters differ'):
        StairTask(restored).advance(old, StairPacket.empty(1, 'cpu'),
            now_s=native.time + .02, estimate=estimate_at(native.time, 1),
            admitted=torch.tensor([[.8, 0., .2, .26]]), base_mode=torch.zeros(1, dtype=torch.long),
            jump_active=torch.zeros(1, dtype=torch.bool))


def test_candidate_receipt_changes_schema_source_without_rewriting_runtime_pin():
    from supermimic_rl.cli import _native_schema_source

    native = native_stairs(1, labels=True, assist=True)
    fixed = install_tof_stair_label(native)
    request = install_tof_stair_label(native, height_hint_lifetime='request')
    provenance = {'upstream_commit': 'frozen', 'runtime_sha256': 'a' * 64, 'stair_label': fixed}
    old = json.loads(_native_schema_source(provenance))
    new = json.loads(_native_schema_source({**provenance, 'stair_label': request}))
    assert new['runtime_sha256'] == old['runtime_sha256'] == 'a' * 64
    assert new != old
    assert new['stair_label']['height_hint']['effective_hold_s'] == .2
    assert 'height_hint' not in old['stair_label']


@pytest.mark.parametrize('lifetime', [True, None, 'forever'])
def test_unknown_lifetime_rejected_before_constructing_a_simulator(lifetime):
    with pytest.raises(ValueError, match='lifetime'):
        create_course({'stair_height_hint_lifetime': lifetime})


def test_request_lifetime_needs_explicit_label_and_existing_enabled_hint():
    with pytest.raises(ValueError, match='explicit ToF'):
        create_course({'stair_height_hint_lifetime': 'request'})
    native = native_stairs(1, labels=True, assist=False)
    with pytest.raises(ValueError, match='enabled native height assist'):
        install_tof_stair_label(native, height_hint_lifetime='request')
