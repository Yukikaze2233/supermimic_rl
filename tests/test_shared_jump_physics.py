"""Flat native parity and request-bound direct platform landing evidence."""
from copy import deepcopy
from dataclasses import fields, replace

import pytest
import torch

from supermimic_rl.course.jump_physics import (
    PlatformLandingEvidence, PlatformLandingPhase, PlatformLandingState,
    advance_platform_landing, observe_jump_physics,
)
from supermimic_rl.course.reward import install_flight_latch
from supermimic_rl.world_model_v2.jump_task import JumpTask, JumpTaskConfig, JumpTaskState
from support.world_v2.jump_task import native_jumps
from test_world_v2_jump_task import physical_ticks


def active_native(n=1):
    native = native_jumps(n)
    owner = native._terrain_course.tasks
    owner.protocol.active_id.copy_(torch.arange(n))
    owner.active_id.copy_(owner.protocol.active_id)
    owner.physical_id.copy_(owner.active_id)
    owner.active_rise.fill_(.2)
    owner.latched_height.fill_(.32)
    owner.supported_start.fill_(True)
    return native


def evidence(n=1, *, clear=False, wheels=(False, False), target=(False, False),
             other=False, valid=True, supported=False, height=.52, selected=True):
    return PlatformLandingEvidence(
        torch.full((n,), selected), torch.full((n,), valid), torch.full((n,), clear),
        torch.tensor(wheels).expand(n, 2).clone(), torch.tensor(target).expand(n, 2).clone(),
        torch.full((n,), other), torch.full((n,), supported), torch.full((n,), height))


def observe(owner, state, sample, *, z=.52, vz=0., force=None, omega=0., tilt=0.):
    n = len(owner.physical_id)
    if force is None:
        force = sample.wheel_touch.float() * 20.
    return observe_jump_physics(owner, force, torch.full((n,), vz), torch.full((n,), tilt),
                                torch.full((n,), z), angular_speed_rad_s=torch.full((n,), omega),
                                platform_state=state, platform_evidence=sample)


def lift(owner, state):
    for _ in range(2):
        state = observe(owner, state, evidence(len(owner.physical_id), clear=True), vz=.4)
    assert (state.phase == int(PlatformLandingPhase.FLIGHT)).all()
    return state


def assert_tensor_owners_equal(actual, expected):
    for name, value in vars(expected).items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(getattr(actual, name), value, atol=0, rtol=0, msg=name)


@pytest.mark.parametrize('explicit_flat', [False, True])
def test_flat_updates_match_real_native_helper_every_tick_and_single_payment(explicit_flat):
    expected = active_native(7)._terrain_course.tasks
    actual = deepcopy(expected)
    expected._physical_cancelled[1] = actual._physical_cancelled[1] = True
    expected.supported_start[2] = actual.supported_start[2] = False
    state = PlatformLandingState.empty(actual.physical_id) if explicit_flat else None
    sample = evidence(7, selected=False) if explicit_flat else None
    for tick in range(425):
        force = torch.full((7, 2), 20.)
        z, vz, tilt, omega = torch.full((7,), .32), torch.zeros(7), torch.zeros(7), torch.zeros(7)
        if tick < 4:
            force.zero_()
            z.fill_(.52)
            vz.fill_(.4)
            z[3] = .57  # Existing peak overshoot does not qualify.
            z[4] = .40  # Nor does a small hop.
            vz[5] = -.4  # A falling birth never gains ascent evidence.
        tilt[6] = .3
        expected.observe_physics(force, vz, tilt, z, angular_speed_rad_s=omega)
        state = observe_jump_physics(actual, force, vz, tilt, z, angular_speed_rad_s=omega,
                                     platform_state=state, platform_evidence=sample)
        assert_tensor_owners_equal(actual, expected)
    assert actual.physical_outcome_counts.tolist() == [1, 0, 0]
    assert actual.paid_id.tolist() == [0, -1, -1, -1, -1, -1, -1]


def test_two_unloaded_wheels_and_body_support_do_not_certify_platform_flight():
    owner = active_native()._terrain_course.tasks
    state = PlatformLandingState.empty(owner.physical_id)
    for _ in range(4):
        state = observe(owner, state, evidence(other=True), vz=.4)
    assert owner.flight_seen.item()  # Retained raw native wheel-unload history.
    assert state.phase.item() == PlatformLandingPhase.BEFORE_FLIGHT
    landing = evidence(wheels=(True, True), target=(True, True), supported=True)
    for _ in range(405):
        state = observe(owner, state, landing)
    assert not owner.physical_outcome_counts[0]
    assert owner.landed_s.item() == 0.


def test_single_target_wheel_then_both_can_land_once_and_continue_on_real_ground():
    owner = active_native()._terrain_course.tasks
    state = lift(owner, PlatformLandingState.empty(owner.physical_id))
    state = observe(owner, state, evidence(wheels=(True, False), target=(True, False)))
    assert state.phase.item() == PlatformLandingPhase.TARGET_SINGLE_WHEEL
    assert owner.landed_s.item() == 0.
    state = observe(owner, state, evidence(wheels=(True, True), target=(True, True), supported=True))
    assert state.phase.item() == PlatformLandingPhase.TARGET_BOTH_WHEELS
    # A moving robot need not remain inside a short target for the whole 2 s.
    continuation = evidence(wheels=(True, True), supported=True, height=.50)
    for _ in range(405):
        state = observe(owner, state, continuation, z=.50)
    assert owner.physical_outcome_counts.tolist() == [1, 0, 0]
    assert owner.physical_id.item() == -1 and owner.paid_id.item() == 0
    assert state.phase.item() == PlatformLandingPhase.TARGET_BOTH_WHEELS


@pytest.mark.parametrize('bad', [
    evidence(wheels=(True, False)),
    evidence(wheels=(True, True), target=(True, False)),
    evidence(other=True),
    evidence(wheels=(True, True), target=(True, True), other=True),
])
def test_wrong_platform_or_nonwheel_first_contact_cannot_be_repaired_by_driving_on(bad):
    owner = active_native()._terrain_course.tasks
    state = lift(owner, PlatformLandingState.empty(owner.physical_id))
    state = observe(owner, state, bad)
    assert state.phase.item() == PlatformLandingPhase.MISSED
    landing = evidence(wheels=(True, True), target=(True, True), supported=True)
    for _ in range(405):
        state = observe(owner, state, landing)
    assert state.phase.item() == PlatformLandingPhase.MISSED
    assert owner.landed_s.item() == 0. and owner.paid_id.item() == -1


def test_missing_contact_between_flight_and_landing_is_unknown_not_a_success_or_known_miss():
    owner = active_native()._terrain_course.tasks
    state = lift(owner, PlatformLandingState.empty(owner.physical_id))
    state = observe(owner, state, evidence(valid=False))
    assert state.phase.item() == PlatformLandingPhase.UNKNOWN_CONTACT
    state = observe(owner, state, evidence(wheels=(True, True), target=(True, True), supported=True))
    assert state.phase.item() == PlatformLandingPhase.UNKNOWN_CONTACT
    assert owner.landed_s.item() == 0.


@pytest.mark.parametrize('bad', [
    evidence(wheels=(True, True), supported=False),
    evidence(wheels=(True, True), supported=True, valid=False),
    evidence(wheels=(True, True), supported=True, other=True),
    evidence(wheels=(True, True), supported=True, height=.62),
])
def test_current_contact_validity_height_and_support_still_break_stable_streak(bad):
    owner = active_native()._terrain_course.tasks
    state = lift(owner, PlatformLandingState.empty(owner.physical_id))
    landing = evidence(wheels=(True, True), target=(True, True), supported=True)
    for _ in range(100):
        state = observe(owner, state, landing)
    assert owner.landed_s.item() > .49
    state = observe(owner, state, bad)
    assert owner.landed_s.item() == 0.
    assert state.phase.item() == PlatformLandingPhase.TARGET_BOTH_WHEELS
    state = observe(owner, state, landing)
    torch.testing.assert_close(owner.landed_s, torch.tensor([.005]))


@pytest.mark.parametrize('defect', ['peak_low', 'peak_high', 'unsupported', 'cancelled', 'tilt', 'omega', 'vz'])
def test_target_contact_never_bypasses_existing_peak_start_quiet_and_cancel_rules(defect):
    owner = active_native()._terrain_course.tasks
    state = lift(owner, PlatformLandingState.empty(owner.physical_id))
    z, vz, tilt, omega = .52, 0., 0., 0.
    if defect == 'peak_low':
        owner.active_rise.fill_(.4)
    elif defect == 'peak_high':
        owner.peak_rise.fill_(.3)
    elif defect == 'unsupported':
        owner.supported_start.zero_()
    elif defect == 'cancelled':
        owner._physical_cancelled.fill_(True)
    elif defect == 'tilt':
        tilt = .3
    elif defect == 'omega':
        omega = .6
    else:
        vz = .2
    landing = evidence(wheels=(True, True), target=(True, True), supported=True)
    for _ in range(405):
        state = observe(owner, state, landing, z=z, vz=vz, tilt=tilt, omega=omega)
    assert owner.paid_id.item() == -1
    assert not owner.physical_outcome_counts[0]


def test_bound_platform_request_cannot_fall_back_to_flat_success_when_selected_is_dropped():
    owner = active_native()._terrain_course.tasks
    state = lift(owner, PlatformLandingState.empty(owner.physical_id))
    # This return would satisfy the flat task. Losing target ownership does not
    # turn an existing platform request into a different, easier task.
    dropped = evidence(selected=False, wheels=(True, True), supported=True, height=.32)
    for _ in range(405):
        state = observe(owner, state, dropped, z=.32)
    assert owner.paid_id.item() == -1 and owner.landed_s.item() == 0.
    assert state.phase.item() == PlatformLandingPhase.UNKNOWN_CONTACT


def test_new_request_and_reset_do_not_inherit_target_landing_or_mutate_old_history():
    owner = active_native(2)._terrain_course.tasks
    state = lift(owner, PlatformLandingState.empty(owner.physical_id))
    state = observe(owner, state, evidence(2, wheels=(True, True), target=(True, True), supported=True))
    snapshot = deepcopy(state)
    reset = state.reset(torch.tensor([True, False]))
    assert reset.request_id.tolist() == [-1, 1]
    assert reset.phase.tolist() == [0, int(PlatformLandingPhase.TARGET_BOTH_WHEELS)]
    following = advance_platform_landing(state, evidence(2, wheels=(True, True), target=(True, True)),
        physical_id=torch.tensor([7, 8]), ascent_seen=torch.ones(2, dtype=torch.bool))
    assert following.request_id.tolist() == [7, 8]
    assert following.phase.tolist() == [0, 0]
    for field in fields(state):
        assert torch.equal(getattr(state, field.name), getattr(snapshot, field.name))


def test_mixed_flat_platform_rows_pay_only_their_own_landing_branch():
    owner = active_native(2)._terrain_course.tasks
    state = lift(owner, PlatformLandingState.empty(owner.physical_id))
    # Rebind row 0 as a new flat request; row 1 remains a platform request.
    owner.physical_id[0] = owner.active_id[0] = 2
    selected = torch.tensor([False, True])
    sample = replace(evidence(2, wheels=(True, True), target=(True, True), supported=True), selected=selected)
    for _ in range(405):
        state = observe_jump_physics(owner, torch.full((2, 2), 20.), torch.zeros(2), torch.zeros(2),
            torch.tensor([.32, .52]), angular_speed_rad_s=torch.zeros(2),
            platform_state=state, platform_evidence=sample)
    assert owner.paid_id.tolist() == [2, 1]
    assert owner.physical_outcome_counts[0].item() == 2


def test_imagined_four_ticks_share_updater_and_latch_order_including_midinterval_payment():
    native = active_native()
    owner = native._terrain_course.tasks
    pstate = lift(owner, PlatformLandingState.empty(owner.physical_id))
    landing = evidence(wheels=(True, True), target=(True, True), supported=True)
    pstate = observe(owner, pstate, landing)
    owner.landed_s.fill_(1.992)
    install_flight_latch(owner)
    task, state = JumpTask(JumpTaskConfig.capture(native)), JumpTaskState.capture(native)
    physical = physical_ticks(1, z=.52)
    stack = PlatformLandingEvidence(**{field.name: getattr(landing, field.name)[:, None].expand(
        1, 4, *getattr(landing, field.name).shape[1:]).clone() for field in fields(landing)})
    expected_platform = pstate
    expected_active, expected_stable = [], []
    # Native installation replaces the underlying updater before the latch.
    from supermimic_rl.reward_kernel import advance_flight_latch
    latch = state._flight_latch_state()
    for tick in range(4):
        expected_platform = observe(owner, expected_platform, stack.tick(tick))
        latch = advance_flight_latch(*latch, current=owner.physical_id,
                                    ascent_seen=owner.ascent_seen, air_ticks=owner.air_ticks)
        expected_active.append((owner.physical_id >= 0).clone())
        expected_stable.append(owner.landed_s.clone())
    actual = task.observe_interval(state, physical=physical, wheel_support_load=torch.full((1, 4, 2), 20.),
        base_com_offset=torch.zeros(1, 3), base_mode=owner.base_mode, command_height=torch.tensor([.32]),
        ground_z=torch.full((1, 4), .2), platform_state=pstate, platform_evidence=stack)
    assert actual.success_credited.item()
    assert actual.physical_active.tolist() == [[True, False, False, False]]
    torch.testing.assert_close(actual.physical_active, torch.stack(expected_active, -1))
    torch.testing.assert_close(actual.landed_s, torch.stack(expected_stable, -1), atol=0, rtol=0)
    for field in fields(pstate):
        assert torch.equal(getattr(actual.platform_state, field.name), getattr(expected_platform, field.name))
    torch.testing.assert_close(actual.state.flight_latch,
                               torch.stack((latch[0].long(), latch[1].long(), latch[2].long()), -1))


def test_optional_platform_input_requires_complete_typed_evidence_before_mutating_owner():
    owner = active_native()._terrain_course.tasks
    original = deepcopy(owner)
    with pytest.raises(ValueError, match='both request history'):
        observe_jump_physics(owner, torch.zeros(1, 2), torch.zeros(1), torch.zeros(1), torch.ones(1),
            angular_speed_rad_s=torch.zeros(1), platform_state=PlatformLandingState.empty(owner.physical_id))
    bad = replace(evidence(), measurement_valid=torch.ones(1))
    with pytest.raises(ValueError, match='typed physical rows'):
        observe(owner, PlatformLandingState.empty(owner.physical_id), bad)
    assert_tensor_owners_equal(owner, original)
