"""Recovery state and handoff agree with the pinned native task methods."""
from dataclasses import asdict, fields, replace

import pytest
import torch

from supermimic_rl.world_model_v2 import so3
from supermimic_rl.world_model_v2.motion_task import MotionTask
from supermimic_rl.world_model_v2.physical import PhysicalState
from supermimic_rl.world_model_v2.recovery_task import (
    RECOVERY_FEATURE_DIM, RECOVERY_RECORD_DTYPES, RECOVERY_RECORD_FIELDS,
    RecoveryBoundary, RecoveryTask, RecoveryTaskConfig, RecoveryTaskState,
)
from transformer_rl.formal.control.observation import CommandMode
from transformer_rl.formal.tasks.curriculum import TaskFamily
from transformer_rl.formal.tasks.protocol import CompletedEstimate
from support.world_v2.recovery_task import native_recovery


def physical_ticks(n, *, requires_grad=False):
    position = torch.zeros(n, 4, 3)
    position[..., 2] = .32
    return PhysicalState(position.requires_grad_(requires_grad), torch.eye(3).expand(n, 4, 3, 3).clone(),
        torch.zeros(n, 4, 3), torch.zeros(n, 4, 3), torch.zeros(n, 4, 6), torch.zeros(n, 4, 8))


def completed(now, *, contact=.9, tilt=0.):
    n = len(now)
    return CompletedEstimate(torch.zeros(n), torch.full((n, 2), contact), torch.full((n,), tilt),
        torch.zeros(n), now.clone(), torch.ones(n, dtype=torch.bool))


def observe_native(native, physical, wheel, base, owned, *, com_offset=None, ground=None):
    n = len(wheel)
    offset = torch.zeros(n, 3) if com_offset is None else com_offset
    ground = torch.zeros(n, 4) if ground is None else ground
    for tick in range(4):
        sample = physical.tick(tick)
        com = sample.velocity + torch.linalg.cross(sample.angular_velocity,
            (sample.rotation @ offset[..., None]).squeeze(-1))
        gravity = sample.gravity_body
        native._terrain_course.recovery.observe_physics(
            height_m=sample.position[:, 2] - ground[:, tick],
            command_height_m=torch.full((n,), .32),
            tilt_rad=torch.atan2(gravity[:, :2].norm(dim=-1), -gravity[:, 2]),
            angular_speed_rad_s=sample.angular_velocity_body.norm(dim=-1),
            velocity_z_m_s=com[:, 2], wheel_load_n=wheel[:, tick], base_load_n=base[:, tick],
            policy_owned=owned if owned.ndim == 1 else owned[:, tick])


def same_state(actual, native):
    for name, expected in RecoveryTaskState.capture(native).record().items():
        torch.testing.assert_close(actual.record()[name], expected, rtol=0, atol=0)


def observe(task, state, physical, wheel, base, owned, *, offset=None, ground=None):
    n = len(wheel)
    return task.observe_interval(state, physical=physical, wheel_support_load=wheel,
        base_normal_load=base, base_com_offset=torch.zeros(n, 3) if offset is None else offset,
        command_height=torch.full((n,), .32), ground_z=torch.zeros(n, 4) if ground is None else ground,
        policy_owned=owned)


def test_capture_keeps_exact_clocks_flags_and_no_alias_to_native_state():
    native = native_recovery(2)
    owner = native._terrain_course.recovery
    native.time.fill_(1_000_000.015)
    owner._last_estimate_s[1] = 1_000_000.
    state = RecoveryTaskState.capture(native)
    assert {name: tuple(value.shape[1:]) for name, value in state.record().items()} == RECOVERY_RECORD_FIELDS
    assert all(value.dtype == RECOVERY_RECORD_DTYPES[name] for name, value in state.record().items())
    assert state.now_s[0].item() == 1_000_000.015
    assert state.clock[0, 4].isneginf()
    before = {name: value.clone() for name, value in state.record().items()}
    owner.active.zero_()
    owner.policy_elapsed_s.add_(1)
    for name, value in RecoveryTaskState.decode(before).record().items():
        torch.testing.assert_close(state.record()[name], value, rtol=0, atol=0)
    config = RecoveryTaskConfig.capture(native)
    assert RecoveryTaskConfig.from_dict(asdict(config)) == config
    assert config.handoff.duration_s == 1.7
    assert config.handoff.forward_m_s == .6
    native.physics_dt = .01
    with pytest.raises(ValueError, match='physics clocks differ'):
        RecoveryTaskConfig.capture(native)


def test_four_birth_orientations_and_upright_handoff_match_native_over_time():
    native = native_recovery()
    task = RecoveryTask(RecoveryTaskConfig.capture(native))
    state = RecoveryTaskState.capture(native)
    initial = {name: value.clone() for name, value in state.record().items()}
    captured = state
    owned = torch.ones(4, dtype=torch.bool)
    success_count = torch.zeros(4, dtype=torch.long)
    handoff_count = torch.zeros_like(success_count)
    # Native catalogue root rolls: left, right, belly upright, back inverted.
    # These are input trajectories, not a claim of simulated robot recovery.
    roll = torch.tensor([-torch.pi / 2, torch.pi / 2, 0., torch.pi])
    for step in range(40):
        physical = physical_ticks(4)
        wheel, base = torch.full((4, 4, 2), 20.), torch.zeros(4, 4)
        estimate = completed(native.time)
        if step < 3:
            rotation = torch.zeros(4, 4, 3)
            rotation[..., 0] = roll[:, None]
            physical = replace(physical, rotation=so3.exp(rotation))
            physical.position[..., 2] = .10
            wheel.zero_()
            base.fill_(30.)
            estimate.wheel_contact_probability.fill_(.1)
            estimate.tilt_rad.copy_(roll.abs())
        observe_native(native, physical, wheel, base, owned)
        state = observe(task, state, physical, wheel, base, owned)
        same_state(state, native)
        now = task.boundary_time(state)
        expected = native._terrain_course.recovery.finish_interval(now, estimate,
            policy_owned=owned, failed=~owned)
        result = task.finish_interval(state, now_s=now, estimate=estimate, policy_owned=owned, failed=~owned)
        native.time = now
        same_state(result.state, native)
        torch.testing.assert_close(result.handoff, expected.handoff)
        torch.testing.assert_close(result.timeout, expected.timeout)
        handoff_count += result.handoff.long()
        success_count += result.verified_success.long()
        state = result.state
    assert handoff_count.tolist() == [1, 1, 1, 1]
    assert success_count.tolist() == [1, 1, 1, 1]
    assert not state.active.any()
    for name, value in initial.items():
        torch.testing.assert_close(captured.record()[name], value, rtol=0, atol=0)


def test_physical_success_cannot_select_mode_and_estimated_handoff_does_not_certify_success():
    native = native_recovery(2)
    task = RecoveryTask(RecoveryTaskConfig.capture(native))
    state = RecoveryTaskState.capture(native)
    owned = torch.ones(2, dtype=torch.bool)
    counts = torch.zeros(2, dtype=torch.long)
    for _ in range(26):
        physical = physical_ticks(2)
        wheel, base = torch.full((2, 4, 2), 20.), torch.zeros(2, 4)
        base[1] = 30.  # Bad true support despite an optimistic estimate.
        state = observe(task, state, physical, wheel, base, owned)
        estimate = completed(state.now_s)
        estimate.wheel_contact_probability[0] = .1  # Good physics cannot bypass this.
        result = task.finish_interval(state, now_s=task.boundary_time(state), estimate=estimate,
            policy_owned=owned, failed=~owned)
        state = result.state
        counts += result.verified_success.long()
    assert state.active.tolist() == [True, False]
    assert state.verified.tolist() == [True, False]
    assert counts.tolist() == [1, 0]
    # Diagnostic certification is not a manufactured reward/success payment.
    assert {field.name for field in fields(result)} == {'state', 'handoff', 'timeout', 'verified_success'}
    assert native._terrain_course.recovery.metrics()['success_reward_added'] is False


def test_timeout_is_separate_from_physical_failure_and_unowned_time():
    native = native_recovery(3)
    owner = native._terrain_course.recovery
    owner.policy_elapsed_s.fill_(4.98)
    native.time.fill_(4.98)
    state = RecoveryTaskState.capture(native)
    task = RecoveryTask(RecoveryTaskConfig.capture(native))
    owned = torch.tensor([True, True, False])
    failed = torch.tensor([False, True, False])
    now = task.boundary_time(state)
    expected = owner.finish_interval(now, None, policy_owned=owned, failed=failed)
    actual = task.finish_interval(state, now_s=now, estimate=None, policy_owned=owned, failed=failed)
    native.time = now
    same_state(actual.state, native)
    assert actual.timeout.tolist() == [True, False, False]
    assert actual.handoff.tolist() == [False, False, False]
    assert actual.state.active.tolist() == [False, False, True]
    torch.testing.assert_close(actual.timeout, expected.timeout)
    assert not actual.verified_success.any()
    assert failed.tolist() == [False, True, False]  # Timeout never mutates physical failure.


def test_stale_duplicate_and_current_estimates_match_native_without_early_handoff():
    native = native_recovery(4)
    owner = native._terrain_course.recovery
    owner.estimated_stable_s.fill_(.48)
    owner._last_estimate_s.fill_(1.)
    native.time.fill_(1.02)
    state = RecoveryTaskState.capture(native)
    task = RecoveryTask(RecoveryTaskConfig.capture(native))
    now = task.boundary_time(state)
    estimate = completed(torch.tensor([.9, 1., now[2], 1.02], dtype=torch.float64))
    owned = torch.ones(4, dtype=torch.bool)
    expected = owner.finish_interval(now, estimate, policy_owned=owned, failed=~owned)
    actual = task.finish_interval(state, now_s=now, estimate=estimate, policy_owned=owned, failed=~owned)
    native.time = now
    same_state(actual.state, native)
    assert actual.handoff.tolist() == [False, False, False, True]
    torch.testing.assert_close(actual.handoff, expected.handoff)


def test_physical_evidence_uses_com_velocity_base_normal_load_and_tick_ownership():
    native = native_recovery(6)
    native._terrain_course.recovery.physical_stable_s.fill_(.48)
    task = RecoveryTask(RecoveryTaskConfig.capture(native))
    state = RecoveryTaskState.capture(native)
    physical = physical_ticks(6, requires_grad=True)
    physical.angular_velocity[:2, :, 1] = -.3
    offset = torch.zeros(6, 3)
    offset[0, 0] = .6  # v_COM,z=.18 while origin v_z=0; must reject only row0.
    wheel, base = torch.full((6, 4, 2), 20.), torch.zeros(6, 4)
    base[2] = 3.
    wheel[3] = 0.
    owned = torch.ones(6, 4, dtype=torch.bool)
    owned[4, -1] = False
    ground = torch.zeros(6, 4)
    ground[5] = .1
    observe_native(native, physical, wheel, base, owned, com_offset=offset, ground=ground)
    result = observe(task, state, physical, wheel, base, owned, offset=offset, ground=ground)
    same_state(result, native)
    assert (result.clock[:, 3] > .49).tolist() == [False, True, False, False, False, False]
    assert not result.clock.requires_grad
    assert physical.position.requires_grad  # Shared continuous reward still owns its graph.


@pytest.mark.parametrize('managed', [True, False])
def test_handoff_reuses_native_tracking_command_and_preserves_other_rows(managed):
    native = native_recovery(5, managed=managed)
    tasks = native._terrain_course.tasks
    tasks.elapsed[:] = torch.arange(5) + 2.
    tasks.segment_clock[:] = torch.arange(5) + 1.
    tasks.segment_end_s[:] = tasks.elapsed + .2
    tasks.heading_enabled.fill_(True)
    tasks.heading_valid.fill_(True)
    tasks.stream.current[:, 3] = torch.linspace(.28, .36, 5)
    motion = MotionTask.decode(MotionTask.capture(native))
    original = motion.encode().clone()
    family, deadline = tasks.family.clone(), tasks.segment_end_s[:, None].clone()
    mask = torch.tensor([True, False, True, False, True])
    boundary = RecoveryBoundary(RecoveryTaskState.capture(native), mask, ~mask & False, ~mask & False)
    task = RecoveryTask(RecoveryTaskConfig.capture(native))
    uniform = torch.zeros(5, 2)
    if managed:
        rng = torch.Generator().set_state(tasks.generator.get_state())
        uniform[mask, 0] = torch.rand(int(mask.sum()), generator=rng)
        uniform[mask, 1] = torch.rand(int(mask.sum()), generator=rng)
    tasks.base_mode.masked_fill_(mask, int(CommandMode.NORMAL))
    tasks.begin_tracking(mask)
    expected = MotionTask.capture(native)
    actual = task.apply_handoff(boundary, motion, task_family=family, segment_end=deadline,
        handoff_uniform=uniform if managed else None)
    torch.testing.assert_close(actual.motion.encode(), expected, rtol=0, atol=0)
    torch.testing.assert_close(actual.task_family, tasks.family, rtol=0, atol=0)
    torch.testing.assert_close(actual.segment_end[:, 0], tasks.segment_end_s, rtol=0, atol=0)
    torch.testing.assert_close(motion.encode(), original, rtol=0, atol=0)
    torch.testing.assert_close(actual.motion.encode()[~mask], original[~mask], rtol=0, atol=0)
    torch.testing.assert_close(actual.motion.current, motion.current, rtol=0, atol=0)
    if managed:
        torch.testing.assert_close(actual.segment_end[mask, 0], tasks.elapsed[mask] + 1.7)
        assert (actual.task_family == int(TaskFamily.RECOVERY)).all()
        assert (actual.motion.start[mask, 0].abs() <= .6).all()
    else:
        assert (actual.task_family[mask] == int(TaskFamily.HEIGHT)).all()


def test_prepared_handoff_is_batch_independent_and_keeps_command_height_gradient():
    native = native_recovery(6)
    task = RecoveryTask(RecoveryTaskConfig.capture(native))
    state = RecoveryTaskState.capture(native)
    motion = MotionTask.decode(MotionTask.capture(native))
    current = motion.current.clone().requires_grad_(True)
    motion = replace(motion, current=current)
    uniform = torch.rand(6, 2, generator=torch.Generator().manual_seed(33))
    family = native._terrain_course.tasks.family.clone()
    deadline = native._terrain_course.tasks.segment_end_s[:, None].clone()
    native._terrain_course.recovery.estimated_stable_s.fill_(.48)
    state = RecoveryTaskState.capture(native)
    owned = torch.ones(6, dtype=torch.bool)
    estimate = completed(state.now_s)
    estimate.wheel_contact_probability[1] = 0.
    now = task.boundary_time(state)
    boundary = task.finish_interval(state, now_s=now, estimate=estimate, policy_owned=owned, failed=~owned)
    full = task.apply_handoff(boundary, motion, task_family=family, segment_end=deadline,
                             handoff_uniform=uniform)
    pieces = []
    for rows in (slice(0, 2), slice(2, 5), slice(5, 6)):
        local_state = RecoveryTaskState(state.flags[rows], state.clock[rows])
        local_estimate = replace(estimate, **{field.name: getattr(estimate, field.name)[rows]
                                             for field in fields(estimate)})
        local_boundary = task.finish_interval(local_state, now_s=now[rows], estimate=local_estimate,
            policy_owned=owned[rows], failed=(~owned)[rows])
        local_motion = replace(motion, **{field.name: getattr(motion, field.name)[rows]
                                         for field in fields(motion)})
        pieces.append(task.apply_handoff(local_boundary, local_motion, task_family=family[rows],
            segment_end=deadline[rows], handoff_uniform=uniform[rows]))
    torch.testing.assert_close(full.motion.encode(), torch.cat([part.motion.encode() for part in pieces]),
                               rtol=0, atol=0)
    torch.testing.assert_close(full.segment_end, torch.cat([part.segment_end for part in pieces]), rtol=0, atol=0)
    full.motion.start[boundary.handoff, 3].sum().backward()
    torch.testing.assert_close(current.grad[:, 3], boundary.handoff.float(), rtol=0, atol=0)
    assert current.grad[:, :3].count_nonzero() == 0
    assert uniform.grad is None


def test_critic_features_have_relative_clocks_and_no_nonfinite_initial_placeholder():
    native = native_recovery(2)
    native.time.fill_(100.)
    native._terrain_course.recovery._last_estimate_s[1] = 99.98
    state, config = RecoveryTaskState.capture(native), RecoveryTaskConfig.capture(native)
    feature = state.critic_features(config)
    assert feature.shape == (2, RECOVERY_FEATURE_DIM)
    assert feature.dtype == torch.float32 and torch.isfinite(feature).all()
    shifted = state.clock.clone()
    shifted[:, [0, 4]] += 4096.
    torch.testing.assert_close(replace(state, clock=shifted).critic_features(config), feature,
                               rtol=0, atol=1e-9)


def test_invalid_record_clock_and_unprepared_handoff_are_not_silently_invented():
    native = native_recovery(1)
    task = RecoveryTask(RecoveryTaskConfig.capture(native))
    state = RecoveryTaskState.capture(native)
    with pytest.raises(ValueError, match='recovery_clock'):
        replace(state, clock=state.clock.float())._restore(task.config)
    mask = torch.ones(1, dtype=torch.bool)
    boundary = RecoveryBoundary(state, mask, ~mask, ~mask)
    motion = MotionTask.decode(MotionTask.capture(native))
    for uniform in (None, torch.tensor([[float('nan'), .5]]), torch.tensor([[1., .5]])):
        with pytest.raises(ValueError, match='prepared'):
            task.apply_handoff(boundary, motion, task_family=native._terrain_course.tasks.family,
                segment_end=native._terrain_course.tasks.segment_end_s[:, None], handoff_uniform=uniform)
