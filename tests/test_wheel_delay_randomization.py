"""Episode aggregate delay preserves time integration and existing DR streams."""
from pathlib import Path

import pytest
import torch

from supermimic_rl.contracts import Field, StateSchema
from supermimic_rl.course.factory import create_course
from supermimic_rl.course.identified_execution import IdentifiedExecution
from supermimic_rl.course.randomization import CourseRandomization, RequestedRandomizationConfig
from supermimic_rl.course.wheel_delay import WheelDelayRandomization
from supermimic_rl.world_model_v2.layout import SnapshotLayout


ROOT = Path(__file__).resolve().parents[1]


def execution(count=8, dt=.005, *, enabled=True, dtype=torch.float64):
    return IdentifiedExecution(count, 'cpu', dt,
        asset_directory=ROOT / 'assets/full_vehicle_v2p0p1_source_audited_v2',
        bundle_directory=ROOT / 'configs/identified/20261031',
        friction_mode='native_floor', dtype=dtype, wheel_delay_randomization=enabled)


def owner(count=8, seed=31, config=None):
    return WheelDelayRandomization(count, seed, config=config, execution=execution(count))


@pytest.mark.parametrize('dt', [.001, .0025, .005, .01])
def test_varying_delay_integrates_shifted_command_intervals_not_rounded_ticks(dt):
    model = execution(4, dt)
    delay = torch.tensor([[.003, .006], [.005, .005], [.0055, .0035], [.0042, .0058]],
                         dtype=torch.float64)
    model.set_wheel_aggregate_delay(torch.arange(4), delay)
    generator = torch.Generator().manual_seed(19)
    commands = torch.rand(31, 4, 2, generator=generator, dtype=torch.float64) * 4 - 2
    commands[0] = 1.
    zero = torch.zeros(4, 6, dtype=torch.float64)
    actual = []
    for wheel in commands:
        command = zero.clone()
        command[:, 4:] = wheel
        actual.append(model.step(command, zero).motor_effort_nm[:, 4:])
    # Independent continuous-time overlap integral of each held command.
    expected = torch.zeros_like(commands)
    for target in range(len(commands)):
        for source in range(len(commands)):
            left = torch.maximum(torch.full_like(delay, target * dt), source * dt + delay)
            right = torch.minimum(torch.full_like(delay, (target + 1) * dt), (source + 1) * dt + delay)
            expected[target] += commands[source] * (right - left).clamp_min(0) / dt
    torch.testing.assert_close(torch.stack(actual), expected, rtol=0, atol=3e-14)


def test_partial_reset_changes_only_selected_delay_and_clears_its_pending_commands():
    model = execution(3)
    zero = torch.zeros(3, 6, dtype=torch.float64)
    model.step(zero + 1., zero)
    previous_history = model.history.clone()
    previous_delay = model.wheel_aggregate_delay_s.clone()
    model.set_wheel_aggregate_delay(torch.tensor([1]), torch.full((1, 2), .006, dtype=torch.float64))
    assert torch.equal(model.history[:, [0, 2]], previous_history[:, [0, 2]])
    assert not model.history[:, 1].any()
    assert torch.equal(model.wheel_aggregate_delay_s[[0, 2]], previous_delay[[0, 2]])
    output = model.step(zero, zero).motor_effort_nm[:, 4:]
    assert not output[1].any()
    torch.testing.assert_close(output[[0, 2]], torch.tensor([[.9, .7], [.9, .7]], dtype=torch.float64))


def test_disabled_motor_clears_history_without_redrawing_delay_or_disabling_passive_loss():
    randomization = owner(3)
    randomization.reset(torch.ones(3, dtype=torch.bool), 5000)
    model = randomization.execution
    delay, rng = model.wheel_aggregate_delay_s.clone(), randomization.delay_generator.get_state()
    command = torch.ones(3, 6, dtype=torch.float64)
    for _ in range(5):
        model.step(command, command)
    output = model.step(command, command, motor_enabled=torch.tensor([True, False, True]))
    assert not output.motor_effort_nm[1].any()
    assert not model.history[:, 1].any()
    assert output.passive_effort_nm[1].abs().sum() > 0
    resumed = model.step(torch.zeros_like(command), command).motor_effort_nm
    assert not resumed[1].any()
    assert torch.equal(delay, model.wheel_aggregate_delay_s)
    assert torch.equal(rng, randomization.delay_generator.get_state())


@pytest.mark.parametrize('coordinate, difficulty', [(0, 0.), (2500, .5), (5000, 1.), (9000, 1.)])
def test_shared_nominal_and_ramp_replace_aggregate_without_consuming_base_rng(coordinate, difficulty):
    config = RequestedRandomizationConfig(shared_encoder_bias=True)
    randomized = owner(1024, config=config)
    baseline = CourseRandomization(1024, seed=31, config=config)
    mask = torch.ones(1024, dtype=torch.bool)
    generator = torch.Generator().manual_seed(31 ^ WheelDelayRandomization.RNG_XOR)
    uniform = .003 + .003 * torch.rand(1024, 2, generator=generator, dtype=torch.float64)
    randomized.reset(mask, coordinate)
    baseline.reset(mask, coordinate)
    expected = randomized.nominal_delay_s + difficulty * (~randomized.nominal[:, None]) * (
        uniform - randomized.nominal_delay_s)
    actual = randomized.execution.wheel_aggregate_delay_s
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert .20 < randomized.nominal.float().mean() < .30
    assert torch.equal(actual[randomized.nominal], randomized.nominal_delay_s.expand(
        int(randomized.nominal.sum()), 2))
    assert actual.min() >= .003 and actual.max() <= .006
    for key, value in baseline.context_parameters().items():
        assert torch.equal(randomized.context_parameters()[key], value), key
    assert torch.equal(randomized.generator.get_state(), baseline.generator.get_state())
    assert torch.equal(randomized.delay_generator.get_state(), generator.get_state())


def test_actual_context_is_live_owned_and_delay_is_held_until_selected_reset():
    randomized = owner(16)
    randomized.reset(torch.ones(16, dtype=torch.bool), 5000)
    context = randomized.context_parameters()
    old = context['wheel_aggregate_delay_s'].clone()
    model = randomized.execution
    zero = torch.zeros(16, 6, dtype=torch.float64)
    for _ in range(20):
        model.step(zero, zero)
    assert torch.equal(old, model.wheel_aggregate_delay_s)
    mask = torch.arange(16) % 2 == 0
    randomized.reset(mask, 5000)
    assert torch.equal(model.wheel_aggregate_delay_s[~mask], old[~mask])
    assert not torch.equal(model.wheel_aggregate_delay_s[mask], old[mask])
    assert torch.equal(context['wheel_aggregate_delay_s'], old)
    assert torch.equal(randomized.context_parameters()['wheel_aggregate_delay_s'], model.wheel_aggregate_delay_s)


def test_empty_reset_consumes_no_delay_rng_and_leaves_queue_and_parameters_unchanged():
    randomized = owner()
    model = randomized.execution
    model.history.fill_(1.)
    before = randomized.delay_generator.get_state().clone()
    delay = model.wheel_aggregate_delay_s.clone()
    randomized.reset(torch.zeros(8, dtype=torch.bool), 5000)
    assert torch.equal(before, randomized.delay_generator.get_state())
    assert torch.equal(delay, model.wheel_aggregate_delay_s)
    assert model.history.eq(1).all()
    assert randomized.delay_reset_rows_consumed == 0


def test_checkpoint_restores_both_rng_streams_for_cold_physics_reset():
    original = owner(16)
    original.reset(torch.ones(16, dtype=torch.bool), 2500)
    original.reset(torch.arange(16) % 3 == 0, 5000)
    checkpoint = original.state_dict()
    restored = owner(16)
    restored.restore_after_physical_reset(checkpoint)
    # Like native restore, physical episodes cold-reset; saved queue state is
    # not implied by this RNG component checkpoint.
    for mask in (torch.ones(16, dtype=torch.bool), torch.arange(16) % 2 == 0):
        original.reset(mask, 5000)
        restored.reset(mask, 5000)
        for key, value in original.context_parameters().items():
            assert torch.equal(value, restored.context_parameters()[key]), key
    assert original.delay_reset_rows_consumed == restored.delay_reset_rows_consumed
    assert original.reset_rows_consumed == restored.reset_rows_consumed
    with pytest.raises(ValueError, match='migration'):
        restored.restore_after_physical_reset(CourseRandomization(16, seed=31).state_dict())
    with pytest.raises(ValueError, match='identity'):
        CourseRandomization(16, seed=31).restore_after_physical_reset(checkpoint)


def test_disabled_default_has_no_extra_context_rng_or_receipt_fields():
    legacy = execution(enabled=False)
    randomization = CourseRandomization(8, seed=31)
    assert 'wheel_aggregate_delay_s' not in legacy.context_parameters()
    assert 'wheel_aggregate_delay_s' not in randomization.context_parameters()
    assert WheelDelayRandomization.STATE_KEY not in randomization.state_dict()
    assert 'wheel_delay_parameter_scope' not in legacy.receipt()
    assert legacy.history.shape == (3, 8, 2)
    with pytest.raises(ValueError, match='opt-in'):
        legacy.set_wheel_aggregate_delay(torch.arange(8), torch.full((8, 2), .004, dtype=torch.float64))
    enabled = owner()
    assert enabled.receipt()['bounds_s'] == [.003, .006]
    assert 'not tick jitter' in enabled.receipt()['scope']
    assert 'replaces aggregate delay' in enabled.execution.receipt()['wheel_delay_parameter_scope']


def test_new_context_schema_is_explicit_and_fixed_delay_world_model_rejects_it():
    baseline = StateSchema((Field('state', (1,)),), (Field('context', (3,)),))
    current = StateSchema(baseline.state_fields, baseline.context_fields + (
        Field('requested.wheel_aggregate_delay_s', (2,), units='s', frame='left,right wheels'),))
    assert current.context_dim == baseline.context_dim + 2
    assert current.fingerprint != baseline.fingerprint
    with pytest.raises(ValueError, match='fixed wheel delay'):
        SnapshotLayout(current)
    with pytest.raises(ValueError, match='recorded substep truth'):
        SnapshotLayout(baseline)


@pytest.mark.parametrize('value', ['true', 1, None])
def test_factory_rejects_nonboolean_opt_in_before_native_launch(value):
    with pytest.raises(ValueError, match='Wheel delay randomization must be a boolean'):
        create_course({'wheel_delay_randomization': value})


@pytest.mark.parametrize('value', [float('nan'), .0029, .0061])
def test_invalid_delay_does_not_mutate_valid_episode(value):
    model = execution(1)
    model.history.fill_(1.)
    previous = model.wheel_aggregate_delay_s.clone()
    with pytest.raises(ValueError, match='3–6 ms'):
        model.set_wheel_aggregate_delay(torch.tensor([0]), torch.full((1, 2), value, dtype=torch.float64))
    assert torch.equal(previous, model.wheel_aggregate_delay_s)
    assert model.history.eq(1).all()
