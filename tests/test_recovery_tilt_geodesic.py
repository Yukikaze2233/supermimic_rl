"""Recovery tilt is continuous through native Euler cuts without changing other terms."""
from copy import deepcopy
from dataclasses import replace
import math

import pytest
import torch

pytest.importorskip('transformer_rl.formal.tasks.reward')
from transformer_rl.formal.control.observation import ObservationAssembler
from transformer_rl.formal.lab_env.formal_reward import _roll_pitch
from transformer_rl.formal.tasks.recovery import RecoveryPoseCompiler
from supermimic_rl.course.reward import install_shared_reward
from supermimic_rl.reward_kernel import (
    RewardExtensions, RewardHistory, RewardParameters, physical_reward_densities,
)
from support.reward_kernel import begin, finish, sample, setup


def physics_at(composer, angles_deg, tick=1):
    angles = torch.tensor(angles_deg, dtype=torch.float64).deg2rad()
    quaternion = RecoveryPoseCompiler.quaternion(*angles.unbind(-1))
    gravity = ObservationAssembler._inverse_rotate(
        quaternion, torch.tensor([[0., 0., -1.]], dtype=torch.float64).expand(2, -1))
    return replace(sample(composer, tick), gravity_body=gravity, roll_pitch_rad=_roll_pitch(quaternion))


def densities(physics, reference, composer, *, enabled, quiet=0.):
    history = RewardHistory(composer._window['dq'], torch.zeros(2), torch.zeros(2), torch.ones(2))
    parameters = replace(RewardParameters.from_composer(composer),
        extensions=RewardExtensions(recovery_tilt_geodesic=enabled, recovery_quiet=quiet))
    return physical_reward_densities(physics, reference, history, parameters)


def test_default_receipt_and_cost_keep_native_identity_and_exact_expression():
    assert not RewardExtensions().active
    assert 'recovery_tilt_geodesic' not in RewardExtensions().receipt()
    extension = RewardExtensions(recovery_tilt_geodesic=True)
    assert extension.active and not extension.needs_jump_progress and not extension.needs_flight_geometry
    assert extension.receipt()['recovery_tilt_geodesic'] is True
    for invalid in (1, 'true', None):
        with pytest.raises(ValueError, match='boolean'):
            RewardExtensions(recovery_tilt_geodesic=invalid)
    composer, reference = setup('recovery')
    begin(composer, reference, torch.zeros(2, 6))
    physics = physics_at(composer, [[17., 90.1, 47.], [-60., 120., -32.]])
    reference = replace(reference, target_roll_pitch_rad=torch.tensor([[.2, -.3], [-.1, .4]]))
    actual = densities(physics, reference, composer, enabled=False)
    pose = physics.roll_pitch_rad - reference.target_roll_pitch_rad
    expected = 2 * (torch.hypot(torch.ones_like(pose), pose / .8) - 1).sum(-1)
    assert torch.equal(actual.cost['attitude'], expected)


@pytest.mark.parametrize('roll', [0., 30., -75.])
@pytest.mark.parametrize('direction', [-1, 1])
def test_native_pitch_branch_cut_has_continuous_geodesic_cost(roll, direction):
    composer, reference = setup('recovery')
    begin(composer, reference, torch.zeros(2, 6))
    physics = physics_at(composer, [[roll, direction * 89.9, 73.], [roll, direction * 90.1, 73.]])
    before = densities(physics, reference, composer, enabled=False)
    after = densities(physics, reference, composer, enabled=True)
    assert (before.cost['attitude'][1] - before.cost['attitude'][0]).abs() > 1.
    # d[2 h(theta/.8)]/dtheta is bounded by 2/.8; rotation differs by .2 degrees.
    assert (after.cost['attitude'][1] - after.cost['attitude'][0]).abs() <= 2.5 * math.radians(.2)


@pytest.mark.parametrize('roll,pitch', [(0., 89.9), (0., 90.1), (30., 105.), (90., 0.), (180., 0.)])
def test_cost_is_independent_of_absolute_world_yaw(roll, pitch):
    composer, reference = setup('recovery')
    begin(composer, reference, torch.zeros(2, 6))
    physics = physics_at(composer, [[roll, pitch, -157.], [roll, pitch, 73.]])
    actual = densities(physics, reference, composer, enabled=True).cost['attitude']
    torch.testing.assert_close(actual[0], actual[1], atol=1e-12, rtol=0)


@pytest.mark.parametrize('axis,angles', [(0, [0., 75.]), (0, [-90., 150.]), (1, [-45., 89.9])])
def test_single_axis_legacy_scale_is_retained_where_euler_has_no_cut(axis, angles):
    composer, reference = setup('recovery')
    begin(composer, reference, torch.zeros(2, 6))
    values = [[0., 0., 0.], [0., 0., 0.]]
    for row, angle in zip(values, angles):
        row[axis] = angle
    physics = physics_at(composer, values)
    old = densities(physics, reference, composer, enabled=False).cost['attitude']
    new = densities(physics, reference, composer, enabled=True).cost['attitude']
    torch.testing.assert_close(new, old, atol=1e-12, rtol=1e-12)


def test_combined_small_tilts_keep_the_original_upright_quadratic_scale():
    composer, reference = setup('recovery')
    begin(composer, reference, torch.zeros(2, 6))
    physics = physics_at(composer, [[.01, -.02, 0.], [-.02, .01, 79.]])
    old = densities(physics, reference, composer, enabled=False).cost['attitude']
    new = densities(physics, reference, composer, enabled=True).cost['attitude']
    torch.testing.assert_close(new, old, atol=1e-14, rtol=1e-6)


def test_nonzero_target_up_is_honored_without_changing_upright_or_quiet_rewards():
    composer, reference = setup('recovery')
    begin(composer, reference, torch.zeros(2, 6))
    reference = replace(reference, target_roll_pitch_rad=torch.tensor(
        [[30., 20.], [-40., 110.]], dtype=torch.float64).deg2rad())
    physics = physics_at(composer, [[30., 20., 73.], [-40., 110., -157.]])
    old = densities(physics, reference, composer, enabled=False, quiet=1.)
    new = densities(physics, reference, composer, enabled=True, quiet=1.)
    torch.testing.assert_close(new.cost['attitude'], torch.zeros(2, dtype=torch.float64), atol=1e-14, rtol=0)
    assert old.cost['attitude'][1] > 0  # Equivalent non-canonical Euler targets still specify the same up.
    assert new.components.keys() == old.components.keys()
    for name, expected in old.components.items():
        if name != 'cost.attitude':
            assert torch.equal(new.components[name], expected), name


@pytest.mark.parametrize('case', ['ordinary', 'terrain', 'gimbal', 'jump_takeoff', 'jump_flight',
                                 'jump_landing', 'air_flight', 'air_landing'])
def test_enabled_extension_preserves_every_other_mode_bitwise(case):
    composer, reference = setup(case)
    begin(composer, reference, torch.zeros(2, 6))
    physics = physics_at(composer, [[20., 89.9, 73.], [-20., 90.1, -157.]])
    before = densities(physics, reference, composer, enabled=False)
    after = densities(physics, reference, composer, enabled=True)
    for name, expected in before.components.items():
        assert torch.equal(after.components[name], expected), name
    assert torch.equal(after.invalid, before.invalid)


def test_upright_inverted_and_antipodal_nonzero_targets_have_finite_gradients():
    composer, reference = setup('recovery')
    begin(composer, reference, torch.zeros(2, 6))
    physics = sample(composer, 1)
    for angles in ([[0., 0.], [0., 0.]], [[.3, -.4], [-.2, .5]]):
        target = torch.tensor(angles, dtype=torch.float64, requires_grad=True)
        roll, pitch = target.detach().unbind(-1)
        up = torch.stack((-pitch.sin(), roll.sin() * pitch.cos(), roll.cos() * pitch.cos()), -1)
        gravity = (-up * torch.tensor([[1.], [-1.]])).requires_grad_()
        value = densities(replace(physics, gravity_body=gravity),
            replace(reference, target_roll_pitch_rad=target), composer, enabled=True).cost['attitude']
        gradients = torch.autograd.grad(value.sum(), (gravity, target))
        assert torch.isfinite(value).all()
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        torch.testing.assert_close(value, torch.tensor(
            [0., 2 * (math.hypot(1., math.pi / .8) - 1)], dtype=torch.float64), atol=1e-12, rtol=0)


def test_gradients_match_finite_differences_away_from_antipodes():
    composer, reference = setup('recovery')
    begin(composer, reference, torch.zeros(2, 6))
    physics = physics_at(composer, [[30., 90.1, 73.], [-45., 120., -157.]])
    gravity = physics.gravity_body.detach().requires_grad_()
    target = torch.tensor([[.2, -.1], [-.3, .4]], dtype=torch.float64, requires_grad=True)

    def objective(g, t):
        return densities(replace(physics, gravity_body=g),
            replace(reference, target_roll_pitch_rad=t), composer, enabled=True).cost['attitude'].sum()

    assert torch.autograd.gradcheck(objective, (gravity, target), eps=1e-6, atol=1e-6, rtol=1e-5)


def test_native_install_records_flag_and_integrates_four_unchanged_physics_ticks():
    composer, reference = setup('recovery')
    old = deepcopy(composer)
    receipt = install_shared_reward(composer, extensions=RewardExtensions(recovery_tilt_geodesic=True))
    assert receipt['extensions']['recovery_tilt_geodesic'] is True
    assert 'kernel_sha256' in receipt
    begin(composer, reference, torch.zeros(2, 6))
    begin(old, reference, torch.zeros(2, 6))
    integral = torch.zeros(2, dtype=torch.float64)
    for tick in range(1, 5):
        physics = physics_at(composer, [[0., 89.8 + .1 * tick, 73.], [0., -89.8 - .1 * tick, -157.]], tick)
        physics = replace(physics, gravity_body=physics.gravity_body.float(),
                          roll_pitch_rad=physics.roll_pitch_rad.float())
        result = densities(physics, reference, composer, enabled=True)
        integral += result.components['cost.attitude'].double() * .005
        composer.observe(physics, reference)
        old.observe(physics, reference)
    actual, baseline = finish(composer), finish(old)
    torch.testing.assert_close(actual.component_densities['cost.attitude'] * .02, integral, atol=1e-14, rtol=0)
    for name, expected in baseline.component_densities.items():
        if name != 'cost.attitude':
            torch.testing.assert_close(actual.component_densities[name], expected, atol=0, rtol=0)
    for name in ('fully_policy_owned', 'success_credited', 'physical_success_request_id',
                 'timestamp_s', 'episode_id', 'control_epoch'):
        assert torch.equal(getattr(actual, name), getattr(baseline, name))
