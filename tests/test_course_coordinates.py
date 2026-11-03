"""Regression of the native tick-47 wrap, with sampling and reset guards."""
import math
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.course.coordinates import (
    NewtonContinuousJointCoordinates,
    install_newton_coordinates,
)


def reader(initial, names=("L_joint3",), continuous=("L_joint3",), dt=.005):
    return NewtonContinuousJointCoordinates(torch.tensor(initial, dtype=torch.float32), names, continuous, dt)


def test_actual_native_tick47_negative_wrap_and_mirror_positive_wrap():
    previous = -6.1233992577
    current = -.2125248909
    coordinates = reader([[previous], [-previous]])
    coordinates.read(torch.tensor([[previous], [-previous]]), torch.zeros(2, 1), 46)
    raw = torch.tensor([[current], [-current]])
    speed = torch.tensor([[-74.46058655], [74.46058655]])
    actual = coordinates.read(raw, speed, 47)
    expected = torch.tensor([[current - math.tau], [-current + math.tau]])
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
    assert coordinates._turns[:, 0].tolist() == [-1, 1]
    # The observed displacement agrees with the native velocity at this tick.
    torch.testing.assert_close(actual - torch.tensor([[previous], [-previous]]), speed * .005,
                               rtol=0, atol=1e-5)
    assert torch.equal(raw, torch.tensor([[current], [-current]]))


@pytest.mark.parametrize("sign", [-1., 1.])
def test_multiple_turns_are_unwrapped_without_changing_bounded_joint_coordinates(sign):
    coordinates = reader([[0., .3]], names=("wheel", "knee"), continuous=("wheel",))
    last = None
    for tick in range(160):
        physical = sign * tick * .25
        raw = torch.tensor([[math.fmod(physical, math.tau), .3 + tick / 100]])
        last = coordinates.read(raw, torch.tensor([[sign * 50., 2.]]), tick)
        assert last[0, 0] == pytest.approx(physical, abs=5e-6)
        assert last[0, 1] == raw[0, 1]
    assert abs(coordinates._turns.item()) >= 6
    assert last is coordinates.read(torch.full((1, 2), float("nan")), torch.zeros(1, 2), 159)


def test_reset_keeps_pre_reset_endpoint_and_other_environment_turns():
    coordinates = reader([[6.1], [-6.1]])
    old = coordinates.read(torch.tensor([[.1], [-.1]]), torch.tensor([[50.], [-50.]]), 1)
    frozen = old.clone()
    coordinates.reset_rows(torch.tensor([0]), torch.tensor([[0.]]))
    assert torch.equal(old, frozen)
    assert coordinates._turns[:, 0].tolist() == [0, -1]
    actual = coordinates.read(torch.tensor([[.2], [-.3]]), torch.tensor([[40.], [-40.]]), 2)
    torch.testing.assert_close(actual, torch.tensor([[.2], [-.3 - math.tau]]))


@pytest.mark.parametrize("speed", [math.pi / .005, -math.pi / .005, float("nan"), float("inf")])
def test_high_speed_or_nonfinite_values_remain_failures_without_advancing_state(speed):
    coordinates = reader([[0.]])
    with pytest.raises(RuntimeError, match="nonfinite|ambiguous"):
        coordinates.read(torch.tensor([[.2]]), torch.tensor([[speed]]), 0)
    assert coordinates._tick is None
    assert not coordinates._turns.any()
    assert coordinates._previous_raw.item() == 0.


def test_sample_gap_and_nonfinite_position_remain_failures():
    coordinates = reader([[0.]])
    coordinates.read(torch.zeros(1, 1), torch.zeros(1, 1), 7)
    with pytest.raises(RuntimeError, match="lost a physical sample"):
        coordinates.read(torch.tensor([[.1]]), torch.ones(1, 1), 9)
    with pytest.raises(RuntimeError, match="nonfinite"):
        coordinates.read(torch.tensor([[float("nan")]]), torch.ones(1, 1), 8)
    assert coordinates._tick == 7


@pytest.mark.parametrize("direction", [-1, 1])
def test_float32_half_turn_is_still_ambiguous_after_modulo_rounding(direction):
    coordinates = reader([[0.]])
    with pytest.raises(RuntimeError, match="position increment is ambiguous"):
        coordinates.read(torch.tensor([[direction * math.pi]]), torch.zeros(1, 1), 0)
    assert coordinates._tick is None and not coordinates._turns.any()


def test_installation_uses_actual_asset_names_and_retains_physx_reader():
    raw = torch.zeros(2, 3)
    old = SimpleNamespace(_turns=torch.zeros(2, 2, dtype=torch.int64))
    env = SimpleNamespace(
        _is_newton=True, cfg=SimpleNamespace(physics_backend_name="kamino",
            sim=SimpleNamespace(physics=SimpleNamespace(solver_cfg=SimpleNamespace(rotation_correction="twopi")))),
        _joint_coordinates=old, robot=SimpleNamespace(data=SimpleNamespace(joint_pos=raw)),
        _as_torch=lambda value: value, _binding=SimpleNamespace(native_joint_names=("knee", "wheel_r", "wheel_l")),
        _asset=SimpleNamespace(continuous_joint_names=lambda: ("wheel_l", "wheel_r")), physics_dt=.005,
    )
    receipt = install_newton_coordinates(env)
    assert receipt["native_period_rad"] == math.tau
    assert env._joint_coordinates.indices.tolist() == [2, 1]
    assert env._joint_coordinates.physics_dt_s == .005
    env._is_newton = False
    env._joint_coordinates = old
    assert install_newton_coordinates(env) is None
    assert env._joint_coordinates is old


def test_unknown_correction_mode_is_not_silently_wrapped():
    env = SimpleNamespace(_is_newton=True, cfg=SimpleNamespace(physics_backend_name="kamino",
        sim=SimpleNamespace(physics=SimpleNamespace(solver_cfg=SimpleNamespace(rotation_correction="none")))))
    with pytest.raises(ValueError, match="verified for Kamino"):
        install_newton_coordinates(env)
