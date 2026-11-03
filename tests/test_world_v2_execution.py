"""Differentiable execution matches the actual capped PD and identified motor chain."""
from pathlib import Path

import pytest
import torch

from supermimic_rl.course.identified_execution import IdentifiedExecution
from supermimic_rl.world_model_v2.execution import LegExecution, WheelExecution

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'configs/identified/20261031'


def test_wheel_effort_matches_native_at_high_speed_in_both_directions():
    native = IdentifiedExecution(4, 'cpu', .005,
        asset_directory=ROOT / 'assets/full_vehicle_v2p0p1_source_audited_v2',
        bundle_directory=BUNDLE, friction_mode='software_dynamic')
    wheels = WheelExecution.from_bundle(BUNDLE, .005)
    history = torch.zeros(4, wheels.history_length, 2)
    generator = torch.Generator().manual_seed(37)
    for _ in range(32):
        speed = torch.randn(4, 6, generator=generator) * 100
        action = torch.randn(4, 6, generator=generator) * 4
        strength = torch.rand(4, 2, generator=generator) * .32 + .84
        command = wheels.motor_input(wheels.target(action), speed[:, 4:], strength)
        history = wheels.push(history, command)
        expected = native.step(torch.cat((torch.zeros(4, 4), command), -1), speed).motor_effort_nm[:, 4:]
        torch.testing.assert_close(wheels.applied(history, speed[:, 4:]), expected, rtol=0, atol=0)


def test_leg_targets_share_one_winding_and_match_native_identified_pd():
    upstream = pytest.importorskip('transformer_rl.formal.control.actuation')
    from supermimic_rl.course.randomization import install_identified_feedback

    native = upstream.OutputFeedbackController([0.] * 6, 2, pair_domain=None)
    install_identified_feedback(native)
    legs = LegExecution(native.nominal, native.stiffness, native.damping, native.effort_bounds)
    action = torch.zeros(2, 6, requires_grad=True)
    q = torch.tensor([[3., 3.4, -3., -3.4, 0., 0.], [.1, .2, .3, .4, 0., 0.]])
    velocity = torch.arange(12).reshape(2, 6).float() / 10
    strength = torch.tensor([[.84, 1.16, 1.1, .9, 1., 1.], [1.1, .9, .84, 1.16, 1., 1.]])
    clock = torch.zeros(2, dtype=torch.float64)
    targets = native.decode(action, q, clock)
    predicted = legs.target(action, q)
    torch.testing.assert_close(predicted, targets.admitted_leg_position_rad, rtol=0, atol=0)
    torch.testing.assert_close(predicted[0], torch.tensor([2 * torch.pi, 2 * torch.pi, -2 * torch.pi, -2 * torch.pi]))
    effort = legs.motor_effort(predicted, q, velocity, strength)
    expected = native.effort(targets, q, velocity, clock, strength=strength).capped_effort[:, :4]
    torch.testing.assert_close(effort, expected, rtol=0, atol=0)
    effort.sum().backward()
    assert torch.isfinite(action.grad).all() and action.grad[1, 0] > 0
