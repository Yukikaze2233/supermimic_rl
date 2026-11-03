"""Real confirmation times, contact exceptions and native pose/contact parity."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from supermimic_rl.world_model_v2.termination import PoseContactLimits, PoseContactTermination

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = json.loads((ROOT / "configs/gpufree/runtime-physx-tgs12-6.json").read_text())
LIMITS = PoseContactLimits.from_runtime(RUNTIME)
DT = RUNTIME["sensors"]["physics_dt_s"]


def samples(n=1, k=4, *, load=0., height=.3):
    return (torch.full((n, k, 1), height),
            torch.tensor([0., 0., -1.]).expand(n, k, 3).clone(),
            torch.full((n, k, 3), load))


def test_healthy_robot_has_no_invented_failure_probability():
    termination = PoseContactTermination(LIMITS, DT)
    memory = torch.zeros(2, 3)
    for _ in range(100):
        interval = termination.advance(memory, *samples(2), recovery=torch.zeros(2, dtype=torch.bool),
                                       body_contact_allowed=torch.zeros(2, dtype=torch.bool))
        assert not interval.failed.any()
        memory = interval.memory
    assert (memory == 0).all()


def test_contact_confirmed_at_native_tick_even_after_float32_serialization():
    termination = PoseContactTermination(LIMITS, DT)
    ticks = round(LIMITS.forbidden_contact_confirm_s / DT)
    memory = torch.zeros(1, 3)
    for tick in range(ticks):
        interval = termination.advance(memory, *samples(k=1, load=2.), recovery=torch.tensor([False]),
                                       body_contact_allowed=torch.tensor([False]))
        assert interval.failed.item() == (tick == ticks - 1)
        memory = interval.memory.clone()  # replay stores float32 between policy intervals
    assert interval.first_failure.item()
    interval = termination.advance(memory, *samples(), recovery=torch.tensor([True]),
                                   body_contact_allowed=torch.tensor([True]))
    assert interval.failed.item() and not interval.first_failure.item()


def test_break_in_contact_resets_confirmation_and_exception_only_covers_base():
    termination = PoseContactTermination(LIMITS, DT)
    memory = torch.tensor([[.395, 0., 0.], [.395, 0., 0.]])
    height, gravity, load = samples(2, k=1)
    load[0, 0, 0] = 3.  # Allowed base contact.
    load[1, 0, 1] = 3.  # Gimbal contact remains forbidden.
    interval = termination.advance(memory, height, gravity, load, recovery=torch.zeros(2, dtype=torch.bool),
                                   body_contact_allowed=torch.ones(2, dtype=torch.bool))
    assert interval.failed.tolist() == [False, True]
    assert interval.memory[0, 0] == 0


def test_full_sequence_matches_native_failure_latch():
    path = ROOT.parent / "transformer_rl-worktrees/full-vehicle-gimbal/src/transformer_rl/formal/tasks/termination.py"
    if not path.exists():
        pytest.skip("Native source checkout is required for parity")
    spec = importlib.util.spec_from_file_location("native_failure_parity", path)
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    config = dict(RUNTIME["termination_limits"])
    config["forbidden_body_names"] = tuple(config["forbidden_body_names"])
    n = 6
    reference = native.PhysicalFailureLatch(native.NativeTaskLimits(**config), n, physics_dt_s=DT)
    termination = PoseContactTermination(LIMITS, DT)
    memory = torch.zeros(n, 3)
    for tick in range(120):
        height, gravity, loads = samples(n, k=1)
        height[0] = .03  # Persistently collapsed.
        gravity[1] = torch.tensor([1., 0., 0.])  # Persistently tilted.
        loads[2:5, 0, 0] = 2.
        loads[5, 0, 1] = 2.
        if tick == 60:
            loads[2] = 0.  # Break consecutive duration.
        recovery = torch.tensor([False, False, False, False, True, False])
        allowed = torch.tensor([False, False, False, True, False, True])
        result = termination.advance(memory, height, gravity, loads, recovery=recovery,
                                     body_contact_allowed=allowed)
        expected = reference.observe(height_m=height[:, 0, 0], gravity_body=gravity[:, 0],
                                     forbidden_body_load_n=loads[:, 0], knee_signed_clearance_rad=torch.ones(n, 2),
                                     knee_pivot_residual_m=torch.zeros(n, 2), within_pressure_stroke=torch.ones(n, 2, dtype=torch.bool),
                                     recovery=recovery, body_contact_allowed=allowed)
        torch.testing.assert_close(result.failed, expected.true_failure)
        torch.testing.assert_close(result.memory[:, 0].double(), reference.contact_time_s, atol=1e-7, rtol=0)
        torch.testing.assert_close(result.memory[:, 1].double(), reference.pose_time_s, atol=1e-7, rtol=0)
        torch.testing.assert_close(result.memory[:, 2].long(), expected.failure_reasons)
        memory = result.memory
