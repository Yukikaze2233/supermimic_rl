"""Native health gates capacity/finite failures, never lack of convergence."""
import json
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.course.health import CoursePhysicsHealth, constraint_capacity_receipt


class LoggerHealth:
    receipt = {"backend": "kamino"}

    def __init__(self):
        self.checks = 0
        self.closed = False

    def check(self):
        self.checks += 1

    def close(self):
        self.closed = True


def fixture(*, internal_detector=False, loops=True):
    data = SimpleNamespace(_contact_overflow_warning_emitted=torch.zeros(1, dtype=torch.int32))
    contacts = SimpleNamespace(world_max_contacts_host=[128, 128], model_max_contacts_host=256,
        world_active_contacts=torch.tensor([20, 40]), model_active_contacts=torch.tensor([60]), data=data)
    packet = SimpleNamespace(rigid_contact_max=512, rigid_contact_count=torch.tensor([80]))
    solver = SimpleNamespace(_contacts_kamino=contacts,
        _config=SimpleNamespace(use_collision_detector=internal_detector),
        _model_kamino=SimpleNamespace(info=SimpleNamespace(
            num_joint_bilateral_cts=torch.tensor([100, 100], dtype=torch.int32),
            num_joint_bounded_cts=torch.tensor([5, 5], dtype=torch.int32),
            max_total_cts=torch.tensor([509, 509], dtype=torch.int32))),
        _solver_kamino=SimpleNamespace(_limits=SimpleNamespace(
            world_max_limits_host=[20, 20], model_max_limits_host=40)))
    manager = SimpleNamespace(_solver=solver, get_contacts=lambda: packet)
    status = torch.tensor([[100., .01, .1, .001, 0.], [120., .02, .2, .002, 0.]])
    loop = SimpleNamespace(enabled=loops, _environment_peak=torch.tensor([.002, .003]),
        _lifetime_peak=torch.tensor(.004), _lifetime_invalid=torch.tensor(0),
        _sum=torch.tensor(.006), _sum_sq=torch.tensor(.00002), _count=torch.tensor(2),
        _velocity_invalid=torch.tensor(0))
    wrapped = LoggerHealth()
    health = CoursePhysicsHealth(wrapped, manager, loop_telemetry=loop,
                                status_reader=lambda: status, native_to_tensor=lambda x: x)
    return health, wrapped, contacts, packet, status, loop


def test_nonconvergence_and_large_finite_residuals_are_recorded_without_failure():
    health, logger, _, _, status, _ = fixture()
    status[0, 1] = 1e12
    health.check()
    health.check()
    assert health.summary["policy_samples"] == 0
    result = health.sample_policy()
    assert logger.checks == 3
    assert result["converged_fraction"] == 0.
    assert result["requested_contact_count"] == 80
    assert result["requested_contacts_per_world"] is None
    assert result["kamino_retained_count"] == 60
    assert result["world_capacity_utilization_max"] == 40 / 128
    assert result["loop_closure_peak_m"] == pytest.approx(.003)
    assert result["loop_mean_error_m"] == pytest.approx(.003)
    assert health.summary["max_r_p"] > 1e11
    json.dumps(health.summary, allow_nan=False)
    health.close()
    assert logger.closed
    with pytest.raises(RuntimeError, match="closed"):
        health.sample_policy()


@pytest.mark.parametrize("cause", ["latched_flag", "world_count", "packet_count"])
def test_overflow_records_diagnostics_then_fails(cause):
    health, _, contacts, packet, _, _ = fixture()
    if cause == "latched_flag":
        contacts.data._contact_overflow_warning_emitted.fill_(1)
    elif cause == "world_count":
        contacts.world_active_contacts[0] = 129
    else:
        packet.rigid_contact_count.fill_(513)
    with pytest.raises(RuntimeError, match="capacity overflow"):
        health.sample_policy()
    assert health.latest["contact_overflow"] == 1
    assert health.summary["overflow_observed"]
    json.dumps(health.summary, allow_nan=False)


@pytest.mark.parametrize("cause", ["status", "loop", "velocity"])
def test_nonfinite_values_or_upstream_latched_nonfinite_observations_fail(cause):
    health, _, _, _, status, loop = fixture()
    if cause == "status":
        status[0, 1] = float("nan")
    elif cause == "loop":
        loop._lifetime_invalid.fill_(1)
    else:
        loop._velocity_invalid.fill_(1)
    with pytest.raises(FloatingPointError, match="Nonfinite native"):
        health.sample_policy()
    assert health.latest["nonfinite_observed"]
    json.dumps(health.summary, allow_nan=False)


def test_internal_detector_does_not_mislabel_output_count_as_requested():
    health, _, _, _, _, _ = fixture(internal_detector=True, loops=False)
    result = health.sample_policy()
    assert result["requested_contact_count"] is None
    assert "output" in result["requested_contact_count_scope"]
    assert result["loop_telemetry_enabled"] is False
    assert "loop_closure_peak_m" not in result


def test_policy_samples_accumulate_counts_without_clearing_native_counters():
    health, _, contacts, _, status, _ = fixture()
    health.sample_policy()
    status[:, 4] = 1
    health.sample_policy()
    assert health.summary["policy_samples"] == 2
    assert health.summary["converged_fraction"] == .5
    assert contacts.world_active_contacts.tolist() == [20, 40]


def test_substep_hook_keeps_early_overflow_and_peak_when_last_native_flag_is_clear(monkeypatch):
    health, _, contacts, packet, _, _ = fixture()

    class Environment:
        _physical_tick = 0
        observed = 0

        def _observe_substep(self):
            self.observed += 1

    env = Environment()
    original = env._observe_substep
    health.install_substep_hook(env)
    assert health.receipt["substep_hook_installed"]
    contacts.world_active_contacts[:] = torch.tensor([128, 90])
    contacts.model_active_contacts.fill_(218)
    contacts.data._contact_overflow_warning_emitted.fill_(1)
    packet.rigid_contact_count.fill_(240)
    # Device observation must not perform any host extraction.
    with monkeypatch.context() as patch:
        for name in ("cpu", "item", "tolist"):
            patch.setattr(torch.Tensor, name, lambda *args, **kwargs: pytest.fail("host read in substep"))
        env._physical_tick = 1
        env._observe_substep()
        env._observe_substep()  # Same native tick is not sampled twice.
        contacts.world_active_contacts.fill_(2)
        contacts.model_active_contacts.fill_(4)
        contacts.data._contact_overflow_warning_emitted.zero_()
        packet.rigid_contact_count.fill_(5)
        env._physical_tick = 2
        env._observe_substep()
    assert env.observed == 3
    with pytest.raises(RuntimeError, match="capacity overflow"):
        health.sample_policy()
    assert health.latest["world_retained_max"] == 2
    assert health.latest["observed_substep_world_retained_peak"] == 128
    assert health.latest["observed_substep_model_retained_peak"] == 218
    assert health.latest["observed_substep_newton_packet_peak"] == 240
    assert health.latest["physics_substeps_observed"] == 2
    assert health.summary["max_world_capacity_utilization"] == 1
    health.close()
    assert env._observe_substep == original
    assert "_observe_substep" not in vars(env)
    assert not hasattr(env, "_supermimic_health_observer")


def test_hook_observes_before_original_failure_and_restores_existing_override():
    health, _, contacts, _, _, _ = fixture()
    contacts.world_active_contacts[0] = 110
    original = lambda: (_ for _ in ()).throw(ValueError("original failure"))
    env = SimpleNamespace(_observe_substep=original, _physical_tick=1)
    health.install_substep_hook(env)
    with pytest.raises(ValueError, match="original failure"):
        env._observe_substep()
    assert health.sample_policy()["observed_substep_world_retained_peak"] == 110
    health.close()
    assert env._observe_substep is original


def test_native_constraint_capacity_uses_actual_components_and_unbounded_python_sum():
    n = 4096
    solver = SimpleNamespace(
        _contacts_kamino=SimpleNamespace(world_max_contacts_host=[256] * n),
        _model_kamino=SimpleNamespace(info=SimpleNamespace(
            num_joint_bilateral_cts=torch.full((n,), 100, dtype=torch.int32),
            num_joint_bounded_cts=torch.full((n,), 5, dtype=torch.int32),
            max_total_cts=torch.full((n,), 893, dtype=torch.int32))),
        _solver_kamino=SimpleNamespace(_limits=SimpleNamespace(
            world_max_limits_host=[20] * n, model_max_limits_host=20 * n)))
    result = constraint_capacity_receipt(solver, lambda x: x)
    assert result["component_histogram"] == [{"values": [100, 5, 20, 256], "worlds": n}]
    assert result["dense_delassus_float32_elements"] == 3266351104
    assert result["signed_int32_element_offsets_safe"] is False
    assert result["native_max_total_cts_matches_components"] is True


def test_actual_41e_deferred_invariants_keep_early_tick_overflow_until_policy_check():
    upstream = Path(os.environ.get("SUPERMIMIC_UPSTREAM_ROOT", str(
        Path(__file__).resolve().parents[2] / "transformer_rl-worktrees/full-vehicle-gimbal")))
    path = upstream / "src/transformer_rl/formal/control/invariants.py"
    if not path.is_file():
        pytest.skip("Pinned upstream source is unavailable")
    spec = importlib.util.spec_from_file_location("course_health_upstream_invariants", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    invariants = module.DeferredInvariants("cpu")
    module.require(invariants, "Kamino contact capacity overflow", torch.tensor([True]))
    module.require(invariants, "Kamino contact capacity overflow", torch.tensor([False]))
    with pytest.raises(ValueError, match="contact capacity overflow"):
        invariants.raise_if_violated()
    invariants.raise_if_violated()
