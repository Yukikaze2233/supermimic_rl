"""Backend strategies isolate Kamino/PhysX differences from the course factory."""
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.course.backend import (
    KaminoCourseBackend, PhysxCourseBackend, PhysxCourseHealth, select_course_backend)


def backend(name):
    return SimpleNamespace(name=name, max_iterations=200, max_contacts_per_world=64, use_cuda_graph=False)


def test_selection_and_identity_bind_the_solver_settings():
    physics = {"solver_position_iterations": 12, "solver_velocity_iterations": 6}
    physx = select_course_backend(backend("physx"), {"physics": physics}, {})
    assert isinstance(physx, PhysxCourseBackend)
    assert physx.identity["position_iterations"] == 12 and physx.identity["velocity_iterations"] == 6
    assert physx.applies_requested_randomization and physx.execution_backend == "physx"
    with pytest.raises(ValueError, match="Kamino backend"):
        select_course_backend(backend("physx"), {"physics": physics}, {"kamino_constraints": {"joint_alpha": .3}})
    with pytest.raises(ValueError, match="explicitly"):
        select_course_backend(backend("physx"), {"physics": {}}, {})
    with pytest.raises(ValueError, match="Unknown"):
        select_course_backend(backend("mujoco"), {"physics": {}}, {})


def test_kamino_identity_adds_only_explicit_constraint_overrides(monkeypatch):
    import dataclasses

    @dataclasses.dataclass
    class Config:
        name: str = "kamino"
        max_iterations: int = 200

    plain = KaminoCourseBackend(Config(), {})
    tuned = KaminoCourseBackend(Config(), {"kamino_constraints": {"joint_alpha": .3}})
    assert "kamino_constraints" not in plain.identity
    assert tuned.identity["kamino_constraints"] == {"joint_alpha": .3}


def test_identified_friction_mode_is_opt_in_and_rejects_unsupported_backend():
    runtime = {"physics": {"solver_position_iterations": 64, "solver_velocity_iterations": 16}}
    legacy = select_course_backend(backend("physx"), runtime, {})
    native = select_course_backend(backend("physx"), runtime, {"identified_friction_mode": "native_floor"})
    assert legacy.identified_friction_mode == "software_dynamic"
    assert native.identified_friction_mode == "native_floor"
    assert legacy.identity == native.identity  # Execution receipt owns this difference.
    with pytest.raises(ValueError, match="Kamino cannot express"):
        select_course_backend(backend("kamino"), runtime, {"identified_friction_mode": "native_floor"})
    for value in (None, True, "static_only", {}):
        with pytest.raises(ValueError, match="identified_friction_mode"):
            select_course_backend(backend("physx"), runtime, {"identified_friction_mode": value})


class LogHealth:
    def __init__(self):
        self.closed = False

    def check(self):
        pass

    def close(self):
        self.closed = True


def telemetry(invalid=0):
    zero = torch.zeros(())
    return SimpleNamespace(enabled=True, _environment_peak=torch.tensor([.001, .003]),
                           _lifetime_peak=torch.tensor(.004), _lifetime_invalid=torch.tensor(invalid),
                           _sum=torch.tensor(.02), _sum_sq=torch.tensor(1e-4), _count=torch.tensor(10.),
                           _velocity_invalid=zero)


def test_physx_health_reports_loop_geometry_and_rejects_nonfinite():
    log = LogHealth()
    health = PhysxCourseHealth(log, telemetry())
    assert health.install_substep_hook(object()) is health
    latest = health.sample_policy()
    assert latest["loop_closure_peak_m"] == pytest.approx(.003)
    assert health.summary["policy_samples"] == 1 and health.summary["loop_closure_peak_m"] == pytest.approx(.004)
    assert "solver_status" in health.receipt
    health.close()
    assert log.closed
    with pytest.raises(FloatingPointError):
        PhysxCourseHealth(LogHealth(), telemetry(invalid=1)).sample_policy()
