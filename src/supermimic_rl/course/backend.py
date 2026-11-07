"""Backend-specific construction of the native course: Kamino or PhysX TGS.

The course factory owns everything backend-independent (pins, asset, task,
identified execution, curriculum). Each strategy here owns only what differs:
physics build scope, native health monitors, execution/mechanics readers,
requested randomization support and the backend identity bound to schemas.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
import math


class PhysxCourseHealth:
    """Native PhysX log errors plus the upstream closed-loop telemetry.

    PhysX articulations expose no per-world solver residuals, so this monitor
    reports loop-anchor geometry only; it never claims solver convergence.
    """

    def __init__(self, log_health, loop_telemetry):
        self.health, self.loop = log_health, loop_telemetry
        self.closed = False
        enabled = bool(loop_telemetry is not None and loop_telemetry.enabled)
        self.receipt = {"backend": "physx", "solver_status": "not exposed by PhysX articulations",
                        "loop_telemetry_enabled": enabled}
        self._summary = {"policy_samples": 0, "loop_closure_peak_m": 0.}
        self.latest = {}

    def install_substep_hook(self, env):
        # The environment drives its own loop telemetry at every physics tick.
        return self

    def check(self):
        if self.closed:
            raise RuntimeError("PhysX course health monitor is closed")
        self.health.check()

    def sample_policy(self):
        self.check()
        self._summary["policy_samples"] += 1
        if not self.receipt["loop_telemetry_enabled"]:
            return {}
        import torch

        loop = self.loop
        values = torch.stack([v.to(torch.float64) for v in (
            loop._environment_peak.amax(), loop._lifetime_peak, loop._lifetime_invalid,
            loop._sum, loop._sum_sq, loop._count, loop._velocity_invalid)]).cpu().tolist()
        peak, lifetime, invalid, total, square, count, velocity_invalid = values
        self.latest = {"loop_closure_peak_m": peak, "loop_lifetime_peak_m": lifetime,
                       "loop_mean_error_m": total / count if count else None,
                       "loop_rms_error_m": math.sqrt(square / count) if count else None,
                       "loop_sample_count": count, "loop_nonfinite_observations": invalid,
                       "joint_velocity_nonfinite_observations": velocity_invalid}
        self._summary["loop_closure_peak_m"] = max(self._summary["loop_closure_peak_m"], lifetime)
        if invalid or velocity_invalid or not all(math.isfinite(v) for v in values):
            raise FloatingPointError("Nonfinite native loop diagnostics; see physics health latest")
        return dict(self.latest)

    @property
    def summary(self):
        return {**self._summary, "receipt": dict(self.receipt), "latest": dict(self.latest)}

    def close(self):
        if not self.closed:
            self.closed = True
            self.health.close()


class KaminoCourseBackend:
    name = "kamino"
    execution_backend = mechanics_backend = "newton"
    sdk_packages = ("isaaclab", "isaaclab_newton", "newton", "warp-lang", "torch")
    applies_requested_randomization = True

    def __init__(self, backend, configuration):
        from .kamino_solver import KaminoConstraintSettings

        self.backend = backend
        self.constraints = KaminoConstraintSettings.from_dict(configuration.get("kamino_constraints"))

    @property
    def identity(self):
        identity = asdict(self.backend)
        if self.constraints.identity():
            identity["kamino_constraints"] = self.constraints.identity()
        return identity

    def build_scope(self):
        """Both the launch and the environment configuration build Kamino physics."""
        from transformer_rl.formal.lab_env import newton_runtime
        from .kamino_solver import kamino_constraint_scope

        return kamino_constraint_scope(self.constraints, newton_runtime)

    @staticmethod
    def log_health(log):
        from transformer_rl.formal.lab_env.newton_runtime import NewtonPhysicsHealth
        return NewtonPhysicsHealth.for_native_process(log)

    @staticmethod
    def course_health(log_health, native):
        from isaaclab_newton.physics import NewtonManager
        from .health import CoursePhysicsHealth

        return CoursePhysicsHealth(log_health, NewtonManager,
                                   loop_telemetry=native._loop_telemetry).install_substep_hook(native)

    @staticmethod
    def environment_type(base, randomization, execution):
        from .randomization import make_randomized_environment
        return make_randomized_environment(base, randomization, execution, backend="newton")


class PhysxCourseBackend:
    name = "physx"
    execution_backend = mechanics_backend = "physx"
    sdk_packages = ("isaaclab", "isaaclab_physx", "torch")
    applies_requested_randomization = True

    def __init__(self, backend, physics):
        iterations = {kind: physics.get(f"solver_{kind}_iterations") for kind in ("position", "velocity")}
        if any(type(value) is not int or value < 1 for value in iterations.values()):
            raise ValueError("PhysX course runtime must state solver_position/velocity_iterations explicitly")
        self.backend, self.iterations = backend, iterations
        self.solver_type = physics.get("solver_type", "isaaclab_default")

    @property
    def identity(self):
        return {"name": "physx", "solver_type": self.solver_type,
                "position_iterations": self.iterations["position"],
                "velocity_iterations": self.iterations["velocity"],
                "contact_friction": "native PhysX combine with runtime wheel_material buckets"}

    @staticmethod
    def build_scope():
        return nullcontext()

    @staticmethod
    def log_health(log):
        from transformer_rl.formal.simulation.health import NativePhysicsHealth
        return NativePhysicsHealth.for_native_process(log)

    @staticmethod
    def course_health(log_health, native):
        return PhysxCourseHealth(log_health, native._loop_telemetry)

    @staticmethod
    def environment_type(base, randomization, execution):
        from .randomization import make_randomized_environment
        return make_randomized_environment(base, randomization, execution, backend="physx")


def select_course_backend(backend, runtime, configuration):
    friction_mode = configuration.get("identified_friction_mode", "software_dynamic")
    if friction_mode not in ("software_dynamic", "native_floor"):
        raise ValueError("identified_friction_mode must be software_dynamic or native_floor")
    if backend.name == "kamino":
        if friction_mode != "software_dynamic":
            raise ValueError("Kamino cannot express the identified independent static/dynamic friction pair")
        strategy = KaminoCourseBackend(backend, configuration)
    elif backend.name == "physx":
        if configuration.get("kamino_constraints") is not None:
            raise ValueError("kamino_constraints applies only to the Kamino backend")
        strategy = PhysxCourseBackend(backend, runtime["physics"])
    else:
        raise ValueError("Unknown course physics backend: " + backend.name)
    strategy.identified_friction_mode = friction_mode
    return strategy
