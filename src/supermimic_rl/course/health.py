"""Policy-boundary Kamino diagnostics with one consolidated host transfer.

The pinned 41e terrain reader already latches contact overflow on the device at
every physical tick into DeferredInvariants. An instance-local substep hook also
retains contact high-water marks and overflow for the diagnostic receipt.
Convergence flags and finite residual magnitudes are diagnostic, never gates.
"""
from contextlib import nullcontext
from collections import Counter
from copy import deepcopy
import hashlib
import math
from types import MethodType
from typing import Any

import torch


def constraint_capacity_receipt(solver, native_to_tensor):
    """Read actual constraint components; sum dense sizes as Python integers.

    SDK get_max_constraints_per_world uses these same native fields. Its numpy
    int32 accumulation can overflow before allocation; this diagnostic does not
    repeat that arithmetic or assume a fixed number of robot constraints.
    """
    info = solver._model_kamino.info
    bilateral = native_to_tensor(info.num_joint_bilateral_cts)
    bounded = native_to_tensor(info.num_joint_bounded_cts)
    native_maximum = native_to_tensor(info.max_total_cts)
    count = len(solver._contacts_kamino.world_max_contacts_host)
    if any(value.shape != (count,) for value in (bilateral, bounded, native_maximum)):
        raise ValueError("Native constraint counts differ from the contact world layout")
    # Startup-only consolidated read. Policy sampling never re-reads static data.
    raw = torch.stack((bilateral, bounded, native_maximum), -1).to(torch.int64).cpu()
    rows = raw.tolist()
    limits = solver._solver_kamino._limits
    limit_capacity = (list(limits.world_max_limits_host)
                      if limits is not None and limits.model_max_limits_host > 0 else [0] * count)
    contacts = list(solver._contacts_kamino.world_max_contacts_host)
    if len(limit_capacity) != count:
        raise ValueError("Native joint-limit capacities differ from the contact world layout")
    components = [(int(row[0]), int(row[1]), int(limit_capacity[i]), int(contacts[i]))
                  for i, row in enumerate(rows)]
    dimensions = [a + b + c + 3 * d for a, b, c, d in components]
    elements = sum(dim * dim for dim in dimensions)
    histogram = Counter(components)
    return {
        "source": "SolverKamino._model_kamino.info and _solver_kamino._limits / _contacts_kamino",
        "component_order": ["joint_bilateral", "joint_bounded", "joint_limit_capacity", "contact_capacity"],
        "world_count": count,
        "component_histogram": [{"values": list(key), "worlds": value} for key, value in sorted(histogram.items())],
        "native_info_sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
        "dimension_formula": "bilateral + bounded + limits + 3 * contact_capacity",
        "max_constraint_dimension_range": [min(dimensions), max(dimensions)],
        "native_max_total_cts_matches_components": all(row[2] == dimensions[i] for i, row in enumerate(rows)),
        "dense_delassus_float32_elements": elements,
        "dense_delassus_bytes": elements * 4,
        "signed_int32_element_offsets_safe": elements <= 2**31 - 1,
        "allocation_scope": "dense Delassus matrix only; excludes solver workspaces and all other allocations",
    }


class _NativeStatusReader:
    """Copy named fields from the SDK Warp struct without a status.numpy read."""

    def __init__(self, solver):
        import warp as wp

        @wp.kernel
        def pack_status(status: wp.array(dtype=Any), output: wp.array2d(dtype=wp.float32)):
            i = wp.tid()
            output[i, 0] = wp.float32(status[i].iterations)
            output[i, 1] = status[i].r_p
            output[i, 2] = status[i].r_d
            output[i, 3] = status[i].r_c
            output[i, 4] = wp.float32(status[i].converged)

        self.solver, self.wp, self.kernel = solver, wp, pack_status
        self.output = wp.zeros((solver.status.shape[0], 5), dtype=wp.float32,
                               device=solver.status.device)
        self.tensor = wp.to_torch(self.output)

    def __call__(self):
        wp, device = self.wp, self.tensor.device
        stream = (wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(device)))
                  if device.type == "cuda" else nullcontext())
        with stream:
            wp.launch(self.kernel, dim=self.tensor.shape[0],
                      inputs=[self.solver.status, self.output], device=self.output.device)
        return self.tensor


class CoursePhysicsHealth:
    """Wrap logger health; explicitly sample once after each policy interval.

    ``check`` preserves the existing cheap logger check before/after stepping.
    The adapter calls ``sample_policy`` once after stepping, then may save
    ``summary``/``latest`` as JSON. No method changes physics or clears native
    counters. The optional reader/converter arguments support SDK-free tests.
    """

    def __init__(self, health, manager, *, loop_telemetry=None,
                 status_reader=None, native_to_tensor=None):
        self.health, self.manager, self.loop = health, manager, loop_telemetry
        self.closed = False
        self.latest = None
        self.solver = manager._solver
        self.contacts = self.solver._contacts_kamino
        if native_to_tensor is None:
            from transformer_rl.formal.lab_env.newton_properties import native_tensor
            native_to_tensor = native_tensor
        self._tensor = native_to_tensor
        self._status_reader = _NativeStatusReader(self.solver) if status_reader is None else status_reader
        self._world_capacity = tuple(int(v) for v in self.contacts.world_max_contacts_host)
        self.num_envs = len(self._world_capacity)
        self.model_capacity = int(self.contacts.model_max_contacts_host)
        if (not self.num_envs or min(self._world_capacity) <= 0
                or sum(self._world_capacity) != self.model_capacity):
            raise ValueError("Kamino contact capacities must be positive and consistent")
        self._external_contacts = not self.solver._config.use_collision_detector
        device = self._tensor(self.contacts.world_active_contacts).device
        self._world_peak = torch.zeros(self.num_envs, dtype=torch.int64, device=device)
        self._model_peak = torch.zeros((), dtype=torch.int64, device=device)
        self._packet_peak = torch.zeros((), dtype=torch.int64, device=device)
        self._overflow_seen = torch.zeros((), dtype=torch.bool, device=device)
        self._physical_samples = 0
        self._last_physical_tick = None
        self._hook_env = self._hook_method = None
        self.receipt = {
            **getattr(health, "receipt", {}),
            "diagnostics": "Kamino status, native contact counters, native six-loop anchor telemetry",
            "sampling_scope": "one final-substep snapshot per policy interval; not all-substep contact peaks",
            "host_read": "one packed tensor transfer in sample_policy; check only reads logger state",
            "contact_overflow_all_substeps": "41e NewtonTerrainContactReader -> DeferredInvariants",
            "kamino_counts": "retained after capacity clipping; per-world requested counts unavailable",
            "newton_count": ("input collision packet before speculative culling/Kamino capacity clipping"
                             if self._external_contacts else "Kamino-generated output, not requested contacts"),
            "solver_convergence_gate": False,
            "solver_residuals": {"r_p": "impulse N*s or N*m*s", "r_d": "velocity m/s or rad/s",
                                 "r_c": "complementarity J"},
            "loop_residual": "separate actual spherical-anchor Euclidean separation in metres",
            "loop_telemetry_enabled": bool(loop_telemetry is not None and loop_telemetry.enabled),
            "contact_capacity_per_world_range": [min(self._world_capacity), max(self._world_capacity)],
            "contact_capacity_total": self.model_capacity,
            "native_constraint_capacity": constraint_capacity_receipt(self.solver, self._tensor),
            "substep_hook_installed": False,
        }
        self._summary = {
            "policy_samples": 0, "solver_world_samples": 0, "converged_world_samples": 0,
            "max_newton_input_count": 0, "max_kamino_retained_count": 0,
            "max_world_retained_count": 0, "max_world_capacity_utilization": 0.,
            "overflow_observed": False, "max_iterations": 0.,
            "max_r_p": 0., "max_r_d": 0., "max_r_c": 0.,
        }

    def install_substep_hook(self, env):
        """Observe before the existing post-physics method, without replacing it."""
        if self.closed or self._hook_env is not None:
            raise RuntimeError("Substep health hook requires one live unbound observer")
        if getattr(env, "_supermimic_health_observer", None) is not None:
            raise RuntimeError("Native environment already has a substep health observer")
        original = env._observe_substep
        self._original_substep = original
        self._had_instance_substep = "_observe_substep" in vars(env)

        def observe_then_original(owner, *args, **kwargs):
            self.observe_physics(owner._physical_tick)
            return original(*args, **kwargs)

        self._hook_env = env
        self._hook_method = MethodType(observe_then_original, env)
        env._observe_substep = self._hook_method
        env._supermimic_health_observer = self
        self.receipt["substep_hook_installed"] = True
        self.receipt["sampling_scope"] = (
            "contact peaks/overflow at every returned physics substep after hook installation; "
            "solver status sampled at policy boundaries")
        return self

    def _contact_snapshot(self):
        count = self._tensor(self.contacts.world_active_contacts)
        model_count = self._tensor(self.contacts.model_active_contacts)
        overflow = self._tensor(self.contacts.data._contact_overflow_warning_emitted)
        packet = self.manager.get_contacts()
        if packet is None or int(packet.rigid_contact_max) <= 0:
            raise ValueError("Course health requires the native Newton contact packet")
        newton_count = self._tensor(packet.rigid_contact_count)
        if (count.shape != (self.num_envs,) or model_count.numel() != 1
                or overflow.numel() != 1 or newton_count.numel() != 1
                or any(v.device != self._world_peak.device for v in (count, model_count, overflow, newton_count))):
            raise ValueError("Native contact counters differ from the bound world/device layout")
        capacity = count.new_tensor(self._world_capacity)
        native_overflow = ((overflow != 0).any() | (count > capacity).any() | (count < 0).any()
                           | (model_count > self.model_capacity).any() | (model_count < 0).any()
                           | (newton_count > packet.rigid_contact_max).any() | (newton_count < 0).any())
        return count, model_count, newton_count, capacity, native_overflow, packet

    @torch.no_grad()
    def observe_physics(self, physical_tick=None):
        """GPU-only sticky counters; no tensor truth values or host transfers."""
        if self.closed:
            raise RuntimeError("Course physics health monitor is closed")
        if physical_tick is not None and physical_tick == self._last_physical_tick:
            return
        count, model_count, newton_count, _, overflow, _ = self._contact_snapshot()
        self._world_peak.copy_(torch.maximum(self._world_peak, count))
        self._model_peak.copy_(torch.maximum(self._model_peak, model_count.reshape(())))
        self._packet_peak.copy_(torch.maximum(self._packet_peak, newton_count.reshape(())))
        self._overflow_seen.logical_or_(overflow)
        self._physical_samples += 1
        self._last_physical_tick = physical_tick

    def check(self):
        if self.closed:
            raise RuntimeError("Course physics health monitor is closed")
        self.health.check()

    @property
    def summary(self):
        result = deepcopy(self._summary)
        count = result["solver_world_samples"]
        result["converged_fraction"] = result["converged_world_samples"] / count if count else None
        result["receipt"] = deepcopy(self.receipt)
        result["latest"] = deepcopy(self.latest)
        result["physics_substeps_observed"] = self._physical_samples
        return result

    @torch.no_grad()
    def sample_policy(self):
        if self.closed:
            raise RuntimeError("Course physics health monitor is closed")
        status = self._status_reader()
        if not isinstance(status, torch.Tensor) or status.shape != (self.num_envs, 5):
            raise ValueError("Kamino named status requires [num_envs,5]")
        count, model_count, newton_count, capacity, native_overflow, packet = self._contact_snapshot()
        if count.device != status.device:
            raise ValueError("Native solver status and contact devices differ")
        world_peak = torch.maximum(self._world_peak, count)
        pieces = {
            "nonfinite_solver_values": (~torch.isfinite(status)).sum(),
            "iterations_max": status[:, 0].amax(), "r_p_max": status[:, 1].amax(),
            "r_d_max": status[:, 2].amax(), "r_c_max": status[:, 3].amax(),
            "converged_worlds": (status[:, 4] != 0).sum(),
            "newton_packet_count": newton_count.reshape(()),
            "kamino_retained_count": model_count.reshape(()),
            "world_retained_min": count.amin(), "world_retained_max": count.amax(),
            "world_capacity_utilization_max": (count.to(torch.float64) / capacity).amax(),
            "worlds_at_capacity": (count == capacity).sum(),
            "contact_overflow": native_overflow | self._overflow_seen,
            "retained_world_sum": count.sum(),
            "observed_substep_world_retained_peak": world_peak.amax(),
            "observed_substep_model_retained_peak": torch.maximum(self._model_peak, model_count.reshape(())),
            "observed_substep_newton_packet_peak": torch.maximum(self._packet_peak, newton_count.reshape(())),
            "observed_substep_capacity_utilization_peak": (world_peak.to(torch.float64) / capacity).amax(),
        }
        loop_enabled = self.receipt["loop_telemetry_enabled"]
        if loop_enabled:
            # These are the upstream observer's device accumulators. They cover
            # every returned physics substep/reset, unlike the contact snapshot.
            pieces.update({
                "loop_closure_peak_m": self.loop._environment_peak.amax(),
                "loop_lifetime_peak_m": self.loop._lifetime_peak,
                "loop_nonfinite_observations": self.loop._lifetime_invalid,
                "loop_sum_m": self.loop._sum, "loop_sum_square_m2": self.loop._sum_sq,
                "loop_sample_count": self.loop._count,
                "joint_velocity_nonfinite_observations": self.loop._velocity_invalid,
            })
        packed = torch.stack([v.to(dtype=torch.float64) for v in pieces.values()])
        # This is the only device -> host transfer made by the observer.
        values = packed.detach().cpu().tolist()
        observed = dict(zip(pieces, values))
        bad = int(observed["nonfinite_solver_values"]) or any(not math.isfinite(v) for v in values)
        if loop_enabled:
            bad = bad or observed["loop_nonfinite_observations"] or observed[
                "joint_velocity_nonfinite_observations"]
        observed.update({
            "converged_fraction": observed["converged_worlds"] / self.num_envs,
            "newton_packet_capacity": int(packet.rigid_contact_max),
            "kamino_capacity_total": self.model_capacity,
            "requested_contact_count": int(observed["newton_packet_count"]) if self._external_contacts else None,
            "requested_contact_count_scope": self.receipt["newton_count"],
            "requested_contacts_per_world": None,
            "loop_telemetry_enabled": loop_enabled,
            "physics_substeps_observed": self._physical_samples,
        })
        if loop_enabled:
            n = observed["loop_sample_count"]
            observed["loop_mean_error_m"] = observed["loop_sum_m"] / n if n else None
            observed["loop_rms_error_m"] = math.sqrt(observed["loop_sum_square_m2"] / n) if n else None
        # JSON-safe receipts preserve a failure flag instead of writing NaN.
        observed = {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                    for k, v in observed.items()}
        observed["nonfinite_observed"] = bool(bad)
        self.latest = observed
        self._summary["policy_samples"] += 1
        self._summary["solver_world_samples"] += self.num_envs
        self._summary["converged_world_samples"] += int(observed["converged_worlds"])
        self._summary["overflow_observed"] |= bool(observed["contact_overflow"])
        mappings = {"max_newton_input_count": "observed_substep_newton_packet_peak",
                    "max_kamino_retained_count": "observed_substep_model_retained_peak",
                    "max_world_retained_count": "observed_substep_world_retained_peak",
                    "max_world_capacity_utilization": "observed_substep_capacity_utilization_peak",
                    **{"max_" + key: key + "_max" for key in ("iterations", "r_p", "r_d", "r_c")}}
        for target, source in mappings.items():
            if observed[source] is not None:
                self._summary[target] = max(self._summary[target], observed[source])
        if bad:
            raise FloatingPointError("Nonfinite native solver/loop diagnostics; see physics health latest")
        if observed["contact_overflow"]:
            raise RuntimeError("Native contact capacity overflow; see physics health latest")
        self.check()
        return deepcopy(self.latest)

    def close(self):
        if not self.closed:
            self.closed = True
            if self._hook_env is not None:
                if self._hook_env._observe_substep is self._hook_method:
                    if self._had_instance_substep:
                        self._hook_env._observe_substep = self._original_substep
                    else:
                        del self._hook_env._observe_substep
                if getattr(self._hook_env, "_supermimic_health_observer", None) is self:
                    del self._hook_env._supermimic_health_observer
                self._hook_env = self._hook_method = None
            self.health.close()
