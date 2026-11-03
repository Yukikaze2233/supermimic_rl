"""Independent left/right 15.62 wheel response in output-shaft units.

Native PhysX owns added armature and baseline static/dynamic friction. This
component owns the residual speed-dependent passive loss, effective I/O delay,
and directional high-speed motoring envelope. It never scales torque by 15.62
again. The returned passive effort is separate from motor-current diagnostics.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

WHEEL_NAMES = ("L_joint3", "R_joint3")
WHEEL_IDS = (2, 5)


def configure_identified_wheels(mechanism, control, model_path, root, *, apply_aggregate_delay=True,
                               match_recorded_controller=True):
    """Derive opt-in runtime inputs, preserving all leg response parameters.

    This creates no training admission or policy. Existing independent design
    and evidence bindings must be versioned for the returned configuration;
    never reuse a September-wheel admission to certify this hardware.
    """
    from .actuator_response import JOINT_NAMES, VECTOR_FIELDS, SCHEMA, VERSION, validate_actuator_response
    from .control_contract import NEW_WHEEL_CONTROL_ID

    root = Path(root).resolve()
    path = Path(model_path)
    path = (path if path.is_absolute() else root / path).resolve()
    try:
        relative = str(path.relative_to(root))
    except ValueError as error:
        raise ValueError("Wheel model must be inside the source snapshot root") from error
    wheel_cfg = {"enabled": True, "model_path": relative,
                 "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                 "apply_aggregate_delay": apply_aggregate_delay}
    _, model = load_wheel_response(wheel_cfg, root)
    if control["actuators"]["wheel"]["gear_ratio"] != 15.62:
        raise ValueError("Cannot install new gearbox dynamics onto an old-ratio control contract")
    derived, prior = copy.deepcopy(mechanism), copy.deepcopy(control)
    old = validate_actuator_response(derived.get("actuator_response"))
    if not old["enabled"]:
        old = {"enabled": True, "schema": SCHEMA, "version": VERSION,
               "joint_names": list(JOINT_NAMES), "gain": 1., **{key: [0.] * 6 for key in VECTOR_FIELDS}}
    for key in VECTOR_FIELDS:
        for wheel in WHEEL_IDS:
            old[key][wheel] = 0.
    derived["actuator_response"] = old
    derived["identified_wheel_response"] = wheel_cfg
    if match_recorded_controller:
        prior["control_id"] = NEW_WHEEL_CONTROL_ID
        prior["actuators"]["wheel"]["kd"] = model["controller_velocity_p_nm_per_rad_s"]
        prior["actuators"]["wheel"]["scope"] = "new_gearbox_airborne_identified_candidate"
    return derived, prior


def load_wheel_response(config, root):
    if config is None or config == {"enabled": False}:
        return {"enabled": False}, None
    required = {"enabled", "model_path", "model_sha256", "apply_aggregate_delay"}
    if not isinstance(config, dict) or set(config) != required or type(config["enabled"]) is not bool:
        raise ValueError("Identified wheel response requires an explicit, hash-bound configuration")
    if type(config["apply_aggregate_delay"]) is not bool:
        raise ValueError("Wheel aggregate delay enablement must be boolean")
    path = Path(config["model_path"])
    if not path.is_absolute():
        path = Path(root) / path
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != config["model_sha256"]:
        raise ValueError("Identified wheel model hash mismatch")
    model = json.loads(content)
    validate_wheel_model(model)
    return copy.deepcopy(config), model


def validate_wheel_model(model):
    if (model.get("schema") != "identified-wheel-response-v1" or model.get("coordinate") != "wheel_output"
            or model.get("gear_ratio") != 15.62 or model.get("bus_voltage_v") != 24.
            or set(model.get("axes", {})) != set(WHEEL_NAMES)):
        raise ValueError("Expected the new 15.62, 24V wheel-output model with both independent axes")
    cap = model.get("software_torque_cap_nm")
    if type(cap) not in (float, int) or not math.isfinite(cap) or not 0 < cap <= model["current_conversion_peak_nm"]:
        raise ValueError("Wheel software torque cap must be positive and within the declared current conversion")
    for joint in WHEEL_NAMES:
        row = model["axes"][joint]
        scalar = ("effective_total_inertia_kg_m2", "cad_inertia_kg_m2", "additional_armature_kg_m2",
                  "aggregate_delay_s", "static_breakaway_nominal_nm")
        if any(type(row[k]) not in (float, int) or not math.isfinite(row[k]) or row[k] < 0 for k in scalar):
            raise ValueError("Wheel inertia, friction and delay must be finite and nonnegative")
        if row["effective_total_inertia_kg_m2"] <= 0 or not math.isclose(
                row["effective_total_inertia_kg_m2"], row["cad_inertia_kg_m2"] + row["additional_armature_kg_m2"], abs_tol=1e-12):
            raise ValueError("Total inertia must equal existing wheel CAD plus added armature, exactly once")
        x, y, bias = row["friction_speed_knots_rad_s"], row["moving_friction_nm"], row["direction_bias_nm"]
        if (len(x) < 2 or len(x) != len(y) or any(not math.isfinite(v) for v in [*x, *y, bias])
                or x[0] < 0 or any(a >= b for a, b in zip(x, x[1:])) or min(y) <= abs(bias)):
            raise ValueError("Wheel moving-friction curve must be ordered, finite and dissipative in both directions")
        if row["static_breakaway_nominal_nm"] < min(y) - abs(bias):
            raise ValueError("Static breakaway must cover the native moving-friction floor")
        envelope = row["high_speed_envelope"]
        for key in ("zero_torque_speed_rad_s", "rolloff_slope_nm_per_rad_s"):
            values = envelope[key]
            if len(values) != 2 or any(not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError("Wheel high-speed envelope requires positive and negative direction values")
    return model


class IdentifiedWheelResponse:
    def __init__(self, count, device, dt, config=None, *, root=None, dtype=None):
        import torch

        self.torch, self.dt = torch, float(dt)
        self.cfg, self.model = load_wheel_response(config, root or Path(__file__).resolve().parents[3])
        self.enabled = self.cfg["enabled"]
        if not self.enabled:
            return
        if type(count) is not int or count < 1 or not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("Wheel response requires positive environment count and physics dt")
        self.dtype = dtype or torch.float32
        self.device = torch.device(device)
        rows = [self.model["axes"][name] for name in WHEEL_NAMES]
        self.rows = rows
        self.knots = [torch.tensor(r["friction_speed_knots_rad_s"], device=device, dtype=self.dtype) for r in rows]
        self.losses = [torch.tensor(r["moving_friction_nm"], device=device, dtype=self.dtype) for r in rows]
        self.bias = torch.tensor([r["direction_bias_nm"] for r in rows], device=device, dtype=self.dtype)
        self.floor = torch.tensor([min(r["moving_friction_nm"]) - abs(r["direction_bias_nm"]) for r in rows], device=device, dtype=self.dtype)
        self.inertia = torch.tensor([r["effective_total_inertia_kg_m2"] for r in rows], device=device, dtype=self.dtype)
        self.zero_speed = torch.tensor([r["high_speed_envelope"]["zero_torque_speed_rad_s"] for r in rows], device=device, dtype=self.dtype)
        self.slope = torch.tensor([r["high_speed_envelope"]["rolloff_slope_nm_per_rad_s"] for r in rows], device=device, dtype=self.dtype)
        delays = [r["aggregate_delay_s"] / self.dt if self.cfg["apply_aggregate_delay"] else 0. for r in rows]
        self.whole_delay = [int(math.floor(v + 1e-10)) for v in delays]
        self.fractional_delay = [v - n for v, n in zip(delays, self.whole_delay)]
        self.history = torch.zeros(max(self.whole_delay) + 2, count, 2, device=device, dtype=self.dtype)
        self.head = 0
        self.passive_effort = torch.zeros(count, 2, device=device, dtype=self.dtype)
        self.drive_bound = torch.full_like(self.passive_effort, self.model["software_torque_cap_nm"])

    def native_parameters(self):
        if not self.enabled:
            return {}
        fields = {k: {} for k in ("armature", "frictionloss", "dynamic_friction", "damping")}
        for i, name in enumerate(WHEEL_NAMES):
            row = self.rows[i]
            fields["armature"][name] = row["additional_armature_kg_m2"]
            fields["frictionloss"][name] = row["static_breakaway_nominal_nm"]
            fields["dynamic_friction"][name] = min(row["moving_friction_nm"]) - abs(row["direction_bias_nm"])
            fields["damping"][name] = 0.
        return fields

    @property
    def native_effort_limit(self):
        # Residual passive effort is a separate physical contribution, not motor
        # current. Native explicit-actuator clipping must leave room for it.
        return self.model["software_torque_cap_nm"] + max(
            max(r["moving_friction_nm"]) + abs(r["direction_bias_nm"]) for r in self.rows)

    def _check(self, effort, velocity):
        if (effort.shape != (len(self.passive_effort), 6) or velocity.shape != effort.shape
                or effort.device != self.device or velocity.device != self.device
                or effort.dtype != self.dtype or velocity.dtype != self.dtype):
            raise ValueError("Wheel response requires matching N x 6 output-axis torque and actual velocity")

    def apply_torque(self, effort, actual_velocity):
        """Advance once per physics tick. Returns motor effort; no passive loss."""
        if not self.enabled:
            return effort
        self._check(effort, actual_velocity)
        torch = self.torch
        with torch.no_grad():
            result = effort.clone()
            limit = self.model["software_torque_cap_nm"]
            self.history[self.head] = effort[:, WHEEL_IDS].clamp(-limit, limit)
            delayed = torch.empty_like(self.passive_effort)
            for k in range(2):
                a = self.history[(self.head - self.whole_delay[k]) % len(self.history), :, k]
                b = self.history[(self.head - self.whole_delay[k] - 1) % len(self.history), :, k]
                delayed[:, k] = a * (1. - self.fractional_delay[k]) + b * self.fractional_delay[k]
            self.head = (self.head + 1) % len(self.history)
            w = actual_velocity[:, WHEEL_IDS]
            direction = (w < 0).long()
            capacity = (self.slope.gather(1, direction.T).T * (
                self.zero_speed.gather(1, direction.T).T - w.abs())).clamp(0., limit)
            # At high speed, a braking command retains its separate current cap.
            self.drive_bound.copy_(torch.where(w * delayed > 0, capacity, limit))
            result[:, WHEEL_IDS] = delayed.clamp(min=-self.drive_bound, max=self.drive_bound)
            magnitude = torch.empty_like(w)
            for k in range(2):
                x, y = self.knots[k], self.losses[k]
                speed = w[:, k].abs().clamp(x[0], x[-1])
                index = (torch.searchsorted(x, speed.contiguous(), right=True) - 1).clamp(0, len(x) - 2)
                alpha = (speed - x[index]) / (x[index + 1] - x[index])
                magnitude[:, k] = y[index] + alpha * (y[index + 1] - y[index])
            residual = (magnitude + w.sign() * self.bias - self.floor).clamp_min(0.)
            # An isolated passive term must not reverse a resting wheel by itself.
            residual = torch.minimum(residual, self.inertia * w.abs() / self.dt)
            self.passive_effort.copy_(-w.sign() * residual)
        return result

    def reset(self, ids=None):
        if not self.enabled:
            return
        if ids is None:
            self.history.zero_()
            self.passive_effort.zero_()
        else:
            self.history[:, ids] = 0.
            self.passive_effort[ids] = 0.

    def summary(self):
        if not self.enabled:
            return {"enabled": False}
        return {"enabled": True, "config": self.cfg, "coordinate": "wheel_output", "joint_names": list(WHEEL_NAMES),
                "native_parameters": self.native_parameters(), "physics_dt_s": self.dt,
                "passive_effort_separate_from_motor_current": True,
                "aggregate_delay_applied": self.cfg["apply_aggregate_delay"],
                "limitations": self.model["limitations"]}
