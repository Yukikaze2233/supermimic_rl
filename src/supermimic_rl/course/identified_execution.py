"""Pinned output-shaft response from the running October 31 identification chain.

All six-axis tensors follow ACTION_ORDER below, in native joint coordinates.
The physics engine owns body inertia and the returned additional armature.
This module owns wheel aggregate delay exactly once and software dissipation.
Measured spring pressure is applied either at sliders or mounting points, never
both. Encoder holding compensation is deliberately not transferred: its source
Jacobian table belongs to a different model coordinate system.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import Tensor


ACTION_ORDER = ("L_joint1", "LL_joint1", "R_joint1", "RR_joint1", "L_joint3", "R_joint3")
SPRING_ORDER = ("L_spring_slide", "R_spring_slide")
MODEL_PINS = {
    "wheel_model.json": "01d52214771bb9dea2cb592eff4dc6fbd27bb096faa90ff1e17b5d898a1e877c",
    "leg_model.json": "90d299995f24ba38e6c1ed67e32dd40b9b2fa26e041f2f814db1e177ac040b52",
    "gas_spring_model.json": "698f67ccc7d6364b73f1b426ed3e6386ffc9e11912ef95a89d716eb7ee0e06b9",
}


@dataclass(frozen=True)
class ExecutionEffort:
    motor_effort_nm: Tensor
    passive_effort_nm: Tensor
    native_effort_nm: Tensor
    wheel_motor_bound_nm: Tensor


@dataclass(frozen=True)
class SpringForces:
    forces_world: Tensor
    torques_about_com_world: Tensor
    mount_positions_world: Tensor
    mount_distance_m: Tensor
    compression_m: Tensor
    pressure_force_n: Tensor
    valid: Tensor
    normalized_compression: Tensor
    within_mechanical_stroke: Tensor
    within_fitted_domain: Tensor


class IdentifiedExecution:
    """Stateful physical response, called once per physics tick after motor PD.

    ``friction_mode='native_floor'`` reproduces the source division of ownership:
    native static/dynamic friction and software residual moving friction.
    ``'software_dynamic'`` supplies the full moving friction in software and asks
    for zero native friction; it has NO static breakaway model and is not the same
    physical model at rest. Its use must be recorded by the backend integration.
    No first-order motor lag, USB delay or old response coefficient is added.
    """

    def __init__(self, count: int, device: str | torch.device, dt: float, *,
                 asset_directory: str | Path, bundle_directory: str | Path,
                 friction_mode: str, dtype: torch.dtype = torch.float32):
        if type(count) is not int or count < 1 or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Response requires positive batch size and physics dt")
        if friction_mode not in ("native_floor", "software_dynamic"):
            raise ValueError("Choose explicit native_floor or software_dynamic friction ownership")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Response requires float32 or float64")
        self.count, self.device, self.dtype, self.dt = count, torch.device(device), dtype, float(dt)
        self.invariants = None
        self.friction_mode = friction_mode
        root = Path(bundle_directory).resolve()
        source_manifest = json.loads((root / "source_manifest.json").read_text())
        self.source_manifest = source_manifest
        self.source_manifest_sha256 = self._digest(root / "source_manifest.json")
        if source_manifest.get("source_commit") != "444749daa37bba264bcfa014a641216018702d55":
            raise ValueError("Unexpected identification source commit")
        for name, pin in source_manifest["files"].items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or self._digest(path) != pin["sha256"]:
                raise ValueError(f"Identification source differs: {name}")
        for name, pin in MODEL_PINS.items():
            if self._digest(root / name) != pin:
                raise ValueError(f"Identification model differs: {name}")
        self.wheel_model = json.loads((root / "wheel_model.json").read_text())
        self.leg_model = json.loads((root / "leg_model.json").read_text())
        self.spring_model = json.loads((root / "gas_spring_model.json").read_text())
        leg_parameters = json.loads((root / "leg_parameters.json").read_text())
        spring_source = json.loads((root / "gas_spring_source.json").read_text())
        if (self._digest(root / "leg_parameters.json") != self.leg_model["source_parameter_sha256"]
                or self._digest(root / "gas_spring_source.json") != self.spring_model["source_curve_sha256"]
                or leg_parameters["right_symmetry_authorized"] is not True
                or self.spring_model["right_leg_symmetry_assumed"] is not True):
            raise ValueError("Identified source or symmetry authorization differs")
        asset = Path(asset_directory)
        self.spec = json.loads((asset / "model_spec.json").read_text())
        self.asset_spec_sha256 = self._digest(asset / "model_spec.json")
        self.asset_manifest_sha256 = self._digest(asset / "manifest.json")
        asset_manifest = json.loads((asset / "manifest.json").read_text())
        self.auxiliary_body_names = tuple(asset_manifest.get("auxiliary_body_names", ()))
        if asset_manifest["files_sha256"]["model_spec.json"] != self.asset_spec_sha256:
            raise ValueError("Selected asset specification differs from its manifest")
        self._joints = {row["name"]: row for row in self.spec["joints"]}
        self._bodies = {row["name"]: row for row in self.spec["bodies"]}
        if any(name not in self._joints for name in (*ACTION_ORDER, *SPRING_ORDER)):
            raise ValueError("Selected asset lacks the named active axes or spring sliders")
        rows = [self.wheel_model["axes"][name] for name in ACTION_ORDER[4:]]
        self.wheel_rows = rows
        self.inertia = self._tensor([r["effective_total_inertia_kg_m2"] for r in rows])
        self.wheel_inertia_scale = torch.ones(count, 2, device=self.device, dtype=dtype)
        cad_inertia = []
        for name in ACTION_ORDER[4:]:
            joint = self._joints[name]
            body = self._bodies[joint["child"]]
            axis = torch.tensor(joint["axis"], dtype=torch.float64)
            if joint["type"] != "continuous" or not torch.isclose(axis.norm(), axis.new_tensor(1.), atol=1e-8):
                raise ValueError("Wheel rebind requires a unit native continuous-joint axis")
            inertia = torch.tensor(body["inertia"], dtype=torch.float64)
            com = torch.tensor(body["com"], dtype=torch.float64)
            # Native body dynamics already includes the parallel-axis term.
            axial = axis @ inertia @ axis + body["mass"] * torch.linalg.cross(axis, com).square().sum()
            cad_inertia.append(float(axial))
        self.cad_inertia = self._tensor(cad_inertia)
        self.armature = self.inertia - self.cad_inertia
        self._cad_inertia_values = cad_inertia
        self._armature_values = [row["effective_total_inertia_kg_m2"] - cad
                                 for row, cad in zip(rows, cad_inertia)]
        if (self.armature < 0).any():
            raise ValueError("Selected wheel CAD inertia exceeds identified total inertia")
        self.knots = [self._tensor(r["friction_speed_knots_rad_s"]) for r in rows]
        self.losses = [self._tensor(r["moving_friction_nm"]) for r in rows]
        self.bias = self._tensor([r["direction_bias_nm"] for r in rows])
        self.floor = self._tensor([min(r["moving_friction_nm"]) - abs(r["direction_bias_nm"]) for r in rows])
        self.zero_speed = self._tensor([r["high_speed_envelope"]["zero_torque_speed_rad_s"] for r in rows])
        self.slope = self._tensor([r["high_speed_envelope"]["rolloff_slope_nm_per_rad_s"] for r in rows])
        delay_ticks = [r["aggregate_delay_s"] / self.dt for r in rows]
        self.whole_delay = [int(math.floor(value + 1e-10)) for value in delay_ticks]
        self.fractional_delay = [value - whole for value, whole in zip(delay_ticks, self.whole_delay)]
        self.history = torch.zeros(max(self.whole_delay) + 2, count, 2, device=self.device, dtype=dtype)
        self.head = 0
        self.passive_effort = torch.zeros(count, 6, device=self.device, dtype=dtype)
        self.drive_bound = torch.full((count, 2), self.wheel_model["software_torque_cap_nm"], device=self.device, dtype=dtype)
        self.leg_coulomb = self._tensor(self.leg_model["coulomb_nm"])
        self.leg_band = self.leg_model["numerical_regularization"]["velocity_rad_s"]
        self.spring_compression = self._tensor(self.spring_model["compression_knots_m"])
        self.spring_force = self._tensor(self.spring_model["force_knots_n"])
        endpoints = [value for row in self.spring_model["sides"].values()
                     for value in (row["compression_m"][0], row["compression_m"][-1])]
        self.geometry_compression_bounds = (min(endpoints), max(endpoints))
        bindings = [self.spec["spring_binding"][name] for name in SPRING_ORDER]
        for name, binding in zip(SPRING_ORDER, bindings):
            if (self._joints[name]["type"] != "prismatic"
                    or binding["compression_expression"] != "compression_at_q_zero_m - q_m"
                    or binding["effort_sign"] != 1
                    or not math.isclose(binding["full_extension_pin_distance_m"], spring_source["extended_pin_distance_m"], abs_tol=1e-10)
                    or not math.isclose(binding["stroke_m"], spring_source["stroke_m"], abs_tol=1e-10)):
                raise ValueError("Selected spring datum or direction differs from identified hardware")
        self.spring_zero = self._tensor([b["compression_at_q_zero_m"] for b in bindings])
        self.spring_full_length = self._tensor([b["full_extension_pin_distance_m"] for b in bindings])
        self.spring_stroke = self._tensor([b["stroke_m"] for b in bindings])

    @staticmethod
    def _digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _tensor(self, values) -> Tensor:
        return torch.tensor(values, device=self.device, dtype=self.dtype)

    def _check(self, tensor: Tensor, width: int, name: str) -> None:
        if (not isinstance(tensor, Tensor) or tensor.shape != (self.count, width)
                or tensor.dtype != self.dtype or tensor.device != self.device):
            raise ValueError(f"{name} requires matching batch, device and dtype")
        self._require(f"Nonfinite {name}", ~torch.isfinite(tensor))

    def _require(self, label: str, violated: Tensor) -> None:
        if self.invariants is not None:
            self.invariants.require(label, violated)
        elif violated.any():
            raise ValueError(label)

    @staticmethod
    def _interpolate(knots: Tensor, values: Tensor, x: Tensor) -> Tensor:
        bounded = x.clamp(knots[0], knots[-1])
        index = (torch.searchsorted(knots, bounded.contiguous(), right=True) - 1).clamp(0, len(knots) - 2)
        fraction = (bounded - knots[index]) / (knots[index + 1] - knots[index])
        return values[index] + fraction * (values[index + 1] - values[index])

    @torch.no_grad()
    def step(self, motor_effort_nm: Tensor, actual_velocity_rad_s: Tensor, *,
             motor_enabled: Tensor | None = None) -> ExecutionEffort:
        """Apply delay/envelope to capped PD; passive forces survive motor disable."""
        self._check(motor_effort_nm, 6, "motor effort")
        self._check(actual_velocity_rad_s, 6, "actual velocity")
        if motor_enabled is not None:
            if (motor_enabled.shape != (self.count,) or motor_enabled.dtype != torch.bool
                    or motor_enabled.device != self.device):
                raise ValueError("motor_enabled requires a device-local bool per environment")
            self.reset(~motor_enabled)
        result = motor_effort_nm.clone()
        result[:, :4].clamp_(-40., 40.)
        limit = self.wheel_model["software_torque_cap_nm"]
        wheel_input = result[:, 4:].clamp(-limit, limit)
        if motor_enabled is not None:
            wheel_input = torch.where(motor_enabled[:, None], wheel_input, 0.)
        self.history[self.head].copy_(wheel_input)
        delayed = torch.empty_like(wheel_input)
        for k in range(2):
            a = self.history[(self.head - self.whole_delay[k]) % len(self.history), :, k]
            b = self.history[(self.head - self.whole_delay[k] - 1) % len(self.history), :, k]
            delayed[:, k] = a * (1. - self.fractional_delay[k]) + b * self.fractional_delay[k]
        self.head = (self.head + 1) % len(self.history)
        speed = actual_velocity_rad_s[:, 4:]
        direction = (speed < 0).long()
        capacity = (self.slope.gather(1, direction.T).T * (
            self.zero_speed.gather(1, direction.T).T - speed.abs())).clamp(0., limit)
        self.drive_bound.copy_(torch.where(speed * delayed > 0, capacity, limit))
        result[:, 4:] = delayed.clamp(min=-self.drive_bound, max=self.drive_bound)
        if motor_enabled is not None:
            result.masked_fill_(~motor_enabled[:, None], 0.)
        magnitude = torch.stack([self._interpolate(x, y, speed[:, k].abs())
                                 for k, (x, y) in enumerate(zip(self.knots, self.losses))], -1)
        loss = magnitude + speed.sign() * self.bias
        if self.friction_mode == "native_floor":
            loss = loss - self.floor
        loss = torch.minimum(loss.clamp_min(0.), self.inertia * self.wheel_inertia_scale * speed.abs() / self.dt)
        self.passive_effort[:, 4:] = -speed.sign() * loss
        self.passive_effort[:, :4] = -self.leg_coulomb * (actual_velocity_rad_s[:, :4] / self.leg_band).clamp(-1., 1.)
        passive = self.passive_effort.clone()
        return ExecutionEffort(result, passive, result + passive, self.drive_bound.clone())

    @torch.no_grad()
    def reset(self, ids=None) -> None:
        """Clear queued drive commands only for reset/disabled environments."""
        if ids is None:
            self.history.zero_()
            self.passive_effort.zero_()
            self.drive_bound.fill_(self.wheel_model["software_torque_cap_nm"])
        else:
            self.history[:, ids] = 0.
            self.passive_effort[ids] = 0.
            self.drive_bound[ids] = self.wheel_model["software_torque_cap_nm"]

    def state_memory(self) -> dict[str, Tensor]:
        """Newest-first input history is sufficient without an absolute ring head."""
        order = (self.head - 1 - torch.arange(len(self.history), device=self.device)) % len(self.history)
        return {"wheel_torque_history_nm": self.history.index_select(0, order).transpose(0, 1).clone()}

    def set_wheel_inertia_scale(self, scale: Tensor) -> None:
        """Use the native body's SAME startup inertia draw, without resampling.

        The backend must separately apply this scale to the requested wheel
        additional armature and verify native readback. This setter changes only
        the software residual's non-reversal bound and exposes the context.
        """
        self._check(scale, 2, "wheel inertia scale")
        self._require("Wheel inertia scale must be positive", scale <= 0)
        self.wheel_inertia_scale.copy_(scale)

    def context_parameters(self) -> dict[str, Tensor]:
        """Per-environment requested mechanics; native readback remains separate."""
        return {
            "wheel_inertia_scale": self.wheel_inertia_scale.clone(),
            "wheel_total_inertia_kg_m2": self.inertia * self.wheel_inertia_scale,
            "wheel_additional_armature_kg_m2": self.armature * self.wheel_inertia_scale,
        }

    def native_parameters(self) -> dict[str, dict[str, float]]:
        """Explicit required native coefficients, with no gearbox rescaling."""
        result = {name: dict(armature=0., friction=0., dynamic_friction=0., viscous_friction=0.)
                  for name in ACTION_ORDER}
        for k, name in enumerate(ACTION_ORDER[4:]):
            result[name]["armature"] = self._armature_values[k]
            if self.friction_mode == "native_floor":
                result[name]["friction"] = self.wheel_rows[k]["static_breakaway_nominal_nm"]
                row = self.wheel_rows[k]
                result[name]["dynamic_friction"] = min(row["moving_friction_nm"]) - abs(row["direction_bias_nm"])
        return result

    def native_effort_limits(self) -> dict[str, float]:
        result = {name: 40. + self.leg_model["coulomb_nm"][i] for i, name in enumerate(ACTION_ORDER[:4])}
        result.update({name: self.wheel_model["software_torque_cap_nm"] + max(row["moving_friction_nm"]) + abs(row["direction_bias_nm"])
                       for name, row in zip(ACTION_ORDER[4:], self.wheel_rows)})
        return result

    def physical_spring_force(self, compression_m: Tensor) -> Tensor:
        """Source 698f67 measured segments and its declared endpoint trend prior."""
        if compression_m.device != self.device or compression_m.dtype != self.dtype:
            raise ValueError("Spring compression device/dtype mismatch")
        self._require("Nonfinite spring compression", ~torch.isfinite(compression_m))
        s = compression_m.clamp(*self.geometry_compression_bounds)
        measured = self._interpolate(self.spring_compression, self.spring_force, s)
        extension = self.spring_model["force_extension"]
        low = self.spring_force[0] * torch.exp(extension["low_compression_rate_per_m"] * (s - self.spring_compression[0]))
        high = self.spring_force[-1] + extension["high_compression_slope_n_per_m"] * (s - self.spring_compression[-1])
        return torch.where(s < self.spring_compression[0], low,
                           torch.where(s > self.spring_compression[-1], high, measured))

    def spring_potential_j(self, compression_m: Tensor) -> Tensor:
        """Integral of the exact applied force from zero compression, in joules."""
        self.physical_spring_force(compression_m)
        x, y = self.spring_compression, self.spring_force
        widths = x[1:] - x[:-1]
        cumulative = torch.cat((y.new_zeros(1), ((y[1:] + y[:-1]) * widths / 2).cumsum(0)))
        extension = self.spring_model["force_extension"]

        def primitive(value):
            s = value.clamp(*self.geometry_compression_bounds)
            index = (torch.searchsorted(x, s.contiguous(), right=True) - 1).clamp(0, len(x) - 2)
            delta = s - x[index]
            inside = cumulative[index] + y[index] * delta + .5 * (y[index + 1] - y[index]) / widths[index] * delta.square()
            rate = extension["low_compression_rate_per_m"]
            low = y[0] / rate * torch.expm1(rate * (s - x[0]))
            high_delta = s - x[-1]
            high = cumulative[-1] + y[-1] * high_delta + .5 * extension["high_compression_slope_n_per_m"] * high_delta.square()
            area = torch.where(s < x[0], low, torch.where(s > x[-1], high, inside))
            return area + self.physical_spring_force(s) * (value - s)

        return primitive(compression_m) - primitive(torch.zeros_like(compression_m))

    def slider_effort(self, native_slider_position_m: Tensor) -> Tensor:
        """Positive slider effort extends the actual native spring coordinate."""
        self._check(native_slider_position_m, 2, "spring position")
        return self.physical_spring_force(self.spring_zero - native_slider_position_m)

    def mount_force_element(self, body_names) -> IdentifiedSpringForceElement:
        return IdentifiedSpringForceElement(self, body_names)

    def receipt(self) -> dict:
        return {
            "source_commit": self.source_manifest["source_commit"], "model_pins": dict(MODEL_PINS),
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_files_sha256": {name: row["sha256"] for name, row in self.source_manifest["files"].items()},
            "asset_model_spec_sha256": self.asset_spec_sha256,
            "asset_manifest_sha256": self.asset_manifest_sha256,
            "action_order": list(ACTION_ORDER), "coordinate": "native_output_shaft",
            "physics_dt_s": self.dt, "friction_mode": self.friction_mode,
            "native_parameters": self.native_parameters(), "native_effort_limits_nm": self.native_effort_limits(),
            "wheel_total_inertia_kg_m2": [r["effective_total_inertia_kg_m2"] for r in self.wheel_rows],
            "wheel_cad_axis_inertia_kg_m2": self._cad_inertia_values,
            "wheel_aggregate_delay_s": [r["aggregate_delay_s"] for r in self.wheel_rows],
            "wheel_delay_owner": "IdentifiedExecution.step torque history; not USB one-way measurement",
            "wheel_velocity_p_nm_per_rad_s": self.wheel_model["controller_velocity_p_nm_per_rad_s"],
            "static_breakaway_reproduced": self.friction_mode == "native_floor",
            "native_coefficients_are_requested_not_readback": True,
            "wheel_inertia_scale_range": [float(self.wheel_inertia_scale.min()), float(self.wheel_inertia_scale.max())],
            "spring_geometry_force_hold_domain_m": list(self.geometry_compression_bounds),
            "spring_measured_domain_m": [float(self.spring_compression[0]), float(self.spring_compression[-1])],
            "spring_extrapolation": "positive_c1_endpoint_trend_v1; hold geometry endpoint outside",
            "spring_pressure_pa": None, "spring_right_symmetry_assumed": True,
            "encoder_holding_compensation": False,
            "legacy_items_to_disable": ["438 response armature/friction/damping/tau", "old wheel envelope",
                "extra aggregate wheel delay", "catalogue 10MPa pressure", "duplicate slider/mount spring forces"],
            "limitations": self.wheel_model["limitations"] + [
                "Leg Coulomb outside measured local speed/posture is a prior; no static/viscous/Stribeck model.",
                "No canonical encoder Jacobian/holding compensation is imported into the selected asset.",
                "Software-dynamic mode omits native static breakaway; source equivalence is not claimed.",
            ],
        }


class IdentifiedSpringForceElement:
    """Equal-and-opposite mount forces; the native closed chain supplies J^T F.

No measured encoder Jacobian is reused across model frames. Applying these
wrenches and slider pressure together would double the physical spring load.
"""

    def __init__(self, execution: IdentifiedExecution, body_names):
        self.execution = execution
        self.body_names = tuple(body_names)
        core = set(execution._bodies)
        full = core | set(execution.auxiliary_body_names)
        if len(set(self.body_names)) != len(self.body_names) or set(self.body_names) not in (core, full):
            raise ValueError("Spring body ordering must match the selected core model")
        constraints = {row["name"]: row for row in execution.spec["constraints"]}
        names, points = [], []
        for side in ("L", "R"):
            upper = constraints[side + "_spring_upper_mount"]
            lower = execution._joints[side * 3 + "_joint2"]
            slide = execution._joints[side + "_spring_slide"]
            if (upper["type"] != "spherical" or upper["body0"] != side + "_link1"
                    or upper["body1"] != slide["child"] or lower["parent"] != side + "_link2"
                    or lower["child"] != slide["parent"]):
                raise ValueError("Spring mounting topology differs")
            names.extend((upper["body0"], lower["parent"]))
            points.extend((upper["local_pos0_m"], [lower["origin"][i][3] for i in range(3)]))
        self.main_link_body_names = tuple(names)
        self.body_ids = tuple(self.body_names.index(name) for name in names)
        self._indices = torch.tensor(self.body_ids, device=execution.device)
        self._points = execution._tensor(points)

    def evaluate(self, body_link_pose_world: Tensor, body_com_position_world: Tensor,
                 spring_strength: Tensor) -> SpringForces:
        """Link pose is xyz+xyzw; COM position is world xyz, strength explicit."""
        e = self.execution
        if (body_link_pose_world.shape != (e.count, len(self.body_names), 7)
                or body_com_position_world.shape != (e.count, len(self.body_names), 3)
                or spring_strength.shape not in ((e.count, 1), (e.count, 2))
                or any(t.device != e.device or t.dtype != e.dtype for t in (
                    body_link_pose_world, body_com_position_world, spring_strength))):
            raise ValueError("Spring geometry shape/device/dtype mismatch")
        pose = body_link_pose_world.index_select(1, self._indices)
        com = body_com_position_world.index_select(1, self._indices)
        norm = pose[..., 3:].norm(dim=-1, keepdim=True)
        q = pose[..., 3:] / norm
        local = self._points.expand(e.count, -1, -1)
        cross = torch.linalg.cross(q[..., :3], local)
        mounts = pose[..., :3] + local + 2 * (q[..., 3:] * cross + torch.linalg.cross(q[..., :3], cross))
        pairs = mounts.unflatten(1, (2, 2))
        separation = pairs[:, :, 0] - pairs[:, :, 1]
        length = separation.norm(dim=-1)
        compression = e.spring_full_length - length
        valid = (torch.isfinite(pose).flatten(1).all(-1) & torch.isfinite(com).flatten(1).all(-1)
                 & (norm.squeeze(-1) > 0).all(-1) & (length > 0).all(-1)
                 & torch.isfinite(spring_strength).all(-1) & (spring_strength >= 0).all(-1))
        e._require("Nonfinite or degenerate spring mounting geometry", ~valid)
        magnitude = e.physical_spring_force(compression) * spring_strength
        force = magnitude[..., None] * separation / length[..., None]
        forces = torch.stack((force, -force), 2).flatten(1, 2)
        torques = torch.linalg.cross(mounts - com, forces)
        return SpringForces(forces, torques, mounts, length, compression, magnitude, valid,
                            compression / e.spring_stroke,
                            (compression >= 0) & (compression <= e.spring_stroke),
                            (compression >= e.spring_compression[0]) & (compression <= e.spring_compression[-1]))
