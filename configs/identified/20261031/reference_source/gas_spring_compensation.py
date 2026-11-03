"""Measured elastic-load holding effort, shared with RMCS's frozen C++ table.

The force curve includes the mean effective gas-spring load, not separately
identified Coulomb friction or directional damping. Right force is a declared
left/right symmetry assumption; signed geometry is evaluated per side.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch


def load_compensation(settings, root):
    cfg = dict(settings)
    if cfg.get("enabled") is not True:
        raise ValueError("Gas-spring compensation must be explicitly enabled")
    path = Path(cfg["model_file"])
    if not path.is_absolute():
        path = Path(root) / path
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != cfg["model_sha256"]:
        raise ValueError("Gas-spring compensation model hash differs")
    model = json.loads(content)
    if (model.get("schema") != "wheel-leg-gas-spring-compensation-v2"
            or model.get("coordinate") != "active_model_output"
            or model.get("direction_friction_compensated") is not False):
        raise ValueError("Unsupported gas-spring compensation model")
    gain = cfg.get("gain", 1.)
    if not math.isfinite(gain) or not 0. <= gain <= 1.:
        raise ValueError("Invalid gas-spring compensation gain")
    if cfg.get("force_extrapolation") != "positive_c1_endpoint_trend_v1":
        raise ValueError("Declare the unmeasured endpoint-trend extrapolation explicitly")
    if cfg.get("physical_curve_outside_geometry") != "hold_geometry_endpoint":
        raise ValueError("Declare the physical-force prior outside the geometry lookup")
    bounds = model["soft_limit_active_difference_rad"]
    if (len(bounds) != 2 or not all(math.isfinite(v) for v in bounds) or bounds[0] >= bounds[1]
            or cfg.get("relative_soft_limits") != dict(minimum=bounds[0],maximum=bounds[1],
                margin=.03,projection="outward_relative_torque_v1")):
        raise ValueError("Gas-compensated training requires relative soft limits matching its authenticated model")
    return cfg, model


class GasSpringCompensation:
    def __init__(self, settings, *, root, device, dtype=torch.float32):
        self.cfg, self.model = load_compensation(settings, root)
        self.gain = self.cfg.get("gain", 1.)
        self.sides = []
        for side in ("left", "right"):
            row = self.model["sides"][side]
            self.sides.append({key: torch.tensor(row[key], device=device, dtype=dtype)
                for key in ("delta_rad", "beta_deg", "compression_m", "length_jacobian_m_per_rad")})
        self.compression = torch.tensor(self.model["compression_knots_m"], device=device, dtype=dtype)
        self.force_n = torch.tensor(self.model["force_knots_n"], device=device, dtype=dtype)
        ends = torch.stack([row["compression_m"][[0,-1]] for row in self.sides])
        self.geometry_compression_min = ends.min()
        self.geometry_compression_max = ends.max()
        source = Path(root) / "model/gas_spring_identified_symmetric_20261028/model.json"
        if hashlib.sha256(source.read_bytes()).hexdigest() != self.model["source_curve_sha256"]:
            raise ValueError("Gas-spring source fit hash differs")
        fit = json.loads(source.read_text())
        self.extended_length_m = fit["extended_pin_distance_m"]
        from .identified_gas_spring import IdentifiedGasSpringCurve
        self.curve = IdentifiedGasSpringCurve(tuple(self.model["compression_knots_m"]),
            tuple(self.model["force_knots_n"]))
        self.right_curve = self.curve.for_side("right", assume_symmetry=True)
        self.settings = dict(self.cfg, sha256=self.cfg["model_sha256"],
            outside_identified_domain="dev_compensation_v2",
            installed_stroke_m=fit["stroke_m"])
        for knots in (self.compression, *(row["delta_rad"] for row in self.sides)):
            if not bool(torch.isfinite(knots).all() and (knots[1:] > knots[:-1]).all()):
                raise ValueError("Gas-spring interpolation needs finite increasing knots")

    @staticmethod
    def interpolate(knots, values, x):
        bounded = x.clamp(knots[0], knots[-1])
        index = torch.searchsorted(knots, bounded.contiguous()).clamp(1, len(knots) - 1)
        u = (bounded - knots[index - 1]) / (knots[index] - knots[index - 1])
        if values.ndim > 1:
            u = u.unsqueeze(-1)
        return torch.lerp(values[index - 1], values[index], u)

    def physical_force(self, compression):
        if not bool(torch.isfinite(compression).all()):
            raise ValueError("Nonfinite spring compression")
        replay = getattr(self, "force_tensor_replay", None)
        force, outside = self._evaluate_force(compression) if replay is None else replay(compression)
        self.last_compression, self.last_force, self.last_outside = compression, force, outside
        return force

    def _evaluate_force(self, compression):
        # Preserve every measured segment. Outside, extend the endpoint slope
        # continuously; the low-compression exponential stays strictly positive.
        s = compression.clamp(self.geometry_compression_min,self.geometry_compression_max)
        measured = self.interpolate(self.compression, self.force_n, s)
        extension = self.model["force_extension"]
        low = self.force_n[0] * torch.exp(extension["low_compression_rate_per_m"] * (s-self.compression[0]))
        high = self.force_n[-1] + extension["high_compression_slope_n_per_m"] * (s-self.compression[-1])
        force = torch.where(s<self.compression[0],low,torch.where(s>self.compression[-1],high,measured))
        outside = (compression < self.compression[0]) | (compression > self.compression[-1])
        return force, outside

    def evaluate(self, q):
        """q[...,4] = (left hip, left aux, right hip, right aux), model radians."""
        if q.shape[-1] != 4:
            raise ValueError("Compensation requires four active model angles")
        replay = getattr(self, "tensor_replay", None)
        torque, self.last = self._evaluate(q) if replay is None else replay(q)
        valid = self.last["geometry_valid"]
        return torque, dict(beta_deg=torch.where(valid, torch.rad2deg(self.last["knee_inner_rad"]), float("nan")),
            force_n=self.last["axial_force_n"]*valid, weight=self.gain*valid.to(q.dtype),
            extrapolated=self.last["outside_identified"] & valid)

    def _evaluate(self, q):
        deltas = (q[..., 1] - q[..., 0], q[..., 2] - q[..., 3])
        efforts, betas, forces, compressions, jacobians, validities, extrapolated = [], [], [], [], [], [], []
        for delta, row in zip(deltas, self.sides):
            knots = row["delta_rad"]
            inside = (delta >= knots[0]) & (delta <= knots[-1])
            beta = self.interpolate(knots, row["beta_deg"], delta)
            compression = self.interpolate(knots, row["compression_m"], delta)
            jacobian = self.interpolate(knots, row["length_jacobian_m_per_rad"], delta)
            force, outside = self._evaluate_force(compression)
            weight = self.gain * inside.to(q.dtype)
            efforts.append(-jacobian * (force * weight).unsqueeze(-1))
            betas.append(torch.deg2rad(beta))
            forces.append(force)
            compressions.append(compression)
            jacobians.append(jacobian)
            validities.append(inside)
            extrapolated.append(outside)
        torque = torch.cat(efforts, dim=-1)
        # Preserve nonfinite input faults instead of hiding them behind a zero
        # out-of-domain mask. The environment's existing finite guard owns stops.
        torque = torch.where(torch.isfinite(q).all(-1, keepdim=True), torque, float("nan"))
        compression = torch.stack(compressions, -1)
        return torque, dict(encoder_position=q.clone(), knee_inner_rad=torch.stack(betas, -1),
            length_m=self.extended_length_m-compression, compression_m=compression,
            axial_force_n=torch.stack(forces, -1), length_jacobian_m_rad=torch.cat(jacobians, -1),
            holding_effort_nm=torque, geometry_valid=torch.stack(validities, -1),
            outside_identified=torch.stack(extrapolated, -1))

    def motor_efforts(self, pd_effort, encoder_position):
        holding, _ = self.evaluate(encoder_position)
        requested = pd_effort+holding
        limited = requested.clamp(-40., 40.)
        limits = self.cfg["relative_soft_limits"]
        coefficients = encoder_position.new_tensor([[-1., 1.], [1., -1.]])
        relative = (encoder_position.unflatten(-1, (2, 2))*coefficients).sum(-1)
        efforts = limited.unflatten(-1, (2, 2))
        outward = (efforts*coefficients).sum(-1)
        blocked = (((relative >= limits["maximum"]-limits["margin"]) & (outward > 0.))
            | ((relative <= limits["minimum"]+limits["margin"]) & (outward < 0.)))
        projected = (efforts-coefficients*torch.where(blocked, outward/2., 0.).unsqueeze(-1)).flatten(-2)
        self.last.update(pd_effort_nm=pd_effort.clone(), requested_effort_nm=requested,
            limited_effort_nm=limited, projected_effort_nm=projected.clone())
        return projected

    def install(self, env):
        env.identified_spring = env.spring_feedforward = self
        env.startup_report["identified_gas_spring"] = dict(self.summary(),
            sha256=self.cfg["model_sha256"], default_catalogue_force_replaced_for_this_task=True,
            applied_to=["L_spring_slide", "R_spring_slide"], physical_call_hz=round(1./env.dt),
            model_source="dev/self_righting_policy", right_leg_measured=False)
        env.startup_report["encoder_spring_feedforward"] = dict(self.summary(),
            physical_spring_retained=True, privileged_force_or_slider_input=False,
            controller_hz=round(1./env.dt), total_leg_limit_nm=40.,
            formula="PD + dev fitted-load holding -> +/-40Nm -> outward relative-torque projection",
            input_source="same sampled/delayed low-level encoders as PD, before target winding",
            input=["L_joint1", "LL_joint1", "R_joint1", "RR_joint1"],
            disabled_motor_zero_after_compensation=True)

    def summary(self):
        return dict(self.cfg, model_spec_sha256=self.model["model_spec_sha256"],
            measured_beta_domain_deg=self.model["measured_beta_domain_deg"],
            geometry_beta_domain_deg=self.model["geometry_beta_domain_deg"],
            right_leg_symmetry_assumed=self.model["right_leg_symmetry_assumed"],
            full_soft_limit_coverage=True,force_extrapolation="positive_c1_endpoint_trend_v1",
            direction_friction_compensated=False)
