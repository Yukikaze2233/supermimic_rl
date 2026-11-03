#!/usr/bin/env python3
"""Build the 1D closed-chain leg table from the pinned asset and upstream kinematics.

For each side the secondary hip motor sweeps phi = q_secondary - q_primary
with the primary fixed at its nominal angle; the other leg stays nominal.
Passive joints are solved exactly (closure tolerance 1e-10 m) by continuation
from the nominal assembly, so the table follows a single assembly branch.
Slopes come from the implicit-function theorem (passive) and from central
differences along that exact tangent (derived geometry). The domain ends are
located by bisection where the solve stops converging (the knee stops).
Interpolation is validated at every grid midpoint against an exact solve.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

FORMAT = "supermimic-leg-linkage-table-v1"
SIDES = {"left": ("L_joint1", "LL_joint1", "L_joint3", "L"),
         "right": ("R_joint1", "RR_joint1", "R_joint3", "R")}
TOLERANCE_M = 1e-10
DOMAIN_MARGIN_RAD = 1e-4


class Builder:
    def __init__(self, upstream_root: Path, asset_directory: Path):
        sys.path.insert(0, str(upstream_root / "src"))
        from transformer_rl.formal.control.observation import MechanicalBinding
        from transformer_rl.formal.tasks.reset_kinematics import AssetResetKinematics

        self.upstream_commit = subprocess.check_output(
            ["git", "-C", str(upstream_root), "rev-parse", "HEAD"], text=True).strip()
        manifest_bytes = (asset_directory / "manifest.json").read_bytes()
        spec_bytes = (asset_directory / "model_spec.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        self.binding = MechanicalBinding.from_manifest(
            manifest, manifest["tree_joint_names"], manifest["rigid_body_names"], include_auxiliary=False)
        self.kin = AssetResetKinematics(manifest_bytes, spec_bytes, self.binding)
        self.names = self.binding.tree_joint_names
        self.policy = self.kin.policy_tree_indices
        self.nominal = self.kin.nominal_tree_position[None].clone()
        self.hashes = {"asset_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                       "model_spec_sha256": hashlib.sha256(spec_bytes).hexdigest()}

    def solve(self, seed, primary, secondary, phi):
        target = seed[:, list(self.policy)].clone()
        columns = [self.names[i] for i in self.policy]
        target[:, columns.index(secondary)] = target[:, columns.index(primary)] + phi
        q, _, maximum, _, converged, _, _ = self.kin._solve_fixed(
            seed, self.policy, target, closure_tolerance_m=TOLERANCE_M, maximum_iterations=64)
        return q, bool(converged), float(maximum)

    def side_passive(self, prefix):
        """Passive tree joints of one leg, in tree order; every left/right name starts with L/R."""
        policy = set(self.policy)
        return [i for i, name in enumerate(self.names) if i not in policy and name.startswith(prefix)]

    def features(self, q, wheel_joint):
        state = self.kin.forward(q)
        knee = self.kin.inner_knee_angles(q)
        return state.joint_origin_world_m[:, self.names.index(wheel_joint)], knee

    def tangent(self, q, secondary, passive):
        _, jac = self.kin.closure(q, jacobian=True)
        all_passive = [i for i in range(18) if i not in self.policy]
        rhs = -jac[0, :, self.names.index(secondary)]
        solution = torch.linalg.lstsq(jac[0][:, all_passive], rhs[:, None]).solution[:, 0]
        direction = torch.zeros(18, dtype=q.dtype)
        direction[self.names.index(secondary)] = 1
        direction[all_passive] = solution
        return direction

    def domain_end(self, seed, phi_good, phi_bad, primary, secondary):
        for _ in range(40):
            middle = .5 * (phi_good + phi_bad)
            q, converged, _ = self.solve(seed, primary, secondary, middle)
            if converged:
                phi_good, seed = middle, q
            else:
                phi_bad = middle
        return phi_good

    def build_side(self, side, spacing):
        primary, secondary, wheel, prefix = SIDES[side]
        passive = self.side_passive(prefix)
        nominal_phi = float(self.nominal[0, self.names.index(secondary)] - self.nominal[0, self.names.index(primary)])
        ends = []
        for direction in (-1., 1.):
            seed, phi = self.nominal.clone(), nominal_phi
            while True:
                q, converged, _ = self.solve(seed, primary, secondary, phi + direction * spacing)
                if not converged:
                    break
                seed, phi = q, phi + direction * spacing
            ends.append(self.domain_end(seed, phi, phi + direction * spacing, primary, secondary))
        # Stay strictly inside the stops: at the exact stop the clamped solve can
        # fall just short of the closure tolerance with a different seed.
        phi_min, phi_max = ends[0] + DOMAIN_MARGIN_RAD, ends[1] - DOMAIN_MARGIN_RAD
        count = int(round((phi_max - phi_min) / spacing)) + 1
        grid = torch.linspace(phi_min, phi_max, count, dtype=torch.float64)
        start = int(torch.argmin((grid - nominal_phi).abs()))
        solutions = [None] * count
        for order in (range(start, count), range(start - 1, -1, -1)):
            seed = self.nominal.clone() if order.start == start else solutions[start]
            for index in order:
                q, converged, residual = self.solve(seed, primary, secondary, float(grid[index]))
                if not converged:
                    raise RuntimeError(f"{side}: knot {index} phi={float(grid[index])} did not close ({residual:.2e} m)")
                solutions[index] = seed = q
        values, slopes = [], []
        epsilon = 1e-6
        for q in solutions:
            direction = self.tangent(q, secondary, passive)
            center, knee = self.features(q, wheel)
            plus_center, plus_knee = self.features(q + epsilon * direction, wheel)
            minus_center, minus_knee = self.features(q - epsilon * direction, wheel)
            knee_index = 0 if side == "left" else 1
            values.append(torch.cat((q[0, passive], center[0], knee[0, knee_index:knee_index + 1])))
            slopes.append(torch.cat((direction[passive], (plus_center - minus_center)[0] / (2 * epsilon),
                                     ((plus_knee - minus_knee)[0, knee_index:knee_index + 1]) / (2 * epsilon))))
        state = self.kin.forward(self.nominal)
        axis = state.joint_axis_world[0, self.names.index(primary)]
        pivot = state.joint_origin_world_m[0, self.names.index(primary)]
        return {"primary_joint": primary, "secondary_joint": secondary,
                "passive_joints": [self.names[i] for i in passive],
                "phi_min": phi_min, "phi_max": phi_max, "nominal_phi": nominal_phi,
                "reference_primary_rad": float(self.nominal[0, self.names.index(primary)]),
                "hip_axis_base": axis.tolist(), "hip_pivot_base_m": pivot.tolist(),
                "values": torch.stack(values).tolist(), "slopes": torch.stack(slopes).tolist(),
                "_solutions": solutions, "_passive": passive}


def validate(builder, side, table):
    """Hermite interpolation against exact solves at every grid midpoint."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from supermimic_rl.world_model_v2.leg_linkage import LinkageSide

    primary, secondary, wheel, _ = SIDES[side]
    geometry = LinkageSide(side, primary, secondary, table["phi_min"], table["phi_max"],
                           torch.tensor(table["values"]), torch.tensor(table["slopes"]),
                           table["reference_primary_rad"], torch.tensor(table["hip_axis_base"]),
                           torch.tensor(table["hip_pivot_base_m"]))
    solutions, passive = table["_solutions"], table["_passive"]
    grid = torch.linspace(table["phi_min"], table["phi_max"], len(solutions), dtype=torch.float64)
    middles = .5 * (grid[1:] + grid[:-1])
    predicted, valid = geometry.evaluate(middles)
    passive_error, center_error, closure = [], [], []
    for index, phi in enumerate(middles.tolist()):
        q, converged, _ = builder.solve(solutions[index], primary, secondary, phi)
        if not converged:
            raise RuntimeError(f"{side}: midpoint {phi} did not close")
        center, _ = builder.features(q, wheel)
        passive_error.append((predicted[index, :len(passive)] - q[0, passive]).abs().max())
        center_error.append((predicted[index, len(passive):len(passive) + 3] - center[0]).norm())
        interpolated = q.clone()
        interpolated[0, passive] = predicted[index, :len(passive)]
        closure.append(builder.kin.closure(interpolated).norm(dim=-1).max())
    return {"midpoints": len(middles), "all_valid": bool(valid.all()),
            "max_passive_error": float(torch.stack(passive_error).max()),
            "max_wheel_center_error_m": float(torch.stack(center_error).max()),
            "max_interpolated_closure_residual_m": float(torch.stack(closure).max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--asset-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spacing", type=float, default=2e-3)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    builder = Builder(args.upstream_root.resolve(), args.asset_directory.resolve())
    sides, validation = {}, {}
    for side in SIDES:
        table = builder.build_side(side, args.spacing)
        validation[side] = validate(builder, side, table)
        sides[side] = {key: value for key, value in table.items() if not key.startswith("_")}
    passive_count = len(sides["left"]["passive_joints"])
    payload = {
        "format": FORMAT,
        "features": ["passive_joint", "wheel_center_reference_base_m", "knee_inner_angle_rad"],
        "feature_sizes": {"passive_joint": passive_count, "wheel_center_reference_base_m": 3,
                          "knee_inner_angle_rad": 1},
        "provenance": {**builder.hashes, "upstream_commit": builder.upstream_commit,
                       "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "closure_tolerance_m": TOLERANCE_M, "spacing_rad": args.spacing,
                       "domain_margin_rad": DOMAIN_MARGIN_RAD,
                       "phi_definition": "q_secondary - q_primary; primary rotates the whole leg",
                       "validation": validation},
        "sides": sides,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload) + "\n")
    print(json.dumps({"validation": validation,
                      "domains": {s: [sides[s]["phi_min"], sides[s]["phi_max"]] for s in sides},
                      "passive_joints": {s: sides[s]["passive_joints"] for s in sides}}, indent=2))


if __name__ == "__main__":
    main()
