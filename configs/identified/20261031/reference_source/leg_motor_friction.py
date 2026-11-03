"""Measured output-shaft moving friction, separate from motor current and gas load.

Only the local Coulomb plateau is admitted. The small linear velocity band is
an explicit numerical regularization, not an identified viscous/Stribeck law or
static breakaway threshold. No friction compensation is added to the controller.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import torch

JOINTS = ("L_joint1", "LL_joint1", "R_joint1", "RR_joint1")
MOTORS = ("left_hip", "left_knee", "right_hip", "right_knee")
ACTIVE_IDS = (0, 1, 3, 4)
MODEL = "model/dm_leg_friction_20261031/model.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def declaration(root, path=MODEL):
    return dict(enabled=True, model_path=path, model_sha256=digest(Path(root)/path))


def load(settings, root):
    if not settings or settings == {"enabled": False}:
        return dict(enabled=False), None
    if (set(settings) != {"enabled", "model_path", "model_sha256"}
            or settings["enabled"] is not True):
        raise ValueError("Leg friction needs an explicit enabled model/hash binding")
    root = Path(root)
    path = root/settings["model_path"]
    if digest(path) != settings["model_sha256"]:
        raise ValueError("Leg friction model SHA differs")
    model = json.loads(path.read_text())
    if (model.get("schema") != "dm-leg-moving-friction-v1"
            or model.get("joint_names") != list(JOINTS)
            or model.get("motor_names") != list(MOTORS)
            or model.get("torque_units") != "Nm at DM active output"
            or model.get("bias_applied") is not False
            or model.get("controller_compensation") is not False
            or model.get("outside_identified_domain") != "constant_coulomb_prior_not_identified"):
        raise ValueError("Unknown leg moving-friction interpretation or axis mapping")
    for name, expected in model["files_sha256"].items():
        target = (path.parent/name).resolve()
        if not target.is_relative_to(path.parent.resolve()) or digest(target) != expected:
            raise ValueError("Leg friction source changed: "+name)
    parameters = json.loads((path.parent/"parameters.json").read_text())
    if parameters.get("right_symmetry_authorized") is not True:
        raise ValueError("Right leg friction needs its explicit symmetry assumption")
    values = []
    for motor in MOTORS:
        row = parameters["motors"][motor]
        value = row["coulomb_plateau_candidate_nm"]
        if not math.isfinite(value) or value <= 0.:
            raise ValueError("Invalid output-shaft Coulomb torque")
        for field in ("viscous_coefficient_nm_s_rad", "static_breakaway_nm", "stribeck_velocity_rad_s"):
            if row[field] is not None or model[field] is not None:
                raise ValueError("Unidentified friction parameters must remain null")
        if motor.startswith("right_"):
            source = "left_"+motor.split("_", 1)[1]
            if row.get("symmetry_source") != source or value != parameters["motors"][source]["coulomb_plateau_candidate_nm"]:
                raise ValueError("Right leg friction differs from the declared left approximation")
        values.append(value)
    numerical = model["numerical_regularization"]
    if (model["coulomb_nm"] != values
            or numerical != dict(kind="linear_velocity_band", velocity_rad_s=.05, identified=False)
            or model["physical_static_model_enabled"] is not False):
        raise ValueError("Leg friction differs from the admitted Coulomb/numerical model")
    return deepcopy(settings), model


class LegMotorFriction:
    def __init__(self, count, device, settings=None, *, root):
        self.cfg, self.model = load(settings, root)
        self.enabled = self.cfg["enabled"]
        if not self.enabled:
            return
        self.coulomb = torch.tensor(self.model["coulomb_nm"], device=device, dtype=torch.float32)
        self.velocity = torch.zeros(count, 4, device=device)
        self.passive_effort = torch.zeros_like(self.velocity)
        self.band = self.model["numerical_regularization"]["velocity_rad_s"]

    @property
    def native_effort_limit(self):
        # The electrical motor is still clipped at40Nm BEFORE adding dissipation.
        return 40.+max(self.model["coulomb_nm"])

    def effort(self, actual_velocity):
        if (actual_velocity.shape != self.velocity.shape or actual_velocity.device != self.velocity.device
                or actual_velocity.dtype != self.velocity.dtype):
            raise ValueError("Leg friction needs actual N x4 output-shaft velocities")
        with torch.no_grad():
            self.velocity.copy_(actual_velocity)
            self.passive_effort.copy_(-self.coulomb*(actual_velocity/self.band).clamp(-1., 1.))
        return self.passive_effort

    def summary(self):
        return dict(config=self.cfg, joint_names=list(JOINTS), coulomb_nm=self.model["coulomb_nm"],
            numerical_regularization=self.model["numerical_regularization"],
            physical_static_model_enabled=False, viscous_identified=False, stribeck_identified=False,
            bias_applied=False, right_symmetry_assumed=True,
            source_domain="local common-angle moving test; all other poses/speeds are an explicit prior",
            motor_torque_cap_nm=40., passive_loss_separate_from_motor=True,
            native_effort_limit_nm=self.native_effort_limit)
