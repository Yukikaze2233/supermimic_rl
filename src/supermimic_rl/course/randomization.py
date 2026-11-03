"""Requested-v2 DR on native Newton or PhysX mechanics and the 41e sensor/control path.

Ranges are an explicitly requested simulation stress distribution, not measured
hardware uncertainty. Source identities and the SI conversion are in provenance.
No upstream file is modified, and motor strength never raises software caps.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from types import MethodType

import torch

from ..adapters.mechanics import read_articulation_mechanics
from .materials import make_material_reader_type, material_mapping_provenance


SOURCE_PINS = {
    "contract": "5e2e64523bd1c3ed0a40b537b8fef7cfc48d9b8173860550d470e98da8272157",
    "robust_getup.py": "ace9a3f0552c2aa7be16beae8203602ecdd41eafcd2fd473c527bc9c95235717",
    "imu_mount.py": "49cb2dae1ab2f18ae456c871c1633382b6413d217d89405ca8b917dd77a4429d",
    "randomization_study.py": "80ec179c8af89f85f0439c0fd15e78815860a78c98ec2ab99eb6de6af187dec1",
}


@dataclass(frozen=True)
class RequestedRandomizationConfig:
    nominal_fraction: float = .25
    mass_fraction: float = .08
    independent_inertia_fraction: float = .06
    base_com_offset_m: tuple[float, float, float] = (.012, .0064, .008)
    motor_strength_fraction: float = .16
    encoder_bias_rad: float = .004
    gyro_bias_rad_s: float = .00525
    imu_fixed_degrees: tuple[float, float, float] = (.8, .8, .8 / 3)
    encoder_noise_rad: float = .00035
    leg_velocity_noise_rad_s: float = .0175
    wheel_velocity_noise_rad_s: float = .0875
    gyro_noise_rad_s: float = .0035
    gravity_noise: float = .0014
    full_at_coordinate: int = 5000
    # Disturbances beyond requested-v2, with the native episode-randomization
    # semantics: a held base-frame wrench redrawn per interval, and world
    # velocity steps on ordinary/slope/stair rows (SCUT push_robot and
    # base_external_force_torque). Zero keeps the requested-v2 contract.
    external_force_n: float = 0.
    external_torque_nm: float = 0.
    push_linear_m_s: float = 0.
    push_yaw_rad_s: float = 0.
    push_interval_s: tuple[float, float] = (5., 10.)
    wrench_interval_s: tuple[float, float] = (5., 10.)

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name in ("push_interval_s", "wrench_interval_s"):
                if (not isinstance(value, (tuple, list)) or len(value) != 2
                        or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
                        or not 0 < value[0] <= value[1]):
                    raise ValueError(name + " requires two ordered positive bounds")
                object.__setattr__(self, name, tuple(float(v) for v in value))
            elif name in ("base_com_offset_m", "imu_fixed_degrees"):
                if (not isinstance(value, (tuple, list)) or len(value) != 3
                        or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in value)):
                    raise ValueError(name + " requires three nonnegative finite bounds")
                object.__setattr__(self, name, tuple(value))
            elif type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(name + " must be finite and nonnegative")
        if (self.nominal_fraction > 1 or any(getattr(self, name) >= 1 for name in
                ("mass_fraction", "independent_inertia_fraction", "motor_strength_fraction"))
                or type(self.full_at_coordinate) is not int or self.full_at_coordinate < 1):
            raise ValueError("Invalid nominal fraction, positive multiplicative range or course horizon")

    @property
    def pushes(self) -> bool:
        return bool(self.push_linear_m_s or self.push_yaw_rad_s)

    @property
    def wrenches(self) -> bool:
        return bool(self.external_force_n or self.external_torque_nm)

    def provenance(self):
        return {
            "source_root": "/home/kaiser/workspace/isaac_sim-6-task-200hz",
            "source_contract": "contracts/wheel_leg_randomization_study_v2.json",
            "source_code_directory": "src/wheeled_tasks/chassis",
            "source_sha256": dict(SOURCE_PINS), "configuration": asdict(self),
            "classification": "requested simulation stress ranges; not hardware measurements",
            "distribution": "independent uniform bounded parameters; Gaussian policy white noise",
            "nominal_mask": "fixed startup Bernoulli lanes shared across body/motor/sensor groups",
            "inertia_rule": "mass scale times independent scalar inertia scale per named body",
            "encoder_bias": "independent policy and PD draws on four leg positions; wheels zero",
            "imu": "R_BS=Rz(yaw)Ry(pitch)Rx(roll), uniform per-reset fixed bounds; no wobble",
            "gyro_units": "SI rad/s; source gyro feature was multiplied by .5 before bias/noise",
            "sensor_noise_order": "fixed transforms in physical ring; causal delay then Gaussian policy noise",
            "motor_strength_order": "nominal capped PD, strength once, fixed software cap, identified delay",
            "course_clock": "mapped real-exposure coordinate, not actor or PPO iteration count",
            "disturbances": ("none (requested-v2)" if not (self.pushes or self.wrenches) else
                             "uniform held base-frame wrench and world velocity steps, redrawn per uniform "
                             "interval, scaled by course difficulty; nominal lanes undisturbed"),
            "contact_material": material_mapping_provenance(),
        }


def sensor_to_body_rotation(rpy):
    """Return Rz(yaw) Ry(pitch) Rx(roll); use transpose on body measurements."""
    r, p, y = rpy.unbind(-1)
    cr, sr, cp, sp, cy, sy = r.cos(), r.sin(), p.cos(), p.sin(), y.cos(), y.sin()
    return torch.stack((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
                        sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
                        -sp, cp * sr, cp * cr), -1).reshape(*rpy.shape[:-1], 3, 3)


class CourseRandomization:
    """Shared nominal lanes and reset-local actual controller/sensor parameters."""

    def __init__(self, num_envs, seed=0, device="cpu", config=None):
        self.config = RequestedRandomizationConfig() if config is None else config
        if (not isinstance(self.config, RequestedRandomizationConfig)
                or type(num_envs) is not int or num_envs < 1
                or type(seed) is not int or not 0 <= seed < 2**63):
            raise ValueError("Randomization requires parsed configuration, positive batch and seed")
        self.num_envs, self.seed, self.device = num_envs, seed, torch.device(device)
        mask_generator = torch.Generator(device="cpu").manual_seed(seed)
        self.nominal = (torch.rand(num_envs, generator=mask_generator) < self.config.nominal_fraction).to(self.device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed ^ 1299709)
        self.strength = torch.ones(num_envs, 1, device=self.device)
        self.scale = torch.zeros(num_envs, device=self.device)
        self.motor_strength = torch.ones(num_envs, 6, device=self.device)
        self.encoder_bias_rad = torch.zeros(num_envs, 6, device=self.device)
        self.policy_encoder_bias_rad = torch.zeros_like(self.encoder_bias_rad)
        self.gyro_bias_rad_s = torch.zeros(num_envs, 3, device=self.device)
        self.imu_rpy_rad = torch.zeros_like(self.gyro_bias_rad_s)
        self.no_load_scale = torch.ones_like(self.motor_strength)
        self.time_constant_scale = torch.ones_like(self.motor_strength)
        self.base_wrench = torch.zeros(num_envs, 6, device=self.device)
        self.disturbance_scale = torch.zeros(num_envs, device=self.device)
        self.push_timer_s = torch.full((num_envs,), math.inf, device=self.device)
        self.wrench_timer_s = torch.full((num_envs,), math.inf, device=self.device)
        self.reset_rows_consumed = 0

    def _uniform(self, count, width):
        return torch.rand(count, width, generator=self.generator, device=self.device) * 2 - 1

    def reset(self, mask, update):
        if (not isinstance(mask, torch.Tensor) or mask.shape != (self.num_envs,)
                or mask.dtype != torch.bool or mask.device != self.device
                or type(update) is not int or update < 0):
            raise ValueError("Randomization reset requires a boolean row mask and course coordinate")
        rows = mask.nonzero().flatten()
        fraction = min(update / self.config.full_at_coordinate, 1.)
        difficulty = fraction * fraction * (3 - 2 * fraction)
        scale = difficulty * (~self.nominal[rows, None])
        count, c = len(rows), self.config
        self.scale[rows] = difficulty
        self.motor_strength[rows] = 1 + scale * self._uniform(count, 6) * c.motor_strength_fraction
        self.encoder_bias_rad[rows, :4] = scale * self._uniform(count, 4) * c.encoder_bias_rad
        self.policy_encoder_bias_rad[rows, :4] = scale * self._uniform(count, 4) * c.encoder_bias_rad
        self.gyro_bias_rad_s[rows] = scale * self._uniform(count, 3) * c.gyro_bias_rad_s
        self.imu_rpy_rad[rows] = (scale * self._uniform(count, 3)
                                 * self.imu_rpy_rad.new_tensor(c.imu_fixed_degrees) * math.pi / 180)
        # Disturbance draws come last and only when enabled, so requested-v2
        # streams are unchanged.
        self.disturbance_scale[rows] = scale[:, 0]
        self.base_wrench[rows] = 0
        self.push_timer_s[rows] = self._interval(count, c.push_interval_s) if c.pushes else math.inf
        self.wrench_timer_s[rows] = self._interval(count, c.wrench_interval_s) if c.wrenches else math.inf
        self.reset_rows_consumed += count
        return difficulty

    def _interval(self, count, bounds):
        low, high = bounds
        return low + (high - low) * torch.rand(count, generator=self.generator, device=self.device)

    def advance_wrench(self, dt_s):
        """Redraw due wrenches; the held [N, 6] base-frame force/torque."""
        self.wrench_timer_s.sub_(dt_s)
        rows = (self.wrench_timer_s <= 0).nonzero().flatten()
        if rows.numel():
            c = self.config
            limit = self.base_wrench.new_tensor([c.external_force_n] * 3 + [c.external_torque_nm] * 3)
            self.base_wrench[rows] = self._uniform(len(rows), 6) * limit * self.disturbance_scale[rows, None]
            self.wrench_timer_s[rows] = self._interval(len(rows), c.wrench_interval_s)
        return self.base_wrench

    def push(self, dt_s, eligible):
        """Rows due for a push and their world [vx, vy, wz] velocity steps."""
        self.push_timer_s.sub_(dt_s)
        rows = ((self.push_timer_s <= 0) & eligible).nonzero().flatten()
        c = self.config
        limit = self.base_wrench.new_tensor([c.push_linear_m_s, c.push_linear_m_s, c.push_yaw_rad_s])
        delta = self._uniform(len(rows), 3) * limit * self.disturbance_scale[rows, None]
        if rows.numel():
            self.push_timer_s[rows] = self._interval(len(rows), c.push_interval_s)
        return rows, delta

    def transform_measurement(self, state):
        # 41e's _encoder_state has already added the PD bias. Replace, do not
        # stack it with the independent policy encoder bias.
        inverse = sensor_to_body_rotation(self.imu_rpy_rad).transpose(-1, -2)
        return replace(state,
            policy_position_rad=(state.policy_position_rad - self.encoder_bias_rad
                                 + self.policy_encoder_bias_rad),
            angular_velocity_body_rad_s=(inverse @ state.angular_velocity_body_rad_s[..., None]).squeeze(-1)
                                       + self.gyro_bias_rad_s,
            gravity_body=(inverse @ state.gravity_body[..., None]).squeeze(-1))

    def context_parameters(self):
        return {
            "nominal": self.nominal[:, None].float(), "scale": self.scale[:, None],
            "motor_strength": self.motor_strength,
            "pd_encoder_bias_rad": self.encoder_bias_rad,
            "policy_encoder_bias_rad": self.policy_encoder_bias_rad,
            "gyro_bias_rad_s": self.gyro_bias_rad_s,
            "imu_sensor_to_body_rotation": sensor_to_body_rotation(self.imu_rpy_rad).flatten(-2),
            **getattr(self, "material_context", {}),
        }

    def state_dict(self):
        return {"format": "requested_course_randomization_v1", "configuration": asdict(self.config),
                "seed": self.seed, "num_envs": self.num_envs,
                "rng": self.generator.get_state().clone(), "reset_rows_consumed": self.reset_rows_consumed}

    def restore_after_physical_reset(self, state):
        expected = self.state_dict()
        if isinstance(state, dict) and isinstance(state.get("configuration"), dict):
            # Checkpoints from before the disturbance intervals used their defaults.
            defaults = RequestedRandomizationConfig()
            state = {**state, "configuration": {
                "push_interval_s": defaults.push_interval_s, "wrench_interval_s": defaults.wrench_interval_s,
                **state["configuration"]}}
        if (not isinstance(state, dict) or state.keys() != expected.keys()
                or any(state[key] != expected[key] for key in ("format", "configuration", "seed", "num_envs"))
                or type(state["reset_rows_consumed"]) is not int or state["reset_rows_consumed"] < 0):
            raise ValueError("Requested randomization checkpoint identity differs")
        generator = torch.Generator(device=self.device)
        generator.set_state(state["rng"].cpu())
        self.generator, self.reset_rows_consumed = generator, state["reset_rows_consumed"]


def make_sensor_model_type(base_type, randomization):
    """Keep upstream clocks/caches; fixed transforms precede history insertion."""
    class RequestedSensorModel(base_type):
        noise_distribution = 1  # 0: legacy uniform halfwidth; 1: Gaussian SI sigma.

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.num_envs != randomization.num_envs or self.device != randomization.device:
                raise ValueError("Sensor randomization batch/device differs")
            if self.uplink_delay_s is not None:
                raise ValueError("Requested identified execution owns aggregate I/O; extra uplink is unsupported")
            c = randomization.config
            self._halfwidth = self._halfwidth.new_tensor(
                [c.encoder_noise_rad] * 4 + [0., 0.]
                + [c.leg_velocity_noise_rad_s] * 4 + [c.wheel_velocity_noise_rad_s] * 2
                + [c.gyro_noise_rad_s] * 3 + [c.gravity_noise] * 3)

        def reset(self, mask, now_s, *, difficulty=1.):
            config = self.config
            # A second independently drawn nominal mask would suppress delays
            # on some randomized lanes. Only the shared mask owns this choice.
            self.config = replace(config, nominal_episode_fraction=0.)
            try:
                super().reset(mask, now_s, difficulty=difficulty)
            finally:
                self.config = config
            self.nominal[mask] = randomization.nominal[mask]
            self.lags[self.nominal & mask] = 0

        def push(self, state, physical_tick, **kwargs):
            return super().push(randomization.transform_measurement(state), physical_tick, **kwargs)

        def seed_after_reset(self, mask, state, **kwargs):
            return super().seed_after_reset(mask, randomization.transform_measurement(state), **kwargs)

        def _read_rows(self, rows, generator):
            from transformer_rl.formal.control.observation import SensorSample

            lags = torch.minimum(self.lags[rows], (self.count[rows] - 1)[:, None])
            slots = (self.physical_tick - lags) % self.capacity
            values = self.values[slots[:, self._groups], rows[:, None], self._columns[None]]
            stamps = self.times[slots, rows[:, None]]
            noise = torch.randn(values.shape, generator=generator, device=self.device)
            values = values + noise * self._halfwidth * ~self.nominal[rows, None] * self.noise_scale[rows, None]
            gravity = values[:, 15:18]
            gravity = gravity / gravity.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            return SensorSample(values[:, :6], values[:, 6:12], values[:, 12:15], gravity, stamps)

    return RequestedSensorModel


def sample_mechanics(parameters, randomization, seed):
    """Named startup draws; one positive scalar preserves each inertia tensor."""
    names, c = parameters.body_names, randomization.config
    rng = torch.Generator(device="cpu").manual_seed(seed)
    shape = parameters.mass_kg.shape
    order = torch.tensor([sorted(names).index(name) for name in names])
    nominal = randomization.nominal.cpu()[:, None]
    mass_scale = 1 + (torch.rand(shape, generator=rng) * 2 - 1)[:, order] * c.mass_fraction
    independent = 1 + (torch.rand(shape, generator=rng) * 2 - 1)[:, order] * c.independent_inertia_fraction
    mass_scale[nominal.expand_as(mass_scale)] = 1
    independent[nominal.expand_as(independent)] = 1
    offset = (torch.rand(shape[0], 3, generator=rng) * 2 - 1) * torch.tensor(c.base_com_offset_m)
    offset[nominal[:, 0]] = 0
    mass_scale, independent = mass_scale.to(parameters.mass_kg), independent.to(parameters.mass_kg)
    com = parameters.com_pose_local_xyzw.clone()
    com[:, names.index("base_link"), :3] += offset.to(com)
    return (parameters.mass_kg * mass_scale, com,
            parameters.inertia_kg_m2 * (mass_scale * independent)[..., None], mass_scale, independent)


class NewtonMechanicsWriter:
    """Public Newton setters; COM translation in link axes, explicit property notification."""

    backend = "newton"

    def __init__(self, notify=None):
        self.notify = notify

    def write(self, robot, mass, com, inertia):
        robot.set_masses_index(masses=mass.contiguous(), env_ids=None, body_ids=None)
        robot.set_coms_index(coms=com[..., :3].contiguous(), env_ids=None, body_ids=None)
        robot.set_inertias_index(inertias=inertia.contiguous(), env_ids=None, body_ids=None)
        notify = self.notify
        if notify is None:
            from transformer_rl.formal.lab_env.newton_properties import notify_newton_properties
            notify = notify_newton_properties
        notify("BODY_INERTIAL_PROPERTIES")
        return read_articulation_mechanics(robot, backend="newton")

    @staticmethod
    def armature(robot, ids):
        from transformer_rl.formal.lab_env.newton_properties import native_tensor
        return native_tensor(robot.data.joint_armature)[:, ids]


class PhysxMechanicsWriter:
    """The pinned PhysX public-setter sequence: COM pose, then republish its final principal frame."""

    backend = "physx"

    @staticmethod
    def write(robot, mass, com, inertia):
        robot.set_masses_index(masses=mass.contiguous(), env_ids=None, body_ids=None, full_data=True)
        robot.set_coms_index(coms=com.contiguous(), env_ids=None, body_ids=None, full_data=True)
        robot.set_inertias_index(inertias=inertia.contiguous(), env_ids=None, body_ids=None, full_data=True)
        actual = read_articulation_mechanics(robot, backend="physx")
        # PhysX may reselect principal axes when writing inertia; publish its final COM pose.
        robot.set_coms_index(coms=actual.com_pose_local_xyzw.to(com.device).contiguous(),
                             env_ids=None, body_ids=None, full_data=True)
        return read_articulation_mechanics(robot, backend="physx")

    @staticmethod
    def armature(robot, ids):
        value = robot.data.joint_armature
        value = getattr(value, "torch", value)
        if not isinstance(value, torch.Tensor):
            import warp as wp
            value = wp.to_torch(value)
        return value[:, ids]


def make_startup_dynamics_type(base_type, randomization, execution=None, writer=None):
    writer = NewtonMechanicsWriter() if writer is None else writer

    class RequestedStartupDynamics(base_type):
        def __init__(self, robot, binding, num_envs, nominal_mass_kg, seed, config, **kwargs):
            # Authenticate/read nominal assets through the original implementation;
            # its historical draw is intentionally replaced by one requested draw.
            super().__init__(robot, binding, num_envs, nominal_mass_kg, seed, None, **kwargs)
            parameters = read_articulation_mechanics(robot, backend=writer.backend)
            mass, com, inertia, mass_scale, independent = sample_mechanics(parameters, randomization, seed)
            if "notify" in kwargs and writer.backend == "newton":
                writer.notify = kwargs["notify"]
            # PhysX getters return host tensors while public setters take simulation-device tensors.
            device = kwargs.get("device", mass.device)
            actual = writer.write(robot, mass.to(device), com.to(device), inertia.to(device))
            for label, observed, target in (("mass", actual.mass_kg, mass),
                    ("COM", actual.com_pose_local_xyzw[..., :3], com[..., :3]),
                    ("inertia", actual.inertia_kg_m2, inertia)):
                if not torch.allclose(observed, target.to(observed.device), rtol=1e-4, atol=1e-6):
                    raise RuntimeError("Requested native " + label + " readback differs")
            # Downstream consumers (reward composer, COM transforms) run on the simulation device.
            nominal_com = parameters.com_pose_local_xyzw.to(device)
            actual_mass, actual_com = actual.mass_kg.to(device), actual.com_pose_local_xyzw.to(device)
            self.body_names = parameters.body_names
            self.body_mass_scale, self.independent_inertia_scale = mass_scale.to(device), independent.to(device)
            self.body_inertia_scale = self.body_mass_scale * self.independent_inertia_scale
            self.mass_kg_by_body, self.com_local_m_by_body = actual_mass, actual_com[..., :3]
            self.mass_kg_by_environment = actual_mass.sum(-1)
            self.mass_scale = (self.mass_kg_by_environment / parameters.mass_kg.to(device).sum(-1))[:, None]
            base = self.body_names.index("base_link")
            self.nominal_base_com_local_m = nominal_com[:, base, :3]
            self.base_com_local_m = actual_com[:, base, :3]
            self.base_com_delta_local_m = self.base_com_local_m - self.nominal_base_com_local_m
            packed = torch.cat((actual.mass_kg.flatten(), actual.com_pose_local_xyzw.flatten(),
                                actual.inertia_kg_m2.flatten())).cpu().numpy().tobytes()
            configuration = randomization.config.provenance()
            self._receipt.update({"configuration": configuration,
                "configuration_sha256": hashlib.sha256(json.dumps(configuration, sort_keys=True,
                    separators=(",", ":")).encode()).hexdigest(),
                "nominal_rows": int(randomization.nominal.sum()), "randomization_active": True,
                "native_parameters_sha256": hashlib.sha256(packed).hexdigest(),
                "property_backend": writer.backend,
                "inertia_rule": "I=I0*mass_scale*independent_inertia_scale; one scalar per body",
                "mass_kg_min": float(self.mass_kg_by_environment.min()),
                "mass_kg_max": float(self.mass_kg_by_environment.max())})
            if execution is not None:
                wheels, joints = ("L_link3", "R_link3"), ("L_joint3", "R_joint3")
                scale = self.body_inertia_scale[:, [self.body_names.index(name) for name in wheels]]
                ids = [robot.joint_names.index(name) for name in joints]
                native = execution.native_parameters()
                armature = (scale * scale.new_tensor([native[name]["armature"] for name in joints])).to(device)
                robot.write_joint_armature_to_sim_index(armature=armature.contiguous(), joint_ids=ids, env_ids=None)
                readback = writer.armature(robot, ids)
                if not torch.allclose(readback, armature.to(readback), rtol=1e-6, atol=1e-8):
                    raise RuntimeError("Wheel armature did not use the body's single inertia draw")
                # PhysX property getters return host tensors; the execution chain runs on its own device.
                execution.set_wheel_inertia_scale(scale.to(getattr(execution, "device", scale.device)))
                self._receipt["wheel_armature_same_body_draw"] = True
                self._receipt["wheel_armature_kg_m2"] = readback.cpu().tolist()

    return RequestedStartupDynamics


def install_identified_feedback(controller, wheel_p=.5):
    """Preserve controller identity and validation; change only explicit P/DR math."""
    if getattr(controller, "_requested_feedback_installed", False):
        raise ValueError("Requested feedback cannot be installed twice")
    if type(wheel_p) not in (int, float) or not math.isfinite(wheel_p) or wheel_p <= 0:
        raise ValueError("Wheel velocity P must be explicit and positive")
    original = controller.effort

    @torch.no_grad()
    def effort(self, targets, feedback_position_rad, feedback_velocity_rad_s, pd_computed_at_s,
               *, strength=None, envelope=None, motor_velocity_rad_s=None, no_load_scale=None):
        if envelope is not None:
            raise ValueError("Identified wheel execution must own the only velocity envelope")
        checked = original(targets, feedback_position_rad, feedback_velocity_rad_s, pd_computed_at_s,
                           strength=None, envelope=None, motor_velocity_rad_s=motor_velocity_rad_s,
                           no_load_scale=no_load_scale)
        scale = torch.ones_like(checked.requested_effort) if strength is None else strength
        if (scale.shape != checked.requested_effort.shape or scale.device != checked.requested_effort.device
                or scale.dtype != checked.requested_effort.dtype
                or not torch.isfinite(scale).all() or (scale <= 0).any()):
            raise ValueError("Motor strength must contain finite positive six-axis gains")
        requested = checked.requested_effort.clone()
        requested[:, 4:] = wheel_p * (targets.wheel_velocity_rad_s - feedback_velocity_rad_s[:, 4:])
        bounds = checked.effort_bounds
        nominal = requested.clamp(min=-bounds, max=bounds)
        strengthened = nominal * scale
        capped = strengthened.clamp(min=-bounds, max=bounds)
        return replace(checked, requested_effort=strengthened, capped_effort=capped,
                       saturated=(requested != nominal) | (strengthened != capped))

    controller.effort = MethodType(effort, controller)
    controller.damping[4:] = wheel_p
    controller._requested_feedback_installed = True


def make_randomized_environment(base_type, randomization, execution, *, backend="newton"):
    """Scope native constructor substitutions, then retain only owned objects.

    Newton additionally maps the requested average contact friction onto Kamino's
    multiply rule. PhysX keeps its native wheel material reader (runtime
    ``wheel_material``) and combines friction natively.
    """
    if backend not in ("newton", "physx"):
        raise ValueError("Requested randomization backend must be newton or physx")

    class RequestedCourseEnvironment(base_type):
        def __init__(self, *args, **kwargs):
            from transformer_rl.formal.lab_env import dynamics, materials, env as native_env

            newton = backend == "newton"
            dynamics_owner, dynamics_name = ((dynamics, "NewtonStartupDynamics") if newton
                                             else (native_env, "LabStartupDynamics"))
            writer = NewtonMechanicsWriter() if newton else PhysxMechanicsWriter()

            @contextmanager
            def bindings():
                previous = (getattr(dynamics_owner, dynamics_name), native_env.EpisodeRandomization,
                            native_env.LabSensorModel, materials.NewtonWheelMaterialReader)
                setattr(dynamics_owner, dynamics_name,
                        make_startup_dynamics_type(previous[0], randomization, execution, writer))

                def episode(num_envs, config, *, seed, device):
                    if num_envs != randomization.num_envs or torch.device(device) != randomization.device:
                        raise ValueError("Native episode randomization batch/device differs")
                    return randomization

                native_env.EpisodeRandomization = episode
                native_env.LabSensorModel = make_sensor_model_type(previous[2], randomization)
                if newton:
                    materials.NewtonWheelMaterialReader = make_material_reader_type(previous[3], randomization)
                try:
                    yield
                finally:
                    setattr(dynamics_owner, dynamics_name, previous[0])
                    (native_env.EpisodeRandomization, native_env.LabSensorModel,
                     materials.NewtonWheelMaterialReader) = previous[1:]

            with bindings():
                super().__init__(*args, **kwargs)
            if (self._is_newton != newton or self._episode_randomization is not randomization
                    or self._sensor_model is None
                    or getattr(self._sensor_model, "noise_distribution", None) != 1
                    or not getattr(self._startup_dynamics, "_receipt", {}).get("randomization_active")):
                raise ValueError("Requested DR requires the selected backend's startup, episode and sensor paths")
            install_identified_feedback(self._control.controller)
            self._requested_randomization = randomization
            self.requested_randomization_receipt = randomization.config.provenance()

    return RequestedCourseEnvironment
