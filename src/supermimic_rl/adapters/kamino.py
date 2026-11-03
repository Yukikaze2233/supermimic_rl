"""Nominal, flat Newton/Kamino backend for the pinned Lab environment.

This module owns application startup and the small backend-dependent services.
The upstream controller, actuator response, spring wrenches, reset solver and
reward are reused. It deliberately does not emulate a PhysX tensor view.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata
import math
import re
import sys
from threading import RLock

import numpy as np
import torch

from .mechanics import read_articulation_mechanics


_CONSTRUCTION_LOCK = RLock()
_GROUND_PATH = "/World/supermimic_ground"


def _tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    proxy = getattr(value, "torch", None)
    if isinstance(proxy, torch.Tensor):
        return proxy
    import warp as wp

    return wp.to_torch(value)


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value.numpy()


@dataclass(frozen=True)
class KaminoSettings:
    padmm_iterations: int = 200
    robot_friction: float = 0.7
    ground_friction: float = 0.7
    max_contacts_per_world: int = 512
    integrator: str = "euler"
    use_cuda_graph: bool = False
    rotation_correction: str = "continuous"

    def __post_init__(self):
        for name in ("padmm_iterations", "max_contacts_per_world"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"Kamino {name} must be a positive integer")
        for name in ("robot_friction", "ground_friction"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Kamino {name} must be finite and nonnegative")
        if self.integrator not in ("euler", "moreau") or type(self.use_cuda_graph) is not bool:
            raise ValueError("Invalid Kamino integrator or graph setting")
        if self.rotation_correction != "continuous":
            raise ValueError("The pinned encoder requires Kamino continuous coordinates; twopi wraps by 2*pi")


@contextmanager
def _module_bindings(module, replacements):
    """Scope the pinned environment's constructor-only module dependencies."""
    with _CONSTRUCTION_LOCK:
        original = {name: getattr(module, name) for name in replacements}
        try:
            for name, value in replacements.items():
                setattr(module, name, value)
            yield
        finally:
            for name, value in original.items():
                setattr(module, name, value)


def _nominal_configuration(config):
    if config is not None and getattr(config, "nominal_fraction", None) != 1.0:
        raise ValueError("Kamino currently supports nominal startup mechanics only")


def make_nominal_dynamics_type(base_type):
    """Inherit upstream COM transformation/receipt behavior without its writes."""

    class NewtonNominalDynamics(base_type):
        def __init__(self, robot, binding, num_envs, nominal_mass_kg, seed, config, *,
                     device, **kwargs):
            _nominal_configuration(config)
            parameters = read_articulation_mechanics(robot, backend="newton")
            if (parameters.body_names != tuple(binding.native_body_names)
                    or parameters.mass_kg.shape[0] != num_envs
                    or type(nominal_mass_kg) not in (int, float)
                    or not math.isfinite(nominal_mass_kg) or nominal_mass_kg <= 0):
                raise ValueError("Newton nominal mechanics differ from the bound core rows")
            mass = parameters.mass_kg.to(device)
            com = parameters.com_pose_local_xyzw[..., :3].to(device)
            inertia = parameters.inertia_kg_m2.to(device)
            total = mass.sum(-1)
            if not torch.allclose(total, torch.full_like(total, nominal_mass_kg), atol=1e-5, rtol=0):
                raise ValueError("Newton native mass differs from the selected core asset")
            self.mass_kg_by_body = mass
            self.com_local_m_by_body = com
            self.inertia_local_kg_m2_by_body = inertia
            self.mass_kg_by_environment = total
            self.mass_scale = torch.ones(num_envs, 1, device=device)
            index = parameters.body_names.index("base_link")
            self.base_com_local_m = com[:, index].clone()
            self.nominal_base_com_local_m = self.base_com_local_m.clone()
            self.base_com_delta_local_m = torch.zeros_like(self.base_com_local_m)
            packed = torch.cat((mass.flatten(), com.flatten(), inertia.flatten())).cpu().numpy().tobytes()
            self._receipt = {
                "status": "newton_native_nominal_readback", "seed": seed,
                "randomization_active": False, "property_writes_performed": False,
                "body_order": list(parameters.body_names), "num_envs": num_envs,
                "mass_kg_min": float(total.min()), "mass_kg_max": float(total.max()),
                "source": parameters.source, "com_frame": parameters.com_frame,
                "inertia_frame": parameters.inertia_frame,
                "native_parameters_sha256": hashlib.sha256(packed).hexdigest(),
                "physical_qualification": False,
            }

    return NewtonNominalDynamics


class NewtonWorldBinding:
    """Verify Newton world ownership instead of invoking PhysX GA isolation."""

    def __init__(self, count, manager):
        self.count, self.manager = count, manager
        self._receipt = {"method": "newton_native_worlds", "world_count": count,
                         "physx_pair_allocator_used": False, "verified": False}

    @property
    def receipt(self):
        return deepcopy(self._receipt)

    def confirm_shared_ground_groups(self, groups):
        self._receipt["ground_binding"] = deepcopy(groups)

    def finish_warmup(self):
        model = self.manager.backend.model
        if model.world_count != self.count or model.articulation_count != self.count:
            raise ValueError("Kamino requires one full articulation in each native world")
        start, end = _numpy(model.articulation_start), _numpy(model.articulation_end)
        loops = start[1:] - end
        if loops.shape != (self.count,) or not np.all(loops == 6):
            raise ValueError("Kamino must preserve all six loop-closing joints in every world")
        self._receipt.update(verified=True, loop_joints_per_world=loops.tolist())


class NewtonNominalWheelMaterial:
    """Read one Coulomb coefficient without pretending it has two regimes."""

    def __init__(self, robot, wheel_body_indices, *, num_envs, device, manager):
        self.names = [robot.body_names[int(i)] for i in wheel_body_indices]
        if len(self.names) != 2 or len(set(self.names)) != 2:
            raise ValueError("Newton wheel material requires two distinct wheel bodies")
        self.num_envs, self.device, self.manager = num_envs, device, manager
        self._values = torch.empty(num_envs, 6, device=device)
        self._receipt = {}
        self.refresh()

    def randomize_startup(self, *args, **kwargs):
        raise ValueError("Kamino nominal backend does not support wheel material randomization")

    def refresh(self):
        model = self.manager.backend.model
        if model.world_count != self.num_envs:
            raise ValueError("Newton material worlds differ from the environment count")
        shape_body = _numpy(model.shape_body)
        body_world = _numpy(model.body_world)
        mu = _numpy(model.shape_material_mu)
        restitution = _numpy(model.shape_material_restitution)
        if (not np.isfinite(mu).all() or not np.isfinite(restitution).all()
                or (mu < 0).any() or (restitution < 0).any() or (restitution > 1).any()):
            raise ValueError("Newton native wheel material contains invalid coefficients")
        labels = [label.rsplit("/", 1)[-1] for label in model.body_label]
        values = np.empty((self.num_envs, 6), dtype=np.float32)
        for wheel, name in enumerate(self.names):
            for world in range(self.num_envs):
                bodies = [i for i, label in enumerate(labels) if label == name and body_world[i] == world]
                if len(bodies) != 1:
                    raise ValueError("Newton wheel ownership must be unique in each native world")
                selected = shape_body == bodies[0]
                if (not selected.any() or np.ptp(mu[selected]) > 1e-6
                        or np.ptp(restitution[selected]) > 1e-6):
                    raise ValueError("All collision shapes on a wheel must share one nominal material")
                friction, bounce = float(mu[selected][0]), float(restitution[selected][0])
                values[world, 3 * wheel:3 * wheel + 3] = (friction, friction, bounce)
        self._values.copy_(torch.as_tensor(values, device=self.device))
        self._receipt = {
            "source": "Newton shape_material_mu/restitution, body_world ownership",
            "wheel_body_names": self.names, "values": values.tolist(),
            "static_dynamic_scope": "one Coulomb mu repeated in legacy static/dynamic columns",
        }

    def features(self):
        return self._values

    @property
    def receipt(self):
        return deepcopy(self._receipt)


class NewtonFlatContactLoads:
    """Reduce filtered plane-normal forces in the projected core body order."""

    def __init__(self, sensor, binding, environment_paths, *, device, **kwargs):
        self.sensor, self.num_envs = sensor, len(environment_paths)
        self.device = torch.device(device)
        names, core = tuple(sensor.body_names), tuple(binding.native_body_names)
        if len(set(names)) != len(names) or len(set(core)) != len(core) or not set(core).issubset(names):
            raise ValueError("Newton plane contact names do not contain the complete core")
        self.indices = torch.tensor([names.index(name) for name in core], device=self.device)
        self.wheels = torch.tensor(binding.wheel_body_indices, device=self.device)
        self.normal = torch.zeros(self.num_envs, len(core), device=self.device)
        self.upward = torch.zeros_like(self.normal)
        self.valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._tick = None
        self.receipt = {"source": "Newton ContactSensor.normal_force_matrix_w",
                        "body_order": list(core), "scope": "one horizontal plane only"}

    def reset_rows(self, ids, physical_tick):
        self.normal[ids] = self.upward[ids] = 0
        self.valid[ids] = False
        self._tick = physical_tick

    def read(self, physical_tick, physics_dt_s):
        if self._tick != physical_tick:
            packet = self.sensor.data.normal_force_matrix_w
            if packet is None:
                raise ValueError("Newton filtered plane-normal forces are unavailable")
            force = _tensor(packet)
            if (force.ndim != 4 or force.shape[:2] != (self.num_envs, len(self.sensor.body_names))
                    or force.shape[2] < 1 or force.shape[-1] != 3 or not torch.isfinite(force).all()):
                raise ValueError("Newton filtered contact packet has invalid environment/body/partner axes")
            force = force.sum(-2).index_select(1, self.indices)
            self.normal.copy_(force.norm(dim=-1))
            self.upward.copy_(force[..., 2].clamp_min(0))
            self.valid.fill_(True)
            self._tick = physical_tick
        return self.normal, self.upward.index_select(1, self.wheels)


class KaminoHealth:
    """Reject nonfinite native solver diagnostics; retain nonconvergence honestly."""

    def __init__(self, manager):
        self.manager = manager
        self.latest = None
        self.closed = False
        self._summary = {
            "sampling_scope": "factory/reset/policy-boundary reads; repeated states may be counted; not all substeps",
            "checks": 0, "solver_world_samples": 0, "converged_world_samples": 0,
            "max_iterations": 0.0, "max_r_p": 0.0, "max_r_d": 0.0, "max_r_c": 0.0,
        }

    @property
    def summary(self):
        result = deepcopy(self._summary)
        count = result["solver_world_samples"]
        result["converged_fraction"] = result["converged_world_samples"] / count if count else None
        return result

    def check(self):
        if self.closed:
            raise RuntimeError("Kamino health monitor is closed")
        status = self.manager._solver.status.numpy()
        required = {"iterations", "r_p", "r_d", "r_c", "converged"}
        if not status.size or not required.issubset(status.dtype.names or ()):
            raise ValueError("Kamino solver status lacks its native convergence diagnostics")
        if any(not np.isfinite(status[name]).all() for name in required):
            raise FloatingPointError("Kamino solver status contains nonfinite values")
        self.latest = {"converged_fraction": float(status["converged"].mean()),
                       **{name + "_max": float(status[name].max()) for name in required - {"converged"}}}
        self._summary["checks"] += 1
        self._summary["solver_world_samples"] += int(status.size)
        self._summary["converged_world_samples"] += int(status["converged"].sum())
        for name in ("iterations", "r_p", "r_d", "r_c"):
            key = "max_" + name
            self._summary[key] = max(self._summary[key], float(status[name].max()))
        # PADMM stopping criteria are not geometric closure tolerances. A false
        # flag is published, not relabeled as a simulation crash or convergence.

    def close(self):
        self.closed = True


class KaminoBackend:
    """Own a Newton launch scope and nominal adapters for one flat worker."""

    declarative_scene = True

    def __init__(self, device, configuration=None):
        self.settings = KaminoSettings(**({} if configuration is None else configuration))
        self.device = device
        self._closed = False
        self._actuator_receipt = {}
        self._readback = None
        from isaaclab import app
        from isaaclab_newton.physics import (
            KaminoMaterialsCfg, KaminoPADMMCfg, KaminoPADMMSolverCfg,
            NewtonCfg, NewtonManager, NewtonShapeCfg,
        )

        self.manager = NewtonManager
        self.physics_configuration = NewtonCfg(
            solver_cfg=KaminoPADMMSolverCfg(
                integrator=self.settings.integrator, use_fk_solver=True,
                use_collision_detector=False,
                # Native CPU checks show TWOPI jumps by 2*pi despite its
                # interval documentation. CONTINUOUS preserves small signed
                # increments, so the unchanged upstream tracker adds no turns.
                rotation_correction=self.settings.rotation_correction,
                dynamics_solver_cfg=KaminoPADMMCfg(max_iterations=self.settings.padmm_iterations),
                materials=KaminoMaterialsCfg(friction_mix_mode="multiply", restitution_mix_mode="multiply"),
                collect_solver_info=False, compute_solution_metrics=False,
                max_contacts_per_world=self.settings.max_contacts_per_world),
            num_substeps=1, use_cuda_graph=self.settings.use_cuda_graph, load_visual_shapes=False,
            default_shape_cfg=NewtonShapeCfg(mu=self.settings.robot_friction, margin=0.0, gap=0.01))
        previous = sys.argv
        try:
            sys.argv = [previous[0]]
            self._scope = app.launch_simulation(
                self.physics_configuration, dict(headless=True, device=device, livestream=0))
            self.physics = self._scope.__enter__()
        finally:
            sys.argv = previous
        self.health = KaminoHealth(self.manager)

    def configure_settings(self):
        """Configuration was passed to launch_simulation; no Kit/PhysX writes."""
        if self._closed:
            raise RuntimeError("Kamino launch scope is closed")

    def prepare_scene(self, cfg):
        if (not cfg.declarative_scene or getattr(cfg, "course_manifest", None) is not None
                or getattr(cfg, "static_plane_height_m", None) != 0.0
                or cfg.height_scanner is not None or cfg.terrain_scanner is not None or cfg.forward_scanners):
            raise ValueError("Kamino backend currently requires the declarative analytical flat plane")
        _nominal_configuration(getattr(cfg, "startup_dynamics_spec", None))
        if getattr(cfg, "wheel_material_spec", None) is not None:
            raise ValueError("Kamino backend requires wheel_material=None")
        randomization = getattr(cfg, "episode_randomization_spec", None)
        if randomization is not None and randomization.enabled:
            raise ValueError("Kamino backend requires episode randomization disabled")
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.sim import GroundPlaneCfg, RigidBodyMaterialCfg

        cfg.sim.physics = self.physics_configuration
        cfg.sim.use_newton_actuators = False
        # Geometric anchor separation complements PADMM residual diagnostics.
        # The upstream observer samples native core link poses each substep.
        cfg.loop_telemetry_enabled = True
        ground = AssetBaseCfg(
            prim_path=_GROUND_PATH, collision_group=-1,
            spawn=GroundPlaneCfg(physics_material=RigidBodyMaterialCfg(
                static_friction=self.settings.ground_friction, dynamic_friction=self.settings.ground_friction,
                restitution=0.0, friction_combine_mode="multiply", restitution_combine_mode="multiply")))
        object.__setattr__(cfg.scene, "supermimic_ground", ground)
        self._world_binding = NewtonWorldBinding(cfg.scene.num_envs, self.manager)
        object.__setattr__(cfg, "_native_clone_binding", self._world_binding)
        # Newton supports armature, Coulomb friction and viscous damping, but
        # has no separate dynamic-friction coefficient. Record that omission
        # explicitly instead of allowing an unreported SDK fallback.
        robots = (("robot_cfg", cfg.robot_cfg), ("scene.robot", cfg.scene.robot))
        for label, robot in robots:
            records = {}
            for name, actuator in robot.actuators.items():
                records[name] = {field: deepcopy(getattr(actuator, field, None)) for field in (
                    "armature", "friction", "dynamic_friction", "viscous_friction", "damping")}
            self._actuator_receipt[label] = records
        for _, robot in robots:
            for actuator in robot.actuators.values():
                value = getattr(actuator, "dynamic_friction", None)
                if value is not None:
                    actuator.dynamic_friction = ({key: 0.0 for key in value}
                                                 if isinstance(value, dict) else 0.0)

    def _contact_sensor_type(self):
        from isaaclab_newton.sensors.contact_sensor.contact_sensor import ContactSensor

        manager = self.manager

        class PlaneContactSensor(ContactSensor):
            def _initialize_impl(self):
                model = manager.backend.model
                ids = np.flatnonzero(_numpy(model.shape_body) < 0)
                names = [model.shape_label[int(index)] for index in ids]
                if not names or any(not name.startswith(_GROUND_PATH + "/") for name in names):
                    raise ValueError("Kamino flat backend requires only its declared static plane")
                self.kamino_contact_partner_receipt = {
                    "shape_ids": ids.tolist(), "shape_labels": names,
                    "scope": "static horizontal plane; robot self-contacts excluded"}
                self.cfg.filter_shape_prim_expr = [re.escape(name) for name in names]
                self.cfg.filter_prim_paths_expr = []
                super()._initialize_impl()

        return PlaneContactSensor

    def environment_type(self, base_type):
        """Wrap the core class before make_full_vehicle_env adds auxiliary axes."""
        from transformer_rl.formal.lab_env import env as env_module

        backend = self
        dynamics_type = make_nominal_dynamics_type(env_module.LabStartupDynamics)
        sensor_type = self._contact_sensor_type()

        def wheel_material(robot, indices, **kwargs):
            return NewtonNominalWheelMaterial(robot, indices, manager=backend.manager, **kwargs)

        def ground_binding(scene):
            if len(scene.env_prim_paths) != backend._world_binding.count:
                raise ValueError("Newton scene world count differs from its clone plan")
            return {"method": "newton_native_worlds", "global_paths": [_GROUND_PATH],
                    "physx_collision_groups_used": False}

        class KaminoEnvironment(base_type):
            mechanics_backend = "newton"

            def __init__(self, *args, **kwargs):
                with _module_bindings(env_module, {
                    "WheelMaterialReader": wheel_material, "ContactSensor": sensor_type,
                    "author_shared_ground_groups": ground_binding,
                }):
                    super().__init__(*args, **kwargs)
                self._contact_loads = NewtonFlatContactLoads(
                    self.contact_sensor, self._binding, self.scene.env_prim_paths, device=self.device)

            def _build_startup_dynamics(self, *args, **kwargs):
                return dynamics_type(*args, **kwargs)

        return KaminoEnvironment

    def finalize_environment(self, env):
        """Publish stable backend identity separately from sampled native state."""
        from newton import ModelFlags

        model = self.manager.backend.model
        self._world_binding.finish_warmup()
        shape_body = _numpy(model.shape_body)
        robot_shapes = shape_body >= 0
        mu = np.where(robot_shapes, self.settings.robot_friction, self.settings.ground_friction).astype(np.float32)
        model.shape_material_mu.assign(mu)
        model.shape_material_restitution.zero_()
        self.manager._solver.notify_model_changed(ModelFlags.SHAPE_PROPERTIES)
        if not np.array_equal(_numpy(model.shape_material_mu), mu):
            raise RuntimeError("Kamino nominal material write differs from native readback")
        env._wheel_materials.refresh()
        full = getattr(env, "_full_robot", env.robot)
        mechanics = read_articulation_mechanics(full, backend="newton")
        mechanical_receipt = {}
        for name in ("mass_kg", "com_pose_local_xyzw", "inertia_kg_m2"):
            value = getattr(mechanics, name).cpu().numpy()
            mechanical_receipt[name] = {
                "row0": value[0].tolist(), "minimum_by_body": value.min(0).tolist(),
                "maximum_by_body": value.max(0).tolist(),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
        native_parameters = {}
        for name in ("joint_armature", "joint_damping", "joint_friction"):
            value = _numpy(getattr(model, name))
            if not np.isfinite(value).all() or value.size % env.num_envs:
                raise ValueError("Kamino native joint parameter layout is inconsistent: " + name)
            rows = value.reshape(env.num_envs, -1)
            native_parameters[name] = {"row0": rows[0].tolist(), "minimum": float(value.min()),
                                       "maximum": float(value.max()), "shape": list(value.shape),
                                       "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
        versions = {}
        for package in ("isaaclab", "isaaclab_newton", "newton", "warp-lang", "torch"):
            try:
                versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                versions[package] = "unknown"
        self.health.check()
        self._readback = {
            "world_binding": self._world_binding.receipt,
            "body_count": model.body_count, "joint_count_including_root_and_loops": model.joint_count,
            "body_order": list(mechanics.body_names), "mechanics": mechanical_receipt,
            "mechanics_source": mechanics.source, "com_frame": mechanics.com_frame,
            "inertia_frame": mechanics.inertia_frame, "native_joint_parameters": native_parameters,
            "wheel_material": env._wheel_materials.receipt,
            "contact_partners": deepcopy(env.contact_sensor.kamino_contact_partner_receipt),
            "solver_status": deepcopy(self.health.latest),
        }
        return {"identity": {
            "backend": "newton_kamino", "settings": asdict(self.settings), "sdk_versions": versions,
            "loop_telemetry_enabled": True,
            "joint_parameter_requests": deepcopy(self._actuator_receipt),
            "joint_dynamic_friction_semantics": "unsupported; explicit zero override, original request recorded",
            "friction_semantics": "single Coulomb mu; multiply robot/plane; restitution zero",
            "reset_semantics": "Kamino FK from joint coordinates, six loops preserved",
            "continuous_joint_coordinates": "continuous native coordinates; upstream tracker receives small increments without wrap correction",
            "external_wrench_semantics": "upstream spring forces via native WrenchComposer, once per substep",
            "normal_load_semantics": "filtered horizontal-plane normal force; flat only",
            "physics_health": "finite native solver diagnostics; nonconvergence retained as diagnostic",
        }, "readback": deepcopy(self._readback)}

    def close(self):
        if not self._closed:
            self._closed = True
            self.health.close()
            self._scope.__exit__(None, None, None)
