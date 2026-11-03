"""Explicit lazy construction of the pinned upstream flat Isaac environment.

Only ``create_isaac`` starts PhysX/Kit or the selected Newton/Kamino launcher.
Importing this module and validating pins do not start Isaac. Native execution
uses the already installed upstream Isaac Python environment.
"""

from __future__ import annotations

import hashlib
import importlib
from importlib import metadata
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import traceback

from .isaac import IsaacAdapter


def _validate_training_observer(sensors, sections):
    """The P0 learner observes clean35, not the upstream delayed/noisy sensor."""
    episode = sections["episode_randomization"]
    if episode is not None and episode.enabled:
        raise ValueError(
            "enabled episode_randomization is unsupported: flat resets have no task_sampler, "
            "so update=0 keeps exposure=0; support requires an explicit fixed-exposure contract")
    widths = ("leg_position_halfwidth_rad", "leg_velocity_halfwidth_rad_s",
              "wheel_velocity_halfwidth_rad_s", "gyro_halfwidth_rad_s", "gravity_halfwidth")
    if any(getattr(sensors, name) != 0 for name in widths):
        raise ValueError("P0 instantaneous_noiseless_clean35 requires every sensor halfwidth to equal zero")
    if any(delay != 0 for delay in sensors.observation_delays_s):
        raise ValueError("P0 instantaneous_noiseless_clean35 requires all observation_delays_s to equal zero")
    communication = sections["communication"]
    if communication is not None and communication.enabled:
        raise ValueError("P0 instantaneous_noiseless_clean35 requires communication.enabled=False")


def _validate_episode_commands(configuration):
    if configuration is None:
        return None
    required = {"forward_m_s", "yaw_rad_s", "height_m", "standing_fraction"}
    if not isinstance(configuration, dict) or set(configuration) != required:
        raise ValueError("episode_commands requires explicit forward/yaw/height ranges and standing_fraction")
    for name in ("forward_m_s", "yaw_rad_s", "height_m"):
        values = configuration[name]
        if (not isinstance(values, (list, tuple)) or len(values) != 2
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)
                or values[0] > values[1] or name == "height_m" and values[0] <= 0):
            raise ValueError("episode command range must be finite and ordered: " + name)
    fraction = configuration["standing_fraction"]
    if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("standing_fraction must lie in [0, 1]")
    return {key: (list(value) if isinstance(value, (list, tuple)) else value)
            for key, value in configuration.items()}


class _EpisodeCommandSampler:
    """Use the native reset-only command hook, with an independent RNG stream."""

    def __init__(self, configuration, *, seed, device):
        import torch

        self.configuration = _validate_episode_commands(configuration)
        if self.configuration is None:
            raise ValueError("an episode command sampler requires explicit ranges")
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        ranges = [self.configuration[name] for name in ("forward_m_s", "yaw_rad_s", "height_m")]
        self.low = torch.tensor([values[0] for values in ranges], dtype=torch.float32, device=self.device)
        self.high = torch.tensor([values[1] for values in ranges], dtype=torch.float32, device=self.device)

    def __call__(self, ids):
        import torch

        if not isinstance(ids, torch.Tensor) or ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("native reset command sampling requires integer row IDs")
        values = self.low + torch.rand((len(ids), 3), generator=self.generator, device=self.device) * (self.high - self.low)
        standing = torch.rand(len(ids), generator=self.generator, device=self.device) < self.configuration["standing_fraction"]
        values[standing, :2] = 0
        # vy is the explicit unsupported lateral command, not missing data.
        return torch.stack((values[:, 0], torch.zeros_like(values[:, 0]), values[:, 1], values[:, 2]), dim=-1)


class _OwnedIsaacAdapter(IsaacAdapter):
    def __init__(self, env, launcher, health, provenance):
        self._compact_state = provenance.get("compact_state", False)
        super().__init__(env)
        self._launcher = launcher
        self._health = health
        self.provenance = provenance
        self._closed = False
        self._native_policy_steps = 0
        self._compact_projector = None
        if provenance.get("compact_state", False):
            from supermimic_rl.compact_state import CompactStateProjector
            self._compact_projector = CompactStateProjector(self.body_names, self.joint_names)

    def _read_static_context(self):
        context = super()._read_static_context()
        if self._compact_state:
            context["policy_joint_indices"] = context["controller.nominal"].new_tensor(
                self.env._binding.policy_joint_indices)[None].expand(self.num_envs, -1).clone()
        native = getattr(self.env, "_full_robot", None)
        if native is not None:
            from .mechanics import read_articulation_mechanics

            properties = read_articulation_mechanics(
                native, backend=getattr(self.env, "mechanics_backend", "physx"),
                body_names=self.env._gimbal_model["body_names"],
                tensor_converter=self._native_tensor)
            for name, value in (("gimbal.body_mass_kg", properties.mass_kg),
                                ("gimbal.body_com_pose_local_xyzw", properties.com_pose_local_xyzw),
                                ("gimbal.body_inertia_kg_m2", properties.inertia_kg_m2)):
                context[name] = self._owned(name, value)
            controller = self.env._gimbal
            gains = [controller.config.yaw_damping, controller.config.pitch_stiffness,
                     controller.config.pitch_damping]
            context["gimbal.servo_gains"] = controller.command.new_tensor(gains)[None].expand(self.num_envs, -1).clone()
            context["gimbal.effort_limits_nm"] = controller.effort_limits[None].expand(self.num_envs, -1).clone()
            context["gimbal.control_joint_signs"] = controller.signs[None].expand(self.num_envs, -1).clone()
        return context

    def _snapshot(self, *, terminal):
        snapshot = super()._snapshot(terminal=terminal)
        native = getattr(self.env, "_full_robot", None)
        if native is not None:
            ids = [list(native.joint_names).index(name) for name in self.env._gimbal_model["joint_names"]]
            for name, source in (("gimbal.position_rad", native.data.joint_pos),
                                 ("gimbal.velocity_rad_s", native.data.joint_vel)):
                snapshot.state[name] = self._owned(name, source)[:, ids].clone()
            snapshot.state["gimbal.effort_nm"] = self._owned("gimbal effort", self.env._gimbal.last_effort)
            # Fixed within an episode; context changes only at a real reset.
            snapshot.context["gimbal.command"] = self._owned("gimbal command", self.env._gimbal.command)
            for name in ("gimbal.position_rad", "gimbal.velocity_rad_s", "gimbal.effort_nm"):
                snapshot.field_metadata["state"][name] = {
                    "units": {"gimbal.position_rad": "rad", "gimbal.velocity_rad_s": "rad/s",
                              "gimbal.effort_nm": "N*m"}[name], "frame": "native yaw,pitch joint axes"}
            snapshot.field_metadata["context"]["gimbal.command"] = {
                "units": "yaw:rad/s; pitch:rad", "frame": "independent gimbal control coordinates"}
        return self._compact_projector.project(snapshot) if self._compact_projector is not None else snapshot

    def reset(self):
        self._health.check()
        result = super().reset()
        try:
            self._health.check()
        except Exception:
            self.current = None
            raise
        return result

    def step(self, action):
        self._health.check()
        result = super().step(action)
        try:
            self._health.check()
        except Exception:
            self.current = None
            raise
        self._native_policy_steps += 1
        if self._native_policy_steps % 256 == 0:
            self._write_native_diagnostics()
        return result

    def _write_native_diagnostics(self):
        if not hasattr(self._health, "summary"):
            return
        reports = {"kamino-health.json": self._health.summary}
        telemetry = getattr(self.env, "_loop_telemetry", None)
        if telemetry is not None and telemetry.enabled:
            reports["kamino-loop-geometry.json"] = {
                "policy_steps": self._native_policy_steps, **telemetry.summary()}
        for name, value in reports.items():
            path = self._native_log_directory / name
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
            temporary.replace(path)

    def close(self):
        """Release the native environment and the application created here."""
        if self._closed:
            return
        self._closed = True
        self.current = None
        try:
            try:
                self._write_native_diagnostics()
            finally:
                self.env.close()
        finally:
            try:
                self._health.close()
            finally:
                self._launcher.close()


def _verified_inputs(configuration):
    required = {
        "upstream_root", "expected_commit", "runtime_json", "runtime_sha256",
        "asset_directory", "asset_manifest_sha256", "num_envs", "device",
        "native_log_directory",
    }
    optional = {"seed", "formal_reward", "normal_load_source", "terrain_profile", "episode_commands",
                "gimbal", "compact_state", "physics_backend", "kamino"}
    if not isinstance(configuration, dict) or not required <= set(configuration) <= required | optional:
        raise ValueError("factory configuration requires exact source/runtime/asset pins and native batch/device")
    for field, size in (("expected_commit", 40), ("runtime_sha256", 64), ("asset_manifest_sha256", 64)):
        if not isinstance(configuration[field], str) or not re.fullmatch(f"[0-9a-f]{{{size}}}", configuration[field]):
            raise ValueError("factory pin is not a complete lowercase digest: " + field)
    root = Path(configuration["upstream_root"]).expanduser().resolve(strict=True)
    source = root / "src"
    if not (source / "transformer_rl" / "formal" / "lab_env" / "env.py").is_file():
        raise ValueError("upstream_root must be the existing Transformer source repository")
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True, text=True,
                              capture_output=True, timeout=15).stdout.strip()
    if git("rev-parse", "HEAD") != configuration["expected_commit"]:
        raise ValueError("upstream commit does not match the explicit expected_commit")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("upstream source has local changes; a clean pinned checkout is required")
    runtime_path = Path(configuration["runtime_json"]).expanduser().resolve(strict=True)
    raw = runtime_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != configuration["runtime_sha256"]:
        raise ValueError("runtime JSON bytes do not match runtime_sha256")
    runtime = json.loads(raw)
    if not isinstance(runtime, dict):
        raise ValueError("runtime JSON must be a mapping")
    log_directory = Path(configuration["native_log_directory"]).expanduser().resolve(strict=True)
    if not log_directory.is_dir():
        raise ValueError("native_log_directory must be an existing run directory")
    asset = Path(configuration["asset_directory"]).expanduser().resolve(strict=True)
    if hashlib.sha256((asset / "manifest.json").read_bytes()).hexdigest() != configuration["asset_manifest_sha256"]:
        raise ValueError("asset manifest bytes do not match asset_manifest_sha256")
    if type(configuration["num_envs"]) is not int or configuration["num_envs"] < 1:
        raise ValueError("num_envs must be positive")
    if not isinstance(configuration["device"], str) or not re.fullmatch(r"cuda:\d+", configuration["device"]):
        raise ValueError("the native factory requires an explicit CUDA device")
    if type(configuration.get("seed", 0)) is not int or not 0 <= configuration.get("seed", 0) < 2**63:
        raise ValueError("seed must be a nonnegative signed-64-bit integer")
    _validate_episode_commands(configuration.get("episode_commands"))
    if type(configuration.get("compact_state", False)) is not bool:
        raise ValueError("compact_state must be boolean")
    backend = configuration.get("physics_backend", "physx")
    if backend not in ("physx", "kamino"):
        raise ValueError("physics_backend must be physx or kamino")
    if backend != "kamino" and "kamino" in configuration:
        raise ValueError("kamino solver settings require physics_backend=kamino")
    if configuration.get("gimbal") is not None:
        from supermimic_rl.gimbal import GimbalConfig
        GimbalConfig(**configuration["gimbal"])
    if configuration.get("terrain_profile", "plane") != "plane":
        raise ValueError("this adapter supports only the explicit plane profile")
    if type(configuration.get("formal_reward", True)) is not bool:
        raise ValueError("formal_reward must be explicit boolean")
    contact_source = configuration.get("normal_load_source", "contact_force_world_z_abs")
    if contact_source not in ("contact_force_world_z_abs", "contact_force_norm"):
        raise ValueError("native patch loads require the unsupported course adapter; choose an explicit flat source")
    for name, module in tuple(sys.modules.items()):
        if name == "transformer_rl" or name.startswith("transformer_rl."):
            path = getattr(module, "__file__", None)
            if path is not None and not Path(path).resolve().is_relative_to(source):
                raise ValueError("a different Transformer checkout is already imported")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    importlib.invalidate_caches()
    return root, asset, runtime, contact_source


def create_isaac(configuration: dict) -> IsaacAdapter:
    """Validate frozen inputs, then instantiate the original flat native scene.

    Required keys are upstream_root, expected_commit (40 hex), runtime_json,
    runtime_sha256, asset_directory, asset_manifest_sha256, num_envs, device and
    native_log_directory. Redirect this worker's stdout and stderr to dedicated
    files under native_log_directory so upstream NativePhysicsHealth can reject
    dropped-contact/PhysX-error intervals before they enter replay.
    Optional keys: seed=0, formal_reward=True, terrain_profile='plane', and
    normal_load_source='contact_force_world_z_abs'. Optional episode_commands
    must explicitly bind forward_m_s, yaw_rad_s and height_m [min,max] ranges
    and standing_fraction. Samples remain constant for the full episode;
    standing sets forward/yaw to zero while retaining its sampled height.
    Omitting this section keeps the original nominal standing command.
    physics_backend='kamino' selects the nominal flat Newton path, configured
    by factory.kamino; its runtime.physics must be empty and wheel_material=None.
    The runtime uses the
    existing managed/diagnostic section definitions, including a verified
    response_source_pin, task, termination_limits, sensors, dynamics and
    physics capacities. Unsupported course/command-pool/birth settings and
    episode_randomization must be disabled in a separately pinned runtime.
    P0 requires every sensor halfwidth and observation delay to be zero and
    communication.enabled=False. This is the instantaneous noiseless clean35
    contract; actuator response and action transport/latch remain supported.

    Formal rewards come directly from upstream locomotion_v4. Flat normal
    loads use its existing diagnostic producer, not the course patch-load
    producer. ``provenance`` records this distinction and missing terms.
    """
    root, asset_directory, runtime, contact_source = _verified_inputs(configuration)
    from transformer_rl.common.training_control import verify_pin
    from transformer_rl.formal.control.actuation import PairTargetDomain, ResponseParameters
    from transformer_rl.formal.control.sensors import SensorModelConfig
    from transformer_rl.formal.simulation.dynamics import StartupDynamicsConfig
    from transformer_rl.formal.lab_env.action_transport import ActionTransportConfig
    from transformer_rl.formal.lab_env.contract import FlatTaskSpec
    from transformer_rl.formal.lab_env.formal_reward import FormalRewardConfig
    from transformer_rl.formal.lab_env.sections import optional_sections, termination_limits
    from transformer_rl.formal.lab_env.settings import load_asset_description
    from transformer_rl.formal.lab_env.sdk import LabSimulation
    from transformer_rl.formal.tasks.reward import REWARD_VERSION
    from transformer_rl.formal.simulation.health import NativePhysicsHealth

    # These parsers are upstream contracts. No controller/physics constants
    # are reimplemented in this project.
    asset = load_asset_description(str(asset_directory),
                                   expected_manifest_sha256=configuration["asset_manifest_sha256"])
    response_path = verify_pin(runtime["response_source_pin"])
    response = ResponseParameters.from_recorded_response(json.loads(response_path.read_text())["actuator_response"])
    task = FlatTaskSpec(**runtime["task"])
    sensors = SensorModelConfig(**runtime["sensors"])
    dynamics = StartupDynamicsConfig(**runtime["dynamics"])
    sections = optional_sections(runtime)
    for name in ("command_pool", "birth_randomization"):
        section = sections[name]
        if section is not None and section.enabled:
            raise ValueError("flat factory does not support enabled " + name)
    _validate_training_observer(sensors, sections)
    if sensors.physics_dt_s != 0.005 or sensors.policy_dt_s != 0.02:
        raise ValueError("the initial native adapter requires the 200Hz/50Hz upstream clock")
    pair_fields = runtime.get("pair_domain")
    pair = None if pair_fields is None else PairTargetDomain(**pair_fields)
    transport = (None if runtime.get("action_transport") is None else
                 ActionTransportConfig.from_dict(runtime["action_transport"]))
    formal = (FormalRewardConfig.from_manifest(asset.manifest, normal_load_source=contact_source)
              if configuration.get("formal_reward", True) else None)
    limits = termination_limits(runtime["termination_limits"]) if formal is not None else None
    physics = runtime["physics"]
    expected_physics = {
        "gpu_found_lost_pairs_capacity", "gpu_found_lost_aggregate_pairs_capacity",
        "gpu_total_aggregate_pairs_capacity", "gpu_max_rigid_contact_count",
        "gpu_max_rigid_patch_count", "gpu_collision_stack_size",
    }
    backend = configuration.get("physics_backend", "physx")
    if backend == "physx" and (not isinstance(physics, dict) or set(physics) != expected_physics
            or any(type(value) is not int or value < 1 for value in physics.values())):
        raise ValueError("runtime requires the complete explicit PhysX capacity contract")
    if backend == "kamino" and physics != {}:
        raise ValueError("Kamino runtime.physics must be empty; configure the solver in factory.kamino")
    launcher = native = health = None
    try:
        if backend == "kamino":
            from .kamino import KaminoBackend

            launcher = KaminoBackend(configuration["device"], configuration.get("kamino", {}))
            health = launcher.health
        else:
            health = NativePhysicsHealth.for_native_process(
                Path(configuration["native_log_directory"]).expanduser().resolve(strict=True))
            launcher = LabSimulation(configuration["device"], gui=False)
        launcher.configure_settings()
        import torch
        from transformer_rl.formal.lab_env.cfg import build_env_cfg, configure_course_physics
        from transformer_rl.formal.lab_env.env import WheeledLabEnv

        if backend == "physx":
            import omni.kit.app

            extensions = omni.kit.app.get_app().get_extension_manager()
            if not extensions.is_extension_enabled("omni.physx.gpu"):
                extensions.set_extension_enabled_immediate("omni.physx.gpu", True)
        torch.cuda.set_device(torch.device(configuration["device"]))
        torch.cuda.init()
        torch.manual_seed(configuration.get("seed", 0))
        cfg = build_env_cfg(
            str(asset_directory), num_envs=configuration["num_envs"], device=configuration["device"],
            seed=configuration.get("seed", 0), env_spacing_m=runtime["environment_spacing_m"],
            response=response, pair_domain=pair, task=task, formal_reward=formal,
            termination_limits=limits, sensors=sensors, dynamics=dynamics,
            action_transport=transport, **sections, terrain_profile="plane", forward_scan=False,
            declarative_scene=launcher.declarative_scene,
            expected_manifest_sha256=configuration["asset_manifest_sha256"],
        )
        # Both the course and analytical flat plane need no native ray queries.
        # Reuse the upstream capacity/GA-isolation binding verbatim.
        if backend == "kamino":
            environment_type = launcher.environment_type(WheeledLabEnv)
        else:
            configure_course_physics(cfg, physics, num_envs=configuration["num_envs"],
                                     device=configuration["device"], declarative_scene=launcher.declarative_scene)
            environment_type = WheeledLabEnv
        if configuration.get("gimbal") is not None:
            from isaaclab.actuators import IdealPDActuatorCfg
            from .full_vehicle import make_full_vehicle_env

            gimbal_model = json.loads(asset.bound_bytes(asset.manifest["auxiliary_model"]))
            limits_by_name = {joint["name"]: joint["effort_limit_nm"] for joint in gimbal_model["joints"]}
            cfg.robot_cfg.actuators["independent_gimbal"] = IdealPDActuatorCfg(
                joint_names_expr=list(gimbal_model["joint_names"]), stiffness=0.0, damping=0.0,
                effort_limit=limits_by_name, effort_limit_sim=limits_by_name)
            if cfg.declarative_scene:
                # The GA scene config owns a copied robot configuration.
                # Preserve the clone-isolation callbacks registered on it.
                cfg.scene.robot.actuators["independent_gimbal"] = cfg.robot_cfg.actuators["independent_gimbal"]
            environment_type = make_full_vehicle_env(
                environment_type, asset, configuration["gimbal"],
                mechanics_backend="newton" if backend == "kamino" else "physx")
        elif asset.manifest.get("auxiliary_joint_names"):
            raise ValueError("a full-vehicle asset requires an explicit independent gimbal configuration")
        if backend == "kamino":
            launcher.prepare_scene(cfg)
        native = environment_type(cfg)
        backend_receipt = launcher.finalize_environment(native) if backend == "kamino" else None
        health.check()
        episode_commands = _validate_episode_commands(configuration.get("episode_commands"))
        command_seed = configuration.get("seed", 0) ^ 104729
        if episode_commands is not None:
            native.set_command_sampler(_EpisodeCommandSampler(
                episode_commands, seed=command_seed, device=configuration["device"]))
        provenance = {
            "upstream_root": str(root), "upstream_commit": configuration["expected_commit"],
            "runtime_sha256": configuration["runtime_sha256"],
            "asset_manifest_sha256": configuration["asset_manifest_sha256"],
            "response_source_sha256": runtime["response_source_pin"]["sha256"],
            "terrain_profile": "plane", "reward_version": REWARD_VERSION if formal is not None else "flat_task",
            "normal_load_source": contact_source if formal is not None else "contact_force_norm",
            "missing_reward_terms": [] if formal is None else list(native._formal_reward.missing_terms),
            "native_execution_validated": False,
            "observer_contract": "instantaneous_noiseless_clean35_v1",
            "body_names": list(native._binding.native_body_names),
            "joint_names": list(native._binding.native_joint_names),
            "policy_joint_indices": list(native._binding.policy_joint_indices),
            "continuous_joint_names": list(asset.continuous_joint_names()),
            "native_joint_types": {joint["name"]: joint["type"] for joint in asset._source_joints()},
            "physics_health": "upstream dedicated native stdout/stderr monitor",
            "episode_commands": episode_commands,
            "episode_command_seed": None if episode_commands is None else command_seed,
            "command_schedule": "source nominal" if episode_commands is None else "sampled once per physical episode reset",
            "state_scope": "native rigid bodies, joints, supported controller and sensor memory; excludes solver warm starts",
        }
        provenance["sdk_versions"] = {}
        if backend == "kamino":
            provenance["physics_backend"] = "kamino"
            provenance["backend"] = backend_receipt
            provenance["physics_health"] = "finite Kamino solver diagnostics and transition validation; convergence recorded separately"
        if configuration.get("compact_state", False):
            provenance["compact_state"] = True
        if configuration.get("gimbal") is not None:
            provenance["gimbal"] = configuration["gimbal"]
            provenance["full_vehicle"] = native.full_vehicle_receipt
        packages = (("isaaclab", "isaaclab_newton", "newton", "warp-lang", "torch")
                    if backend == "kamino" else ("isaaclab", "isaacsim", "isaaclab_physx", "torch"))
        for package in packages:
            try:
                provenance["sdk_versions"][package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                provenance["sdk_versions"][package] = "unknown"
        adapter = _OwnedIsaacAdapter(native, launcher, health, provenance)
        adapter._native_log_directory = Path(configuration["native_log_directory"])
        return adapter
    except BaseException as error:
        # Kit shutdown can terminate the interpreter; persist construction
        # failures before closing it so the native error is not lost.
        failure = {"type": type(error).__name__, "message": str(error),
                   "traceback": traceback.format_exc(), "phase": "native_factory"}
        (Path(configuration["native_log_directory"]) / "failure.json").write_text(
            json.dumps(failure, indent=2) + "\n")
        print(failure["traceback"], file=sys.stderr, flush=True)
        for resource in (native, health, launcher):
            if resource is not None:
                try:
                    resource.close()
                except BaseException as cleanup_error:
                    error.add_note(f"Native cleanup failed: {cleanup_error}")
        raise
