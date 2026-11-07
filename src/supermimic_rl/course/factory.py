"""Construct the pinned native full-vehicle training course lazily (Kamino or PhysX TGS)."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import importlib
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
import traceback


def _read_pinned(path, digest):
    path = Path(path).expanduser().resolve(strict=True)
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("Pinned input changed: " + str(path))
    return path, json.loads(content)


def create_course(configuration):
    """Full45 history MLP course using the source FSM and exact static terrain."""
    motion_trajectory_owner = configuration.get('motion_trajectory_owner', False)
    if type(motion_trajectory_owner) is not bool:
        raise ValueError('Motion trajectory owner must be a boolean')
    stair_timeout_retry = configuration.get("stair_timeout_retry", False)
    if type(stair_timeout_retry) is not bool:
        raise ValueError('Stair timeout retry must be a boolean')
    hint_lifetime = configuration.get("stair_height_hint_lifetime", "fixed")
    if hint_lifetime not in ("fixed", "request"):
        raise ValueError('Stair height hint lifetime must be fixed or request')
    if hint_lifetime == "request" and not configuration.get("tof_stair_label", False):
        raise ValueError('Request-lifetime stair hints require the explicit ToF stair label')
    root = Path(configuration["upstream_root"]).resolve(strict=True)
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != configuration["expected_commit"]:
        raise ValueError("Native source commit differs")
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip():
        raise ValueError("Native source checkout must match the frozen commit")
    source = root / "src"
    for name, module in tuple(sys.modules.items()):
        if name == "transformer_rl" or name.startswith("transformer_rl."):
            path = getattr(module, "__file__", None)
            if path and not Path(path).resolve().is_relative_to(source):
                raise ValueError("Another native source checkout is already imported")
    sys.path.insert(0, str(source))
    importlib.invalidate_caches()
    _, runtime = _read_pinned(configuration["runtime_json"], configuration["runtime_sha256"])
    _, course_configuration = _read_pinned(configuration["course_json"], configuration["course_sha256"])
    platform_plan = None
    if configuration.get('platform_task') is not None:
        from .platform_course import PlatformCoursePlan

        platform_plan = PlatformCoursePlan.read(configuration['platform_task'],
            num_envs=configuration['num_envs'], atlas_sha256=runtime['terrain_atlas_pin']['sha256'],
            record_substeps=configuration.get('record_substeps', False),
            jump_takeoff=configuration.get('jump_takeoff'))
    asset_directory = Path(configuration["asset_directory"]).resolve(strict=True)
    _read_pinned(asset_directory / "manifest.json", configuration["asset_manifest_sha256"])
    log = Path(configuration["native_log_directory"])
    log.mkdir(parents=True, exist_ok=True)

    from transformer_rl.common.training_control import verify_pin
    from transformer_rl.formal.control.actuation import PairTargetDomain
    from transformer_rl.formal.control.sensors import SensorModelConfig
    from transformer_rl.formal.simulation.dynamics import StartupDynamicsConfig
    from transformer_rl.formal.lab_env.action_transport import ActionTransportConfig
    from transformer_rl.formal.lab_env.backend import PhysicsBackendConfig
    from transformer_rl.formal.lab_env.contract import FlatTaskSpec
    from transformer_rl.formal.lab_env.sections import optional_sections, termination_limits
    from transformer_rl.formal.lab_env.settings import load_asset_description
    from transformer_rl.formal.lab_env.sdk import LabSimulation
    from transformer_rl.formal.lab_env.formal_curriculum import LabTaskDrawSampler
    from .jump_first import make_course_curriculum, install_jump_first_command_pool
    from .identified_execution import IdentifiedExecution
    from .randomization import CourseRandomization, RequestedRandomizationConfig
    from .backend import select_course_backend

    backend = PhysicsBackendConfig.from_dict(runtime["physics_backend"])
    strategy = select_course_backend(backend, runtime, configuration)
    if platform_plan is not None and strategy.name != 'physx':
        raise ValueError('Native platform patch collection currently requires the PhysX contact interface')
    asset = load_asset_description(str(asset_directory),
                                  expected_manifest_sha256=configuration["asset_manifest_sha256"])
    atlas = verify_pin(runtime["terrain_atlas_pin"])
    sections = optional_sections(runtime)
    if sections["communication"] is not None and sections["communication"].enabled:
        raise ValueError("Identified wheel aggregate delay must not also receive legacy USB delay")
    if sections["motor_envelope"] is not None and sections["motor_envelope"].enabled:
        raise ValueError("Identified wheel envelope replaces the legacy envelope")
    launcher = native = health = None
    build_scope = ExitStack()
    try:
        build_scope.enter_context(strategy.build_scope())
        launcher = LabSimulation(configuration["device"], backend=backend)
        launcher.configure_settings()
        import torch
        from transformer_rl.formal.lab_env.cfg import build_course_env_cfg, configure_course_physics
        from transformer_rl.formal.lab_env.env import WheeledLabEnv
        from .execution import configure_identified_actuators, make_environment
        from .twist import RECEIPT as ROOT_TWIST_RECEIPT, make_root_twist_corrected_environment
        from .adapter import CourseAdapter
        from .routing import install_authored_routing

        torch.cuda.set_device(torch.device(configuration["device"]))
        torch.manual_seed(configuration.get("seed", 7))
        health = strategy.log_health(log)
        sensors = SensorModelConfig(**runtime["sensors"])
        execution = IdentifiedExecution(
            configuration["num_envs"], configuration["device"], sensors.physics_dt_s,
            asset_directory=asset_directory, bundle_directory=configuration["identified_bundle"],
            friction_mode="software_dynamic")
        cfg = build_course_env_cfg(
            asset, atlas, num_envs=configuration["num_envs"], device=configuration["device"],
            seed=configuration.get("seed", 7), environment_spacing_m=runtime["environment_spacing_m"],
            response=None, pair_domain=(None if runtime.get("pair_domain") is None else PairTargetDomain(**runtime["pair_domain"])),
            task=FlatTaskSpec(**runtime["task"]), termination_limits=termination_limits(runtime["termination_limits"]),
            sensors=sensors, dynamics=StartupDynamicsConfig(**runtime["dynamics"]),
            action_transport=ActionTransportConfig.from_dict(runtime["action_transport"]),
            sections=sections, declarative_scene=launcher.declarative_scene, backend=backend)
        configure_course_physics(cfg, runtime["physics"], num_envs=configuration["num_envs"],
                                 device=configuration["device"], declarative_scene=launcher.declarative_scene, backend=backend)
        configure_identified_actuators(cfg, execution, backend=strategy.execution_backend)
        cfg.loop_telemetry_enabled = True
        randomization = CourseRandomization(configuration["num_envs"], configuration.get("seed", 7),
                                            configuration["device"], RequestedRandomizationConfig(**runtime["requested_randomization"]))
        randomized_type = strategy.environment_type(WheeledLabEnv, randomization, execution)
        environment_type = make_environment(make_root_twist_corrected_environment(randomized_type), execution)
        native = environment_type(cfg)
        from .action_contract import install_action_bounds

        action_contract = install_action_bounds(native, runtime["task"]["action_bounds"])
        from .reward import (install_jump_tracking_reference, install_shared_reward,
                             install_stair_support_reference)

        from ..reward_kernel import RewardExtensions

        build_scope.close()
        health = strategy.course_health(health, native)
        curriculum = make_course_curriculum(configuration["num_envs"], configuration["device"],
                                            configuration.get("seed", 7), course_configuration)
        native.set_task_sampler(LabTaskDrawSampler(curriculum))
        from .recovery_bridge import install_recovery_roll_bridge

        recovery_bridge_receipt = install_recovery_roll_bridge(
            native, curriculum, log, evaluation=configuration.get('evaluation') is not None)
        platform_owner = None
        if platform_plan is not None:
            from .platform_course import NativePlatformCourse

            # Bind the final router before any actual-body clearance provider.
            install_authored_routing(native, rehearsal_curriculum=(
                curriculum if curriculum.config.foundation_authored_rehearsal else None))
            platform_owner = NativePlatformCourse(native, platform_plan).install_physics()
        # The native sampler installation creates LabTerrainCourse and its
        # jump task owner; the environment constructor leaves it unset.
        extensions = RewardExtensions(**configuration.get("reward_extensions", {}))
        jump_geometry = None
        if extensions.needs_flight_geometry:
            from .jump_reward_geometry import JumpRewardGeometry, NativeJumpRewardGeometry

            linkage_pin = configuration.get('jump_reward_linkage', {})
            jump_geometry = NativeJumpRewardGeometry(JumpRewardGeometry.load(asset_directory,
                linkage_pin.get('path'), expected_linkage_sha256=linkage_pin.get('sha256')),
                native._terrain_course)
        reward_kernel_receipt = install_shared_reward(
            native._formal_reward.composer, verify_native=configuration.get("audit_reward_kernel", False),
            extensions=extensions, flight_geometry=jump_geometry,
            jump_tasks=native._terrain_course.tasks,
            **({} if platform_owner is None else {'platform_landing': platform_owner.landing}))
        if extensions.stair_support_transfer:
            reward_kernel_receipt['stair_support_transfer'] = install_stair_support_reference(native)
        course_schedule_receipt = install_jump_first_command_pool(native._terrain_course.tasks, curriculum)
        from .gimbal_arbitration import install_request_gimbal_exclusion

        gimbal_arbitration_receipt = install_request_gimbal_exclusion(
            native, enabled=configuration.get('request_gimbal_exclusion', False))
        from .motion_trajectory import install_motion_trajectory_owner

        motion_trajectory_receipt = install_motion_trajectory_owner(native, enabled=motion_trajectory_owner)
        jump_takeoff_receipt = stair_label_receipt = None
        if configuration.get("jump_takeoff") is not None:
            from .jump_takeoff import install_jump_takeoff

            jump_takeoff_receipt = install_jump_takeoff(native, configuration["jump_takeoff"])
        if native._formal_reward.composer.supermimic_reward_extensions.needs_jump_progress:
            install_jump_tracking_reference(native)
            reward_kernel_receipt['jump_tracking_reference'] = 'first_real_flight_landing_v1'
        if configuration.get("tof_stair_label", False):
            from .stair_label import install_tof_stair_label

            stair_label_receipt = install_tof_stair_label(native, height_hint_lifetime=hint_lifetime)
        from .stair_retry import install_stair_timeout_retry

        stair_retry_receipt = install_stair_timeout_retry(native, enabled=stair_timeout_retry)
        if platform_owner is not None:
            # Caller reservation is outermost; ToF measurement/end/retry owners
            # continue running underneath, as does the original take-off gate.
            platform_owner.install_caller()
        jump_assist_receipt = None
        if configuration.get("jump_assist") is not None:
            if configuration.get("evaluation") is not None:
                raise ValueError("Fixed evaluation must measure unassisted jumps")
            from .jump_assist import install_jump_assist

            jump_assist_receipt = install_jump_assist(native, curriculum, configuration["jump_assist"]).receipt()
        if platform_owner is None:
            install_authored_routing(native, rehearsal_curriculum=(
                curriculum if curriculum.config.foundation_authored_rehearsal else None))
        provenance = {
            "upstream_commit": commit, "runtime_sha256": configuration["runtime_sha256"],
            "course_sha256": configuration["course_sha256"],
            "asset_manifest_sha256": configuration["asset_manifest_sha256"],
            "terrain_sha256": runtime["terrain_atlas_pin"]["sha256"],
            "physics_backend": strategy.name, "backend": {"identity": strategy.identity},
            "observer_contract": "full45_physical_sensor_history_v1",
            "physics_dt_s": sensors.physics_dt_s, "policy_dt_s": sensors.policy_dt_s,
            "action_bounds": action_contract["action_bounds"],
            "execution": execution.receipt(), "curriculum": course_configuration,
            "course_schedule": course_schedule_receipt,
            "jump_takeoff": jump_takeoff_receipt,
            "stair_label": stair_label_receipt,
            "jump_assist": jump_assist_receipt,
            "desynchronize_episodes": configuration.get("desynchronize_episodes", False),
            "reward_kernel": reward_kernel_receipt,
            "reward_config": asdict(native._formal_reward.config),
            "terrain_support_config": asdict(native._terrain_course.support_reference.config),
            "randomization": {**randomization.config.provenance(),
                              "applied": strategy.applies_requested_randomization},
            "continuous_coordinates": getattr(native, "continuous_coordinate_receipt", None),
            "numerical_isolation": native.numerical_isolation_receipt,
            "root_twist": ROOT_TWIST_RECEIPT if strategy.name == "kamino" else None,
            "authored_routing": native.authored_routing_receipt,
            "model_state": "full physical articulation, controller memories, local terrain scan, course memories",
            "native_execution_validated": False,
            "sdk_versions": {package: metadata.version(package) for package in strategy.sdk_packages}}
        if stair_retry_receipt is not None:
            provenance['stair_timeout_retry'] = stair_retry_receipt
        if gimbal_arbitration_receipt is not None:
            provenance['request_gimbal_exclusion'] = gimbal_arbitration_receipt
        if motion_trajectory_receipt is not None:
            provenance['motion_trajectory_owner'] = motion_trajectory_receipt
        if recovery_bridge_receipt is not None:
            provenance['recovery_roll_bridge'] = recovery_bridge_receipt
        if platform_owner is not None:
            provenance['platform_task'] = platform_owner.receipt()
        (log / "course-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        setup = {"action_contract": action_contract,
                 "materials": getattr(native._wheel_materials, "receipt", None),
                 "dynamics": getattr(native._startup_dynamics, "receipt", None),
                 "coordinates": getattr(native, "continuous_coordinate_receipt", None),
                 "physics_health": health.receipt}
        (log / "native-setup-readback.json").write_text(json.dumps(setup, indent=2) + "\n")
        return CourseAdapter(native, launcher, health, curriculum, provenance, log,
                             mechanics_backend=strategy.mechanics_backend,
                             evaluation_configuration=configuration.get("evaluation"),
                             record_substeps=configuration.get("record_substeps", False),
                             substep_audit=configuration.get("substep_audit", False),
                             record_disturbances=configuration.get("record_disturbances", False),
                             profile_steps=configuration.get("profile_steps", False),
                             desynchronize_episodes=configuration.get("desynchronize_episodes", False),
                             seed=configuration.get("seed", 7))
    except BaseException as error:
        build_scope.close()
        failure = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        (log / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        print(failure["traceback"], file=sys.stderr, flush=True)
        for resource in (native, health, launcher):
            if resource is not None:
                try:
                    resource.close()
                except BaseException as cleanup_error:
                    error.add_note(str(cleanup_error))
        raise
