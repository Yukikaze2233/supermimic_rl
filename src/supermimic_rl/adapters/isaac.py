"""Strict, simulator-independent capture boundary for the existing WheeledLabEnv.

Construct the original ``WheeledLabEnv(cfg)`` after starting its Isaac Kit
application, then pass it to ``IsaacAdapter``. This module neither starts Kit
nor imports Isaac. Use the original flat configuration with an explicit
``static_plane_height_m``; task courses and scripted recovery are rejected.

The snapshots contain observable native rigid-body/joint state and the
supported controller/sensor memories. They are NOT restorable PhysX snapshots:
contact solver warm starts and global random-generator states are not exposed.
Learned rollouts therefore require validation against the real simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any

import torch

from .mechanics import read_articulation_mechanics, validate_mechanics_backend


@dataclass(frozen=True)
class RawSnapshot:
    state: dict[str, torch.Tensor]
    context: dict[str, torch.Tensor]
    observation: torch.Tensor
    actor_observation: torch.Tensor
    goal: torch.Tensor
    episode_id: torch.Tensor
    control_epoch: torch.Tensor
    policy_owned: torch.Tensor
    # Clock metadata stays float64 and must never enter the float32 model state.
    timestamp_s: torch.Tensor | None = None
    field_metadata: dict[str, dict[str, dict[str, str]]] | None = None
    estimator_target: torch.Tensor | None = None
    estimator_valid: torch.Tensor | None = None


@dataclass(frozen=True)
class RawTransition:
    start: RawSnapshot
    end: RawSnapshot
    action: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    valid: torch.Tensor
    # Named [batch, substeps, ...] physics-tick truth, when the adapter records it.
    substeps: dict[str, torch.Tensor] | None = None


class IsaacAdapter:
    """Capture all rows before auto-reset without copying the physics loop.

    The supported boundary is flat, policy-owned locomotion. Component replacement,
    script handoffs, missing native fields and missing terminal hooks fail
    explicitly. ``current`` is the next decision endpoint after any auto-reset;
    ``RawTransition.end`` always precedes that reset.
    """

    _UNSUPPORTED = (
        "_terrain_course", "_task_sampler", "_recover_task",
        "_recovery_preparation", "_birth_randomization",
    )
    _OPTIONAL = (
        "_sensor_model", "_action_transport", "_action_latch", "_communication",
        "_episode_randomization",
    )
    _TARGET_FIELDS = (
        "issued_action", "raw_leg_position_rad", "admitted_leg_position_rad",
        "wheel_velocity_rad_s", "pair_projected", "common_turns",
        "controller_issued_at_s", "control_source",
    )
    observation_dim = 35
    supports_mixed_ownership = False

    def __init__(self, env: Any):
        self.env = env
        self.mechanics_backend = validate_mechanics_backend(getattr(env, "mechanics_backend", "physx"))
        self.num_envs = self._required(env, "num_envs")
        self.device = torch.device(self._required(env, "device"))
        if type(self.num_envs) is not int or self.num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        self._busy = False
        self.current: RawSnapshot | None = None
        self._episode_id = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._control_epoch = self._episode_id.clone()
        self._validate_scope()
        self._components = {name: self._required(env, name) for name in self._OPTIONAL}
        binding = self._required(env, "_binding")
        self.body_names = tuple(self._required(binding, "native_body_names"))
        self.joint_names = tuple(self._required(binding, "native_joint_names"))
        if (len(self.body_names) != 19 or len(set(self.body_names)) != 19
                or len(self.joint_names) != 18 or len(set(self.joint_names)) != 18
                or tuple(env.robot.body_names) != self.body_names
                or tuple(env.robot.joint_names) != self.joint_names):
            raise ValueError("the named native 19-body/18-joint topology must match")
        self.joint_types = None
        asset = getattr(env, "_asset", None)
        if asset is not None:
            joint_types = {joint["name"]: joint["type"] for joint in asset._source_joints()}
            if set(joint_types) != set(self.joint_names):
                raise ValueError("native joint units require the exact pinned model_spec joint names")
            self.joint_types = tuple(joint_types[name] for name in self.joint_names)
        self.policy_dt_s = float(self._required(env, "step_dt"))
        self.physics_dt_s = float(self._required(env, "physics_dt"))
        if (not math.isfinite(self.policy_dt_s) or not math.isfinite(self.physics_dt_s)
                or self.physics_dt_s <= 0 or self.policy_dt_s <= 0
                or abs(self.policy_dt_s - env.cfg.decimation * self.physics_dt_s) > 1e-10):
            raise ValueError("policy and physics periods must match native decimation")
        self._static_context = self._read_static_context()

    @classmethod
    def create(cls, cfg, *, render_mode=None):
        """Build the original native environment after the caller starts Kit.

        Build ``cfg`` with ``transformer_rl.formal.lab_env.cfg.build_env_cfg``
        using ``terrain_profile='plane'``, ``forward_scan=False``, explicit
        ``formal_reward`` and ``termination_limits`` for locomotion_v4. The
        adapter accepts the original reward unchanged, including its missing
        contact-slip term. This factory does not launch a simulator application.
        """
        from transformer_rl.formal.lab_env.env import WheeledLabEnv

        return cls(WheeledLabEnv(cfg, render_mode=render_mode))

    @staticmethod
    def _required(owner: Any, name: str) -> Any:
        if not hasattr(owner, name):
            raise ValueError(f"required native field is missing: {type(owner).__name__}.{name}")
        return getattr(owner, name)

    @staticmethod
    def _native_tensor(value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        proxy = getattr(value, "torch", None)
        if isinstance(proxy, torch.Tensor):
            return proxy
        # PhysX root_view getters return Warp arrays in IsaacLab 3.0.
        if type(value).__module__.startswith("warp"):
            import warp

            return warp.to_torch(value)
        raise TypeError("native data must be a torch, ProxyArray or Warp array")

    def _owned(self, name: str, value: Any, shape=None, *, dtype=None) -> torch.Tensor:
        value = self._native_tensor(value)
        if (value.ndim < 1 or value.shape[0] != self.num_envs
                or shape is not None and value.shape != (self.num_envs, *shape)
                or dtype is not None and value.dtype != dtype):
            raise ValueError(f"native tensor layout differs: {name}")
        if value.is_complex() or not torch.isfinite(value).all():
            raise ValueError(f"native tensor must be real and finite: {name}")
        return value.detach().to(self.device).clone()

    def _validate_scope(self) -> None:
        for name in self._UNSUPPORTED:
            if self._required(self.env, name) is not None:
                raise ValueError(f"flat policy-only adapter does not support {name}")
        height = self._required(self.env.cfg, "static_plane_height_m")
        if type(height) not in (int, float) or not math.isfinite(height):
            raise ValueError("flat capture needs an explicit finite static plane height")
        if not callable(self._required(self.env, "_capture_terminal")):
            raise ValueError("the native pre-reset capture hook is required")
        if self._required(self.env, "_forward_scanners"):
            raise ValueError("flat capture does not support automatic stair triggers")
        episode = self._required(self.env, "_episode_randomization")
        if episode is not None and self._required(episode.config, "enabled"):
            raise ValueError(
                "enabled episode_randomization is unsupported: this flat adapter has no task_sampler, "
                "so upstream reset uses update=0 and exposure=0; explicit fixed exposure is required")

    def _read_static_context(self) -> dict[str, torch.Tensor]:
        parameters = read_articulation_mechanics(
            self.env.robot, backend=self.mechanics_backend, tensor_converter=self._native_tensor)
        context = {}
        for name, value, shape in (
            ("body_mass_kg", parameters.mass_kg, (19,)),
            ("body_com_pose_local_xyzw", parameters.com_pose_local_xyzw, (19, 7)),
            ("body_inertia_kg_m2", parameters.inertia_kg_m2, (19, 9)),
        ):
            context[name] = self._owned(name, value, shape)
        controller = self.env._control.controller
        for name in ("nominal", "action_bounds", "effort_bounds", "stiffness", "damping"):
            value = self._native_tensor(self._required(controller, name))
            if value.shape != (6,):
                raise ValueError(f"controller.{name} must contain all six axes")
            context[f"controller.{name}"] = self._owned(name, value[None].expand(self.num_envs, -1))
        context["wheel_material"] = self._owned(
            "wheel material", self.env._wheel_materials.features(), (6,))
        action_bounds = self._native_tensor(self._required(self.env, "_action_bounds"))
        if action_bounds.shape != (6,) or (action_bounds <= 0).any():
            raise ValueError("native action bounds must contain six positive values")
        context["policy_action_bounds"] = self._owned(
            "policy action bounds", action_bounds[None].expand(self.num_envs, -1), (6,))
        reference = context["body_mass_kg"]
        for name, value in (("policy_dt_s", self.policy_dt_s), ("physics_dt_s", self.physics_dt_s),
                            ("plane_height_m", self.env.cfg.static_plane_height_m)):
            context[name] = reference.new_full((self.num_envs, 1), value)
        drive = self._required(self.env._control, "drive")
        if drive is not None:
            for name in ("additional_armature", "coulomb_effort", "viscous_effort_per_speed",
                         "effective_time_constant_s"):
                values = reference.new_tensor(self._required(drive.parameters, name))
                if values.shape != (6,):
                    raise ValueError(f"drive response {name} must contain six axes")
                context[f"drive.{name}"] = values[None].expand(self.num_envs, -1).clone()
        return context

    def _component_fields(self, output, prefix, component, names):
        for name in names:
            output[f"{prefix}.{name}"] = self._owned(
                f"{prefix}.{name}", self._required(component, name))

    def _age_seconds(self, name, timestamp, clock):
        """Subtract native absolute clocks in float64 before any model packing."""
        timestamp = self._owned(name, timestamp, dtype=torch.float64)
        reference = clock.reshape(self.num_envs, *([1] * (timestamp.ndim - 1)))
        return reference - timestamp

    def _field_metadata(self, state, context):
        """Annotate audited units; preserve source-defined for unspecified SDK axes."""
        joint_position = "source-defined per-native-joint"
        joint_velocity = "source-defined per-native-joint/time"
        if self.joint_types is not None:
            units = ["m" if kind == "prismatic" else "rad" for kind in self.joint_types]
            joint_position = "native order: " + ",".join(units)
            joint_velocity = "native order: " + ",".join(unit + "/s" for unit in units)
        joint_frame = "native joint order: " + ",".join(self.joint_names)
        body_frame = "world; native body order: " + ",".join(self.body_names)
        known_state = {
            "body_link_pose_world_xyzw": ("xyz:m; quaternion_xyzw:1", body_frame),
            "body_com_pose_world_xyzw": ("xyz:m; quaternion_xyzw:1", body_frame),
            "body_com_velocity_world": ("linear:m/s; angular:rad/s", body_frame),
            "joint_position_native": (joint_position, joint_frame),
            "joint_position_continuous": (joint_position, joint_frame),
            "joint_velocity_native": (joint_velocity, joint_frame),
            "continuous._previous_raw": (joint_position, joint_frame),
            "continuous._position": (joint_position, joint_frame),
            "continuous._turns": ("4*pi-radian native wraps", "selected continuous joint order"),
            "terrain_contact_force_world_n": ("N", body_frame),
            "clean_observation": ("mixed normalized features", "upstream 35D actor observation contract"),
            "control._last_effort": ("N*m", "policy axis order"),
            "control._last_policy_velocity": ("rad/s", "policy axis order"),
            "control._last_joint_acceleration": ("rad/s^2", "policy axis order"),
            "control.episode_length_buf": ("policy steps", "episode-relative counter"),
            "target.raw_leg_position_rad": ("rad", "four leg output axes"),
            "target.admitted_leg_position_rad": ("rad", "four leg output axes"),
            "target.wheel_velocity_rad_s": ("rad/s", "left,right wheel output axes"),
            "target.common_turns": ("2*pi-radian turns", "left,right coaxial pairs"),
            "drive.state": ("N*m", "policy axis order"),
            "communication.current": ("q[6]:rad; dq[6]:rad/s", "policy axis order; feedback encoder coordinates"),
            "communication.previous": ("q[6]:rad; dq[6]:rad/s", "policy axis order; feedback encoder coordinates"),
            "communication.held_effort": ("N*m", "policy axis order"),
            "sensor.values": ("q[6]:rad; dq[6]:rad/s; gyro[3]:rad/s; gravity[3]:1", "accessible history newest-first; encoder/IMU"),
            "sensor.heading": ("cos(yaw),sin(yaw):1", "accessible history newest-first"),
            "reward._last_dq": ("rad/s", "policy axis order"),
            "failure.contact_time_s": ("s", "consecutive forbidden-contact duration"),
            "failure.pose_time_s": ("s", "consecutive failed-pose duration"),
        }
        for name in (
            "target.controller_issued_age_s", "target.generated_age_s",
            "communication.applied_age_s", "transport.arrived_generated_age_s",
            "reward.endpoint_age_s", "sensor.sample_ages_s", "transport.generated_age_history_s",
        ):
            known_state[name] = ("s", "current physical endpoint minus source timestamp; float64 subtraction")
        for name in ("communication.held_origin_ages_s", "communication.applied_origin_ages_s"):
            known_state[name] = ("s", "endpoint-relative ages [PD,policy,feedback,target-latch]")
        for name in ("control._previous_action", "control._issued_action", "control._before_action",
                     "target.issued_action", "transport.history", "reward._previous", "reward._previous2"):
            known_state[name] = ("upstream controller action units", "policy axis order")
        known_context = {
            "body_mass_kg": ("kg", "native body order"),
            "body_com_pose_local_xyzw": ("xyz:m; quaternion_xyzw:1", "native rigid-body local frame"),
            "body_inertia_kg_m2": ("kg*m^2", "native SDK declared inertia frame"),
            "controller.nominal": ("rad", "policy axis order"),
            "controller.effort_bounds": ("N*m", "policy axis order"),
            "controller.stiffness": ("N*m/rad", "policy axis order"),
            "controller.damping": ("N*m/(rad/s)", "policy axis order"),
            "wheel_material": ("1", "left/right [static friction,dynamic friction,restitution]"),
            "plane_height_m": ("m", "world Z"),
            "environment_origin_world_m": ("m", "world"),
            "ground_at_root_world_z_m": ("m", "world Z"),
            "ground_at_wheel_world_z_m": ("m", "world Z; left,right"),
            "drive.additional_armature": ("kg*m^2", "policy output axes"),
            "drive.coulomb_effort": ("N*m", "policy output axes"),
            "drive.viscous_effort_per_speed": ("N*m/(rad/s)", "policy output axes"),
            "drive._tau_scale": ("1", "policy output axes"),
            "latch.lags": ("physics steps", "per-environment target latch delay"),
            "transport.lags": ("policy steps", "per-environment action queue delay"),
            "sensor.lags": ("physics steps", "q,dq,gyro,gravity groups"),
        }
        if self.mechanics_backend == "newton":
            known_context["body_com_pose_local_xyzw"] = (
                "xyz:m; quaternion_xyzw:1", "COM translation in link axes; Newton identity COM orientation")
            known_context["body_inertia_kg_m2"] = (
                "kg*m^2", "Newton COM-centered inertia tensor expressed in link axes")
        for name in ("policy_dt_s", "physics_dt_s", "drive.effective_time_constant_s",
                     "communication.total_s", "communication.uplink_s", "communication.downlink_s",
                     "sensor.uplink_delay_s"):
            known_context[name] = ("s", "duration")
        result = {}
        for group, values, known in (("state", state, known_state), ("context", context, known_context)):
            result[group] = {}
            for name in values:
                units, frame = known.get(name, ("source-defined", "source-defined"))
                result[group][name] = {"units": units, "frame": frame}
        return result

    def _capture_memory(self, state, context, clock):
        env = self.env
        self._component_fields(state, "control", env, (
            "_previous_action", "_issued_action", "_before_action", "_last_effort",
            "_last_policy_velocity", "_last_joint_acceleration", "_previous_control_source",
            "episode_length_buf",
        ))
        targets = self._required(env, "_targets")
        if targets is None:
            raise ValueError("native reset must establish actual controller targets")
        self._component_fields(state, "target", targets, self._TARGET_FIELDS)
        state["target.controller_issued_age_s"] = self._age_seconds(
            "controller issuance timestamp", state.pop("target.controller_issued_at_s"), clock)
        if not self.supports_mixed_ownership and (state["target.control_source"] != 0).any():
            raise ValueError("script-owned native targets cannot receive policy credit")
        # At the next pre-physics boundary, env promotes _targets to held
        # targets. The old _held_targets is overwritten before it is read.
        generated = getattr(env, "_policy_generated_at_s", None)
        if generated is None:
            generated = clock
        state["target.generated_age_s"] = self._age_seconds("target generation timestamp", generated, clock)
        state["target.generated_timestamp_present"] = torch.full(
            (self.num_envs,), hasattr(env, "_policy_generated_at_s"), device=self.device, dtype=torch.bool)
        coordinates = env._joint_coordinates
        self._component_fields(state, "continuous", coordinates, ("_previous_raw", "_position", "_turns"))
        drive = env._control.drive
        if drive is not None:
            self._component_fields(state, "drive", drive, ("state",))
            self._component_fields(context, "drive", drive, ("_tau_scale",))
        communication = self._components["_communication"]
        if communication is not None:
            self._component_fields(state, "communication", communication, (
                "current", "previous", "has_previous", "held_effort", "held_origins",
                "applied_origins", "applied_at_s", "policy_allowed",
            ))
            # current/previous are q[6]+dq[6], with no time columns. All four
            # origin columns are absolute seconds: PD, policy, feedback, latch.
            for source, destination in (
                ("held_origins", "held_origin_ages_s"),
                ("applied_origins", "applied_origin_ages_s"),
                ("applied_at_s", "applied_age_s"),
            ):
                state[f"communication.{destination}"] = self._age_seconds(
                    source, state.pop(f"communication.{source}"), clock)
            self._component_fields(context, "communication", communication, ("total_s", "uplink_s", "downlink_s"))
        transport = self._components["_action_transport"]
        if transport is not None:
            capacity = self._required(transport, "capacity")
            index = self._required(transport, "index")
            ages = torch.arange(capacity, device=self.device)
            slots = (index - ages) % capacity
            for name in ("history", "_stamp_history"):
                value = self._native_tensor(self._required(transport, name))
                if value.shape[:2] != (capacity, self.num_envs):
                    raise ValueError("action transport queue layout differs")
                state[f"transport.{name}"] = self._owned(name, value[slots].transpose(0, 1))
            self._component_fields(state, "transport", transport, ("_policy_allowed", "arrived_generated_at_s"))
            state["transport.generated_age_history_s"] = self._age_seconds(
                "transport generation history", state.pop("transport._stamp_history"), clock)
            state["transport.arrived_generated_age_s"] = self._age_seconds(
                "arrived packet generation timestamp", state.pop("transport.arrived_generated_at_s"), clock)
            self._component_fields(context, "transport", transport, ("lags",))
        latch = self._components["_action_latch"]
        if latch is not None:
            self._component_fields(state, "latch", latch, ("held",))
            self._component_fields(context, "latch", latch, ("lags",))
        episode = self._components["_episode_randomization"]
        if episode is not None:
            self._component_fields(context, "episode", episode, (
                "strength", "scale", "motor_strength", "encoder_bias_rad",
                "no_load_scale", "time_constant_scale",
            ))
        sensor = self._components["_sensor_model"]
        if sensor is not None:
            self._capture_sensor_memory(sensor, state, context, clock)
        formal = self._required(env, "_formal_reward")
        if formal is not None:
            composer = self._required(formal, "composer")
            if self._required(composer, "_window") is not None:
                raise ValueError("capture requires a completed native reward interval")
            self._component_fields(state, "reward", composer, (
                "_previous", "_previous2", "_action_count", "_last_dq", "_dq_valid",
                "_time", "_ended", "_success_seen",
            ))
            # PhysicalRewardComposer._time is the absolute reset/finish clock.
            state["reward.endpoint_age_s"] = self._age_seconds(
                "reward endpoint timestamp", state.pop("reward._time"), clock)
            failure = self._required(env, "_failure_latch")
            if failure is None:
                raise ValueError("formal reward requires the original physical failure latch")
            self._component_fields(state, "failure", failure, (
                "contact_time_s", "pose_time_s", "failure_reasons",
            ))

    def _capture_sensor_memory(self, sensor, state, context, clock):
        capacity = self._required(sensor, "capacity")
        count = self._owned("sensor.count", self._required(sensor, "count"), (), dtype=torch.int64)
        if (count < 1).any() or (count > capacity).any():
            raise ValueError("sensor histories must contain a real reset sample")
        tick = self._required(sensor, "physical_tick")
        if type(tick) is not int or tick != self.env._physical_tick:
            raise ValueError("sensor memory belongs to a different physical endpoint")
        ages = torch.arange(capacity, device=self.device)[None].expand(self.num_envs, -1)
        valid = ages < count[:, None]
        # Native reads clamp unavailable ages to the oldest real sample.
        # This avoids reading uninitialized storage without inventing data.
        slots = (tick - ages.minimum(count[:, None] - 1)) % capacity
        rows = torch.arange(self.num_envs, device=self.device)[:, None]
        for name in ("values", "times", "heading"):
            value = self._native_tensor(self._required(sensor, name))
            if value.shape[:2] != (capacity, self.num_envs):
                raise ValueError(f"sensor queue layout differs: {name}")
            selected = value[slots, rows]
            if name == "times":
                state["sensor.sample_ages_s"] = self._age_seconds("sensor sample timestamps", selected, clock)
            else:
                state[f"sensor.{name}"] = self._owned(name, selected)
        state["sensor.valid_age"] = valid.clone()
        state["sensor.count"] = count
        self._component_fields(state, "sensor", sensor, ("initialized",))
        self._component_fields(context, "sensor", sensor, ("lags", "nominal", "noise_scale"))
        halfwidth = self._native_tensor(self._required(sensor, "_halfwidth"))
        context["sensor.noise_halfwidth"] = self._owned(
            "sensor.noise_halfwidth", halfwidth[None].expand(self.num_envs, -1))
        if self._required(sensor, "uplink_delay_s") is not None:
            self._component_fields(context, "sensor", sensor, ("uplink_delay_s",))

    @torch.no_grad()
    def _snapshot(self, *, terminal: bool) -> RawSnapshot:
        self._validate_scope()
        for name, component in self._components.items():
            if self._required(self.env, name) is not component:
                raise ValueError("native component changed after adapter construction: " + name)
        env = self.env
        sample = env._sample()
        measured = env._observations.assembler.measure(sample)
        clock = self._owned("physical clock", env._clock(), (), dtype=torch.float64)
        if not torch.equal(clock, measured.timestamp_s):
            raise ValueError("native observation and physical snapshot clocks differ")
        commands = env._commands()
        clean, actor = self._actor_frames(measured, commands, clock, terminal=terminal)
        data = env.robot.data
        state = {}
        for name, field, shape in (
            ("body_link_pose_world_xyzw", "body_link_pose_w", (19, 7)),
            ("body_com_pose_world_xyzw", "body_com_pose_w", (19, 7)),
            ("body_com_velocity_world", "body_com_vel_w", (19, 6)),
            ("joint_position_native", "joint_pos", (18,)),
            ("joint_velocity_native", "joint_vel", (18,)),
        ):
            state[name] = self._owned(name, self._required(data, field), shape)
        for name in ("body_link_pose_world_xyzw", "body_com_pose_world_xyzw"):
            if (state[name][..., 3:].norm(dim=-1) - 1).abs().gt(2e-4).any():
                raise ValueError("native body orientations must be unit xyzw quaternions")
        state["joint_position_continuous"] = self._owned(
            "unwrapped joint coordinates", sample.joint_position_rad, (18,))
        state["terrain_contact_force_world_n"] = self._owned(
            "terrain contacts", sample.terrain_contact_force_world_n, (19, 3))
        state["clean_observation"] = clean.clone()
        context = {name: value.clone() for name, value in self._static_context.items()}
        context["environment_origin_world_m"] = self._owned(
            "environment origins", sample.environment_origin_world_m, (3,))
        context["ground_at_root_world_z_m"] = self._owned(
            "ground height", sample.ground_at_root_world_z_m, ())
        context["ground_at_wheel_world_z_m"] = self._owned(
            "wheel ground height", sample.ground_at_wheel_world_z_m, (2,))
        self._capture_memory(state, context, clock)
        owned = state["target.control_source"].eq(0)
        return RawSnapshot(
            state, context, clean, actor,
            torch.cat((clean[:, :4], clean[:, 28:35]), dim=-1),
            self._episode_id.clone(), self._control_epoch.clone(), owned,
            timestamp_s=clock.clone(), field_metadata=self._field_metadata(state, context),
        )

    def _actor_frames(self, measured, commands, clock, *, terminal):
        env = self.env
        if (commands.mode != 0).any() or (commands.request_height_m != 0).any():
            raise ValueError("only ordinary flat command mode is supported")
        skill = self._required(env, "_skill_fsm")
        if skill is not None and self._required(skill, "active").any():
            raise ValueError("active skill state is outside the flat adapter boundary")
        clean = env._observations.assembler.actor(
            env._observations.assembler.clean_sensors(measured), commands, now_s=clock)
        if terminal:
            actor = self._required(env, "_terminal_frame")
        else:
            policy = self._required(env, "obs_buf")["policy"]
            if policy.ndim != 2 or policy.shape[1] % self.observation_dim:
                raise ValueError("native policy history must contain complete observation frames")
            actor = policy.reshape(self.num_envs, -1, self.observation_dim)[:, -1]
        return (self._owned("clean actor observation", clean, (self.observation_dim,), dtype=torch.float32),
                self._owned("actual actor observation", actor, (self.observation_dim,), dtype=torch.float32))

    @torch.no_grad()
    def reset(self) -> RawSnapshot:
        if self._busy:
            raise RuntimeError("cannot reset during an adapter step")
        self.current = None
        self.env.reset()
        self._episode_id.add_(1)
        self._control_epoch.add_(1)
        self.current = self._snapshot(terminal=False)
        return self.current

    def close(self) -> None:
        """Close the supplied native environment; application ownership is external."""
        if self._busy:
            raise RuntimeError("cannot close during an adapter step")
        self.current = None
        self.env.close()

    def _continuous_joint_failure_note(self, error: Exception, proposed: torch.Tensor) -> None:
        messages = (
            "continuous joint state is nonfinite or angular sampling is ambiguous",
            "continuous joint position increment is ambiguous",
        )
        if not isinstance(error, RuntimeError) or str(error) not in messages:
            return

        def scalar(value):
            value = float(value)
            return value if math.isfinite(value) else str(value)

        try:
            reader = self.env._joint_coordinates
            indices = reader.indices.detach().cpu()
            raw = self._native_tensor(self.env.robot.data.joint_pos).detach().cpu()
            velocity = self._native_tensor(self.env.robot.data.joint_vel).detach().cpu()
            previous = reader._previous_raw.detach().cpu()
            q = raw.index_select(1, indices)
            dq = velocity.index_select(1, indices)
            old = previous.index_select(1, indices)
            dt = float(reader.physics_dt_s)
            masks = {
                "nonfinite_position": ~torch.isfinite(q),
                "nonfinite_velocity": ~torch.isfinite(dq),
                "angular_speed_ambiguous": torch.isfinite(dq) & (dq.abs() * dt >= math.pi),
            }
            delta = None
            if str(error) == messages[1]:
                # Match the source reader's native 4*pi representation correction.
                period = 2 * math.tau
                turns = torch.round((old.double() - q.double()) / period).long()
                delta = q.double() - old.double() + turns.double() * period
                masks = {"angular_increment_ambiguous": delta.abs() >= math.pi}
            invalid = torch.stack(tuple(masks.values())).any(0)
            issued = self._native_tensor(self.env._issued_action).detach().cpu()
            proposed = proposed.detach().cpu()
            rows = []
            for row in invalid.any(1).nonzero(as_tuple=False).flatten().tolist():
                joints = []
                for column in invalid[row].nonzero(as_tuple=False).flatten().tolist():
                    joint_index = int(indices[column])
                    joint = {
                        "joint_index": joint_index,
                        "joint_name": self.joint_names[joint_index],
                        "q_rad": scalar(q[row, column]),
                        "dq_rad_s": scalar(dq[row, column]),
                        "previous_raw_rad": scalar(old[row, column]),
                        "reasons": [name for name, mask in masks.items() if mask[row, column]],
                    }
                    if delta is not None:
                        joint["corrected_increment_rad"] = scalar(delta[row, column])
                    joints.append(joint)
                rows.append({
                    "row": row, "joints": joints,
                    "issued_action": [scalar(value) for value in issued[row]],
                    "proposed_action": [scalar(value) for value in proposed[row]],
                })
            note = {
                "continuous_joint_diagnostic": {
                    "physical_tick": self.env._physical_tick,
                    "reader_tick": reader._tick,
                    "physics_dt_s": dt,
                    "rows": rows,
                },
            }
            error.add_note(json.dumps(note, allow_nan=False))
        except Exception as diagnostic_error:
            # Diagnostic collection must not replace the original native failure.
            error.add_note(json.dumps({"continuous_joint_diagnostic_error": str(diagnostic_error)}))

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> RawTransition:
        if self._busy or self.current is None:
            raise RuntimeError("reset the adapter before a non-reentrant step")
        action = self._owned("proposed action", action, (6,), dtype=torch.float32)
        start = self.current
        captured: list[tuple[RawSnapshot, torch.Tensor]] = []
        original = self.env._capture_terminal
        absent = object()
        previous_override = self.env.__dict__.get("_capture_terminal", absent)

        def capture():
            try:
                original()
            except ValueError as error:
                if "actor sensor time" in str(error):
                    state = self.env._last_state
                    packet = self.env._sensor_packet(state)
                    commands = self.env._commands()
                    age = state.timestamp_s[:, None] - packet.sampled_at_s
                    error.add_note(str({
                        "physical_tick": self.env._physical_tick,
                        "sensor_age_min_max_s": [float(age.min()), float(age.max())],
                        "command_min": commands.admitted_body_command.amin(0).tolist(),
                        "command_max": commands.admitted_body_command.amax(0).tolist(),
                        "mode": commands.mode.tolist(),
                        "request_height": commands.request_height_m.tolist(),
                        "command_clock": commands.command_clock_s.tolist(),
                    }))
                raise
            if captured:
                raise RuntimeError("native step invoked the terminal boundary more than once")
            captured.append((self._snapshot(terminal=True), self._owned(
                "actual issued action", self.env._issued_action, (6,), dtype=torch.float32)))

        self._busy = True
        self.env._capture_terminal = capture
        try:
            _, reward, terminated, truncated, extras = self.env.step(action)
            if len(captured) != 1:
                raise RuntimeError("native step did not expose exactly one pre-reset terminal boundary")
            end, issued = captured[0]
            bounds = start.context["policy_action_bounds"]
            compared = start.policy_owned if self.supports_mixed_ownership else torch.ones_like(start.policy_owned)
            if not torch.equal(issued[compared], action.clamp(-bounds, bounds)[compared]):
                raise ValueError("native issued action differs from the policy clipping contract")
            reward = self._owned("reward", reward, (), dtype=torch.float32)
            terminated = self._owned("terminated", terminated, (), dtype=torch.bool)
            truncated = self._owned("truncated", truncated, (), dtype=torch.bool)
            valid = self._owned("interval validity", extras["interval_observation_valid"], (), dtype=torch.bool)
            if (terminated & truncated).any():
                raise ValueError("terminated and truncated rows must be disjoint")
            if start.timestamp_s is None or end.timestamp_s is None:
                raise ValueError("native transition requires separate float64 clock metadata")
            if not torch.allclose(end.timestamp_s - start.timestamp_s,
                                  torch.full_like(end.timestamp_s, self.policy_dt_s), atol=1e-8, rtol=0):
                raise ValueError("native transition did not advance exactly one policy interval")
            done = terminated | truncated
            self._episode_id.add_(done.long())
            changed_owner = start.policy_owned != end.policy_owned
            self._control_epoch.add_((done | changed_owner).long())
            self.current = self._snapshot(terminal=False)
            if not self.supports_mixed_ownership:
                valid &= start.policy_owned & end.policy_owned
            return RawTransition(start, end, issued, reward, terminated, truncated, valid)
        except Exception as error:
            # A failed native call may already have advanced physics.
            self.current = None
            self._continuous_joint_failure_note(error, action)
            raise
        finally:
            if previous_override is absent:
                del self.env.__dict__["_capture_terminal"]
            else:
                self.env._capture_terminal = previous_override
            self._busy = False
