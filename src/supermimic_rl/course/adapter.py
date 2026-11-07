"""Full-vehicle course capture, retaining actual sensors and task ownership.

The native environment owns physics, tasks, rewards and the deployable FSM.
This adapter owns only the SuperMimic capture boundary and three additional
gimbal encoder channels. Privileged terrain and labels never enter the actor.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json

import torch

from ..adapters.isaac import IsaacAdapter, RawSnapshot
from ..adapters.mechanics import read_articulation_mechanics
from ..compact_state import CompactStateProjector


class GimbalSensorHistory:
    """Sample independent gimbal encoders on the existing physical sensor clock."""

    def __init__(self, env):
        self.env, self.sensor = env, env._sensor_model
        if self.sensor is None or env._gimbal is None:
            raise ValueError("Formal observation needs physical sensors and the independent gimbal")
        self.values = torch.zeros(self.sensor.capacity, env.num_envs, 5, device=env.device)
        native_push, native_seed = self.sensor.push, self.sensor.seed_after_reset

        def push(*args, **kwargs):
            native_push(*args, **kwargs)
            self._write(self.sensor._rows)

        def seed(mask, *args, **kwargs):
            native_seed(mask, *args, **kwargs)
            self._write(mask.nonzero().flatten())

        self.sensor.push, self.sensor.seed_after_reset = push, seed

    def instantaneous(self):
        gimbal = self.env._gimbal
        q = self.env._as_torch(self.env.robot.data.joint_pos)[:, gimbal.joint_indices]
        dq = self.env._as_torch(self.env.robot.data.joint_vel)[:, gimbal.joint_indices]
        position, velocity = (q - gimbal._zero) * gimbal._sign, dq * gimbal._sign
        return torch.cat((position[:, :1].sin(), position[:, :1].cos(),
                          position[:, 1:2], velocity), -1)

    def _write(self, rows):
        self.values[self.sensor.physical_tick % self.sensor.capacity, rows] = self.instantaneous()[rows]

    def ordered(self):
        sensor = self.sensor
        ages = torch.arange(sensor.capacity, device=sensor.device)[None]
        ages = torch.minimum(ages, (sensor.count - 1)[:, None])
        slots = (sensor.physical_tick - ages) % sensor.capacity
        rows = torch.arange(sensor.num_envs, device=sensor.device)[:, None]
        return self.values[slots, rows].clone()

    def delayed_extra(self):
        sensor = self.sensor
        history = self.ordered()
        rows = torch.arange(sensor.num_envs, device=sensor.device)
        delay = sensor.lags[:, [0, 1, 1]].float()
        if sensor.uplink_delay_s is not None:
            delay = delay + sensor.uplink_delay_s[:, None] / sensor.config.physics_dt_s
        delay = torch.minimum(delay, (sensor.count - 1)[:, None].float())
        lo = delay.floor().long()
        hi = torch.minimum(lo + 1, (sensor.count - 1)[:, None])
        channels = torch.tensor([2, 3, 4], device=sensor.device)
        a, b = history[rows[:, None], lo, channels], history[rows[:, None], hi, channels]
        return a + (b - a) * (delay - lo).to(a.dtype)


class CourseAdapter(IsaacAdapter):
    observation_dim = 45
    supports_mixed_ownership = True

    def _owned(self, name, value, shape=None, *, dtype=None):
        checks = getattr(self, "_capture_finite_checks", None)
        if checks is None:
            return super()._owned(name, value, shape, dtype=dtype)
        value = self._native_tensor(value)
        if (value.ndim < 1 or value.shape[0] != self.num_envs
                or shape is not None and value.shape != (self.num_envs, *shape)
                or dtype is not None and value.dtype != dtype or value.is_complex()):
            raise ValueError("Native tensor layout differs: " + name)
        # Keep one synchronization at the endpoint rather than one per field.
        checks.append((name, torch.isfinite(value).all()))
        return value.detach().to(self.device).clone()

    def __init__(self, env, launcher, health, curriculum, provenance, log_directory, *,
                 mechanics_backend="newton", evaluation_configuration=None, record_substeps=False, substep_audit=False,
                 record_disturbances=False, profile_steps=False, desynchronize_episodes=False, seed=0):
        if type(desynchronize_episodes) is not bool:
            raise ValueError("desynchronize_episodes must be boolean")
        if desynchronize_episodes and evaluation_configuration is not None:
            raise ValueError("Fixed evaluation keeps its scripted episode clocks")
        self._desynchronize = desynchronize_episodes
        self._desync_generator = torch.Generator(device="cpu").manual_seed(seed)
        self.env, self._launcher, self._health = env, launcher, health
        self.curriculum, self.provenance = curriculum, provenance
        self._native_log_directory = Path(log_directory)
        self._evaluation_configuration = evaluation_configuration
        self.num_envs, self.device = env.num_envs, torch.device(env.device)
        if mechanics_backend not in ("newton", "physx"):
            raise ValueError("Course mechanics backend must be newton or physx")
        self.mechanics_backend = mechanics_backend
        self._busy, self._closed, self.current = False, False, None
        self._episode_id = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._control_epoch = self._episode_id.clone()
        self.body_names = tuple(env._binding.native_body_names)
        self.joint_names = tuple(env._binding.native_joint_names)
        if len(self.body_names) != 21 or len(self.joint_names) != 20:
            raise ValueError("Full course requires the audited 21-body/20-joint vehicle")
        kinds = {j["name"]: j["type"] for j in env._asset._source_joints()}
        self.joint_types = tuple(kinds.get(name, "revolute") for name in self.joint_names)
        self.policy_dt_s, self.physics_dt_s = float(env.step_dt), float(env.physics_dt)
        self._components = {name: self._required(env, name) for name in self._OPTIONAL}
        self._gimbal_sensors = GimbalSensorHistory(env)
        self._projector = CompactStateProjector(self.body_names, self.joint_names)
        self._static_context = self._read_static_context()
        self._validate_scope()
        if type(record_substeps) is not bool or type(substep_audit) is not bool:
            raise ValueError("record_substeps and substep_audit must be boolean")
        if substep_audit and not record_substeps:
            raise ValueError("substep_audit extends recorded substeps; enable record_substeps")
        if type(record_disturbances) is not bool or record_disturbances and not record_substeps:
            raise ValueError("record_disturbances must be boolean and requires record_substeps")
        self._substeps = None
        self.substep_fields, self.substeps_per_step = (), 0
        if record_substeps:
            from ..adapters.substeps import SubstepRecorder
            self._substeps = SubstepRecorder(env, audit=substep_audit,
                                             disturbances=record_disturbances).install()
            self.substep_fields = self._substeps.fields
            self.substeps_per_step = self._substeps.count
        if type(profile_steps) is not bool:
            raise ValueError("profile_steps must be boolean")
        self._profiler = None
        if profile_steps:
            from ..adapters.profiling import StepProfiler
            # Outermost first: adapter step > native step > its phases. Nested
            # labels overlap; each label's total is inclusive of its callees.
            self._profiler = (StepProfiler(self.device)
                              .wrap(self, "step", "adapter.step")
                              .wrap(env, "step", "native.step")
                              .wrap(env.sim, "step", "native.physics")
                              .wrap(env, "_observe_substep", "native.observe_substep")
                              .wrap(self, "_snapshot", "adapter.snapshot"))
            course = env._terrain_course
            for owner, method, label in (
                    (env, "_pre_physics_step", "native.pre_physics"),
                    (env, "_apply_action", "native.tick.apply_action"),
                    (env.scene, "write_data_to_sim", "native.tick.write_data"),
                    (env.scene, "update", "native.tick.scene_update"),
                    (env, "_sample", "native.sample"),
                    (env._sensor_model, "push", "native.tick.sensor_push"),
                    (course, "observe_support", "native.tick.terrain_support"),
                    (course, "observe_reward", "native.tick.reward_observe"),
                    (env, "_update_skill", "native.post.update_skill"),
                    (env, "_update_gimbal_course", "native.post.gimbal_course"),
                    (env, "_reset_idx", "native.post.reset"),
                    (env, "_get_observations", "native.post.observations"),
                    (env._observations.assembler, "measure", "native.tick.measure"),
                    (env, "_observe_spring_state", "native.tick.spring_state"),
                    (getattr(env, "_boundary_geometry", None), "evaluate", "native.tick.boundary_geometry"),
                    (getattr(env, "_pressure", None), "evaluate", "native.tick.pressure"),
                    (getattr(env, "_failure_latch", None), "observe", "native.tick.failure_latch"),
                    (course, "termination", "native.tick.course_termination"),
                    (getattr(course, "stair_evidence", None), "observe", "native.tick.reward.stair_evidence"),
                    (getattr(course, "recovery", None), "observe_physics", "native.tick.reward.recovery"),
                    (getattr(course, "tasks", None), "observe_physics", "native.tick.reward.tasks"),
                    (getattr(env, "_formal_reward", None), "observe", "native.tick.reward.formal"),
                    (getattr(env, "_loop_telemetry", None), "observe_physics", "native.tick.loop_telemetry"),
                    (self._health, "observe_physics", "native.tick.health"),
                    (getattr(env, "_gimbal", None), "step", "native.tick.gimbal"),
                    (env, "_capture_terminal", "native.post.capture_terminal"),
                    (self, "_capture_snapshot", "adapter.capture_snapshot")):
                if owner is not None and callable(getattr(owner, method, None)):
                    self._profiler.wrap(owner, method, label)

    @torch.no_grad()
    def reset(self):
        """Cold reset; optionally spread the first episodes' time limits uniformly.

        All rows are born together, so with equal episode lengths every row
        times out and is reborn in the same interval. Like RSL-RL's
        init_at_random_ep_len, the first episode of each row is truncated at a
        uniform point of its length, after which episodes stay staggered.
        Only the time-limit counter moves; task clocks are untouched.
        """
        snapshot = super().reset()
        if self._desynchronize:
            limit = int(self.env.max_episode_length)
            offset = torch.randint(0, limit, (self.num_envs,), generator=self._desync_generator)
            self.env.episode_length_buf.copy_(offset.to(self.env.episode_length_buf))
        return snapshot

    def evaluation_begin(self, schema):
        from .evaluation import CourseEvaluation
        self._course_evaluator = CourseEvaluation(self, self._evaluation_configuration)
        self._course_evaluator.begin(schema)

    def evaluation_step(self, transition, tracking):
        self._course_evaluator.step(transition, tracking)

    def evaluation_sample(self):
        evaluator = getattr(self, '_course_evaluator', None)
        return None if evaluator is None else evaluator.sample()

    def evaluation_summary(self):
        return self._course_evaluator.summary()

    def _validate_scope(self):
        if self.env._terrain_course is None or self.env._gimbal_course is None:
            raise ValueError("Formal course and moving independent gimbal must be connected")
        if not callable(self.env._capture_terminal):
            raise ValueError("Pre-reset physical endpoint capture is required")

    def _read_static_context(self):
        env = self.env
        mechanics = read_articulation_mechanics(env.robot, backend=self.mechanics_backend,
                                                tensor_converter=self._native_tensor)
        context = {name: self._owned(name, value) for name, value in (
            ("body_mass_kg", mechanics.mass_kg),
            ("body_com_pose_local_xyzw", mechanics.com_pose_local_xyzw),
            ("body_inertia_kg_m2", mechanics.inertia_kg_m2))}
        reference = context["body_mass_kg"]
        controller = env._control.controller
        for name in ("nominal", "action_bounds", "effort_bounds", "stiffness", "damping"):
            context["controller." + name] = getattr(controller, name)[None].expand(self.num_envs, -1).clone()
        context["policy_joint_indices"] = reference.new_tensor(env._binding.policy_joint_indices)[None].expand(self.num_envs, -1).clone()
        context["policy_action_bounds"] = env._action_bounds[None].expand(self.num_envs, -1).clone()
        context["wheel_material"] = env._wheel_materials.features().clone()
        for name, value in (("policy_dt_s", self.policy_dt_s), ("physics_dt_s", self.physics_dt_s)):
            context[name] = reference.new_full((self.num_envs, 1), value)
        context["gimbal.control_joint_signs"] = env._gimbal._sign[None].expand(self.num_envs, -1).clone()
        context["gimbal.joint_zero_rad"] = env._gimbal._zero[None].expand(self.num_envs, -1).clone()
        for name, value in env._identified_execution.context_parameters().items():
            context["identified." + name] = value.clone()
        return context

    def _actor_frames(self, measured, commands, clock, *, terminal):
        env = self.env
        clean35 = env._observations.assembler.actor(
            env._observations.assembler.clean_sensors(measured), commands, now_s=clock)
        clean_gimbal = self._gimbal_sensors.instantaneous()
        context = env._gimbal_course.context(env._gimbal_encoder_yaw())
        clean = torch.cat((clean35, context[:, :5], clean_gimbal), -1)
        if terminal:
            actual42 = env._terminal_frame
        else:
            actual42 = env.obs_buf["policy"].reshape(self.num_envs, -1, 42)[:, -1]
        actual = torch.cat((actual42, self._gimbal_sensors.delayed_extra()), -1)
        return self._owned("clean observation", clean, (45,)), self._owned("actor observation", actual, (45,))

    def _capture_course(self, state, context, clock):
        env, course = self.env, self.env._terrain_course
        state["sensor.gimbal_history"] = self._gimbal_sensors.ordered()
        # Local coordinates and measured terrain geometry retain stair location
        # when the compact physical state removes global body translations.
        state["course.root_position_local_m"] = env._root_link_position() - course.router.origins
        state["course.terrain_tile"] = course.router.tile.clone()
        state["course.body_contact_allowed"] = course.body_contact_allowed.clone()
        state["course.command_mode"] = env._command_mode.clone()
        state["course.terrain_heights_relative_m"] = course._scan_heights - env._root_link_position()[:, 2:3]
        state["course.command_remaining_s"] = (course.tasks.segment_end_s - course.tasks.elapsed).clamp(0, 120)[:, None]
        remaining = torch.minimum(state["course.command_remaining_s"], env._gimbal_course.remaining_s[:, None].clamp_min(0))
        next_request = course.tasks.next_request_s - course.tasks.elapsed
        # Planned jump births remain NORMAL until their request is admitted.
        has_request = (course.tasks.family == 4) & (next_request > 0)
        remaining = torch.where(has_request[:, None], torch.minimum(remaining, next_request[:, None]), remaining)
        state["course.imagination_remaining_s"] = remaining
        state["course.dynamic_goal"] = (course.tasks.heading_enabled
                                        | (course.tasks.trajectory.kind != 0))
        for prefix, owner, names in (
            ("course.tasks", course.tasks, ("elapsed", "segment_clock", "family", "base_mode", "terrain_kind",
                "terrain_step_height_m", "approach_sign", "heading_enabled", "heading_target_rad", "request_clock",
                "phase", "active_rise", "latched_height", "flight_seen", "air_ticks", "ascent_seen", "landed_s",
                "ground_anchor", "_last_world_height", "_last_supported", "peak_rise", "supported_start", "_physical_cancelled")),
            ("course.trajectory", course.tasks.trajectory, ("start", "end", "duration_s", "kind",
                                                          "frequency_start_hz", "frequency_end_hz")),
            ("course.stream", course.tasks.stream, ("current",)),
            ("course.recovery", course.recovery, ("active", "policy_elapsed_s", "estimated_stable_s", "physical_stable_s", "_verified")),
            ("course.gimbal", env._gimbal_course, ("remaining_s", "mode", "target", "rate", "special", "speed", "direction", "chassis_rate")),
            ("gimbal.controller", env._gimbal.controller, ("mode", "target_angles", "yaw_rate_target", "_last_joint", "_last_world")),
        ):
            self._component_fields(state, prefix, owner, names)
        support = getattr(course, "support_reference", None)
        if support is not None:
            # These clocks are physical reward state, even on flat landing
            # births. Reconstructing them as zero would change rough_dash phases.
            self._component_fields(state, "course.support", support, (
                "_contact", "_airborne", "_unsupported_s", "_landing_remaining_s",
                "_phase_active", "_phase_elapsed_s", "_wheel_contact_s", "_base_contact_s"))
        preparation = env._recovery_preparation
        if preparation is not None:
            self._component_fields(state, "preparation", preparation, (
                "active", "elapsed", "frames", "_start", "_delta", "_duration", "passive", "_release_frames"))
        protocol = course.tasks.protocol
        self._component_fields(state, "course.protocol", protocol, (
            "height_m", "cancelled", "flight_seen", "air_ticks"))
        state["course.protocol.active"] = protocol.active_id >= 0
        tasks = course.tasks
        for name in ("active_id", "physical_id", "success_id", "paid_id"):
            state["course.tasks." + name + ".valid"] = getattr(tasks, name) >= 0
        state["course.tasks.physical_paid"] = (tasks.physical_id >= 0) & (tasks.physical_id == tasks.paid_id)
        state["course.tasks.physical_success"] = (tasks.physical_id >= 0) & (tasks.physical_id == tasks.success_id)
        for name in ("accepted_at_s", "last_time_s", "landing_since_s"):
            stamp = getattr(protocol, name)
            valid = torch.isfinite(stamp)
            state["course.protocol." + name + ".valid"] = valid.clone()
            state["course.protocol." + name + ".age_s"] = torch.where(valid, clock - stamp, 0.).clamp(0, 120).float()
        for name, stamp in (("course.recovery.last_estimate", course.recovery._last_estimate_s),
                            ("course.tasks.physical_accepted", tasks._physical_accepted_at_s)):
            valid = torch.isfinite(stamp)
            state[name + ".valid"] = valid.clone()
            state[name + ".age_s"] = torch.where(valid, clock - stamp, 0.).clamp(0, 120).float()
        randomization = getattr(env, "_requested_randomization", None)
        if randomization is not None:
            for name, value in randomization.context_parameters().items():
                context["requested." + name] = value.clone()
        recorder = getattr(self, "_substeps", None)
        if recorder is not None and recorder.disturbances is not None:
            for name, value in recorder.disturbances.snapshot().items():
                state[name] = value.clone()
        if hasattr(env._sensor_model, "noise_distribution"):
            context["sensor.noise_distribution"] = clock.new_full(
                (self.num_envs, 1), env._sensor_model.noise_distribution).float()
        # Two PID layers have layer-major storage rather than row-major storage.
        for name in ("_integral", "_previous_error", "_has_previous"):
            value = getattr(env._gimbal.controller, name)
            state["gimbal.controller." + name] = value.movedim(1, 0).clone()
        for name, value in env._identified_execution.state_memory().items():
            state["identified." + name] = value.clone()

    @torch.no_grad()
    def _snapshot(self, *, terminal):
        self._capture_finite_checks = []
        try:
            result = self._capture_snapshot(terminal=terminal)
            names, values = zip(*self._capture_finite_checks)
            flags = torch.stack(values)
            if not flags.all():
                invalid = [name for name, flag in zip(names, flags.cpu().tolist()) if not flag]
                raise ValueError("Nonfinite native snapshot fields: " + ", ".join(invalid))
            return result
        finally:
            del self._capture_finite_checks

    def _capture_snapshot(self, *, terminal):
        env = self.env
        sample = env._sample()
        measured = env._observations.assembler.measure(sample)
        clock = self._owned("clock", env._clock(), (), dtype=torch.float64)
        clean, actor = self._actor_frames(measured, env._commands(), clock, terminal=terminal)
        state = {name: self._owned(name, getattr(env.robot.data, field)) for name, field in (
            ("body_link_pose_world_xyzw", "body_link_pose_w"),
            ("body_com_pose_world_xyzw", "body_com_pose_w"),
            ("body_com_velocity_world", "body_com_vel_w"),
            ("joint_position_native", "joint_pos"), ("joint_velocity_native", "joint_vel"))}
        state["joint_position_continuous"] = sample.joint_position_rad.clone()
        state["terrain_contact_force_world_n"] = sample.terrain_contact_force_world_n.clone()
        state["clean_observation"] = clean.clone()
        context = {name: value.clone() for name, value in self._static_context.items()}
        context.update(environment_origin_world_m=sample.environment_origin_world_m.clone(),
                       ground_at_root_world_z_m=sample.ground_at_root_world_z_m.clone(),
                       ground_at_wheel_world_z_m=sample.ground_at_wheel_world_z_m.clone())
        self._capture_memory(state, context, clock)
        indices = env._gimbal.joint_indices
        state["gimbal.position_rad"] = state["joint_position_native"][:, indices].clone()
        state["gimbal.velocity_rad_s"] = state["joint_velocity_native"][:, indices].clone()
        state["gimbal.yaw_sin_cos"] = self._gimbal_sensors.instantaneous()[:, :2]
        self._capture_course(state, context, clock)
        bridge = getattr(env, 'supermimic_recovery_bridge', None)
        if bridge is not None and not terminal:
            base = self.body_names.index('base_link')
            bridge.record_births(self._episode_id, clock, state['body_link_pose_world_xyzw'][:, base],
                                 state['body_com_pose_world_xyzw'][:, base, 2])
        owned = state["target.control_source"].eq(0)
        if env._recovery_preparation is not None:
            owned &= ~env._recovery_preparation.active
        if env._recover_task is not None:
            owned &= ~env._recover_task.hold_mask
        labels = torch.cat((measured.velocity_origin_body_m_s,
                            (measured.wheel_ground_force_n > 2.).float()), -1)
        valid = torch.isfinite(labels).all(-1)
        measurement_valid = getattr(measured, "contact_measurement_valid", None)
        if measurement_valid is not None:
            valid &= measurement_valid
        raw = RawSnapshot(state, context, clean, actor,
                          torch.cat((clean[:, :4], clean[:, 28:35], clean[:, 35:40]), -1),
                          self._episode_id.clone(), self._control_epoch.clone(), owned,
                          clock, self._field_metadata(state, context), labels.clone(), valid)
        projected = self._projector.project(raw)
        # These yaw coordinates are used through circular differences by the
        # source controller. Keep their memory, without teaching a 2*pi jump.
        # PID previous_error/integral are real-valued histories: the source
        # differentiates/accumulates them directly, so they must stay unchanged.
        if env._gimbal.controller.config.joint_limits_rad[0] is not None:
            raise ValueError("Periodic course yaw encoding requires an unlimited yaw joint")
        metadata = projected.field_metadata["state"]
        for name in ("course.gimbal.target", "gimbal.controller.target_angles",
                     "gimbal.controller._last_joint", "gimbal.controller._last_world"):
            angles = projected.state.pop(name)
            yaw = angles[:, :1]
            projected.state[name + ".yaw_sin_cos"] = torch.cat((yaw.sin(), yaw.cos()), -1)
            projected.state[name + ".pitch_rad"] = angles[:, 1:2].clone()
            metadata.pop(name)
            metadata[name + ".yaw_sin_cos"] = {
                "units": "1", "frame": "sin,cos of source " + name + " yaw",
            }
            metadata[name + ".pitch_rad"] = {
                "units": "rad", "frame": "source " + name + " pitch",
            }
        direction = projected.state.pop("course.gimbal.direction")[:, None]
        projected.state["course.gimbal.direction_sin_cos"] = torch.cat(
            (direction.sin(), direction.cos()), -1)
        metadata.pop("course.gimbal.direction")
        metadata["course.gimbal.direction_sin_cos"] = {
            "units": "1", "frame": "sin,cos of requested translation direction",
        }
        # The 20-axis vector also contains yaw. Its six policy axes retain
        # their native indices and physical units; yaw uses the same phase as
        # the explicit gimbal fields instead of retaining a second raw angle.
        yaw_index = self.joint_names.index("M_joint1")
        if yaw_index in env._binding.policy_joint_indices:
            raise ValueError("Independent gimbal yaw cannot be a locomotion policy axis")
        positions = projected.state.pop("joint_position_continuous")
        positions[:, yaw_index] = projected.state["gimbal.yaw_sin_cos"][:, 0]
        name = "joint_position_gimbal_yaw_sine"
        projected.state[name] = positions
        metadata.pop("joint_position_continuous")
        units = ["m" if kind == "prismatic" else "rad" for kind in self.joint_types]
        units[yaw_index] = "1"
        metadata[name] = {
            "units": "native order: " + ",".join(units),
            "frame": ("native joint order: " + ",".join(self.joint_names)
                      + "; M_joint1 is sin(native yaw), cosine in gimbal.yaw_sin_cos;"
                        " all other columns are continuous physical coordinates"),
        }
        return projected

    def set_policy_estimate(self, frame, raw_state):
        from transformer_rl.formal.tasks.protocol import CompletedEstimate
        self.env._completed_estimate = CompletedEstimate._from_actor_observation(
            frame[:, :35], raw_state, self.env._clock().clone())

    def set_training_progress(self, real_transitions, actor_updates):
        self.curriculum.set_progress(real_transitions=real_transitions, actor_updates=actor_updates)

    def training_state_dict(self):
        state = {"format": "supermimic_course_v1", "curriculum": self.curriculum.state_dict(),
                 "course": self.env._terrain_course.state_dict(),
                 "control_timing": self.env.control_timing_state_dict(),
                 "sensors": self.env._sensor_model.checkpoint_state_dict()}
        platform = getattr(self.env, 'supermimic_platform_course', None)
        if platform is not None:
            state['platform_course'] = platform.state_dict()
        return state

    def load_training_state_dict(self, state):
        if state["format"] != "supermimic_course_v1":
            raise ValueError("Course checkpoint format differs")
        platform = getattr(self.env, 'supermimic_platform_course', None)
        if (platform is None) != ('platform_course' not in state):
            raise ValueError('Platform course checkpoint requires an explicit source migration')
        self.curriculum.load_state_dict(state["curriculum"])
        self.env._terrain_course.restore_after_physical_reset(state["course"])
        if platform is not None:
            platform.restore_after_physical_reset(state['platform_course'])
        self.env.restore_control_timing(state["control_timing"])
        sensor = self.env._sensor_model
        sensor.initialized.zero_()
        sensor.count.zero_()
        sensor._sample = None
        sensor._sample_tick = None
        sensor.load_after_explicit_cold_setup(state["sensors"])
        # New physics episodes, restored sampling streams; never restore solver state.
        self.reset()

    def step(self, action):
        self._health.check()
        recorder = self._substeps
        if recorder is not None:
            recorder.begin()
        try:
            result = super().step(action)
        except BaseException:
            if recorder is not None:
                recorder.abort()
            raise
        if recorder is not None:
            result = replace(result, substeps=self._finish_substeps(result))
        self._health.sample_policy()
        self._health.check()
        samples = self._health.summary["policy_samples"]
        if samples == 1 or samples % 100 == 0:
            self._save_physics_health()
        return result

    def _finish_substeps(self, transition):
        values, timestamps = self._substeps.finish()
        start, end = transition.start.timestamp_s, transition.end.timestamp_s
        offsets = torch.arange(1, self.substeps_per_step + 1, dtype=torch.float64, device=self.device)
        expected = start[:, None] + offsets * self.physics_dt_s
        # The last physics tick is the pre-reset endpoint the replay supervises.
        if (not torch.allclose(timestamps, expected, atol=1e-8, rtol=0)
                or not torch.allclose(timestamps[:, -1], end, atol=1e-8, rtol=0)):
            raise ValueError("Recorded substeps do not cover exactly this policy interval")
        return values

    def _save_physics_health(self):
        target = self._native_log_directory / "native-health.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._health.summary, indent=2) + "\n")
        temporary.replace(target)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._substeps is not None:
                self._substeps.close()
            if self._profiler is not None:
                (self._native_log_directory / "step-profile.json").write_text(
                    json.dumps(self._profiler.summary(), indent=2) + "\n")
                self._profiler.close()
            self._save_physics_health()
            report = {"course": self.env._terrain_course.metrics(),
                      "curriculum": self.curriculum.state_dict()}
            torch.save(report, self._native_log_directory / "course-final.pt")
            self.env.close()
        finally:
            self._health.close()
            self._launcher.close()
