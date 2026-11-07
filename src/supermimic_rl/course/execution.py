"""Identified execution at the pinned full-vehicle environment's effort boundary.

The parent owns action latching, controller feedback, curriculum, gimbal and
reset behavior. A temporary transparent robot proxy intercepts its final
native effort write. Numerically rejected rows are parked until normal reset.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import SimpleNamespace

import torch
from torch import Tensor

from .identified_execution import ACTION_ORDER, IdentifiedExecution


def configure_identified_actuators(cfg, execution: IdentifiedExecution, *, backend: str) -> dict:
    """Set requested active-axis properties before native scene creation.

Newton has no independent static/dynamic friction pair. Its current integration
must explicitly choose software_dynamic; these returned values are requests,
not native readback. Passive-joint damping and auxiliary actuators stay intact.
"""
    if backend not in ("newton", "physx"):
        raise ValueError("Execution backend must be newton or physx")
    if backend == "newton" and execution.friction_mode != "software_dynamic":
        raise ValueError("Newton requires explicit software_dynamic mode; separate native dynamic friction is unsupported")
    if getattr(cfg, "response_spec", None) is not None:
        raise ValueError("Disable legacy response_spec before configuring identified execution")
    parameters, limits = execution.native_parameters(), execution.native_effort_limits()
    selected = []
    robot_configs = [cfg.robot_cfg]
    scene_robot = getattr(getattr(cfg, "scene", None), "robot", None)
    if scene_robot is not None and scene_robot is not cfg.robot_cfg:
        robot_configs.append(scene_robot)
    # Validate the entire assignment before changing any actuator object.
    for robot in robot_configs:
        groups = []
        for name, actuator in robot.actuators.items():
            joints = tuple(actuator.joint_names_expr)
            if any(joint in ACTION_ORDER for joint in joints):
                if not set(joints) <= set(ACTION_ORDER) or len(set(joints)) != len(joints):
                    raise ValueError("Active actuator groups require explicit names without auxiliary/regex entries")
                groups.append((name, actuator, joints))
        owned = [joint for _, _, joints in groups for joint in joints]
        if sorted(owned) != sorted(ACTION_ORDER):
            raise ValueError("Each identified active axis must belong to exactly one explicit actuator group")
        selected.extend(groups)
    for _, actuator, joints in selected:
        for field in ("armature", "friction", "dynamic_friction", "viscous_friction"):
            setattr(actuator, field, {joint: parameters[joint][field] for joint in joints})
        actuator.stiffness = 0.
        actuator.damping = 0.
        actuator.effort_limit = {joint: limits[joint] for joint in joints}
        actuator.effort_limit_sim = dict(actuator.effort_limit)
    return dict(backend=backend, friction_mode=execution.friction_mode,
                requested_active_properties=parameters, requested_effort_limits_nm=limits,
                modified_actuator_groups=list(dict.fromkeys(name for name, _, _ in selected)),
                declarative_scene_copy_updated=len(robot_configs) == 2,
                native_readback_required=True)


def read_identified_native_friction(robot, execution: IdentifiedExecution, *, backend: str) -> dict:
    """Verify the solver's properties, not the actuator configuration/cache.

    The current PhysX tensor API reports [static effort, dynamic effort,
    viscous coefficient] in backend joint order. The first two are N.m for
    these revolute axes (Isaac Sim 5+), not legacy dimensionless coefficients.
    This reads all six active axes so an unintended leg friction term is also
    detected. Passive joints and the independent gimbal remain unmodified.
    """
    if backend != "physx" or execution.friction_mode != "native_floor":
        raise ValueError("Native-floor readback requires the explicit PhysX friction mode")
    view = robot.root_view
    names = tuple(view.shared_metatype.dof_names)
    if len(set(names)) != len(names) or set(names) != set(robot.joint_names):
        raise ValueError("PhysX friction readback joint names differ from the articulation")
    values = view.get_dof_friction_properties()
    values = getattr(values, "torch", values)
    if not isinstance(values, Tensor):
        import warp

        values = warp.to_torch(values)
    if values.shape != (execution.count, len(names), 3):
        raise ValueError("PhysX friction readback requires all environments, joints and three components")
    indices = torch.tensor([names.index(name) for name in ACTION_ORDER], device=values.device)
    actual = values.index_select(1, indices).detach().clone()
    fields = ("friction", "dynamic_friction", "viscous_friction")
    parameters = execution.native_parameters()
    expected = actual.new_tensor([[parameters[name][field] for field in fields] for name in ACTION_ORDER])
    if (not torch.isfinite(actual).all()
            or not torch.allclose(actual, expected.expand_as(actual), rtol=1e-6, atol=1e-8)):
        raise RuntimeError("PhysX identified joint friction readback differs from the pinned model")
    actual = actual.cpu().contiguous()
    return {
        "contract": "physx_identified_native_friction_readback_v1", "verified": True,
        "reader": "robot.root_view.get_dof_friction_properties",
        "joint_order": list(ACTION_ORDER), "component_order": list(fields),
        "units": ["N.m", "N.m", "N.m/(rad/s)"], "environment_count": execution.count,
        "requested": expected.cpu().tolist(),
        "actual_min": actual.amin(0).tolist(), "actual_max": actual.amax(0).tolist(),
        "actual_sha256": hashlib.sha256(actual.numpy().tobytes()).hexdigest(),
        "static_identification": [{
            "joint": name, "nominal_nm": row["static_breakaway_nominal_nm"],
            "observed_interval_nm": row["static_breakaway_observed_interval_nm"],
            "exactly_identified": row["static_breakaway_exactly_identified"],
        } for name, row in zip(ACTION_ORDER[4:], execution.wheel_rows)],
        "scope": "Wheel static/dynamic solver ownership; leg static friction remains unidentified and zero",
        "dynamics_validated": False,
    }


@dataclass(frozen=True)
class _PressureState:
    slider_effort_n: Tensor
    compression_m: Tensor
    normalized_compression: Tensor
    within_fitted_curve: Tensor
    within_mechanical_stroke: Tensor
    endpoint_clamped: Tensor
    model_pressure_potential_j: Tensor


class _MeasuredSliderPressure:
    """Diagnostics for the applied measured curve; no second physical force."""

    def __init__(self, execution):
        self.execution = execution
        self.parameters = SimpleNamespace(
            joint_names=("L_spring_slide", "R_spring_slide"),
            compression_at_zero_m=tuple(execution.spring_zero.tolist()),
            stroke_m=tuple(execution.spring_stroke.tolist()),
            force_coefficients_n=None, fitted_u_domain=None,
        )

    def evaluate(self, slider_position_m, *, strength=None):
        e = self.execution
        e._check(slider_position_m, 2, "spring diagnostic position")
        scale = torch.ones(e.count, 2, device=e.device, dtype=e.dtype) if strength is None else strength
        if (scale.shape not in ((e.count, 1), (e.count, 2))
                or scale.device != e.device or scale.dtype != e.dtype):
            raise ValueError("Spring diagnostic strength shape/device/dtype mismatch")
        e._require("Invalid spring diagnostic strength", ~torch.isfinite(scale) | (scale < 0))
        compression = e.spring_zero - slider_position_m
        force = e.physical_spring_force(compression) * scale
        lo, hi = e.geometry_compression_bounds
        return _PressureState(
            force, compression, compression / e.spring_stroke,
            (compression >= e.spring_compression[0]) & (compression <= e.spring_compression[-1]),
            (compression >= 0) & (compression <= e.spring_stroke),
            (compression < lo) | (compression > hi),
            e.spring_potential_j(compression) * scale,
        )


class _EffortIntercept:
    def __init__(self, environment, robot):
        self.environment, self.native = environment, robot
        self.calls = 0
        self.result = None

    def __getattr__(self, name):
        return getattr(self.native, name)

    def set_joint_effort_target(self, target, *args, **kwargs):
        env = self.environment
        if args or kwargs or self.calls:
            raise RuntimeError("Pinned execution requires one full native effort write per physics tick")
        expected = (env.num_envs, len(env._binding.native_joint_names))
        if target.shape != expected:
            raise ValueError("Native effort write must contain the complete articulation")
        indices = env._active_indices
        spring_indices = env._spring_indices
        env._identified_execution._require(
            "Slider pressure would duplicate the measured mount force", target.index_select(1, spring_indices) != 0)
        velocity = env._as_torch(self.native.data.joint_vel).index_select(1, indices)
        preparation = getattr(env, "_recovery_preparation", None)
        enabled = None if preparation is None else ~preparation.passive
        self.result = env._identified_execution.step(
            target.index_select(1, indices), velocity, motor_enabled=enabled)
        applied = target.clone()
        applied.index_copy_(1, indices, self.result.native_effort_nm)
        if env._numerical_had_event:
            applied[env._numerical_invalid] = 0.
        self.calls += 1
        return self.native.set_joint_effort_target(applied)


def make_environment(base_type, execution: IdentifiedExecution):
    """Wrap the 41e full-articulation environment without copying its control loop."""

    class IdentifiedCourseEnvironment(base_type):
        def __init__(self, *args, **kwargs):
            self._identified_execution = execution
            self._numerical_interval_active = False
            self._numerical_invalid = None
            self._numerical_had_event = False
            self._numerical_invalid_total = 0
            self._numerical_start_pose = self._numerical_start_position = None
            super().__init__(*args, **kwargs)
            if (self.num_envs != execution.count or self._device != execution.device
                    or abs(float(self.physics_dt) - execution.dt) > 1e-12):
                raise ValueError("Identified response batch/device/physics clock differs from native environment")
            names = tuple(self._binding.native_joint_names)
            active = tuple(names[index] for index in self._active_indices.tolist())
            if active != ACTION_ORDER:
                raise ValueError("Native active indices differ from identified output-axis order")
            if self._control.drive is not None or getattr(self, "_communication", None) is not None:
                raise ValueError("Legacy response/communication must be disabled for single-owner identified delay")
            if self._motor_envelope is not None:
                raise ValueError("Legacy wheel envelope must be disabled for the identified speed envelope")
            execution.invariants = getattr(self, "_invariants", None)
            self._control.controller.damping[4:] = execution.wheel_model["controller_velocity_p_nm_per_rad_s"]
            self._last_native_effort = torch.zeros_like(self._last_effort)
            self._last_passive_effort = torch.zeros_like(self._last_effort)
            self.identified_execution_receipt = execution.receipt()
            from .coordinates import install_newton_coordinates

            install_newton_coordinates(self)
            self._numerical_invalid = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
            self.numerical_isolation_receipt = {
                'contract': 'physx_finite_coordinate_sampling_row_isolation_v3',
                'native_period_rad': 2 * math.tau,
                'scope': 'finite ambiguous position/speed samples; reject the entire policy interval',
                'rejected_rows': 'restore interval-start pose/joints with zero velocity, then normal episode reset',
                'native_state_modified': 'rejected rows only; never used for learning',
                'unknown_or_nonfinite_errors': 'raise',
            }

        def _joint_positions(self):
            try:
                return super()._joint_positions()
            except RuntimeError as error:
                # These pinned reader failures identify the finite bad rows.
                # Healthy rows retain the native 4*pi coordinate decoder.
                if str(error) not in (
                        'continuous joint position increment is ambiguous',
                        'continuous joint state is nonfinite or angular sampling is ambiguous'):
                    raise
                from transformer_rl.formal.lab_env.joint_coordinates import ContinuousJointCoordinates

                reader = self._joint_coordinates
                if type(reader) is not ContinuousJointCoordinates:
                    raise
                position = self._as_torch(self.robot.data.joint_pos)
                velocity = self._as_torch(self.robot.data.joint_vel)
                nonfinite_position, nonfinite_velocity = ~torch.isfinite(position), ~torch.isfinite(velocity)
                if nonfinite_position.any() or nonfinite_velocity.any():
                    samples = []
                    for field, values, mask in (('position', position, nonfinite_position),
                                                ('velocity', velocity, nonfinite_velocity)):
                        for row, joint in mask.nonzero()[:16].cpu().tolist():
                            samples.append({'field': field, 'row': row,
                                'joint': self._binding.native_joint_names[joint],
                                'value': str(values[row, joint].item())})
                    self._numerical_last_event = {
                        'event': 'nonfinite_native_joint_state', 'physical_tick': self._physical_tick,
                        'action': 'raise_original_error', 'samples': samples}
                    print(json.dumps(self._numerical_last_event), flush=True)
                    raise
                if (not self._numerical_interval_active or self._numerical_invalid is None
                        or reader._tick is None or self._physical_tick != reader._tick + 1):
                    raise
                raw = position.index_select(1, reader.indices)
                previous = reader._previous_raw.index_select(1, reader.indices)
                speed = velocity.index_select(1, reader.indices)
                if not torch.isfinite(previous).all():
                    raise
                period = 2 * math.tau
                turns = torch.round((previous.double() - raw.double()) / period)
                delta = raw.double() - previous.double() + turns * period
                bad_increment = delta.abs() >= math.pi
                bad_speed = speed.abs() * reader.physics_dt_s >= math.pi
                bad = bad_increment | bad_speed
                rows = bad.any(-1).nonzero().flatten()
                if not len(rows):
                    raise
                new_rows = rows[~self._numerical_invalid[rows]]
                self._numerical_invalid_total += len(new_rows)
                self._numerical_invalid[rows] = True
                self._numerical_had_event = True
                self._terminated[rows] = True
                pairs = bad.nonzero()[:16]
                self._numerical_last_event = {
                    'event': 'numerical_interval_rejected', 'physical_tick': self._physical_tick,
                    'row_count': len(rows), 'rows': rows[:16].cpu().tolist(),
                    'row_joint_samples': pairs.cpu().tolist(),
                    'joint_names': [self._binding.native_joint_names[index]
                                    for index in reader.indices.cpu().tolist()],
                    'delta_rad': delta[pairs[:, 0], pairs[:, 1]].cpu().tolist(),
                    'speed_rad_s': speed[pairs[:, 0], pairs[:, 1]].cpu().tolist(),
                    'ambiguous_increment': bad_increment[pairs[:, 0], pairs[:, 1]].cpu().tolist(),
                    'ambiguous_speed': bad_speed[pairs[:, 0], pairs[:, 1]].cpu().tolist(),
                    'invalid_intervals_total': self._numerical_invalid_total,
                }
                print(json.dumps(self._numerical_last_event), flush=True)
                self._park_numerical_rows(rows)
                return reader.read(self._as_torch(self.robot.data.joint_pos),
                                   self._as_torch(self.robot.data.joint_vel), self._physical_tick)

        def _park_numerical_rows(self, rows):
            # Waiting for the policy boundary lets a failed articulation turn
            # into NaNs during the remaining physics ticks. Restore only its
            # last finite pose now. Keep task/reward identity until normal
            # reset: the whole transition is already excluded from learning.
            position = self._numerical_start_position.index_select(0, rows)
            pose = self._numerical_start_pose.index_select(0, rows)
            self.robot.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=rows)
            self.robot.write_root_com_velocity_to_sim_index(
                root_velocity=pose.new_zeros(len(rows), 6), env_ids=rows)
            self.robot.write_joint_position_to_sim_index(position=position, env_ids=rows)
            self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(position), env_ids=rows)
            self._joint_coordinates.reset_rows(rows, position)
            self.scene.write_data_to_sim()
            self.sim.forward()
            self.scene.update(dt=0.)

        def _observe_substep(self):
            # Resolve a failed joint row before terrain queries dereference its
            # root/wheel poses. The parent otherwise queries terrain first.
            self._joint_positions()
            return super()._observe_substep()

        def _pre_physics_step(self, actions):
            self._numerical_invalid.zero_()
            self._numerical_had_event = False
            self._numerical_start_position = self._as_torch(self.robot.data.joint_pos).clone()
            self._numerical_start_pose = torch.cat((self._root_link_position(), self._root_link_quat()), -1)
            return super()._pre_physics_step(actions)

        def step(self, action):
            self._numerical_interval_active = True
            try:
                result = super().step(action)
            finally:
                self._numerical_interval_active = False
            extras = result[4]
            invalid = self._numerical_invalid.clone()
            original_valid = extras.get('interval_observation_valid', torch.ones_like(invalid))
            extras['interval_observation_valid'] = original_valid & ~invalid
            extras['numerical_invalid'] = invalid
            extras['numerical_rejected_intervals'] = invalid.sum()
            extras['numerical_rejected_intervals_total'] = self._numerical_invalid_total
            if self._numerical_had_event and 'termination_counts' in extras:
                counts = extras['termination_counts'].clone()
                counts[0] -= (invalid & original_valid & result[2]).sum()
                extras['termination_counts'] = counts
            return result

        def _build_pressure(self):
            return _MeasuredSliderPressure(self._identified_execution)

        def _build_spring_force(self):
            self._spring_source_sha256 = {
                "model_spec.json": self._identified_execution.asset_spec_sha256,
                "gas_spring_model.json": self._identified_execution.receipt()["model_pins"]["gas_spring_model.json"],
            }
            return self._identified_execution.mount_force_element(self.robot.body_names)

        def _apply_action(self):
            if self._numerical_had_event:
                # The parent clears _terminated after _pre_physics_step. Keep
                # a failure found there queued for the normal interval reset.
                self._terminated |= self._numerical_invalid
            native = self.robot
            intercept = _EffortIntercept(self, native)
            self.robot = intercept
            try:
                super()._apply_action()
            finally:
                self.robot = native
            if intercept.calls != 1:
                raise RuntimeError("Parent control did not issue exactly one native effort write")
            result = intercept.result
            # Reward/current diagnostics use actual motor torque after delay;
            # passive loss remains separately auditable and is not motor current.
            self._last_effort.copy_(result.motor_effort_nm)
            self._last_native_effort.copy_(result.native_effort_nm)
            self._last_passive_effort.copy_(result.passive_effort_nm)

        def _reset_idx(self, env_ids):
            self._identified_execution.reset(env_ids)
            result = super()._reset_idx(env_ids)
            for name in ("_last_native_effort", "_last_passive_effort"):
                value = getattr(self, name, None)
                if value is not None:
                    value[env_ids] = 0.
            return result

    return IdentifiedCourseEnvironment
