"""Synthetic native-boundary checks; these never import or start Isaac."""

from types import SimpleNamespace
import hashlib
import json
import math

import pytest
import torch

from supermimic_rl.adapters.isaac import IsaacAdapter
from supermimic_rl.adapters.packed import PackedEnvironment
from supermimic_rl.adapters.factory import (
    _EpisodeCommandSampler, _OwnedIsaacAdapter, _validate_episode_commands,
    _validate_training_observer, _verified_inputs,
)


def test_newton_static_mechanics_use_public_data_with_explicit_frame_metadata():
    native = NativeFixture()
    native.mechanics_backend = "newton"
    del native.robot.root_view
    native.robot.data.body_mass = torch.full((2, 19), 2.)
    native.robot.data.body_com_pos_b = torch.full((2, 19, 3), .02)
    native.robot.data.body_inertia = torch.eye(3).flatten().expand(2, 19, 9).clone()
    adapter = IsaacAdapter(native)
    frame = adapter.reset()
    assert torch.equal(frame.context["body_mass_kg"], native.robot.data.body_mass)
    assert torch.equal(frame.context["body_com_pose_local_xyzw"][..., :3], native.robot.data.body_com_pos_b)
    assert torch.equal(frame.context["body_com_pose_local_xyzw"][..., 3:],
                       torch.tensor([0., 0., 0., 1.]).expand(2, 19, 4))
    metadata = frame.field_metadata["context"]
    assert "Newton identity COM orientation" in metadata["body_com_pose_local_xyzw"]["frame"]
    assert "expressed in link axes" in metadata["body_inertia_kg_m2"]["frame"]


class NativeFixture:
    """Mutable native buffers with the same pre-reset callback ordering."""

    def __init__(self, *, sensors=False, formal=False):
        self.num_envs, self.device = 2, "cpu"
        self.step_dt, self.physics_dt = 0.02, 0.005
        self.cfg = SimpleNamespace(static_plane_height_m=0.0, decimation=4)
        self._physical_tick = 0
        self._forward_scanners = ()
        self._terrain_course = self._task_sampler = self._recover_task = None
        self._recovery_preparation = self._birth_randomization = self._skill_fsm = None
        self._formal_reward = self._failure_latch = None
        self._sensor_model = self._action_transport = self._action_latch = None
        self._communication = self._episode_randomization = None
        self._binding = SimpleNamespace(
            native_body_names=tuple(f"body_{i}" for i in range(19)),
            native_joint_names=tuple(f"joint_{i}" for i in range(18)),
        )
        pose = torch.zeros(2, 19, 7)
        pose[..., 6] = 1
        local_com = pose.clone()
        self.robot = SimpleNamespace(
            body_names=self._binding.native_body_names,
            joint_names=self._binding.native_joint_names,
            data=SimpleNamespace(
                body_link_pose_w=pose.clone(), body_com_pose_w=pose.clone(),
                body_com_vel_w=torch.zeros(2, 19, 6),
                joint_pos=torch.zeros(2, 18), joint_vel=torch.zeros(2, 18),
            ),
            root_view=SimpleNamespace(
                get_masses=lambda: torch.ones(2, 19),
                get_coms=lambda: local_com,
                get_inertias=lambda: torch.ones(2, 19, 9),
            ),
        )
        self._action_bounds = torch.tensor([3., 3., 3., 3., 9., 9.])
        controller = SimpleNamespace(**{name: torch.ones(6) for name in (
            "nominal", "effort_bounds", "stiffness", "damping")})
        controller.action_bounds = self._action_bounds
        parameters = SimpleNamespace(**{name: (0.1,) * 6 for name in (
            "additional_armature", "coulomb_effort", "viscous_effort_per_speed", "effective_time_constant_s")})
        self._control = SimpleNamespace(controller=controller, drive=SimpleNamespace(
            state=torch.zeros(2, 6), _tau_scale=torch.ones(2, 6), parameters=parameters))
        self._wheel_materials = SimpleNamespace(features=lambda: torch.ones(2, 6))
        self._joint_coordinates = SimpleNamespace(
            _previous_raw=torch.zeros(2, 18), _position=torch.zeros(2, 18),
            _turns=torch.zeros(2, 4, dtype=torch.long))
        for name in ("_previous_action", "_issued_action", "_before_action", "_last_effort",
                     "_last_policy_velocity", "_last_joint_acceleration"):
            setattr(self, name, torch.zeros(2, 6))
        self._previous_control_source = torch.zeros(2, dtype=torch.long)
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self._targets = SimpleNamespace(
            issued_action=torch.zeros(2, 6), raw_leg_position_rad=torch.zeros(2, 4),
            admitted_leg_position_rad=torch.zeros(2, 4), wheel_velocity_rad_s=torch.zeros(2, 2),
            pair_projected=torch.zeros(2, 2, dtype=torch.bool), common_turns=torch.zeros(2, 2),
            controller_issued_at_s=torch.zeros(2, dtype=torch.float64),
            control_source=torch.zeros(2, dtype=torch.long),
        )
        self.command_mode = torch.zeros(2, dtype=torch.long)
        self.request_height = torch.zeros(2)
        self._observations = SimpleNamespace(assembler=SimpleNamespace(
            measure=lambda sample: sample, clean_sensors=lambda state: state,
            actor=lambda sample, commands, now_s: self._clean_frame(sample),
        ))
        self.fail_after_capture = False
        self.skip_capture = False
        self.terminal_action = None
        if sensors:
            self._sensor_model = SimpleNamespace(
                capacity=3, count=torch.ones(2, dtype=torch.long), physical_tick=0,
                values=torch.full((3, 2, 18), float("nan")),
                times=torch.full((3, 2), float("nan"), dtype=torch.float64),
                heading=torch.full((3, 2, 2), float("nan")),
                initialized=torch.ones(2, dtype=torch.bool), lags=torch.ones(2, 4, dtype=torch.long),
                nominal=torch.zeros(2, dtype=torch.bool), noise_scale=torch.ones(2),
                _halfwidth=torch.ones(18), uplink_delay_s=None,
            )
        if formal:
            composer = SimpleNamespace(_window=None)
            for name in ("_previous", "_previous2", "_last_dq"):
                setattr(composer, name, torch.zeros(2, 6))
            for name in ("_action_count", "_success_seen"):
                setattr(composer, name, torch.zeros(2, dtype=torch.long))
            composer._dq_valid = composer._ended = torch.zeros(2, dtype=torch.bool)
            composer._time = torch.zeros(2, dtype=torch.float64)
            self._formal_reward = SimpleNamespace(composer=composer)
            self._failure_latch = SimpleNamespace(
                contact_time_s=torch.zeros(2, dtype=torch.float64),
                pose_time_s=torch.zeros(2, dtype=torch.float64),
                failure_reasons=torch.zeros(2, dtype=torch.long))
            self._skill_fsm = SimpleNamespace(active=torch.zeros(2, dtype=torch.bool))

    def _clock(self):
        return torch.full((2,), self._physical_tick * self.physics_dt, dtype=torch.float64)

    def _sample(self):
        return SimpleNamespace(
            joint_position_rad=self._joint_coordinates._position,
            terrain_contact_force_world_n=torch.ones(2, 19, 3),
            environment_origin_world_m=torch.zeros(2, 3),
            ground_at_root_world_z_m=torch.zeros(2),
            ground_at_wheel_world_z_m=torch.zeros(2, 2), timestamp_s=self._clock(),
        )

    def _commands(self):
        return SimpleNamespace(mode=self.command_mode, request_height_m=self.request_height)

    @staticmethod
    def _clean_frame(sample):
        frame = torch.arange(35, dtype=torch.float32)[None].repeat(2, 1)
        frame[:, 10] = sample.joint_position_rad[:, 0]
        return frame

    def _capture_terminal(self):
        self._terminal_frame = self._clean_frame(self._sample()) + 100

    def _publish(self):
        self.obs_buf = {
            "policy": (self._clean_frame(self._sample()) + 100).repeat(1, 5),
            "critic": torch.full((2, 182), -999.0),
        }

    def reset(self):
        self.robot.data.joint_pos.zero_()
        self._joint_coordinates._position.fill_(10)
        self._control.drive.state.zero_()
        self._targets.control_source.zero_()
        self._targets.controller_issued_at_s.copy_(self._clock())
        if self._sensor_model is not None:
            sensor = self._sensor_model
            sensor.physical_tick = self._physical_tick
            slot = self._physical_tick % sensor.capacity
            sensor.values[slot].fill_(7)
            sensor.times[slot] = self._clock()
            sensor.heading[slot].fill_(1)
        self._publish()

    def step(self, action):
        self._physical_tick += 4
        self._policy_generated_at_s = self._clock() - self.step_dt
        self._issued_action.copy_(action.clamp(-self._action_bounds, self._action_bounds))
        self._previous_action.copy_(self._issued_action)
        self._targets.issued_action.copy_(self._issued_action)
        self.robot.data.joint_pos.fill_(12)
        self._joint_coordinates._position.fill_(123)
        self.robot.data.body_com_vel_w.fill_(9)
        self._control.drive.state.fill_(42)
        self.episode_length_buf.add_(1)
        if self._formal_reward is not None:
            self._formal_reward.composer._previous.fill_(11)
            self._failure_latch.pose_time_s.fill_(0.12)
        if not self.skip_capture:
            self._capture_terminal()
        if self.fail_after_capture:
            raise RuntimeError("simulated native failure")
        self.terminal_action = self._issued_action.clone()
        # The real environment mutates the done row during auto-reset.
        self.robot.data.joint_pos[0] = -7
        self._joint_coordinates._position[0] = -8
        self._control.drive.state[0] = 0
        self._issued_action[0] = 0
        self._publish()
        return self.obs_buf, torch.tensor([1., 2.]), torch.tensor([True, False]), torch.zeros(2, dtype=torch.bool), {
            "interval_observation_valid": torch.tensor([True, False])}


def test_terminal_capture_precedes_reset_and_owns_native_buffers():
    native = NativeFixture()
    adapter = IsaacAdapter(native)
    initial = adapter.reset()
    transition = adapter.step(torch.full((2, 6), 99.0))
    assert torch.equal(transition.action, native.terminal_action)
    assert transition.end.episode_id.tolist() == [1, 1]
    assert adapter.current.episode_id.tolist() == [2, 1]
    assert adapter.current.control_epoch.tolist() == [2, 1]
    assert transition.valid.tolist() == [True, False]
    assert transition.end.state["joint_position_native"][0, 0] == 12
    assert transition.end.state["joint_position_continuous"][0, 0] == 123
    assert adapter.current.state["joint_position_native"][0, 0] == -7
    assert transition.end.state["drive.state"][0, 0] == 42
    assert adapter.current.state["drive.state"][0, 0] == 0
    assert initial.state["joint_position_continuous"][0, 0] == 10
    native.robot.data.body_com_vel_w.fill_(999)
    assert transition.end.state["body_com_velocity_world"][0, 18, 5] == 9
    assert "_capture_terminal" not in native.__dict__


def test_clean_observation_is_distinct_from_noisy_actor_and_critic():
    adapter = IsaacAdapter(NativeFixture())
    snapshot = adapter.reset()
    assert torch.equal(snapshot.actor_observation, snapshot.observation + 100)
    assert snapshot.goal.shape == (2, 11)
    assert snapshot.goal[0].tolist() == [0., 1., 2., 3., 28., 29., 30., 31., 32., 33., 34.]
    assert "critic" not in snapshot.state
    assert snapshot.context["body_mass_kg"].shape == (2, 19)


def test_sensor_partial_history_uses_real_oldest_sample_and_explicit_mask():
    adapter = IsaacAdapter(NativeFixture(sensors=True))
    snapshot = adapter.reset()
    assert snapshot.state["sensor.values"].shape == (2, 3, 18)
    assert snapshot.state["sensor.values"].eq(7).all()
    assert snapshot.state["sensor.valid_age"].tolist() == [[True, False, False]] * 2
    assert torch.isfinite(snapshot.state["sensor.sample_ages_s"]).all()


def test_real_formal_reward_and_failure_memory_are_preserved():
    adapter = IsaacAdapter(NativeFixture(formal=True))
    adapter.reset()
    transition = adapter.step(torch.zeros(2, 6))
    assert transition.reward.tolist() == [1., 2.]
    assert transition.end.state["reward._previous"].eq(11).all()
    assert transition.end.state["failure.pose_time_s"].eq(0.12).all()


@pytest.mark.parametrize("name", ["body_com_vel_w", "joint_pos"])
def test_missing_native_state_is_an_error_not_a_zero_fill(name):
    native = NativeFixture()
    adapter = IsaacAdapter(native)
    adapter.reset()
    delattr(native.robot.data, name)
    native.reset = lambda: None
    with pytest.raises(ValueError, match="required native field"):
        adapter.reset()


@pytest.mark.parametrize("field", ["_terrain_course", "_recovery_preparation", "_task_sampler"])
def test_unsupported_task_ownership_is_rejected_before_collection(field):
    native = NativeFixture()
    setattr(native, field, object())
    with pytest.raises(ValueError, match="does not support"):
        IsaacAdapter(native)


def test_script_target_is_never_reported_as_policy_data():
    native = NativeFixture()
    adapter = IsaacAdapter(native)
    adapter.reset()
    native._targets.control_source[1] = 2
    with pytest.raises(ValueError, match="script-owned"):
        adapter.step(torch.zeros(2, 6))
    assert adapter.current is None
    assert "_capture_terminal" not in native.__dict__


@pytest.mark.parametrize("failure", ["fail_after_capture", "skip_capture"])
def test_failed_step_restores_original_hook_and_invalidates_current(failure):
    native = NativeFixture()
    adapter = IsaacAdapter(native)
    adapter.reset()
    original = native._capture_terminal
    setattr(native, failure, True)
    with pytest.raises(RuntimeError) as caught:
        adapter.step(torch.zeros(2, 6))
    assert not getattr(caught.value, "__notes__", [])
    assert native._capture_terminal == original
    assert adapter.current is None
    assert not adapter._busy


@pytest.mark.parametrize("bad_row", [0, 1])
def test_continuous_joint_failure_records_only_invalid_axes_and_rows(bad_row):
    native = NativeFixture()
    adapter = IsaacAdapter(native)
    adapter.reset()
    original = native._capture_terminal
    if bad_row == 1:
        native._capture_terminal = original
    reader = native._joint_coordinates
    reader.indices = torch.tensor([0, 3, 5])
    reader.physics_dt_s = native.physics_dt
    reader._tick = 171
    reader._previous_raw.fill_(0.25)
    failure = RuntimeError("continuous joint state is nonfinite or angular sampling is ambiguous")
    proposed = torch.full((2, 6), 99.0)

    def fail(action):
        native._physical_tick = 172
        native._issued_action.copy_(action.clamp(-native._action_bounds, native._action_bounds))
        native.robot.data.joint_pos[:, 1] = float("nan")  # Not a continuous axis.
        native.robot.data.joint_vel[:, 2] = float("inf")  # Not a continuous axis.
        native.robot.data.joint_vel[1 - bad_row, reader.indices] = 600.0
        native.robot.data.joint_pos[bad_row, 0] = float("nan")
        native.robot.data.joint_vel[bad_row, 3] = float("inf")
        native.robot.data.joint_vel[bad_row, 5] = -700.0
        raise failure

    native.step = fail
    with pytest.raises(RuntimeError) as caught:
        adapter.step(proposed)
    assert caught.value is failure
    note = json.loads(failure.__notes__[0])
    json.dumps(note, allow_nan=False)
    diagnostic = note["continuous_joint_diagnostic"]
    assert diagnostic["physical_tick"] == 172
    assert diagnostic["reader_tick"] == 171
    assert diagnostic["physics_dt_s"] == 0.005
    assert len(diagnostic["rows"]) == 1
    row = diagnostic["rows"][0]
    assert row["row"] == bad_row
    assert row["proposed_action"] == [99.0] * 6
    assert row["issued_action"] == [3., 3., 3., 3., 9., 9.]
    assert row["joints"] == [
        {"joint_index": 0, "joint_name": "joint_0", "q_rad": "nan", "dq_rad_s": 0.,
         "previous_raw_rad": 0.25, "reasons": ["nonfinite_position"]},
        {"joint_index": 3, "joint_name": "joint_3", "q_rad": 0., "dq_rad_s": "inf",
         "previous_raw_rad": 0.25, "reasons": ["nonfinite_velocity"]},
        {"joint_index": 5, "joint_name": "joint_5", "q_rad": 0., "dq_rad_s": -700.,
         "previous_raw_rad": 0.25, "reasons": ["angular_speed_ambiguous"]},
    ]
    assert adapter.current is None
    assert not adapter._busy
    assert native._capture_terminal == original
    if bad_row == 1:
        assert native.__dict__["_capture_terminal"] is original
    else:
        assert "_capture_terminal" not in native.__dict__
    assert reader._tick == 171
    assert reader._previous_raw.eq(0.25).all()
    assert torch.isnan(native.robot.data.joint_pos[bad_row, 0])


def test_continuous_increment_diagnostic_matches_native_four_pi_correction():
    native = NativeFixture()
    adapter = IsaacAdapter(native)
    adapter.reset()
    original = native._capture_terminal
    reader = native._joint_coordinates
    reader.indices = torch.tensor([0, 3, 5])
    reader.physics_dt_s = native.physics_dt
    reader._tick = 20
    failure = RuntimeError("continuous joint position increment is ambiguous")

    def fail(action):
        native._physical_tick = 21
        native._issued_action.copy_(action)
        native.robot.data.joint_pos[1, 0] = 4 * math.pi + 0.25
        native.robot.data.joint_pos[1, 3] = 3.5
        native.robot.data.joint_pos[1, 5] = -3.5
        native.robot.data.joint_pos[0, 1] = 3.5  # Not a continuous axis.
        raise failure

    native.step = fail
    with pytest.raises(RuntimeError) as caught:
        adapter.step(torch.zeros(2, 6))
    assert caught.value is failure
    diagnostic = json.loads(failure.__notes__[0])["continuous_joint_diagnostic"]
    assert diagnostic["physical_tick"] == 21
    assert diagnostic["reader_tick"] == 20
    assert [row["row"] for row in diagnostic["rows"]] == [1]
    joints = diagnostic["rows"][0]["joints"]
    assert [joint["joint_index"] for joint in joints] == [3, 5]
    assert [joint["joint_name"] for joint in joints] == ["joint_3", "joint_5"]
    assert [joint["corrected_increment_rad"] for joint in joints] == [3.5, -3.5]
    assert all(joint["reasons"] == ["angular_increment_ambiguous"] for joint in joints)
    assert all(joint["previous_raw_rad"] == joint["dq_rad_s"] == 0 for joint in joints)
    assert adapter.current is None
    assert not adapter._busy
    assert native._capture_terminal == original
    assert "_capture_terminal" not in native.__dict__


def test_missing_diagnostic_field_cannot_replace_original_native_failure():
    native = NativeFixture()
    adapter = IsaacAdapter(native)
    adapter.reset()
    original = native._capture_terminal
    failure = RuntimeError("continuous joint state is nonfinite or angular sampling is ambiguous")

    def fail(action):
        raise failure

    native.step = fail
    with pytest.raises(RuntimeError) as caught:
        adapter.step(torch.zeros(2, 6))
    assert caught.value is failure
    assert "indices" in json.loads(failure.__notes__[0])["continuous_joint_diagnostic_error"]
    assert adapter.current is None
    assert not adapter._busy
    assert native._capture_terminal == original
    assert "_capture_terminal" not in native.__dict__


def test_existing_instance_hook_is_restored():
    native = NativeFixture()
    existing = native._capture_terminal
    native._capture_terminal = existing
    adapter = IsaacAdapter(native)
    adapter.reset()
    adapter.step(torch.zeros(2, 6))
    assert native.__dict__["_capture_terminal"] is existing


def test_nonfinite_physics_and_shape_drift_are_rejected():
    native = NativeFixture()
    adapter = IsaacAdapter(native)
    adapter.reset()
    native.robot.data.body_com_pose_w[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        adapter.step(torch.zeros(2, 6))


def test_named_topology_must_match_exactly():
    native = NativeFixture()
    native.robot.body_names = native.robot.body_names[::-1]
    with pytest.raises(ValueError, match="topology"):
        IsaacAdapter(native)


def test_delay_queues_are_snapshotted_in_native_semantics():
    native = NativeFixture()
    native._action_transport = SimpleNamespace(
        capacity=3, index=1, history=torch.arange(36.).reshape(3, 2, 6),
        _stamp_history=torch.arange(6., dtype=torch.float64).reshape(3, 2),
        _policy_allowed=torch.ones(2, dtype=torch.bool),
        arrived_generated_at_s=torch.ones(2, dtype=torch.float64),
        lags=torch.tensor([1, 2]),
    )
    native._action_latch = SimpleNamespace(held=torch.tensor([True, False]), lags=torch.tensor([1, 2]))
    communication = SimpleNamespace()
    for name, width in (("current", 12), ("previous", 12), ("held_effort", 6),
                        ("held_origins", 4), ("applied_origins", 4)):
        dtype = torch.float64 if name.endswith("origins") else torch.float32
        setattr(communication, name, torch.full((2, width), 7.0, dtype=dtype))
    communication.has_previous = communication.policy_allowed = torch.ones(2, dtype=torch.bool)
    for name in ("applied_at_s", "total_s", "uplink_s", "downlink_s"):
        setattr(communication, name, torch.ones(2, dtype=torch.float64))
    native._communication = communication
    adapter = IsaacAdapter(native)
    snapshot = adapter.reset()
    assert torch.equal(snapshot.state["transport.history"][:, 0], native._action_transport.history[1])
    assert torch.equal(snapshot.state["transport.history"][:, 2], native._action_transport.history[2])
    native._action_transport.history.zero_()
    communication.held_effort.zero_()
    assert snapshot.state["transport.history"].sum() > 0
    assert snapshot.state["communication.held_effort"].eq(7).all()
    assert snapshot.context["latch.lags"].tolist() == [1, 2]


def test_large_clock_ages_survive_actual_float32_learning_pack():
    native = NativeFixture(sensors=True, formal=True)
    native._physical_tick = 6_553_600  # 32768 seconds: float32 cannot resolve 0.1 ms.
    clock = native._clock()
    ages = torch.tensor([0.0001, 0.0025, 0.0051, 0.01], dtype=torch.float64)
    origins = clock[:, None] - ages[None]
    native._communication = SimpleNamespace(
        current=torch.zeros(2, 12), previous=torch.zeros(2, 12),
        held_effort=torch.zeros(2, 6), held_origins=origins.clone(),
        applied_origins=origins.clone(), applied_at_s=clock - ages[0],
        has_previous=torch.ones(2, dtype=torch.bool), policy_allowed=torch.ones(2, dtype=torch.bool),
        total_s=torch.ones(2), uplink_s=torch.ones(2), downlink_s=torch.ones(2),
    )
    native._action_transport = SimpleNamespace(
        capacity=3, index=2, history=torch.zeros(3, 2, 6),
        _stamp_history=clock[None] - ages[:3, None],
        _policy_allowed=torch.ones(2, dtype=torch.bool),
        arrived_generated_at_s=clock - ages[1], lags=torch.ones(2, dtype=torch.long),
    )
    adapter = IsaacAdapter(native)
    adapter.reset()
    native._targets.controller_issued_at_s.copy_(clock - ages[0])
    native._policy_generated_at_s = clock - ages[1]
    native._formal_reward.composer._time.copy_(clock - ages[0])
    sensor = native._sensor_model
    sensor.times[sensor.physical_tick % sensor.capacity] = clock - ages[0]
    raw = adapter._snapshot(terminal=False)
    packer = PackedEnvironment(adapter, source="synthetic-clock-regression")
    packed = packer._pack(raw)
    expected = {
        "target.controller_issued_age_s": [0.0001],
        "target.generated_age_s": [0.0025],
        "communication.held_origin_ages_s": ages.tolist(),
        "communication.applied_origin_ages_s": ages.tolist(),
        "communication.applied_age_s": [0.0001],
        "transport.generated_age_history_s": ages[:3].flip(0).tolist(),
        "transport.arrived_generated_age_s": [0.0025],
        "reward.endpoint_age_s": [0.0001],
        "sensor.sample_ages_s": [0.0001] * 3,
    }
    offset = 0
    for field in packer.schema.state_fields:
        if field.name in expected:
            actual = packed.state[:, offset:offset + field.size]
            wanted = torch.tensor(expected[field.name])[None].expand(2, -1)
            assert torch.allclose(actual, wanted, rtol=0, atol=1e-8), field.name
            assert field.units == "s"
        offset += field.size
    assert raw.timestamp_s.dtype == torch.float64
    assert torch.equal(raw.timestamp_s, clock)
    assert not {"timestamp_s", "target.controller_issued_at_s", "communication.held_origins",
                "communication.applied_origins", "communication.applied_at_s", "transport._stamp_history",
                "transport.arrived_generated_at_s", "reward._time", "sensor.times"} & raw.state.keys()
    assert (clock.float() - (clock - ages[0]).float()).eq(0).all()


def test_step_interval_validation_uses_separate_float64_clock():
    native = NativeFixture()
    native._physical_tick = 6_553_600
    adapter = IsaacAdapter(native)
    adapter.reset()
    transition = adapter.step(torch.zeros(2, 6))
    assert torch.allclose(transition.end.timestamp_s - transition.start.timestamp_s,
                          torch.full((2,), 0.02, dtype=torch.float64), atol=1e-10, rtol=0)
    assert "timestamp_s" not in transition.end.state


def test_float32_absolute_timestamp_is_rejected_instead_of_reconstructing_precision():
    native = NativeFixture()
    native._targets.controller_issued_at_s = native._targets.controller_issued_at_s.float()
    with pytest.raises(ValueError, match="controller issuance timestamp"):
        IsaacAdapter(native).reset()


def test_joint_units_follow_source_types_in_actual_native_order():
    native = NativeFixture()
    joints = [{"name": name, "type": "prismatic" if i in (3, 12) else "revolute"}
              for i, name in enumerate(native._binding.native_joint_names)]
    native._asset = SimpleNamespace(_source_joints=lambda: joints[::-1])
    snapshot = IsaacAdapter(native).reset()
    metadata = snapshot.field_metadata
    units = metadata["state"]["joint_position_native"]["units"].removeprefix("native order: ").split(",")
    assert units == ["m" if i in (3, 12) else "rad" for i in range(18)]
    velocity = metadata["state"]["joint_velocity_native"]["units"]
    assert velocity == "native order: " + ",".join(unit + "/s" for unit in units)
    assert metadata["context"]["body_inertia_kg_m2"]["frame"] == "native SDK declared inertia frame"
    assert set(metadata["state"]) == set(snapshot.state)
    assert set(metadata["context"]) == set(snapshot.context)


def test_flat_episode_randomization_is_rejected_instead_of_claiming_exposure():
    native = NativeFixture()
    native._episode_randomization = SimpleNamespace(config=SimpleNamespace(enabled=True))
    with pytest.raises(ValueError, match="update=0 and exposure=0"):
        IsaacAdapter(native)


@pytest.mark.parametrize("mismatch", [None, "episode", "communication", "delay",
    "leg_position_halfwidth_rad", "leg_velocity_halfwidth_rad_s",
    "wheel_velocity_halfwidth_rad_s", "gyro_halfwidth_rad_s", "gravity_halfwidth"])
def test_factory_requires_exact_instantaneous_noiseless_observer(mismatch):
    sensors = SimpleNamespace(
        leg_position_halfwidth_rad=0., leg_velocity_halfwidth_rad_s=0.,
        wheel_velocity_halfwidth_rad_s=0., gyro_halfwidth_rad_s=0., gravity_halfwidth=0.,
        observation_delays_s=(0.,), nominal_episode_fraction=1., extra_delay_probability=0.,
    )
    sections = {"episode_randomization": SimpleNamespace(enabled=False),
                "communication": SimpleNamespace(enabled=False, total_delay_s=0.)}
    if mismatch in ("episode", "communication"):
        sections["episode_randomization" if mismatch == "episode" else mismatch].enabled = True
    elif mismatch == "delay":
        sensors.observation_delays_s = (0., 0.005)
    elif mismatch is not None:
        setattr(sensors, mismatch, 0.01)
    if mismatch is None:
        _validate_training_observer(sensors, sections)
    else:
        with pytest.raises(ValueError, match="exposure=0|instantaneous_noiseless_clean35"):
            _validate_training_observer(sensors, sections)


@pytest.mark.parametrize("mismatch", ["commit", "dirty", "runtime", "asset"])
def test_factory_pin_validation_precedes_any_simulator_import(tmp_path, monkeypatch, mismatch):
    from supermimic_rl.adapters import factory

    root = tmp_path / "upstream"
    module = root / "src/transformer_rl/formal/lab_env/env.py"
    module.parent.mkdir(parents=True)
    module.write_text("raise AssertionError('Isaac must not be imported')")
    asset = tmp_path / "asset"
    asset.mkdir()
    manifest = b'{}'
    (asset / "manifest.json").write_bytes(manifest)
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps({"task": {}}))
    commit = "a" * 40
    config = {
        "upstream_root": str(root), "expected_commit": commit,
        "runtime_json": str(runtime), "runtime_sha256": hashlib.sha256(runtime.read_bytes()).hexdigest(),
        "asset_directory": str(asset), "asset_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "num_envs": 2, "device": "cuda:0",
        "native_log_directory": str(tmp_path),
    }
    def git_result(args, **kwargs):
        if "rev-parse" in args:
            return SimpleNamespace(stdout=("b" * 40 if mismatch == "commit" else commit))
        return SimpleNamespace(stdout=" M src/file.py" if mismatch == "dirty" else "")
    monkeypatch.setattr(factory.subprocess, "run", git_result)
    if mismatch == "runtime":
        config["runtime_sha256"] = "0" * 64
    if mismatch == "asset":
        config["asset_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        _verified_inputs(config)


def test_native_health_error_blocks_transition_publication_and_close_owns_resources():
    native = NativeFixture()
    closed = []
    native.close = lambda: closed.append("env")
    launcher = SimpleNamespace(close=lambda: closed.append("application"))
    health = SimpleNamespace(calls=0, close=lambda: closed.append("health"))

    def check():
        health.calls += 1
        if health.calls == 4:
            raise RuntimeError("PhysX error: simulation will miss interactions")

    health.check = check
    adapter = _OwnedIsaacAdapter(native, launcher, health, {})
    adapter.reset()
    with pytest.raises(RuntimeError, match="miss interactions"):
        adapter.step(torch.zeros(2, 6))
    assert adapter.current is None
    adapter.close()
    adapter.close()
    assert closed == ["env", "health", "application"]


def test_episode_commands_use_explicit_ranges_and_reproducible_standing_fraction():
    configuration = {"forward_m_s": [-0.5, 0.5], "yaw_rad_s": [-1., 1.],
                     "height_m": [0.23, 0.32], "standing_fraction": 0.2}
    sampler = _EpisodeCommandSampler(configuration, seed=42, device="cpu")
    repeat = _EpisodeCommandSampler(configuration, seed=42, device="cpu")
    ids = torch.arange(4096)
    commands = sampler(ids)
    assert torch.equal(commands, repeat(ids))
    assert commands.shape == (4096, 4)
    assert commands[:, 1].eq(0).all()
    assert commands[:, 0].abs().le(0.5).all()
    assert commands[:, 2].abs().le(1.0).all()
    assert commands[:, 3].ge(0.23).all() and commands[:, 3].le(0.32).all()
    standing = commands[:, [0, 2]].eq(0).all(-1)
    assert 0.17 < standing.float().mean() < 0.23
    assert commands[standing, 3].std() > 0
    assert not torch.equal(sampler(ids), commands)


@pytest.mark.parametrize("replacement", [
    {"height_m": [0., 0.32]}, {"yaw_rad_s": [2., -1.]},
    {"standing_fraction": 1.2}, {"forward_m_s": [float("nan"), 1.]},
])
def test_episode_command_configuration_rejects_invalid_domains(replacement):
    configuration = {"forward_m_s": [-0.5, 0.5], "yaw_rad_s": [-1., 1.],
                     "height_m": [0.23, 0.32], "standing_fraction": 0.2}
    configuration.update(replacement)
    with pytest.raises(ValueError):
        _validate_episode_commands(configuration)
