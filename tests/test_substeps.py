"""Substep truth: schema compatibility, native tick recording, packing and replay."""
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.adapters.packed import PackedEnvironment
from supermimic_rl.adapters.substeps import SubstepRecorder
from supermimic_rl.contracts import Field, Snapshot, StateSchema, Transition
from supermimic_rl.replay import ReplayBuffer


SUBSTEP_FIELDS = (Field("velocity", (2,), "m/s", "body"), Field("load", (1,), "N", "wheel"))


def schema(substeps=True):
    return StateSchema((Field("position", (2,), "m", "world"),), (Field("mass", (1,)),),
                       source="substep-test",
                       substep_fields=SUBSTEP_FIELDS if substeps else (),
                       substeps_per_step=4 if substeps else 0)


def test_schemas_without_substeps_keep_their_original_serialization():
    legacy = schema(substeps=False)
    payload = legacy.to_dict()
    assert "substep_fields" not in payload and "substeps_per_step" not in payload
    assert StateSchema.from_dict(payload) == legacy
    recorded = schema()
    assert recorded.substep_dim == 3
    assert StateSchema.from_dict(recorded.to_dict()) == recorded
    assert recorded.fingerprint != legacy.fingerprint
    with pytest.raises(ValueError, match="substep count"):
        StateSchema(legacy.state_fields, legacy.context_fields, substep_fields=SUBSTEP_FIELDS)


def test_substeps_pack_in_field_order_and_reject_drift():
    packed = schema().pack_substeps({"load": torch.full((2, 4, 1), 7.),
                                     "velocity": torch.arange(16.).reshape(2, 4, 2)})
    assert packed.shape == (2, 4, 3)
    torch.testing.assert_close(packed[1, 3], torch.tensor([14., 15., 7.]))
    with pytest.raises(ValueError, match="substep count"):
        schema().pack_substeps({"load": torch.zeros(2, 3, 1), "velocity": torch.zeros(2, 3, 2)})
    with pytest.raises(ValueError, match="do not match"):
        schema().pack_substeps({"velocity": torch.zeros(2, 4, 2)})
    with pytest.raises(ValueError, match="Nonfinite"):
        schema().pack_substeps({"load": torch.full((2, 4, 1), float("nan")),
                                "velocity": torch.zeros(2, 4, 2)})


def snapshot(value, *, episode=0):
    state = torch.full((2, 2), float(value))
    state[1] += 1000  # Distinguish the environments in sampled windows.
    return Snapshot(state, torch.zeros(2, 35), torch.zeros(2, 11),
                    torch.ones(2, 1), torch.full((2,), episode, dtype=torch.int64),
                    torch.zeros(2, dtype=torch.int64), torch.ones(2, dtype=torch.bool))


def transition(step, substeps):
    return Transition(snapshot(step), torch.zeros(2, 6), snapshot(step + 1), torch.zeros(2),
                      torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool),
                      torch.ones(2, dtype=torch.bool), substeps)


def test_transition_requires_declared_substeps_only():
    rows = torch.zeros(2, 4, 3)
    transition(0, rows).validate(schema())
    with pytest.raises(ValueError, match="substeps must be"):
        transition(0, None).validate(schema())
    with pytest.raises(ValueError, match="substeps must be"):
        transition(0, torch.zeros(2, 3, 3)).validate(schema())
    with pytest.raises(ValueError, match="declares no substep"):
        transition(0, rows).validate(schema(substeps=False))


def test_replay_windows_align_substeps_with_actions_and_round_trip(tmp_path):
    replay = ReplayBuffer(schema(), num_envs=2, capacity=16)
    for step in range(5):
        rows = torch.full((2, 4, 3), float(step))
        rows[1] += 100
        replay.add(transition(step, rows))
    batch = replay.sample(32, 3, generator=torch.Generator().manual_seed(3))
    assert batch.substeps.shape == (32, 3, 4, 3)
    # Each window's substeps belong to the same environment and interval as its states.
    environment = batch.states[:, 0, 0].ge(1000).float()
    first_step = batch.states[:, 0, 0] - 1000 * environment
    expected = first_step[:, None] + torch.arange(3.)[None] + 100 * environment[:, None]
    torch.testing.assert_close(batch.substeps[:, :, 0, 0], expected)
    torch.testing.assert_close(batch.substeps[:, :, 3, 2], expected)
    path = tmp_path / "replay.pt"
    replay.save(path)
    restored = ReplayBuffer(schema(), num_envs=2, capacity=16)
    restored.load(path)
    torch.testing.assert_close(restored._storage["substeps"], replay._storage["substeps"])
    legacy = ReplayBuffer(schema(substeps=False), num_envs=2, capacity=16)
    assert "substeps" not in legacy._storage
    assert legacy.sample is not None


class FakeNative:
    """The pinned loop: decimation physics ticks, each followed by _observe_substep."""

    def __init__(self):
        self.cfg = SimpleNamespace(decimation=4)
        self.num_envs, self.device = 2, "cpu"
        self._binding = SimpleNamespace(native_body_names=("base_link", "L_link3", "R_link3"),
                                        native_joint_names=("a", "b"))
        self.tick = 0
        self._command = torch.tensor([[1., 0., .5, .3], [-1., 0., 0., .25]])
        self._command_mode = torch.tensor([0, 4])
        self._request_height = torch.tensor([0., .1])

    def _observe_substep(self):
        t = float(self.tick)
        n = self.num_envs
        self._last_sample = SimpleNamespace(
            root_link_position_world_m=torch.full((n, 3), t),
            root_link_orientation_xyzw=torch.tensor([[0., 0., 0., 1.]]).expand(n, -1),
            root_com_position_world_m=torch.full((n, 3), t + .01),
            root_com_velocity_world_m_s=torch.full((n, 3), t), angular_velocity_world_rad_s=torch.zeros(n, 3),
            joint_position_rad=torch.full((n, 2), t), joint_velocity_rad_s=torch.full((n, 2), -t),
            ground_at_root_world_z_m=torch.zeros(n), ground_at_wheel_world_z_m=torch.zeros(n, 2),
            body_normal_load_n=torch.full((n, 3), t), wheel_support_load_n=torch.full((n, 2), 2 * t),
            terrain_contact_force_world_n=torch.zeros(n, 3, 3),
            contact_measurement_valid=torch.ones(n, dtype=torch.bool),
            timestamp_s=torch.full((n,), t * .005, dtype=torch.float64))
        self._last_state = SimpleNamespace(
            velocity_origin_body_m_s=torch.full((n, 3), t), angular_velocity_body_rad_s=torch.zeros(n, 3),
            gravity_body=torch.tensor([[0., 0., -1.]]).expand(n, -1), height_above_ground_m=torch.full((n,), .3))
        self._last_effort = torch.full((n, 6), t)
        self.robot = SimpleNamespace(data=SimpleNamespace(
            body_link_pose_w=torch.full((n, 3, 7), t), body_com_vel_w=torch.full((n, 3, 6), -t)))

    @staticmethod
    def _as_torch(value):
        return value

    def step(self, ticks=4):
        for _ in range(ticks):
            self.tick += 1
            self._observe_substep()


def test_recorder_copies_each_tick_and_restores_the_native_method():
    native = FakeNative()
    recorder = SubstepRecorder(native).install()
    native.step()  # Outside an adapter step: not recorded.
    recorder.begin()
    native.step()
    values, stamps = recorder.finish()
    torch.testing.assert_close(values["root_link_position_world_m"][0, :, 0], torch.tensor([5., 6., 7., 8.]))
    torch.testing.assert_close(values["wheel_support_load_n"][1, -1], torch.tensor([16., 16.]))
    torch.testing.assert_close(values["command_mode"][:, 0, 0], torch.tensor([0., 4.]))
    torch.testing.assert_close(stamps[0], torch.tensor([5., 6., 7., 8.], dtype=torch.float64) * .005)
    # Returned rows are owned; the next interval reuses the internal buffer.
    recorder.begin()
    native.step()
    assert values["root_link_position_world_m"][0, 0, 0] == 5
    recorder.finish()
    recorder.begin()
    with pytest.raises(RuntimeError, match="more physics substeps"):
        native.step(ticks=5)
    recorder.begin()
    native.step(ticks=3)
    with pytest.raises(RuntimeError, match="Recorded 3 of 4"):
        recorder.finish()
    recorder.close()
    assert "_observe_substep" not in vars(native)
    assert not hasattr(native, "_supermimic_substep_recorder")


def test_recorder_requires_lab_contact_measurements():
    native = FakeNative()
    original = native._observe_substep

    def without_loads():
        original()
        del native._last_sample.wheel_support_load_n

    native._observe_substep = without_loads
    recorder = SubstepRecorder(native).install()
    recorder.begin()
    with pytest.raises(ValueError, match="Lab contact measurement"):
        native.step(ticks=1)


class SubstepAdapter:
    substep_fields = SUBSTEP_FIELDS
    substeps_per_step = 4

    def __init__(self):
        self.current = self._raw(0.)

    @staticmethod
    def _raw(value):
        return SimpleNamespace(
            state={"position": torch.full((2, 2), value)}, context={"mass": torch.ones(2, 1)},
            observation=torch.zeros(2, 35), actor_observation=None, goal=torch.zeros(2, 11),
            episode_id=torch.zeros(2, dtype=torch.int64), control_epoch=torch.zeros(2, dtype=torch.int64),
            policy_owned=torch.ones(2, dtype=torch.bool),
            field_metadata={"state": {"position": {"units": "m", "frame": "world"}}})

    def reset(self):
        return self.current

    def step(self, action):
        start, end = self.current, self._raw(1.)
        self.current = end
        return SimpleNamespace(start=start, end=end, action=action, reward=torch.zeros(2),
                               terminated=torch.zeros(2, dtype=torch.bool),
                               truncated=torch.zeros(2, dtype=torch.bool), valid=torch.ones(2, dtype=torch.bool),
                               substeps={"velocity": torch.ones(2, 4, 2), "load": torch.zeros(2, 4, 1)})


def test_packed_environment_freezes_declared_substeps_and_packs_each_transition():
    environment = PackedEnvironment(SubstepAdapter(), source="substep-adapter")
    environment.reset()
    assert environment.schema.substeps_per_step == 4
    assert environment.schema.substep_fields == SUBSTEP_FIELDS
    result = environment.step(torch.zeros(2, 6))
    assert result.substeps.shape == (2, 4, 3)
    torch.testing.assert_close(result.substeps[..., :2], torch.ones(2, 4, 2))
    plain = PackedEnvironment(SubstepAdapter(), source="substep-adapter")
    plain.adapter.substep_fields = ()
    plain.reset()
    assert plain.step(torch.zeros(2, 6)).substeps is None


def course_adapter_with_recorder(monkeypatch):
    from supermimic_rl.course import adapter as course
    from supermimic_rl.adapters.isaac import IsaacAdapter, RawTransition

    native = FakeNative()
    adapter = course.CourseAdapter.__new__(course.CourseAdapter)
    adapter.env, adapter.num_envs, adapter.device = native, 2, torch.device("cpu")
    adapter.physics_dt_s = .005
    adapter._health = SimpleNamespace(check=lambda: None, sample_policy=lambda: None,
                                      summary={"policy_samples": 2})
    adapter._substeps = SubstepRecorder(native).install()
    adapter.substep_fields, adapter.substeps_per_step = adapter._substeps.fields, 4
    stamp = SimpleNamespace(skew=0.)

    def native_step(self, action):
        start_time = torch.full((2,), native.tick * .005, dtype=torch.float64)
        native.step()
        end_time = torch.full((2,), native.tick * .005 + stamp.skew, dtype=torch.float64)
        raw = lambda time: SimpleNamespace(timestamp_s=time)
        return RawTransition(raw(start_time), raw(end_time), action, torch.zeros(2),
                             torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool),
                             torch.ones(2, dtype=torch.bool))

    monkeypatch.setattr(IsaacAdapter, "step", native_step)
    return adapter, native, stamp


def test_course_step_attaches_substeps_covering_exactly_its_interval(monkeypatch):
    adapter, native, stamp = course_adapter_with_recorder(monkeypatch)
    result = adapter.step(torch.zeros(2, 6))
    assert result.substeps["applied_effort_nm"].shape == (2, 4, 6)
    torch.testing.assert_close(result.substeps["root_link_position_world_m"][0, :, 0],
                               torch.tensor([1., 2., 3., 4.]))
    stamp.skew = .005
    with pytest.raises(ValueError, match="exactly this policy interval"):
        adapter.step(torch.zeros(2, 6))


def test_course_step_failure_leaves_the_recorder_idle(monkeypatch):
    from supermimic_rl.adapters.isaac import IsaacAdapter

    adapter, native, _ = course_adapter_with_recorder(monkeypatch)

    def failing(self, action):
        native.step(ticks=2)
        raise RuntimeError("native failure")

    monkeypatch.setattr(IsaacAdapter, "step", failing)
    with pytest.raises(RuntimeError, match="native failure"):
        adapter.step(torch.zeros(2, 6))
    assert adapter._substeps._index is None
    native.step()  # Later ticks outside a step are ignored.


def test_audit_fields_record_backend_body_motion_as_returned():
    native = FakeNative()
    recorder = SubstepRecorder(native, audit=True).install()
    names = [field.name for field in recorder.fields]
    assert names[:2] == ["body_link_pose_world_xyzw", "body_com_velocity_reported_world"]
    assert "native body order: base_link,L_link3,R_link3" in recorder.fields[0].frame
    recorder.begin()
    native.step()
    values, _ = recorder.finish()
    torch.testing.assert_close(values["body_com_velocity_reported_world"][0, :, 0, 0],
                               -torch.tensor([1., 2., 3., 4.]))
    assert "body_link_pose_world_xyzw" not in dict.fromkeys(
        field.name for field in SubstepRecorder(FakeNative()).fields)


def test_step_profiler_times_wrapped_methods_and_restores_them():
    from supermimic_rl.adapters.profiling import StepProfiler

    native = FakeNative()
    profiler = StepProfiler("cpu").wrap(native, "step", "native.step").wrap(native, "_observe_substep", "tick")
    native.step()
    summary = profiler.summary()
    assert summary["native.step"]["calls"] == 1 and summary["tick"]["calls"] == 4
    profiler.close()
    assert "step" not in vars(native) and "_observe_substep" not in vars(native)


def test_step_profiler_skips_frozen_owners():
    from dataclasses import dataclass
    from supermimic_rl.adapters.profiling import StepProfiler

    @dataclass(frozen=True)
    class Frozen:
        def evaluate(self):
            return 1

    profiler = StepProfiler("cpu").wrap(Frozen(), "evaluate", "frozen")
    assert profiler.summary() == {"_skipped": ["frozen"]}
