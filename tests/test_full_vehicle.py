"""Full-vehicle orchestration checks without a simulator or Isaac imports."""

from dataclasses import dataclass
import json
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.adapters.full_vehicle import make_full_vehicle_env
from support.core_projection import CORE_BODIES, NativeArticulation, NewtonNativeArticulation, to_native, to_tensor
from support.full_vehicle import BaseEnv, BoundAsset, StartupDynamics


class FullNative(NativeArticulation):
    def __init__(self):
        super().__init__()
        masses = {"base": 17.16794, "wheel": 1., "leg": 1., "auxYaw": 1.0263, "auxPitch": 1.2475}
        self.root_view.buffers["masses"][:] = torch.tensor([masses[name] for name in self.backend_body_names])
        self.root_view.buffers["coms"].zero_()
        self.root_view.buffers["coms"][..., 6] = 1
        self.root_view.buffers["inertias"][:] = torch.eye(3).flatten()

    def set_joint_effort_target(self, target, joint_ids=None, env_ids=None):
        env_ids = torch.arange(3) if env_ids is None else env_ids
        joint_ids = torch.arange(5) if joint_ids is None else joint_ids
        super().set_joint_effort_target(target, joint_ids, env_ids)

    def write_joint_position_to_sim_index(
        self, *, position, joint_ids=None, env_ids=None, full_data=False, skip_forward=False,
    ):
        super().write_joint_position_to_sim_index(
            position=position, joint_ids=joint_ids, env_ids=env_ids,
            full_data=full_data, skip_forward=skip_forward)

    def write_joint_velocity_to_sim_index(
        self, *, velocity, joint_ids=None, env_ids=None, full_data=False, skip_forward=False,
    ):
        super().write_joint_velocity_to_sim_index(
            velocity=velocity, joint_ids=joint_ids, env_ids=env_ids,
            full_data=full_data, skip_forward=skip_forward)


def make_case(*, scale=None, corrupt_aux=False, native=None, asset=None):
    asset = BoundAsset() if asset is None else asset
    cfg = SimpleNamespace(native=FullNative() if native is None else native, asset=asset,
                          seed=27, scale=scale, corrupt_aux=corrupt_aux, events=[])
    cls = make_full_vehicle_env(
        BaseEnv, asset, {"yaw_velocity_rad_s": (.8, .8), "pitch_position_rad": (.1, .1)},
        native_to_tensor=to_tensor, tensor_to_native=to_native)
    return cls, cfg


@pytest.mark.parametrize("scale", [None, [1., 1.1, .9]])
def test_startup_uses_core_nominal_then_exposes_full_mass_before_reward(scale):
    cls, cfg = make_case(scale=scale)
    env = cls(cfg)
    assert env._full_robot is cfg.native and env.scene.articulations["robot"] is cfg.native
    assert env.events[0] == ("bootstrap_reset", None)
    assert env.events[1] == ("startup_nominal", 19.16794)
    expected = torch.tensor([1., 1., 1.] if scale is None else scale) * 19.16794 + 2.2738
    assert torch.allclose(env.reward_mass, expected, atol=1e-5, rtol=0)
    assert torch.allclose(env._startup_dynamics.mass_scale, (expected / 21.44174)[:, None])
    assert torch.equal(env._startup_dynamics.base_com_local_m, torch.tensor([1., 2., 3.]))
    assert env._gimbal_body_indices.tolist() == [3, 0]
    assert env._gimbal_joint_indices.tolist() == [1, 3]
    assert env._gimbal.command.count_nonzero() == 0
    assert env._gimbal_model == cfg.asset.model


def test_full_receipt_is_json_readback_with_actual_orders_and_not_qualification():
    cls, cfg = make_case(scale=[1., 1.1, .9])
    env = cls(cfg)
    receipt = env.full_vehicle_receipt
    assert receipt["physical_qualification"] is False
    assert receipt["source_com_inertia_validation"] == "readback recorded; source-frame comparison not performed"
    assert receipt["public_body_names"] == list(cfg.native.body_names)
    assert receipt["auxiliary_properties_unchanged_by_core_startup"] is True
    after = receipt["after_core_startup"]
    assert after["body_order"] == list(cfg.native.backend_body_names)
    for name, field in (("mass_kg", "masses"), ("com_pose_local_xyzw", "coms"), ("inertia_kg_m2", "inertias")):
        values = cfg.native.root_view.buffers[field]
        actual = after["properties"][name]
        assert actual["row0"] == values[0].tolist()
        assert actual["minimum_by_body"] == values.amin(0).tolist()
        assert actual["maximum_by_body"] == values.amax(0).tolist()
        assert len(actual["sha256"]) == 64
    assert receipt["gimbal_control"]["effort_limits_nm"] == [4.5, 4.5]
    assert receipt["gimbal_control"]["control_joint_signs"] == [1, -1]
    json.dumps(receipt, allow_nan=False)
    receipt["after_core_startup"]["body_order"].clear()
    assert env.full_vehicle_receipt["after_core_startup"]["body_order"]
    startup_receipt = env._startup_dynamics.receipt
    assert startup_receipt["body_order"] == list(cfg.native.backend_body_names)
    assert startup_receipt["core_startup"]["body_order"] == list(CORE_BODIES)
    assert startup_receipt["native_parameters_sha256"] == after["native_parameters_sha256"]


def test_aux_reset_occurs_before_parent_forward_without_touching_other_rows():
    cls, cfg = make_case()
    env = cls(cfg)
    q_before, dq_before = cfg.native.data.joint_pos.clone(), cfg.native.data.joint_vel.clone()
    env._gimbal.command[:] = 9
    env._gimbal.last_effort[:] = 4
    env._reset_idx(torch.tensor([2]))
    q_expected, dq_expected = q_before.clone(), dq_before.clone()
    q_expected[2, [1, 3]] = torch.tensor([.12, -.34])
    q_expected[2, [2, 4, 0]] = 0
    dq_expected[2] = 0
    event = env.events[-1]
    assert event[0] == "parent_forward"
    assert torch.equal(event[1], q_expected) and torch.equal(event[2], dq_expected)
    assert torch.equal(env._gimbal.command[:2], torch.full((2, 2), 9.))
    assert torch.equal(env._gimbal.command[2], torch.tensor([.8, .1]))
    assert torch.equal(env._gimbal.last_effort[:2], torch.full((2, 2), 4.))
    assert env._gimbal.last_effort[2].count_nonzero() == 0
    assert cfg.native.effort[2].count_nonzero() == 0


def test_apply_action_keeps_core_effort_and_drives_only_named_aux_joints():
    cls, cfg = make_case()
    env = cls(cfg)
    env._reset_idx(torch.arange(3))
    cfg.native.data.joint_pos[:, [1, 3]] = torch.tensor([.2, -.1])
    cfg.native.data.joint_vel[:, [1, 3]] = torch.tensor([.3, -.2])
    env._apply_action()
    assert torch.equal(cfg.native.effort[:, [2, 4, 0]], torch.full((3, 3), 2.))
    assert torch.equal(cfg.native.effort[:, [1, 3]], env._gimbal.last_effort)
    assert torch.allclose(env._gimbal.last_effort, torch.tensor([.25, .1]).expand(3, 2), atol=1e-6)
    assert cfg.native.public_calls[-2][1].tolist() == [2, 4, 0]
    assert cfg.native.public_calls[-1][1].tolist() == [1, 3]


def test_empty_reset_does_not_advance_gimbal_sampler():
    cls, cfg = make_case()
    env = cls(cfg)
    rng = env._gimbal.generator.get_state().clone()
    env._reset_idx(torch.empty(0, dtype=torch.long))
    assert torch.equal(rng, env._gimbal.generator.get_state())


@pytest.mark.parametrize("preserve_total", [False, True])
def test_mismatched_native_mass_fails_before_core_randomization(preserve_total):
    native = FullNative()
    native.root_view.buffers["masses"][:, 1] += .01
    if preserve_total:
        native.root_view.buffers["masses"][:, 4] -= .01
    cls, cfg = make_case(native=native)
    with pytest.raises(ValueError, match="mass"):
        cls(cfg)
    assert not any(event[0] == "startup_nominal" for event in cfg.events)


def test_auxiliary_properties_cannot_be_randomized_by_core_startup():
    cls, cfg = make_case(corrupt_aux=True)
    with pytest.raises(RuntimeError, match="changed auxiliary"):
        cls(cfg)


def test_sdk_roundoff_and_com_quaternion_sign_are_equivalent_and_recorded():
    cls, cfg = make_case(corrupt_aux="roundoff")
    env = cls(cfg)
    comparison = env.full_vehicle_receipt["auxiliary_property_comparison"]
    assert comparison["inertia_kg_m2"]["matched"]
    assert comparison["inertia_kg_m2"]["max_absolute_difference"] > 0
    assert comparison["com_pose_local_xyzw"]["max_absolute_difference"] == 2
    assert comparison["com_pose_local_xyzw"]["quaternion_sign_equivalent_max_absolute_difference"] == 0


def test_all_nominal_configuration_never_rewrites_native_mechanical_properties():
    @dataclass(frozen=True)
    class NominalConfig:
        nominal_fraction: float = 1.0
        base_mass_scale: tuple = (.9, 1.3)

    cls, cfg = make_case(scale=NominalConfig())
    before = {name: value.clone() for name, value in cfg.native.root_view.buffers.items()}
    env = cls(cfg)
    assert not any(call[0] in ("masses", "coms", "inertias") for call in cfg.native.public_calls)
    assert not cfg.native.root_view.writes
    for name, values in before.items():
        assert torch.equal(values, cfg.native.root_view.buffers[name])
    receipt = env.full_vehicle_receipt
    assert receipt["write_skipped_all_nominal"] is True
    assert receipt["requested_startup_configuration"]["base_mass_scale"] == (.9, 1.3)
    assert receipt["randomized_auxiliary_property_writes_validated"] is False
    assert env._startup_dynamics.receipt["write_skipped_all_nominal"] is True
    assert torch.allclose(env.reward_mass, torch.full((3,), 21.44174), atol=1e-5, rtol=0)


def test_unexpected_native_joint_is_rejected_before_startup():
    native = FullNative()
    native._joint_names = ("motor2", "auxYaw", "motor1", "unexpected", "passive")
    cls, cfg = make_case(native=native)
    with pytest.raises(ValueError, match="axes differ"):
        cls(cfg)
    assert not any(event[0] == "startup_nominal" for event in cfg.events)


def test_source_mass_and_auxiliary_identity_are_checked_before_constructing_native_env():
    asset = BoundAsset()
    asset.manifest["total_mass_kg"] += .1
    with pytest.raises(ValueError, match="sum"):
        make_case(asset=asset)
    asset = BoundAsset()
    asset.model["joint_names"].reverse()
    with pytest.raises(ValueError, match="named auxiliary"):
        make_case(asset=asset)
    with pytest.raises(TypeError, match="hook"):
        make_full_vehicle_env(object, BoundAsset(), {})


@pytest.mark.parametrize("nominal_fraction", [1.0, 0.5])
def test_newton_full_vehicle_reads_public_mass_and_preserves_aux_control(nominal_fraction):
    @dataclass(frozen=True)
    class NominalConfig:
        nominal_fraction: float

    class NewtonBase(BaseEnv):
        def _build_startup_dynamics(self, robot, binding, count, nominal, seed, config, **kwargs):
            assert config is None
            mass = robot.data.body_mass
            assert torch.allclose(mass.sum(-1), torch.full((count,), nominal), atol=1e-5, rtol=0)
            return StartupDynamics(mass.sum(-1), nominal)

    native = NewtonNativeArticulation()
    masses = {"base": 17.16794, "wheel": 1., "leg": 1., "auxYaw": 1.0263, "auxPitch": 1.2475}
    native.data.body_mass[:] = torch.tensor([masses[name] for name in native.body_names])
    asset = BoundAsset()
    cfg = SimpleNamespace(native=native, asset=asset, seed=27,
                          scale=NominalConfig(nominal_fraction), corrupt_aux=False, events=[])
    cls = make_full_vehicle_env(
        NewtonBase, asset, {"yaw_velocity_rad_s": (.8, .8), "pitch_position_rad": (.1, .1)},
        mechanics_backend="newton")
    if nominal_fraction != 1:
        with pytest.raises(ValueError, match="nominal startup only"):
            cls(cfg)
        return
    env = cls(cfg)
    assert env.mechanics_backend == env.robot.mechanics_backend == "newton"
    assert torch.allclose(env.reward_mass, torch.full((3,), 21.44174), atol=1e-5, rtol=0)
    receipt = env.full_vehicle_receipt["after_core_startup"]
    assert receipt["body_order"] == list(native.body_names)
    assert "Newton link-aligned COM axes" in receipt["com_frame"]
    assert "expressed in link axes" in receipt["inertia_frame"]
    assert env._gimbal_backend_body_indices.tolist() == [3, 0]
    env._reset_idx(torch.tensor([2]))
    assert torch.equal(native.data.joint_pos[2, [1, 3]], torch.tensor([.12, -.34]))
    native.data.joint_pos[:, [1, 3]] = torch.tensor([.2, -.1])
    native.data.joint_vel[:, [1, 3]] = torch.tensor([.3, -.2])
    env._apply_action()
    assert torch.equal(native.effort[:, [2, 4, 0]], torch.full((3, 3), 2.))
    assert torch.equal(native.effort[:, [1, 3]], env._gimbal.last_effort)
