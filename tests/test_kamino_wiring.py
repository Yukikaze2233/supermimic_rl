"""Check backend composition and the pinned encoder contract without Isaac."""

from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from supermimic_rl.adapters.core_projection import CoreArticulationProjection
from supermimic_rl.adapters.full_vehicle import make_full_vehicle_env
from supermimic_rl.adapters.kamino import KaminoBackend, NewtonFlatContactLoads
from support.core_projection import NewtonNativeArticulation
from support.full_vehicle import BaseEnv, BoundAsset, StartupDynamics


def coordinate_reader_type():
    path = (Path(__file__).resolve().parents[1] / ".sources/transformer_rl-full-vehicle"
            / "src/transformer_rl/formal/lab_env/joint_coordinates.py")
    if not path.is_file():
        pytest.skip("Pinned upstream checkout is required for encoder integration")
    spec = importlib.util.spec_from_file_location("pinned_joint_coordinates_for_kamino", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ContinuousJointCoordinates


@pytest.mark.parametrize("turns, sdk_continuous_output", [
    (1, 6.293185234069824), (-1, -6.293185234069824),
    (10, 62.841854095458984), (-10, -62.841854095458984),
])
def test_official_continuous_sdk_outputs_survive_pinned_reader_and_row_reset(
    turns, sdk_continuous_output,
):
    # These values were evaluated by the installed official Warp function on CPU.
    # Its version, source hashes and complete results are retained in the audit.
    sign = 1 if turns > 0 else -1
    before = turns * math.tau - sign * .01
    initial = torch.tensor([[before, .4], [before, -.4]], dtype=torch.float32)
    reader = coordinate_reader_type()(initial, ("wheel", "knee"), ("wheel",), .005)
    raw = initial.clone()
    raw[:, 0] = sdk_continuous_output
    velocity = torch.tensor([[sign * 4., 0.], [sign * 4., 0.]])
    assert torch.equal(reader.read(raw, velocity, 0), raw)
    assert reader._turns.count_nonzero() == 0
    reset_value = torch.tensor([[.12, -.2]])
    reader.reset_rows(torch.tensor([0]), reset_value)
    raw[0] = reset_value[0]
    raw[:, 0] += sign * .02
    assert torch.equal(reader.read(raw, velocity, 1), raw)
    assert reader._turns.count_nonzero() == 0
    assert reader._position[1, 0] == raw[1, 0]


@pytest.mark.parametrize("sign", [-1, 1])
def test_twopi_sdk_boundary_is_incompatible_with_pinned_four_pi_reader(sign):
    initial = torch.tensor([[sign * (math.tau - .01)]], dtype=torch.float32)
    reader = coordinate_reader_type()(initial, ("wheel",), ("wheel",), .005)
    # TWOPI uses signed modulo 2*pi despite the enum's [-2*pi, 2*pi] interval.
    raw = torch.tensor([[sign * .009999752044677734]])
    with pytest.raises(RuntimeError, match="position increment is ambiguous"):
        reader.read(raw, torch.tensor([[sign * 4.]]), 0)


@dataclass
class NominalConfiguration:
    nominal_fraction: float = 1.0


def test_composed_full_vehicle_kamino_mro_preserves_mass_reset_and_effort(monkeypatch):
    module = ModuleType("transformer_rl.formal.lab_env.env")
    module.LabStartupDynamics = StartupDynamics
    original = {name: object() for name in (
        "WheelMaterialReader", "ContactSensor", "author_shared_ground_groups")}
    for name, value in original.items():
        setattr(module, name, value)
    package = ModuleType("transformer_rl.formal.lab_env")
    package.env = module
    monkeypatch.setitem(sys.modules, "transformer_rl.formal.lab_env", package)
    monkeypatch.setitem(sys.modules, "transformer_rl.formal.lab_env.env", module)
    sensor_type = type("NewtonSensorForComposition", (), {})
    backend = KaminoBackend.__new__(KaminoBackend)
    backend.manager = object()
    backend._world_binding = SimpleNamespace(count=3)
    backend._contact_sensor_type = lambda: sensor_type

    class NativeBase(BaseEnv):
        def __init__(self, cfg):
            self.cfg, self.num_envs, self.device = cfg, 3, "cpu"
            self.events = cfg.events
            assert module.ContactSensor is sensor_type
            assert module.WheelMaterialReader is not original["WheelMaterialReader"]
            self._setup_scene()
            self.scene.env_prim_paths = [f"/e{i}" for i in range(3)]
            self._binding = SimpleNamespace(
                native_body_names=cfg.asset.manifest["rigid_body_names"],
                wheel_body_indices=[1, 2])
            cfg.native.ready = True
            assert isinstance(self.robot, CoreArticulationProjection)
            assert self.scene.articulations["robot"] is cfg.native
            self.contact_sensor = SimpleNamespace(
                body_names=list(cfg.native.body_names),
                data=SimpleNamespace(normal_force_matrix_w=torch.zeros(3, 5, 1, 3)))
            self.booting = True
            self._reset_idx(torch.tensor([1]))
            self.booting = False
            self._startup_dynamics = self._build_startup_dynamics(
                self.robot, self._binding, 3, cfg.asset.manifest["total_mass_kg"], 5,
                NominalConfiguration(), device="cpu")
            self.reward_mass = self._startup_dynamics.mass_kg_by_environment.clone()
            self._apply_action()

        def _build_startup_dynamics(self, *args, **kwargs):
            raise AssertionError("The PhysX startup hook must be replaced by Kamino")

    asset, native = BoundAsset(), NewtonNativeArticulation()
    asset.manifest["rigid_body_names"] = ("base_link", "wheel", "leg")
    native._body_names = tuple("base_link" if name == "base" else name
                               for name in native._body_names)
    masses = {"base_link": 17.16794, "wheel": 1., "leg": 1.,
              "auxYaw": 1.0263, "auxPitch": 1.2475}
    native.data.body_mass[:] = torch.tensor([masses[name] for name in native.body_names])
    cfg = SimpleNamespace(native=native, asset=asset, seed=27, events=[])
    kamino_type = backend.environment_type(NativeBase)
    full_type = make_full_vehicle_env(
        kamino_type, asset,
        {"yaw_velocity_rad_s": (.8, .8), "pitch_position_rad": (.1, .1)},
        mechanics_backend="newton")
    env = full_type(cfg)
    assert full_type.__mro__[1:3] == (kamino_type, NativeBase)
    assert isinstance(env._contact_loads, NewtonFlatContactLoads)
    assert torch.allclose(env.reward_mass, torch.full((3,), 21.44174), atol=1e-5, rtol=0)
    assert env.events[0] == ("bootstrap_reset", None)
    assert env.full_vehicle_receipt["before_core_startup"]["body_order"] == list(native.body_names)
    assert all(getattr(module, name) is value for name, value in original.items())
    q_before = native.data.joint_pos.clone()
    env._reset_idx(torch.tensor([2]))
    assert torch.equal(native.data.joint_pos[:2], q_before[:2])
    assert torch.allclose(native.data.joint_pos[2, [1, 3]], torch.tensor([.12, -.34]))
    assert native.data.joint_pos[2, [2, 4, 0]].count_nonzero() == 0
    assert native.data.joint_vel[2].count_nonzero() == 0
    env._apply_action()
    assert torch.equal(native.effort[:, [2, 4, 0]], torch.full((3, 3), 2.))
    assert torch.equal(native.effort[:, [1, 3]], env._gimbal.last_effort)
