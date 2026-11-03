"""Bound full-vehicle asset, startup dynamics and base environment fakes."""

from copy import deepcopy
import json
from types import SimpleNamespace
import torch
from supermimic_rl.adapters.core_projection import CoreArticulationProjection
from support.core_projection import CORE_BODIES, CORE_JOINTS, to_tensor


class BoundAsset:
    def __init__(self):
        self.manifest_sha256 = "a" * 64
        self.manifest = {
            "rigid_body_names": CORE_BODIES, "tree_joint_names": CORE_JOINTS,
            "auxiliary_body_names": ["auxYaw", "auxPitch"],
            "auxiliary_joint_names": ["auxYaw", "auxPitch"],
            "auxiliary_model": "gimbal_model.json",
            "core_mass_kg": 19.16794, "auxiliary_mass_kg": 2.2738, "total_mass_kg": 21.44174,
            "nominal_joint_pos": {"auxYaw": .12, "auxPitch": -.34},
        }
        self.model = {
            "body_names": ["auxYaw", "auxPitch"], "joint_names": ["auxYaw", "auxPitch"],
            "control_joint_signs": [1, -1],
            "joints": [{"name": name, "effort_limit_nm": 4.5} for name in ("auxYaw", "auxPitch")],
            "bodies": [{"name": "auxYaw", "mass": 1.0263}, {"name": "auxPitch", "mass": 1.2475}],
        }

    def bound_bytes(self, name, *, maximum_bytes):
        assert name == "gimbal_model.json" and maximum_bytes == 256 * 1024
        return json.dumps(self.model).encode()


class StartupDynamics:
    def __init__(self, mass, nominal):
        self.mass_kg_by_environment = mass
        self.mass_scale = (mass / nominal)[:, None]
        self.base_com_local_m = torch.tensor([1., 2., 3.])
        self._receipt = {"body_order": list(CORE_BODIES), "native_parameters_sha256": "core digest",
                         "physical_qualification": False, "mass_kg_min": float(mass.min()),
                         "mass_kg_max": float(mass.max())}

    @property
    def receipt(self):
        return deepcopy(self._receipt)


class BaseEnv:
    def __init__(self, cfg):
        self.cfg, self.num_envs, self.device = cfg, 3, "cpu"
        self.events = cfg.events
        self._setup_scene()
        assert isinstance(self.robot, CoreArticulationProjection)
        assert self.scene.articulations["robot"] is cfg.native
        cfg.native.ready = True
        self.booting = True
        self._reset_idx(torch.tensor([1]))
        self.booting = False
        self._startup_dynamics = self._build_startup_dynamics(
            self.robot, object(), 3, cfg.asset.manifest["total_mass_kg"], 5,
            cfg.scale, device="cpu", use_public_setters=True)
        # Consumers run inside the original constructor, before the independent
        # controller exists. They must already receive full mass accounting.
        self.reward_mass = self._startup_dynamics.mass_kg_by_environment.clone()
        self._apply_action()

    def _setup_scene(self):
        self.robot = self.cfg.native
        self.robot.ready = False
        self.scene = SimpleNamespace(articulations={"robot": self.robot})

    def _build_startup_dynamics(self, robot, binding, count, nominal, seed, config, **kwargs):
        self.events.append(("startup_nominal", nominal))
        core_mass = to_tensor(robot.root_view.get_masses())
        assert torch.allclose(core_mass.sum(-1), torch.full((count,), nominal), atol=1e-5, rtol=0)
        if config is not None:
            core_mass = core_mass * torch.tensor(config)[:, None]
            robot.set_masses_index(masses=core_mass, full_data=True)
        if self.cfg.corrupt_aux:
            index = self.cfg.native.backend_body_names.index("auxYaw")
            if self.cfg.corrupt_aux == "roundoff":
                self.cfg.native.root_view.buffers["inertias"][:, index, 0] += 5e-7
                self.cfg.native.root_view.buffers["coms"][:, index, 3:] *= -1
            else:
                self.cfg.native.root_view.buffers["inertias"][:, index, 0] += 1
        return StartupDynamics(core_mass.sum(-1), nominal)

    def _reset_idx(self, ids):
        if self.booting:
            self.events.append(("bootstrap_reset", self._gimbal))
            return
        ids = torch.as_tensor(ids, dtype=torch.long)
        if not len(ids):
            return
        self.events.append(("parent_reset", ids.tolist()))
        # PhysX Articulation.reset resets actuator commands and wrenches, not q/dq.
        self.cfg.native.effort[ids] = 0
        self.robot.write_joint_position_to_sim_index(position=torch.zeros(len(ids), 3), env_ids=ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros(len(ids), 3), env_ids=ids)
        self.events.append(("parent_forward", self.cfg.native.data.joint_pos.clone(),
                            self.cfg.native.data.joint_vel.clone()))

    def _apply_action(self):
        self.events.append(("core_action",))
        self.robot.set_joint_effort_target(torch.full((3, 3), 2.))
