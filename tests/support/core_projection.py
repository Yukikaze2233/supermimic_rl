"""Synthetic native articulation buffers in reordered backend layouts (no Kit)."""

from types import SimpleNamespace
import torch


CORE_BODIES = ("base", "wheel", "leg")
CORE_JOINTS = ("motor1", "passive", "motor2")


def to_tensor(value):
    return value.tensor


def to_native(value):
    assert value.is_contiguous()
    return NativeArray(value)


class NativeArray:
    """Stand in for a Warp array at the raw tensor-view boundary."""

    def __init__(self, tensor):
        self.tensor = tensor
        self.shape = tuple(tensor.shape)


class NativeView:
    def __init__(self, owner):
        self.owner = owner
        self.count = 3
        self.link_paths = [[f"/env_{row}/{name}" for name in owner.backend_body_names]
                           for row in range(self.count)]
        self.max_shapes = sum(owner.shape_counts.values())
        self.buffers = {}
        for name, width, tail in (
            ("masses", 5, ()), ("coms", 5, (7,)), ("inertias", 5, (9,)),
            ("material_properties", self.max_shapes, (3,)),
        ):
            shape = (self.count, width) + tail
            self.buffers[name] = torch.arange(torch.tensor(shape).prod()).float().reshape(shape)
        self.writes = []

    def _get(self, name):
        return NativeArray(self.buffers[name])

    def _set(self, name, data, indices):
        assert isinstance(data, NativeArray)
        assert isinstance(indices, NativeArray)
        assert indices.tensor.dtype == torch.int32
        assert data.shape == tuple(self.buffers[name].shape)
        rows = indices.tensor.long()
        self.buffers[name][rows] = data.tensor[rows]
        self.writes.append((name, rows.clone()))

    def get_masses(self):
        return self._get("masses")

    def get_coms(self):
        return self._get("coms")

    def get_inertias(self):
        return self._get("inertias")

    def get_material_properties(self):
        return self._get("material_properties")

    def set_masses(self, data, indices):
        self._set("masses", data, indices)

    def set_coms(self, data, indices):
        self._set("coms", data, indices)

    def set_inertias(self, data, indices):
        self._set("inertias", data, indices)

    def set_material_properties(self, data, indices):
        self._set("material_properties", data, indices)


class NativeData:
    def __init__(self, owner):
        self.owner = owner
        self.body_link_pose_w = torch.arange(3 * 5 * 7).float().reshape(3, 5, 7)
        self.body_com_pose_w = self.body_link_pose_w + 100
        self.body_com_vel_w = torch.arange(3 * 5 * 6).float().reshape(3, 5, 6)
        self.body_lin_vel_w = self.body_com_vel_w[..., :3]
        self.joint_pos = torch.arange(15).float().reshape(3, 5)
        self.joint_vel = self.joint_pos + 30
        self.default_joint_pos = self.joint_pos + 60
        self.root_link_pos_w = torch.arange(9).float().reshape(3, 3)
        self.root_link_quat_w = torch.zeros(3, 4)
        self.root_com_vel_w = torch.arange(18).float().reshape(3, 6)

    @property
    def body_com_pose_b(self):
        indices = [self.owner.backend_body_names.index(name) for name in self.owner.body_names]
        return SimpleNamespace(torch=self.owner.root_view.buffers["coms"][:, indices])


class NativeWrench:
    def __init__(self):
        self.forces = torch.ones(3, 5, 3)
        self.torques = torch.ones(3, 5, 3) * 2
        self.calls = []

    def add_forces_and_torques_index(
        self, forces=None, torques=None, positions=None, body_ids=None,
        env_ids=None, is_global=False,
    ):
        if forces is not None:
            self.forces[env_ids[:, None], body_ids[None, :]] += forces
        if torques is not None:
            self.torques[env_ids[:, None], body_ids[None, :]] += torques
        self.calls.append(dict(forces=forces, torques=torques, positions=positions,
                               body_ids=body_ids, env_ids=env_ids, is_global=is_global))


class NativeArticulation:
    def __init__(self, *, ready=True):
        self.ready = ready
        self.device = "cpu"
        self._body_names = ("auxPitch", "leg", "base", "auxYaw", "wheel")
        self._joint_names = ("motor2", "auxYaw", "motor1", "auxPitch", "passive")
        self.backend_body_names = ("wheel", "auxYaw", "base", "leg", "auxPitch")
        self.shape_counts = {"wheel": 2, "auxYaw": 1, "base": 0, "leg": 3, "auxPitch": 2}
        self.shape_requests = []
        self._physics_sim_view = SimpleNamespace(create_rigid_body_view=self._shape_view)
        self._root_view = NativeView(self)
        self.data = NativeData(self)
        self.effort = torch.arange(15).float().reshape(3, 5) + 90
        self.instantaneous_wrench_composer = NativeWrench()
        self.public_calls = []

    @property
    def body_names(self):
        assert self.ready, "native not initialized"
        return self._body_names

    @property
    def joint_names(self):
        assert self.ready, "native not initialized"
        return self._joint_names

    @property
    def root_view(self):
        assert self.ready, "native not initialized"
        return self._root_view

    def _shape_view(self, path):
        self.shape_requests.append(path)
        return SimpleNamespace(max_shapes=self.shape_counts[path.rsplit("/", 1)[-1]])

    def _set_property(self, name, value, body_ids, env_ids, full_data):
        assert full_data is False
        backend_ids = torch.tensor([self.backend_body_names.index(self.body_names[index])
                                    for index in body_ids.tolist()])
        self.root_view.buffers[name][env_ids[:, None], backend_ids[None, :]] = value
        self.public_calls.append((name, body_ids.clone(), env_ids.clone(), full_data))

    def set_masses_index(self, *, masses, body_ids, env_ids, full_data):
        self._set_property("masses", masses, body_ids, env_ids, full_data)

    def set_coms_index(self, *, coms, body_ids, env_ids, full_data):
        self._set_property("coms", coms, body_ids, env_ids, full_data)

    def set_inertias_index(self, *, inertias, body_ids, env_ids, full_data):
        self._set_property("inertias", inertias, body_ids, env_ids, full_data)

    def write_joint_position_to_sim_index(
        self, *, position, joint_ids, env_ids, full_data, skip_forward,
    ):
        assert full_data is False
        self.data.joint_pos[env_ids[:, None], joint_ids[None, :]] = position
        self.public_calls.append(("position", joint_ids, env_ids, skip_forward))

    def write_joint_velocity_to_sim_index(
        self, *, velocity, joint_ids, env_ids, full_data, skip_forward,
    ):
        assert full_data is False
        self.data.joint_vel[env_ids[:, None], joint_ids[None, :]] = velocity
        self.public_calls.append(("velocity", joint_ids, env_ids, skip_forward))

    def set_joint_effort_target(self, target, joint_ids, env_ids):
        self.effort[env_ids[:, None], joint_ids[None, :]] = target
        self.public_calls.append(("effort", joint_ids, env_ids))

    def write_root_link_pose_to_sim_index(self, **kwargs):
        self.public_calls.append(("root_pose", kwargs))

    def write_root_com_velocity_to_sim_index(self, **kwargs):
        self.public_calls.append(("root_velocity", kwargs))


class NewtonNativeArticulation(NativeArticulation):
    """Public Newton axes with no usable PhysX tensor view."""

    def __init__(self):
        super().__init__()
        self.num_instances = 3
        self.data = SimpleNamespace(**vars(self.data))
        self.data.body_mass = torch.arange(15).float().reshape(3, 5) + 1
        self.data.body_com_pos_b = torch.arange(45).float().reshape(3, 5, 3) / 100
        self.data.body_inertia = torch.eye(3).flatten().expand(3, 5, 9).clone()
        com = torch.zeros(3, 5, 7)
        com[..., :3] = self.data.body_com_pos_b
        com[..., 6] = 1
        self.data.body_com_pose_b = SimpleNamespace(torch=com)

    @property
    def root_view(self):
        raise AssertionError("Newton must not use a PhysX root_view")

    def set_joint_effort_target(self, target, joint_ids=None, env_ids=None):
        super().set_joint_effort_target(
            target, torch.arange(5) if joint_ids is None else joint_ids,
            torch.arange(3) if env_ids is None else env_ids)

    def write_joint_position_to_sim_index(
        self, *, position, joint_ids=None, env_ids=None, skip_forward=False,
    ):
        super().write_joint_position_to_sim_index(
            position=position, joint_ids=joint_ids, env_ids=env_ids,
            full_data=False, skip_forward=skip_forward)

    def write_joint_velocity_to_sim_index(
        self, *, velocity, joint_ids=None, env_ids=None, skip_forward=False,
    ):
        super().write_joint_velocity_to_sim_index(
            velocity=velocity, joint_ids=joint_ids, env_ids=env_ids,
            full_data=False, skip_forward=skip_forward)

    def write_root_link_pose_to_sim_index(self, *, root_pose, env_ids=None, skip_forward=False):
        super().write_root_link_pose_to_sim_index(
            root_pose=root_pose, env_ids=env_ids, skip_forward=skip_forward)

    def write_root_com_velocity_to_sim_index(self, *, root_velocity, env_ids=None, skip_forward=False):
        super().write_root_com_velocity_to_sim_index(
            root_velocity=root_velocity, env_ids=env_ids, skip_forward=skip_forward)
