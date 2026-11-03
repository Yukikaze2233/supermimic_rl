"""Named core views of a full articulation, without owning its lifecycle.

The scene must retain the real articulation. Only the flat locomotion environment's
``robot`` reference is replaced by this projection. Scene reset, simulation writes,
and updates still run on the full articulation; auxiliary joint reset and control
remain the caller's responsibility. Root states describe the original root link,
not a newly computed center of mass of the selected bodies.

This implements the explicit API consumed by the upstream flat Lab environment.
It is not a general Articulation replacement. Public property setters preserve SDK
cache invalidation; raw ``root_view`` setters retain the SDK's raw-write semantics.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .mechanics import validate_mechanics_backend


@dataclass(frozen=True)
class _TensorView:
    """Expose the SDK data accessor's ``.torch`` boundary after gathering."""

    torch: torch.Tensor


class CoreArticulationProjection:
    """Project named bodies/joints while leaving every auxiliary axis untouched.

    Construction does not access the native object: Isaac initializes its views
    after ``_setup_scene``. Binding happens on first use and rejects subsequent
    root-view replacement. Core order is the supplied name order. Public data and
    setters use public indices; PhysX property getters use backend link indices.

    Optional conversion callbacks accept/return Torch tensors at the native tensor
    view boundary. They allow CPU tests without importing Warp. Production defaults
    import Warp lazily and return scalar Warp arrays with the original flat shape
    (COM poses ``[N, B, 7]``, inertias ``[N, B, 9]``).
    """

    def __init__(
        self,
        native: Any,
        core_body_names: Sequence[str],
        core_joint_names: Sequence[str],
        *,
        native_to_tensor: Callable[[Any], torch.Tensor] | None = None,
        tensor_to_native: Callable[[torch.Tensor], Any] | None = None,
        mechanics_backend: str = "physx",
    ) -> None:
        self.native = native
        self.mechanics_backend = validate_mechanics_backend(mechanics_backend)
        self._body_names = self._validate_names(core_body_names, "core body")
        self._joint_names = self._validate_names(core_joint_names, "core joint")
        if any(callback is not None and not callable(callback)
               for callback in (native_to_tensor, tensor_to_native)):
            raise TypeError("native conversion callbacks must be callable")
        self._convert_in = native_to_tensor
        self._convert_out = tensor_to_native
        self._bound_view = None
        self._shape_indices = None
        self._data = _CoreData(self)
        self._view = _CoreRootView(self)
        self._wrench = _CoreWrenchComposer(self)

    @staticmethod
    def _validate_names(names: Sequence[str], label: str) -> tuple[str, ...]:
        if isinstance(names, (str, bytes)):
            raise ValueError(f"{label} names must be a nonempty unique sequence")
        names = tuple(names)
        if (not names or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names)):
            raise ValueError(f"{label} names must be a nonempty unique sequence")
        return names

    def _bind(self) -> None:
        view = self.native.root_view if self.mechanics_backend == "physx" else self.native.data
        if self._bound_view is not None:
            if view is not self._bound_view:
                raise RuntimeError("native articulation view changed; rebuild the core projection")
            return
        public_bodies = self._validate_names(self.native.body_names, "native body")
        public_joints = self._validate_names(self.native.joint_names, "native joint")
        if self.mechanics_backend == "physx":
            paths = tuple(tuple(row) for row in view.link_paths)
            count = int(view.count)
            if count < 1 or len(paths) != count:
                raise ValueError("native link paths must contain every environment row")
            backend_bodies = self._validate_names(
                tuple(path.rsplit("/", 1)[-1] for path in paths[0]), "backend body")
            if (set(public_bodies) != set(backend_bodies)
                    or any(tuple(path.rsplit("/", 1)[-1] for path in row) != backend_bodies
                           for row in paths)):
                raise ValueError("native public bodies and backend link paths disagree")
            declared_backend = getattr(self.native, "backend_body_names", backend_bodies)
            if tuple(declared_backend) != backend_bodies:
                raise ValueError("native backend body names differ from PhysX link paths")
        else:
            count = int(self.native.num_instances)
            if count < 1:
                raise ValueError("Newton articulation must contain at least one environment")
            paths, backend_bodies = None, public_bodies
        if (not set(self._body_names).issubset(public_bodies)
                or not set(self._joint_names).issubset(public_joints)):
            raise ValueError("core names are absent from the native articulation")
        device = torch.device(self.native.device)
        self._body_indices = torch.tensor(
            [public_bodies.index(name) for name in self._body_names], device=device)
        self._joint_indices = torch.tensor(
            [public_joints.index(name) for name in self._joint_names], device=device)
        self._backend_indices = torch.tensor(
            [backend_bodies.index(name) for name in self._body_names], dtype=torch.long)
        self._paths = paths
        self._public_body_count = len(public_bodies)
        self._public_joint_count = len(public_joints)
        self._count = count
        self._device = device
        self._bound_view = view

    def _tensor(self, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        direct = getattr(value, "torch", None)
        if isinstance(direct, torch.Tensor):
            return direct
        if self._convert_in is None:
            import warp as wp

            result = wp.to_torch(value)
        else:
            result = self._convert_in(value)
        if not isinstance(result, torch.Tensor):
            raise TypeError("native conversion must produce a Torch tensor")
        return result

    def _native(self, value: torch.Tensor) -> Any:
        value = value.contiguous()
        if self._convert_out is not None:
            return self._convert_out(value)
        import warp as wp

        return wp.from_torch(value)

    def _ids(self, value: Any, size: int, device: torch.device) -> torch.Tensor:
        if value is None:
            return torch.arange(size, dtype=torch.long, device=device)
        if isinstance(value, slice):
            return torch.tensor(range(size)[value], dtype=torch.long, device=device)
        if isinstance(value, (Sequence, range)):
            value = torch.as_tensor(value) if len(value) else torch.empty(0, dtype=torch.long)
        value = self._tensor(value)
        if (value.ndim != 1 or value.dtype == torch.bool
                or value.dtype.is_floating_point or value.dtype.is_complex):
            raise ValueError("selection must contain one-dimensional integer indices")
        # Validate Python selections on CPU before transferring them. Spring
        # body lists are constant per substep and must not force a GPU sync.
        value = value.to(dtype=torch.long)
        if (bool((value < 0).any()) or bool((value >= size).any())
                or value.unique().numel() != value.numel()):
            raise ValueError("selection indices must be unique and in range")
        return value.to(device=device)

    def _selection(self, env_ids: Any, axis_ids: Any, *, joint: bool = False):
        self._bind()
        mapping = self._joint_indices if joint else self._body_indices
        rows = self._ids(env_ids, self._count, self._device)
        columns = self._ids(axis_ids, len(mapping), self._device)
        return rows, columns, mapping.index_select(0, columns)

    def _selected_values(self, value, rows, columns, axis_size, tail, full_data):
        if type(full_data) is not bool:
            raise TypeError("full_data must be a bool")
        value = self._tensor(value)
        expected = ((self._count, axis_size) if full_data
                    else (len(rows), len(columns))) + tail
        if tuple(value.shape) != expected or value.dtype != torch.float32:
            raise ValueError(f"selected native values must be float32 with shape {expected}")
        if full_data:
            value = value.index_select(0, rows.to(value.device)).index_select(
                1, columns.to(value.device))
        return value.to(self._device).contiguous()

    @property
    def body_names(self) -> tuple[str, ...]:
        self._bind()
        return self._body_names

    @property
    def joint_names(self) -> tuple[str, ...]:
        self._bind()
        return self._joint_names

    @property
    def backend_body_names(self) -> tuple[str, ...]:
        self._bind()
        return self._body_names

    @property
    def num_bodies(self) -> int:
        return len(self._body_names)

    @property
    def num_joints(self) -> int:
        return len(self._joint_names)

    @property
    def num_instances(self) -> int:
        self._bind()
        return self._count

    @property
    def device(self) -> torch.device:
        self._bind()
        return self._device

    @property
    def data(self):
        return self._data

    @property
    def root_view(self):
        return self._view

    @property
    def instantaneous_wrench_composer(self):
        return self._wrench

    @property
    def _physics_sim_view(self):
        self._bind()
        if self.mechanics_backend != "physx":
            raise NotImplementedError("Newton has no PhysX simulation view")
        return self.native._physics_sim_view

    def _set_property(self, name, value, tail, body_ids, env_ids, full_data):
        if self.mechanics_backend != "physx":
            raise NotImplementedError("Newton mechanical randomization is not supported")
        rows, columns, native_columns = self._selection(env_ids, body_ids)
        selected = self._selected_values(
            value, rows, columns, self.num_bodies, tail, full_data)
        getattr(self.native, f"set_{name}_index")(
            **{name: selected}, body_ids=native_columns, env_ids=rows, full_data=False)

    def set_masses_index(self, *, masses, body_ids=None, env_ids=None, full_data=False):
        self._set_property("masses", masses, (), body_ids, env_ids, full_data)

    def set_coms_index(self, *, coms, body_ids=None, env_ids=None, full_data=False):
        self._set_property("coms", coms, (7,), body_ids, env_ids, full_data)

    def set_inertias_index(self, *, inertias, body_ids=None, env_ids=None, full_data=False):
        self._set_property("inertias", inertias, (9,), body_ids, env_ids, full_data)

    def _write_joint(self, name, value, joint_ids, env_ids, full_data, skip_forward):
        rows, columns, native_columns = self._selection(env_ids, joint_ids, joint=True)
        selected = self._selected_values(value, rows, columns, self.num_joints, (), full_data)
        # Newton accepts only selected rows/columns and has no full_data keyword.
        # PhysX exposes both layouts, so retain its explicit partial-data flag.
        layout = {"full_data": False} if self.mechanics_backend == "physx" else {}
        getattr(self.native, f"write_joint_{name}_to_sim_index")(
            **{name: selected}, joint_ids=native_columns, env_ids=rows,
            skip_forward=skip_forward, **layout)

    def write_joint_position_to_sim_index(
        self, *, position, joint_ids=None, env_ids=None, full_data=False, skip_forward=False,
    ):
        self._write_joint("position", position, joint_ids, env_ids, full_data, skip_forward)

    def write_joint_velocity_to_sim_index(
        self, *, velocity, joint_ids=None, env_ids=None, full_data=False, skip_forward=False,
    ):
        self._write_joint("velocity", velocity, joint_ids, env_ids, full_data, skip_forward)

    def set_joint_effort_target(self, target, joint_ids=None, env_ids=None):
        rows, columns, native_columns = self._selection(env_ids, joint_ids, joint=True)
        selected = self._selected_values(target, rows, columns, self.num_joints, (), False)
        self.native.set_joint_effort_target(selected, joint_ids=native_columns, env_ids=rows)

    def write_root_link_pose_to_sim_index(self, *, root_pose, env_ids=None, **kwargs):
        self._bind()
        self.native.write_root_link_pose_to_sim_index(
            root_pose=root_pose, env_ids=env_ids, **kwargs)

    def write_root_com_velocity_to_sim_index(self, *, root_velocity, env_ids=None, **kwargs):
        self._bind()
        self.native.write_root_com_velocity_to_sim_index(
            root_velocity=root_velocity, env_ids=env_ids, **kwargs)


class _CoreData:
    _BODY_FIELDS = frozenset({
        "body_link_pose_w", "body_com_pose_w", "body_com_vel_w", "body_lin_vel_w",
        "body_com_pose_b", "body_mass", "body_com_pos_b", "body_inertia",
    })
    _JOINT_FIELDS = frozenset({"joint_pos", "joint_vel", "default_joint_pos"})
    _ROOT_FIELDS = frozenset({"root_link_pos_w", "root_link_quat_w", "root_com_vel_w"})

    def __init__(self, owner):
        self._owner = owner

    def __getattr__(self, name):
        if name not in self._BODY_FIELDS | self._JOINT_FIELDS | self._ROOT_FIELDS:
            raise AttributeError(f"core projection does not expose data.{name}")
        owner = self._owner
        owner._bind()
        value = getattr(owner.native.data, name)
        if name in self._ROOT_FIELDS:
            return value
        value = owner._tensor(value)
        joint = name in self._JOINT_FIELDS
        expected = owner._public_joint_count if joint else owner._public_body_count
        if value.ndim < 2 or tuple(value.shape[:2]) != (owner._count, expected):
            raise ValueError(f"native data.{name} has incompatible environment/body/joint axes")
        indices = owner._joint_indices if joint else owner._body_indices
        result = value.index_select(1, indices.to(value.device))
        return _TensorView(result) if name == "body_com_pose_b" else result


class _CoreRootView:
    def __init__(self, owner):
        self._owner = owner

    @property
    def count(self):
        return self._owner.num_instances

    @property
    def link_paths(self):
        owner = self._owner
        owner._bind()
        if owner.mechanics_backend != "physx":
            raise NotImplementedError("Newton public bodies do not expose PhysX link_paths")
        indices = owner._backend_indices.tolist()
        return [[row[index] for index in indices] for row in owner._paths]

    def _shapes(self):
        owner = self._owner
        owner._bind()
        if owner.mechanics_backend != "physx":
            raise NotImplementedError("Newton material access requires the Newton backend adapter")
        if owner._shape_indices is None:
            counts = [int(owner.native._physics_sim_view.create_rigid_body_view(path).max_shapes)
                      for path in owner._paths[0]]
            if any(count < 0 for count in counts) or sum(counts) != owner._bound_view.max_shapes:
                raise ValueError("native link shape counts differ from the full material table")
            offsets = [0]
            for count in counts:
                offsets.append(offsets[-1] + count)
            owner._shape_indices = torch.tensor([
                shape for index in owner._backend_indices.tolist()
                for shape in range(offsets[index], offsets[index + 1])
            ], dtype=torch.long)
        return owner._shape_indices

    @property
    def max_shapes(self):
        return len(self._shapes())

    def _read(self, name, tail, *, material=False):
        owner = self._owner
        owner._bind()
        if owner.mechanics_backend != "physx":
            raise NotImplementedError("Newton properties must be read through public Articulation.data")
        indices = self._shapes() if material else owner._backend_indices
        full = owner._tensor(getattr(owner._bound_view, f"get_{name}")())
        width = owner._bound_view.max_shapes if material else owner._public_body_count
        if tuple(full.shape) != (owner._count, width) + tail:
            raise ValueError(f"native root_view.{name} has incompatible shape")
        return full, indices.to(full.device)

    def _get(self, name, tail, *, material=False):
        value, indices = self._read(name, tail, material=material)
        return self._owner._native(value.index_select(1, indices))

    def get_masses(self):
        return self._get("masses", ())

    def get_coms(self):
        return self._get("coms", (7,))

    def get_inertias(self):
        return self._get("inertias", (9,))

    def get_material_properties(self):
        return self._get("material_properties", (3,), material=True)

    def _set(self, name, data, indices, tail, *, material=False):
        owner = self._owner
        full, columns = self._read(name, tail, material=material)
        rows = owner._ids(indices, owner._count, full.device)
        data = owner._tensor(data)
        expected = (owner._count, len(columns)) + tail
        if tuple(data.shape) != expected or data.dtype != full.dtype:
            raise ValueError(f"raw projected property write requires full core shape {expected}")
        # PhysX consumes a full table and selects rows itself. Read-modify-write
        # preserves auxiliary shapes/bodies and every unselected environment.
        merged = full.clone()
        merged[rows[:, None], columns[None, :]] = data.to(full.device).index_select(0, rows)
        getattr(owner._bound_view, f"set_{name}")(
            owner._native(merged), owner._native(rows.to(dtype=torch.int32)))

    def set_masses(self, data, indices=None):
        self._set("masses", data, indices, ())

    def set_coms(self, data, indices=None):
        self._set("coms", data, indices, (7,))

    def set_inertias(self, data, indices=None):
        self._set("inertias", data, indices, (9,))

    def set_material_properties(self, data, indices=None):
        self._set("material_properties", data, indices, (3,), material=True)


class _CoreWrenchComposer:
    def __init__(self, owner):
        self._owner = owner

    def add_forces_and_torques_index(
        self, forces=None, torques=None, positions=None, body_ids=None,
        env_ids=None, is_global=False,
    ):
        owner = self._owner
        rows, columns, native_columns = owner._selection(env_ids, body_ids)
        values = {
            key: None if value is None else owner._selected_values(
                value, rows, columns, owner.num_bodies, (3,), False)
            for key, value in (("forces", forces), ("torques", torques), ("positions", positions))
        }
        owner.native.instantaneous_wrench_composer.add_forces_and_torques_index(
            **values, body_ids=native_columns, env_ids=rows, is_global=is_global)
