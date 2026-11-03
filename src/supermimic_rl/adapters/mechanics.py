"""Explicit mechanical-property readback for PhysX and nominal Newton bodies.

No simulator is imported or inferred from available attributes. Newton expresses
its inertia tensor in link-aligned COM axes; the identity COM quaternion below
declares that representation, rather than pretending to read a principal frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch


def validate_mechanics_backend(backend: str) -> str:
    if backend not in ("physx", "newton"):
        raise ValueError("mechanics backend must be explicitly physx or newton")
    return backend


@dataclass(frozen=True)
class MechanicalParameters:
    body_names: tuple[str, ...]
    mass_kg: torch.Tensor
    com_pose_local_xyzw: torch.Tensor
    inertia_kg_m2: torch.Tensor
    source: str
    com_frame: str
    inertia_frame: str


def read_articulation_mechanics(
    robot: Any,
    *,
    backend: str = "physx",
    body_names: Sequence[str] | None = None,
    tensor_converter: Callable[[Any], torch.Tensor] | None = None,
) -> MechanicalParameters:
    """Read owned float32 tensors, optionally gathering a requested name order.

    PhysX defaults to its existing backend property order. Newton uses public
    articulation body order. This function never writes or randomizes mechanics.
    """
    backend = validate_mechanics_backend(backend)

    def tensor(value):
        if isinstance(value, torch.Tensor):
            return value
        proxy = getattr(value, "torch", None)
        if isinstance(proxy, torch.Tensor):
            return proxy
        if tensor_converter is not None:
            converted = tensor_converter(value)
        elif type(value).__module__.startswith("warp"):
            import warp

            converted = warp.to_torch(value)
        else:
            raise TypeError("mechanical properties require Torch, ProxyArray or Warp tensors")
        if not isinstance(converted, torch.Tensor):
            raise TypeError("mechanical tensor conversion must return a Torch tensor")
        return converted

    if backend == "physx":
        view = robot.root_view
        names = tuple(getattr(robot, "backend_body_names", robot.body_names))
        paths = getattr(view, "link_paths", None)
        if paths is not None:
            if not paths:
                raise ValueError("PhysX link paths must contain at least one environment")
            path_names = tuple(path.rsplit("/", 1)[-1] for path in paths[0])
            if (hasattr(robot, "backend_body_names") and names != path_names
                    or any(tuple(path.rsplit("/", 1)[-1] for path in row) != path_names
                           for row in paths)):
                raise ValueError("PhysX mechanical property body order differs from link paths")
            names = path_names
        mass = tensor(view.get_masses())
        com = tensor(view.get_coms())
        inertia = tensor(view.get_inertias())
        source = "PhysX root_view.get_masses/get_coms/get_inertias"
        com_frame = "SDK COM frame pose relative to link frame; xyz m + xyzw quaternion"
        inertia_frame = "native PhysX SDK declared; source link-frame components not assumed equivalent"
    else:
        names = tuple(robot.body_names)
        mass = tensor(robot.data.body_mass)
        position = tensor(robot.data.body_com_pos_b)
        inertia = tensor(robot.data.body_inertia)
        if position.ndim != 3 or position.shape[-1] != 3:
            raise ValueError("Newton body_com_pos_b must have shape [N, B, 3]")
        com = torch.zeros((*position.shape[:-1], 7), device=position.device, dtype=position.dtype)
        com[..., :3] = position
        com[..., 6] = 1
        if inertia.ndim == 4 and inertia.shape[-2:] == (3, 3):
            inertia = inertia.flatten(-2)
        source = "Newton Articulation.data.body_mass/body_com_pos_b/body_inertia"
        com_frame = "COM translation in link axes; identity xyzw denotes Newton link-aligned COM axes"
        inertia_frame = "Newton COM-centered inertia tensor expressed in link axes"
    if (not names or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names) or set(names) != set(robot.body_names)):
        raise ValueError("mechanical property body names must match the public articulation")
    if mass.ndim != 2 or mass.shape[0] < 1 or mass.shape[1] != len(names):
        raise ValueError("mechanical mass must have shape [N, B] in the declared body order")
    count = mass.shape[0]
    declared_count = getattr(robot, "num_instances", count)
    if declared_count != count:
        raise ValueError("mechanical properties differ from the articulation environment count")
    for name, value, tail in (("mass", mass, ()), ("COM pose", com, (7,)),
                              ("inertia", inertia, (9,))):
        if (tuple(value.shape) != (count, len(names)) + tail or value.dtype != torch.float32
                or value.device != mass.device or not torch.isfinite(value).all()):
            raise ValueError(f"mechanical {name} requires finite float32 tensors with consistent axes")
    if not (mass > 0).all():
        raise ValueError("mechanical body masses must be positive")
    if body_names is not None:
        selected = tuple(body_names)
        if (not selected or len(set(selected)) != len(selected)
                or not set(selected).issubset(names)):
            raise ValueError("selected mechanical body names must be nonempty, unique and present")
        indices = torch.tensor([names.index(name) for name in selected], device=mass.device)
        mass, com, inertia = (value.index_select(1, indices) for value in (mass, com, inertia))
        names = selected
    return MechanicalParameters(names, mass.detach().clone(), com.detach().clone(),
                                inertia.detach().clone(), source, com_frame, inertia_frame)
