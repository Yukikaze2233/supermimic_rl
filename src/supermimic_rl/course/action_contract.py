"""Runtime action bounds for the pinned native control contract.

The pinned native source checks issued policy/script actions against the legacy
(3, 3, 3, 3, 9, 9) contract in ``control.ownership`` and its output controller
clips with an attribute hard-coding the same values, while the environment
clips with ``task.action_bounds``. With +-3 legs each hip target stays within
+-43 deg of nominal, although the coaxial hip turns a full circle. The knee
keeps its mechanical stop from the asset; no target projection is added.

``install_action_bounds`` makes all three follow the runtime value. Validation
is delegated to the original function on rescaled policy/script rows, so every
other native check (finiteness, control sources, preparation-hold coordinates,
ownership) stays exactly as pinned. The legacy bound changes nothing.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import sys

import torch

LEGACY_ACTION_BOUNDS = (3., 3., 3., 3., 9., 9.)
OWNERSHIP_SOURCE_SHA256 = "6f78192db5a47e11b740fa841f8a8d86f6a90d4ca90bbcead018ac6e8d06c725"


def _bounded_validator(original, bounds, hold_source):
    legacy = torch.tensor(LEGACY_ACTION_BOUNDS)
    ratio = legacy / torch.tensor(bounds, dtype=torch.float32)

    def validate_issued_control(action, source, **kwargs):
        if not isinstance(action, torch.Tensor) or not isinstance(source, torch.Tensor):
            return original(action, source, **kwargs)
        # Holds keep their own canonical coordinate bound in the original.
        hold = (source == hold_source)[..., None]
        scaled = torch.where(hold, action, action * ratio.to(action.device, action.dtype))
        return original(scaled, source, **kwargs)

    validate_issued_control.supermimic_action_bounds = tuple(bounds)
    validate_issued_control.supermimic_original = original
    return validate_issued_control


def install_action_bounds(native, bounds) -> dict:
    """Align controller and ownership checks with the environment's runtime bound."""
    from transformer_rl.formal.control import ownership

    bounds = tuple(float(value) for value in bounds)
    issued = native._action_bounds
    value = torch.tensor(bounds, dtype=issued.dtype, device=issued.device)
    if not torch.equal(issued, value):
        raise ValueError("Native issued-action bounds differ from runtime task.action_bounds")
    controller = native._control.controller
    controller.action_bounds.copy_(value.to(controller.action_bounds.device))
    receipt = {"action_bounds": list(bounds), "ownership_validation": "pinned native"}
    if bounds == LEGACY_ACTION_BOUNDS:
        return receipt
    source = Path(inspect.getfile(ownership))
    if hashlib.sha256(source.read_bytes()).hexdigest() != OWNERSHIP_SOURCE_SHA256:
        raise ValueError("Action-bound installation requires the verified native ownership source")
    current = ownership.validate_issued_control
    installed = getattr(current, "supermimic_action_bounds", None)
    if installed is not None:
        if installed != bounds:
            raise ValueError("A different action bound is already installed in this process")
        original = current.supermimic_original
    else:
        original = current
    validator = (current if installed is not None
                 else _bounded_validator(original, bounds, int(ownership.ControlSource.PREPARATION_HOLD)))
    patched = []
    for name, module in tuple(sys.modules.items()):
        if (name.startswith("transformer_rl.")
                and getattr(module, "validate_issued_control", None) in (original, validator)):
            module.validate_issued_control = validator
            patched.append(name)
    return receipt | {"ownership_validation": "runtime bound via rescaled pinned check",
                      "ownership_source_sha256": OWNERSHIP_SOURCE_SHA256, "patched_modules": sorted(patched)}
