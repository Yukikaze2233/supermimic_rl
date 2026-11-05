"""Versioned command preprocessing for the unchanged 5x35 chassis interface.

Native task_heading is root_yaw + signed gimbal encoder yaw, not the tilted
camera view ray. Rotate its requested horizontal velocity into root-heading
coordinates using the delayed encoder already in each recorded frame. No
privileged pose, extra network input, actuator command or task state is used.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

LEGACY_COMMANDS = "native_first35_v1"
HEADING_COMMANDS = "gimbal_target_heading_v1"
COMMAND_CONTRACTS = (LEGACY_COMMANDS, HEADING_COMMANDS)
ONNX_COMMAND_METADATA = "supermimic.chassis_command_contract"


def chassis_frame(frame: Tensor, contract: str = LEGACY_COMMANDS) -> Tensor:
    """Native42/formal45 -> chassis35; existing chassis35 is already prepared.

    Apply to every history frame, before flattening. Channels 0:2 are SI
    horizontal velocity in the chassis heading frame; 2:35 are unchanged.
    The full raw frame remains in replay so old data needs no rewriting.
    """
    if contract not in COMMAND_CONTRACTS or frame.shape[-1] not in (35, 42, 45):
        raise ValueError("Chassis input requires a known command contract and 35/42/45 channels")
    if contract == LEGACY_COMMANDS or frame.shape[-1] == 35:
        return frame[..., :35]
    active = frame[..., 36:37] > .5
    speed = frame[..., 37:38]
    direction_sin, direction_cos = frame[..., 38:39], frame[..., 39:40]
    yaw_sin, yaw_cos = frame[..., 40:41], frame[..., 41:42]
    horizontal = speed * torch.cat((direction_cos * yaw_cos - direction_sin * yaw_sin,
                                     direction_sin * yaw_cos + direction_cos * yaw_sin), -1)
    return torch.cat((torch.where(active, horizontal, frame[..., :2]), frame[..., 2:35]), -1)


def annotate_onnx_commands(path: Path, contract: str) -> None:
    """Keep the 35D ONNX input; tell deployment which preprocessing it expects."""
    import onnx

    if contract not in COMMAND_CONTRACTS:
        raise ValueError("Unknown chassis command contract")
    model = onnx.load(str(path))
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    metadata[ONNX_COMMAND_METADATA] = contract
    onnx.helper.set_model_props(model, metadata)
    onnx.save(model, str(path))
