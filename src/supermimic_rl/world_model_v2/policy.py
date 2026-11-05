"""Deployable history MLP with a supervised velocity/contact estimator.

The legacy actor consumes formal45; the chassis actor consumes 35 channels with
versioned command preprocessing. Replay retains all recorded gimbal channels.
During BPTT estimator weights are constants, while the derivative with respect
to imagined chassis observations stays in the graph.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from torch import Tensor, nn

from ..contracts import ACTION_BOUNDS
from ..chassis_observation import (
    COMMAND_CONTRACTS, LEGACY_COMMANDS, annotate_onnx_commands, chassis_frame,
)

FRAME_DIM = 45
CHASSIS_FRAME_DIM = 35
ESTIMATE_DIM = 5   # body velocity xyz, left/right wheel contact logits


def elu_mlp(inputs: int, hidden: tuple[int, ...], outputs: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for width in hidden:
        layers += [nn.Linear(inputs, width), nn.ELU()]
        inputs = width
    layers.append(nn.Linear(inputs, outputs))
    return nn.Sequential(*layers)


class EstimatorActor(nn.Module):
    def __init__(self, history_length: int = 5, estimator_hidden: tuple[int, ...] = (128, 64),
                 actor_hidden: tuple[int, ...] = (512, 256, 128), *, frame_dim: int = FRAME_DIM,
                 action_bounds: tuple[float, ...] = ACTION_BOUNDS,
                 chassis_command_contract: str = LEGACY_COMMANDS):
        super().__init__()
        if type(frame_dim) is not int or frame_dim not in (CHASSIS_FRAME_DIM, FRAME_DIM):
            raise ValueError('Actor frame dimension must be 35 (chassis) or 45 (legacy gimbal)')
        self.history_length, self.frame_dim = history_length, frame_dim
        if (chassis_command_contract not in COMMAND_CONTRACTS
                or frame_dim != CHASSIS_FRAME_DIM and chassis_command_contract != LEGACY_COMMANDS):
            raise ValueError("Projected chassis commands require the 35D actor")
        self.chassis_command_contract = chassis_command_contract
        flat = history_length * frame_dim
        self.estimator = elu_mlp(flat, estimator_hidden, ESTIMATE_DIM)
        self.decoder = elu_mlp(flat + ESTIMATE_DIM, actor_hidden, 6)
        nn.init.uniform_(self.decoder[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.decoder[-1].bias)
        self.register_buffer("action_bounds", torch.tensor(action_bounds, dtype=torch.float32))

    def _flat(self, history: Tensor) -> Tensor:
        if (history.ndim < 2 or history.shape[-2] != self.history_length
                or history.shape[-1] not in (self.frame_dim, FRAME_DIM)):
            raise ValueError(f'Actor requires chronological [batch, {self.history_length}, '
                             f'{self.frame_dim}] frames, or recorded formal45')
        frames = (chassis_frame(history, self.chassis_command_contract)
                  if self.frame_dim == CHASSIS_FRAME_DIM else history)
        return frames.flatten(-2)

    def estimate(self, history: Tensor) -> Tensor:
        return self.estimator(self._flat(history))

    def forward(self, history: Tensor) -> Tensor:
        flat = self._flat(history)
        # Detaching the output would also cut the physical-state gradient on
        # subsequent imagined steps. Only supervised training changes weights.
        if torch.jit.is_tracing() or not torch.is_grad_enabled():
            # Export/inference has no estimator optimization. TorchScript's
            # tracer cannot enter functional_call; both paths have equal values.
            estimate = self.estimator(flat)
        else:
            frozen = {name: parameter.detach() for name, parameter in self.estimator.named_parameters()}
            estimate = torch.func.functional_call(self.estimator, frozen, (flat,))
        return torch.tanh(self.decoder(torch.cat((flat, estimate), -1))) * self.action_bounds

    def estimator_loss(self, history: Tensor, velocity_body: Tensor, wheel_contact: Tensor) -> Tensor:
        estimate = self.estimate(history)
        return (nn.functional.mse_loss(estimate[..., :3], velocity_body)
                + nn.functional.binary_cross_entropy_with_logits(estimate[..., 3:], wheel_contact))


def export_onnx(actor: EstimatorActor, path: Path) -> Path:
    """Export only the actor's selected channels, keeping the live learner on its device."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    deployment = deepcopy(actor).cpu().eval()
    example = torch.zeros(1, actor.history_length, actor.frame_dim)
    torch.onnx.export(deployment, example, str(path), input_names=["history"], output_names=["action"],
                      opset_version=17, dynamo=False)
    if actor.frame_dim == CHASSIS_FRAME_DIM:
        annotate_onnx_commands(path, actor.chassis_command_contract)
    return path
