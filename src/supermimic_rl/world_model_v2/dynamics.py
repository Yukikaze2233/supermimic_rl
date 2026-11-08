"""One forward pass predicts four full 5 ms motion increments and contact labels.

Architecture follows supermimic_vibed (MIT): LayerNorm + SiLU MLP, zero-initialized
output, running input/output normalization updated from real data only.
All four increments are relative to the same interval start. The final tick
is the 20 ms endpoint; there is no independent endpoint head that can disagree.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import Tensor, nn

from .records import TICK_LABEL_DIM
from .physical import INCREMENT_DIM

MODEL_FIT_PRECISIONS = ('float32', 'bfloat16')


class RunningNorm(nn.Module):
    """Running mean/variance without gradients; frozen after ``max_count`` samples."""

    def __init__(self, dim: int, *, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count, self.eps = max_count, eps

    @torch.no_grad()
    def update(self, value: Tensor) -> None:
        value = value.reshape(-1, value.shape[-1]).float()
        if self.count >= self.max_count or not len(value):
            return
        n = value.shape[0]
        mean, var = value.mean(0), value.var(0, unbiased=False)
        total = self.count + n
        delta = mean - self.mean
        self.mean += delta * n / total
        self.var.copy_((self.var * self.count + var * n + delta.square() * self.count * n / total) / total)
        self.count.copy_(total)

    def std(self) -> Tensor:
        return torch.sqrt(self.var + self.eps ** 2)

    def forward(self, value: Tensor) -> Tensor:
        return (value - self.mean) / self.std()


def mlp(inputs: int, hidden: tuple[int, ...], outputs: int, *, layer_norm: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    for width in hidden:
        layers.append(nn.Linear(inputs, width))
        if layer_norm:
            layers.append(nn.LayerNorm(width))
        layers.append(nn.SiLU())
        inputs = width
    layers.append(nn.Linear(inputs, outputs))
    return nn.Sequential(*layers)


class MotionModel(nn.Module):
    TICKS = 4

    def __init__(self, input_dim: int, hidden: tuple[int, ...] = (2048, 2048, 2048)):
        super().__init__()
        self.output_dim = self.TICKS * TICK_LABEL_DIM
        self.net = mlp(input_dim, hidden, self.output_dim, layer_norm=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.input_norm = RunningNorm(input_dim)
        self.output_norm = RunningNorm(self.output_dim, eps=1e-4)
        # Runtime scope only: checkpoints retain FP32 parameters and buffers.
        self._fit_precision = 'float32'

    @contextmanager
    def fitting_precision(self, precision: str):
        """Select network arithmetic for one fit; imagination keeps its default.

        CPU uses the FP32 reference path even when a resumed configuration
        requests CUDA BF16. Nested/disposable fits restore the preceding scope,
        including when a failed fit raises before backward.
        """
        if precision not in MODEL_FIT_PRECISIONS:
            raise ValueError('Model fit precision must be float32 or bfloat16')
        previous = self._fit_precision
        self._fit_precision = precision
        try:
            yield
        finally:
            self._fit_precision = previous

    def targets(self, ticks: Tensor) -> Tensor:
        return ticks.flatten(-2)

    def split(self, output: Tensor) -> tuple[Tensor, Tensor]:
        ticks = output.unflatten(-1, (self.TICKS, TICK_LABEL_DIM))
        return ticks[..., -1, :INCREMENT_DIM], ticks

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Final increment ``[N, 26]`` and four full tick labels ``[N, 4, 56]``."""
        normalized = self.input_norm(inputs).to(self.net[0].weight.dtype)
        if self._fit_precision == 'bfloat16' and inputs.is_cuda:
            # Keep normalization, physical integration, losses and optimizer
            # state outside autocast; only the motion MLP uses Tensor Cores.
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                raw = self.net(normalized)
            raw = raw.to(inputs.dtype)
        else:
            raw = self.net(normalized).to(inputs.dtype)
        return self.split(raw * self.output_norm.std() + self.output_norm.mean)
