"""Opt-in wall-clock breakdown of named methods, synchronized with the device.

Wrapping is instance-level and reversible; behavior is unchanged except for a
device synchronization around each timed call, so profiled runs measure where
time goes but run slightly slower than unprofiled ones.
"""
from __future__ import annotations

from collections import defaultdict
import time
from types import MethodType

import torch


class StepProfiler:
    def __init__(self, device):
        self.device = torch.device(device)
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)
        self._restore = []
        self.skipped = []

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def wrap(self, owner, method_name: str, label: str | None = None):
        """Time ``owner.method_name``; nested timed calls are counted in both labels."""
        label = label or f"{type(owner).__name__}.{method_name}"
        original = getattr(owner, method_name)
        had_instance = method_name in vars(owner)
        profiler = self

        def timed(_, *args, **kwargs):
            profiler._synchronize()
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                profiler._synchronize()
                profiler.seconds[label] += time.perf_counter() - started
                profiler.calls[label] += 1

        try:
            setattr(owner, method_name, MethodType(timed, owner))
        except (AttributeError, TypeError):
            # Frozen dataclasses and slotted objects cannot take instance methods.
            self.skipped.append(label)
            return self
        self._restore.append((owner, method_name, original, had_instance))
        return self

    def summary(self) -> dict:
        result = {label: {"seconds": self.seconds[label], "calls": self.calls[label],
                          "mean_ms": 1e3 * self.seconds[label] / max(self.calls[label], 1)}
                  for label in sorted(self.seconds)}
        if self.skipped:
            result["_skipped"] = list(self.skipped)
        return result

    def close(self):
        for owner, name, original, had_instance in reversed(self._restore):
            if had_instance:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        self._restore.clear()
