"""Observe the pinned course's velocity pushes and held base-frame wrenches.

This recorder draws no randomness and never changes native scheduling. Pushes
are issued before the first physics tick; wrench values belong to individual
ticks, because a redraw may occur partway through a policy interval.
"""
from __future__ import annotations

from types import MethodType

import torch

from ..contracts import Field


DISTURBANCE_FIELDS = (
    Field("velocity_push_world", (3,), "m/s,m/s,rad/s",
          "issued root COM world vx,vy,wz step before this policy interval; repeated on its ticks"),
    Field("external_wrench_body", (6,), "N,N*m",
          "held base-frame force xyz and torque xyz used for this physics tick"),
)


class DisturbanceRecorder:
    """Owned by the substep recorder; disabled recordings keep the old schema."""

    def __init__(self, env):
        self.owner = getattr(env, "_requested_randomization", None)
        if self.owner is None or self.owner is not getattr(env, "_episode_randomization", None):
            raise ValueError("Disturbance truth requires the native course randomization owner")
        self.push_delta = torch.zeros(env.num_envs, 3, device=env.device)
        self._original = None
        self._had_instance_method = False

    def install(self):
        owner = self.owner
        if self._original is not None or getattr(owner, "_disturbance_recorder", None) is not None:
            raise RuntimeError("The randomization owner already has a disturbance recorder")
        original = owner.push
        recorder = self

        def push_and_record(instance, *args, **kwargs):
            rows, delta = original(*args, **kwargs)
            recorder.push_delta.zero_()
            recorder.push_delta.index_copy_(0, rows, delta)
            return rows, delta

        self._had_instance_method = "push" in vars(owner)
        self._original = original
        owner.push = MethodType(push_and_record, owner)
        owner._disturbance_recorder = self

    def begin(self):
        # Native _apply_push returns early when disabled. Never reuse last step.
        self.push_delta.zero_()

    def tick(self):
        return {"velocity_push_world": self.push_delta,
                "external_wrench_body": self.owner.base_wrench}

    def snapshot(self):
        owner, config = self.owner, self.owner.config
        zero = torch.zeros_like(owner.disturbance_scale)
        # Disabled clocks are +inf natively, but the packed state is finite.
        # Configuration in the pinned source distinguishes disabled from due.
        return {
            "disturbance.push_remaining_s": owner.push_timer_s if config.pushes else zero,
            "disturbance.wrench_remaining_s": owner.wrench_timer_s if config.wrenches else zero,
            "disturbance.held_wrench_body": owner.base_wrench,
            "disturbance.scale": owner.disturbance_scale,
        }

    def close(self):
        if self._original is None:
            return
        if self._had_instance_method:
            self.owner.push = self._original
        else:
            del self.owner.__dict__["push"]
        del self.owner.__dict__["_disturbance_recorder"]
        self._original = None
