"""Opt-in ordinary MOTION trajectories through the existing native command stream.

The native pool otherwise replaces every episode trajectory with HOLD. This
owner retains the birth's waveform for subsequent ordinary motion segments;
other families keep their original sampling, request and handoff owners.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import torch

from transformer_rl.formal.tasks.command_pool import SegmentKind
from transformer_rl.formal.tasks.curriculum import TaskFamily
from transformer_rl.formal.tasks.trajectory import TrajectoryKind


class MotionTrajectoryOwner:
    """Bind one native instance; no new actor input, controller or hardware bound."""

    def __init__(self, native):
        from .jump_first import JumpFirstCommandPool

        self.tasks = native._terrain_course.tasks
        self.gimbal = native._gimbal_course
        self.training_commands_enabled = True
        if not isinstance(self.tasks.command_pool, JumpFirstCommandPool):
            raise ValueError('Motion trajectories require the installed jump-first command pool')
        if self.gimbal is None:
            raise ValueError('Motion trajectory ownership requires the independent gimbal course')
        self.birth_kind = torch.zeros_like(self.tasks.family)
        self.pending_kind = torch.zeros_like(self.tasks.family)
        self.pending_duration = torch.ones_like(self.tasks.elapsed)
        self._original_sample = self.tasks.command_pool.sample
        self._original_begin = self.tasks._begin_episode_segments
        self._original_draw = self.tasks._draw_segments
        self._original_apply = self.tasks._apply_segment
        self._original_gimbal_advance = self.gimbal.advance

    @staticmethod
    def receipt():
        return {
            'implementation': 'motion_trajectory_owner_v1',
            'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'ownership': 'MOTION births: ordinary STAND/NORMAL segments; other families unchanged',
            'waveform': 'birth TaskDraw.trajectory_kind retained across subsequent segments',
            'heading': 'MOTION uses waveform yaw; non-MOTION heading controllers unchanged',
            'yaw': 'zero for MOTION, matching native TaskFamily.MOTION birth semantics',
            'duration_s': [.5, 3.],
            'duration_source': 'native LabCourseTasks._begin_episode_trajectory: .5 + 2.5*U(0,1)',
            'duration_sampling': 'reuse normalized existing segment-duration draw, no extra RNG call',
            'start_stop': 'zero planar start, sampled signed endpoint; standing segments are HOLD zero',
            'gimbal': 'MOTION excludes special chassis spin-translation; independent servo remains active',
            'evaluation': 'explicit fixed evaluation temporarily owns commands and bypasses training sampling',
            'scope': 'command trajectories, not measured acceleration limits or capability evidence',
        }

    def install(self):
        self.tasks.command_pool.sample = self._sample
        self.tasks._begin_episode_segments = self._begin
        self.tasks._draw_segments = self._draw
        self.tasks._apply_segment = self._apply
        self.gimbal.advance = self._advance_gimbal
        return self.receipt()

    def _sample(self, family, terrain_kind, elapsed_s, approach_sign, context, generator, **kwargs):
        if not self.training_commands_enabled:
            return self._original_sample(family, terrain_kind, elapsed_s, approach_sign, context, generator, **kwargs)
        # Native sampling already moves unavailable special shares to NORMAL.
        # Use that path to draw legal signed ordinary targets for MOTION; do
        # not reinterpret a sampled SPIN's zero vx as a new motion command.
        elapsed = torch.where(family == int(TaskFamily.MOTION), 0., elapsed_s)
        draw = self._original_sample(family, terrain_kind, elapsed, approach_sign, context, generator, **kwargs)
        motion = family == int(TaskFamily.MOTION)
        command = draw.command.clone()
        # Native TaskDraw defines MOTION as signed forward motion with zero
        # yaw. The generic pool must not replace it with a turning objective.
        command[:, 2] = torch.where(motion, 0., command[:, 2])
        return replace(draw, command=command, heading_enabled=draw.heading_enabled & ~motion)

    def _begin(self, ids, task):
        self.birth_kind[ids] = task.draw.trajectory_kind
        result = self._original_begin(ids, task)
        # Native begin initializes stream.current only after applying its first
        # segment. Rebase now, so a reset cannot inherit the previous episode's
        # velocity/height as the start of its new waveform.
        self._configure(ids)
        return result

    def _draw(self, rows):
        draw = self._original_draw(rows)
        self.pending_kind[rows] = self.birth_kind[rows]
        low, high = self.tasks.command_pool.config.segment_s
        uniform = ((draw.duration_s - low) / (high - low)).clamp(0, 1)
        # This is the native legacy waveform duration, not an actuator slew
        # limit. The segment still ends after its original 5--15 s draw.
        self.pending_duration[rows] = .5 + 2.5 * uniform
        return draw

    def _apply(self, rows, command, kind, duration_s, heading_enabled, heading_offset_rad):
        result = self._original_apply(rows, command, kind, duration_s, heading_enabled, heading_offset_rad)
        self._configure(rows)
        return result

    def _configure(self, rows):
        if not self.training_commands_enabled:
            return
        tasks = self.tasks
        ids = rows[tasks.family[rows] == int(TaskFamily.MOTION)]
        if not len(ids):
            return
        standing = tasks.segment_kind[ids] == int(SegmentKind.STAND)
        kind = torch.where(standing, int(TrajectoryKind.HOLD), self.pending_kind[ids])
        target = tasks.trajectory.end[ids].clone()
        start = tasks.stream.current[ids].clone()
        hold = kind == int(TrajectoryKind.HOLD)
        start[hold] = target[hold]
        # START_STOP must actually stop, rather than returning to a previous
        # nonzero segment speed. Keep the sampled height target constant.
        stop = kind == int(TrajectoryKind.START_STOP)
        start[stop, :3] = 0.
        start[stop, 3] = target[stop, 3]
        tasks.trajectory.start[ids] = start
        tasks.trajectory.kind[ids] = kind
        # The waveform owns yaw as well as vx. A second heading controller
        # would turn a START_STOP zero command into an unintended rotation.
        tasks.heading_enabled[ids] = False
        duration = self.pending_duration[ids]
        tasks.trajectory.duration_s[ids] = duration
        tasks.trajectory.frequency_start_hz[ids] = 1 / duration
        tasks.trajectory.frequency_end_hz[ids] = 2 / duration

    def _advance_gimbal(self, dt_s, world_angles, eligible, **kwargs):
        if not self.training_commands_enabled:
            return self._original_gimbal_advance(dt_s, world_angles, eligible, **kwargs)
        return self._original_gimbal_advance(
            dt_s, world_angles, eligible & (self.tasks.family != int(TaskFamily.MOTION)), **kwargs)


def install_motion_trajectory_owner(native, *, enabled=False):
    """Default-off installation keeps existing callables and source receipts."""
    if type(enabled) is not bool:
        raise ValueError('Motion trajectory owner must be a boolean')
    if not enabled:
        return None
    if getattr(native, '_motion_trajectory_owner', None) is not None:
        raise ValueError('Motion trajectory owner is already installed')
    owner = MotionTrajectoryOwner(native)
    receipt = owner.install()
    native._motion_trajectory_owner = owner
    return receipt
