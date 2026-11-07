"""Functional native command trajectories between known resampling boundaries.

Clocks, trajectory parameters and stream limits are recorded from the live
native task, not inferred from goal vectors. Heading reads the same delayed
IMU ring. Continuous heading/admission derivatives remain available to BPTT;
mode changes and eligibility decisions have their usual discrete semantics.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
from torch import Tensor

from ..course.command_override import override_command_height
from ..formal_observation import _delayed
from . import so3

MOTION_TASK_DIM = 40


@dataclass(frozen=True)
class MotionTask:
    start: Tensor             # 4: raw trajectory start, SI command units.
    end: Tensor               # 4: raw trajectory end.
    current: Tensor           # 4: admitted command stream, before gimbal override.
    timing: Tensor            # 5: episode time, segment clock, duration, f0, f1.
    route: Tensor             # 5: trajectory kind, terrain kind, segment kind, base mode, managed pool.
    heading: Tensor           # 6: enabled, valid, target, offset, gain, yaw limit.
    drive: Tensor             # 3: actual wheel radius, track width, target speed limit.
    rates: Tensor             # 4: enabled, forward acceleration, yaw acceleration, height rate.
    gimbal: Tensor            # 5: special, speed, direction, chassis yaw rate, eligibility time.

    @classmethod
    def decode(cls, flat):
        return cls(*flat.split((4, 4, 4, 5, 5, 6, 3, 4, 5), -1))

    def encode(self):
        return torch.cat((self.start, self.end, self.current, self.timing, self.route,
                          self.heading, self.drive, self.rates, self.gimbal), -1)

    @classmethod
    @torch.no_grad()
    def capture(cls, native):
        """Read before env.step(), while the start snapshot still owns these fields."""
        task, gimbal = native._terrain_course.tasks, native._gimbal_course
        trajectory, stream = task.trajectory, task.stream
        like = stream.current
        constant = lambda value: like.new_full((len(like),), float(value))
        rates = (like.new_zeros(len(like), 4) if stream.rates is None else
                 like.new_tensor([1., stream.rates.forward_acceleration_m_s2,
                                  stream.rates.yaw_acceleration_rad_s2, stream.rates.height_rate_m_s])
                 .expand(len(like), -1))
        result = cls(trajectory.start, trajectory.end, stream.current,
            torch.stack((task.elapsed, task.segment_clock, trajectory.duration_s,
                         trajectory.frequency_start_hz, trajectory.frequency_end_hz), -1),
            torch.stack((trajectory.kind, task.terrain_kind, task.segment_kind, task.base_mode,
                         constant(task.command_pool is not None)), -1),
            torch.stack((task.heading_enabled, task.heading_valid, task.heading_target_rad,
                         task.heading_offset_rad, constant(task.heading_gain), constant(task.heading_yaw_limit_rad_s)), -1),
            like.new_tensor([stream.envelope.wheel_radius_m, stream.envelope.track_width_m,
                             stream.envelope.target_wheel_speed_rad_s]).expand(len(like), -1), rates,
            torch.stack((gimbal.special, gimbal.speed, gimbal.direction, gimbal.chassis_rate,
                         constant(gimbal.config.spin_after_episode_s)), -1))
        return result.encode().to(like).clone()

    def critic_features(self, state):
        # Future heading costs depend on a relative target, never absolute yaw.
        target = so3.wrap(self.heading[..., 2:3] - so3.yaw(state.rotation)[..., None])
        heading = torch.cat((self.heading[..., :2], target, self.heading[..., 3:]), -1)
        return replace(self, heading=heading).encode()

    def _trajectory(self, elapsed):
        duration, first, last = self.timing[..., 2:].unbind(-1)
        kind = self.route[..., 0].long()
        cyclic = (kind == 2) | (kind == 3) | (kind == 5)
        clock = torch.where(cyclic, elapsed.remainder(duration), elapsed).clamp_min(0).minimum(duration)
        fraction = clock / duration
        sine = (1 - (2 * math.pi * first * clock).cos()) / 2
        phase = first * clock + (last - first) * clock.square() / (2 * duration)
        chirp = (1 - (2 * math.pi * phase).cos()) / 2
        smooth = fraction.square() * (3 - 2 * fraction)
        stop = torch.where(fraction < .1, fraction / .1,
                           torch.where(fraction < .5, 1., torch.where(fraction < .6, (.6 - fraction) / .1, 0.)))
        weights = torch.stack((torch.zeros_like(fraction), fraction, sine, chirp, smooth, stop,
                               (fraction >= .5).to(fraction.dtype)), -1)
        weight = weights.gather(-1, kind[..., None]).clamp(0, 1)
        return self.start + (self.end - self.start) * weight

    def _admit(self, target, policy_dt_s):
        radius, track, speed_limit = self.drive.unbind(-1)
        wheels = torch.stack((target[..., 0] - target[..., 2] * track / 2,
                              target[..., 0] + target[..., 2] * track / 2), -1) / radius[..., None]
        scale = (speed_limit / wheels.abs().amax(-1).clamp_min(1e-20)).clamp_max(1)
        projected = torch.stack((target[..., 0] * scale, torch.zeros_like(scale),
                                 target[..., 2] * scale, target[..., 3]), -1)
        delta = projected - self.current
        rates = self.rates[..., 1:] * policy_dt_s / delta[..., [0, 2, 3]].abs().clamp_min(1e-20)
        fraction = torch.where(self.rates[..., 0] > .5, rates.amin(-1).clamp_max(1), 1.)
        limited = self.current + delta * fraction[..., None]
        return torch.where((self.rates[..., 0] > .5)[..., None], limited, projected)

    def advance_admission(self, queues, readout, policy_dt_s, *,
                          height_override=None, height_override_mask=None):
        """Admit the ordinary/caller target before requests latch its height.

        Caller height replaces only the requested target; the existing coupled
        wheel projection and slew limiter still determine the admitted command.
        """
        elapsed = self.timing[..., 0] + policy_dt_s
        segment_clock = self.timing[..., 1] + policy_dt_s
        target = self._trajectory(segment_clock)
        phase = _delayed(queues.heading, readout.ages[..., 3:4], queues.count, (0, 0))
        heading_yaw = torch.atan2(phase[..., 1], phase[..., 0])
        enabled, valid = self.heading[..., 0] > .5, self.heading[..., 1] > .5
        terrain = self.route[..., 1]
        approach = (terrain == 3) | (terrain == 4) | (terrain == 5) | (terrain == 7)
        initialize = (enabled | (approach & (self.route[..., 4] > .5))) & ~valid
        wrap = lambda value: torch.remainder(value + math.pi, 2 * math.pi) - math.pi
        heading_target = torch.where(initialize, wrap(heading_yaw + self.heading[..., 3]), self.heading[..., 2])
        yaw = (self.heading[..., 4] * wrap(heading_target - heading_yaw)).clamp(
            -self.heading[..., 5], self.heading[..., 5])
        target = torch.cat((target[..., :2], torch.where(enabled, yaw, target[..., 2])[..., None], target[..., 3:]), -1)
        target = override_command_height(target, height_override, height_override_mask)
        admitted = self._admit(target, policy_dt_s)
        heading = torch.cat((self.heading[..., :1], (valid | initialize).to(admitted.dtype)[..., None],
                              heading_target[..., None], self.heading[..., 3:]), -1)
        following = replace(self, current=admitted,
            timing=torch.cat((elapsed[..., None], segment_clock[..., None], self.timing[..., 2:]), -1),
            heading=heading)
        return following, admitted

    def finish_command(self, admitted, goal, *, mode=None, request_owned=None):
        """Apply gimbal eligibility after the request-conditioned command/mode.

        The controller stream retains the unassisted command in ``current``;
        a stair height hint must not feed back into that stream's slew limiter.
        """
        selected_mode = self.route[..., 3] if mode is None else mode
        eligible = ((self.route[..., 1] == 0) & (self.route[..., 2] == 1) & (selected_mode == 0)
                    & (self.timing[..., 0] >= self.gimbal[..., 4]) & (admitted[..., :3] != 0).any(-1))
        if request_owned is not None:
            if request_owned.shape != eligible.shape or request_owned.dtype != torch.bool:
                raise ValueError('Chassis request ownership requires one bool per command row')
            eligible = eligible & ~request_owned
        special = (self.gimbal[..., 0] > .5) & eligible
        command = torch.cat((torch.where(special[..., None], 0., admitted[..., :2]),
                              torch.where(special, self.gimbal[..., 3], admitted[..., 2])[..., None],
                              admitted[..., 3:4] * 5.), -1)
        context = torch.stack((~special, special, torch.where(special, self.gimbal[..., 1], 0.),
                               torch.where(special, self.gimbal[..., 2].sin(), 0.),
                               torch.where(special, self.gimbal[..., 2].cos(), 0.)), -1)
        modes = (goal[..., 4:10] if mode is None else
                 torch.nn.functional.one_hot(mode.long(), 6).to(goal.dtype))
        next_goal = torch.cat((command, modes, torch.zeros_like(goal[..., 10:11]), context), -1)
        return replace(self, gimbal=torch.cat((special.to(goal.dtype)[..., None], self.gimbal[..., 1:]), -1)), next_goal

    def advance(self, queues, readout, goal, policy_dt_s):
        """Advance only after this interval's physical rewards have been integrated."""
        following, admitted = self.advance_admission(queues, readout, policy_dt_s)
        return following.finish_command(admitted, goal)
