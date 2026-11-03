"""Formal deployment observer, separate from clean physical supervision.

The native SensorModel stores encoder-biased samples before observation noise.
Histories here are newest first, on the physical clock. Their future values are
part of the learned dynamics state; instantaneous physics never bypasses them.
Noise follows an explicit context contract: legacy uniform half-widths or the
requested Gaussian standard deviations. Neither is a measured probability law.
Added gimbal pitch/rate channels use SI units.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from .compact_state import analytic_actor_observation, named_field
from .contracts import StateSchema


def _optional(tensor, schema, name, *, context=False):
    fields = schema.context_fields if context else schema.state_fields
    if any(field.name == name for field in fields):
        return named_field(tensor, schema, name, context=context)
    return None


def _previous(state, previous_action, schema):
    return (named_field(state, schema, "control._previous_action")
            if previous_action is None else previous_action)


def _check_goal(state, goal, schema):
    if (schema.observation_dim != 45 or schema.goal_dim < 16
            or goal.shape != (*state.shape[:-1], schema.goal_dim)):
        raise ValueError("Formal45 requires the base11 and five gimbal command features")


def _gimbal_clean(state, context, schema):
    phase = named_field(state, schema, "gimbal.yaw_sin_cos")
    pitch = named_field(state, schema, "gimbal.pitch_position_rad")
    velocity = named_field(state, schema, "gimbal.velocity_rad_s")
    signs = named_field(context, schema, "gimbal.control_joint_signs", context=True)
    zero = _optional(context, schema, "gimbal.joint_zero_rad", context=True)
    if zero is not None:
        sine, cosine = phase[..., :1], phase[..., 1:]
        offset = zero[..., :1]
        phase = torch.cat((sine * offset.cos() - cosine * offset.sin(),
                           cosine * offset.cos() + sine * offset.sin()), -1)
        pitch = pitch - zero[..., 1:2]
    # Native compact yaw is periodic; the controller sign acts on sine only.
    phase = torch.cat((phase[..., :1] * signs[..., :1], phase[..., 1:]), -1)
    phase = _unit_phase(phase)
    return torch.cat((phase, pitch * signs[..., 1:2], velocity * signs), -1)


def _unit_phase(phase):
    norm = phase.norm(dim=-1, keepdim=True)
    fallback = torch.cat((torch.zeros_like(norm), torch.ones_like(norm)), -1)
    return torch.where(norm > 1e-8, phase / norm.clamp_min(1e-8), fallback)


def observe_clean(state: Tensor, goal: Tensor, context: Tensor,
                  previous_action: Tensor | None, schema: StateSchema) -> Tensor:
    """Instantaneous unbiased physical labels; never the delayed Actor input."""
    _check_goal(state, goal, schema)
    base = analytic_actor_observation(state, context, goal[..., :11], previous_action, schema)
    return torch.cat((base, goal[..., 11:16], _gimbal_clean(state, context, schema)), -1).clamp(-100, 100)


def _delayed(history, ages, count, groups):
    capacity, width = history.shape[-2:]
    if len(groups) != width:
        raise ValueError("Sensor history width differs from its channel groups")
    # Native count is integral. Project its learned representation to valid
    # available history, exactly clamping fresh reset ages to its sole sample.
    maximum = count.reshape(*history.shape[:-2], 1).round().clamp(1, capacity) - 1
    ages = torch.minimum(ages.clamp_min(0), maximum)
    recent = ages.floor().long()
    older = torch.minimum(recent + 1, maximum.long())
    group = torch.tensor(groups, device=history.device)
    a = history.gather(-2, recent[..., group].unsqueeze(-2)).squeeze(-2)
    b = history.gather(-2, older[..., group].unsqueeze(-2)).squeeze(-2)
    return a + (ages - recent)[..., group] * (b - a)


def observe(state: Tensor, goal: Tensor, context: Tensor,
            previous_action: Tensor | None, schema: StateSchema, *, noise: bool = True) -> Tensor:
    """Read causal physical queues, then configured noise and actor scaling.

    ``sensor.gimbal_history`` stores controller-coordinate sin(yaw), cos(yaw),
    pitch, yaw velocity and pitch velocity. Positions share group0 and rates
    group1. Gimbal yaw is not resampled from clean native state.
    """
    _check_goal(state, goal, schema)
    return deployed_frame(
        named_field(state, schema, "sensor.values"), named_field(state, schema, "sensor.count"),
        named_field(state, schema, "sensor.gimbal_history"),
        _previous(state, previous_action, schema), goal, SensorReadout.from_context(context, schema),
        noise=noise)


class SensorReadout:
    """Per-environment read parameters: ages in physical ticks and noise settings."""

    def __init__(self, ages, gaussian, halfwidth, amplitude, nominal_q):
        self.ages, self.gaussian, self.halfwidth = ages, gaussian, halfwidth
        self.amplitude, self.nominal_q = amplitude, nominal_q

    def draw_noise(self, like: Tensor) -> Tensor:
        """Draw unit sensor noise, optionally with time axes before the channels."""
        gaussian = self.gaussian
        while gaussian.ndim < like.ndim:
            gaussian = gaussian.unsqueeze(-2)
        if gaussian.all():
            return torch.randn_like(like)
        if gaussian.any():
            return torch.where(gaussian, torch.randn_like(like), torch.rand_like(like) * 2 - 1)
        return torch.rand_like(like) * 2 - 1

    @classmethod
    def from_context(cls, context: Tensor, schema: StateSchema) -> SensorReadout:
        ages = named_field(context, schema, "sensor.lags", context=True)
        uplink = _optional(context, schema, "sensor.uplink_delay_s", context=True)
        if uplink is not None:
            dt = named_field(context, schema, "physics_dt_s", context=True)
            shape = (*ages.shape[:-1], 1)
            ages = ages + uplink.reshape(shape) / dt.reshape(shape)
        distribution = _optional(context, schema, "sensor.noise_distribution", context=True)
        shape = (*ages.shape[:-1], 1)
        gaussian = (torch.zeros(shape, dtype=torch.bool, device=ages.device) if distribution is None
                    else distribution.reshape(shape) > .5)
        # The inherited field name is retained for schema compatibility: for
        # distribution=1 these values are SI standard deviations, not widths.
        halfwidth = named_field(context, schema, "sensor.noise_halfwidth", context=True)
        nominal = named_field(context, schema, "sensor.nominal", context=True)
        scale = named_field(context, schema, "sensor.noise_scale", context=True)
        amplitude = (nominal < .5).to(halfwidth.dtype).reshape(shape) * scale.reshape(shape)
        nominal_q = named_field(context, schema, "controller.nominal", context=True)
        return cls(ages, gaussian, halfwidth, amplitude, nominal_q)


def deployed_frame(values: Tensor, count: Tensor, gimbal_history: Tensor, previous_action: Tensor,
                   goal: Tensor, readout: SensorReadout, *, noise: bool = True,
                   noise_draw: Tensor | None = None) -> Tensor:
    """Assemble the deployed 45D frame from newest-first physical-tick queues."""
    raw = _delayed(values, readout.ages, count, [0] * 6 + [1] * 6 + [2] * 3 + [3] * 3)
    gaussian = readout.gaussian
    if noise:
        draw = readout.draw_noise(raw) if noise_draw is None else noise_draw
        if draw.shape != raw.shape or draw.device != raw.device or draw.dtype != raw.dtype:
            raise ValueError('Supplied sensor noise must match the raw observation shape/device/dtype')
        raw = raw + draw * readout.halfwidth * readout.amplitude
    gravity = raw[..., 15:18]
    gravity = torch.where(gaussian, gravity / gravity.norm(dim=-1, keepdim=True).clamp_min(1e-6), gravity)
    raw = torch.cat((raw[..., :15], gravity), -1)
    pairs = (raw[..., :4].double() - readout.nominal_q[..., :4].double()).unflatten(-1, (2, 2))
    turns = torch.floor(pairs.mean(-1) / (2 * math.pi) + .5)
    relative = (pairs - turns[..., None] * (2 * math.pi)).flatten(-2).to(raw.dtype)
    base = torch.cat((goal[..., :4], raw[..., 12:15] * .5, raw[..., 15:18], relative,
                      torch.zeros_like(raw[..., 4:6]), raw[..., 6:12] * .1,
                      previous_action, goal[..., 4:11]), -1)
    gimbal = _delayed(gimbal_history, readout.ages, count, [0, 0, 0, 1, 1])
    gimbal = torch.cat((_unit_phase(gimbal[..., :2]), gimbal[..., 2:]), -1)
    return torch.cat((base, goal[..., 11:16], gimbal), -1).clamp(-100, 100)


def advance_conditions(state: Tensor, goal: Tensor, context: Tensor,
                       previous_action: Tensor, schema: StateSchema):
    """Advance the visible clock while retaining the actual issued task goal.

    No learned course.goal or privileged physical phase selects a new request.
    The learner bounds each rollout at its start's known condition boundary.
    """
    _check_goal(state, goal, schema)
    return state, advance_goal(goal, schema.policy_dt_s), context


def advance_goal(goal: Tensor, policy_dt_s: float) -> Tensor:
    """Advance the visible request clock by one policy interval.

    LabCourseTasks exposes RequestProtocol.clock_s, which is zero without an
    active jump request (unlike the legacy flat environment's wall clock).
    """
    clock = torch.where(goal[..., 8:9] > .5, (goal[..., 10:11] + policy_dt_s).clamp_max(5),
                        torch.zeros_like(goal[..., 10:11]))
    return torch.cat((goal[..., :10], clock, goal[..., 11:]), -1)


def imagination_horizon(state: Tensor, goal: Tensor, context: Tensor,
                        schema: StateSchema) -> Tensor:
    """Read the real start's deadline once; never trust a predicted deadline.

    Ordinary rollout endpoints stay strictly before the next known exogenous
    condition change. A zero result means no safe ordinary interval at this
    start. Until a deployable request transition model is integrated, explicit
    jump/recovery requests and dynamic command controllers use one physical
    policy interval plus critic value. Predicted flags cannot extend a rollout.
    """
    remaining = named_field(state, schema, "course.imagination_remaining_s").reshape(state.shape[:-1])
    dynamic = _optional(state, schema, "course.dynamic_goal")
    if dynamic is None:
        dynamic = _optional(context, schema, "course.dynamic_goal", context=True)
    dynamic = (torch.zeros_like(remaining) if dynamic is None else dynamic.reshape(state.shape[:-1]))
    return known_horizon(remaining, dynamic, goal, schema.policy_dt_s)


def known_horizon(remaining_s: Tensor, dynamic: Tensor, goal: Tensor, policy_dt_s: float) -> Tensor:
    """Steps strictly before the next known command change; requests/dynamic goals use one."""
    steps = (torch.ceil(remaining_s.clamp_min(0) / policy_dt_s - 1e-5).long() - 1).clamp_min(0)
    request = (goal[..., 7:9].amax(-1) > .5) | (dynamic > .5)
    return torch.where(request, torch.ones_like(steps), steps)
