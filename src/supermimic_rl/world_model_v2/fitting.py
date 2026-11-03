"""Multi-step autoregressive fitting of the v2 world model on compact replay windows.

The rollout starts at the logged first state and feeds its own predictions back,
as in the paper (k = 8). Each step is scored in the true state's local frame,
normalized by the running output scale with physical floors (vibed's ERR_FLOOR:
a lower bound on the loss denominator, not a constraint on predictions).
Exogenous gimbal commands are read from data at every step; the wheel queue
and the previous action are advanced by code from the model's own ticks.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch
from torch import Tensor

from .model import FlatTerrain, Ticks, WorldModelV2
from .physical import increments
from .records import CONTACT_LABEL_DIM, decode_memory, decode_physical

# Increment floors: position, rotation, velocity, angular velocity, joint angles, joint rates.
INCREMENT_FLOOR = (3e-3,) * 3 + (5e-3,) * 3 + (2e-2,) * 3 + (5e-2,) * 3 + (3e-3,) * 6 + (5e-2,) * 8
# Full physical increments plus wheel and forbidden-body loads [N].
TICK_FLOOR = INCREMENT_FLOOR + (5.,) * CONTACT_LABEL_DIM + (3e-3,) * 2 + (5e-2,) * 2


def tick_residual(predicted: Ticks, actual: Ticks) -> Tensor:
    """Local-frame motion residuals handle rotations and gimbal yaw periodically."""
    return torch.cat((increments(actual.state, predicted.state),
                      predicted.wheel_load - actual.wheel_load,
                      predicted.body_normal_load - actual.body_normal_load,
                      predicted.resultant_contact_force - actual.resultant_contact_force,
                      predicted.passive_knee_position - actual.passive_knee_position,
                      predicted.passive_knee_velocity - actual.passive_knee_velocity), -1)


@dataclass(frozen=True)
class FitTerms:
    state: Tensor
    ticks: Tensor
    per_step_state: Tensor
    terrain_truncated_fraction: Tensor


def recorded_rollout(model, windows):
    """Yield original row IDs and predictions up to each terrain-query boundary.

    The physical prediction that leaves the known scan domain is still scored:
    its position error supplies the correction gradient. Only later steps are
    absent. No invented floor, recorded future scan or teacher-forced state is
    fed into that trajectory. Nonfinite physical outputs remain visible to the
    optimizer's existing skip handling rather than being masked as good data.
    """
    rows = torch.arange(len(windows['action']), device=windows['action'].device)
    state = decode_physical(windows['physical'][:, 0])
    height, scan = windows['height'][:, 0], windows['scan'][:, 0]
    memory = decode_memory(windows['memory'][:, 0])
    for step in range(windows['action'].shape[1]):
        logged = decode_memory(windows['memory'][rows, step])
        memory = replace(memory, gimbal_mode=logged.gimbal_mode, gimbal_target_pitch=logged.gimbal_target_pitch,
                         gimbal_target_yaw=logged.gimbal_target_yaw,
                         gimbal_yaw_rate_target=logged.gimbal_yaw_rate_target)
        terrain = ({'terrain_tile': windows['terrain_tile'][rows, step, 0]}
                   if 'terrain_tile' in windows else {})
        prediction = model.step(state, height, scan, memory, windows['sim_context'][rows, step],
                                windows['motor_strength'][rows, step, 4:6], windows['action'][rows, step], **terrain)
        tick_valid = getattr(prediction, 'tick_terrain_valid', None)
        supported = (torch.ones_like(rows, dtype=torch.bool) if tick_valid is None else tick_valid.all(-1))
        yield step, rows, prediction, supported
        keep = supported & windows['valid'][rows, step]
        if step + 1 == windows['action'].shape[1] or not keep.any():
            break
        rows = rows[keep]
        state = prediction.state.map(lambda value: value[keep])
        height, scan = prediction.height_above_ground[keep], prediction.terrain_scan[keep]
        memory = replace(prediction.memory, **{field.name: getattr(prediction.memory, field.name)[keep]
                                             for field in fields(prediction.memory)})


class WorldModelFit:
    RECORD_FIELDS = ('physical', 'end_physical', 'height', 'scan', 'memory', 'sim_context',
                     'motor_strength', 'action', 'ticks', 'valid')
    OPTIONAL_RECORD_FIELDS = ('reward_route', 'terrain_tile')

    def __init__(self, model: WorldModelV2):
        self.model = model
        motion = model.motion
        floor = motion.output_norm.mean.new_tensor(TICK_FLOOR * motion.TICKS)
        if floor.numel() != motion.output_dim:
            raise ValueError("Error floors must cover every motion output")
        motion.register_buffer("error_floor", floor, persistent=False)

    @torch.no_grad()
    def observe(self, windows: dict[str, Tensor]) -> None:
        """Update running input/output statistics from real single steps only."""
        model = self.model
        start, end = decode_physical(windows["physical"]), decode_physical(windows["end_physical"])
        features, _ = model.features(start, windows["height"], windows["scan"], decode_memory(windows["memory"]),
                                     windows["sim_context"], windows["action"])
        valid = windows["valid"]
        targets = model.motion.targets(windows["ticks"])
        model.motion.input_norm.update(features[valid])
        model.motion.output_norm.update(targets[valid])

    def terms(self, windows: dict[str, Tensor]) -> FitTerms:
        model, motion = self.model, self.model.motion
        if (isinstance(model.terrain, FlatTerrain) and 'reward_route' in windows
                and (windows['reward_route'][..., 1] != 0).any()):
            raise ValueError('Nonflat fitting requires the pinned atlas, not frozen start scans')
        scale = torch.maximum(motion.output_norm.std(), motion.error_floor)
        state_scale, tick_scale = motion.split(scale)
        batch, horizon = windows['action'].shape[:2]
        empty = windows['action'].new_zeros(batch)
        truncated = torch.zeros(batch, dtype=torch.bool, device=empty.device)
        state_terms, tick_terms = [], []
        for step, rows, prediction, supported in recorded_rollout(model, windows):
            true_start = decode_physical(windows["physical"][rows, step])
            true_end = decode_physical(windows["end_physical"][rows, step])
            true_ticks = Ticks.decode(windows["ticks"][rows, step], true_start)
            alive = windows["valid"][rows, step].to(empty.dtype)
            state_error = (increments(true_end, prediction.state) / state_scale).square().mean(-1)
            tick_error = (tick_residual(prediction.ticks, true_ticks)
                          / tick_scale).square().mean((-2, -1))
            state_terms.append(empty.index_copy(0, rows, state_error * alive))
            tick_terms.append(empty.index_copy(0, rows, tick_error * alive))
            truncated[rows[~supported]] = True
        state_terms.extend([empty] * (horizon - len(state_terms)))
        tick_terms.extend([empty] * (horizon - len(tick_terms)))
        per_step = torch.stack(state_terms, -1)
        count = per_step.numel()
        return FitTerms(per_step.sum() / count, torch.stack(tick_terms, -1).sum() / count,
                        per_step.detach(), truncated.float().mean())
