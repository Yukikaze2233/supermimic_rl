"""Held-out autoregressive physical errors using real actions and independent environment rows."""
from dataclasses import fields
import torch
from .fitting import recorded_rollout
from .model import Ticks
from .physical import increments
from .records import CONTACT_BODIES, FORBIDDEN_BODIES, WHEEL_BODIES, decode_physical

REPORT_SLICES = {"position_m": slice(0, 3), "rotation_rad": slice(3, 6), "velocity_m_s": slice(6, 9),
                 "angular_velocity_rad_s": slice(9, 12), "leg_q_rad": slice(12, 16),
                 "gimbal_q_rad": slice(16, 18), "leg_qd_rad_s": slice(18, 22), "wheel_qd_rad_s": slice(22, 24)}


def _confusion(predicted, actual):
    """Absent positive labels cannot establish recall, even with zero errors."""
    tp, fp = int((predicted & actual).sum()), int((predicted & ~actual).sum())
    tn, fn = int((~predicted & ~actual).sum()), int((~predicted & actual).sum())
    return {'true_positive': tp, 'false_positive': fp, 'true_negative': tn, 'false_negative': fn,
            'positive_labels': tp + fn, 'negative_labels': tn + fp,
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
            'false_positive_rate': fp / (fp + tn) if fp + tn else None,
            'status': 'observed_positives' if tp + fn else 'no_positive_labels'}


def _contact_events(predicted, actual, keep, threshold):
    """Native instantaneous contact decisions, separate from confirmed failure."""
    nonwheel = [index for index, name in enumerate(CONTACT_BODIES) if name not in WHEEL_BODIES]

    def events(ticks):
        # terrain.py passes wheel_ground_force_n (normal support), not the
        # wheel resultant force, to both jump and recovery physical observers.
        on, off = ticks.wheel_load >= 5., ticks.wheel_load.clamp_min(0.) <= 2.
        return {
            'left_wheel_supported': on[..., 0], 'right_wheel_supported': on[..., 1],
            'both_wheels_supported': on.all(-1), 'both_wheels_airborne': off.all(-1),
            'one_wheel_supported': on.any(-1) & off.any(-1),
            'forbidden_body_contact': (ticks.forbidden_load > threshold).any(-1),
            'nonwheel_body_contact': (ticks.body_normal_load[..., nonwheel] > threshold).any(-1),
            'base_normal_contact': ticks.body_normal_load[..., CONTACT_BODIES.index('base_link')] > threshold,
            'base_resultant_contact': ticks.base_contact_force > threshold,
        }

    predicted_events, actual_events = events(predicted), events(actual)
    return {name: _confusion(value[keep], actual_events[name][keep])
            for name, value in predicted_events.items()}


class _ContactClockAudit:
    """Isolate contact prediction's effect using the actual native failure clock.

    Both paths read recorded task exceptions. Pose inputs are deliberately
    neutral so a tilt/height failure cannot be misreported as a contact error.
    This is a held-out contact-only audit, not an imagined task success metric.
    """
    def __init__(self, termination, windows):
        self.termination = termination
        memory = windows['failure_memory'][:, 0].clone()
        memory[:, 1] = 0.
        memory[:, 2] = memory[:, 2].long() & 4
        self.predicted, self.actual = memory.clone(), memory
        self.valid = (windows['failure_conditions_valid'][..., 0] > .5).all(-1)

    def advance(self, predicted, actual, conditions, rows):
        n, count, _ = predicted.forbidden_load.shape
        height = predicted.forbidden_load.new_full((n, count, 1), float('inf'))
        gravity = predicted.forbidden_load.new_zeros(n, count, 3)
        gravity[..., 2] = -1.
        for name, ticks in (('predicted', predicted), ('actual', actual)):
            memory = getattr(self, name)
            result = self.termination.advance(memory[rows], height, gravity, ticks.forbidden_load,
                recovery=conditions[:, 0] > .5, body_contact_allowed=conditions[:, 1] > .5)
            memory[rows] = result.memory

    def report(self, complete):
        valid = self.valid & complete
        p, a = (self.predicted[:, 2] != 0)[valid], (self.actual[:, 2] != 0)[valid]
        error = (self.predicted[:, 0] - self.actual[:, 0]).abs()[valid]
        return {
            'scope': 'contact-only latch over the full horizon, with recorded task exceptions',
            'audited_windows': int(valid.sum()), 'invalid_condition_windows': int((~self.valid).sum()),
            'incomplete_prediction_windows': int((~complete).sum()),
            'confirm_s': self.termination.limits.forbidden_contact_confirm_s,
            'physics_dt_s': self.termination.dt,
            **_confusion(p, a),
            'clock_absolute_error_mean_s': float(error.mean()) if len(error) else None,
        }


@torch.no_grad()
def validate(fit, replay, env_ids, horizon, batch, generator, device, *,
             contact_threshold_n=None, new_episode_namespace=None, termination=None,
             required_rows=None, start_rows=None, contact_floor=None):
    """Sample each horizon independently, including short/failing episodes at H1.

    Reusing H32 windows for H1 excludes every episode shorter than 32 steps and
    all newly collected 24-step segments. Such a curve is conditional on long
    survival; it cannot validate one-step errors on the full replay population.

    Population masks use physical ring slots, shaped [capacity, num_envs].
    ``required_rows`` must hold throughout the window (e.g. recorded source
    metadata); ``start_rows`` selects its initial task/terrain/event only.
    Absent populations are reported explicitly, never assigned zero error.
    """
    model = fit.model
    for mask in (required_rows, start_rows):
        if mask is not None and (mask.dtype != torch.bool or mask.device != replay.device
                                 or mask.shape != (replay.capacity, replay.num_envs)):
            raise ValueError('Validation population must be a boolean mask over physical replay slots')
    report = {}
    for length in sorted({value for value in (1, 4, 8, 16, horizon) if value <= horizon}):
        candidates = replay.window_candidates(length, env_ids)
        original_count = candidates.numel()
        offset, env = candidates // replay.num_envs, candidates % replay.num_envs
        slots = (offset + replay.head - replay.size) % replay.capacity
        selected = torch.ones_like(candidates, dtype=torch.bool)
        if start_rows is not None:
            selected &= start_rows[slots, env]
        if required_rows is not None:
            for step in range(length):
                selected &= required_rows[(slots + step) % replay.capacity, env]
        candidates = candidates[selected]
        if not candidates.numel():
            report[f'{length}_step'] = {'status': 'no_eligible_windows', 'eligible_windows': 0,
                'unfiltered_eligible_windows': original_count, 'sampled_windows': 0}
            continue
        windows = replay.sample_windows(batch, length, generator=generator, env_ids=env_ids,
                                        candidates=candidates)
        windows = {name: value.to(device) for name, value in windows.items()}
        contact_audit = None if termination is None else _ContactClockAudit(termination, windows)
        floored_audit = (None if termination is None or contact_floor is None
                         else _ContactClockAudit(termination, windows))
        truncated = torch.zeros(batch, dtype=torch.bool, device=device)
        nonfinite_motion = torch.zeros_like(truncated)
        complete = torch.zeros_like(truncated)
        final = None
        for step, rows, prediction, supported in recorded_rollout(model, windows):
            truncated[rows[~supported]] = True
            finite = torch.ones(len(rows), dtype=torch.bool, device=device)
            for field in fields(prediction.ticks.state):
                finite &= torch.isfinite(getattr(prediction.ticks.state, field.name)).flatten(1).all(-1)
            for value in (prediction.ticks.body_normal_load, prediction.ticks.wheel_load,
                          prediction.ticks.resultant_contact_force, prediction.ticks.passive_knee_position,
                          prediction.ticks.passive_knee_velocity):
                finite &= torch.isfinite(value).flatten(1).all(-1)
            nonfinite_motion[rows[~finite]] = True
            if contact_audit is not None:
                true_ticks = Ticks.decode(windows['ticks'][rows, step], decode_physical(windows['physical'][rows, step]))
                contact_audit.advance(prediction.ticks, true_ticks, windows['failure_conditions'][rows, step], rows)
                if floored_audit is not None:
                    floored_audit.advance(prediction.ticks.with_contact_floor(*contact_floor), true_ticks,
                                          windows['failure_conditions'][rows, step], rows)
            if step == length - 1:
                keep = supported & ~nonfinite_motion[rows]
                complete[rows[keep]] = True
                final = (rows, keep, prediction)
        values = {'sampling': 'independent eligible windows for this horizon',
            'eligible_windows': candidates.numel(), 'unfiltered_eligible_windows': original_count,
            'sampled_windows': batch, 'complete_prediction_windows': int(complete.sum()),
            'terrain_truncated_fraction': float(truncated.float().mean()),
            'nonfinite_motion_fraction': float(nonfinite_motion.float().mean()),
            'error_scope': 'complete finite predictions only; missing terrain is reported as truncation',
            'terminal_endpoint_fraction': float(windows['terminated'][:, -1].float().mean())}
        if new_episode_namespace is not None:
            values['new_physics_generation_fraction'] = float(
                (windows['start_episode_id'][:, 0] >= new_episode_namespace).float().mean())
        if contact_audit is not None:
            values['forbidden_contact_latch'] = contact_audit.report(complete)
        if floored_audit is not None:
            values['forbidden_contact_latch_floored'] = floored_audit.report(complete)
        if final is None or not complete.any():
            values['status'] = 'no_complete_predictions'
            report[f'{length}_step'] = values
            continue
        rows, keep, prediction = final
        error = increments(decode_physical(windows['end_physical'][rows, -1]), prediction.state)[keep]
        truth = Ticks.decode(windows['ticks'][rows, -1], decode_physical(windows['physical'][rows, -1]))
        height = prediction.height_above_ground
        values.update({
            **{name: error[:, part].square().mean().sqrt().item() for name, part in REPORT_SLICES.items()},
            'height_m': (height - windows['end_height'][rows, -1])[keep].square().mean().sqrt().item(),
            'body_normal_load_n': (prediction.ticks.body_normal_load - truth.body_normal_load)[keep].square().mean().sqrt().item(),
            'wheel_support_load_n': (prediction.ticks.wheel_load - truth.wheel_load)[keep].square().mean().sqrt().item(),
            'wheel_resultant_force_n': (prediction.ticks.wheel_contact_force - truth.wheel_contact_force)[keep].square().mean().sqrt().item(),
            'base_resultant_force_n': (prediction.ticks.base_contact_force - truth.base_contact_force)[keep].square().mean().sqrt().item(),
            'passive_knee_position_rad': (prediction.ticks.passive_knee_position - truth.passive_knee_position)[keep].square().mean().sqrt().item(),
            'passive_knee_velocity_rad_s': (prediction.ticks.passive_knee_velocity - truth.passive_knee_velocity)[keep].square().mean().sqrt().item()})
        if contact_threshold_n is not None:
            predicted = prediction.ticks.forbidden_load > contact_threshold_n
            actual = truth.forbidden_load > contact_threshold_n
            values['forbidden_contact'] = {'threshold_n': contact_threshold_n, 'bodies': {}}
            for index, name in enumerate(FORBIDDEN_BODIES):
                p, a = predicted[keep, ..., index], actual[keep, ..., index]
                values['forbidden_contact']['bodies'][name] = _confusion(p, a)
            events = {'scope': 'four physical ticks in the final interval of complete finite predictions',
                      'support_on_n': 5., 'support_off_n': 2., 'body_contact_n': contact_threshold_n,
                      'raw': _contact_events(prediction.ticks, truth, keep, contact_threshold_n)}
            if contact_floor is not None:
                floored = prediction.ticks.with_contact_floor(*contact_floor)
                events['floored'] = _contact_events(floored, truth, keep, contact_threshold_n)
                values['forbidden_contact_floored'] = {'threshold_n': contact_threshold_n, 'bodies': {
                    name: _confusion((floored.forbidden_load > contact_threshold_n)[keep, ..., index],
                                     actual[keep, ..., index]) for index, name in enumerate(FORBIDDEN_BODIES)}}
            values['contact_events'] = events
        report[f'{length}_step'] = values
    return report




@torch.no_grad()
def calibrate_contact_floor(fit, replay, env_ids, batch, generator, device, *, threshold_n: float,
                            quantile: float = .99, horizon: int = 4, min_negatives: int = 100) -> dict:
    """Per-channel floors for non-wheel contact loads from held-out recorded windows.

    A floor is the ``quantile`` of raw predictions on ticks whose true load is at
    most ``threshold_n`` (no contact at the native threshold). Predictions below
    it are noise and imagination treats them as zero. The report gives false
    positives and recall at the threshold before and after flooring; wheels are
    never floored because their support load decides airborne phases.
    """
    names = (*CONTACT_BODIES, 'base_resultant')
    floor = torch.zeros(len(names), device=device)
    report = {'threshold_n': threshold_n, 'quantile': quantile, 'ticks': 0, 'channels': {}}
    candidates = replay.window_candidates(horizon, env_ids)
    if not candidates.numel():
        # Zero floors leave imagination on raw loads until held-out windows exist.
        return {'body_floor': floor[:-1], 'base_floor': floor[-1], 'report': {**report, 'status': 'no_eligible_windows'}}
    windows = replay.sample_windows(batch, horizon, generator=generator, env_ids=env_ids, candidates=candidates)
    windows = {name: value.to(device) for name, value in windows.items()}
    predicted, actual = [], []
    for step, rows, prediction, supported in recorded_rollout(fit.model, windows):
        keep = supported & torch.isfinite(prediction.ticks.body_normal_load).flatten(1).all(-1)
        keep &= torch.isfinite(prediction.ticks.base_contact_force).flatten(1).all(-1)
        truth = Ticks.decode(windows['ticks'][rows, step], decode_physical(windows['physical'][rows, step]))
        for target, ticks in ((predicted, prediction.ticks), (actual, truth)):
            target.append(torch.cat((ticks.body_normal_load, ticks.base_contact_force[..., None]), -1)[keep].flatten(0, 1))
    predicted, actual = torch.cat(predicted), torch.cat(actual)
    report['ticks'] = len(predicted)
    for index, name in enumerate(names):
        if name in WHEEL_BODIES:
            continue
        negative, positive = actual[:, index] <= threshold_n, actual[:, index] > threshold_n
        if int(negative.sum()) >= min_negatives:
            floor[index] = torch.quantile(predicted[negative, index].float(), quantile).clamp_min(0)
        raw = predicted[:, index] > threshold_n
        floored = predicted[:, index] > torch.maximum(floor[index], torch.tensor(threshold_n, device=device))
        report['channels'][name] = {
            'floor_n': float(floor[index]), 'negatives': int(negative.sum()), 'positives': int(positive.sum()),
            'false_positive_raw': int((raw & negative).sum()), 'false_positive_floored': int((floored & negative).sum()),
            'true_positive_raw': int((raw & positive).sum()), 'true_positive_floored': int((floored & positive).sum())}
    return {'body_floor': floor[:-1], 'base_floor': floor[-1], 'report': report}
