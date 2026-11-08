"""First-interval action augmentation certified on recorded controller feedback.

Only dynamics fitting may consume the returned windows. Reward, Actor frames,
and estimator labels still describe the original action and must not be used
as counterfactual labels. Normalization must observe the original windows.

With the pinned memoryless leg execution, equal capped PD effort at all four
true control starts also gives equal downstream effort: identified clipping
and passive friction see the same input and velocity. Wheel actions, their
initial queue, external forces, and the physical labels are unchanged. This
is an open-loop H-step equivalence, not a closed-loop policy counterfactual.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping

import torch
from torch import Tensor

from ..contracts import StateSchema, action_bounds_from_source
from ..course.identified_execution import ACTION_ORDER, MODEL_PINS
from .execution import LegExecution
from .model import Ticks
from .records import MEMORY_DIM, PHYSICAL_DIM, TICK_LABEL_DIM, decode_physical


@dataclass(frozen=True)
class ActionEquivalenceResult:
    windows: dict[str, Tensor]
    eligible_pairs: Tensor       # [B, 2], a nonempty certified plateau interval exists
    changed_rows: Tensor         # [B], the sampled action passed exact effort certification
    delta_phi: Tensor            # [B], actual accepted target-phi change; zero otherwise
    selected_pair: Tensor        # [B], left=0, right=1, unchanged=-1
    max_effort_difference: Tensor  # [B], proposed capped-PD difference over four ticks/legs
    delta_bounds: Tensor         # [B, 2, 2], feasible phi interval [lower, upper]; zero if ineligible


@dataclass(frozen=True)
class FirstStepLegActionEquivalence:
    action_bounds: tuple[float, ...]

    @classmethod
    def from_runtime(cls, schema: StateSchema, runtime_json: bytes) -> FirstStepLegActionEquivalence:
        """Bind to the actual pinned runtime, not a similarly named template.

        Old-data compatibility belongs to the auxiliary loader. This checks
        the execution contract of the learner receiving those migrated rows.
        """
        source, runtime = json.loads(schema.source), json.loads(runtime_json)
        if hashlib.sha256(runtime_json).hexdigest() != source.get('runtime_sha256'):
            raise ValueError('Action equivalence runtime SHA differs from the schema source')
        for name in ('action_transport', 'action_latch', 'communication'):
            if runtime.get(name, {}).get('enabled') is not False:
                raise ValueError('Action equivalence requires explicitly disabled ' + name)
        if any(name not in runtime or runtime[name] is not None for name in ('pair_domain', 'motor_envelope')):
            raise ValueError('Action equivalence requires no pair domain or legacy motor envelope')
        sensors = runtime.get('sensors', {})
        if (schema.substeps_per_step != 4 or schema.policy_dt_s != .02
                or sensors.get('physics_dt_s') != .005 or sensors.get('policy_dt_s') != .02):
            raise ValueError('Action equivalence requires four 5 ms ticks per 20 ms interval')
        bounds = action_bounds_from_source(source)
        if tuple(runtime.get('task', {}).get('action_bounds', ())) != bounds:
            raise ValueError('Action equivalence runtime action bounds differ from the schema source')
        execution = source.get('execution', {})
        if (execution.get('coordinate') != 'native_output_shaft'
                or execution.get('encoder_holding_compensation') is not False
                or tuple(execution.get('action_order', ())) != ACTION_ORDER
                or execution.get('model_pins') != MODEL_PINS):
            raise ValueError('Action equivalence requires the pinned identified output-shaft execution')
        return cls(bounds)

    @torch.no_grad()
    def augment(self, windows: Mapping[str, Tensor], unit_samples: Tensor) -> ActionEquivalenceResult:
        """Sample one pair's plateau interval and change only the first action.

        ``unit_samples`` is an independent caller-owned CPU RNG draw [B, 2]
        in [0, 1): the first column selects a feasible pair, the second samples
        its interval. No simulator, replay, or learner RNG is consumed here.
        Missing evidence leaves samples untouched, rather than dropping them.
        """
        action = windows['action']
        if action.ndim != 3 or action.shape[-1] != 6 or action.shape[1] < 2:
            raise ValueError('Action equivalence needs batched multi-step six-axis actions')
        batch, horizon = action.shape[:2]
        if (unit_samples.shape != (batch, 2) or unit_samples.device.type != 'cpu'
                or not unit_samples.is_floating_point() or not torch.isfinite(unit_samples).all()
                or not ((unit_samples >= 0) & (unit_samples < 1)).all()):
            raise ValueError('Action equivalence requires finite CPU unit samples with shape [B, 2] in [0, 1)')
        empty_pairs = torch.zeros((batch, 2), dtype=torch.bool, device=action.device)
        empty_rows = empty_pairs[:, 0]
        zero = action.new_zeros(batch)
        unchanged = ActionEquivalenceResult(dict(windows), empty_pairs, empty_rows, zero,
            torch.full((batch,), -1, dtype=torch.long, device=action.device), zero,
            action.new_zeros((batch, 2, 2)))
        required = ('physical', 'ticks', 'memory', 'controller_parameters', 'motor_strength',
                    'valid', 'policy_owned', 'end_policy_owned', 'reward_context_valid')
        if any(name not in windows for name in required):
            return unchanged
        expected = {'physical': (PHYSICAL_DIM,), 'ticks': (4, TICK_LABEL_DIM), 'memory': (MEMORY_DIM,),
                    'controller_parameters': (30,), 'motor_strength': (6,)}
        for name, shape in expected.items():
            if windows[name].shape != (batch, horizon, *shape) or windows[name].device != action.device:
                raise ValueError('Action equivalence record shape/device differs: ' + name)
        eligible = torch.ones(batch, dtype=torch.bool, device=action.device)
        for name in ('valid', 'policy_owned', 'end_policy_owned', 'reward_context_valid'):
            flag = windows[name]
            if flag.shape not in ((batch, horizon), (batch, horizon, 1)) or flag.device != action.device:
                raise ValueError('Action equivalence evidence shape/device differs: ' + name)
            eligible &= (flag.reshape(batch, horizon) == 1).all(-1)

        original = action[:, 0]
        parameters = windows['controller_parameters'][:, 0]
        nominal, stiffness, damping, cap, bias = parameters.split(6, -1)
        strength = windows['motor_strength'][:, 0]
        start = decode_physical(windows['physical'][:, 0])
        ticks = Ticks.decode(windows['ticks'][:, 0], start)
        feedback = torch.cat((start.joint_position[:, None, :4], ticks.leg_position[:, :3]), 1)
        feedback = feedback + bias[:, None, :4]
        velocity = torch.cat((start.joint_velocity[:, None, :4], ticks.joint_velocity[:, :3, :4]), 1)
        bounds = original.new_tensor(self.action_bounds)
        eligible &= (torch.isfinite(parameters).all(-1) & torch.isfinite(strength).all(-1)
                     & torch.isfinite(feedback).all((1, 2)) & torch.isfinite(velocity).all((1, 2))
                     & torch.isfinite(original).all(-1) & (original.abs() <= bounds).all(-1)
                     & (stiffness[:, :4] > 0).all(-1) & (damping[:, :4] >= 0).all(-1)
                     & (cap[:, :4] > 0).all(-1) & (strength[:, :4] > 0).all(-1))
        legs = LegExecution(nominal, stiffness, damping, cap, self.action_bounds[0])
        tick_legs = LegExecution(nominal[:, None], stiffness[:, None], damping[:, None], cap[:, None],
                                 self.action_bounds[0])
        target = legs.target(original, feedback[:, 0])
        effort = tick_legs.motor_effort(target[:, None], feedback, velocity, strength[:, None])

        # The two serial caps produce a plateau at C*min(s,1), reached when
        # requested PD crosses C/max(s,1). Intersect that exact region over
        # every recorded tick; an unsaturated tick admits no nonzero shift.
        plateau = cap[:, None, :4] * strength[:, None, :4].clamp(max=1.)
        positive, negative = effort == plateau, effort == -plateau
        motor_saturated = (positive | negative).all(1)
        threshold = cap[:, None, :4].double() / strength[:, None, :4].double().clamp(min=1.)
        center = feedback.double() + damping[:, None, :4].double() * velocity.double() / stiffness[:, None, :4].double()
        offset = threshold / stiffness[:, None, :4].double()
        low = torch.where(positive, center + offset, -torch.inf).amax(1)
        high = torch.where(negative, center - offset, torch.inf).amin(1)
        target64, raw64 = target.double().unflatten(-1, (2, 2)), original[:, :4].double().unflatten(-1, (2, 2))
        low, high = low.unflatten(-1, (2, 2)), high.unflatten(-1, (2, 2))
        bound = float(self.action_bounds[0])
        lower = torch.stack((2 * (target64[..., 0] - high[..., 0]),
                             2 * (low[..., 1] - target64[..., 1]),
                             (raw64[..., 0] - bound) / 2, (-bound - raw64[..., 1]) / 2), -1).amax(-1)
        upper = torch.stack((2 * (target64[..., 0] - low[..., 0]),
                             2 * (high[..., 1] - target64[..., 1]),
                             (raw64[..., 0] + bound) / 2, (bound - raw64[..., 1]) / 2), -1).amin(-1)
        pairs = (eligible[:, None] & motor_saturated.unflatten(-1, (2, 2)).all(-1)
                 & torch.isfinite(lower) & torch.isfinite(upper) & (upper > lower))
        delta_bounds = torch.where(pairs[..., None], torch.stack((lower, upper), -1), 0.)
        units = unit_samples.to(device=action.device, dtype=torch.float64)
        chosen = torch.where(pairs.all(-1), (units[:, 0] >= .5).long(), pairs[:, 1].long())
        chosen_bounds = delta_bounds.gather(1, chosen[:, None, None].expand(-1, 1, 2))[:, 0]
        delta = (chosen_bounds[:, 0] + units[:, 1] * (chosen_bounds[:, 1] - chosen_bounds[:, 0])).to(action.dtype)
        proposed = original.clone()
        shift = torch.zeros((batch, 2), dtype=action.dtype, device=action.device).scatter(1, chosen[:, None], delta[:, None])
        proposed[:, :4] += torch.stack((-2 * shift, 2 * shift), -1).flatten(-2)
        proposed_target = legs.target(proposed, feedback[:, 0])
        proposed_effort = tick_legs.motor_effort(proposed_target[:, None], feedback, velocity, strength[:, None])
        difference = (proposed_effort - effort).abs().amax((1, 2))
        difference = torch.where(pairs.any(-1), difference, 0.)

        # Rounding a common-mean raw pair must not switch its whole-turn
        # admission branch. Final certification deliberately uses no epsilon.
        def winding(target, raw):
            desired = nominal[:, :4] + .25 * raw[:, :4]
            return torch.round((target.double() - desired.double()) / (2 * math.pi))

        changed = (pairs.any(-1) & (difference == 0) & (proposed.abs() <= bounds).all(-1)
                   & (winding(proposed_target, proposed) == winding(target, original)).all(-1)
                   & (proposed != original).any(-1))
        accepted = torch.where(changed[:, None], proposed, original)
        output = dict(windows)
        output['action'], output['memory'] = action.clone(), windows['memory'].clone()
        output['action'][:, 0] = accepted
        output['memory'][:, 1, 6:12] = torch.where(changed[:, None], accepted, windows['memory'][:, 1, 6:12])
        phi_change = ((proposed_target - target).unflatten(-1, (2, 2)) * target.new_tensor([-1., 1.])).sum(-1)
        phi_change = phi_change.gather(1, chosen[:, None])[:, 0]
        return ActionEquivalenceResult(output, pairs, changed, torch.where(changed, phi_change, 0.),
            torch.where(changed, chosen, -1), difference, delta_bounds.to(action.dtype))
