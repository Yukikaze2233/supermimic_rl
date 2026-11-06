"""Use the shared reward kernel in one native composer without changing upstream files."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import inspect
from pathlib import Path
from types import MethodType

import torch
from transformer_rl.formal.tasks.reward import PhysicalRewardComposer, RewardPhysics, RewardReference
from ..reward_kernel import (NATIVE_SOURCE_SHA256, JumpProgress, RewardExtensions, RewardHistory,
                             RewardParameters, advance_flight_latch, jump_tracking_targets,
                             physical_reward_densities)


class SharedRewardComposer(PhysicalRewardComposer):
    """Keep native event ownership/clock semantics around the differentiable kernel."""

    @torch.no_grad()
    def observe(self, sample: RewardPhysics, reference: RewardReference):
        window = self._window
        if window is None or window['steps'] >= self.substeps:
            raise RuntimeError('physical reward has no open complete-policy interval capacity')
        invalid = self._inputs(sample, reference)
        invalid |= (~torch.isfinite(sample.timestamp_s)
                    | (sample.timestamp_s - window['start'] - (window['steps'] + 1) * self.physics_dt_s).abs().gt(1e-8)
                    | (sample.episode_id != window['episode']) | (sample.control_epoch != window['control'])
                    | (reference.admitted_body_command != window['command']).any(-1)
                    | (reference.mode != window['mode']))
        progress = None
        if self.supermimic_reward_extensions.needs_jump_progress:
            # The request clock is held across the policy interval. Read its
            # stage once, while the physical phase and the ascent latch update
            # at every substep (after the task owner observed this substep).
            tasks = self.supermimic_jump_tasks
            if window['steps'] == 0:
                curve = window.get('jump_tracking_curve')
                window['jump_reference_stage'] = (tasks.jump_reference() if curve is None else curve).clock_stage
            latch = tasks.supermimic_flight_latch
            progress = JumpProgress(window['jump_reference_stage'], tasks.ascent_seen.clone(), latch.closed.clone(),
                                    tasks.landed_s.clone(), latch.lifted.clone(),
                                    tasks.peak_rise.clone() if self.supermimic_reward_extensions.jump_peak_completion else None,
                                    tasks.active_rise.clone() if self.supermimic_reward_extensions.jump_peak_completion else None)
        densities = physical_reward_densities(
            sample, reference,
            RewardHistory(window['dq'], window['action_cost'], window['action2_cost'], self._action_count),
            RewardParameters.from_composer(self), jump_progress=progress)
        if (invalid | densities.invalid).any():
            raise ValueError('reward input, PRE-reset identity, timing, phase or finite density bound differs')
        positive, cost = densities.positive, densities.cost
        components, gates, errors = densities.components, densities.gates, densities.errors
        ordinary = gates['ordinary']
        gimbal_active = gates['gimbal_spin_translate']
        dq = sample.policy_velocity_rad_s
        window['failure_scale'] = densities.failure_scale
        # Count only the physical exposure for which an error is defined.
        # In particular, a special row's body-vx=0 is a placeholder, whereas
        # zero-filled inactive gimbal errors are not successful tracking.
        scopes = {'ordinary_horizontal_tracking': (
            ordinary & ~gimbal_active, ('forward_m_s', 'yaw_rad_s', 'support_height_m'))}
        if reference.gimbal is not None:
            scopes['gimbal_spin_translate'] = (
                gimbal_active, ('gimbal_velocity_m_s', 'yaw_rad_s', 'support_height_m'))
            heading_valid = (gimbal_active & (reference.gimbal.speed_m_s > .1)
                             & sample.velocity_origin_world_m_s[:, :2].square().sum(-1).gt(0.))
            scopes['gimbal_heading_tracking'] = (heading_valid, ('gimbal_heading_rad',))
        for scope, (mask, names) in scopes.items():
            for name in names:
                key = scope + '/' + name
                error = torch.where(mask, errors[name].double(), 0.)
                for group, value in (
                    ('scoped_errors', error), ('scoped_error_squares', error.square()),
                    ('scoped_counts', mask.to(torch.float64)),
                ):
                    contribution = value * self.physics_dt_s
                    if key in window[group]:
                        window[group][key].add_(contribution)
                    else:
                        window[group][key] = contribution.clone()
        for key, values in (('positive', positive), ('raw_cost', cost), ('components', components), ('gates', gates),
                            ('errors', errors), ('error_squares', {key: value.double().square() for key, value in errors.items()})):
            for name, value in values.items():
                contribution = value.to(torch.float64) * self.physics_dt_s
                if name in window[key]:
                    window[key][name].add_(contribution)
                else:
                    window[key][name] = contribution.clone()
        window['owned'] &= sample.policy_owned
        window['dq'], window['steps'] = dq.detach().clone(), window['steps'] + 1
        window['last_timestamp'] = sample.timestamp_s.detach().clone()


class FirstFlightLatch:
    """Native owner of each request's first-real-flight latch (``advance_flight_latch``).

    Updated after every native ``observe_physics`` call, before the reward
    composer reads the same substep; cleared whenever the physical request changes.
    """

    def __init__(self, tasks):
        self.request = tasks.physical_id.clone()
        self.lifted = torch.zeros_like(tasks.ascent_seen)
        self.closed = torch.zeros_like(tasks.ascent_seen)

    def observe(self, tasks) -> None:
        self.request, self.lifted, self.closed = advance_flight_latch(
            self.request, self.lifted, self.closed, current=tasks.physical_id,
            ascent_seen=tasks.ascent_seen, air_ticks=tasks.air_ticks)


def install_flight_latch(tasks) -> FirstFlightLatch:
    latch = FirstFlightLatch(tasks)
    observe_physics = tasks.observe_physics

    def observe_and_latch(*args, **kwargs):
        result = observe_physics(*args, **kwargs)
        latch.observe(tasks)
        return result

    tasks.observe_physics = observe_and_latch
    tasks.supermimic_flight_latch = latch
    return latch


def install_jump_tracking_reference(native) -> None:
    """Correct the reference before native reward sampling and trace observers.

    Physics, the request protocol and its original contact phase stay with the
    native task. Only the reward's decision to switch to standing targets uses
    the existing first-real-flight evidence, shared with imagination.
    """
    from transformer_rl.formal.control.observation import CommandMode

    tasks, reward = native._terrain_course.tasks, native._formal_reward
    observe = reward.observe

    def observe_with_tracking_reference(sample, state, **kwargs):
        # The visible clock is constant for the four physical ticks. Share its
        # curve with the composer instead of adding four reference evaluations.
        window = reward.composer._window
        if 'jump_tracking_curve' not in window:
            window['jump_tracking_curve'] = tasks.jump_reference()
        curve = window['jump_tracking_curve']
        height, velocity = jump_tracking_targets(
            kwargs['phase'], curve.height_world_z_m, curve.velocity_world_z_m_s,
            kwargs['jump_height_world_z_m'],
            first_flight_closed=tasks.supermimic_flight_latch.closed)
        jump = kwargs['mode'] == int(CommandMode.JUMP)
        kwargs['jump_height_world_z_m'] = torch.where(jump, height, kwargs['jump_height_world_z_m'])
        kwargs['jump_velocity_world_z_m_s'] = torch.where(jump, velocity, kwargs['jump_velocity_world_z_m_s'])
        return observe(sample, state, **kwargs)

    reward.observe = observe_with_tracking_reference


def install_shared_reward(composer, *, verify_native=False, extensions: RewardExtensions = RewardExtensions(),
                          jump_tasks=None):
    """Instance-local installation; the frozen native source stays untouched.

    Extensions add opt-in terms (and may replace the success event); they are
    recorded in the receipt only when active, so native-parity runs keep their
    original source identity. The native audit compares unextended rewards.
    """
    source = Path(inspect.getfile(PhysicalRewardComposer))
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != NATIVE_SOURCE_SHA256:
        raise ValueError("Shared reward kernel requires its verified native reward source")
    if not isinstance(composer, PhysicalRewardComposer):
        raise TypeError("Shared reward installation requires the native composer")
    if verify_native and extensions.active:
        raise ValueError("The native reward audit compares the unextended reward")
    if extensions.needs_jump_progress and jump_tasks is None:
        raise ValueError('Jump flight/missed rewards require the native jump task owner')
    composer.supermimic_reward_extensions = extensions
    composer.supermimic_jump_tasks = jump_tasks
    if extensions.needs_jump_progress:
        install_flight_latch(jump_tasks)
    if extensions.success_event is not None:
        composer.tuning = replace(composer.tuning, success_event=float(extensions.success_event))
    if verify_native:
        audit = RewardKernelAudit(composer)
        composer.observe = audit.observe
        composer.shared_kernel_audit = audit
    else:
        composer.observe = MethodType(SharedRewardComposer.observe, composer)
    receipt = {"implementation": ("native_with_shared_density_verification" if verify_native
                                  else "shared_physical_reward_densities"), "physics_dt_s": composer.physics_dt_s,
               "native_source_sha256": actual,
               "kernel_sha256": hashlib.sha256(Path(inspect.getfile(physical_reward_densities)).read_bytes()).hexdigest()}
    if extensions.active:
        receipt["extensions"] = asdict(extensions)
        if not extensions.jump_peak_completion:
            # An absent opt-in keeps old source receipts and replay contracts.
            receipt['extensions'].pop('jump_peak_completion')
    return receipt


class RewardKernelAudit:
    """Compare the pure kernel to real native substep contributions before reset.

    This short audit runs the original composer. Production uses the shared
    composer above after parity is established, without paying for both paths.
    """

    def __init__(self, composer):
        self.composer, self.original = composer, composer.observe
        self.max_errors = {}
        self.samples = 0

    @torch.no_grad()
    def observe(self, sample, reference):
        composer, window = self.composer, self.composer._window
        history = RewardHistory(window['dq'], window['action_cost'], window['action2_cost'], composer._action_count)
        densities = physical_reward_densities(sample, reference, history, RewardParameters.from_composer(composer))
        if densities.invalid.any():
            raise ValueError('Shared density rejected a native reward sample')
        before = {name: value.clone() for name, value in window['components'].items()}
        self.original(sample, reference)
        if window['components'].keys() != densities.components.keys():
            raise ValueError('Shared/native reward component names differ')
        for name, value in densities.components.items():
            actual = window['components'][name] - before.get(name, 0.)
            error = (actual - value.double() * composer.physics_dt_s).abs().amax()
            self.max_errors[name] = torch.maximum(self.max_errors.get(name, torch.zeros_like(error)), error)
        self.samples += composer.num_envs

    def summary(self):
        errors = {name: float(value) for name, value in self.max_errors.items()}
        return {'physical_samples': self.samples, 'component_step_max_abs': errors,
                'max_abs': max(errors.values(), default=0.),
                'scope': 'recorded native RewardPhysics/RewardReference to shared densities; not imagined-state assembly'}
