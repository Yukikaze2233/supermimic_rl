"""Use the shared reward kernel in one native composer without changing upstream files."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import math
from pathlib import Path
from types import MethodType

import torch
from transformer_rl.formal.tasks.reward import PhysicalRewardComposer, RewardPhysics, RewardReference
from ..reward_kernel import (NATIVE_SOURCE_SHA256, JumpPhysicalReference, JumpProgress, RewardExtensions, RewardHistory,
                             RewardParameters, advance_flight_latch, jump_tracking_targets,
                             physical_reward_densities)


class SharedRewardComposer(PhysicalRewardComposer):
    """Keep native event ownership/clock semantics around the differentiable kernel."""

    @torch.no_grad()
    def begin(self, *args, **kwargs):
        PhysicalRewardComposer.begin(self, *args, **kwargs)
        if self.supermimic_reward_extensions.coaxial_action_smoothing:
            window = self._window
            history = RewardHistory.from_actions(
                window['dq'], window['issued'], self._previous, self._previous2, self._action_count,
                coaxial_action_smoothing=True)
            window['action_cost'], window['action2_cost'] = history.action_cost, history.action2_cost

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
            platform = self.supermimic_platform_landing
            platform_bound = platform_target = None
            if platform is not None:
                if platform.last_evidence is None:
                    raise ValueError('Platform reward needs the current observed physics evidence')
                platform_bound = (platform.last_evidence.selected & (tasks.physical_id >= 0)
                                  & (platform.state.request_id == tasks.physical_id))
                capture_target = getattr(platform, 'target_progress', None)
                if capture_target is not None and self.supermimic_reward_extensions.jump_missed:
                    platform_target = capture_target(platform_bound)
            include_peak = (self.supermimic_reward_extensions.jump_peak_completion
                            or self.supermimic_reward_extensions.jump_flight_clearance)
            progress = JumpProgress(window['jump_reference_stage'], tasks.ascent_seen.clone(), latch.closed.clone(),
                                    tasks.landed_s.clone(), latch.lifted.clone(),
                                    tasks.peak_rise.clone() if include_peak else None,
                                    tasks.active_rise.clone() if include_peak else None,
                                    (tasks.physical_id >= 0).clone(), platform_bound, platform_target,
                                    flight_geometry=(None if self.supermimic_jump_geometry is None
                                                     else self.supermimic_jump_geometry(sample)))
        physical_reference = None
        if self.supermimic_jump_physical_reference is not None:
            curve = window.get('jump_tracking_curve')
            if curve is None:
                curve = self.supermimic_jump_tasks.jump_reference()
                window['jump_tracking_curve'] = curve
            physical_reference = self.supermimic_jump_physical_reference.sample(sample, reference, curve)
        densities = physical_reward_densities(
            sample, reference,
            RewardHistory(window['dq'], window['action_cost'], window['action2_cost'], self._action_count),
            RewardParameters.from_composer(self), jump_progress=progress,
            stair_base_net_upward_n=getattr(self, 'supermimic_stair_base_net_upward_n', None),
            jump_physical_reference=physical_reference,
            stair_platform_geometry=(None if self.supermimic_stair_platform_geometry is None
                                     else self.supermimic_stair_platform_geometry(sample)))
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
        exposure = self.supermimic_joint_speed_turn_exposure
        if exposure is not None:
            # These are measurements of the admitted command during this
            # physical tick, not birth draws or a post-reset command snapshot.
            eligible = ordinary & ~gimbal_active & sample.policy_owned
            command = reference.admitted_body_command
            low, high = exposure['initial_forward_m_s']
            gates = dict(gates)
            gates['joint_speed_turn/ordinary_policy_owned'] = eligible
            far_error = errors['forward_m_s'].abs() >= exposure['far_forward_error_m_s']
            for forward_sign, forward_mask in (('positive', command[:, 0] > max(high, 0.)),
                                               ('negative', command[:, 0] < min(low, 0.))):
                for yaw_sign, yaw_mask in (('positive', command[:, 2] > 0),
                                           ('negative', command[:, 2] < 0)):
                    scope = f'joint_speed_turn/forward_{forward_sign}_yaw_{yaw_sign}'
                    mask = eligible & forward_mask & yaw_mask
                    # Always publish every quadrant, including zero exposure.
                    scopes[scope] = (mask, ('forward_m_s', 'yaw_rad_s'))
                    gates[scope + '/far_forward_error'] = mask & far_error
                    gates[scope + '/height_gate'] = torch.where(mask, gates['velocity_height_gate'], 0.)
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


class NativeJumpPhysicalReference:
    """Reward tracking follows the first real flight, without changing commands.

    Each observed latch endpoint is the zero-time initial condition. The
    origin velocity matches RewardPhysics' root-position coordinate; it is
    not whole-robot COM velocity. The ballistic curve is only a tracking
    reference. Peak completion and qualification still use the original
    requested rise and measured request-wide peak.
    """

    def __init__(self, tasks):
        self.tasks = tasks
        self.request = tasks.physical_id.clone()
        self.episode = torch.full_like(self.request, -1)
        self.anchored = torch.zeros_like(self.request, dtype=torch.bool)
        self.start_s = torch.zeros_like(self.request, dtype=torch.float64)
        self.start_z = torch.zeros_like(self.request, dtype=torch.float32)
        self.start_vz = torch.zeros_like(self.start_z)
        self.gravity = float(tasks.reference.config.gravity_m_s2)

    @torch.no_grad()
    def sample(self, sample, reference, curve):
        from transformer_rl.formal.control.observation import CommandMode
        from transformer_rl.formal.tasks.reward import RewardPhase

        tasks, latch = self.tasks, self.tasks.supermimic_flight_latch
        changed = (tasks.physical_id != self.request) | (sample.episode_id != self.episode)
        self.anchored &= ~changed
        self.request.copy_(tasks.physical_id)
        self.episode.copy_(sample.episode_id)
        jump = reference.mode == int(CommandMode.JUMP)
        first = (jump & (tasks.physical_id >= 0) & tasks.ascent_seen & latch.lifted
                 & ~latch.closed & ~self.anchored)
        self.start_s = torch.where(first, sample.timestamp_s, self.start_s)
        self.start_z = torch.where(first, sample.root_world_z_m, self.start_z)
        self.start_vz = torch.where(first, sample.velocity_origin_world_m_s[:, 2], self.start_vz)
        self.anchored |= first
        flight = (jump & self.anchored & tasks.ascent_seen & latch.lifted & ~latch.closed
                  & (reference.phase == int(RewardPhase.FLIGHT)))
        elapsed = (sample.timestamp_s - self.start_s).clamp_min(0).to(self.start_z)
        # The planned release already incorporates extension displacement;
        # sqrt(2*g*requested_rise) would incorrectly demand the entire rise
        # from flight alone when the supported push raised the root first.
        launch_v = curve.launch_velocity_m_s.to(self.start_z)
        release_z = curve.target_apex_world_z_m - launch_v.square() / (2 * self.gravity)
        late = (jump & (tasks.physical_id >= 0) & ~latch.lifted & ~latch.closed
                & (tasks.request_clock.double() >= curve.expected_launch_time_s))
        height = torch.where(late, release_z, reference.jump_height_world_z_m)
        velocity = torch.where(late, launch_v, reference.jump_velocity_world_z_m_s)
        height = torch.where(flight,
            self.start_z + self.start_vz * elapsed - .5 * self.gravity * elapsed.square(), height)
        velocity = torch.where(flight, self.start_vz - self.gravity * elapsed, velocity)
        return JumpPhysicalReference(height, velocity, launch_v, late, flight)

    def receipt(self):
        return {'format': 'native-jump-physical-reference-v1',
                'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'release': 'existing_request_apex_minus_planned_launch_velocity_squared_over_two_g',
                'push_fraction': 'actual_root_origin_vz_over_existing_planned_launch_velocity',
                'first_flight': 'same_episode_request_ascent_first_flight_latch_endpoint',
                'flight_tracking': 'measured_root_origin_z_and_vz_plus_elapsed_gravity_ballistic_reference',
                'initial_elapsed_s': 0., 'gravity_m_s2': self.gravity,
                'qualification': 'original_requested_apex_peak_completion_and_native_success_unchanged',
                'actor_clock': 'original_visible_planned_reference_unchanged',
                'resume': 'cold_physical_episode_reset_repopulates_reward_event_memory',
                'compact_world_model': 'unsupported_without_explicit_event_memory_producer'}


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


def install_stair_support_reference(native) -> dict:
    """Pass the current native endpoint's base net force through its composer.

    The upstream RewardPhysics has scalar contact loads but no force direction.
    Keep this reward-only evidence scoped to the synchronous observe call so a
    later call cannot accidentally consume the preceding physics tick.
    """
    reward = native._formal_reward
    composer, observe = reward.composer, reward.observe
    if not composer.supermimic_reward_extensions.stair_support_transfer:
        raise ValueError('Native stair support requires its explicit reward extension')
    base = native._base_body_index
    composer.supermimic_stair_base_net_upward_n = None

    def observe_with_stair_support(sample, state, **kwargs):
        force = sample.terrain_contact_force_world_n
        if (force.ndim != 3 or force.shape[0] != composer.num_envs
                or force.shape[2] != 3 or base >= force.shape[1]):
            raise ValueError('Native stair support requires full body world-force vectors')
        composer.supermimic_stair_base_net_upward_n = force[:, base, 2].clamp_min(0)
        try:
            return observe(sample, state, **kwargs)
        finally:
            composer.supermimic_stair_base_net_upward_n = None

    reward.observe = observe_with_stair_support
    return {'source': 'NativePhysicalSample.terrain_contact_force_world_n',
            'measure': 'positive_world_z_component_of_base_net_contact_force',
            'timing': 'current_physical_endpoint_during_native_reward_observe',
            'support_threshold_n': 3.}


def install_shared_reward(composer, *, verify_native=False, extensions: RewardExtensions = RewardExtensions(),
                          jump_tasks=None, platform_landing=None, flight_geometry=None,
                          joint_speed_turn_initial_forward_m_s=None, stair_platform_geometry=None):
    """Instance-local installation; the frozen native source stays untouched.

    Extensions add opt-in terms (and may replace the success event); they are
    recorded in the receipt only when active, so native-parity runs keep their
    original source identity. The native audit compares unextended rewards.
    Optional ``platform_landing`` is the already installed physics observer;
    its same-tick request binding extends physical tails onto selected terrain.
    ``joint_speed_turn_initial_forward_m_s`` adds measurement-only exposure
    from the caller's actual curriculum. Native parity audits omit it.
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
        raise ValueError('Request-aware rewards require the native jump task owner')
    if platform_landing is not None and (jump_tasks is None or platform_landing.tasks is not jump_tasks):
        raise ValueError('Platform reward evidence must share the native jump task owner')
    if extensions.needs_flight_geometry != (flight_geometry is not None):
        raise ValueError('Jump geometry producer must match the enabled geometry reward extensions')
    if extensions.stair_platform_height != (stair_platform_geometry is not None):
        raise ValueError('Stair platform geometry producer must match stair_platform_height')
    exposure = None
    if joint_speed_turn_initial_forward_m_s is not None and not verify_native:
        low, high = map(float, joint_speed_turn_initial_forward_m_s)
        if not (math.isfinite(low) and math.isfinite(high) and low <= high):
            raise ValueError('Joint speed/turn exposure requires ordered finite initial forward bounds')
        exposure = {
            'version': 1, 'initial_forward_m_s': [low, high],
            'policy_dt_s': composer.tuning.policy_dt_s,
            'far_forward_error_m_s': 1.,  # Existing forward positive-reward clip boundary.
            'scope': 'ordinary & not special gimbal & physical sample policy_owned',
            'command': 'admitted vx outside initial forward domain and strictly nonzero yaw',
            'timing': 'pre-reset physical substeps; only valid completed intervals enter PPO statistics',
        }
    composer.supermimic_joint_speed_turn_exposure = exposure
    composer.supermimic_jump_geometry = flight_geometry
    composer.supermimic_stair_platform_geometry = stair_platform_geometry
    composer.supermimic_reward_extensions = extensions
    composer.supermimic_jump_tasks = jump_tasks
    composer.supermimic_platform_landing = platform_landing
    if extensions.needs_jump_progress:
        install_flight_latch(jump_tasks)
    composer.supermimic_jump_physical_reference = (
        NativeJumpPhysicalReference(jump_tasks) if extensions.jump_physical_reference else None)
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
    if exposure is not None:
        receipt['joint_speed_turn_exposure'] = dict(exposure)
    if flight_geometry is not None:
        receipt['jump_flight_geometry'] = flight_geometry.receipt
    if stair_platform_geometry is not None:
        receipt['stair_platform_height'] = stair_platform_geometry.receipt
    if composer.supermimic_jump_physical_reference is not None:
        receipt['jump_physical_reference'] = composer.supermimic_jump_physical_reference.receipt()
    if extensions.active:
        receipt["extensions"] = extensions.receipt()
    if extensions.coaxial_action_smoothing:
        from transformer_rl.formal.control.ownership import canonical_coaxial_displacement
        from transformer_rl.formal.contracts import ACTION_ORDER
        from .action_contract import OWNERSHIP_SOURCE_SHA256

        ownership = Path(inspect.getfile(canonical_coaxial_displacement))
        ownership_sha = hashlib.sha256(ownership.read_bytes()).hexdigest()
        if ownership_sha != OWNERSHIP_SOURCE_SHA256:
            raise ValueError('Coaxial action smoothing requires the verified common-winding source')
        order = tuple(composer.binding.native_joint_names[i] for i in composer.binding.policy_joint_indices)
        if order != ACTION_ORDER:
            raise ValueError('Coaxial action smoothing requires the named native policy joint order')
        composer.begin = MethodType(SharedRewardComposer.begin, composer)
        receipt['coaxial_action_smoothing'] = {
            'contract': 'coaxial-action-smoothing-v1',
            'leg_action_scale_rad': .25,
            'pairs': [list(ACTION_ORDER[:2]), list(ACTION_ORDER[2:4])],
            'first_difference': 'nearest common full-turn displacement of successive leg targets',
            'second_difference': 'difference of consecutive common-aligned increments; no second wrap',
            'wheels': 'original nonperiodic speed action differences',
            'history': 'original raw issued actions and native action_count; no new state',
            'ownership_source_sha256': ownership_sha,
        }
    if extensions.jump_tuck_endpoint_normalized:
        receipt['jump_tuck_objective'] = extensions.tuck_objective_receipt()
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
