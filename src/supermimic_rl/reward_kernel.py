"""Shared differentiable 200 Hz reward densities for native and imagined states.

Extracted from the pinned native locomotion reward; stateful identity checking,
metrics and one-shot skill events remain in the native composer. This function
has no simulator, no no_grad decorator, and no mutation of its tensor inputs.
The native adapter and imagination must call this same formula implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from transformer_rl.formal.tasks.reward import (
    CommandMode, RewardPhase, RewardPhysics, RewardReference, RewardTuning,
    FORK_TOLERANCE_M, VELOCITY_HEIGHT_GATE_M, ORDINARY_FORWARD_SQUARE_WEIGHT,
    VELOCITY_SQUARE_WEIGHT, STAND_STILL_WEIGHT, SPIN_CENTERING_WEIGHT,
)
from transformer_rl.formal.tasks.gimbal_course import GimbalCourse

JUMP_PUSH_MIN_REFERENCE_M_S = .1
JUMP_SETTLE_STREAK_S = 2.     # the native physical success needs this uninterrupted stable landing
JUMP_LANDING_RATE_FREE_RAD_S = .4   # inside the native quiet limit of 0.5 rad/s
NATIVE_SOURCE_SHA256 = 'b03282693376a8af082d8433badfcba917270394113aaf2bec271289f7decdd8'


@dataclass(frozen=True)
class RewardExtensions:
    """Opt-in terms beyond the pinned native reward; the defaults keep native parity.

    The native jump objective tracks a time-based reference. A 3-10 cm jump
    flies for about 0.15-0.3 s of its request window, so a policy that never
    leaves the ground keeps most of that reward. Following Fudan's jump policy
    (flight and upward-speed rewards), these terms pay only for actual flight
    in a requested jump; the success event can be raised from the native +1.
    Flight terms alone never reach a policy that has not left the ground, so
    ``jump_push`` (Fudan's takeoff_extend/encourage_jump) pays the achieved
    fraction of the reference's upward speed while supported, only in the
    request's own push window.

    Native FLIGHT only means both wheel loads stayed <= 2 N for 10 ms, which a
    fast leg retraction reaches without rising. Flight terms therefore pay
    only in the request's first real flight: after real ascent and until the
    first touchdown, which bounds them per request; repeated bouncing earns
    nothing. See :class:`JumpProgress` and :func:`advance_flight_latch`.

    The native success needs 2 s of uninterrupted stable landing (contact,
    quiet, near the start height, peak inside the target window). The single
    success event gives a policy that never succeeded no gradient toward it, so
    ``jump_settle`` pays while those exact conditions hold. That happens only
    after a correct jump and lasts at most the 2 s before the request ends.
    It grows with the native streak (``landed_s``) from 25% to 100% at 2 s:
    R8b met every condition on 79% of those ticks but never for 2 s in a row,
    so a per-tick reward alone did not value the unbroken streak. The streaks
    were broken mostly by roll/pitch wobble while balancing after the landing
    (rate > 0.5 rad/s on ~10% of those ticks); ``jump_landing_rate`` costs the
    roll/pitch rate above 0.4 rad/s from the first real touchdown on.
    """

    jump_flight: float = 0.        # per second in the request's first real flight (after real ascent)
    jump_lift: float = 0.          # per second and m/s of upward world velocity under the same condition
    jump_lift_cap_m_s: float = 2.
    jump_push: float = 0.          # per second at full reference upward speed while supported
    success_event: float | None = None
    jump_missed: float = 0.        # per second in reference HOLD without real ascent
    jump_settle: float = 0.        # per second at a 2 s native stable-landing streak (25% at its start)
    jump_landing_rate: float = 0.  # per second and (rad/s)^2 of roll/pitch rate above 0.4 rad/s after a real landing

    def __post_init__(self):
        values = (self.jump_flight, self.jump_lift, self.jump_lift_cap_m_s, self.jump_push, self.jump_missed,
                  self.jump_settle, self.jump_landing_rate,
                  1. if self.success_event is None else self.success_event)
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Reward extensions must be finite and nonnegative")

    @property
    def active(self) -> bool:
        return bool(self.jump_flight or self.jump_lift or self.jump_push or self.jump_missed or self.jump_settle
                    or self.jump_landing_rate or self.success_event is not None)

    @property
    def needs_jump_progress(self) -> bool:
        return bool(self.jump_flight or self.jump_lift or self.jump_push or self.jump_missed or self.jump_settle
                    or self.jump_landing_rate)


@dataclass(frozen=True)
class JumpProgress:
    """Per-row request state the jump extensions need beyond the native phase.

    ``reference_stage`` is the reference clock stage (int64), held across one
    policy interval. ``ascent_seen`` is the native per-request latch of real
    ascent (upward COM speed > 0.15 m/s and rise > 1 cm). ``flight_closed``
    marks that the request's first real flight has touched down (see
    :func:`advance_flight_latch`); producers without that latch pass None and
    then cannot pay flight terms on jump rows. All are read after the task
    owner observed the physical tick.
    """

    reference_stage: Tensor
    ascent_seen: Tensor
    flight_closed: Tensor | None = None
    landed_s: Tensor | None = None       # native uninterrupted stable-landing time [s]; > 0 while stable

    def validate(self, mode: Tensor) -> None:
        flags = (self.ascent_seen,) + (() if self.flight_closed is None else (self.flight_closed,))
        if (self.reference_stage.shape != mode.shape or self.reference_stage.device != mode.device
                or self.reference_stage.dtype != torch.long
                or any(flag.shape != mode.shape or flag.device != mode.device or flag.dtype != torch.bool
                       for flag in flags)
                or self.landed_s is not None and (self.landed_s.shape != mode.shape or self.landed_s.device != mode.device
                                                  or not self.landed_s.is_floating_point())):
            raise ValueError('Jump progress needs matching int64 reference stages, boolean latches and float times')

    @staticmethod
    def cat(items) -> JumpProgress | None:
        items = tuple(items)
        if all(item is None for item in items):
            return None
        if (any(item is None for item in items) or len({item.flight_closed is None for item in items}) != 1
                or len({item.landed_s is None for item in items}) != 1):
            raise ValueError('Jump progress coverage changes within a physical interval')
        optional = lambda name: (None if getattr(items[0], name) is None
                                 else torch.cat([getattr(item, name) for item in items]))
        return JumpProgress(torch.cat([item.reference_stage for item in items]),
                            torch.cat([item.ascent_seen for item in items]),
                            optional('flight_closed'), optional('landed_s'))


def advance_flight_latch(request: Tensor, lifted: Tensor, closed: Tensor, *, current: Tensor,
                         ascent_seen: Tensor, air_ticks: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """One physical tick of a request's first-real-flight latch, after the task owner observed it.

    The latch belongs to ``request``; a different ``current`` physical request
    clears it. ``lifted``: airborne for the native two ticks with real ascent
    (wheel unloading alone sets native flight_seen). ``closed``: touched down
    after that liftoff. Returns the new (request, lifted, closed).
    """
    changed = current != request
    lifted = (lifted & ~changed) | ((current >= 0) & ascent_seen & (air_ticks >= 2))
    closed = (closed & ~changed) | (lifted & (air_ticks == 0))
    return current.clone(), lifted, closed


@dataclass(frozen=True)
class RewardParameters:
    tuning: RewardTuning
    physics_dt_s: float
    mass_kg: Tensor
    wheel_indices: list[int]
    harmful_indices: list[int]
    base_index: int
    extensions: RewardExtensions = RewardExtensions()

    @classmethod
    def from_composer(cls, composer):
        return cls(composer.tuning, composer.physics_dt_s, composer._mass_kg,
                   composer._wheels, composer._harmful,
                   composer.binding.native_body_names.index("base_link"),
                   getattr(composer, "supermimic_reward_extensions", RewardExtensions()))


@dataclass(frozen=True)
class RewardHistory:
    velocity_rad_s: Tensor
    action_cost: Tensor
    action2_cost: Tensor
    action_count: Tensor

    @classmethod
    def from_actions(cls, velocity, action, previous, previous2, action_count):
        first = (action - previous).square()
        second = (action - 2 * previous + previous2).square()
        first = .0025 * first[:, :4].sum(-1) + .01 * first[:, 4:].sum(-1)
        second = .0125 * second[:, :4].sum(-1) + .01 * second[:, 4:].sum(-1)
        return cls(velocity, torch.where(action_count > 0, first, 0.),
                   torch.where(action_count > 1, second, 0.), action_count)


@dataclass(frozen=True)
class RewardDensities:
    positive: dict[str, Tensor]
    cost: dict[str, Tensor]
    components: dict[str, Tensor]
    gates: dict[str, Tensor]
    errors: dict[str, Tensor]
    failure_scale: Tensor
    invalid: Tensor

    @property
    def total(self):
        return torch.stack(tuple(self.components.values())).sum(0)


def gimbal_tracking_densities(reference, velocity_world):
    """Native gimbal objective with a defined derivative at zero horizontal speed.

    Heading has no physical meaning at rest. Select a constant direction before
    atan2 there; multiplying atan2(0, 0) by a zero gate still yields NaN gradients.
    The heading gate already removes that direction from all paid terms.
    """
    yaw = reference.yaw_world_rad
    vx, vy = velocity_world[:, 0], velocity_world[:, 1]
    velocity = torch.stack((yaw.cos() * vx + yaw.sin() * vy,
                            -yaw.sin() * vx + yaw.cos() * vy), -1)
    speed = velocity.norm(dim=-1)
    target = reference.speed_m_s[:, None] * torch.stack(
        (reference.direction_rad.cos(), reference.direction_rad.sin()), -1)
    error2 = (velocity - target).square().sum(-1)
    heading_velocity = torch.where((speed > 0)[:, None], velocity, velocity.new_tensor([1., 0.]))
    heading_error = GimbalCourse._wrap(
        torch.atan2(heading_velocity[:, 1], heading_velocity[:, 0]) - reference.direction_rad)
    heading_valid = (reference.speed_m_s > .1) & (speed > 0.)
    active = reference.active
    positive = {
        'gimbal_velocity': active * torch.exp(-error2 / .25),
        'gimbal_speed': active * torch.exp(-(speed - reference.speed_m_s).square() / .25),
        'gimbal_heading': 5. * active * heading_valid * torch.exp(-heading_error.square() / .025),
    }
    cost = {
        'gimbal_velocity': .2 * active * .5 ** 2 * error2,
        'gimbal_heading': .2 * active * heading_valid * (4. * heading_error).square(),
        'gimbal_still': active * (reference.speed_m_s < .05) * velocity.abs().sum(-1),
    }
    errors = {'gimbal_velocity_m_s': torch.where(active, error2.sqrt(), 0.),
              'gimbal_heading_rad': torch.where(active & heading_valid, heading_error, 0.)}
    return positive, cost, errors


def physical_reward_densities(sample: RewardPhysics, reference: RewardReference,
                             history: RewardHistory, parameters: RewardParameters, *,
                             jump_progress: JumpProgress | None = None) -> RewardDensities:
    """Pure densities; integrate with the physical dt exactly once outside."""
    invalid = torch.zeros_like(reference.mode, dtype=torch.bool)
    failure_scale = torch.ones_like(sample.height_above_ground_m)
    jump = reference.mode == int(CommandMode.JUMP)
    recovery = reference.mode == int(CommandMode.RECOVERY)
    ordinary = ~jump & ~recovery
    gimbal_active = (torch.zeros_like(ordinary) if reference.gimbal is None
                     else reference.gimbal.active)
    if gimbal_active.shape != ordinary.shape or gimbal_active.dtype != torch.bool:
        raise ValueError('gimbal reward requires a boolean row mask')
    invalid |= gimbal_active & ~ordinary
    terrain_active = torch.zeros_like(ordinary)
    terrain_airborne = torch.zeros_like(ordinary)
    terrain_landing = torch.zeros_like(ordinary)
    terrain_extension = torch.zeros_like(sample.height_above_ground_m)
    if reference.terrain is not None:
        terrain_active = reference.terrain.active
        terrain_airborne = reference.terrain.airborne
        terrain_landing = reference.terrain.landing
        terrain_extension = reference.terrain.body_extension_m
    rough_dash = parameters.tuning.profile == 'rough_dash'
    rough_active = (reference.rough_dash.active if rough_dash else torch.zeros_like(ordinary))
    if rough_dash:
        # The active physical window includes flat landing births. It is
        # separate from terrain height-reference eligibility.
        terrain_airborne = rough_active
    phase = reference.phase
    invalid |= ((jump & ((phase < int(RewardPhase.TAKEOFF)) | (phase > int(RewardPhase.SETTLING))))
                | (recovery & (phase != int(RewardPhase.RECOVERY)))
                | (ordinary & (phase != int(RewardPhase.SUPPORT))))
    velocity, command = sample.velocity_origin_body_m_s, reference.admitted_body_command
    # Heading frame from the measured up axis u = -g_body: forward f is
    # body x projected normal to u, left = u x f. Velocities in m/s, yaw
    # rate about the world vertical in rad/s; pitching cannot leak v_up.
    up = -sample.gravity_body
    forward = torch.zeros_like(up)
    forward[:, 0] = 1
    forward = forward - (forward * up).sum(-1, keepdim=True) * up
    forward = forward / forward.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    left = torch.linalg.cross(up, forward)
    v_forward, v_left = (velocity * forward).sum(-1), (velocity * left).sum(-1)
    v_up = (velocity * up).sum(-1)
    yaw_rate = (sample.angular_velocity_body_rad_s * up).sum(-1)
    # Reward and actor share the admitted target, including small commands.
    target_v, target_w = command[:, 0], command[:, 2]
    ev, ew = v_forward - target_v, yaw_rate - target_w
    support_eh = sample.height_above_ground_m - command[:, 3]
    terrain_eh = terrain_extension - command[:, 3]
    ordinary_eh = torch.where(terrain_active, terrain_eh, support_eh)
    if rough_dash:
        # SCUT uses a small base ray grid on ground and wheel extension in
        # flight. Keep our terrain extension datum to avoid a root-ray
        # jump at sharp steps; flat ground still uses body clearance.
        # ToF owns visible height changes. No hidden +0.12/.36 target.
        ordinary_eh = torch.where(rough_active,
            reference.rough_dash.body_extension_m - command[:, 3], ordinary_eh)
    full, zero = VELOCITY_HEIGHT_GATE_M
    velocity_gate = torch.where((reference.mode == int(CommandMode.NORMAL)) & ~terrain_active,
                                ((zero - support_eh.abs()) / (zero - full)).clamp(0, 1), 1.)
    if rough_dash:
        velocity_gate = torch.where(ordinary,
            ((zero - ordinary_eh.abs()) / (zero - full)).clamp(0, 1), velocity_gate)
    jump_eh = sample.root_world_z_m - reference.jump_height_world_z_m
    eh = torch.where(jump, jump_eh, ordinary_eh)
    evz = sample.velocity_origin_world_m_s[:, 2] - reference.jump_velocity_world_z_m_s
    pose = sample.roll_pitch_rad - reference.target_roll_pitch_rad
    upright = (-sample.gravity_body[:, 2]).clamp(-1, 1)
    height_gate = torch.where(recovery, upright.clamp_min(0), torch.ones_like(upright))
    # Ordinary forward tracking uses g [exp(-clip(e,1)^2/0.5) - (0.25 e)^2]
    # per second: only its square weight rises from 0.1 to 1.0. Jump retains
    # its existing square weight of 0.1 and positive weight of 0.8.
    # Yaw remains g [exp(-clip(e,0.8)^2/0.25) - 0.1 (0.5 e)^2]; g is the
    # velocity height gate. Height remains exp(-clip(e_h,0.15)^2/0.001) - (10 e_h)^2.
    locomotion = torch.where(ordinary, 1., torch.where(jump, .8, 0.))
    forward_square_weight = torch.where(ordinary, ORDINARY_FORWARD_SQUARE_WEIGHT, VELOCITY_SQUARE_WEIGHT)
    gx, gy = sample.gravity_body[:, 0], sample.gravity_body[:, 1]
    # Densities per second. Ordinary <= 1+1+1+2 = 5; jump <= 0.8+1+1+1+1;
    # recovery <= 1+4; all below the composer's 8/s bound.
    positive = {'forward': locomotion * velocity_gate * torch.exp(-ev.clamp(-1., 1.).square() / .5),
                'yaw': (~recovery) * velocity_gate * torch.exp(-ew.clamp(-.8, .8).square() / .25),
                'height': height_gate * torch.exp(-eh.clamp(-.15, .15).square() / .001),
                # SCUT flat_orientation_{y,x}_exp on projected gravity.
                'attitude': ordinary * (torch.exp(-gx.square() / .02) + torch.exp(-gy.square() / .01)),
                'jump_vertical': jump * torch.exp(-(evz / .35).square()),
                'jump_attitude': jump * torch.exp(-(pose / .25).square().sum(-1)),
                'recovery_upright': recovery * (2 * (1 + upright))}
    h = lambda value: torch.hypot(torch.ones_like(value), value) - 1
    effort, dq = sample.active_explicit_effort_nm, sample.policy_velocity_rad_s
    ddq = (dq - history.velocity_rad_s) / parameters.physics_dt_s
    exact_still = ordinary & ~gimbal_active & command[:, :3].eq(0).all(-1)
    still = ordinary & ~gimbal_active & (target_v == 0)
    strict_still = still & (target_w == 0)
    spin_centering = still & (target_w != 0)
    standing_weight = STAND_STILL_WEIGHT * strict_still + SPIN_CENTERING_WEIGHT * spin_centering
    horizontal = torch.stack((v_forward, v_left), -1)
    # SCUT flat_orientation_{y,x}_v: 2[(2 exp(-v^2/3) + 2) g]^2 = 32 c(v)^2 g^2
    # with c(v) = (exp(-v^2/3) + 1)/2, 1 at rest and 0.54 at 2.7 m/s.
    relief = .5 * (torch.exp(-command[:, 0].square() / 3) + 1)
    fork = (torch.zeros_like(v_forward) if sample.wheel_position_base_m is None
            else sample.wheel_position_base_m[:, 0, 0] - sample.wheel_position_base_m[:, 1, 0])
    wheel_load = sample.body_normal_load_n[:, parameters.wheel_indices]
    slip = wheel_load > 5
    landing = (jump & ((phase == int(RewardPhase.LANDING)) | (phase == int(RewardPhase.SETTLING)))) | terrain_landing
    flight = (jump & (phase == int(RewardPhase.FLIGHT))) | terrain_airborne
    if rough_dash:
        # The profile below replaces the legacy terrain-flight effort and
        # normalized landing heuristics; explicit JUMP keeps its objective.
        flight = jump & (phase == int(RewardPhase.FLIGHT))
        landing = jump & ((phase == int(RewardPhase.LANDING)) | (phase == int(RewardPhase.SETTLING)))
    weight_n = parameters.mass_kg * parameters.tuning.gravity_m_s2
    excess_landing = (wheel_load.sum(-1) / weight_n - parameters.tuning.landing_load_multiple).clamp_min(0)
    # SCUT published rough_dash releases height-square and pitch penalties
    # while airborne, retains roll and doubles vertical-speed damping. gx
    # is the pitch component of projected gravity; gy is the roll component.
    ordinary_attitude = 32 * relief.square() * (
        torch.where(terrain_airborne, 0., gx.square()) + gy.square())
    cost = {'forward': (~recovery) * velocity_gate * forward_square_weight * (.25 * ev).square(),
            'yaw': (~recovery) * velocity_gate * VELOCITY_SQUARE_WEIGHT * (.5 * ew).square(),
            'height': height_gate * (10 * eh).square() * (~terrain_airborne),
            'attitude': torch.where(recovery, 2 * h(pose / .8).sum(-1),
                                     torch.where(jump, .25 * h(pose / .5).sum(-1),
                                                 ordinary_attitude)),
            # SCUT ang_vel_xy 0.05; recovery flips are ungated, jump uses its own gain.
            'roll_pitch_angular_velocity': torch.where(
                recovery, parameters.tuning.angular_damping_recovery,
                torch.where(jump, parameters.tuning.angular_damping_jump,
                            parameters.tuning.angular_damping_ordinary))
                * sample.angular_velocity_body_rad_s[:, :2].square().sum(-1),
            # Penalize drift only for explicit zero-translation commands.
            'stand_world_xy': standing_weight * horizontal.abs().sum(-1),
            'ordinary_vertical': ordinary * .5 * v_up.square() * torch.where(terrain_airborne, 2., 1.),
            'jump_vertical': jump * h(evz / .5),
            # SCUT no_fork + no_fork_square: 1(|dx| > 0.05 m) + (5 dx)^2, dx = x_L - x_R.
            'leg_fork': (~recovery) * ((fork.abs() > FORK_TOLERANCE_M).float() + (5 * fork).square()),
            'effort': 1e-4 * effort.square().sum(-1),
            # SCUT wheel_power: positive wheel mechanical power only, W.
            'mechanical_power_proxy': 1e-4 * (effort[:, 4:] * dq[:, 4:]).clamp_min(0).sum(-1),
            'wheel_velocity': 1e-5 * dq[:, 4:].square().sum(-1),
            'wheel_acceleration': 1e-8 * ddq[:, 4:].square().sum(-1),
            'leg_velocity': 5e-3 * dq[:, :4].square().sum(-1),
            'leg_acceleration': 5e-7 * ddq[:, :4].square().sum(-1),
            'issued_first_difference': history.action_cost, 'issued_second_difference': history.action2_cost,
            # SCUT undesired_contact: any non-wheel body above 3 N.
            'harmful_contact': 2 * (sample.body_normal_load_n[:, parameters.harmful_indices] > 3).any(-1).float(),
            'contact_slip': .5 * (slip * (sample.wheel_contact_tangent_speed_m_s / .2).square()).sum(-1),
            'joint_boundary': 2 * (sample.joint_boundary_shortfall_rad / .05).square().sum(-1),
            'flight_wheel_effort': .2 * flight * (effort[:, 4:] / 4.5).square().sum(-1),
            'landing_impact': .5 * landing * h(excess_landing)}
    if reference.intentional_base_contact is not None and not rough_dash:
        base = parameters.base_index
        # Explicitly permitted terrain base contact exempts only the base.
        others = [index for index in parameters.harmful_indices if index != base]
        cost['harmful_contact'] = torch.where(
            reference.intentional_base_contact,
            2 * (sample.body_normal_load_n[:, others] > 3).any(-1).float(), cost['harmful_contact'])
    if rough_dash:
        # Active additions in the published rough_dash 2026-07-19 recipe.
        # Values are densities, integrated once with the actual physics dt.
        air = reference.rough_dash
        wheel_timer = air.wheel_contact_time_s.amax(-1)
        before_contact = air.active & (wheel_timer < .02 - 1e-6)
        after_contact = air.active & (wheel_timer >= .02 - 1e-6)
        centering = before_contact & (air.base_contact_time_s < .02 - 1e-6)
        wheel_x = (sample.wheel_position_base_m * forward[:, None]).sum(-1)
        wheel_z = (sample.wheel_position_base_m * up[:, None]).sum(-1)
        centering &= (wheel_z < -.1).all(-1)
        positive['airborne_wheel_centering'] = 10 * centering * torch.exp(-wheel_x.square() / .02).mean(-1)
        max_torque = effort[:, 4:].abs().amax(-1)
        positive['airborne_zero_wheel_torque'] = 20 * before_contact * torch.exp(-(max_torque / 1.5).square())
        # Both physical output axes are signed forward by the native
        # producer. Using raw mirrored wheel dq would reward one side
        # backwards. This term follows body-x commands, not cloud motion.
        directional = (((target_v > 1) & (velocity[:, 0] > 1))
                       | ((target_v < -1) & (velocity[:, 0] < -1))) & ~gimbal_active
        direction = target_v.sign()
        wheel_score = (sample.wheel_forward_velocity_rad_s * direction[:, None] / 10).clamp(0, 1).amin(-1)
        positive['airborne_wheel_directional_speed'] = 10 * before_contact * directional * wheel_score
        shortfall_window = air.active & (wheel_timer > 0) & (wheel_timer < .05 - 1e-6)
        cost['airborne_wheel_speed_shortfall'] = 10 * shortfall_window * directional * (1 - wheel_score)
        cost['airborne_landing_down_velocity'] = 2 * after_contact * sample.velocity_origin_world_m_s[:, 2].clamp_max(0).square()
        force = sample.wheel_contact_force_n
        cost['airborne_landing_contact_force'] = (10 * after_contact * (force > 1).any(-1)
            * ((force.amax(-1) - 200) / 200).clamp(0, 1))
        limits = sample.passive_knee_limits_rad
        span = limits[:, 1] - limits[:, 0]
        lower, upper = limits[:, 0] + .10 * span, limits[:, 1] - .05 * span
        q, qd = sample.passive_knee_position_rad, sample.passive_knee_velocity_rad_s
        violation = (lower - q).clamp_min(0) + (q - upper).clamp_min(0)
        outward = ((q < lower) & (qd < -3 * math.pi)) | ((q > upper) & (qd > 3 * math.pi))
        cost['airborne_knee_position'] = 10 * air.active * violation.sum(-1)
        cost['airborne_knee_velocity'] = 100 * air.active * outward.float().sum(-1)
        # The selected-asset knee terms replace only the knee part of the
        # generic boundary cost; output-admission corrections remain paid.
        admission_cost = 2 * (sample.joint_boundary_shortfall_rad[:, 2:] / .05).square().sum(-1)
        cost['joint_boundary'] = torch.where(air.active, admission_cost, cost['joint_boundary'])
        cost['harmful_contact'] = cost['harmful_contact'] * torch.where(air.active, 25., 1.)
        failure_scale = torch.where(air.active, 6., 1.)
    gimbal_errors = {}
    if reference.gimbal is not None:
        for key in ('yaw_world_rad', 'speed_m_s', 'direction_rad'):
            value = getattr(reference.gimbal, key)
            if value.shape != ordinary.shape or value.device != ordinary.device:
                raise ValueError('gimbal reward targets require matching row shapes/devices')
            invalid |= ~torch.isfinite(value)
        gp, gc, gimbal_errors = gimbal_tracking_densities(
            reference.gimbal, sample.velocity_origin_world_m_s)
        # Body-frame zero vx/vy is a placeholder for this task, not a
        # request to stop translating. Do not charge both objectives.
        positive['forward'] = positive['forward'] * ~gimbal_active
        cost['forward'] = cost['forward'] * ~gimbal_active
        cost['stand_world_xy'] = cost['stand_world_xy'] * ~gimbal_active
        positive.update(gp)
        cost.update(gc)
    if sample.available_terms is not None:
        available = set(sample.available_terms)
        positive = {key: value if key in available else torch.zeros_like(value)
                    for key, value in positive.items()}
        cost = {key: value if key in available else torch.zeros_like(value)
                for key, value in cost.items()}
    extensions = parameters.extensions
    missed_jump = torch.zeros_like(jump)
    stage = ascended = closed = landed = None
    if extensions.needs_jump_progress:
        if jump_progress is None:
            if jump.any():
                raise ValueError('Jump flight/missed rewards require the request progress (reference stage, ascent)')
        else:
            jump_progress.validate(reference.mode)
            stage, ascended, closed, landed = (jump_progress.reference_stage, jump_progress.ascent_seen,
                                               jump_progress.flight_closed, jump_progress.landed_s)
    if extensions.jump_missed:
        from transformer_rl.formal.tasks.jump import ReferenceStage

        if stage is not None:
            # Only the explicit end-of-reference HOLD certifies a missed push;
            # a zero velocity target also occurs at compression start and apex.
            # Wheel unloading without rising (native flight_seen) is still missed.
            missed_jump = jump & (stage == int(ReferenceStage.HOLD)) & ~ascended
        # A missed request must not keep earning locomotion tracking rewards.
        # Keep jump_attitude to encourage upright support while waiting,
        # instead of turning the entire HOLD interval into pure punishment.
        for name in ('forward', 'yaw', 'height', 'jump_vertical'):
            positive[name] = torch.where(missed_jump, 0., positive[name])
        cost['jump_missed'] = extensions.jump_missed * missed_jump.to(v_up.dtype)
    if extensions.jump_flight or extensions.jump_lift:
        # As SCUT's takeoff state machine, the flight window opens at the
        # physical takeoff event (not the reference clock), so late real
        # jumps still count, and closes at the first touchdown of the request.
        flying = torch.zeros_like(jump)
        if stage is not None:
            if closed is None:
                if jump.any():
                    raise ValueError('Jump flight rewards require the first-flight latch of the request')
            else:
                flying = jump & (phase == int(RewardPhase.FLIGHT)) & ascended & ~closed
        flying = flying.to(v_up.dtype)
        rise = sample.velocity_origin_world_m_s[:, 2].clamp(0, extensions.jump_lift_cap_m_s)
        positive['jump_flight'] = extensions.jump_flight * flying
        positive['jump_lift'] = extensions.jump_lift * flying * rise
    if extensions.jump_push:
        # Supported (not airborne) while the reference demands upward speed,
        # i.e. in its extension/ballistic push. Native TAKEOFF alone is not
        # enough: early wheel unloading moves the native phase to LANDING.
        target = reference.jump_velocity_world_z_m_s
        pushing = (jump & (phase != int(RewardPhase.FLIGHT)) & (target > JUMP_PUSH_MIN_REFERENCE_M_S))
        if closed is None:
            if jump.any():
                raise ValueError('Jump push rewards require the first-flight latch of the request')
        else:
            pushing = pushing & ~closed   # no second push after the first real flight landed
        achieved = (sample.velocity_origin_world_m_s[:, 2] / target.clamp_min(JUMP_PUSH_MIN_REFERENCE_M_S)).clamp(0, 1)
        positive['jump_push'] = extensions.jump_push * pushing.to(v_up.dtype) * achieved
    if extensions.jump_settle:
        settled = torch.zeros_like(v_up)
        if stage is not None:
            if landed is None:
                if jump.any():
                    raise ValueError('Jump settle rewards require the native stable-landing time')
            else:
                streak = (landed.to(v_up.dtype) / JUMP_SETTLE_STREAK_S).clamp(0, 1)
                settled = torch.where(jump & (landed > 0), .25 + .75 * streak, 0.)
        positive['jump_settle'] = extensions.jump_settle * settled
    if extensions.jump_landing_rate:
        landed_after_flight = torch.zeros_like(jump)
        if stage is not None:
            if closed is None:
                if jump.any():
                    raise ValueError('Jump landing-rate costs require the first-flight latch of the request')
            else:
                landed_after_flight = jump & closed & (phase != int(RewardPhase.FLIGHT))
        excess = (sample.angular_velocity_body_rad_s[:, :2].norm(dim=-1) - JUMP_LANDING_RATE_FREE_RAD_S).clamp_min(0)
        cost['jump_landing_rate'] = extensions.jump_landing_rate * landed_after_flight.to(v_up.dtype) * excess.square()
    p, c = torch.stack(tuple(positive.values())).sum(0), torch.stack(tuple(cost.values())).sum(0)
    components = {**{'positive.' + key: value for key, value in positive.items()},
                  **{'cost.' + key: -value for key, value in cost.items()}}
    positive_bound = torch.where(gimbal_active, 11., 8.)
    positive_bound = (positive_bound + extensions.jump_flight + extensions.jump_lift * extensions.jump_lift_cap_m_s
                      + extensions.jump_push + extensions.jump_settle)
    if rough_dash:
        # 10 centering + 20 zero torque + 10 directional wheel speed.
        positive_bound = positive_bound + 40 * rough_active
    invalid |= (~torch.isfinite(p) | ~torch.isfinite(c) | (p < 0) | (p > positive_bound + 1e-5) | (c < 0)
                | ~torch.isfinite(torch.stack(tuple(components.values()))).all(0))
    # Keep legacy metric keys; their gates now follow exact admitted zeros.
    gates = {'ordinary': ordinary, 'jump': jump, 'recovery': recovery, 'exact_stationary': exact_still,
             'gimbal_spin_translate': gimbal_active,
             'ordinary_horizontal_tracking': ordinary & ~gimbal_active,
             'stationary_deadband': still,
             'strict_stationary_deadband': strict_still, 'spin_centering': spin_centering,
             'slip_left': slip[:, 0], 'slip_right': slip[:, 1], 'landing': landing,
             'terrain_active': terrain_active, 'terrain_airborne': terrain_airborne,
             'terrain_landing': terrain_landing,
             'height_reference_weight': height_gate, 'velocity_height_gate': velocity_gate,
             'first_action_difference': history.action_count > 0, 'second_action_difference': history.action_count > 1}
    if extensions.jump_missed:
        gates['jump_missed'] = missed_jump
    errors = {'forward_m_s': ev, 'yaw_rad_s': ew, 'support_height_m': support_eh,
              'terrain_body_extension_m': torch.where(terrain_active, terrain_extension, 0.),
              'terrain_height_error_m': torch.where(terrain_active, terrain_eh, 0.),
              'jump_world_height_m': jump_eh, 'jump_world_vertical_m_s': evz,
              'roll_rad': pose[:, 0], 'pitch_rad': pose[:, 1],
              'stationary_world_xy_speed_m_s': horizontal.norm(dim=-1), 'leg_fork_m': fork}
    errors.update(gimbal_errors)
    return RewardDensities(positive, cost, components, gates, errors, failure_scale, invalid)
