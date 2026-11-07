"""Shared differentiable 200 Hz reward densities for native and imagined states.

Extracted from the pinned native locomotion reward; stateful identity checking,
metrics and one-shot skill events remain in the native composer. This function
has no simulator, no no_grad decorator, and no mutation of its tensor inputs.
The native adapter and imagination must call this same formula implementation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from transformer_rl.formal.tasks.reward import (
    CommandMode, RewardPhase, RewardPhysics, RewardReference, RewardTuning,
    FORK_TOLERANCE_M, VELOCITY_HEIGHT_GATE_M, ORDINARY_FORWARD_SQUARE_WEIGHT,
    VELOCITY_SQUARE_WEIGHT, STAND_STILL_WEIGHT, SPIN_CENTERING_WEIGHT,
)
from transformer_rl.formal.tasks.gimbal_course import GimbalCourse

if TYPE_CHECKING:
    from .course.platform_target_reward import PlatformTargetProgress

JUMP_PUSH_MIN_REFERENCE_M_S = .1
JUMP_TAKEOFF_RISE_M_S = .15    # Fudan takeoff_extend: supported and rising faster than this
JUMP_SETTLE_STREAK_S = 2.     # the native physical success needs this uninterrupted stable landing
JUMP_LANDING_RATE_FREE_RAD_S = .4   # inside the native quiet limit of 0.5 rad/s
# Pinned LabCourseTasks.observe_physics uses min(0.03 m, requested rise * 0.5).
JUMP_PEAK_TOLERANCE_MAX_M = .03
JUMP_PEAK_TOLERANCE_TARGET_FRACTION = .5
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

    Once the reference reaches HOLD, ``jump_missed`` removes tracking profit
    until real liftoff. A supported rise alone does not complete takeoff.
    Compression and extension retain their shaping rewards, and a late real
    jump can still earn flight and landing rewards within the request window.

    ``jump_peak_completion`` retains that missed-jump signal after a small
    first flight. From first touchdown or HOLD, tracking rewards are scaled
    by attained peak / the native acceptable lower peak, and the existing
    missed-jump weight costs the remaining fraction. The landing reference
    stays at the baseline. On its own this only changes JUMP-mode rewards.

    ``jump_physical_tail`` retains peak completion, missed and settling terms
    after the protocol releases to NORMAL while the same physical request
    remains alive. Terrain and special gimbal tasks keep their own rewards.
    It does not prolong the request or add takeoff/flight terms to NORMAL.

    ``jump_extension_height_floor`` treats the clock's EXTENSION height as a
    lower bound until the first real touchdown. Height above that reference
    does not ask the body to compress again while vertical tracking asks it
    to rise. Compression, later clock stages and physical success stay intact.

    The native success needs 2 s of uninterrupted stable landing (contact,
    quiet, near the start height, peak inside the target window). The single
    success event gives a policy that never succeeded no gradient toward it, so
    ``jump_settle`` pays while those exact conditions hold. That happens only
    after a correct jump and lasts at most the 2 s before the request ends.
    It grows with the native streak (``landed_s``) from 25% to 100% at 2 s:
    R8b met every condition on 79% of those ticks but never for 2 s in a row,
    so a per-tick reward alone did not value the unbroken streak. The streaks
    were broken by angular wobble while balancing after the landing; costing
    roll/pitch alone moved the wobble into yaw. ``jump_landing_rate`` costs the
    same quantity the native quiet test bounds by 0.5 rad/s, the body angular
    speed, above 0.4 rad/s from the first real touchdown on.

    ``recovery_quiet`` rewards low three-axis angular speed once recovery has
    attained its existing height/upright objective. It reuses that objective's
    smooth weight and the yaw tracking kernel's 0.25 (rad/s)^2 scale. This
    supplies a settling signal without damping the whole flip or changing the
    recovery handoff. It is not physical success or contact certification.

    ``stationary_quiet`` rewards low angular speed and low horizontal motion
    after an explicit NORMAL zero command, with both wheels supported and no
    harmful body contact. Its translation factor reuses the existing stand
    drift cost; it introduces no additional speed threshold or gain.
    Terrain/flight windows, special gimbal commands and still-live physical
    jump requests are excluded. It adds no mode lock or motion restriction.

    ``ordinary_leg_acceleration`` overrides the physical acceleration cost
    only for normal, slope and stair locomotion. None retains the native
    5e-7 coefficient; jump and recovery always retain their native cost.

    ``jump_flight_clearance`` spends the existing flight reward on continuous
    common wheel clearance instead of a binary contact event. Requested root
    rise normalizes distance; it is not a new clearance success threshold.
    ``jump_tuck`` rewards shortening each virtual leg toward the source's
    closed-chain minimum, normalized by that same source's reachable span.
    Both use the first real flight window, never the supported push. Geometry
    remains reward-only and does not change observations or physical success.
    ``jump_takeoff_extension`` is the supported counterpart (Fudan's
    takeoff_extend): inside ``jump_push``'s window, while the base still rises
    faster than 0.15 m/s, it pays the mean normalized virtual-leg extension
    toward the same source's maximum, so the push uses the full leg stroke.

    ``stair_support_transfer`` keeps velocity tracking available during a
    permitted ordinary STAIR request. Once the base has upward net contact
    force, its extension target is an upper bound, allowing the legs to lift
    onto the step. Only that supported base is exempt from harmful contact;
    other bodies retain their costs. Net world-Z force is separate evidence:
    a scalar contact norm cannot distinguish support from a riser impact.
    """

    jump_flight: float = 0.        # per second in the request's first real flight (after real ascent)
    jump_lift: float = 0.          # per second and m/s of upward world velocity under the same condition
    jump_lift_cap_m_s: float = 2.
    jump_push: float = 0.          # per second at full reference upward speed while supported
    success_event: float | None = None
    jump_missed: float = 0.        # per second in reference HOLD without real liftoff
    jump_settle: float = 0.        # per second at a 2 s native stable-landing streak (25% at its start)
    jump_landing_rate: float = 0.  # per second and (rad/s)^2 of body angular speed above 0.4 rad/s after a real landing
    recovery_quiet: float = 0.    # per second at quiet, upright target height; default retains native parity
    ordinary_leg_acceleration: float | None = None  # per second and (rad/s^2)^2, summed over four legs
    jump_peak_completion: bool = False  # reuse missed-jump weight with continuous requested-peak completion
    stationary_quiet: float = 0.  # per second at upright target height and quiet supported NORMAL standstill
    jump_physical_tail: bool = False  # retain existing completion/landing terms until physical request ends
    jump_extension_height_floor: bool = False  # release only above-reference height in pre-touchdown EXTENSION
    jump_flight_clearance: bool = False  # make the existing flight budget differentiable in wheel clearance
    jump_tuck: float = 0.  # per second at the source's minimum virtual leg length, only in real flight
    stair_support_transfer: bool = False  # permit supported ordinary STAIR leg retraction
    jump_takeoff_extension: float = 0.  # per second at the source's maximum virtual leg length while pushing up

    # Opt-ins added after the first extension receipts. Their disabled values
    # are omitted, so sources recorded before each opt-in keep their identity.
    _RECEIPT_OPT_INS = ('jump_flight_clearance', 'jump_tuck', 'jump_peak_completion', 'stationary_quiet',
                        'jump_physical_tail', 'jump_extension_height_floor', 'stair_support_transfer',
                        'jump_takeoff_extension')

    def __post_init__(self):
        if type(self.stair_support_transfer) is not bool:
            raise ValueError('Stair support transfer requires a boolean flag')
        if (type(self.jump_flight_clearance) is not bool
                or self.jump_flight_clearance and self.jump_flight <= 0):
            raise ValueError('Jump flight clearance requires a boolean flag and positive jump_flight weight')
        if type(self.jump_extension_height_floor) is not bool:
            raise ValueError('Jump extension height floor requires a boolean flag')
        if type(self.jump_peak_completion) is not bool or self.jump_peak_completion and self.jump_missed <= 0:
            raise ValueError('Jump peak completion requires a boolean flag and positive jump_missed weight')
        if type(self.jump_physical_tail) is not bool or self.jump_physical_tail and not self.jump_peak_completion:
            raise ValueError('Jump physical tail requires a boolean flag and enabled jump_peak_completion')
        values = (self.jump_flight, self.jump_lift, self.jump_lift_cap_m_s, self.jump_push, self.jump_missed,
                  self.jump_settle, self.jump_landing_rate, self.recovery_quiet, self.stationary_quiet, self.jump_tuck,
                  self.jump_takeoff_extension,
                  0. if self.ordinary_leg_acceleration is None else self.ordinary_leg_acceleration,
                  1. if self.success_event is None else self.success_event)
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Reward extensions must be finite and nonnegative")

    @property
    def active(self) -> bool:
        return bool(self.jump_flight or self.jump_lift or self.jump_push or self.jump_missed or self.jump_settle
                    or self.jump_landing_rate or self.recovery_quiet or self.success_event is not None
                    or self.ordinary_leg_acceleration is not None or self.stationary_quiet
                    or self.jump_extension_height_floor or self.jump_tuck or self.stair_support_transfer
                    or self.jump_takeoff_extension)

    @property
    def needs_jump_progress(self) -> bool:
        return bool(self.jump_flight or self.jump_lift or self.jump_push or self.jump_missed or self.jump_settle
                    or self.jump_landing_rate or self.stationary_quiet or self.jump_extension_height_floor
                    or self.jump_tuck or self.jump_takeoff_extension)

    @property
    def needs_flight_geometry(self) -> bool:
        """Whether native and imagined rewards must bind the pinned leg/wheel geometry producer."""
        return bool(self.jump_flight_clearance or self.jump_tuck or self.jump_takeoff_extension)

    def receipt(self) -> dict:
        """Source-identity record: every field, except opt-ins left at their disabled value."""
        receipt = asdict(self)
        for name in self._RECEIPT_OPT_INS:
            if not receipt[name]:
                receipt.pop(name)
        return receipt


@dataclass(frozen=True)
class JumpFlightGeometry:
    """Reward-only two-wheel geometry in metres, from the same bound source.

    Clearance uses the actual collision-cylinder bottom and its local ground.
    Virtual length is hip-axis origin to wheel-link origin, not root-to-wheel
    distance or a world-Z projection. Minimum and span are expanded source
    constants over that source's configured closed-chain stop interval.
    ``valid`` carries terrain/FK availability; it is not a learned contact gate.
    """

    wheel_clearance_m: Tensor
    leg_length_m: Tensor
    minimum_leg_length_m: Tensor
    leg_length_range_m: Tensor
    valid: Tensor

    def validate(self, prototype: Tensor) -> None:
        values = (self.wheel_clearance_m, self.leg_length_m,
                  self.minimum_leg_length_m, self.leg_length_range_m)
        if (self.valid.shape != prototype.shape or self.valid.device != prototype.device
                or self.valid.dtype != torch.bool
                or any(value.shape != (*prototype.shape, 2) or value.device != prototype.device
                       or not value.is_floating_point() for value in values)):
            raise ValueError('Jump flight geometry needs matching floating two-wheel rows and a boolean validity mask')

    def numeric(self) -> Tensor:
        """Rows whose values are finite with positive leg lengths and spans."""
        return (torch.isfinite(torch.stack((self.wheel_clearance_m, self.leg_length_m,
                                            self.minimum_leg_length_m, self.leg_length_range_m), -1)).all((-2, -1))
                & (self.leg_length_m > 0).all(-1)
                & (self.minimum_leg_length_m > 0).all(-1)
                & (self.leg_length_range_m > 0).all(-1))

    def selected_legs(self, rows: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Leg length, minimum and span; unselected rows get finite placeholders.

        Selecting before arithmetic keeps an unrelated invalid row from
        poisoning another row's BPTT derivative through ``torch.where``.
        """
        rows = rows[:, None]
        return (torch.where(rows, self.leg_length_m, 0.), torch.where(rows, self.minimum_leg_length_m, 0.),
                torch.where(rows, self.leg_length_range_m, 1.))

    @classmethod
    def cat(cls, items):
        items = tuple(items)
        return cls(*(torch.cat([getattr(item, name) for item in items])
                     for name in cls.__dataclass_fields__))

    def intervals(self, count):
        return type(self)(*(getattr(self, name).unflatten(0, (count, 4))
                            for name in self.__dataclass_fields__))

    def tick(self, index):
        return type(self)(*(getattr(self, name)[:, index] for name in self.__dataclass_fields__))


@dataclass(frozen=True)
class JumpProgress:
    """Per-row request state the jump extensions need beyond the native phase.

    ``reference_stage`` is the reference clock stage (int64), held across one
    policy interval. ``ascent_seen`` is the native per-request latch of real
    ascent (upward root-origin speed > 0.15 m/s and rise > 1 cm). ``flight_started``
    additionally requires the native two airborne ticks: extending a supported
    leg can satisfy ascent alone. ``flight_closed``
    marks that the request's first real flight has touched down (see
    :func:`advance_flight_latch`); producers without that latch pass None and
    then cannot pay flight terms on jump rows. All are read after the task
    owner observed the physical tick. ``platform_bound`` certifies that the
    selected platform evidence belongs to the same still-live physical request.
    """

    reference_stage: Tensor
    ascent_seen: Tensor
    flight_closed: Tensor | None = None
    landed_s: Tensor | None = None       # native uninterrupted stable-landing time [s]; > 0 while stable
    flight_started: Tensor | None = None # first-real-flight latch, retained after touchdown
    peak_rise: Tensor | None = None      # running native root-height rise; differentiable in imagination
    requested_rise: Tensor | None = None
    physical_active: Tensor | None = None  # current physical request >= 0, observed after this physics tick
    platform_bound: Tensor | None = None  # selected platform evidence belongs to that same live request
    platform_target: PlatformTargetProgress | None = None  # continuous cost only before both-wheel landing
    flight_geometry: JumpFlightGeometry | None = None  # true wheel clearance and source-normalized virtual legs

    def validate(self, mode: Tensor) -> None:
        flags = tuple(flag for flag in (self.ascent_seen, self.flight_closed, self.flight_started,
                                       self.physical_active, self.platform_bound)
                      if flag is not None)
        if (self.reference_stage.shape != mode.shape or self.reference_stage.device != mode.device
                or self.reference_stage.dtype != torch.long
                or any(flag.shape != mode.shape or flag.device != mode.device or flag.dtype != torch.bool
                       for flag in flags)
                or self.landed_s is not None and (self.landed_s.shape != mode.shape or self.landed_s.device != mode.device
                                                  or not self.landed_s.is_floating_point())):
            raise ValueError('Jump progress needs matching int64 reference stages, boolean latches and float times')
        if self.platform_target is not None:
            self.platform_target.validate(mode)
        if self.flight_geometry is not None:
            self.flight_geometry.validate(mode)
        peaks = (self.peak_rise, self.requested_rise)
        if ((peaks[0] is None) != (peaks[1] is None)
                or any(value is not None and (value.shape != mode.shape or value.device != mode.device
                       or not value.is_floating_point()) for value in peaks)):
            raise ValueError('Jump peak progress needs matching floating-point achieved and requested rises')

    @staticmethod
    def cat(items) -> JumpProgress | None:
        items = tuple(items)
        if all(item is None for item in items):
            return None
        if (any(item is None for item in items) or len({item.flight_closed is None for item in items}) != 1
                or len({item.landed_s is None for item in items}) != 1
                or len({item.flight_started is None for item in items}) != 1
                or len({item.peak_rise is None for item in items}) != 1
                or len({item.requested_rise is None for item in items}) != 1
                or len({item.physical_active is None for item in items}) != 1
                or len({item.platform_bound is None for item in items}) != 1
                or len({item.platform_target is None for item in items}) != 1
                or len({item.flight_geometry is None for item in items}) != 1):
            raise ValueError('Jump progress coverage changes within a physical interval')
        optional = lambda name: (None if getattr(items[0], name) is None
                                 else torch.cat([getattr(item, name) for item in items]))
        platform = None
        if items[0].platform_target is not None:
            # Ordinary/jump deployments do not install the optional selected-
            # platform producer. Import its implementation only when supplied.
            from .course.platform_target_reward import PlatformTargetProgress

            platform = PlatformTargetProgress.cat(item.platform_target for item in items)
        return JumpProgress(torch.cat([item.reference_stage for item in items]),
                            torch.cat([item.ascent_seen for item in items]),
                            optional('flight_closed'), optional('landed_s'), optional('flight_started'),
                            optional('peak_rise'), optional('requested_rise'), optional('physical_active'),
                            optional('platform_bound'),
                            platform,
                            None if items[0].flight_geometry is None else
                            JumpFlightGeometry.cat(item.flight_geometry for item in items))


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


def jump_tracking_targets(phase: Tensor, curve_height: Tensor, curve_velocity: Tensor,
                          landing_height: Tensor, *, first_flight_closed: Tensor | None = None):
    """Use landing targets only after a real flight has touched down.

    The native contact phase also calls a brief compression unload followed by
    recontact LANDING. That event must not erase the remaining upward target.
    Producers without the optional real-flight latch retain native semantics.
    """
    landing = (phase == int(RewardPhase.LANDING)) | (phase == int(RewardPhase.SETTLING))
    if first_flight_closed is not None:
        landing = landing & first_flight_closed
    return (torch.where(landing, landing_height, curve_height),
            torch.where(landing, 0., curve_velocity))


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
                             jump_progress: JumpProgress | None = None,
                             stair_base_net_upward_n: Tensor | None = None) -> RewardDensities:
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
    stair_transfer = stair_supported = torch.zeros_like(ordinary)
    if parameters.extensions.stair_support_transfer:
        if (stair_base_net_upward_n is None
                or stair_base_net_upward_n.shape != ordinary.shape
                or stair_base_net_upward_n.device != ordinary.device
                or not stair_base_net_upward_n.is_floating_point()):
            raise ValueError('Stair support transfer requires same-tick base net upward force rows')
        if reference.intentional_base_contact is not None:
            stair_transfer = ((reference.mode == int(CommandMode.STAIR))
                              & reference.intentional_base_contact & ~gimbal_active)
        valid_force = torch.isfinite(stair_base_net_upward_n) & (stair_base_net_upward_n >= 0)
        invalid |= stair_transfer & ~valid_force
        stair_base_net_upward_n = torch.where(valid_force, stair_base_net_upward_n, 0.)
        # Reuse the native harmful-contact threshold, without treating a
        # horizontal collision's large normal load as vertical support.
        stair_supported = stair_transfer & (stair_base_net_upward_n > 3.)
        ordinary_eh = torch.where(stair_supported, ordinary_eh.clamp_min(0), ordinary_eh)
    full, zero = VELOCITY_HEIGHT_GATE_M
    velocity_gate = torch.where((reference.mode == int(CommandMode.NORMAL)) & ~terrain_active,
                                ((zero - support_eh.abs()) / (zero - full)).clamp(0, 1), 1.)
    if rough_dash:
        velocity_gate = torch.where(ordinary,
            ((zero - ordinary_eh.abs()) / (zero - full)).clamp(0, 1), velocity_gate)
    velocity_gate = torch.where(stair_transfer, 1., velocity_gate)
    jump_eh = sample.root_world_z_m - reference.jump_height_world_z_m
    eh = torch.where(jump, jump_eh, ordinary_eh)
    if parameters.extensions.jump_extension_height_floor:
        from transformer_rl.formal.tasks.jump import ReferenceStage

        if jump_progress is None or jump_progress.flight_closed is None:
            if jump.any():
                raise ValueError('Jump extension height floor requires explicit first-flight closure and request progress')
        else:
            jump_progress.validate(reference.mode)
            extension = (jump & (jump_progress.reference_stage == int(ReferenceStage.EXTENSION))
                         & ~jump_progress.flight_closed)
            invalid |= extension & ~torch.isfinite(jump_eh)
            # Preserve the signed physical error in telemetry; only the height
            # objective stops pulling down an already above-reference extension.
            eh = torch.where(extension, eh.clamp_max(0), eh)
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
    harmful_contact = (sample.body_normal_load_n[:, parameters.harmful_indices] > 3).any(-1)
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
            'harmful_contact': 2 * harmful_contact.float(),
            'contact_slip': .5 * (slip * (sample.wheel_contact_tangent_speed_m_s / .2).square()).sum(-1),
            'joint_boundary': 2 * (sample.joint_boundary_shortfall_rad / .05).square().sum(-1),
            'flight_wheel_effort': .2 * flight * (effort[:, 4:] / 4.5).square().sum(-1),
            'landing_impact': .5 * landing * h(excess_landing)}
    if parameters.extensions.ordinary_leg_acceleration is not None:
        cost['leg_acceleration'] = torch.where(ordinary,
            parameters.extensions.ordinary_leg_acceleration * ddq[:, :4].square().sum(-1),
            cost['leg_acceleration'])
    if reference.intentional_base_contact is not None and (not rough_dash or parameters.extensions.stair_support_transfer):
        base = parameters.base_index
        # Explicitly permitted terrain base contact exempts only the base.
        others = [index for index in parameters.harmful_indices if index != base]
        exempt_base = (stair_supported if rough_dash else reference.intentional_base_contact)
        cost['harmful_contact'] = torch.where(
            exempt_base,
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
    physical_tail = torch.zeros_like(jump)
    if extensions.jump_physical_tail:
        if jump_progress is None or jump_progress.physical_active is None:
            raise ValueError('Jump physical tail requires explicit per-tick physical jump activity')
        tail_surface = ~terrain_active & ~terrain_airborne & ~rough_active
        if jump_progress.platform_bound is not None:
            # Terrain reward routing must not erase an explicit platform
            # request's physical tail after the protocol returns to NORMAL.
            tail_surface = tail_surface | jump_progress.platform_bound
        physical_tail = (jump_progress.physical_active & (reference.mode == int(CommandMode.NORMAL))
                         & tail_surface & ~gimbal_active)
    settling_scope = jump | physical_tail
    if extensions.jump_missed:
        from transformer_rl.formal.tasks.jump import ReferenceStage

        if stage is not None:
            # Only the explicit end-of-reference HOLD certifies a missed push;
            # a zero velocity target also occurs at compression start and apex.
            # Neither supported extension nor unloading without ascent is a
            # jump. Use the same real-flight latch as the flight rewards.
            started = jump_progress.flight_started
            if started is None:
                if jump.any():
                    raise ValueError('Missed jump reward requires the first-real-flight latch')
            else:
                missed_jump = jump & (stage == int(ReferenceStage.HOLD)) & ~started
        # A missed request must not keep earning locomotion tracking rewards.
        # Keep jump_attitude to encourage upright support while waiting,
        # instead of turning the entire HOLD interval into pure punishment.
        if extensions.jump_peak_completion and stage is not None:
            peak, requested = jump_progress.peak_rise, jump_progress.requested_rise
            if peak is None or requested is None or closed is None or started is None:
                raise ValueError('Jump peak completion requires peak, request and first-flight records')
            tolerance = torch.minimum(torch.full_like(requested, JUMP_PEAK_TOLERANCE_MAX_M),
                                      requested * JUMP_PEAK_TOLERANCE_TARGET_FRACTION)
            # Inactive rows carry a zero request. Keep their division defined
            # without altering any admitted jump target or success tolerance.
            lower = torch.where(settling_scope, requested - tolerance, torch.ones_like(requested))
            if (settling_scope & (lower <= 0)).any():
                raise ValueError('Jump peak completion requires a positive admitted rise')
            completion = started.to(peak.dtype) * (peak / lower).clamp(0, 1)
            # Protocol completion can reset its clock before physical peak or
            # landing qualification. The still-live request owns this tail.
            late = (jump & ((stage == int(ReferenceStage.HOLD)) | closed)) | physical_tail
            gain = torch.where(late, completion, 1.)
            for name in ('forward', 'yaw', 'height', 'jump_vertical'):
                positive[name] = positive[name] * gain
            cost['jump_missed'] = extensions.jump_missed * torch.where(late, 1 - completion, 0.)
            missed_jump = late & (completion < 1)
        else:
            for name in ('forward', 'yaw', 'height', 'jump_vertical'):
                positive[name] = torch.where(missed_jump, 0., positive[name])
            cost['jump_missed'] = extensions.jump_missed * missed_jump.to(v_up.dtype)
    if extensions.jump_missed and jump_progress is not None and jump_progress.platform_target is not None:
        # Keep the peak objective and geometric approach gradients independent.
        # Together their costs can reach 2 * jump_missed; no new multiplier.
        target = jump_progress.platform_target
        target.validate(reference.mode)
        if jump_progress.platform_bound is None or jump_progress.physical_active is None:
            raise ValueError('Platform target cost requires the current physical request binding')
        target_scope = (target.eligible & jump_progress.platform_bound
                        & jump_progress.physical_active)
        cost['jump_target'] = extensions.jump_missed * torch.where(target_scope, target.cost, 0.)
    flight_geometry_available = torch.zeros_like(jump)
    common_clearance = torch.zeros_like(v_up)
    tuck_shortfall = torch.zeros_like(v_up)
    if extensions.jump_flight or extensions.jump_lift or extensions.jump_tuck:
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
        geometry_flight = flying
        flight_score = flying.to(v_up.dtype)
        if extensions.jump_flight_clearance or extensions.jump_tuck:
            geometry = None if jump_progress is None else jump_progress.flight_geometry
            if geometry is None:
                if jump.any():
                    raise ValueError('Jump geometric flight rewards require an explicit flight geometry producer')
                if extensions.jump_tuck:
                    positive['jump_tuck'] = torch.zeros_like(v_up)
            else:
                geometry.validate(reference.mode)
                numeric = geometry.numeric()
                requested = jump_progress.requested_rise
                if extensions.jump_flight_clearance:
                    if requested is None:
                        if jump.any():
                            raise ValueError('Jump flight clearance requires the admitted requested rise')
                        requested = torch.ones_like(v_up)
                    numeric &= torch.isfinite(requested) & (requested > 0)
                invalid |= geometry_flight & geometry.valid & ~numeric
                flight_geometry_available = geometry_flight & geometry.valid & numeric
                # Invalid terrain rows follow the existing producer's reset/mask
                # path. Select finite values before arithmetic so an unrelated
                # missing query cannot poison another row's BPTT derivative.
                available = flight_geometry_available[:, None]
                clearance = torch.where(available, geometry.wheel_clearance_m, 0.)
                common_clearance = clearance.amin(-1)
                if extensions.jump_flight_clearance:
                    scale = torch.where(flight_geometry_available, requested, 1.)
                    raised = common_clearance.clamp_min(0)
                    flight_score = raised / (scale + raised)
                if extensions.jump_tuck:
                    length, minimum, span = geometry.selected_legs(flight_geometry_available)
                    shortfall = (length - minimum).clamp_min(0)
                    tuck_shortfall = shortfall.mean(-1)
                    tuck_score = (1 + (shortfall / span).square()).reciprocal().mean(-1)
                    positive['jump_tuck'] = (extensions.jump_tuck * flight_geometry_available * tuck_score)
        flying = flying.to(v_up.dtype)
        rise = sample.velocity_origin_world_m_s[:, 2].clamp(0, extensions.jump_lift_cap_m_s)
        positive['jump_flight'] = extensions.jump_flight * flight_score
        positive['jump_lift'] = extensions.jump_lift * flying * rise
    if extensions.jump_push or extensions.jump_takeoff_extension:
        # Supported (not airborne) while the reference demands upward speed,
        # i.e. in its extension/ballistic push. Native TAKEOFF alone is not
        # enough: early wheel unloading moves the native phase to LANDING.
        push_target = reference.jump_velocity_world_z_m_s
        pushing = (jump & (phase != int(RewardPhase.FLIGHT)) & (push_target > JUMP_PUSH_MIN_REFERENCE_M_S))
        if closed is None:
            if jump.any():
                raise ValueError('Jump push rewards require the first-flight latch of the request')
        else:
            pushing = pushing & ~closed   # no second push after the first real flight landed
    if extensions.jump_push:
        achieved = (sample.velocity_origin_world_m_s[:, 2]
                    / push_target.clamp_min(JUMP_PUSH_MIN_REFERENCE_M_S)).clamp(0, 1)
        positive['jump_push'] = extensions.jump_push * pushing.to(v_up.dtype) * achieved
    takeoff_available = torch.zeros_like(jump)
    takeoff_extension = torch.zeros_like(v_up)
    if extensions.jump_takeoff_extension:
        geometry = None if jump_progress is None else jump_progress.flight_geometry
        if geometry is None:
            if jump.any():
                raise ValueError('Jump takeoff extension requires an explicit leg geometry producer')
        else:
            geometry.validate(reference.mode)
            numeric = geometry.numeric()
            rising = pushing & (sample.velocity_origin_world_m_s[:, 2] > JUMP_TAKEOFF_RISE_M_S)
            invalid |= rising & geometry.valid & ~numeric
            takeoff_available = rising & geometry.valid & numeric
            length, minimum, span = geometry.selected_legs(takeoff_available)
            takeoff_extension = ((length - minimum) / span).clamp(0, 1).mean(-1)
        positive['jump_takeoff_extension'] = (extensions.jump_takeoff_extension
                                              * takeoff_available * takeoff_extension)
    if extensions.jump_settle:
        settled = torch.zeros_like(v_up)
        if stage is not None:
            if landed is None:
                if jump.any():
                    raise ValueError('Jump settle rewards require the native stable-landing time')
            else:
                streak = (landed.to(v_up.dtype) / JUMP_SETTLE_STREAK_S).clamp(0, 1)
                settled = torch.where(settling_scope & (landed > 0), .25 + .75 * streak, 0.)
        positive['jump_settle'] = extensions.jump_settle * settled
    if extensions.jump_landing_rate:
        landed_after_flight = torch.zeros_like(jump)
        if stage is not None:
            if closed is None:
                if jump.any():
                    raise ValueError('Jump landing-rate costs require the first-flight latch of the request')
            else:
                landed_after_flight = settling_scope & closed & (phase != int(RewardPhase.FLIGHT))
        excess = (sample.angular_velocity_body_rad_s.norm(dim=-1) - JUMP_LANDING_RATE_FREE_RAD_S).clamp_min(0)
        cost['jump_landing_rate'] = extensions.jump_landing_rate * landed_after_flight.to(v_up.dtype) * excess.square()
    if extensions.recovery_quiet:
        # Existing height reward approaches zero while collapsed or inverted.
        # Use all axes: yaw is not otherwise tracked during native recovery.
        angular_square = sample.angular_velocity_body_rad_s.square().sum(-1)
        positive['recovery_quiet'] = (extensions.recovery_quiet * recovery * positive['height']
                                     * torch.exp(-angular_square / .25))
    if extensions.stationary_quiet:
        if jump_progress is None or jump_progress.physical_active is None:
            raise ValueError('Stationary quiet requires explicit per-tick physical jump activity')
        jump_progress.validate(reference.mode)
        # Use the native 5 N stable-wheel support and existing >3 N harmful
        # contact predicates; absent contact coverage is never evidence of rest.
        contact_available = sample.available_terms is None or 'harmful_contact' in sample.available_terms
        supported = (wheel_load >= 5).all(-1) & ~harmful_contact & contact_available
        stationary = ((reference.mode == int(CommandMode.NORMAL)) & exact_still & supported
                      & ~terrain_active & ~terrain_airborne & ~rough_active
                      & ~jump_progress.physical_active)
        angular_square = sample.angular_velocity_body_rad_s.square().sum(-1)
        positive['stationary_quiet'] = (extensions.stationary_quiet * stationary * positive['height']
                                       * upright.clamp_min(0) * torch.exp(-angular_square / .25)
                                       * torch.exp(-cost['stand_world_xy']))
    p, c = torch.stack(tuple(positive.values())).sum(0), torch.stack(tuple(cost.values())).sum(0)
    components = {**{'positive.' + key: value for key, value in positive.items()},
                  **{'cost.' + key: -value for key, value in cost.items()}}
    positive_bound = torch.where(gimbal_active, 11., 8.)
    positive_bound = (positive_bound + extensions.jump_flight + extensions.jump_lift * extensions.jump_lift_cap_m_s
                      + extensions.jump_push + extensions.jump_settle + extensions.recovery_quiet
                      + extensions.stationary_quiet + extensions.jump_tuck + extensions.jump_takeoff_extension)
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
    if extensions.jump_flight_clearance or extensions.jump_tuck:
        gates['jump_flight_geometry_available'] = flight_geometry_available
        errors['jump_common_clearance_m'] = common_clearance
        errors['jump_tuck_length_excess_m'] = tuck_shortfall
    if extensions.jump_takeoff_extension:
        gates['jump_takeoff_extension_available'] = takeoff_available
        errors['jump_takeoff_extension_fraction'] = takeoff_extension
    if extensions.stair_support_transfer:
        gates['stair_support_transfer'] = stair_transfer
        gates['stair_base_supported'] = stair_supported
        errors['stair_objective_height_error_m'] = torch.where(stair_transfer, ordinary_eh, 0.)
        errors['stair_base_net_upward_n'] = torch.where(stair_transfer, stair_base_net_upward_n, 0.)
    if extensions.jump_missed and jump_progress is not None and jump_progress.platform_target is not None:
        target = jump_progress.platform_target
        gates['jump_target_eligible'] = target.eligible
        gates['jump_target_available'] = target.available & target.eligible
        errors['jump_target_horizontal_m'] = torch.where(target.available,
            target.horizontal_distance_m.mean(-1), 0.)
        errors['jump_target_vertical_shortfall_m'] = torch.where(target.available,
            target.vertical_shortfall_m.mean(-1), 0.)
    errors.update(gimbal_errors)
    return RewardDensities(positive, cost, components, gates, errors, failure_scale, invalid)
