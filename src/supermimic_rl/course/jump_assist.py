"""Training-only vertical assist for requested jumps (SCUT jump_takeoff assist).

A policy that has never left the ground never experiences flight, landing or
the success event, so no reward reaches its takeoff. As in SCUT's jump_takeoff
state machine, accepted jump requests receive an upward base force during the
reference push window, until real liftoff (airborne after real ascent). The force closes the shortfall
to the reference launch speed and is capped by weight. Two decays:

* ``probability``: a fading share of requests is fully assisted;
* ``force`` (SCUT ``decay_mode: force``): the force scale fades, so the
  policy's own push keeps a marginal effect. An optional fixed unassisted
  request cohort measures capability while the course coordinate is held.

A request counts as assisted only if assist impulse was actually delivered to
it (selection alone is not assistance).

It is installed only from an explicit training factory section, never with a
fixed evaluation suite (and world-model factories do not request it). Assisted
and unassisted request outcomes are counted separately, so capability gates
can admit only unassisted jumps.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch

# transformer_rl.formal.tasks.jump.ReferenceStage
EXTENSION, BALLISTIC = 1, 2
# Outcomes of all requests and of unassisted requests per requested-rise bin [m].
RISE_BIN_EDGES = (.10, .20, .30)
RISE_BINS = tuple(f'{low:.2f}-{high:.2f}' for low, high in zip((.0, *RISE_BIN_EDGES), (*RISE_BIN_EDGES, .40)))
# "liftoff": the ended request was airborne after real ascent (whether or not it then succeeded).
COUNTERS = ('assisted_requests', 'assisted_ended', 'assisted_success', 'unassisted_ended', 'unassisted_success',
            *(f'{who}{kind}_rise_{name}' for who in ('', 'unassisted_') for name in RISE_BINS
              for kind in ('ended', 'success', 'liftoff')))
DECAYS = ('probability', 'force')


@dataclass(frozen=True)
class JumpAssistConfig:
    probability: float                  # selection chance (probability decay) or force scale at the stage start
    fade_stage_fraction: float = .75    # stage fraction at which the decayed quantity reaches zero
    decay: str = 'probability'
    velocity_gain_per_s: float = 10.    # force = mass * gain * launch-speed shortfall
    max_weight_fraction: float = 1.     # force cap as a fraction of the robot weight
    stage: str = 'active_jump'
    gravity_m_s2: float = 9.81
    seed: int = 0
    assist_request_probability: float = 1.  # independent fixed request cohort; force amplitude still fades

    def __post_init__(self):
        values = (self.probability, self.fade_stage_fraction, self.velocity_gain_per_s,
                  self.max_weight_fraction, self.gravity_m_s2, self.assist_request_probability)
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                or not 0 <= self.probability <= 1 or not 0 < self.fade_stage_fraction <= 1
                or not 0 <= self.assist_request_probability <= 1
                or self.velocity_gain_per_s < 0 or self.max_weight_fraction < 0 or self.gravity_m_s2 <= 0
                or type(self.seed) is not int or self.decay not in DECAYS):
            raise ValueError('Jump assist needs a probability in [0, 1], a fade fraction in (0, 1] '
                             'and nonnegative force settings')


class JumpAssist:
    """Selection, force and outcome accounting; counters stay on device until ``metrics``."""

    def __init__(self, native, curriculum, config: JumpAssistConfig, latch):
        if config.stage not in curriculum.stage_names:
            raise ValueError('Jump assist stage is not part of the native course: ' + config.stage)
        self.native, self.curriculum, self.config = native, curriculum, config
        index = list(curriculum.stage_names).index(config.stage)
        self.stage_start = sum(curriculum.stage_budgets[:index])
        self.stage_budget = curriculum.stage_budgets[index]
        device = native._terrain_course.tasks.physical_id.device
        self.request = torch.full((native.num_envs,), -1, dtype=torch.long, device=device)
        self.selected = torch.zeros(native.num_envs, dtype=torch.bool, device=device)
        # A request counts as assisted only if assist impulse was actually delivered to it.
        self.impulse = torch.zeros(native.num_envs, device=device)
        # Real liftoff (airborne for the native 2 ticks after real ascent) comes
        # from the per-tick first-flight latch the shared reward also reads.
        # Native flight_seen (2 N wheel unloading) is reachable by retracting the legs.
        self.latch = latch
        self.rise = torch.zeros(native.num_envs, device=device)   # requested rise latched at acceptance
        self.edges = torch.tensor(RISE_BIN_EDGES, device=device)
        self.generator = torch.Generator(device=device).manual_seed(config.seed)
        self.counts = torch.zeros(len(COUNTERS), dtype=torch.long, device=device)

    def _progress(self) -> float:
        return (self.curriculum.real_transitions - self.stage_start) / self.stage_budget

    def level(self) -> float:
        """Selection chance (probability decay) or force scale (force decay).

        With a performance jump height ladder the assist fades per opened
        height level (1, 2/3, 1/3, 0 of ``probability`` for 3 gates), so the
        full height range is trained unassisted, independent of the stage
        coordinate; otherwise it fades with the coordinate inside the assist stage.
        """
        ladder = getattr(self.curriculum, 'jump_height_level', lambda: None)()
        if ladder is not None:
            return self.config.probability * (1 - ladder / len(self.curriculum.config.jump_height_gates))
        progress = self._progress()
        if not 0 <= progress < 1:
            return 0.
        return max(0., self.config.probability * (1 - progress / self.config.fade_stage_fraction))

    def _count(self, name, rows):
        index = COUNTERS.index(name)
        self.counts[index] += rows.sum()

    def _count_by_rise(self, kind, rows, helped):
        bins = torch.bucketize(self.rise, self.edges, right=True)
        for index, name in enumerate(RISE_BINS):
            in_bin = rows & (bins == index)
            self._count(f'{kind}_rise_{name}', in_bin)
            self._count(f'unassisted_{kind}_rise_{name}', in_bin & ~helped)

    def lifted(self, request):
        """Real liftoff seen for ``request`` (rows whose latch still belongs to it)."""
        return self.latch.lifted & (self.latch.request == request) & (request >= 0)

    def track(self, tasks, lifted=None):
        """Attribute request endings and select newly accepted requests (idempotent).

        ``lifted`` is the ended requests' liftoff when the latch has already
        moved on to the next request (after an observe_physics call).
        """
        current = tasks.physical_id.long()
        changed = current != self.request
        ended = changed & (self.request >= 0)
        helped = self.impulse > 0
        lifted = self.lifted(self.request) if lifted is None else lifted
        self._count('assisted_ended', ended & helped)
        self._count('unassisted_ended', ended & ~helped)
        self._count_by_rise('ended', ended, helped)
        self._count_by_rise('liftoff', ended & lifted, helped)
        accepted = changed & (current >= 0)
        level = self.level()
        chance = self.config.assist_request_probability * (float(level > 0) if self.config.decay == 'force' else level)
        if self.config.decay == 'force' and chance in (0., 1.):
            # Preserve the default force-decay RNG contract.
            draw = torch.full(current.shape, chance == 1., dtype=torch.bool, device=current.device)
        else:
            draw = torch.rand(current.shape, device=current.device, generator=self.generator) < chance
        self.selected = torch.where(accepted, draw, self.selected & (current >= 0))
        self.impulse = torch.where(changed, 0., self.impulse)
        self.rise = torch.where(accepted, tasks.active_rise, self.rise)
        self._count('assisted_requests', accepted & draw)
        self.request = current.clone()

    def record_success(self, before, tasks, lifted_before):
        """Count successes decided by one native ``observe_physics`` call.

        The native task clears ``success_id`` at the next policy step and at
        resets, so a success is attributed where it is decided, together with
        the ending of the same request.
        """
        success = (before >= 0) & (tasks.physical_id.long() < 0) & (tasks.success_id.long() == before)
        helped = self.impulse > 0
        self._count('assisted_success', success & helped)
        self._count('unassisted_success', success & ~helped)
        self._count_by_rise('success', success, helped)
        self.track(tasks, lifted_before)

    def force_z(self):
        """Upward world force [N] per environment for this substep; None outside the assist stage."""
        native = self.native
        tasks = native._terrain_course.tasks
        self.track(tasks)
        if self.level() <= 0:
            return None
        curve = tasks.jump_reference()
        push = ((curve.clock_stage == EXTENSION)
                | ((curve.clock_stage == BALLISTIC) & (curve.velocity_world_z_m_s > 0)))
        window = self.selected & (tasks.physical_id >= 0) & ~self.lifted(tasks.physical_id.long()) & push
        velocity_z = native._as_torch(native.robot.data.root_com_vel_w)[:, 2]
        mass = native._formal_reward.composer._mass_kg
        shortfall = (curve.launch_velocity_m_s.to(velocity_z.dtype) - velocity_z).clamp_min(0)
        limit = self.config.max_weight_fraction * mass * self.config.gravity_m_s2
        force = torch.minimum(mass * self.config.velocity_gain_per_s * shortfall, limit)
        if self.config.decay == 'force':
            force = force * self.level()
        force = torch.where(window, force, torch.zeros_like(force))
        self.impulse += force * float(native.physics_dt)
        return force

    def metrics(self) -> tuple[dict[str, float], dict[str, float]]:
        """Cumulative counters (one host transfer) and the current assist level gauge."""
        values = self.counts.tolist()
        return ({f'course/jump_assist/{name}': value for name, value in zip(COUNTERS, values)},
                {'course/jump_assist/level': self.level()})

    def receipt(self) -> dict:
        return {**asdict(self.config), 'stage_start_transitions': self.stage_start,
                'stage_budget_transitions': self.stage_budget,
                'force': 'world +z on base_link through the native instantaneous wrench composer',
                'scope': 'training only; evaluation and world-model collection never install it'}


def install_jump_assist(native, curriculum, configuration: dict) -> JumpAssist:
    """Add the assist wrench after the native per-substep action/spring/DR wrenches."""
    from .reward import install_flight_latch

    tasks = native._terrain_course.tasks
    latch = getattr(tasks, 'supermimic_flight_latch', None) or install_flight_latch(tasks)
    assist = JumpAssist(native, curriculum, JumpAssistConfig(**configuration), latch)
    original = native._apply_action
    base = [native._base_body_index]

    def apply_action():
        original()
        force = assist.force_z()
        if force is not None:
            forces = torch.zeros(native.num_envs, 1, 3, dtype=force.dtype, device=force.device)
            forces[:, 0, 2] = force
            native.robot.instantaneous_wrench_composer.add_forces_and_torques_index(
                forces=forces, torques=torch.zeros_like(forces), body_ids=base, is_global=True)

    observe_physics = tasks.observe_physics

    def observe_and_count(*args, **kwargs):
        before = tasks.physical_id.long().clone()
        lifted = assist.lifted(before)
        result = observe_physics(*args, **kwargs)
        # The latch was updated inside (and may have moved to a new request).
        lifted = lifted | assist.lifted(before)
        assist.record_success(before, tasks, lifted)
        return result

    native._apply_action = apply_action
    tasks.observe_physics = observe_and_count
    native.supermimic_jump_assist = assist
    return assist
