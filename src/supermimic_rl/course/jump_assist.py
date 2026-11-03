"""Training-only vertical assist for requested jumps (SCUT jump_takeoff assist).

A policy that has never left the ground never experiences flight, landing or
the success event, so no reward reaches its takeoff. As in SCUT's jump_takeoff
state machine, a decaying share of accepted jump requests receives an upward
base force during the reference push window, until verified flight. The force
closes the shortfall to the reference launch speed and is capped by weight.

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
COUNTERS = ('assisted_requests', 'assisted_ended', 'assisted_success', 'unassisted_ended', 'unassisted_success')


@dataclass(frozen=True)
class JumpAssistConfig:
    probability: float                  # chance an accepted request is assisted at the stage start
    fade_stage_fraction: float = .75    # stage fraction at which that chance reaches zero
    velocity_gain_per_s: float = 10.    # force = mass * gain * launch-speed shortfall
    max_weight_fraction: float = 1.     # force cap as a fraction of the robot weight
    stage: str = 'active_jump'
    gravity_m_s2: float = 9.81
    seed: int = 0

    def __post_init__(self):
        values = (self.probability, self.fade_stage_fraction, self.velocity_gain_per_s,
                  self.max_weight_fraction, self.gravity_m_s2)
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                or not 0 <= self.probability <= 1 or not 0 < self.fade_stage_fraction <= 1
                or self.velocity_gain_per_s < 0 or self.max_weight_fraction < 0 or self.gravity_m_s2 <= 0
                or type(self.seed) is not int):
            raise ValueError('Jump assist needs a probability in [0, 1], a fade fraction in (0, 1] '
                             'and nonnegative force settings')


class JumpAssist:
    """Selection, force and outcome accounting; counters stay on device until ``metrics``."""

    def __init__(self, native, curriculum, config: JumpAssistConfig):
        if config.stage not in curriculum.stage_names:
            raise ValueError('Jump assist stage is not part of the native course: ' + config.stage)
        self.native, self.curriculum, self.config = native, curriculum, config
        index = list(curriculum.stage_names).index(config.stage)
        self.stage_start = sum(curriculum.stage_budgets[:index])
        self.stage_budget = curriculum.stage_budgets[index]
        device = native._terrain_course.tasks.physical_id.device
        self.request = torch.full((native.num_envs,), -1, dtype=torch.long, device=device)
        self.selected = torch.zeros(native.num_envs, dtype=torch.bool, device=device)
        self.generator = torch.Generator(device=device).manual_seed(config.seed)
        self.counts = torch.zeros(len(COUNTERS), dtype=torch.long, device=device)

    def _progress(self) -> float:
        return (self.curriculum.real_transitions - self.stage_start) / self.stage_budget

    def probability(self) -> float:
        progress = self._progress()
        if not 0 <= progress < 1:
            return 0.
        return max(0., self.config.probability * (1 - progress / self.config.fade_stage_fraction))

    def _count(self, name, rows):
        index = COUNTERS.index(name)
        self.counts[index] += rows.sum()

    def track(self, tasks):
        """Attribute request endings and select newly accepted requests (idempotent)."""
        current = tasks.physical_id.long()
        changed = current != self.request
        ended = changed & (self.request >= 0)
        self._count('assisted_ended', ended & self.selected)
        self._count('unassisted_ended', ended & ~self.selected)
        accepted = changed & (current >= 0)
        draw = torch.rand(current.shape, device=current.device, generator=self.generator) < self.probability()
        self.selected = torch.where(accepted, draw, self.selected & (current >= 0))
        self._count('assisted_requests', accepted & draw)
        self.request = current.clone()

    def record_success(self, before, tasks):
        """Count successes decided by one native ``observe_physics`` call.

        The native task clears ``success_id`` at the next policy step and at
        resets, so a success is attributed where it is decided, together with
        the ending of the same request.
        """
        success = (before >= 0) & (tasks.physical_id.long() < 0) & (tasks.success_id.long() == before)
        self._count('assisted_success', success & self.selected)
        self._count('unassisted_success', success & ~self.selected)
        self.track(tasks)

    def force_z(self):
        """Upward world force [N] per environment for this substep; None outside the assist stage."""
        native = self.native
        tasks = native._terrain_course.tasks
        self.track(tasks)
        if not 0 <= self._progress() < 1:
            return None
        curve = tasks.jump_reference()
        push = ((curve.clock_stage == EXTENSION)
                | ((curve.clock_stage == BALLISTIC) & (curve.velocity_world_z_m_s > 0)))
        window = self.selected & (tasks.physical_id >= 0) & ~tasks.flight_seen & push
        velocity_z = native._as_torch(native.robot.data.root_com_vel_w)[:, 2]
        mass = native._formal_reward.composer._mass_kg
        shortfall = (curve.launch_velocity_m_s.to(velocity_z.dtype) - velocity_z).clamp_min(0)
        limit = self.config.max_weight_fraction * mass * self.config.gravity_m_s2
        force = torch.minimum(mass * self.config.velocity_gain_per_s * shortfall, limit)
        return torch.where(window, force, torch.zeros_like(force))

    def metrics(self) -> tuple[dict[str, float], dict[str, float]]:
        """Cumulative counters (one host transfer) and the current assist probability gauge."""
        values = self.counts.tolist()
        return ({f'course/jump_assist/{name}': value for name, value in zip(COUNTERS, values)},
                {'course/jump_assist/probability': self.probability()})

    def receipt(self) -> dict:
        return {**asdict(self.config), 'stage_start_transitions': self.stage_start,
                'stage_budget_transitions': self.stage_budget,
                'force': 'world +z on base_link through the native instantaneous wrench composer',
                'scope': 'training only; evaluation and world-model collection never install it'}


def install_jump_assist(native, curriculum, configuration: dict) -> JumpAssist:
    """Add the assist wrench after the native per-substep action/spring/DR wrenches."""
    assist = JumpAssist(native, curriculum, JumpAssistConfig(**configuration))
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

    tasks = native._terrain_course.tasks
    observe_physics = tasks.observe_physics

    def observe_and_count(*args, **kwargs):
        before = tasks.physical_id.long().clone()
        result = observe_physics(*args, **kwargs)
        assist.record_success(before, tasks)
        return result

    native._apply_action = apply_action
    tasks.observe_physics = observe_and_count
    native.supermimic_jump_assist = assist
    return assist
