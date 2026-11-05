"""Opt-in early jump course using the existing native request/reward protocol.

This changes exposure order, not mechanical parameters or jump success labels.
The legacy eight-stage configuration and its checkpoint format remain intact.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import torch
from transformer_rl.formal.tasks.command_pool import CommandPool
from transformer_rl.formal.tasks.curriculum import TaskFamily, interpolate_domain

from .curriculum import CourseCurriculumConfig, STAGE_NAMES, SuperMimicCurriculum, _TransitionBudget

SCHEDULE = 'jump_first_v1'
STAGE_ORDER = (0, 1, 5, 2, 3, 4, 6, 7)
JUMP_FIRST_STAGE_NAMES = tuple(STAGE_NAMES[index] for index in STAGE_ORDER)


class JumpFirstCommandPool(CommandPool):
    """Jump requests keep their forward speed but never a yaw rate or a random heading.

    The native jump success needs 2 s of stable landing with a body angular
    speed of at most 0.5 rad/s, which no turning command can meet: in R8 the
    yaw part of the stable-landing test started failing when moving jumps
    (with the segment's yaw rate) entered the course. The evaluated moving
    jumps also command no yaw.
    """

    def sample(self, family, *args, **kwargs):
        draw = super().sample(family, *args, **kwargs)
        request = family == int(TaskFamily.REQUEST)
        command = draw.command.clone()
        command[:, 2] = torch.where(request, 0., command[:, 2])
        return replace(draw, command=command, heading_enabled=draw.heading_enabled & ~request)


def install_jump_first_command_pool(tasks, curriculum):
    """Preserve the native pool configuration; affect only the opt-in course."""
    if not isinstance(curriculum, JumpFirstCurriculum):
        return None
    if not isinstance(tasks.command_pool, CommandPool):
        raise ValueError('Jump-first course requires the native command pool')
    tasks.command_pool = JumpFirstCommandPool(tasks.command_pool.config)
    return {'schedule': SCHEDULE, 'request_heading_override': False,
            'native_handoff_preserved': True, 'request_yaw_rate': 0.,
            'early_jump_domain': list(curriculum.initial_domain.jump_m)}


class JumpFirstCurriculum(SuperMimicCurriculum):
    """Balance/height/low speed -> small jumps -> terrain -> recovery/combined.

    Standing and moving jumps use the existing 25%/80% jump-stage milestones.
    Early jumping stays within the pinned initial jump and speed domains. The
    full jump envelope is introduced in combined rehearsal, after terrain and
    recovery. Exposure alone remains a schedule coordinate, never competence.
    """

    _FORMAT = 'supermimic_jump_first_curriculum_v1'

    def __init__(self, num_envs, device='cpu', seed=0, config=None):
        super().__init__(num_envs, device, seed, config)
        self.stage_names = JUMP_FIRST_STAGE_NAMES
        self.stage_budgets = tuple(self.config.stage_transition_budgets[index] for index in STAGE_ORDER)
        weights = [self.config.stage_weights[index] for index in STAGE_ORDER]
        jump = list(weights[2])
        # The former jump mixture contains terrain rehearsal. Before the first
        # terrain stage, retain the other relative shares and normalize them.
        # Recovery births stay as configured, so easy get-up can mix in early.
        jump[int(TaskFamily.TERRAIN)] = 0.
        retained = sum(jump)
        weights[2] = tuple(value / retained for value in jump)
        self.stage_weights = tuple(weights)
        coordinates = self.config.source_schedule_coordinates
        # Do not unlock the later command/gimbal/DR schedules while learning
        # the first jumps. All values derive from the original compatibility
        # coordinates; there is no new hardware or timing parameter here.
        self.source_coordinates = (*coordinates[:3], coordinates[2],
                                   *coordinates[3:6], *coordinates[7:])
        from transformer_rl.formal.tasks.curriculum import ContinuousTaskSampler
        self.sampler = ContinuousTaskSampler(seed, device=device, domain=self.domain,
            stage_weights=self.stage_weights, continuous=True)
        self.budget = self._budget(0)

    # Keys of performance gates the trainer has opened (``stage@fraction``),
    # or None when the jump height follows the coordinate. Not checkpointed
    # here: the trainer restores its gates and re-applies them after resume.
    opened_gates: frozenset[str] | None = None

    def set_opened_gates(self, keys) -> None:
        self.opened_gates = frozenset(keys)

    def ladder_gate_keys(self) -> tuple[str, ...]:
        """Performance gate keys (``stage@fraction``) of the jump height ladder, in order."""
        return tuple(f'{self.config.jump_height_stage}@{fraction:g}'
                     for fraction, _ in self.config.jump_height_gates)

    def jump_height_level(self) -> int | None:
        """Number of jump height gates opened in order, if performance drives the height."""
        if self.opened_gates is None or not self.config.jump_height_gates:
            return None
        level = 0
        for key in self.ladder_gate_keys():
            if key not in self.opened_gates:
                break
            level += 1
        return level

    def _budget(self, real_transitions):
        budget = _TransitionBudget(SimpleNamespace(
            stage_transition_budgets=self.stage_budgets,
            source_schedule_coordinates=self.source_coordinates), real_transitions)
        budget.stage_names = self.stage_names
        return budget

    def set_progress(self, real_transitions, actor_updates):
        super().set_progress(real_transitions, actor_updates)
        self.budget = self._budget(real_transitions)

    def _stage_fraction(self, stage):
        index = self.stage_names.index(stage)
        return self._smooth_fraction(
            self.real_transitions - sum(self.stage_budgets[:index]), self.stage_budgets[index])

    def _effective_domain(self):
        command = interpolate_domain(self.initial_domain, self.domain,
                                     self._stage_fraction('dynamic_commands'))
        height_fraction = self._smooth_fraction(self.real_transitions, sum(self.stage_budgets[:2]))
        height = interpolate_domain(self.initial_domain, self.domain, height_fraction)
        terrain = interpolate_domain(self.initial_domain, self.domain,
                                     self._stage_fraction('terrain_foundation'))
        jump = interpolate_domain(self.initial_domain, self.domain,
                                  self._stage_fraction(self.config.jump_height_stage))
        if self.config.jump_height_gates:
            index = self.stage_names.index(self.config.jump_height_stage)
            start, budget = sum(self.stage_budgets[:index]), self.stage_budgets[index]
            upper = self.initial_domain.jump_m[1]
            level = self.jump_height_level()
            for number, (fraction, cap) in enumerate(self.config.jump_height_gates):
                # Match CourseGates' integer boundary exactly: its held
                # coordinate is boundary - 1, even for a very large budget.
                # With reported (non-holding) gates the opened ladder decides.
                reached = (self.real_transitions >= start + int(fraction * budget) if level is None
                           else number < level)
                if reached:
                    upper = cap
            jump = replace(jump, jump_m=(self.initial_domain.jump_m[0], upper))
        return replace(command, height_m=height.height_m,
            slope_rad=terrain.slope_rad, roughness_m=terrain.roughness_m,
            step_up_m=terrain.step_up_m, step_down_m=terrain.step_down_m,
            jump_m=jump.jump_m)

    def terrain_step_up_cap_m(self):
        if self.stage < self.stage_names.index('terrain_foundation'):
            return 0.
        return float(self._effective_domain().step_up_m[1])

    def terrain_level_cap(self, levels):
        if type(levels) is not int or levels < 1:
            raise ValueError('The actual positive atlas level count is required')
        return int((levels - 1) * self._stage_fraction('terrain_foundation'))

    def _request_motion_scale(self):
        index = self.stage_names.index('active_jump')
        if self.stage < index:
            return 0.
        if self.stage > index:
            return 1.
        start = self.config.jump_standing_fraction_of_stage
        end = self.config.jump_full_motion_fraction_of_stage
        budget = self.stage_budgets[index]
        return self._smooth_fraction(self.budget.consumed[index] - start * budget,
                                     (end - start) * budget)


def make_course_curriculum(num_envs, device, seed, configuration):
    """Parse a legacy course or the explicit versioned jump-first wrapper."""
    if 'schedule' not in configuration:
        return SuperMimicCurriculum(num_envs, device, seed,
                                   CourseCurriculumConfig.from_dict(configuration))
    if set(configuration) != {'schedule', 'base'} or configuration['schedule'] != SCHEDULE:
        raise ValueError('Unknown native course schedule or fields')
    return JumpFirstCurriculum(num_envs, device, seed,
                              CourseCurriculumConfig.from_dict(configuration['base']))
