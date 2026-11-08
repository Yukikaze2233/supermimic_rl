"""Real-transition scheduling for the pinned native formal task interfaces.

Import this module after adding the authenticated upstream source to sys.path.
Budgets and mixture weights are algorithm design choices, not hardware data or
PPO iteration counts. Sampling records requested births, never physical success.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
import math

import torch

from transformer_rl.formal.lab_env.formal_curriculum import (
    FormalCurriculumCapabilities, FormalTaskCurriculum,
)
from transformer_rl.formal.tasks.command_pool import CommandContext
from transformer_rl.formal.tasks.curriculum import (
    INITIAL_DOMAIN, ContinuousTaskSampler, TaskDomain, TaskFamily,
    interpolate_domain,
)
from transformer_rl.formal.tasks.geometry import TerrainKind
from transformer_rl.formal.tasks.recovery import RECOVERY_NAMES


STAGE_NAMES = (
    "contract_world_pretrain", "flat_support", "dynamic_commands",
    "terrain_foundation", "authored_steps", "active_jump", "full_recovery",
    "combined_dr",
)
REQUIRED_RECOVERY = (
    "left_side_90", "right_side_90", "prone_belly_legs_above",
    "prone_back_legs_above",
)


@dataclass(frozen=True)
class CourseCurriculumConfig:
    """Runnable exposure design using the 41e full-vehicle task envelopes.

    ``source_schedule_coordinates`` is an explicit compatibility coordinate for
    the existing command/gimbal/episode-DR thresholds (2000/3000/4000/5000).
    It is interpolated from real exposure, not incremented by actor/PPO steps.
    The world-only actor freeze is enforced by TrainingRun, not this sampler.
    """

    stage_transition_budgets: tuple[int, ...] = (
        262144, 1048576, 2097152, 2097152, 4194304, 4194304, 4194304, 8388608,
    )
    # HEIGHT, MOTION, ROTATION, TERRAIN, REQUEST, RECOVERY.
    stage_weights: tuple[tuple[float, ...], ...] = (
        (.50, .40, .10, .00, .00, .00),
        (.60, .30, .10, .00, .00, .00),
        (.10, .60, .30, .00, .00, .00),
        (.10, .25, .10, .55, .00, .00),
        (.05, .20, .10, .65, .00, .00),
        (.10, .15, .05, .20, .50, .00),
        (.10, .10, .05, .15, .10, .50),
        (.10, .25, .15, .20, .15, .15),
    )
    source_schedule_coordinates: tuple[int, ...] = (
        0, 0, 1000, 5000, 10000, 18000, 24000, 30000, 35000,
    )
    domain: TaskDomain = field(default_factory=lambda: TaskDomain(height_m=(.23, .40)))
    initial_domain: TaskDomain = field(default_factory=lambda: INITIAL_DOMAIN)
    recovery_postures: tuple[str, ...] = REQUIRED_RECOVERY
    foundation_terrain_kinds: tuple[str, ...] = (
        "FLAT", "SLOPE", "ROUGH", "STEP_UP", "STEP_DOWN", "MATERIAL",
    )
    # Positive integer quotas in one deterministic terrain-family cycle.
    # Empty retains the original one-slot-per-kind order and source identity.
    foundation_terrain_kind_weights: dict[str, int] = field(default_factory=dict)
    full_terrain_kinds: tuple[str, ...] = tuple(kind.name for kind in TerrainKind)
    mixture_blend_fraction: float = .25
    jump_standing_fraction_of_stage: float = .25
    jump_full_motion_fraction_of_stage: float = .80
    moving_jump_min_fraction: float = .10
    # Stage over which the requested jump height range widens from the
    # initial to the full domain (jump-first course). "active_jump" grows the
    # upper height inside the jump stage so every later stage samples all heights.
    jump_height_stage: str = "combined_dr"
    # Optional (stage fraction, new upper height) pairs. The gated coordinate
    # stops one integer transition before each pair until the skill passes.
    # Empty retains the older continuously interpolated height curriculum.
    jump_height_gates: tuple[tuple[float, float], ...] = ()
    # Optional difficulty order for recovery births: {posture: first stage that
    # may draw it}. Postures absent from the map use the first stage. Empty keeps
    # every configured posture from the first recovery birth on.
    recovery_posture_stages: dict = field(default_factory=dict)
    # Reuse a periodic share of STEP_UP births for the two named authored
    # assets. The default retains the original one-in-five birth quota.
    foundation_authored_rehearsal: bool = False
    foundation_authored_rehearsal_every: int = 5
    # Six intermediate side-roll births replace alternate easy-pose slots;
    # original difficult-pose and task-family shares remain unchanged.
    recovery_roll_bridge: bool = False
    recovery_midpoint_bridge: bool = False

    def __post_init__(self):
        budgets = tuple(self.stage_transition_budgets)
        weights = tuple(tuple(row) for row in self.stage_weights)
        coordinates = tuple(self.source_schedule_coordinates)
        if (len(budgets) != len(STAGE_NAMES)
                or any(type(v) is not int or v <= 0 for v in budgets)):
            raise ValueError("Eight positive integer real-transition budgets are required")
        if (len(weights) != len(STAGE_NAMES)
                or any(len(row) != len(TaskFamily)
                       or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                              for v in row)
                       or abs(sum(row) - 1.) > 1e-9 for row in weights)):
            raise ValueError("Each stage requires six nonnegative weights summing to one")
        for stage, families in ((3, (3,)), (4, (3,)), (5, (4,)), (6, (5,)), (7, (3, 4, 5))):
            if any(weights[stage][family] <= 0 for family in families):
                raise ValueError("Required terrain, jump and recovery exposure cannot be disabled")
        if (len(coordinates) != len(STAGE_NAMES) + 1 or coordinates[0] != 0
                or any(type(v) is not int or v < 0 for v in coordinates)
                or any(a > b for a, b in zip(coordinates, coordinates[1:]))
                or coordinates[-1] < 5000):
            raise ValueError("Source schedule coordinates must monotonically cover all 5000 unlocks")
        if not isinstance(self.domain, TaskDomain) or not isinstance(self.initial_domain, TaskDomain):
            raise TypeError("Course domains must be parsed TaskDomain values")
        for name, end in asdict(self.domain).items():
            if isinstance(end, tuple):
                start = getattr(self.initial_domain, name)
                if not end[0] <= start[0] < start[1] <= end[1]:
                    raise ValueError("Initial task range must be inside final domain: " + name)
        for name, values, required, allowed in (
            ("recovery_postures", self.recovery_postures, REQUIRED_RECOVERY, RECOVERY_NAMES),
            ("foundation_terrain_kinds", self.foundation_terrain_kinds,
             ("FLAT", "SLOPE", "ROUGH", "STEP_UP", "STEP_DOWN"), TerrainKind.__members__),
            ("full_terrain_kinds", self.full_terrain_kinds,
             ("FLAT", "STEP_UP", "TWO_PLATFORMS"), TerrainKind.__members__),
        ):
            values = tuple(values)
            if (not values or len(set(values)) != len(values)
                    or any(value not in allowed for value in values)
                    or not set(required) <= set(values)):
                raise ValueError("Missing or invalid named course coverage: " + name)
            object.__setattr__(self, name, values)
        terrain_weights = self.foundation_terrain_kind_weights
        if (not isinstance(terrain_weights, dict)
                or (terrain_weights and set(terrain_weights) != set(self.foundation_terrain_kinds))
                or any(type(value) is not int or value < 1 for value in terrain_weights.values())):
            raise ValueError('Foundation terrain weights must give every named kind a positive integer quota')
        object.__setattr__(self, 'foundation_terrain_kind_weights', dict(terrain_weights))
        for name in ("mixture_blend_fraction", "jump_standing_fraction_of_stage",
                     "jump_full_motion_fraction_of_stage", "moving_jump_min_fraction"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(name + " must lie in (0, 1]")
        if self.jump_height_stage not in STAGE_NAMES:
            raise ValueError("jump_height_stage must name a course stage")
        height_gates = tuple(tuple(pair) for pair in self.jump_height_gates)
        previous_fraction, previous_height = 0., self.initial_domain.jump_m[1]
        for pair in height_gates:
            if (len(pair) != 2 or any(type(value) not in (float, int) or not math.isfinite(value)
                                      for value in pair)):
                raise ValueError("jump_height_gates requires finite (fraction, upper height) pairs")
            fraction, upper = pair
            if not (previous_fraction < fraction < 1 and previous_height < upper <= self.domain.jump_m[1]):
                raise ValueError("jump_height_gates must increase inside the stage and jump domain")
            previous_fraction, previous_height = fraction, upper
        if height_gates and height_gates[-1][1] != self.domain.jump_m[1]:
            raise ValueError("jump_height_gates must reach the full jump domain")
        object.__setattr__(self, "jump_height_gates", height_gates)
        stages = dict(self.recovery_posture_stages)
        if (any(name not in self.recovery_postures for name in stages)
                or any(stage not in STAGE_NAMES for stage in stages.values())):
            raise ValueError("recovery_posture_stages maps configured postures to course stages")
        object.__setattr__(self, "recovery_posture_stages", stages)
        if type(self.foundation_authored_rehearsal) is not bool:
            raise ValueError("foundation_authored_rehearsal must be boolean")
        if (type(self.foundation_authored_rehearsal_every) is not int
                or self.foundation_authored_rehearsal_every < 1):
            raise ValueError("foundation_authored_rehearsal_every must be a positive integer")
        if type(self.recovery_roll_bridge) is not bool:
            raise ValueError("recovery_roll_bridge must be boolean")
        if type(self.recovery_midpoint_bridge) is not bool:
            raise ValueError("recovery_midpoint_bridge must be boolean")
        if self.recovery_midpoint_bridge and not self.recovery_roll_bridge:
            raise ValueError("recovery_midpoint_bridge requires recovery_roll_bridge")
        if self.jump_standing_fraction_of_stage >= self.jump_full_motion_fraction_of_stage:
            raise ValueError("Jump moving exposure must follow standing exposure")
        object.__setattr__(self, "stage_transition_budgets", budgets)
        object.__setattr__(self, "stage_weights", weights)
        object.__setattr__(self, "source_schedule_coordinates", coordinates)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise TypeError("Course configuration must be a JSON object")
        value = dict(value)
        for key in ("domain", "initial_domain"):
            if key in value:
                value[key] = TaskDomain(**value[key])
        return cls(**value)

    def to_dict(self):
        result = json.loads(json.dumps(asdict(self)))
        # A disabled opt-in must preserve old checkpoint configuration identity.
        if not self.foundation_terrain_kind_weights:
            result.pop('foundation_terrain_kind_weights')
        if not self.foundation_authored_rehearsal:
            result.pop("foundation_authored_rehearsal")
        # Existing enabled sources also omit the original fixed period.
        if self.foundation_authored_rehearsal_every == 5:
            result.pop("foundation_authored_rehearsal_every")
        if not self.recovery_roll_bridge:
            result.pop("recovery_roll_bridge")
        if not self.recovery_midpoint_bridge:
            result.pop("recovery_midpoint_bridge")
        return result


class _TransitionBudget:
    """Small facade for upstream readers; total_updates is a mapped coordinate."""

    def __init__(self, config, real_transitions):
        self.promotion = "exposure"
        self.stage_names = STAGE_NAMES
        self.stage_budgets = config.stage_transition_budgets
        self.consumed = []
        remaining = real_transitions
        for budget in self.stage_budgets:
            amount = min(remaining, budget)
            self.consumed.append(amount)
            remaining -= amount
        self.stage = min(sum(a == b for a, b in zip(self.consumed, self.stage_budgets)),
                         len(STAGE_NAMES) - 1)
        complete = real_transitions >= sum(self.stage_budgets)
        self.status = "training_budget_complete" if complete else "running"
        lo, hi = config.source_schedule_coordinates[self.stage:self.stage + 2]
        self.total_updates = lo + (hi - lo) * self.consumed[self.stage] // self.stage_budgets[self.stage]


class SuperMimicCurriculum(FormalTaskCurriculum):
    """Eight-stage native sampler whose only scheduling clock is real data.

    Call set_progress after real collection (also during world-only pretraining).
    actor_updates is a separately persisted diagnostic. Birth counts are requests
    issued to the native task adapter, not completed attempts or terrain success.
    """

    _FORMAT = "supermimic_real_transition_curriculum_v1"

    def __init__(self, num_envs, device="cpu", seed=0, config=None):
        if config is None:
            config = CourseCurriculumConfig()
        elif isinstance(config, dict):
            config = CourseCurriculumConfig.from_dict(config)
        if not isinstance(config, CourseCurriculumConfig):
            raise TypeError("Expected CourseCurriculumConfig or its JSON object")
        super().__init__(
            num_envs, device, seed=seed, promotion="exposure",
            capabilities=FormalCurriculumCapabilities(terrain=True, requests=True, recovery=True),
            domain=config.domain, initial_domain=config.initial_domain,
            include_jump=True, continuous=True,
        )
        self.config = config
        self.stage_names = STAGE_NAMES
        self.stage_budgets = config.stage_transition_budgets
        self.stage_weights = config.stage_weights
        self.sampler = ContinuousTaskSampler(
            seed, device=device, domain=self.domain,
            stage_weights=self.stage_weights, continuous=True,
        )
        self.real_transitions = 0
        self.actor_updates = 0
        self.budget = _TransitionBudget(config, 0)
        self.sample_counts = {
            "family": [0] * len(TaskFamily), "terrain": [0] * len(TerrainKind),
            "recovery_posture": [0] * len(RECOVERY_NAMES), "jump_motion": [0, 0],
        }
        self._recovery_birth_variant = None
        self._pending_recovery_variant = None
        if config.recovery_roll_bridge:
            from .recovery_bridge import variant_names

            self.sample_counts['recovery_posture'] = [0] * len(
                variant_names(midpoints=config.recovery_midpoint_bridge))
            self._recovery_birth_variant = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        # A reset samples then routes within the same call. This transient row
        # receipt is never replayed across physical reset/checkpoint restore;
        # the persisted terrain ordinal determines the next draw exactly.
        self._foundation_authored_rows = (torch.zeros(num_envs, dtype=torch.bool, device=device)
                                         if config.foundation_authored_rehearsal else None)
        self._pending_foundation_authored = None
        quotas = config.foundation_terrain_kind_weights
        # Round-robin quota slots spread repeated kinds across the cycle.
        # The unchanged terrain-family ordinal makes partial reset batches
        # and resumed samplers use the same sequence without random draws.
        cycle = [int(TerrainKind[name])
                 for slot in range(max(quotas.values(), default=1))
                 for name in config.foundation_terrain_kinds if slot < quotas.get(name, 1)]
        self._foundation_terrain_pool = torch.tensor(cycle, dtype=torch.long, device=device)
        step_up = self._foundation_terrain_pool == int(TerrainKind.STEP_UP)
        self._foundation_step_up_count = cycle.count(int(TerrainKind.STEP_UP))
        self._foundation_step_up_ordinal = step_up.to(torch.long).cumsum(0) - 1

    def set_progress(self, real_transitions, actor_updates):
        for name, value in (("real_transitions", real_transitions), ("actor_updates", actor_updates)):
            if type(value) is not int or value < getattr(self, name):
                raise ValueError(name + " must be a monotonic nonnegative integer")
        self.real_transitions, self.actor_updates = real_transitions, actor_updates
        self.budget = _TransitionBudget(self.config, real_transitions)

    def begin_update(self):
        raise RuntimeError("Use set_progress(real_transitions, actor_updates); optimizer updates are not exposure")

    @property
    def weights(self):
        target = self.stage_weights[self.stage]
        if self.stage == 0:
            return target
        fraction = self._smooth_fraction(
            self.budget.consumed[self.stage],
            self.config.mixture_blend_fraction * self.stage_budgets[self.stage],
        )
        return tuple(a + (b - a) * fraction
                     for a, b in zip(self.stage_weights[self.stage - 1], target))

    def _effective_domain(self):
        command_fraction = self._smooth_fraction(self.real_transitions, sum(self.stage_budgets[:3]))
        domain = interpolate_domain(self.initial_domain, self.domain, command_fraction)
        terrain_fraction = self._smooth_fraction(
            max(0, self.real_transitions - sum(self.stage_budgets[:3])), self.stage_budgets[3],
        )
        terrain = interpolate_domain(self.initial_domain, self.domain, terrain_fraction)
        jump_fraction = self._smooth_fraction(
            max(0, self.real_transitions - sum(self.stage_budgets[:5])), self.stage_budgets[5],
        )
        jump = interpolate_domain(self.initial_domain, self.domain, jump_fraction)
        return replace(domain, slope_rad=terrain.slope_rad, roughness_m=terrain.roughness_m,
                       step_up_m=terrain.step_up_m, step_down_m=terrain.step_down_m,
                       jump_m=jump.jump_m)

    def terrain_step_up_cap_m(self):
        return 0. if self.stage < 3 else float(self._effective_domain().step_up_m[1])

    def terrain_level_cap(self, levels):
        if type(levels) is not int or levels < 1:
            raise ValueError("The actual positive atlas level count is required")
        if self.stage < 3:
            return 0
        if self.stage > 3:
            return levels - 1
        fraction = self._smooth_fraction(self.budget.consumed[3], self.stage_budgets[3])
        return int((levels - 1) * fraction)

    def _request_motion_scale(self):
        if self.stage < 5:
            return 0.
        if self.stage > 5:
            return 1.
        start = self.config.jump_standing_fraction_of_stage
        end = self.config.jump_full_motion_fraction_of_stage
        return self._smooth_fraction(self.budget.consumed[5] - start * self.stage_budgets[5],
                                     (end - start) * self.stage_budgets[5])

    def command_context(self):
        return CommandContext(self._effective_domain(), self.budget.total_updates,
                              self._request_motion_scale())

    def recovery_posture_pool(self) -> tuple[str, ...]:
        """Configured postures whose first stage has been reached (all postures if unscheduled)."""
        first = self.config.recovery_posture_stages
        names = tuple(name for name in self.config.recovery_postures
                      if self.stage_names.index(first.get(name, self.stage_names[0])) <= self.stage)
        if not names:
            raise ValueError("Recovery births need at least one posture available at this stage")
        return names

    def _condition_task_difficulty(self, draw):
        posture, terrain = draw.recovery_posture.clone(), draw.terrain_kind.clone()
        command, nominal = draw.requested_command.clone(), draw.nominal.clone()
        trajectory, step = draw.trajectory_kind.clone(), draw.step_up_m.clone()
        rehearsal = (torch.zeros_like(draw.family, dtype=torch.bool)
                     if self.config.foundation_authored_rehearsal else None)
        variants = (torch.full_like(draw.family, -1) if self.config.recovery_roll_bridge else None)
        for family in (TaskFamily.RECOVERY, TaskFamily.TERRAIN, TaskFamily.REQUEST):
            rows = (draw.family == int(family)).nonzero().flatten()
            if not len(rows):
                continue
            offset = self.sample_counts["family"][int(family)]
            ordinal = torch.arange(len(rows), device=self.device) + offset
            if family == TaskFamily.RECOVERY:
                names = self.recovery_posture_pool()
                pool = posture.new_tensor([RECOVERY_NAMES.index(name) for name in names])
                posture[rows] = pool[ordinal % len(pool)]
                if variants is not None:
                    from .recovery_bridge import select_variants

                    posture[rows], variants[rows] = select_variants(
                        ordinal, posture[rows], names, midpoints=self.config.recovery_midpoint_bridge)
                nominal[rows], trajectory[rows] = False, 0
            elif family == TaskFamily.TERRAIN:
                foundation = self.stage < self.stage_names.index('authored_steps')
                pool = (self._foundation_terrain_pool if foundation else terrain.new_tensor(
                    [int(TerrainKind[name]) for name in self.config.full_terrain_kinds]))
                terrain[rows] = pool[ordinal % len(pool)]
                # Half the stair cycles request the declared endpoint exactly.
                # The terrain router must still place the actual authored mesh.
                endpoint = ((ordinal // len(pool)) % 2 == 0) & torch.isin(
                    terrain[rows], terrain.new_tensor([
                        int(TerrainKind.STEP_UP), int(TerrainKind.STAIRS),
                        int(TerrainKind.TWO_PLATFORMS),
                    ]))
                if self.stage >= self.stage_names.index('authored_steps'):
                    step[rows[endpoint]] = self.domain.step_up_m[1]
                if rehearsal is not None and self.stage_name == 'terrain_foundation':
                    # Select a periodic share of the existing STEP_UP slots.
                    # Divide by the same period to alternate the two assets
                    # across batches and restores without extra random draws.
                    # Count STEP_UP slots, not complete mixed-terrain cycles:
                    # weighted cycles may contain several slots of this kind.
                    cycle = ((ordinal // len(pool)) * self._foundation_step_up_count
                             + self._foundation_step_up_ordinal[ordinal % len(pool)])
                    every = self.config.foundation_authored_rehearsal_every
                    selected = ((terrain[rows] == int(TerrainKind.STEP_UP)) & (cycle % every == 0))
                    chosen = rows[selected]
                    terrain[chosen] = torch.where((cycle[selected] // every) % 2 == 0,
                                                 int(TerrainKind.STEP_UP), int(TerrainKind.TWO_PLATFORMS))
                    step[chosen] = self.domain.step_up_m[1]
                    rehearsal[chosen] = True
            else:
                scale = self._request_motion_scale()
                moving = (ordinal % 2 == 1) & (scale > 0)
                command[rows, :3] *= scale
                command[rows[~moving], :3] = 0
                # Moving jumps go straight: the native success needs a quiet body
                # angular speed (<= 0.5 rad/s), which a yaw command would prevent.
                command[rows, 2] = 0
                motion_rows = rows[moving]
                if len(motion_rows):
                    domain = self._effective_domain()
                    low = min(-domain.forward_m_s[0], domain.forward_m_s[1])
                    floor = low * self.config.moving_jump_min_fraction * scale
                    vx = command[motion_rows, 0]
                    command[motion_rows, 0] = torch.where(vx < 0, -1., 1.) * vx.abs().clamp_min(floor)
                nominal[rows], trajectory[rows] = False, 0
                self.sample_counts["jump_motion"][0] += int((~moving).sum())
                self.sample_counts["jump_motion"][1] += int(moving.sum())
        self._pending_foundation_authored = rehearsal
        self._pending_recovery_variant = variants
        return replace(draw, recovery_posture=posture, terrain_kind=terrain,
                       requested_command=command, nominal=nominal,
                       trajectory_kind=trajectory, step_up_m=step)

    def sample(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device)
        if (ids.dtype not in (torch.int32, torch.int64) or ids.ndim != 1
                or not len(ids) or len(ids) > self.num_envs
                or bool(((ids < 0) | (ids >= self.num_envs)).any())
                or ids.unique().numel() != len(ids)):
            raise ValueError("Reset IDs must be unique in-range integer environment indices")
        draw = super().sample(ids)
        if self._foundation_authored_rows is not None:
            self._foundation_authored_rows[ids] = self._pending_foundation_authored
            self._pending_foundation_authored = None
        if self._recovery_birth_variant is not None:
            self._recovery_birth_variant[ids] = self._pending_recovery_variant
        for name, values in (
            ("family", draw.family),
            ("terrain", draw.terrain_kind[draw.family == int(TaskFamily.TERRAIN)]),
            ("recovery_posture", (draw.recovery_posture if self._pending_recovery_variant is None
                                  else self._pending_recovery_variant)[draw.family == int(TaskFamily.RECOVERY)]),
        ):
            counts = torch.bincount(values, minlength=len(self.sample_counts[name])).cpu().tolist()
            self.sample_counts[name] = [a + b for a, b in zip(self.sample_counts[name], counts)]
        self._pending_recovery_variant = None
        return draw

    def recovery_birth_variants(self, ids):
        """Actual latest birth labels, distinct from reused native base-pose IDs."""
        if self._recovery_birth_variant is None:
            raise ValueError('Recovery roll bridge is disabled')
        labels = self._recovery_birth_variant[ids]
        if (labels < 0).any():
            raise ValueError('A recovery birth must have a current sampler label')
        return labels.clone()

    def recovery_birth_samples(self):
        """Named cumulative reset draws, not measured placements or successes."""
        names = RECOVERY_NAMES
        if self.config.recovery_roll_bridge:
            from .recovery_bridge import variant_names

            names = variant_names(midpoints=self.config.recovery_midpoint_bridge)
        return dict(zip(names, self.sample_counts['recovery_posture'], strict=True))

    def foundation_authored_rehearsal_receipt(self):
        if not self.config.foundation_authored_rehearsal:
            return None
        result = {
            "contract": "foundation_authored_birth_rehearsal_v1",
            "stage": self.stage_names.index('terrain_foundation'),
            "source_slots": "STEP_UP in the configured foundation terrain cycle",
            "slot_period": self.config.foundation_authored_rehearsal_every,
            "asset_cycle": ["road_step", "central_steps"],
            "requested_step_up_m": self.domain.step_up_m[1],
            "selection": "persisted terrain-family ordinal; no extra random draws",
            "other_routes": "native routing and its 20-percent ceiling exploration unchanged",
            "cap_scope": "selected authored births only; ordinary level and step caps unchanged",
        }
        if self.config.foundation_terrain_kind_weights:
            result['terrain_kind_weights'] = dict(self.config.foundation_terrain_kind_weights)
            result['selection'] = 'persisted terrain-family ordinal with STEP_UP-local cadence; no extra random draws'
        return result

    def foundation_authored_rehearsal_rows(self, ids):
        """Rows certified by their latest birth draw before synchronous routing."""
        if self._foundation_authored_rows is None:
            raise ValueError("Foundation authored rehearsal is disabled")
        if self.stage_name != 'terrain_foundation':
            return torch.zeros(len(ids), dtype=torch.bool, device=self.device)
        return self._foundation_authored_rows[ids].clone()

    def state_dict(self):
        return {
            "format": self._FORMAT, "num_envs": self.num_envs,
            "config": self.config.to_dict(), "real_transitions": self.real_transitions,
            "actor_updates": self.actor_updates, "sampler": self.sampler.state_dict(),
            "sample_counts": {key: list(value) for key, value in self.sample_counts.items()},
        }

    def load_state_dict(self, state):
        expected = self.state_dict()
        if (not isinstance(state, dict) or state.keys() != expected.keys()
                or any(state[key] != expected[key] for key in ("format", "num_envs", "config"))):
            raise ValueError("Course checkpoint identity/configuration differs")
        counts = state["sample_counts"]
        if (not isinstance(counts, dict) or counts.keys() != self.sample_counts.keys()
                or any(not isinstance(counts[key], list) or len(counts[key]) != len(value)
                       or any(type(n) is not int or n < 0 for n in counts[key])
                       for key, value in self.sample_counts.items())):
            raise ValueError("Course checkpoint sampling ledger is invalid")
        if any(sum(counts[key]) != counts["family"][int(family)] for key, family in (
            ("terrain", TaskFamily.TERRAIN), ("recovery_posture", TaskFamily.RECOVERY),
            ("jump_motion", TaskFamily.REQUEST),
        )):
            raise ValueError("Course checkpoint sampling counts disagree")
        prepared = type(self)(self.num_envs, self.device, config=self.config)
        prepared.set_progress(state["real_transitions"], state["actor_updates"])
        prepared.sampler.load_state_dict(state["sampler"])
        self.real_transitions, self.actor_updates = prepared.real_transitions, prepared.actor_updates
        self.budget, self.sampler = prepared.budget, prepared.sampler
        self.sample_counts = {key: list(value) for key, value in counts.items()}
        if self._foundation_authored_rows is not None:
            self._foundation_authored_rows.zero_()
            self._pending_foundation_authored = None
        if self._recovery_birth_variant is not None:
            self._recovery_birth_variant.fill_(-1)
            self._pending_recovery_variant = None
