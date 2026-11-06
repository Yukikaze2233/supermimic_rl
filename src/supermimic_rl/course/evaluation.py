"""Fixed native course requests and endpoint-based evaluation accounting.

Scenarios select existing task/command interfaces. Physics, command admission,
reward and physical success criteria remain upstream. Parameter values below
come from the instantiated course domain, command pool and gimbal contract.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch

from ..compact_state import named_field
from .jump_evaluation import JumpEvaluation
from .recovery_evaluation import RecoveryEvaluation

# Radiant 0bf3456 clips before MAE. Keep these secondary summaries separate
# from raw error and document our native coordinates/sample-weighted reduction.
CLIPPED_MAE_LIMITS = (1., .8, .15)  # Forward m/s, yaw rad/s, support height m.


@dataclass(frozen=True)
class CourseEvaluationConfig:
    evaluation_stage: str = "combined_dr"
    scenarios: tuple[str, ...] = ()
    family: str | None = None
    seed: int = 0
    recovery_qualification: bool = False

    def __post_init__(self):
        if (not isinstance(self.evaluation_stage, str)
                or type(self.recovery_qualification) is not bool
                or type(self.seed) is not int or not 0 <= self.seed < 2**63
                or not isinstance(self.scenarios, (tuple, list))
                or any(not isinstance(name, str) for name in self.scenarios)
                or len(set(self.scenarios)) != len(self.scenarios)
                or self.family is not None and not isinstance(self.family, str)):
            raise ValueError("Invalid fixed course evaluation selection")
        object.__setattr__(self, "scenarios", tuple(self.scenarios))


class FixedCourseScenarios:
    """Assign named scenarios by environment ID, retaining them across resets."""

    def __init__(self, adapter, config):
        from transformer_rl.formal.tasks.curriculum import TaskFamily
        from transformer_rl.formal.tasks.geometry import TerrainKind
        from transformer_rl.formal.tasks.recovery import RECOVERY_NAMES

        self.adapter, self.config = adapter, config
        self.env, self.device = adapter.env, adapter.device
        self.family_type, self.terrain_type = TaskFamily, TerrainKind
        self.posture_names = RECOVERY_NAMES
        stage_names = adapter.curriculum.stage_names
        if config.evaluation_stage not in stage_names:
            raise ValueError("Unknown formal evaluation stage")
        self.curriculum = type(adapter.curriculum)(adapter.num_envs, adapter.device,
                                                  config.seed, adapter.curriculum.config)
        stage = stage_names.index(config.evaluation_stage)
        # The endpoint belongs to this stage, not the following stage. This is
        # an isolated evaluation schedule, never additional training exposure.
        exposure = sum(self.curriculum.stage_budgets[:stage + 1]) - 1
        self.curriculum.set_progress(exposure, 0)
        self.domain = self.curriculum._effective_domain()
        self.pool = self.env._terrain_course.tasks.command_pool
        if self.pool is None:
            raise ValueError("Fixed evaluation requires the connected native command pool")
        bank = self._bank()
        names = list(config.scenarios or bank)
        if any(name not in bank for name in names):
            raise ValueError("Unknown fixed scenario: " + repr(set(names) - bank.keys()))
        if config.family is not None:
            family = config.family.upper()
            if family not in TaskFamily.__members__:
                raise ValueError("Unknown requested evaluation family")
            names = [name for name in names if bank[name]["family"] == family]
        if not names:
            raise ValueError("Evaluation selection contains no scenarios")
        self.scenarios = {name: bank[name] for name in names}
        self.names = tuple(names)
        self.row_names = tuple(names[row % len(names)] for row in range(adapter.num_envs))
        self.births = torch.zeros(adapter.num_envs, dtype=torch.long, device=self.device)
        self.command_source = torch.zeros_like(self.births)
        self.qualification_clock = None
        self.qualification_commands = None
        if config.recovery_qualification:
            recovery = self.env._terrain_course.recovery
            self.qualification_clock = torch.zeros(adapter.num_envs, dtype=torch.float64, device=self.device)
            self.qualification_commands = torch.tensor(
                [self.scenarios[name]['commands'][0] for name in self.row_names], device=self.device)
            self.qualification_rows = torch.tensor(
                [self.scenarios[name]['family'] == 'RECOVERY' for name in self.row_names],
                dtype=torch.bool, device=self.device)
            self.qualification_timeout_s = recovery.config.timeout_s
        self._restore = []
        self.fixed_tiles = self._fixed_tiles()

    def _fixed_tiles(self):
        """Resolve requested cases to actual atlas geometry, without proficiency routing."""
        router = self.env._terrain_course.router
        manifest = getattr(router, 'manifest', None)
        if manifest is None:
            return {}  # CPU command-only fixtures do not claim terrain coverage.
        tiles = manifest['tiles']
        result = {}
        for name, case in self.scenarios.items():
            authored = {'authored_single_step': 'road_step', 'authored_two_platforms': 'central_steps'}.get(name)
            candidates = [index for index, tile in enumerate(tiles)
                if tile['name'] == authored] if authored else [index for index, tile in enumerate(tiles)
                if tile['kind'] == int(self.terrain_type[case['terrain']])
                and tile['name'] not in ('road_step', 'central_steps')]
            key, target = {
                'SLOPE': ('slope_rad', self.domain.slope_rad[1]),
                'ROUGH': ('roughness_m', self.domain.roughness_m[1]),
                'STEP_UP': ('step_height_m', case['step_up_m']),
            }.get(case['terrain'], (None, None))
            if key is not None and not authored:
                candidates = [index for index in candidates
                    if abs(tiles[index].get('parameters', {}).get(key, 0.)) <= abs(target) + 1e-7
                    and tiles[index].get('parameters', {}).get(key, 0.) * target >= 0]
            if not candidates:
                raise ValueError('Pinned atlas cannot instantiate fixed scenario: ' + name)
            selected = min(candidates, key=lambda index: (0. if key is None or authored else
                abs(tiles[index].get('parameters', {}).get(key, 0.) - target), index))
            result[name] = selected
            if authored:
                case['step_up_m'] = tiles[selected]['parameters']['step_height_m']
        return result

    def _bank(self):
        d, p = self.domain, self.pool.config
        h, vx = d.nominal_height_m, d.forward_m_s[1]
        command = lambda forward=0., yaw=0., height=h: [forward, 0., yaw, height]
        approach = p.high_approach_forward_m_s[1]
        bank = {}

        def add(name, family, commands, *, terrain="FLAT", posture=None, gimbal="hold", step=0.,
                jump=None, rotation_sign=1):
            bank[name] = {"family": family, "terrain": terrain, "commands": commands,
                          "posture": posture, "gimbal": gimbal, "step_up_m": step,
                          "jump_height_m": d.jump_m[0] if jump is None else jump,
                          "rotation_sign": rotation_sign}

        add("authored_single_step", "TERRAIN", [command(approach)], terrain="STEP_UP", step=d.step_up_m[1])
        add("authored_two_platforms", "TERRAIN", [command(approach)], terrain="TWO_PLATFORMS", step=d.step_up_m[1])
        levels = (("low", d.jump_m[0]),
                  ("intermediate", max(d.jump_m[0], min(d.jump_m[1], self.curriculum.config.initial_domain.jump_m[1]))),
                  ("high", d.jump_m[1]))
        for motion, forward in (("standing", 0.), ("moving", p.approach_forward_m_s[0])):
            for level, rise in levels:
                add(f"jump_{motion}_{level}", "REQUEST", [command(forward)], jump=rise)
        for name in self.curriculum.config.recovery_postures:
            add("recovery_" + name, "RECOVERY", [command()], posture=name)
        for direction, sign in (("positive", 1), ("negative", -1)):
            add("spin_translation_" + direction, "MOTION",
                [command(self.env._gimbal_course.config.translation_speed_m_s[1])],
                gimbal="spin_translation", rotation_sign=sign)
        add("height_low", "HEIGHT", [command(height=d.height_m[0])])
        add("height_high", "HEIGHT", [command(height=d.height_m[1])])
        add("forward_speed", "MOTION", [command(vx)])
        add("braking", "MOTION", [command(vx), command()])
        add("fast_turning", "ROTATION", [command(vx, p.normal_yaw_rad_s)])
        add("slope", "TERRAIN", [command(p.approach_forward_m_s[0])], terrain="SLOPE")
        add("rough", "TERRAIN", [command(p.approach_forward_m_s[0])], terrain="ROUGH")
        add("stand", "HEIGHT", [command()])
        add("reverse_speed", "MOTION", [command(d.forward_m_s[0])])
        add("in_place_rotation", "ROTATION", [command(yaw=p.spin_mid_rad_s[0])])
        for direction, sign in (("positive", 1), ("negative", -1)):
            add("gimbal_yaw_sweep_" + direction, "MOTION", [command()],
                gimbal="sweep", rotation_sign=sign)
        router = self.env._terrain_course.router
        authored = getattr(router, 'authored', torch.zeros_like(router.kinds, dtype=torch.bool))
        positive = router.step_table[(router.kinds == int(self.terrain_type.STEP_UP))
                                     & (router.step_table > 0) & ~authored]
        if positive.numel():
            add("low_obstacle", "TERRAIN", [command(p.approach_forward_m_s[0])],
                terrain="STEP_UP", step=float(positive.min()))
        return bank

    def _selected(self, ids):
        return [self.scenarios[self.row_names[row]] for row in ids.cpu().tolist()]

    def sample(self, env_ids):
        from transformer_rl.formal.tasks.curriculum import TaskDraw

        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        cases = self._selected(ids)
        self.births[ids] += 1
        scalar = lambda value: torch.full((len(ids),), value, device=self.device)
        integer = lambda values: torch.tensor(values, dtype=torch.long, device=self.device)
        return TaskDraw(
            family=integer([int(self.family_type[c["family"]]) for c in cases]),
            stratum=integer([0] * len(ids)),
            requested_command=torch.tensor([c["commands"][0] for c in cases], device=self.device),
            jump_height_m=torch.tensor([c["jump_height_m"] for c in cases], device=self.device),
            slope_rad=scalar(self.domain.slope_rad[1]), roughness_m=scalar(self.domain.roughness_m[1]),
            step_up_m=torch.tensor([c["step_up_m"] for c in cases], device=self.device),
            step_down_m=scalar(self.domain.step_down_m[1]),
            terrain_kind=integer([int(self.terrain_type[c["terrain"]]) for c in cases]),
            recovery_posture=integer([0 if c["posture"] is None else self.posture_names.index(c["posture"])
                                     for c in cases]),
            trajectory_kind=integer([0] * len(ids)),
            nominal=torch.zeros(len(ids), dtype=torch.bool, device=self.device),
            rotation_mode=integer([0] * len(ids)),
        )

    def draw_segments(self, rows):
        from transformer_rl.formal.tasks.command_pool import SegmentDraw, SegmentKind

        duration = self.pool.config.segment_s[0]
        elapsed = self.env._terrain_course.tasks.elapsed[rows].cpu().tolist()
        commands, kinds = [], []
        for case, clock in zip(self._selected(rows), elapsed):
            sequence = case["commands"]
            command = sequence[int(math.floor((clock + 1e-5) / duration)) % len(sequence)]
            commands.append(command)
            still = command[0] == command[2] == 0 and case["gimbal"] != "spin_translation"
            kinds.append(int(SegmentKind.STAND if still else SegmentKind.NORMAL))
        n, device = len(rows), self.device
        self.command_source[rows] = 1
        return SegmentDraw(torch.tensor(kinds, device=device), torch.tensor(commands, device=device),
                           torch.full((n,), duration, device=device),
                           torch.zeros(n, dtype=torch.long, device=device),
                           torch.zeros(n, dtype=torch.bool, device=device), torch.zeros(n, device=device))

    def draw_gimbal(self, ids, world_angles, eligible, update):
        from transformer_rl.formal.control.gimbal import GimbalMode

        course, c = self.env._gimbal_course, self.env._gimbal_course.config
        for row, case in zip(ids.cpu().tolist(), self._selected(ids)):
            special = case["gimbal"] == "spin_translation" and bool(eligible[row])
            sweep = case["gimbal"] == "sweep"
            mode = GimbalMode.WORLD_HEADING if special else GimbalMode.YAW_RATE if sweep else GimbalMode.BODY_FIXED
            course.mode[row] = int(mode)
            course.special[row] = special
            course.target[row, 0] = world_angles[row, 0] if mode != GimbalMode.BODY_FIXED else 0.
            # Opposite endpoints exist from birth, even if episodes end before
            # the first segment boundary. Repeated rows of the 27-case suite
            # also receive opposite endpoints because its length is odd.
            segment = int(self.env._terrain_course.tasks.elapsed[row] / self.pool.config.segment_s[0])
            course.target[row, 1] = c.pitch_target_rad[(row + segment) % 2]
            course.rate[row] = c.yaw_rate_rad_s[int(case["rotation_sign"] > 0)] if sweep else 0.
            course.speed[row] = c.translation_speed_m_s[1] if special else 0.
            course.direction[row] = 0.
            # Upstream samples a separate +/- sign for this magnitude range.
            course.chassis_rate[row] = case["rotation_sign"] * c.chassis_spin_rad_s[0] if special else 0.
            course.remaining_s[row] = self.pool.config.segment_s[0]
            course.sample_counts[int(mode)] += 1
            course.sample_counts[3] += int(special)

    def install(self):
        # Reuse the already installed sampler, preparation and sensor objects.
        # Calling env.set_task_sampler again would replace recovery components.
        tasks = self.env._terrain_course.tasks
        bridge = getattr(self.env, 'supermimic_recovery_bridge', None)
        if bridge is not None:
            # Every installed fixed plan uses the original endpoints,
            # independently of how the plan was created.
            self._install(bridge, 'evaluation', True)
        original_tracking = tasks.begin_tracking

        def begin_tracking(mask):
            result = original_tracking(mask)
            # Preserve the native ground-birth/recovery handoff and its RNG.
            # A scheduled segment becomes active only on the next real draw.
            self.command_source[mask] = 2
            return result

        self._install(self.curriculum, 'sample', self.sample)
        self._install(self.env._task_sampler, 'curriculum', self.curriculum)
        self._install(self.adapter, 'curriculum', self.curriculum)
        self._install(tasks, '_draw_segments', self.draw_segments)
        self._install(tasks, 'begin_tracking', begin_tracking)
        self._install(self.env._gimbal_course, '_draw', self.draw_gimbal)
        if self.config.recovery_qualification:
            original_step = tasks.stream.step

            def stationary_request(requested):
                # Override the external target, never the admitted command or
                # actuator path. Native handoff still draws its ordinary RNG.
                hold = self.qualification_rows & (
                    self.qualification_clock < self.qualification_timeout_s - 1e-9)
                target = torch.where(hold[:, None], self.qualification_commands, requested)
                result = original_step(target)
                self.command_source[hold] = 1
                return result

            self._install(tasks.stream, 'step', stationary_request)
        if self.fixed_tiles:
            router = self.env._terrain_course.router
            original_reset = router.reset

            def reset(ids, kind, **kwargs):
                # Keep native episode bookkeeping, but remove training-only
                # ceiling exploration and pin the evaluation collision tile.
                original_reset(ids, kind, **{**kwargs, 'step_up_cap_m': None})
                ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
                chosen = torch.tensor([self.fixed_tiles[self.row_names[row]] for row in ids.cpu().tolist()],
                                      dtype=torch.long, device=self.device)
                router.tile[ids] = chosen
                router.origins[ids] = router.origins_table[chosen]
                return router.origins[ids], router.spawn_z[chosen]

            self._install(router, 'reset', reset)

    def _install(self, owner, name, value):
        self._restore.append((owner, name, owner.__dict__.get(name), name in owner.__dict__))
        setattr(owner, name, value)

    def close(self):
        for owner, name, value, existed in reversed(self._restore):
            if existed:
                setattr(owner, name, value)
            else:
                delattr(owner, name)
        self._restore.clear()

    def description(self):
        selection = asdict(self.config)
        if not self.config.recovery_qualification:
            selection.pop('recovery_qualification')
        result = {"selection": selection, "scenarios": self.scenarios,
                "fixed_atlas_tiles": self.fixed_tiles,
                "resolved_curriculum": {"stage": self.curriculum.stage,
                    "stage_name": self.curriculum.stage_name,
                    "source_schedule_coordinate": self.curriculum.budget.total_updates,
                    "evaluation_exposure_coordinate": self.curriculum.real_transitions,
                    "coordinate_semantics": "isolated evaluation difficulty; not claimed training samples"},
                "assigned_rows": {name: self.row_names.count(name) for name in self.names},
                "row_assignment": [{"env_id": row, "scenario": name,
                                    "initial_pitch_target_rad": self.env._gimbal_course.config.pitch_target_rad[row % 2]}
                                   for row, name in enumerate(self.row_names)],
                "unassigned_scenarios": [name for name in self.names if name not in self.row_names],
                "parameters": {"task_domain": asdict(self.domain),
                               "command_pool": asdict(self.pool.config),
                               "gimbal": asdict(self.env._gimbal_course.config)},
                "parameter_source": "instantiated pinned native contracts; jump low/high=effective domain endpoints, intermediate=initial-domain upper endpoint capped by effective domain; not success thresholds",
                "placement": "requested terrain kind/height; actual routed tiles and declared collision profiles are reported separately",
                "randomness": "scheduled command targets and collision tiles fixed; native birth/recovery handoff commands, request timing, birth, dynamics and sensor DR retained; training terrain coverage/proficiency routing excluded"}
        if self.config.recovery_qualification:
            result['randomness'] = ('Native birth, handoff and segment RNG draws remain unchanged; '
                'recovery requests inside the qualification budget use the pre-authored stationary override. '
                'Other fixed scenarios, dynamics and sensor DR retain their existing evaluation behavior.')
            result['recovery_qualification'] = {
                'request': 'Pre-authored vx=vy=yaw_rate=0 and original recovery height; override before native command admission/slew, independently of success or handoff.',
                'budget_clock': 'Policy-owned physical ticks from each recovery birth; successful certification does not release the fixed command early.',
                'timeout_s': self.qualification_timeout_s,
                'randomness': 'Native handoff/segment RNG and underlying trajectories remain unchanged; only admitted requests inside the qualification window are overridden.'}
        return result


class CourseEvaluation:
    """Aggregate actual pre-reset endpoints, distinct from requested coverage."""

    def __init__(self, adapter, configuration=None):
        self.adapter = adapter
        self.config = (configuration if isinstance(configuration, CourseEvaluationConfig)
                       else CourseEvaluationConfig(**(configuration or {})))
        self.plan, self.schema, self.buckets = None, None, {}
        self.phase_buckets = {}
        self.recovery_phase_buckets = {}
        self.recovery_evidence = None
        self.jump_evidence = None
        self._hook_restore = []
        self._authored_pending = None
        self._active = False

    def begin(self, schema):
        self.schema = schema
        self.plan = FixedCourseScenarios(self.adapter, self.config)
        try:
            self.plan.install()
            self.adapter.reset()
            self.before = self.adapter.env._terrain_course.metrics()
            self.buckets = {}
            self.phase_buckets = {}
            self.recovery_phase_buckets = {}
            self._install_authored_evidence()
            if getattr(self.adapter.env._terrain_course, 'recovery', None) is not None:
                options = ({'qualification_commands': self.plan.qualification_commands,
                            'qualification_clock': self.plan.qualification_clock}
                           if self.config.recovery_qualification else {})
                self.recovery_evidence = RecoveryEvaluation(self.adapter.env, self.plan.row_names, **options)
                self.recovery_evidence.install()
            tasks = getattr(self.adapter.env._terrain_course, 'tasks', None)
            if getattr(tasks, 'protocol', None) is not None:
                self.jump_evidence = JumpEvaluation(self.adapter.env, self.plan.row_names)
                self.jump_evidence.install()
        except Exception:
            self.close()
            raise
        self._active = True
        self.adapter.evaluation_end = self.close

    def sample(self):
        """Copy the command producer before stepping or any automatic reset."""
        if not self._active:
            return None
        router = self.adapter.env._terrain_course.router
        tile = router.tile
        result = {'command_source': self.plan.command_source.detach().clone(),
                'routed_step_height_m': router.step_table[tile].detach().clone(),
                'routed_slope_rad': router.slope_table[tile].detach().clone(),
                'routed_roughness_m': router.roughness_table[tile].detach().clone()}
        if self.recovery_evidence is not None:
            result.update(self.recovery_evidence.sample())
        return result

    def _install_authored_evidence(self):
        """Read actual per-row events before reset, without changing the schema."""
        evidence = getattr(self.adapter.env._terrain_course, "stair_evidence", None)
        if evidence is None:
            return
        self._authored_pending = torch.zeros(evidence.num_envs, 5, dtype=torch.long,
                                             device=evidence.device)
        # Unlike the global post-reset delta, per-case attempts include the
        # fresh initial episode. All subsequent ends/births use source events.
        self._authored_pending[:, 0] = evidence.active.long()
        original_observe, original_finish, original_reset = evidence.observe, evidence._finish, evidence.reset

        def observe(*args, **kwargs):
            completed = original_observe(*args, **kwargs)
            self._authored_pending[:, 2] += completed.long()
            return completed

        def finish(mask):
            ended = mask & evidence.active
            completed = ended & evidence.completed
            result = original_finish(mask)
            self._authored_pending[:, 1] += ended.long()
            self._authored_pending[:, 3] += completed.long()
            self._authored_pending[:, 4] += (ended & ~completed).long()
            return result

        def reset(ids, *args, **kwargs):
            result = original_reset(ids, *args, **kwargs)
            self._authored_pending[ids, 0] += evidence.active[ids].long()
            return result

        for name, callback in (("observe", observe), ("_finish", finish), ("reset", reset)):
            self._hook_restore.append((evidence, name, evidence.__dict__.get(name), name in evidence.__dict__))
            setattr(evidence, name, callback)

    def close(self):
        """Restore only evaluation wrappers, also when a rollout raises."""
        if self.jump_evidence is not None:
            self.jump_evidence.close()
        if self.recovery_evidence is not None:
            self.recovery_evidence.close()
        for owner, name, previous, existed in reversed(self._hook_restore):
            if existed:
                setattr(owner, name, previous)
            else:
                delattr(owner, name)
        self._hook_restore.clear()
        if isinstance(self.plan, FixedCourseScenarios):
            self.plan.close()
        self._active = False

    @staticmethod
    def _record_range(ranges, name, values):
        values = values.reshape(-1).tolist()
        previous = ranges.get(name)
        ranges[name] = {"min": values if previous is None else list(map(min, previous["min"], values)),
                        "max": values if previous is None else list(map(max, previous["max"], values))}

    def _record_gimbal(self, bucket, values, row):
        statistics = bucket.setdefault("gimbal", {
            "samples": 0, "ranges": {}, "translation_samples": 0,
            "translation_absolute_sum": [0., 0.], "translation_square_sum": [0., 0.]})
        statistics["samples"] += 1
        for name, tensor in values.items():
            if name not in {"translation_active", "translation_actual_m_s", "translation_target_m_s"}:
                self._record_range(statistics["ranges"], name, tensor[row])
        if values["translation_active"][row]:
            statistics["translation_samples"] += 1
            actual, target = values["translation_actual_m_s"][row], values["translation_target_m_s"][row]
            for name, tensor in (("translation_actual_m_s", actual), ("translation_target_m_s", target)):
                self._record_range(statistics["ranges"], name, tensor)
            for axis, error in enumerate((actual - target).tolist()):
                statistics["translation_absolute_sum"][axis] += abs(error)
                statistics["translation_square_sum"][axis] += error * error

    def _field(self, snapshot, name):
        if any(field.name == name for field in self.schema.state_fields):
            return named_field(snapshot.state, self.schema, name).detach().cpu()
        return None

    @staticmethod
    def _empty_bucket():
        return {
            "samples": 0, "reward_sum": 0., "terminated": 0, "truncated": 0,
            "valid_samples": 0, "invalid_count": 0, "nonfinite_measurement_count": 0,
            "policy_owned_samples": 0, "jump_physical_success_events": 0,
            "recovery_physical_success_events": 0, "actual": {},
            "tracking_absolute_sum": [0., 0., 0.], "tracking_square_sum": [0., 0., 0.],
            "tracking_clipped_absolute_sum": [0., 0., 0.],
            "tracking_samples": 0, "command_min": None, "command_max": None}

    def step(self, transition, tracking):
        actual, command, gimbal = {}, None, None
        if tracking is not None:
            measured, target = tracking
            actual["tracking"] = (measured - target).double().cpu()
            command = target.detach().cpu()
            if getattr(tracking, "gimbal", None) is not None:
                gimbal = {name: value.detach().cpu() for name, value in tracking.gimbal.items()}
        authored = None
        if self._authored_pending is not None:
            authored = self._authored_pending.cpu().clone()
            self._authored_pending.zero_()
        for key, name in (("family", "course.tasks.family"), ("terrain", "course.tasks.terrain_kind"),
                          ("tile", "course.terrain_tile"), ("gimbal_mode", "course.gimbal.mode")):
            value = self._field(transition.start, name)
            if value is not None:
                actual[key] = value.reshape(-1).long().tolist()
        reward = transition.reward.detach().double().cpu()
        terminated, truncated = transition.terminated.cpu(), transition.truncated.cpu()
        finite = torch.isfinite(reward)
        if tracking is not None:
            finite &= torch.isfinite(actual['tracking']).all(-1) & torch.isfinite(command).all(-1)
        if gimbal is not None:
            for key, value in gimbal.items():
                valid_gimbal = torch.isfinite(value).reshape(len(reward), -1).all(-1)
                if key in ('translation_actual_m_s', 'translation_target_m_s'):
                    valid_gimbal |= ~gimbal['translation_active']
                finite &= valid_gimbal
        owned = transition.start.policy_owned.cpu()
        jump = self._field(transition.end, "course.tasks.success_id.valid")
        recovered_start = self._field(transition.start, "course.recovery._verified")
        recovered_end = self._field(transition.end, "course.recovery._verified")
        context = getattr(tracking, 'course', None) if tracking is not None else None
        if context is not None:
            context = {key: value.detach().cpu() for key, value in context.items()}
            producer = context['command_source']
            if (producer.shape != owned.shape or producer.dtype != torch.int64
                    or ((producer < 0) | (producer > 2)).any()):
                raise ValueError('Invalid pre-action command producer')
            for value in context.values():
                finite &= torch.isfinite(value).reshape(len(reward), -1).all(-1)
        valid = transition.valid.detach().cpu() & finite
        if self.recovery_evidence is not None:
            self.recovery_evidence.observe_transition(transition, valid)
        if self.jump_evidence is not None:
            self.jump_evidence.observe_transition(transition, valid)
        for row, name in enumerate(self.plan.row_names):
            total_bucket = self.buckets.setdefault(name, self._empty_bucket())
            if not owned[row]:
                phase = 'script_owned'
            elif context is None:
                phase = 'unclassified'
            else:
                phase = ('unclassified', 'scheduled_segment', 'native_handoff')[int(context['command_source'][row])]
            phases = self.phase_buckets.setdefault(name, {})
            phase_bucket = phases.setdefault(phase, self._empty_bucket())
            row_buckets = [total_bucket, phase_bucket]
            if context is not None and 'recovery_active' in context:
                recovery_phase = ('recovery' if context['recovery_active'][row] else
                    'post_handoff_tracking' if context['recovery_after_handoff'][row] else 'other')
                recovery_phases = self.recovery_phase_buckets.setdefault(name, {})
                row_buckets.append(recovery_phases.setdefault(recovery_phase, self._empty_bucket()))
            for bucket in row_buckets:
                # Preserve the original observed-sample count. Every capability
                # numerator and error denominator below uses only trusted rows.
                bucket['samples'] += 1
                bucket['nonfinite_measurement_count'] += int(~finite[row])
                if not valid[row]:
                    bucket['invalid_count'] += 1
                    continue
                bucket['valid_samples'] += 1
                if context is not None:
                    terrain = bucket.setdefault('routed_terrain_ranges', {})
                    for key, value in context.items():
                        if key not in ('command_source', 'recovery_active', 'recovery_after_handoff'):
                            self._record_range(terrain, key, value[row])
                bucket["reward_sum"] += float(reward[row])
                bucket["terminated"] += int(terminated[row])
                bucket["truncated"] += int(truncated[row])
                bucket["policy_owned_samples"] += int(owned[row])
                if gimbal is not None:
                    self._record_gimbal(bucket, gimbal, row)
                if authored is not None and bucket is total_bucket:
                    events = bucket.setdefault("authored_whole_course", {key: 0 for key in
                        ("started", "ended", "complete", "ended_complete", "incomplete")})
                    for key, occurrences in zip(events, authored[row].tolist()):
                        events[key] += occurrences
                if jump is not None:
                    bucket["jump_physical_success_events"] += int(jump[row] > .5)
                if recovered_end is not None and recovered_start is not None:
                    bucket["recovery_physical_success_events"] += int((recovered_end[row] > .5) & (recovered_start[row] <= .5))
                for key, values in actual.items():
                    if key == "tracking":
                        bucket["tracking_samples"] += 1
                        target = command[row].tolist()
                        if bucket["command_min"] is None:
                            bucket["command_min"], bucket["command_max"] = list(target), list(target)
                        else:
                            bucket["command_min"] = list(map(min, bucket["command_min"], target))
                            bucket["command_max"] = list(map(max, bucket["command_max"], target))
                        for axis in range(3):
                            bucket["tracking_absolute_sum"][axis] += float(values[row, axis].abs())
                            bucket["tracking_square_sum"][axis] += float(values[row, axis].square())
                            bucket['tracking_clipped_absolute_sum'][axis] += min(
                                float(values[row, axis].abs()), CLIPPED_MAE_LIMITS[axis])
                    else:
                        counts = bucket["actual"].setdefault(key, {})
                        label = str(values[row])
                        counts[label] = counts.get(label, 0) + 1

    @staticmethod
    def _summarize_bucket(source):
        bucket = dict(source)
        count = bucket.pop("tracking_samples")
        absolute, square = bucket.pop("tracking_absolute_sum"), bucket.pop("tracking_square_sum")
        clipped = bucket.pop('tracking_clipped_absolute_sum')
        bucket["tracking"] = {"samples": count, "coordinates": ["forward_m_s", "yaw_rad_s", "height_m"],
                              "mae": [v / count for v in absolute] if count else None,
                              "rmse": [math.sqrt(v / count) for v in square] if count else None,
                              "clipped_mae": [v / count for v in clipped] if count else None,
                              "clipped_mae_limits": list(CLIPPED_MAE_LIMITS),
                              "aggregation": "sample-weighted valid pre-action measurements paired with a valid interval, including terminating intervals and later reset episodes",
                              "command_min": bucket.pop("command_min"),
                              "command_max": bucket.pop("command_max")}
        if "gimbal" in bucket:
            statistics = dict(bucket["gimbal"])
            count = statistics.pop("translation_samples")
            absolute, square = (statistics.pop("translation_absolute_sum"),
                                statistics.pop("translation_square_sum"))
            statistics["translation_tracking"] = {
                "samples": count, "coordinates": ["gimbal_yaw_forward_m_s", "gimbal_yaw_lateral_m_s"],
                "mae": [v / count for v in absolute] if count else None,
                "rmse": [math.sqrt(v / count) for v in square] if count else None,
                "frame": "native reward task_heading = root_yaw + controller encoder yaw; not view-ray world yaw",
                "sampling": "pre-action, only native special/active rows; inactive preparation is excluded"}
            statistics["angular_ranges"] = "native/controller q preserves source coordinates; view/task headings wrap to [-pi,pi], min/max is not swept angle"
            bucket["gimbal"] = statistics
        if "authored_whole_course" in bucket:
            events = dict(bucket["authored_whole_course"])
            events["ended_completion_rate"] = events["ended_complete"] / events["ended"] if events["ended"] else None
            bucket["authored_whole_course"] = events
        return bucket

    def summary(self):
        result = {}
        for name in self.plan.names:
            if name not in self.buckets:
                result[name] = {"samples": 0}
                continue
            bucket = self._summarize_bucket(self.buckets[name])
            bucket['command_phases'] = {phase: self._summarize_bucket(values)
                for phase, values in self.phase_buckets.get(name, {}).items()}
            if name in self.recovery_phase_buckets:
                bucket['recovery_phases'] = {phase: self._summarize_bucket(values)
                    for phase, values in self.recovery_phase_buckets[name].items()}
            result[name] = bucket
        after = self.adapter.env._terrain_course.metrics()
        delta = {}
        for key in ("physical_request_outcomes", "request_end_counts"):
            delta[key] = {name: value - self.before[key][name] for name, value in after[key].items()}
        delta["recovery"] = {name: value - self.before["recovery"]["counts"][name]
                             for name, value in after["recovery"]["counts"].items()}
        delta["authored_stair_completion"] = {
            name: [a - b for a, b in zip(after["authored_stair_completion"][name], values)]
            for name, values in self.before["authored_stair_completion"].items()
            if name in {"started_by_tile", "ended_by_tile", "complete_by_tile", "ended_complete_by_tile", "incomplete_by_tile"}}
        router = getattr(self.adapter.env._terrain_course, 'router', None)
        tiles = getattr(router, 'manifest', {}).get('tiles', [])
        observed = {int(tile) for bucket in self.buckets.values() for tile in bucket['actual'].get('tile', {})}
        catalog = {}
        for index in sorted(observed):
            if index < 0 or index >= len(tiles):
                continue
            tile = tiles[index]
            parameters = tile.get('parameters', {})
            profile = parameters.get('profile_m')
            catalog[str(index)] = {key: tile.get(key) for key in ('name', 'kind', 'level', 'source_roi_m')}
            catalog[str(index)].update(parameters=parameters,
                profile_peak_above_first_m=None if not profile else max(profile) - profile[0])
        report = {"format": "supermimic-fixed-course-evaluation-v3", "plan": self.plan.description(),
                "buckets": result, "physical_counter_deltas_after_initial_reset": delta,
                "routed_terrain_catalog": catalog,
                "terrain_semantics": "declared parameters/profiles of observed pinned atlas tiles; first routing rise is not total traversal height; no inferred crossing success",
                "command_phase_semantics": "pre-action native producer: scheduled_segment, native_handoff or unclassified; script_owned overrides producer when RL does not own the interval; phase totals partition all observed samples/endpoints, not skill qualification",
                "native_final_metrics": after,
                "validity_semantics": "samples counts all observed intervals; valid_samples excludes transition.valid=false or nonfinite measured rows; all capability, reward, range and tracking statistics exclude those invalid rows",
                "native_counter_semantics": "raw upstream aggregate counters, including invalid/reset events; not the validity-filtered per-case capability statistics",
                "clipped_mae_reference": "Radiant 0bf3456 clip bounds only; native coordinates and sample weighting differ from its first-episode trajectory-weighted evaluation",
                "success_semantics": "per-case native jump pulse/recovery rising edge/authored observe-finish-reset events; global counters are separate and never apportioned to cases",
                "scope": "observed intervals only; empty buckets and incomplete events are not demonstrated capabilities"}
        if self.recovery_evidence is not None:
            report['recovery_diagnostics'] = self.recovery_evidence.summary()
            report['recovery_phase_semantics'] = (
                'Pre-action native recovery.active: the handoff interval belongs to recovery; '
                'post_handoff_tracking persists through later command draws until episode reset. '
                'Phase totals partition observed samples, not certified skill successes.')
        if self.jump_evidence is not None:
            report['jump_diagnostics'] = self.jump_evidence.summary()
        if self.config.recovery_qualification:
            report['command_phase_semantics'] += (
                ' With recovery_qualification enabled, scheduled_segment also includes the fixed '
                'qualification request override; it does not assert that a new native segment was drawn.')
        return report
