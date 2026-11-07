"""Opt-in native assembly for fixed or actual-route platform-jump plans.

The existing course still owns terrain placement and command sampling. This
owner binds the shared caller, actual patch evidence and whole-wheel clearance
without changing the course's terrain, birth pose or command sampling.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch

from .platform_intent import PLATFORM_INTENT_FORMAT, PLATFORM_TARGET_IDS, PlatformIntentState
from .platform_sampler import PLATFORM_SAMPLER_FORMAT, PlatformSamplerConfig


PLATFORM_COURSE_FORMAT = 'native-explicit-platform-course-v1'


@dataclass(frozen=True)
class PlatformCoursePlan:
    catalog_path: Path
    catalog_sha256: str
    atlas_sha256: str
    geometry_sha256: str
    rows: tuple
    sampler: PlatformSamplerConfig | None = None

    @classmethod
    def read(cls, configuration, *, num_envs, atlas_sha256, record_substeps, jump_takeoff):
        """Validate required experiment inputs before starting the simulator."""
        if configuration is None:
            return None
        common = {'catalog_path', 'catalog_sha256'}
        if (not isinstance(configuration, dict)
                or set(configuration) not in (common | {'rows'}, common | {'sampler'})):
            raise ValueError('Platform task requires catalog pins and either explicit rows or sampler')
        if record_substeps is not True or not isinstance(jump_takeoff, dict):
            raise ValueError('Platform task requires record_substeps and explicit jump_takeoff configuration')
        sampler = (None if 'sampler' not in configuration else
                   PlatformSamplerConfig.read(configuration['sampler']))
        rows = configuration.get('rows', [None] * num_envs)
        if not isinstance(rows, list) or len(rows) != num_envs:
            raise ValueError('Platform task needs one explicit plan or null per environment')
        names = {'target_id', 'earliest_episode_s', 'rise_m', 'support_height_m'}
        copied = []
        for row in rows:
            if row is None:
                copied.append(None)
                continue
            if not isinstance(row, dict) or set(row) != names or row['target_id'] not in PLATFORM_TARGET_IDS:
                raise ValueError('Platform plan needs a supported target, earliest episode time, rise and support height')
            for name in ('earliest_episode_s', 'rise_m', 'support_height_m'):
                value = row[name]
                if (type(value) not in (int, float) or not math.isfinite(value)
                        or value < 0 or name != 'earliest_episode_s' and value == 0):
                    raise ValueError('Platform plan has an invalid explicit ' + name)
            copied.append({key: row[key] for key in sorted(names)})
        path = Path(configuration['catalog_path']).expanduser().resolve(strict=True)
        raw = path.read_bytes()
        pin = configuration['catalog_sha256']
        if hashlib.sha256(raw).hexdigest() != pin:
            raise ValueError('Platform catalog pin differs')
        catalog = json.loads(raw)
        if (catalog.get('format') != 'authored-platform-triangle-targets-v1'
                or catalog['source']['atlas_manifest']['sha256'] != atlas_sha256
                or not set(PLATFORM_TARGET_IDS).issubset(t['target_id'] for t in catalog['targets'])):
            raise ValueError('Platform catalog must identify both supported targets on the pinned native atlas')
        if sampler is not None:
            sampler.validate_catalog(catalog)
        return cls(path, pin, atlas_sha256, catalog['source']['atlas_geometry']['sha256'],
                   tuple(copied), sampler)

    def state(self, device):
        targets = [-1 if row is None else PLATFORM_TARGET_IDS.index(row['target_id']) for row in self.rows]
        values = lambda name: [0. if row is None else row[name] for row in self.rows]
        return PlatformIntentState.plan(
            target_index=torch.tensor(targets, dtype=torch.long, device=device),
            earliest_episode_s=torch.tensor(values('earliest_episode_s'), dtype=torch.float64, device=device),
            rise_m=torch.tensor(values('rise_m'), dtype=torch.float32, device=device),
            support_height_m=torch.tensor(values('support_height_m'), dtype=torch.float32, device=device))


class NativePlatformCourse:
    """One immutable native scene owner, installed in two existing boundaries.

    ``install_physics`` precedes shared reward / first-flight installation.
    ``install_caller`` follows take-off gating, stair labels and timeout retry.
    Scene destruction owns these patches, just like the existing reward/gate
    installers. This object does not replace a live course or resume a scene.
    """

    def __init__(self, native, plan: PlatformCoursePlan):
        from ..world_model_v2.terrain import AtlasTerrain
        from .native_platform_clearance import NativePlatformClearance
        from .platform_geometry import PlatformGeometry

        self.native, self.plan = native, plan
        course, router = native._terrain_course, native._terrain_course.router
        if (router.identity != plan.atlas_sha256
                or router.manifest['files']['geometry.npz']['sha256'] != plan.geometry_sha256):
            raise ValueError('Platform plan differs from the actually installed native atlas geometry')
        if len(plan.rows) != native.num_envs:
            raise ValueError('Platform plan and native environment count differ')
        if not callable(getattr(course.tasks, '_select_request', None)):
            raise ValueError('Platform task needs the pinned native request-selection entry point')
        if hasattr(course.tasks, 'supermimic_flight_latch'):
            raise ValueError('Install platform physics before the shared first-flight latch')
        self.terrain = AtlasTerrain(course.query, router.origins_table, manifest_sha256=router.identity)
        self.geometry = PlatformGeometry.load(plan.catalog_path, expected_sha256=plan.catalog_sha256,
                                              terrain=self.terrain, target_ids=PLATFORM_TARGET_IDS)
        self._initial = plan.state(self.terrain.origins.device)
        self._reset_counts = torch.zeros(native.num_envs, dtype=torch.long, device=self.terrain.origins.device)
        self._birth_routes = torch.full((native.num_envs, 2), -1, dtype=torch.long,
                                        device=self.terrain.origins.device)
        self.caller = self.task_adapter = self.landing = None
        # Reuse the native query and final router. A later router replacement is
        # a real scene identity change, so routing must be installed first.
        self.clearance = NativePlatformClearance(native, self.terrain, selected_capture=self._selected,
                                                 tile_ids=tuple(self.geometry.tiles.cpu().tolist()))

    @property
    def platform_rows(self):
        """Current episode plan rows, including waiting/finished requests."""
        state = (self._initial if self.caller is None or self.plan.sampler is None
                 else self.caller.state)
        return (state.target_index >= 0).clone()

    def _plan_after_reset(self, ids, task):
        """The native router and task reset have already completed for ids."""
        course = self.native._terrain_course
        tile = course.router.tile[ids]
        family = task.draw.family
        if not torch.equal(family, course.tasks.family[ids]):
            raise ValueError('Platform birth family differs from the completed native reset')
        state = self.plan.sampler.select(tile, family, course.tasks.base_mode[ids])
        self._reset_counts[ids] += 1
        self._birth_routes[ids] = torch.stack((tile, family), -1)
        return state

    def _intent(self):
        state = self._initial if self.caller is None else self.caller.state
        selected = ((state.bound_request_id >= 0)
                    & (state.bound_request_id == self.native._terrain_course.tasks.physical_id))
        expected = self.geometry.tiles[state.target_index.clamp_min(0)]
        if (selected & (self.native._terrain_course.router.tile != expected)).any():
            raise ValueError('An active explicit platform request was routed onto a different target tile')
        return state

    def _selected(self):
        state = self._intent()
        return ((state.bound_request_id >= 0)
                & (state.bound_request_id == self.native._terrain_course.tasks.physical_id))

    def install_physics(self):
        from .jump_task_adapter import NativeJumpTaskAdapter
        from .native_platform_landing import NativePlatformLanding

        if self.task_adapter is not None or hasattr(self.native, 'supermimic_platform_course'):
            raise ValueError('Native platform course is already installed')
        self.task_adapter = NativeJumpTaskAdapter(self.native._terrain_course.tasks)
        self.landing = NativePlatformLanding(self.native, self.geometry, intent_capture=self._intent,
                                             clearance_capture=self.clearance.capture,
                                             wheel_pose_capture=self.clearance.capture_pose).install()
        self.native.supermimic_platform_course = self
        return self

    def install_caller(self):
        from .native_platform_intent import NativePlatformIntent

        if self.task_adapter is None or self.caller is not None:
            raise ValueError('Install platform physics once before installing its caller')
        self.caller = NativePlatformIntent(self.native, task_adapter=self.task_adapter,
            state=self._initial, reset_plan=(None if self.plan.sampler is None
                                             else self._plan_after_reset)).install()
        return self

    def recording_options(self):
        if self.caller is None or self.landing is None:
            raise ValueError('Complete native platform installation before collecting its task records')
        return {'record_platform': True, 'platform_landing': self.landing}

    def receipt(self):
        from ..world_model_v2.platform_records import PLATFORM_RECORD_VERSION
        from . import platform_target_reward

        result = {
            'format': PLATFORM_COURSE_FORMAT, 'intent_format': PLATFORM_INTENT_FORMAT,
            'record_version': PLATFORM_RECORD_VERSION,
            'catalog_sha256': self.plan.catalog_sha256, 'atlas_sha256': self.plan.atlas_sha256,
            'geometry_sha256': self.plan.geometry_sha256, 'target_ids': list(PLATFORM_TARGET_IDS),
            'rows': [None if row is None else dict(row) for row in self.plan.rows],
            'plan_scope': 'explicit fixed per-row plans rearmed after native episode reset; no mixed-course sampler',
            'admission': 'existing command projection/slew and exact admitted height; native take-off readiness',
            'tof': 'stair-only; explicit pending intent reserves automatic submission, then cancel/idle/fresh jump ID',
            'physical_evidence': 'actual native contact patches and source-bound whole-cylinder atlas clearance',
            'clearance_reference': 'whole_cylinder_minimum_world_z_gap_over_own_terrain_footprint',
            'recording': 'policy-start caller and landing owner; four actual physics-tick evidence records',
            'geometry_role': 'privileged task verification and flight target shaping; no additional actor inputs',
            'target_reward': {'format': platform_target_reward.PLATFORM_TARGET_REWARD_FORMAT,
                'implementation_sha256': hashlib.sha256(Path(platform_target_reward.__file__).read_bytes()).hexdigest(),
                'coefficient': 'existing jump_missed; additive to unchanged peak cost',
                'scale': 'max(admitted request rise, source wheel radius)',
                'scope': 'same live request after-observe FLIGHT or TARGET_SINGLE_WHEEL only'},
            'trigger_role': 'native supported-stance readiness uses simulation truth, not a deployment sensor contract',
            'joint_imagination_enabled': False,
            'native_collision_equivalence_verified': False,
        }
        if self.plan.sampler is not None:
            from . import platform_sampler

            del result['rows']
            result['format'] = 'native-route-platform-course-v1'
            result['sampler'] = {'format': PLATFORM_SAMPLER_FORMAT,
                'configuration': self.plan.sampler.to_dict(),
                'implementation_sha256': hashlib.sha256(Path(platform_sampler.__file__).read_bytes()).hexdigest(),
                'selection': 'actual router tile and explicit birth task family after native reset',
                'random_draws': 0,
                'reset_order': 'native route/task reset, select only reset rows, publish caller plan',
                'cold_resume': 'preserve counters; discard active caller history and reselect after real reset'}
            result['plan_scope'] = 'deterministic per-episode plans on existing matching course routes; no terrain or pose override'
            result['joint_imagination_supported'] = True
            result['joint_imagination_scope'] = (
                'caller plan is fully recorded; ordinary command known-horizon rules still apply; '
                'does not enable a learner or override contact-supervision-only dataset restrictions')
        return result

    def state_dict(self):
        """Caller history is auditable; a cold restart must not resume its flight."""
        if self.caller is None:
            raise ValueError('Install the platform caller before saving its state')
        return {'format': 'native-platform-course-state-v1', 'task': self.receipt(),
                'caller': self.caller.state_dict(), 'reset_counts': self._reset_counts.clone(),
                'birth_routes': self._birth_routes.clone()}

    def restore_after_physical_reset(self, saved):
        """Restore only accounting before the adapter performs its real reset."""
        if (not isinstance(saved, dict)
                or set(saved) != {'format', 'task', 'caller', 'reset_counts', 'birth_routes'}
                or saved['format'] != 'native-platform-course-state-v1'
                or saved['task'] != self.receipt()):
            raise ValueError('Platform course saved source differs')
        history = PlatformIntentState.from_state_dict(saved['caller'])
        if history.phase.shape != self._initial.phase.shape:
            raise ValueError('Platform course saved caller row count differs')
        for name, current in (('reset_counts', self._reset_counts), ('birth_routes', self._birth_routes)):
            value = saved[name]
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.long or value.shape != current.shape
                    or (value < (0 if name == 'reset_counts' else -1)).any()):
                raise ValueError('Platform course saved birth accounting differs')
        # Native task restore owns watermarks and aborted requests. Its cold
        # reset supplies a fresh pose/route; old active caller state is evidence
        # only, never an instruction to continue an in-flight physical request.
        self._reset_counts.copy_(saved['reset_counts'])
        self._birth_routes.copy_(saved['birth_routes'])
        self.caller.set_plan(self._initial)

    def summary(self):
        """Actual final snapshots only; never aggregate flat-anchor failure tests."""
        from .jump_physics import PlatformLandingPhase
        from .platform_intent import PlatformIntentPhase

        state = self._intent()
        landing = self.landing.capture()
        tasks = self.native._terrain_course.tasks
        rows = []
        for row in self.platform_rows.nonzero().flatten().cpu().tolist():
            rows.append({'env_id': row, 'episode_id': int(self.native._episode_id[row]),
                'target_id': PLATFORM_TARGET_IDS[int(state.target_index[row])],
                'caller_phase': PlatformIntentPhase(int(state.phase[row])).name.lower(),
                'bound_request_id': int(state.bound_request_id[row]),
                'current_native_physical_id': int(tasks.physical_id[row]),
                'last_native_success_id': int(tasks.success_id[row]),
                'landing_request_id': int(landing.request_id[row]),
                'landing_phase': PlatformLandingPhase(int(landing.phase[row])).name.lower(),
                'consecutive_whole_body_clear_ticks': int(landing.free_ticks[row]),
                'actual_tile': int(self.native._terrain_course.router.tile[row])})
        result = {'format': 'native-platform-final-snapshot-v1', 'task': self.receipt(), 'rows': rows,
                'scope': 'final native caller/landing snapshots; not all requests or a skill success rate',
                'flat_jump_failure_conditions_applicable': False,
                'outcome_semantics': 'native shared platform observer owns success; snapshots do not independently qualify it',
                'joint_imagination_enabled': False}
        if self.plan.sampler is not None:
            result['sampler_births'] = {'reset_counts': self._reset_counts.cpu().tolist(),
                'latest_tile_and_family': self._birth_routes.cpu().tolist(),
                'scope': 'current and explicitly restored accounting only; not task success counts'}
        return result
