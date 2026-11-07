"""Explicit native assembly and source identity, without launching Isaac."""
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.course.platform_course import NativePlatformCourse, PlatformCoursePlan
from supermimic_rl.course.platform_intent import PlatformIntentPhase
from supermimic_rl.course.jump_takeoff import install_jump_takeoff
from supermimic_rl.course.reward import install_flight_latch
from supermimic_rl.world_model_v2.platform_records import PLATFORM_RECORD_VERSION

from transformer_rl.formal.lab_env.course_tasks import LabCourseTasks

from support.world_v2.jump_task import native_jumps
from support.world_v2.stair_task import native_stairs, estimate_at
from test_native_platform_landing import native_fixture, tick


def row(target='101:profile:1', earliest=2.):
    return dict(target_id=target, earliest_episode_s=earliest, rise_m=.2, support_height_m=.32)


def configuration(tmp_path, rows=None):
    # Two real triangles per target; actual asset coverage is independently
    # checked by the geometry/binder tests, not fabricated by this timeline.
    targets, faces = [], {}
    for tile, name in enumerate(('101:profile:1', '102:profile:1')):
        triangles = [[[0., -1., .2], [2., -1., .2], [2., 1., .2]],
                     [[0., -1., .2], [2., 1., .2], [0., 1., .2]]]
        ids = [tile*2, tile*2+1]
        for i, triangle in zip(ids, triangles):
            faces[str(i)] = dict(tile_id=tile, source_part_role='prepared_source',
                                 tile_local_vertices_m=triangle)
        targets.append(dict(target_id=name, tile_id=tile, triangle_ids=ids,
            source_roi_xy_m=[[-2., -2.], [2., 2.]], sparse_query=dict(numerical_z_tolerance_m=1e-6)))
    catalog = dict(format='authored-platform-triangle-targets-v1',
        source={'atlas_manifest': {'sha256': 'atlas-pin'}, 'atlas_geometry': {'sha256': 'geometry-pin'}},
        targets=targets, face_catalog=faces)
    path = tmp_path / 'catalog.json'
    raw = json.dumps(catalog).encode()
    path.write_bytes(raw)
    return dict(catalog_path=str(path), catalog_sha256=hashlib.sha256(raw).hexdigest(),
                rows=[row()] if rows is None else rows)


def read(config, *, count=None, **overrides):
    args = dict(num_envs=len(config['rows']) if count is None else count, atlas_sha256='atlas-pin',
                record_substeps=True, jump_takeoff={})
    return PlatformCoursePlan.read(config, **{**args, **overrides})


def test_disabled_plan_has_no_path_or_optional_configuration_dependency():
    assert PlatformCoursePlan.read(None, num_envs=0, atlas_sha256=None,
                                   record_substeps=False, jump_takeoff=None) is None


def test_fixed_plan_records_future_intent_exactly_and_does_not_consume_rng(tmp_path):
    config = configuration(tmp_path, [row(earliest=2.13), None, row('102:profile:1', 3.17)])
    rng = torch.get_rng_state().clone()
    plan = read(config)
    state = plan.state('cpu')
    assert torch.equal(torch.get_rng_state(), rng)
    assert state.target_index.tolist() == [0, -1, 1]
    assert state.phase.tolist() == [int(PlatformIntentPhase.WAITING), 0, int(PlatformIntentPhase.WAITING)]
    assert state.earliest_episode_s.dtype == torch.float64
    assert state.earliest_episode_s.tolist() == [2.13, 0., 3.17]
    config['rows'][0]['rise_m'] = .9
    assert plan.rows[0]['rise_m'] == .2


@pytest.mark.parametrize('field,value', [('earliest_episode_s', -1), ('rise_m', 0),
    ('rise_m', float('inf')), ('support_height_m', None), ('target_id', '102:profile:2')])
def test_missing_or_invalid_plan_values_are_rejected_before_simulator(tmp_path, field, value):
    config = configuration(tmp_path)
    config['rows'][0][field] = value
    with pytest.raises(ValueError, match='Platform plan'):
        read(config)


@pytest.mark.parametrize('options', [{'record_substeps': False}, {'jump_takeoff': None}])
def test_recording_and_gate_are_explicit_prelaunch_requirements(tmp_path, options):
    with pytest.raises(ValueError, match='requires record_substeps'):
        read(configuration(tmp_path), **options)


def test_catalog_pin_atlas_pin_and_exact_row_count_are_prelaunch_requirements(tmp_path):
    config = configuration(tmp_path)
    with pytest.raises(ValueError, match='per environment'):
        read(config, count=2)
    with pytest.raises(ValueError, match='pinned native atlas'):
        read(config, atlas_sha256='wrong')
    with pytest.raises(ValueError, match='pin differs'):
        read({**config, 'catalog_sha256': 'wrong'})
    del config['rows'][0]['rise_m']
    with pytest.raises(ValueError, match='needs a supported target'):
        read(config)


def assembly(tmp_path, monkeypatch):
    if not hasattr(LabCourseTasks, '_select_request'):
        pytest.skip('Native private request-selection entry point is required')
    existing, observer, gap = native_fixture(tmp_path)
    observer.close()
    native = native_stairs(1)
    course = native._terrain_course
    course.tasks = native_jumps()._terrain_course.tasks
    tasks = course.tasks
    for name in ('_binding', '_wheel_body_indices', '_contact_loads', '_last_sample', '_last_state', '_physical_tick'):
        setattr(native, name, getattr(existing, name))
    native._episode_id = torch.tensor([1])
    native._command[:, 3] = .32
    native.time.fill_(2.)
    tasks.elapsed.fill_(2.)
    tasks.next_request_s.fill_(100.)
    tasks.protocol.last_time_s.copy_(native.time)
    tasks.trajectory.start[:, 3] = .32
    tasks.trajectory.end[:, 3] = .32
    tasks.stream.current[:, 3] = .32
    course.stair_requests.reset(torch.arange(1), native.time)
    course.reset_commands = lambda ids, *args, **kwargs: None
    course.router = SimpleNamespace(identity='atlas-pin', tile=torch.tensor([0]),
        origins=torch.zeros(1, 3), origins_table=torch.zeros(2, 3),
        manifest={'files': {'geometry.npz': {'sha256': 'geometry-pin'}}})

    class Query:
        tile_count, device = 2, torch.device('cpu')
        _ceiling_m = 1.

        def __call__(self, points, tiles):
            height = torch.full_like(points[..., 0], .2)
            normal = torch.zeros_like(points)
            normal[..., 2] = 1.
            return height, torch.ones_like(height, dtype=torch.bool), normal, torch.ones_like(height)

    course.query = Query()

    class Clearance:
        def __init__(self, bound_native, terrain, *, selected_capture, tile_ids):
            assert bound_native is native
            assert terrain.query is course.query
            assert torch.equal(terrain.origins, course.router.origins_table)
            self.selected_capture = selected_capture

        def capture(self):
            from transformer_rl.formal.control.wheel_clearance import WheelClearanceSample

            return WheelClearanceSample(gap[0], course.router.origins, native.time,
                native._physical_tick, 0, ('left', 'right'), 'test-source', True, False)

    monkeypatch.setattr('supermimic_rl.course.native_platform_clearance.NativePlatformClearance', Clearance)
    plan = read(configuration(tmp_path))
    owner = NativePlatformCourse(native, plan)
    return native, owner


def test_installation_order_preserves_shared_owner_latch_and_toftime(tmp_path, monkeypatch):
    native, owner = assembly(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='Complete native platform installation'):
        owner.recording_options()
    owner.install_physics()
    with pytest.raises(ValueError, match='take-off gate'):
        owner.install_caller()
    latch = install_flight_latch(native._terrain_course.tasks)
    install_jump_takeoff(native, {})
    owner.install_caller()
    assert owner.recording_options() == {'record_platform': True, 'platform_landing': owner.landing}
    assert native.supermimic_platform_course is owner
    assert native._terrain_course.tasks.supermimic_flight_latch is latch
    # The real native request boundary admits exactly the explicit plan. The
    # SDK contact view and clearance are test doubles, not a simulation result.
    native.time.add_(.02)
    native._completed_estimate = estimate_at(native.time - .02, 1)
    native._terrain_course.advance_skill()
    assert owner.caller.state.phase.tolist() == [int(PlatformIntentPhase.ACCEPTED)]
    request = owner.caller.state.bound_request_id.item()
    assert request == native._terrain_course.tasks.physical_id.item()
    owner.landing.begin_interval()
    for _ in range(4):
        tick(native)
    interval = owner.landing.capture_interval()
    evidence, history = interval.evidence, interval.state
    assert evidence.selected.shape == (1, 4) and evidence.selected.all()
    assert history.request_id.shape == (1, 4)
    assert (history.request_id == request).all()
    summary = owner.summary()
    assert summary['flat_jump_failure_conditions_applicable'] is False
    assert summary['rows'][0]['bound_request_id'] == request
    assert summary['rows'][0]['actual_tile'] == 0
    assert 'success_rate' not in summary


def test_plan_source_receipt_is_stable_under_runtime_updates(tmp_path, monkeypatch):
    native, owner = assembly(tmp_path, monkeypatch)
    owner.install_physics()
    install_flight_latch(native._terrain_course.tasks)
    install_jump_takeoff(native, {})
    owner.install_caller()
    before = owner.receipt()
    owner.caller.state = replace(owner.caller.state, phase=torch.tensor([int(PlatformIntentPhase.DONE)]))
    assert owner.receipt() == before
    assert before['record_version'] == PLATFORM_RECORD_VERSION == 2
    assert before['catalog_sha256'] == owner.geometry.catalog_sha256
    assert before['joint_imagination_enabled'] is False
    assert before['rows'][0]['earliest_episode_s'] == 2.
    mask = owner.platform_rows
    mask.fill_(False)
    assert owner.platform_rows.all()


def test_active_plan_cannot_silently_verify_the_other_tile(tmp_path, monkeypatch):
    native, owner = assembly(tmp_path, monkeypatch)
    owner.install_physics()
    install_jump_takeoff(native, {})
    owner.install_caller()
    owner.caller.state = replace(owner.caller.state, phase=torch.tensor([int(PlatformIntentPhase.ACCEPTED)]),
                                bound_request_id=torch.tensor([7]))
    native._terrain_course.tasks.physical_id.fill_(7)
    native._terrain_course.router.tile.fill_(1)
    with pytest.raises(ValueError, match='different target tile'):
        owner._selected()


def test_actual_router_geometry_identity_and_before_latch_boundary(tmp_path, monkeypatch):
    native, owner = assembly(tmp_path, monkeypatch)
    native._terrain_course.router.manifest['files']['geometry.npz']['sha256'] = 'wrong'
    with pytest.raises(ValueError, match='actually installed'):
        NativePlatformCourse(native, owner.plan)
    native._terrain_course.router.manifest['files']['geometry.npz']['sha256'] = 'geometry-pin'
    install_flight_latch(native._terrain_course.tasks)
    with pytest.raises(ValueError, match='before the shared first-flight'):
        NativePlatformCourse(native, owner.plan)
