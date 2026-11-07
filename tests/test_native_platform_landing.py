"""Actual patch ownership, whole-wheel evidence, and one native payment owner."""
from dataclasses import replace
from types import MethodType, SimpleNamespace

import pytest
import torch

from transformer_rl.formal.control.wheel_clearance import CylinderTerrainGap, WheelClearanceSample
from transformer_rl.formal.lab_env.contact_measurement import LabContactLoads
from transformer_rl.formal.tasks.geometry import HeightPatch, TerrainSurface, TerrainQueryBatch

from supermimic_rl.course.jump_physics import PlatformLandingPhase, observe_jump_physics
from supermimic_rl.course.native_platform_landing import (
    NativePlatformLanding, PlatformContactPatches, PlatformLandingEvidenceBuilder,
)
from supermimic_rl.course.platform_intent import PlatformIntentPhase, PlatformIntentState
from supermimic_rl.course.reward import install_flight_latch
from support.world_v2.jump_task import native_jumps
from test_platform_geometry import Terrain, fixture, rectangle


def geometry(tmp_path):
    terrain = Terrain(lambda p: torch.where(p[..., 0] >= 0, .2, 0.))
    return fixture(tmp_path, rectangle(0, -2, 2, 2), terrain=terrain)


def patches(points=((.01, -.3, .2), (.01, .3, .2)), *, force=(20., 20.),
            body=(1, 2), normals=None, rows=None):
    count = len(force)
    return PlatformContactPatches(
        torch.zeros(count, dtype=torch.long) if rows is None else torch.tensor(rows),
        torch.tensor(body, dtype=torch.long), torch.tensor(points).reshape(count, 3).float(),
        torch.tensor([[0., 0., 1.]] * count) if normals is None else torch.tensor(normals).float(),
        torch.tensor(force).float())


def cylinder_gap(n=1, *, x=-.02, z=.3):
    # Actual native analytic solver: the rim overlaps a 200 mm step even when
    # the wheel centre column is still over the lower approach.
    surface = TerrainSurface((
        HeightPatch((-2., 0.), (-2., 2.), ((0., 0.), (0., 0.)), .8),
        HeightPatch((0., 2.), (-2., 2.), ((.2, .2), (.2, .2)), .8)))
    query = CylinderTerrainGap(TerrainQueryBatch([surface] * n), (.1, .1), (.04, .04))
    centre = torch.tensor([[[x, -.3, z], [x, .3, z]]]).expand(n, -1, -1)
    axis = torch.tensor([[[0., 1., 0.]]]).expand(n, 2, -1)
    return query.at(centre, axis)


def build(builder, patch=None, *, loads=(0., 20., 20., 0.), gap=None,
          valid=True, selected=True):
    return builder.build(patches=patches() if patch is None else patch,
        body_load_n=torch.tensor([loads]), contact_valid=torch.tensor([valid]),
        tile=torch.tensor([0]), selected=torch.tensor([selected]), target_index=torch.tensor([0]),
        command_height_m=torch.tensor([.32]), clearance=gap)


def test_loaded_rim_target_is_not_the_wheel_centre_column(tmp_path):
    builder = PlatformLandingEvidenceBuilder(geometry(tmp_path), [1, 2])
    gap = cylinder_gap()
    assert gap.valid.all() and (gap.vertical_gap_m > 0).all()
    assert (gap.vertical_gap_m < .01).all()  # Centre column would report 200 mm.
    result = build(builder, gap=gap)
    assert result.wheel_touch.all() and result.target_wheel_touch.all()
    assert result.measurement_valid.item() and result.landing_support_valid.item()
    assert result.landing_world_height.item() == pytest.approx(.52)
    assert not result.full_body_clear.item()


def test_real_support_outside_selected_union_is_contact_not_air(tmp_path):
    target = fixture(tmp_path, rectangle(0, -.1, 2, .1), terrain=Terrain(.2))
    builder = PlatformLandingEvidenceBuilder(target, [1, 2])
    result = build(builder, gap=cylinder_gap())
    assert result.wheel_touch.all() and not result.target_wheel_touch.any()
    assert result.measurement_valid.item() and result.landing_support_valid.item()
    assert not result.full_body_clear.item()


def test_light_first_target_contact_is_not_mistaken_for_a_wrong_landing(tmp_path):
    result = build(PlatformLandingEvidenceBuilder(geometry(tmp_path), [1, 2]),
                   patches(force=(3., 3.)), loads=(0., 3., 3., 0.), gap=cylinder_gap())
    assert result.wheel_touch.all() and result.target_wheel_touch.all()
    assert not result.landing_support_valid.item()  # Existing grounded minimum is 5 N.


def test_horizontal_riser_patch_is_not_upward_support(tmp_path):
    builder = PlatformLandingEvidenceBuilder(geometry(tmp_path), [1, 2])
    wall = patches(normals=((1., 0., 0.), (1., 0., 0.)))
    result = build(builder, wall, gap=cylinder_gap())
    assert result.wheel_touch.all() and not result.target_wheel_touch.any()
    assert not result.landing_support_valid.item()
    # Target support plus a significant wrong contact must remain distinguishable.
    mixed = patches(points=((.01, -.3, .2), (.01, .3, .2), (0., -.3, .1)),
                    force=(20., 20., 3.), body=(1, 2, 1),
                    normals=((0., 0., 1.), (0., 0., 1.), (1., 0., 0.)))
    result = build(builder, mixed, loads=(0., 23., 20., 0.), gap=cylinder_gap())
    assert result.target_wheel_touch.tolist() == [[False, True]]


def test_whole_body_departure_and_gap_bounds_are_separate_from_wheel_unloading(tmp_path):
    builder = PlatformLandingEvidenceBuilder(geometry(tmp_path), [1, 2])
    empty = patches(points=(), force=(), body=(), normals=())
    high = cylinder_gap(z=.45)
    result = build(builder, empty, loads=(0., 0., 0., 0.), gap=high)
    assert result.full_body_clear.item() and result.measurement_valid.item()
    body_supported = build(builder, empty, loads=(3., 0., 0., 0.), gap=high)
    assert body_supported.nonwheel_contact.item() and not body_supported.full_body_clear.item()
    result = build(builder, empty, loads=(0., 0., 0., 0.))
    assert not result.full_body_clear.item() and not result.measurement_valid.item()
    # Without an atlas cylinder provider, contact attribution still has real output.
    result = build(builder)
    assert result.measurement_valid.item() and result.target_wheel_touch.all()
    assert not result.landing_support_valid.item()
    result = build(builder, gap=cylinder_gap(z=.28))
    assert not result.landing_support_valid.item()  # Existing penetration band, not a wider one.
    result = build(builder, gap=high, valid=False)
    assert not result.measurement_valid.item() and not result.landing_support_valid.item()


class ContactView:
    sensor_count, filter_count, max_contact_data_count = 4, 1, 16

    def __init__(self):
        self.calls, self.loaded = 0, True

    def get_contact_data(self, dt):
        self.calls += 1
        # Body-major order deliberately differs from native order: wheel R,
        # base, wheel L, auxiliary body. Padding positions may be nonfinite.
        count = torch.tensor([[1], [0], [1], [0]]) if self.loaded else torch.zeros(4, 1, dtype=torch.long)
        starts = torch.tensor([[0], [123], [1], [456]])
        return (torch.tensor([20., 20., 0.]), torch.tensor([[.01, .3, .2], [.01, -.3, .2],
                                                          [torch.nan] * 3]),
                torch.tensor([[0., 0., 1.], [0., 0., 1.], [torch.nan] * 3]),
                torch.zeros(3), count, starts)


def native_fixture(tmp_path):
    native = native_jumps()
    tasks = native._terrain_course.tasks
    tasks.observe_physics = MethodType(observe_jump_physics, tasks)
    tasks.physical_id.fill_(7)
    tasks.active_rise.fill_(.2)
    tasks.latched_height.fill_(.32)
    tasks.supported_start.fill_(True)
    # Real native reduction/read path, with only the SDK contact view mocked.
    loads = object.__new__(LabContactLoads)
    loads.view, loads.surface_count, loads.capacity = ContactView(), 1, 16
    loads.sensor_to_native, loads.wheels = torch.tensor([2, 0, 1, 3]), torch.tensor([1, 2])
    loads.normal, loads.upward = torch.zeros(1, 4), torch.zeros(1, 4)
    loads.valid, loads._tick = torch.zeros(1, dtype=torch.bool), None
    native._contact_loads, native._wheel_body_indices = loads, loads.wheels
    native._binding = SimpleNamespace(native_body_names=('base', 'left', 'right', 'aux'),
                                     wheel_body_indices=(1, 2))
    native._physical_tick = 0
    native._command = torch.tensor([[0., 0., 0., .32]])
    native._terrain_course.router.tile = torch.tensor([0])
    native._terrain_course.router.identity = 'atlas-pin'
    native._last_sample = native._last_state = None
    intent = replace(PlatformIntentState.plan(target_index=torch.tensor([0]),
        earliest_episode_s=torch.zeros(1, dtype=torch.float64), rise_m=torch.tensor([.2]),
        support_height_m=torch.tensor([.32])), phase=torch.tensor([int(PlatformIntentPhase.ACCEPTED)]),
        bound_request_id=torch.tensor([7]))
    current_gap = [cylinder_gap()]

    def capture_gap():
        return WheelClearanceSample(current_gap[0], torch.zeros(1, 3),
            torch.tensor([native._physical_tick * .005], dtype=torch.float64), native._physical_tick,
            0, ('left', 'right'), 'test-urdf', True, False)

    observer = NativePlatformLanding(native, geometry(tmp_path), intent_capture=lambda: intent,
                                     clearance_capture=capture_gap).install()
    return native, observer, current_gap


def tick(native, *, z=.52, vz=0.):
    native._physical_tick += 1
    normal, upward = native._contact_loads.read(native._physical_tick, .005)
    native._last_sample = SimpleNamespace(body_normal_load_n=normal,
        contact_measurement_valid=native._contact_loads.valid)
    native._last_state = SimpleNamespace(contact_measurement_valid=native._contact_loads.valid)
    return native._terrain_course.tasks.observe_physics(upward, torch.tensor([vz]), torch.zeros(1),
        torch.tensor([z]), angular_speed_rad_s=torch.zeros(1))


def test_native_cache_uses_one_backend_read_and_real_body_binding(tmp_path):
    native, observer, _ = native_fixture(tmp_path)
    backend = native._contact_loads.view._view
    returned = tick(native)
    assert backend.calls == 1 and returned is observer.state
    assert observer.last_evidence.target_wheel_touch.all()
    native._contact_loads.read(native._physical_tick, .005)
    assert backend.calls == 1  # The cached native read is not repeated by the observer.
    saved = observer.capture()
    saved.phase.fill_(99)
    assert observer.state.phase.item() != 99
    observer.close()
    assert native._contact_loads.view is backend
    assert native._terrain_course.tasks.observe_physics.__func__ is observe_jump_physics


def test_native_shared_update_before_latch_and_single_payment(tmp_path):
    native, observer, gap = native_fixture(tmp_path)
    backend = native._contact_loads.view._view
    tasks = native._terrain_course.tasks
    latch = install_flight_latch(tasks)
    backend.loaded = False
    gap[0] = cylinder_gap(z=.45)
    for _ in range(2):
        tick(native, vz=.4)
    assert observer.state.phase.item() == PlatformLandingPhase.FLIGHT
    assert latch.lifted.item()
    backend.loaded = True
    gap[0] = cylinder_gap()
    for _ in range(405):
        tick(native)
    assert tasks.paid_id.item() == 7 and tasks.physical_outcome_counts[0].item() == 1
    assert observer.state.phase.item() == PlatformLandingPhase.TARGET_BOTH_WHEELS
    assert backend.calls == 407
    with pytest.raises(RuntimeError, match='outer'):
        observer.close()


def test_stale_packet_is_not_read_again_or_fabricated_as_clear_air(tmp_path):
    native, observer, _ = native_fixture(tmp_path)
    tick(native)
    native._physical_tick += 1
    native._terrain_course.tasks.observe_physics(torch.full((1, 2), 20.), torch.zeros(1),
        torch.zeros(1), torch.tensor([.52]), angular_speed_rad_s=torch.zeros(1))
    assert native._contact_loads.view._view.calls == 1
    assert not observer.last_evidence.measurement_valid.item()


def test_partial_reset_clears_only_selected_landing_rows():
    observer = object.__new__(NativePlatformLanding)
    from supermimic_rl.course.jump_physics import PlatformLandingState
    observer.state = PlatformLandingState(torch.tensor([2, 3]), torch.tensor([2, 2]), torch.tensor([3, 4]))
    observer.reset(torch.tensor([True, False]))
    assert observer.state.request_id.tolist() == [-1, 3]
    assert observer.state.free_ticks.tolist() == [0, 2]
    assert observer.state.phase.tolist() == [0, 4]


def test_ordinary_native_jump_remains_bitwise_equal(tmp_path):
    native, observer, gap = native_fixture(tmp_path)
    actual = native._terrain_course.tasks
    expected = native_jumps()._terrain_course.tasks
    expected.physical_id.fill_(7)
    expected.active_rise.fill_(.2)
    expected.latched_height.fill_(.32)
    expected.supported_start.fill_(True)
    observer.intent_capture = lambda: PlatformIntentState.inactive(1).record()
    for i in range(410):
        air = i < 4
        native._contact_loads.view._view.loaded = not air
        gap[0] = None  # No whole-wheel query is required for unchanged flat rules.
        observer.clearance_capture = None
        force = torch.zeros(1, 2) if air else torch.full((1, 2), 20.)
        z, vz = (.52, .4) if air else (.32, 0.)
        expected.observe_physics(force, torch.tensor([vz]), torch.zeros(1), torch.tensor([z]),
                                 angular_speed_rad_s=torch.zeros(1))
        tick(native, z=z, vz=vz)
        for name, value in vars(expected).items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(getattr(actual, name), value, rtol=0, atol=0, msg=name)
    assert actual.physical_outcome_counts[0].item() == 1


def test_reset_hook_runs_native_reset_once_then_clears_history(tmp_path):
    native, observer, _ = native_fixture(tmp_path)
    observer.state = replace(observer.state, request_id=torch.tensor([7]),
                             free_ticks=torch.tensor([2]), phase=torch.tensor([3]))
    calls = []
    observer._reset = lambda *args, **kwargs: calls.append((args, kwargs)) or 'native-result'
    result = native._terrain_course.tasks.reset(torch.tensor([0]), 'task', now='clock')
    assert result == 'native-result' and len(calls) == 1
    assert observer.capture().request_id.item() == -1


@pytest.mark.parametrize('kind', ['stale', 'centre_column', 'unbound', 'wrong_bodies'])
def test_wrong_clearance_contract_cannot_certify_departure(tmp_path, kind):
    native, observer, _ = native_fixture(tmp_path)
    native._contact_loads.view._view.loaded = False
    capture = observer.clearance_capture

    def invalid():
        sample = capture()
        return replace(sample, **{
            'stale': {'physical_tick': sample.physical_tick - 1},
            'centre_column': {'reference': 'wheel_centre_column'},
            'unbound': {'source_geometry_bound': False},
            'wrong_bodies': {'body_names': ('right', 'left')},
        }[kind])

    observer.clearance_capture = invalid
    tick(native, vz=.4)
    assert not observer.last_evidence.full_body_clear.item()
    assert not observer.last_evidence.measurement_valid.item()
