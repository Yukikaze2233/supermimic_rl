"""Event-aligned jump tracking, request identity and unchanged peak obligation."""
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('transformer_rl.formal.tasks.reward')
from transformer_rl.formal.tasks.jump import JumpReferenceConfig, JumpReferenceProgram, ReferenceStage
from transformer_rl.formal.tasks.reward import RewardPhase

from supermimic_rl.course.reward import NativeJumpPhysicalReference, install_shared_reward
from supermimic_rl.reward_kernel import (
    JumpPhysicalReference, JumpProgress, RewardExtensions, RewardHistory, RewardParameters,
    JumpFlightGeometry, physical_reward_densities,
)
from test_reward_kernel import begin, sample, setup


def owner():
    program = JumpReferenceProgram(JumpReferenceConfig(compression_depth_max_m=.10,
        compression_duration_s=.30, nominal_extension_duration_s=.20))
    tasks = SimpleNamespace(physical_id=torch.tensor([4, 5]), ascent_seen=torch.zeros(2, dtype=torch.bool),
        request_clock=torch.tensor([.64, .64]), reference=program,
        active_rise=torch.full((2,), .2), peak_rise=torch.zeros(2), landed_s=torch.zeros(2),
        air_ticks=torch.zeros(2, dtype=torch.long), observe_physics=lambda *a, **k: None)
    tasks.jump_reference = lambda: program.sample(torch.full((2,), .32), tasks.active_rise,
        tasks.request_clock.double(), torch.zeros(2), active=torch.ones(2, dtype=torch.bool))
    tasks.supermimic_flight_latch = SimpleNamespace(lifted=torch.zeros(2, dtype=torch.bool),
        closed=torch.zeros(2, dtype=torch.bool))
    return tasks


def fixture():
    composer, reference = setup('jump_takeoff')
    begin(composer, reference, torch.zeros(2, 6))
    physics = replace(sample(composer, 1), root_world_z_m=torch.tensor([.40, .40]),
        velocity_origin_world_m_s=torch.tensor([[0., 0., .8], [0., 0., .8]]))
    history = RewardHistory(composer._window['dq'], torch.zeros(2), torch.zeros(2), torch.zeros(2))
    progress = JumpProgress(torch.full((2,), int(ReferenceStage.BALLISTIC)), torch.ones(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool), torch.zeros(2), torch.zeros(2, dtype=torch.bool),
        torch.full((2,), .08), torch.full((2,), .2), torch.ones(2, dtype=torch.bool))
    return composer, physics, reference, history, progress


def test_disabled_receipt_and_kernel_remain_unchanged():
    assert 'jump_physical_reference' not in RewardExtensions(jump_push=2.).receipt()
    assert not RewardExtensions().active
    for value in (1, 'true', None):
        with pytest.raises(ValueError, match='boolean'):
            RewardExtensions(jump_physical_reference=value)
    composer, physics, reference, history, progress = fixture()
    parameters = RewardParameters.from_composer(composer)
    old = physical_reward_densities(physics, reference, history, parameters)
    new = physical_reward_densities(physics, reference, history,
        replace(parameters, extensions=RewardExtensions(jump_physical_reference=False)))
    torch.testing.assert_close(old.total, new.total, rtol=0, atol=0)
    assert old.errors.keys() == new.errors.keys()


def test_late_supported_target_keeps_planned_release_displacement_and_velocity():
    tasks = owner()
    _, physics, reference, _, _ = fixture()
    curve = tasks.jump_reference()
    event = NativeJumpPhysicalReference(tasks).sample(physics, reference, curve)
    assert event.late_supported.all() and not event.flight.any()
    torch.testing.assert_close(event.velocity_world_z_m_s.double(), curve.launch_velocity_m_s)
    expected = .32 - curve.compression_depth_m + .5 * curve.launch_velocity_m_s * curve.extension_duration_s
    torch.testing.assert_close(event.height_world_z_m.double(), expected.double())
    # Part of the requested rise occurred during supported extension.
    assert (event.launch_velocity_m_s < (2 * 9.81 * tasks.active_rise).sqrt()).all()


def test_first_flight_anchor_uses_measured_origin_and_same_endpoint_zero_time():
    tasks = owner()
    _, physics, reference, _, _ = fixture()
    event = NativeJumpPhysicalReference(tasks)
    tasks.ascent_seen[:] = True
    tasks.supermimic_flight_latch.lifted[:] = True
    reference = replace(reference, phase=torch.full((2,), int(RewardPhase.FLIGHT)))
    first = event.sample(physics, reference, tasks.jump_reference())
    torch.testing.assert_close(first.height_world_z_m, physics.root_world_z_m, rtol=0, atol=0)
    torch.testing.assert_close(first.velocity_world_z_m_s, physics.velocity_origin_world_m_s[:, 2], rtol=0, atol=0)
    later = event.sample(replace(physics, timestamp_s=physics.timestamp_s + .05,
        root_world_z_m=torch.full((2,), .9)), reference, tasks.jump_reference())
    torch.testing.assert_close(later.height_world_z_m, torch.full((2,), .4 + .8 * .05 - .5 * 9.81 * .05**2))
    torch.testing.assert_close(later.velocity_world_z_m_s, torch.full((2,), .8 - 9.81 * .05))
    assert not later.late_supported.any()


def test_latch_cannot_be_reused_after_touchdown_or_episode_and_request_change():
    tasks = owner()
    _, physics, reference, _, _ = fixture()
    event = NativeJumpPhysicalReference(tasks)
    tasks.ascent_seen[:] = True
    tasks.supermimic_flight_latch.lifted[:] = True
    flight = replace(reference, phase=torch.full((2,), int(RewardPhase.FLIGHT)))
    event.sample(physics, flight, tasks.jump_reference())
    tasks.supermimic_flight_latch.closed[:] = True
    landed = event.sample(physics, reference, tasks.jump_reference())
    assert not landed.flight.any() and not landed.late_supported.any()
    torch.testing.assert_close(landed.height_world_z_m, reference.jump_height_world_z_m)
    tasks.supermimic_flight_latch.closed[:] = False
    # Request changes in row 0, episode changes with a repeated request ID in row 1.
    tasks.physical_id[0] += 1
    fresh = replace(physics, root_world_z_m=torch.tensor([.5, .6]),
        episode_id=physics.episode_id + torch.tensor([0, 1]))
    reset = event.sample(fresh, flight, tasks.jump_reference())
    torch.testing.assert_close(reset.height_world_z_m, fresh.root_world_z_m)


def test_push_fraction_uses_planned_launch_and_flight_tracking_no_longer_brakes_upward():
    composer, physics, reference, history, progress = fixture()
    parameters = replace(RewardParameters.from_composer(composer),
        extensions=RewardExtensions(jump_push=2., jump_physical_reference=True, jump_extension_height_floor=True))
    event = JumpPhysicalReference(torch.full((2,), .4), torch.tensor([1.6, .8]),
        torch.full((2,), 1.6), torch.tensor([True, False]), torch.tensor([False, True]))
    reference = replace(reference, jump_velocity_world_z_m_s=torch.full((2,), -.1),
        phase=torch.tensor([int(RewardPhase.TAKEOFF), int(RewardPhase.FLIGHT)]))
    progress = replace(progress, flight_started=torch.tensor([False, True]))
    actual = physical_reward_densities(physics, reference, history, parameters,
        jump_progress=progress, jump_physical_reference=event)
    torch.testing.assert_close(actual.positive['jump_push'], torch.tensor([1., 0.]))
    assert actual.cost['jump_vertical'][1] == 0
    assert actual.positive['jump_vertical'][1] == 1
    assert actual.errors['jump_clock_world_vertical_m_s'][1] == pytest.approx(.9)
    assert actual.errors['jump_world_vertical_m_s'][1] == 0
    with pytest.raises(ValueError, match='event producer'):
        physical_reward_densities(physics, reference, history, parameters, jump_progress=progress)


def test_native_four_tick_composer_uses_event_producer_after_real_latch():
    composer, physics, reference, _, _ = fixture()
    tasks = owner()
    receipt = install_shared_reward(composer,
        extensions=RewardExtensions(jump_physical_reference=True, jump_push=2.), jump_tasks=tasks)
    assert receipt['jump_physical_reference']['initial_elapsed_s'] == 0.
    for tick in range(1, 5):
        if tick == 2:
            tasks.ascent_seen[:] = True
            tasks.air_ticks[:] = 2
            tasks.supermimic_flight_latch.observe(tasks)
        phase = RewardPhase.TAKEOFF if tick == 1 else RewardPhase.FLIGHT
        current = replace(sample(composer, tick), root_world_z_m=torch.full((2,), .4),
            velocity_origin_world_m_s=torch.tensor([[0., 0., .8], [0., 0., .8]]))
        composer.observe(current, replace(reference, phase=torch.full((2,), int(phase))))
    assert composer._window['steps'] == 4
    torch.testing.assert_close(composer._window['gates']['jump_late_supported_reference'], torch.full((2,), .005, dtype=torch.float64))
    torch.testing.assert_close(composer._window['gates']['jump_event_flight_reference'], torch.full((2,), .015, dtype=torch.float64))


def test_event_tracking_velocity_gradient_matches_finite_difference():
    composer, physics, reference, history, progress = fixture()
    reference = replace(reference, phase=torch.full((2,), int(RewardPhase.FLIGHT)),
        jump_velocity_world_z_m_s=torch.full((2,), -.2))
    event = JumpPhysicalReference(torch.full((2,), .4), torch.full((2,), .8), torch.full((2,), 1.6),
        torch.zeros(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool))
    params = replace(RewardParameters.from_composer(composer), extensions=RewardExtensions(jump_physical_reference=True))
    velocity = torch.tensor([[0., 0., .6], [0., 0., 1.]], dtype=torch.float64, requires_grad=True)

    def score(v):
        result = physical_reward_densities(replace(physics, velocity_origin_world_m_s=v), reference,
            history, params, jump_progress=progress, jump_physical_reference=event)
        return (result.positive['jump_vertical'] - result.cost['jump_vertical']).sum()

    gradient, = torch.autograd.grad(score(velocity), velocity)
    assert gradient[0, 2] > 0 and gradient[1, 2] < 0
    direction = torch.tensor([[0., 0., .3], [0., 0., -.2]], dtype=torch.float64)
    epsilon = 1e-6
    numeric = (score(velocity + epsilon * direction) - score(velocity - epsilon * direction)) / (2 * epsilon)
    torch.testing.assert_close((gradient * direction).sum(), numeric, atol=1e-7, rtol=1e-7)


def test_compact_world_model_rejects_unrecorded_first_flight_anchor():
    from supermimic_rl.world_model_v2.imagined_reward import OrdinaryFlatReward

    source = {'reward_config': {'physics_dt_s': .005}, 'asset_manifest_sha256': 'test',
              'reward_kernel': {'extensions': {'jump_physical_reference': True}}}
    geometry = SimpleNamespace(linkage=SimpleNamespace(provenance={'asset_manifest_sha256': 'test'}))
    with pytest.raises(ValueError, match='first-flight root position'):
        OrdinaryFlatReward(source, geometry, (0, 1), .02)


def request_counterfactual(requested_rise):
    """Two declared trajectories, not a learned policy or simulator claim."""
    composer, physics, reference, _, _ = fixture()
    dt, baseline, start_z, launch = .005, .32, .35, .64
    time = torch.arange(dt, 6. + dt / 2, dt)
    samples, n = len(time), 2 * len(time)
    time = time.repeat(2)
    weak_v = .8
    strong_v = (2 * 9.81 * (baseline + requested_rise - start_z)) ** .5
    launch_v = torch.cat((torch.full((samples,), weak_v), torch.full((samples,), strong_v)))
    elapsed = (time - launch).clamp_min(0)
    return_time = (launch_v + (launch_v.square() + 2 * 9.81 * (start_z - baseline)).sqrt()) / 9.81
    flying = (time >= launch) & (elapsed < return_time)
    closed = (time >= launch) & ~flying
    z = torch.where(flying, start_z + launch_v * elapsed - .5 * 9.81 * elapsed.square(), baseline)
    vz = torch.where(flying, launch_v - 9.81 * elapsed, 0.)
    rise = torch.maximum(z - baseline, torch.zeros_like(z))
    peak = rise.reshape(2, -1).cummax(-1).values.flatten()
    program = owner().reference
    curve = program.sample(torch.full((n,), baseline), torch.full((n,), requested_rise),
        time.double(), torch.zeros(n), active=torch.ones(n, dtype=torch.bool))

    def expand(instance):
        return replace(instance, **{field.name: getattr(instance, field.name)[0:1].expand(
            n, *getattr(instance, field.name).shape[1:]).clone()
            for field in fields(instance) if isinstance(getattr(instance, field.name), torch.Tensor)})

    physics, reference = expand(physics), expand(reference)
    velocity = torch.zeros(n, 3)
    velocity[:, 2] = vz
    physics = replace(physics, root_world_z_m=z, height_above_ground_m=z,
                      velocity_origin_world_m_s=velocity)
    phase = torch.where(flying, int(RewardPhase.FLIGHT), torch.where(closed,
                       int(RewardPhase.SETTLING), int(RewardPhase.TAKEOFF)))
    reference = replace(reference, phase=phase,
        jump_height_world_z_m=torch.where(closed, baseline, curve.height_world_z_m),
        jump_velocity_world_z_m_s=torch.where(closed, 0., curve.velocity_world_z_m_s))
    minimum, span = .1624632159, .2083671688
    geometry = JumpFlightGeometry((z - baseline).clamp_min(0)[:, None].expand(-1, 2),
        torch.full((n, 2), minimum + .03), torch.full((n, 2), minimum),
        torch.full((n, 2), span), torch.ones(n, dtype=torch.bool))
    progress = JumpProgress(curve.clock_stage, time >= launch, closed,
        torch.where(closed, elapsed - return_time, 0.), time >= launch,
        peak, torch.full((n,), requested_rise), torch.ones(n, dtype=torch.bool), flight_geometry=geometry)
    extensions = RewardExtensions(jump_flight=4., jump_lift=2., jump_push=2., jump_tuck=1.7,
        jump_takeoff_extension=.5, jump_missed=1., jump_peak_completion=True, jump_settle=2.,
        jump_extension_height_floor=True, jump_flight_clearance=True)
    history = RewardHistory(physics.policy_velocity_rad_s, torch.zeros(n), torch.zeros(n), torch.zeros(n))
    parameters = replace(RewardParameters.from_composer(composer), mass_kg=torch.full((n,), 13.886), extensions=extensions)
    late = (time >= curve.expected_launch_time_s) & (time < launch)
    planned_v = curve.launch_velocity_m_s.float()
    release_z = curve.target_apex_world_z_m - planned_v.square() / (2 * 9.81)
    h = torch.where(late, release_z, reference.jump_height_world_z_m)
    v = torch.where(late, planned_v, reference.jump_velocity_world_z_m_s)
    event = JumpPhysicalReference(torch.where(flying, z, h), torch.where(flying, vz, v),
                                  planned_v, late, flying)
    output = {}
    for label, enabled in (('before', False), ('after', True)):
        result = physical_reward_densities(physics, reference, history,
            replace(parameters, extensions=replace(extensions, jump_physical_reference=enabled)),
            jump_progress=progress, jump_physical_reference=event if enabled else None)
        assert not result.invalid.any()
        output[label] = {'request_total_without_terminal_event': (result.total.reshape(2, -1).sum(-1) * dt).tolist(),
            'height_and_vertical': ((result.positive['height'] + result.positive['jump_vertical']
                 - result.cost['height'] - result.cost['jump_vertical']).reshape(2, -1).sum(-1) * dt).tolist(),
            'missed_cost': (result.cost['jump_missed'].reshape(2, -1).sum(-1) * dt).tolist(),
            'hold_height_positive': (torch.where(time > 2., result.positive['height'], 0.).reshape(2, -1).sum(-1) * dt).tolist()}
    return output


@pytest.mark.parametrize('requested_rise', [.2, .4])
def test_reanchored_weak_jump_keeps_request_wide_apex_debt(requested_rise):
    result = request_counterfactual(requested_rise)
    # Re-anchoring makes weak flight easier to track locally, but leaves the
    # independent requested-height debt and tail gain exactly in place.
    assert result['after']['missed_cost'] == result['before']['missed_cost']
    assert result['after']['hold_height_positive'] == result['before']['hold_height_positive']
    assert result['after']['missed_cost'][0] > 0
    assert result['after']['missed_cost'][1] == 0
    weak, achieved = result['after']['request_total_without_terminal_event']
    assert achieved > weak
