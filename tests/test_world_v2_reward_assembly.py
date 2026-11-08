"""Four-tick integration agrees with the native composer's history and event semantics."""
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('transformer_rl.formal.tasks.reward')
from transformer_rl.formal.tasks.reward import RewardReference, RewardPhase
from transformer_rl.formal.tasks.gimbal_course import GimbalRewardReference
from supermimic_rl.world_model_v2 import so3
from supermimic_rl.world_model_v2.reward_assembly import RewardMemory, RewardTick
from supermimic_rl.world_model_v2.support import SupportState
from supermimic_rl.world_model_v2.imagined_reward import OrdinaryFlatReward
from supermimic_rl.world_model_v2.imagined_reward import TerrainLocomotionReward
from supermimic_rl.world_model_v2.model import Ticks
from supermimic_rl.world_model_v2.records import decode_memory
from supermimic_rl.world_model_v2.terrain import TerrainSample
from support.world_v2.reward_assembly import setup


def test_reward_binding_uses_recorded_controller_mass_and_refuses_unknown_task_routes():
    _, assembly, geometry, state, legs, _ = setup('rough_dash')
    source = {'asset_manifest_sha256': geometry.linkage.provenance['asset_manifest_sha256'],
              'reward_config': {'robot_mass_kg': assembly.parameters.tuning.robot_mass_kg,
                  'gamma': assembly.parameters.tuning.gamma, 'profile': 'rough_dash',
                  'physics_dt_s': .005, 'available_terms': list(assembly.available_terms)},
              'terrain_support_config': asdict(assembly.support_config)}
    producer = OrdinaryFlatReward(source, geometry, assembly.wheels, .02)
    start = {'reward_context_valid': torch.ones(2, 1), 'reward_route': torch.zeros(2, 3),
             'horizon': torch.tensor([[1., 0.], [1., 0.]]), 'sim_context': torch.ones(2, 65),
             'controller_parameters': torch.cat((legs.nominal, legs.stiffness, legs.damping,
                                                 legs.effort_bounds, torch.full((2, 6), .017)), -1),
             'motor_strength': torch.ones(2, 6), 'height': state.position[:, 2:3],
             'failure_conditions': torch.zeros(2, 2), 'goal': torch.zeros(2, 16),
             'transforms': torch.cat((torch.zeros(2, 9), torch.eye(3).flatten().repeat(2, 1),
                                      torch.ones(2, 2), torch.zeros(2, 2)), -1)}
    context = producer.bind(start, state)
    torch.testing.assert_close(context.assembly.parameters.mass_kg, torch.full((2,), 21.))
    torch.testing.assert_close(context.encoder_bias, torch.full((2, 6), .017))
    torch.testing.assert_close(context.legs.stiffness, legs.stiffness)
    for field, index in (('reward_route', 0), ('reward_route', 1), ('goal', 7), ('goal', 8)):
        changed = {**start, field: start[field].clone()}
        changed[field][0, index] = 1
        with pytest.raises(ValueError, match='cannot advance'):
            producer.bind(changed, state)
    for field, index in (('reward_route', 2), ('horizon', 1)):
        changed = {**start, field: start[field].clone()}
        changed[field][0, index] = 1
        with pytest.raises(ValueError, match='recorded command and heading state'):
            producer.bind(changed, state)
    with pytest.raises(ValueError, match='recorded native'):
        producer.bind({**start, 'reward_context_valid': torch.zeros(2, 1)}, state)


def test_reward_memory_roundtrip_retains_action_derivatives_and_discrete_support_state():
    _, _, _, _, _, memory = setup('rough_dash')
    action = torch.randn(2, 6, requires_grad=True)
    memory = replace(memory, previous_action=action, action_count=torch.ones(2))
    restored = RewardMemory.decode(memory.encode())
    torch.testing.assert_close(restored.encode(), memory.encode())
    restored.previous_action.square().sum().backward()
    torch.testing.assert_close(action.grad, 2 * action)
    assert restored.support.phase_active.dtype == torch.bool


@pytest.mark.parametrize('profile', ['legacy', 'rough_dash'])
@pytest.mark.parametrize('gimbal', [False, True])
def test_four_tick_reward_matches_native_across_first_second_and_third_actions(profile, gimbal):
    native, assembly, geometry, state, legs, memory = setup(profile)
    wheels, queue = assembly.wheels, torch.zeros(2, 3, 2)
    bias, strength = torch.zeros(2, 6), torch.ones(2, 6)
    ids, owned = torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.bool)
    reference = RewardReference(torch.tensor([[.4, 0., 0., .32], [-.4, 0., .5, .3]]), ids,
                               torch.full((2,), int(RewardPhase.SUPPORT)), torch.zeros(2), torch.zeros(2), torch.zeros(2, 2))
    for interval in range(3):
        action = torch.full((2, 6), .05 * (interval + 1))
        native.begin(action, state.joint_velocity[:, :6], now_s=native._time, episode_id=ids, control_epoch=ids,
                     policy_owned=owned, admitted_body_command=reference.admitted_body_command, command_mode=ids)
        samples = []
        for tick in range(4):
            endpoint = replace(state, velocity=torch.full((2, 3), .1 * tick),
                               joint_velocity=torch.full((2, 8), float(tick)))
            wheel = geometry(endpoint)
            tick_reference = reference
            if gimbal:
                tick_reference = replace(reference, gimbal=GimbalRewardReference(
                    torch.tensor([True, False]), torch.tensor([2.9 + .1 * tick, -.4]),
                    torch.tensor([.5, 0.]), torch.tensor([-.3, 0.])))
            samples.append(RewardTick(endpoint, wheel, torch.zeros(2), torch.zeros(2, 2),
                torch.zeros(2, 21), torch.zeros(2, 2), torch.zeros(2, 3), owned, torch.zeros_like(owned), tick_reference,
                native._time + (tick + 1) * .005, ids, ids, owned))
        start, saved_queue, saved_support = state, queue, memory.support
        result = assembly.integrate(start, tuple(samples), action, memory, queue, legs, bias, strength)
        target, correction = legs.decode_target(action, state.joint_position[:, :4])
        for tick in samples:
            speed = state.joint_velocity[:, 4:6]
            saved_queue = wheels.push(saved_queue, wheels.motor_input(wheels.target(action), speed, strength[:, 4:6]))
            effort = torch.cat((legs.motor_effort(target, state.joint_position, state.joint_velocity, strength),
                                wheels.applied(saved_queue, speed)), -1)
            physics = assembly.physics(tick, effort, correction)
            target_ref, saved_support = assembly.advance_reference(tick, saved_support)
            native.observe(physics, target_ref)
            state = tick.state
        terminated = torch.tensor([interval == 2, False])
        expected = native.finish(terminated=terminated, truncated=torch.zeros_like(owned),
                                  physical_success_request_id=torch.full((2,), -1), entire_skill_policy_owned=owned)
        for name, value in expected.component_densities.items():
            torch.testing.assert_close(result.components[name], value * .02, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(result.finish(terminated, torch.zeros_like(owned), assembly.parameters),
                                   expected.physical_step_reward, atol=1e-12, rtol=1e-12)
        assert not result.invalid.any()
        queue, memory = result.wheel_torque_history, result.memory
        torch.testing.assert_close(queue, saved_queue, atol=0, rtol=0)


def test_assembled_physics_gradient_remains_finite_at_vertical_pitch():
    _, assembly, geometry, state, _, _ = setup('legacy')
    velocity = torch.zeros(2, 3, requires_grad=True)
    rotation = so3.exp(torch.tensor([[0., torch.pi / 2, 0.], [0., 0., 0.]], dtype=torch.float64)).float()
    state = replace(state, velocity=velocity, rotation=rotation.requires_grad_())
    ids, valid = torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.bool)
    ref = RewardReference(torch.zeros(2, 4), ids, ids, torch.zeros(2), torch.zeros(2), torch.zeros(2, 2))
    tick = RewardTick(state, geometry(state), torch.zeros(2), torch.zeros(2, 2), torch.zeros(2, 21),
                      torch.zeros(2, 2), torch.zeros(2, 3), valid, ~valid, ref,
                      torch.zeros(2, dtype=torch.float64), ids, ids, valid)
    result = assembly.physics(tick, torch.zeros(2, 6), torch.zeros(2, 4))
    gradients = torch.autograd.grad(result.roll_pitch_rad.sum() + result.velocity_origin_body_m_s.sum(),
                                     (state.rotation, velocity))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_missed_jump_imagination_matches_native_ticks_and_retains_motion_gradients():
    from transformer_rl.formal.control.observation import CommandMode
    from transformer_rl.formal.tasks.jump import ReferenceStage
    from supermimic_rl.course.reward import install_shared_reward
    from supermimic_rl.reward_kernel import JumpProgress, RewardExtensions

    native, assembly, geometry, start, legs, memory = setup('legacy')
    extensions = RewardExtensions(jump_missed=1.)
    stage = torch.full((2,), int(ReferenceStage.HOLD))
    # Row 0 rises while supported. Only row 1 achieves real liftoff from tick 2.
    ascents = [torch.tensor([True, tick >= 2]) for tick in range(4)]
    flights = [torch.tensor([False, tick >= 2]) for tick in range(4)]
    tasks = SimpleNamespace(jump_reference=lambda: SimpleNamespace(clock_stage=stage), ascent_seen=ascents[0],
                            physical_id=torch.zeros(2, dtype=torch.long), air_ticks=torch.zeros(2, dtype=torch.long),
                            landed_s=torch.zeros(2), observe_physics=lambda *args, **kwargs: None)
    install_shared_reward(native, extensions=extensions, jump_tasks=tasks)
    assembly.parameters = replace(assembly.parameters, extensions=extensions)
    ids, owned = torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.bool)
    mode = torch.full_like(ids, int(CommandMode.JUMP))
    command = torch.tensor([[.4, 0., 0., .32], [.4, 0., 0., .32]])
    action = torch.zeros(2, 6)
    native.begin(action, start.joint_velocity[:, :6], now_s=native._time, episode_id=ids,
                 control_epoch=ids, policy_owned=owned, admitted_body_command=command, command_mode=mode)
    samples = []
    for tick in range(4):
        phase = torch.tensor([int(RewardPhase.TAKEOFF),
                              int(RewardPhase.TAKEOFF if tick < 2 else RewardPhase.FLIGHT)])
        reference = RewardReference(command, mode, phase, torch.full((2,), .32),
                                    torch.zeros(2), torch.zeros(2, 2))
        endpoint = replace(start, velocity=torch.full((2, 3), .1, requires_grad=True))
        samples.append(RewardTick(endpoint, geometry(endpoint), torch.zeros(2), torch.zeros(2, 2),
            torch.zeros(2, 21), torch.zeros(2, 2), torch.zeros(2, 3), owned, ~owned, reference,
            native._time + (tick + 1) * .005, ids, ids, owned,
            jump_progress=JumpProgress(stage, ascents[tick], flight_started=flights[tick])))
    queue = torch.zeros(2, 3, 2)
    result = assembly.integrate(start, tuple(samples), action, memory, queue, legs,
                                torch.zeros(2, 6), torch.ones(2, 6))
    target, correction = legs.decode_target(action, start.joint_position[:, :4])
    previous, support = start, memory.support
    for index, tick in enumerate(samples):
        tasks.ascent_seen = ascents[index]
        tasks.air_ticks = flights[index].long() * 2
        tasks.observe_physics()
        speed = previous.joint_velocity[:, 4:6]
        queue = assembly.wheels.push(queue, assembly.wheels.motor_input(
            assembly.wheels.target(action), speed, torch.ones(2, 2)))
        effort = torch.cat((legs.motor_effort(target, previous.joint_position, previous.joint_velocity,
                                              torch.ones(2, 6)), assembly.wheels.applied(queue, speed)), -1)
        reference, support = assembly.advance_reference(tick, support)
        native.observe(assembly.physics(tick, effort, correction), reference)
        previous = tick.state
    expected = native.finish(terminated=~owned, truncated=~owned,
                             physical_success_request_id=torch.full_like(ids, -1), entire_skill_policy_owned=owned)
    for name, value in expected.component_densities.items():
        torch.testing.assert_close(result.components[name], value * .02, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(result.components['cost.jump_missed'],
                               torch.tensor([-.02, -.01], dtype=torch.float64))
    for name in ('positive.forward', 'positive.yaw'):
        assert result.components[name][0] == 0
        assert result.components[name][1] > 0  # This row achieves real liftoff on tick 3.
    assert (result.components['positive.jump_attitude'] > 0).all()
    loss = sum(result.components.values()).sum()
    gradients = torch.autograd.grad(loss, tuple(tick.state.velocity for tick in samples))
    assert all(torch.isfinite(gradient).all() and gradient.abs().sum() > 0 for gradient in gradients)


class TerrainColumns:
    """Explicit surface fixtures; real triangle queries have separate atlas tests."""
    manifest_sha256 = 'fixture'

    def __init__(self, kind):
        self.kind, self.queries = kind, []
        self.miss = False

    def height_at(self, points, tile):
        self.queries.append(points.detach().clone())
        x = points[..., 0]
        surfaces = (x * 0., .1 * x, .03 * (3 * x).sin(), (x >= 0) * .2,
                    (x >= 0) * -.2, .1 * (x / .2).floor(), x * 0.,
                    (x >= 0) * .2 + (x >= .3) * .15 - (x >= .6) * .05)
        height = surfaces[self.kind].to(points)
        valid = torch.ones_like(x, dtype=torch.bool)
        if self.miss:
            valid[0] = False
        height = torch.where(valid, height, float('nan'))
        normal = torch.zeros_like(points); normal[..., 2] = 1.
        return TerrainSample(height, valid, normal, torch.ones_like(height))


@pytest.mark.parametrize('kind', range(8))
@pytest.mark.parametrize('mode_labels', [False, True])
def test_terrain_reward_and_termination_height_match_native_task_family_rules(kind, mode_labels):
    from transformer_rl.formal.lab_env.terrain import LabTerrainCourse
    from transformer_rl.formal.lab_env.terrain_support import TerrainSupportReference
    from transformer_rl.formal.simulation.ground import WheelCollisionCentres

    native, assembly, geometry, state, legs, memory = setup('rough_dash')
    state.position[:, 0] = -.3
    source = {'asset_manifest_sha256': geometry.linkage.provenance['asset_manifest_sha256'],
              'terrain_sha256': 'fixture', 'terrain_support_config': asdict(assembly.support_config),
              'reward_config': {'robot_mass_kg': assembly.parameters.tuning.robot_mass_kg,
                  'gamma': assembly.parameters.tuning.gamma, 'profile': 'rough_dash',
                  'physics_dt_s': .005, 'available_terms': list(assembly.available_terms)}}
    terrain = TerrainColumns(kind)
    producer = TerrainLocomotionReward(source, geometry, assembly.wheels, .02, terrain=terrain)
    mode = 0 if not mode_labels or kind in (0, 2, 6) else (2 if kind == 1 else 1)
    command = torch.tensor([[.4, 0., .1, .32], [.4, 0., .1, .32]])
    goal = torch.zeros(2, 16)
    goal[:, :4] = command; goal[:, 3] *= 5.; goal[:, 4 + mode] = 1.; goal[:, 11] = 1.
    sim_context = torch.zeros(2, 65); sim_context[:, 0] = assembly.parameters.tuning.robot_mass_kg
    allowed = torch.full((2,), kind in (3, 4, 5, 7))
    record = {'reward_context_valid': torch.ones(2, 1),
        'reward_route': torch.tensor([[mode, kind, 0.]]).repeat(2, 1), 'goal': goal,
        'horizon': torch.ones(2, 2), 'sim_context': sim_context,
        'controller_parameters': torch.cat((legs.nominal, legs.stiffness, legs.damping,
                                            legs.effort_bounds, torch.zeros(2, 6)), -1),
        'motor_strength': torch.ones(2, 6), 'height': state.position[:, 2:3],
        'failure_conditions': torch.stack((torch.zeros(2), allowed), -1),
        'transforms': torch.cat((torch.zeros(2, 9), torch.eye(3).flatten().repeat(2, 1),
                                 torch.ones(2, 2), torch.zeros(2, 2)), -1),
        'terrain_tile': torch.arange(2)[:, None], 'task_family': torch.tensor([[3.], [1.]])}
    context = producer.bind(record, state, known_interval=True)
    course = object.__new__(LabTerrainCourse)
    course.env = SimpleNamespace(_command_mode=torch.full((2,), mode))
    course.tasks = SimpleNamespace(family=record['task_family'][:, 0].long())
    course.router = SimpleNamespace(kinds=torch.full((2,), kind), tile=torch.arange(2))
    course.query_invalid = torch.zeros(2, dtype=torch.bool)
    wheel = geometry(state)
    # This fixture keeps the source joints fixed and translates the upright
    # body, so a native cylinder centred at each measured wheel is equivalent.
    axes = wheel.cylinder_axis_world[0].double()
    axes = axes / axes.norm(dim=-1, keepdim=True)
    wheels = WheelCollisionCentres(('left', 'right'), ((0., 0., 0.),) * 2,
        tuple(side.radius for side in geometry.sides), tuple(side.half_length * 2 for side in geometry.sides),
        tuple(tuple(value) for value in axes.tolist()))
    course.support_reference = TerrainSupportReference(2, wheels, physics_dt_s=.005, config=assembly.support_config)
    knee = torch.stack([geometry.linkage.evaluate(side.name,
        state.joint_position[:, side.secondary] - state.joint_position[:, side.primary])[0]['passive_joint'][:, side.passive_knee]
        for side in geometry.sides], -1)
    control = decode_memory(torch.zeros(2, 16))
    ids, valid = torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.bool)
    for interval in range(3):
        motion = state.with_time_axis().map(lambda value: value.expand(-1, 4, *value.shape[2:]).clone())
        motion.position[:, :, 0] += torch.arange(1, 5) * .08
        motion.velocity[:, :, 0] = .4
        ticks = Ticks(motion, torch.zeros(2, 4, 2), torch.zeros(2, 4, 21), torch.zeros(2, 4, 3),
                      knee[:, None].expand(-1, 4, -1), torch.zeros(2, 4, 2))
        action = torch.full((2, 6), .05)
        result = context.integrate(state, SimpleNamespace(ticks=ticks), action, memory, control, goal, interval)
        geometry_ticks = geometry(motion, passive_knee_position=ticks.passive_knee_position,
                                   passive_knee_velocity=ticks.passive_knee_velocity)
        target, correction = legs.decode_target(action, state.joint_position[:, :4])
        queue, previous = control.wheel_torque_history, state
        native.begin(action, state.joint_velocity[:, :6], now_s=native._time, episode_id=ids,
                     control_epoch=ids, policy_owned=valid, admitted_body_command=command,
                     command_mode=course.env._command_mode)
        for tick in range(4):
            endpoint, wheel = motion.tick(tick), geometry_ticks.tick(tick)
            columns = terrain.height_at(torch.cat((endpoint.position[:, None], wheel.centre_world), 1), torch.arange(2))
            pose = torch.zeros(2, 2, 7); pose[..., :3] = wheel.centre_world; pose[..., 6] = 1.
            course._support_sample = course.support_reference.observe(endpoint.position, pose,
                ticks.wheel_load[:, tick], valid, root_ground_world_z_m=columns.height[:, 0],
                wheel_ground_world_z_m=columns.height[:, 1:], wheel_contact_force_n=ticks.wheel_contact_force[:, tick],
                base_contact_force_n=ticks.base_contact_force[:, tick], eligible=valid)
            height = endpoint.position[:, 2] - columns.height[:, 0]
            expected_height = course.termination_height(SimpleNamespace(height_above_ground_m=height))
            torch.testing.assert_close(result.termination_height[:, tick, 0], expected_height, atol=1e-7, rtol=0)
            assert course.terrain_reward_reference().active.tolist() == [kind != 0, False]
            reference = RewardReference(command, course.env._command_mode,
                torch.full_like(ids, int(RewardPhase.SUPPORT)), torch.zeros(2), torch.zeros(2), torch.zeros(2, 2),
                intentional_base_contact=allowed, terrain=course.terrain_reward_reference(),
                rough_dash=course.rough_dash_reward_reference())
            sample = RewardTick(endpoint, wheel, columns.height[:, 0], columns.height[:, 1:],
                ticks.body_normal_load[:, tick], ticks.wheel_load[:, tick], ticks.resultant_contact_force[:, tick],
                valid, valid, reference, native._time + (tick + 1) * .005, ids, ids, valid)
            speed = previous.joint_velocity[:, 4:6]
            queue = assembly.wheels.push(queue, assembly.wheels.motor_input(assembly.wheels.target(action), speed, record['motor_strength'][:, 4:6]))
            effort = torch.cat((legs.motor_effort(target, previous.joint_position, previous.joint_velocity, record['motor_strength']),
                                assembly.wheels.applied(queue, speed)), -1)
            native.observe(assembly.physics(sample, effort, correction), reference)
            previous = endpoint
        expected = native.finish(terminated=~valid, truncated=~valid,
                                  physical_success_request_id=torch.full_like(ids, -1), entire_skill_policy_owned=valid)
        torch.testing.assert_close(result.physical_reward, expected.physical_step_reward, atol=2e-7, rtol=1e-6)
        torch.testing.assert_close(result.memory.support.encode(), SupportState.from_native(course.support_reference).encode())
        assert not result.invalid.any()
        state, memory = motion.tick(-1), result.memory
        control = replace(control, wheel_torque_history=result.wheel_torque_history)
    # The production path queried all four changing root/left/right columns
    # together, not the start's ground datum copied four times.
    assert terrain.queries[0].shape == (2, 4, 3, 3)
    assert terrain.queries[0][0, :, 0, 0].unique().numel() == 4
    terrain.miss = True
    invalid = context.integrate(state, SimpleNamespace(ticks=ticks), action, memory, control, goal, 3)
    assert invalid.invalid[0] and torch.isnan(invalid.termination_height[0]).all()


@pytest.mark.parametrize('profile', ['legacy', 'rough_dash'])
@pytest.mark.parametrize('skills', [False, True])
@pytest.mark.parametrize('weighted_components', [False, True])
def test_tick_batching_retains_targets_masses_static_limits_and_gradients(
        monkeypatch, profile, skills, weighted_components):
    import supermimic_rl.world_model_v2.reward_assembly as module
    from transformer_rl.formal.tasks.reward import CommandMode

    _, assembly, geometry, start, legs, memory = setup(profile)
    perturbation = torch.tensor([[.013, -.021, .017], [-.019, .023, -.011]], requires_grad=True)
    action = torch.tensor([[.12, -.09, .08, -.11, .3, -.2], [-.14, .07, -.1, .05, -.2, .4]],
                          requires_grad=True)
    mass = torch.tensor([17.3, 27.4], requires_grad=True)
    assembly.parameters = replace(assembly.parameters, mass_kg=mass)
    memory = replace(memory, previous_action=action.detach() * .4,
                     previous_action2=action.detach() * -.3, action_count=torch.tensor([1., 2.]))
    owned, ids = torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.long)
    modes = (torch.tensor([int(CommandMode.JUMP), int(CommandMode.RECOVERY)]) if skills else ids)
    ticks = []
    for index in range(4):
        endpoint = replace(start, position=start.position + perturbation * (index + 1),
            rotation=so3.exp(perturbation * (index + 1)), velocity=perturbation * (index + 2),
            angular_velocity=perturbation * (index + 3),
            joint_velocity=perturbation.repeat(1, 3)[:, :8] * (index + 1))
        command = torch.cat((perturbation * (index + 1), start.position[:, 2:3] + perturbation[:, :1]), -1)
        phases = (torch.tensor([int((RewardPhase.TAKEOFF, RewardPhase.FLIGHT,
                   RewardPhase.LANDING, RewardPhase.SETTLING)[index]), int(RewardPhase.RECOVERY)])
                  if skills else torch.full_like(ids, int(RewardPhase.SUPPORT)))
        reference = RewardReference(command, modes, phases,
            start.position[:, 2] + perturbation[:, 2] * (index + 1),
            perturbation[:, 2] * (index + 2), perturbation[:, :2],
            intentional_base_contact=torch.tensor([True, False]),
            gimbal=(None if skills else GimbalRewardReference(torch.tensor([True, False]),
                perturbation[:, 0] + index * .1, perturbation[:, 1] + .5, perturbation[:, 2] - .3)))
        load = torch.zeros(2, 21)
        load[:, assembly.parameters.wheel_indices] = torch.tensor([[600.], [1100.]])
        ticks.append(RewardTick(endpoint, geometry(endpoint), torch.tensor([-.01, .02]),
            torch.tensor([[.01, .02], [-.03, .01]]), load, torch.full((2, 2), 8.),
            torch.tensor([[650., 700., 1.], [1100., 1150., 3.]]), owned, owned, reference,
            torch.full((2,), (index + 1) * .005, dtype=torch.float64), ids, ids, owned))
    args = (start, tuple(ticks), action, memory, torch.zeros(2, 3, 2), legs,
            torch.full((2, 6), .003), torch.tensor([[.9] * 6, [1.1] * 6]))
    original = module.physical_reward_densities
    calls = []

    def recorded(sample, reference, history, parameters, **kwargs):
        calls.append((sample, reference, parameters))
        return original(sample, reference, history, parameters, **kwargs)

    monkeypatch.setattr(module, 'physical_reward_densities', recorded)
    batched = assembly.integrate(*args)
    assert len(calls) == 1
    sample, reference, parameters = calls[0]
    assert sample.passive_knee_limits_rad.shape == (2, 2)
    assert sample.policy_velocity_rad_s.shape == (8, 6)
    torch.testing.assert_close(parameters.mass_kg, mass.repeat(4), atol=0, rtol=0)
    torch.testing.assert_close(reference.admitted_body_command,
                               torch.cat([tick.reference.admitted_body_command for tick in ticks]), atol=0, rtol=0)
    # Unequal per-component weights also check the individual dictionary views;
    # the summed reward alone could hide a swapped or disconnected component.
    loss = (sum((index + 1) * value.sum() for index, value in enumerate(batched.components.values()))
            if weighted_components else batched.physical_reward.sum())
    gradients = torch.autograd.grad(loss, (action, perturbation, mass),
                                    retain_graph=True, allow_unused=True)
    # Exercise the preserved serial kernel path on exactly the same states,
    # targets, controller history and graph leaves; only batching differs.
    stack_rows = assembly._stack_rows
    monkeypatch.setattr(assembly, '_stack_rows', lambda values, **kw:
                        None if isinstance(values[0], RewardReference) else stack_rows(values, **kw))
    calls.clear()
    serial = assembly.integrate(*args)
    assert len(calls) == 4
    expected_loss = (sum((index + 1) * value.sum() for index, value in enumerate(serial.components.values()))
                     if weighted_components else serial.physical_reward.sum())
    expected_gradients = torch.autograd.grad(expected_loss, (action, perturbation, mass),
                                            allow_unused=True)
    assert batched.components.keys() == serial.components.keys()
    for name in batched.components:
        torch.testing.assert_close(batched.components[name], serial.components[name], atol=0, rtol=0)
    for name in ('physical_reward', 'failure_scale', 'wheel_torque_history', 'invalid'):
        torch.testing.assert_close(getattr(batched, name), getattr(serial, name), atol=0, rtol=0)
    torch.testing.assert_close(batched.memory.encode(), serial.memory.encode(), atol=0, rtol=0)
    for actual, expected in zip(gradients, expected_gradients):
        if expected is None:
            assert actual is None
        else:
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_changing_optional_tick_reference_keeps_serial_kernel_semantics(monkeypatch):
    import supermimic_rl.world_model_v2.reward_assembly as module

    _, assembly, geometry, state, legs, memory = setup('legacy')
    ids, owned = torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.bool)
    reference = RewardReference(torch.zeros(2, 4), ids, ids, torch.zeros(2), torch.zeros(2), torch.zeros(2, 2))
    ticks = []
    for index in range(4):
        tick_reference = (reference if index % 2 == 0 else replace(reference,
            gimbal=GimbalRewardReference(owned, torch.zeros(2), torch.zeros(2), torch.zeros(2))))
        ticks.append(RewardTick(state, geometry(state), torch.zeros(2), torch.zeros(2, 2),
            torch.zeros(2, 21), torch.zeros(2, 2), torch.zeros(2, 3), owned, ~owned,
            tick_reference, torch.full((2,), (index + 1) * .005), ids, ids, owned))
    calls, original = [], module.physical_reward_densities

    def recorded(*args, **kwargs):
        calls.append(args[1].gimbal is not None)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, 'physical_reward_densities', recorded)
    result = assembly.integrate(state, tuple(ticks), torch.zeros(2, 6), memory, torch.zeros(2, 3, 2),
                                legs, torch.zeros(2, 6), torch.ones(2, 6))
    assert calls == [False, True, False, True]
    assert 'positive.gimbal_velocity' in result.components
    assert not result.invalid.any()
