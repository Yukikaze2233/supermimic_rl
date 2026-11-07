"""Real source-tree geometry agrees with differentiable reward reconstruction."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('transformer_rl.formal.tasks.reset_kinematics')
from transformer_rl.formal.control.observation import MechanicalBinding
from transformer_rl.formal.lab_env.terrain_support import TerrainSupportReference
from transformer_rl.formal.simulation.ground import WheelCollisionCentres
from transformer_rl.formal.tasks.reset_kinematics import AssetResetKinematics

from supermimic_rl.course.jump_reward_geometry import JumpRewardGeometry, NativeJumpRewardGeometry
from supermimic_rl.reward_kernel import RewardExtensions
from supermimic_rl.world_model_v2 import so3
from supermimic_rl.world_model_v2.geometry import RewardGeometry
from supermimic_rl.world_model_v2.imagined_reward import OrdinaryFlatReward
from supermimic_rl.world_model_v2.leg_linkage import LegLinkage
from supermimic_rl.world_model_v2.physical import PhysicalState


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / 'assets/full_vehicle_v2p0p1_source_audited_v2_knee40_107'
LINKAGE = ROOT / 'configs/linkage/leg_linkage_08bf7e00.json'


def source_tree(n=12):
    reward = JumpRewardGeometry.load(ASSET, LINKAGE)
    geometry = RewardGeometry(LegLinkage.load(LINKAGE), ASSET)
    manifest = json.loads((ASSET / 'manifest.json').read_text())
    binding = MechanicalBinding.from_manifest(manifest, manifest['tree_joint_names'],
                                              manifest['rigid_body_names'], include_auxiliary=False)
    native = AssetResetKinematics((ASSET / 'manifest.json').read_bytes(),
                                  (ASSET / 'model_spec.json').read_bytes(), binding)
    tree = native.nominal_tree_position[None].repeat(n, 1)
    active = torch.zeros(n, 6, dtype=torch.float64)
    for side in geometry.sides:
        table = geometry.linkage.sides[side.name]
        phi = torch.linspace(table.phi_min + .01, table.phi_max - .01, n, dtype=torch.float64)
        primary = torch.linspace(-2.3, 2.1, n, dtype=torch.float64)
        active[:, side.primary], active[:, side.secondary] = primary, primary + phi
        tree[:, binding.tree_joint_names.index(table.primary_joint)] = primary
        tree[:, binding.tree_joint_names.index(table.secondary_joint)] = primary + phi
        passive = geometry.linkage.evaluate(side.name, phi)[0]['passive_joint']
        for column, name in enumerate(table.passive_joint_names):
            tree[:, binding.tree_joint_names.index(name)] = passive[:, column]
    tree[:, list(binding.wheel_tree_indices)] = torch.linspace(-10., 10., n, dtype=torch.float64)[:, None]
    rotation = so3.exp(torch.linspace(-.6, .7, n * 3).double().reshape(n, 3))
    position = torch.tensor([.2, -.1, .55]).double().repeat(n, 1)
    state = PhysicalState(position, rotation, torch.zeros(n, 3).double(),
                          torch.zeros(n, 3).double(), active, torch.zeros(n, 8).double())
    knee_indices = [binding.tree_joint_names.index(
        geometry.linkage.sides[side.name].passive_joint_names[side.passive_knee])
        for side in geometry.sides]
    # Deliberately include small solver compliance, so parity cannot silently
    # substitute ideal closure for the measured passive knee coordinates.
    tree[:, knee_indices] += torch.tensor([.003, -.004]).double()
    return reward, geometry, native, binding, tree, state, knee_indices


def test_source_endpoint_lengths_span_entire_table_and_pin_both_inputs():
    reward, geometry, *_ = source_tree()
    torch.testing.assert_close(reward.minimum_leg_length_m,
        torch.tensor([.162463215903925, .162463215766070], dtype=torch.float64), atol=1e-12, rtol=0)
    for index, side in enumerate(geometry.sides):
        table = geometry.linkage.sides[side.name]
        points = geometry.linkage.unpack(table.values)['wheel_center_reference_base_m']
        lengths = (points - reward.hip_pivot_base_m[index]).norm(dim=-1)
        torch.testing.assert_close(lengths.amin(), reward.minimum_leg_length_m[index])
        torch.testing.assert_close(lengths.amax() - lengths.amin(), reward.leg_length_range_m[index])
    assert reward.receipt['asset_manifest_sha256'] == hashlib.sha256((ASSET / 'manifest.json').read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='explicit pin'):
        JumpRewardGeometry.load(ASSET, LINKAGE, expected_linkage_sha256='wrong')


def test_native_measured_wheel_cylinders_and_wm_true_state_geometry_agree():
    reward, geometry, native, binding, tree, state, knee_indices = source_tree()
    truth = native.forward(tree)
    bodies = list(binding.wheel_body_indices)
    links = truth.body_position_m[:, bodies]
    world_position = state.position[:, None] + (state.rotation[:, None] @ links[..., None]).squeeze(-1)
    world_rotation = state.rotation[:, None] @ truth.body_rotation[:, bodies]
    poses = torch.cat((world_position, so3.matrix_to_quaternion(world_rotation)), -1).float()
    cylinders = WheelCollisionCentres.from_urdf((ASSET / 'robot.urdf').read_bytes(), binding)
    observer = TerrainSupportReference(len(tree), cylinders, physics_dt_s=.005)
    support = observer.observe(state.position.float(), poses, torch.zeros(len(tree), 2),
                               torch.ones(len(tree), dtype=torch.bool))
    # Unequal wheel columns expose accidental root-ground or min/max reuse.
    ground = torch.tensor([.04, .20]).double().expand(len(tree), -1)
    course = SimpleNamespace(_support_sample=support, wheel_ground=ground,
                             query_invalid=torch.zeros(len(tree), dtype=torch.bool))
    measured = NativeJumpRewardGeometry(reward, course)(SimpleNamespace(wheel_position_base_m=links))
    wheel = geometry(state, passive_knee_position=tree[:, knee_indices],
                     passive_knee_velocity=torch.zeros(len(tree), 2).double())
    imagined = reward.sample(wheel.link_position_base, wheel.bottom_world_z, ground, wheel.valid)
    for name in ('wheel_clearance_m', 'leg_length_m', 'minimum_leg_length_m', 'leg_length_range_m'):
        torch.testing.assert_close(getattr(measured, name), getattr(imagined, name), atol=2e-7, rtol=0)
    assert measured.valid.all() and imagined.valid.all()


def test_geometry_through_actual_fk_and_shared_reward_has_useful_gradients():
    from test_jump_flight_objective import evaluate
    reward, geometry, _, _, _, state, _ = source_tree(n=2)
    # Keep the cylinder off the ground and off abs/min branch boundaries.
    state = replace(state, position=state.position + torch.tensor([0., 0., .4]))
    coordinates = torch.cat((state.position, state.joint_position), -1).requires_grad_()

    def objective(value):
        physical = replace(state, position=value[:, :3], joint_position=value[:, 3:])
        wheel = geometry(physical)
        flight = reward.sample(wheel.link_position_base, wheel.bottom_world_z,
                               torch.tensor([[0., .03], [.02, 0.]]).double(), wheel.valid)
        result = evaluate(flight)
        return (result.positive['jump_flight'] + result.positive['jump_tuck']).sum()

    gradient = torch.autograd.grad(objective(coordinates), coordinates)[0]
    assert torch.isfinite(gradient).all()
    assert (gradient[:, 2] > 0).all()
    assert gradient[:, 3:7].abs().sum() > 0
    direction = torch.linspace(-.3, .4, coordinates.numel()).double().reshape_as(coordinates)
    epsilon = 1e-6
    numeric = (objective(coordinates + epsilon * direction)
               - objective(coordinates - epsilon * direction)) / (2 * epsilon)
    torch.testing.assert_close((gradient * direction).sum(), numeric, atol=2e-7, rtol=1e-5)
    # A full revolution of both hip motors changes neither virtual leg length.
    rotated = state.joint_position.clone()
    rotated[:, :4] += 2 * torch.pi
    before, after = geometry(state), geometry(replace(state, joint_position=rotated))
    l0 = reward.sample(before.link_position_base, before.bottom_world_z, torch.zeros(2, 2), before.valid)
    l1 = reward.sample(after.link_position_base, after.bottom_world_z, torch.zeros(2, 2), after.valid)
    torch.testing.assert_close(l0.leg_length_m, l1.leg_length_m, atol=1e-10, rtol=0)


def test_unavailable_native_row_is_masked_before_differentiable_arithmetic():
    reward, geometry, _, _, _, state, _ = source_tree(n=2)
    wheel = geometry(state)
    link = wheel.link_position_base.detach().clone()
    link[0] = float('nan')
    link.requires_grad_()
    bottom = wheel.bottom_world_z.detach().clone().requires_grad_()
    result = reward.sample(link, bottom, torch.zeros(2, 2), torch.tensor([False, True]))
    loss = result.leg_length_m.sum() + result.wheel_clearance_m.sum()
    gradients = torch.autograd.grad(loss, (link, bottom))
    assert result.valid.tolist() == [False, True]
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient[0].count_nonzero() == 0


def test_imagined_reward_binds_native_receipt_and_disabled_extension_keeps_old_source():
    from support.world_v2.jump_imagination import fixture
    learner, start, _ = fixture()
    old = learner.imagination.reward
    geometry = JumpRewardGeometry(old.geometry)
    extension = RewardExtensions(jump_flight=4., jump_flight_clearance=True, jump_tuck=1.7)
    source = {'asset_manifest_sha256': old.geometry.linkage.provenance['asset_manifest_sha256'],
        'reward_config': {'robot_mass_kg': old.tuning.robot_mass_kg, 'gamma': old.tuning.gamma,
        'profile': 'rough_dash', 'physics_dt_s': .005, 'available_terms': list(old.terms)},
        'terrain_support_config': asdict(old.support_config),
        'reward_kernel': {'extensions': asdict(extension), 'jump_flight_geometry': geometry.receipt}}
    current = OrdinaryFlatReward(source, old.geometry, old.wheels, .02)
    assert current.jump_geometry.receipt == geometry.receipt
    broken = json.loads(json.dumps(source))
    broken['reward_kernel']['jump_flight_geometry']['linkage_data_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='native source receipt'):
        OrdinaryFlatReward(broken, old.geometry, old.wheels, .02)
    broken['reward_kernel']['extensions'] = {'jump_flight': 4.}
    assert OrdinaryFlatReward(broken, old.geometry, old.wheels, .02).jump_geometry is None


def test_four_tick_imagination_produces_geometry_without_peak_completion():
    from support.world_v2.jump_imagination import fixture
    from supermimic_rl.world_model_v2.jump_task import JumpTaskState
    from supermimic_rl.world_model_v2.model import Ticks
    from supermimic_rl.world_model_v2.records import decode_memory, decode_physical
    from supermimic_rl.world_model_v2.reward_assembly import RewardMemory
    learner, start, _ = fixture(gated=True)
    producer = learner.imagination.reward
    producer.extensions = RewardExtensions(jump_flight=4., jump_flight_clearance=True, jump_tuck=1.7)
    producer.jump_geometry = JumpRewardGeometry(producer.geometry)
    producer.jump_task.config = replace(producer.jump_task.config, flight_latch=True)
    start['jump_flight_latch'] = torch.tensor([[7, 1, 0], [7, 1, 0]], dtype=torch.int64)
    jump = JumpTaskState.decode(start)
    jump.ids[:, 1] = jump.ids[:, 4] = 7
    jump.values[:, [1, 2, 9]] = 0.
    jump.values[:, 5] = .10
    jump.values[:, 6] = .35
    jump.values[:, 8] = 3.
    jump.values[:, [10, 11]] = 1.
    start['goal'][:, 4] = 0.
    start['goal'][:, 8] = 1.
    start['reward_route'][:, 0] = 4.
    state = decode_physical(start['physical'])
    state = replace(state, position=state.position + torch.tensor([0., 0., .35]))
    labels = torch.zeros_like(start['ticks'])
    for index, side in enumerate(producer.geometry.sides):
        q = state.joint_position
        values, _ = producer.geometry.linkage.geometry_base(side.name, q[:, side.primary],
                                                            q[:, side.secondary] - q[:, side.primary])
        labels[..., -4 + index] = values['passive_joint'][:, side.passive_knee, None]
    ticks = Ticks.decode(labels, state)
    context = producer.bind(start, state, known_interval=True)
    memory = RewardMemory.decode(torch.cat((start['reward_memory'], start['support_memory']), -1))
    result = context.integrate(state, SimpleNamespace(ticks=ticks, tick_terrain_valid=None),
        start['action'], memory, decode_memory(start['memory']), start['goal'], 0,
        jump_state=jump, base_mode=torch.zeros(2, dtype=torch.long))
    assert not producer.extensions.jump_peak_completion
    assert (result.components['positive.jump_flight'] > 0).all()
    assert (result.components['positive.jump_tuck'] > 0).all()


def test_four_tick_native_reward_matches_world_assembly_and_stops_tuck_on_touchdown():
    from transformer_rl.formal.control.observation import CommandMode
    from transformer_rl.formal.tasks.jump import ReferenceStage
    from transformer_rl.formal.tasks.reward import RewardPhase, RewardReference
    from supermimic_rl.course.reward import install_shared_reward
    from supermimic_rl.reward_kernel import JumpProgress
    from supermimic_rl.world_model_v2.reward_assembly import RewardTick
    from support.world_v2.reward_assembly import setup

    native, assembly, geometry, state, legs, memory = setup('legacy')
    helper = JumpRewardGeometry(geometry)
    extension = RewardExtensions(jump_flight=4., jump_flight_clearance=True, jump_tuck=1.7)
    stage = torch.full((2,), int(ReferenceStage.BALLISTIC))
    owner = SimpleNamespace(jump_reference=lambda: SimpleNamespace(clock_stage=stage),
        ascent_seen=torch.ones(2, dtype=torch.bool), physical_id=torch.zeros(2, dtype=torch.long),
        air_ticks=torch.zeros(2, dtype=torch.long), landed_s=torch.zeros(2),
        peak_rise=torch.full((2,), .10), active_rise=torch.full((2,), .10),
        observe_physics=lambda *args, **kwargs: None)
    course = SimpleNamespace(_support_sample=None, wheel_ground=torch.zeros(2, 2),
                             query_invalid=torch.zeros(2, dtype=torch.bool))
    receipt = install_shared_reward(native, extensions=extension, jump_tasks=owner,
                                    flight_geometry=NativeJumpRewardGeometry(helper, course))
    assert receipt['jump_flight_geometry'] == helper.receipt
    assembly.parameters = replace(assembly.parameters, extensions=extension)
    ids, owned = torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.bool)
    mode = torch.full_like(ids, int(CommandMode.JUMP))
    command = torch.tensor([[0., 0., 0., .32]]).repeat(2, 1)
    action = torch.zeros(2, 6)
    native.begin(action, state.joint_velocity[:, :6], now_s=native._time, episode_id=ids,
        control_epoch=ids, policy_owned=owned, admitted_body_command=command, command_mode=mode)
    ticks = []
    for index in range(4):
        closed = torch.full((2,), index >= 2)
        phase = RewardPhase.FLIGHT if index < 2 else RewardPhase.LANDING
        reference = RewardReference(command, mode, torch.full_like(ids, int(phase)),
                                    state.position[:, 2], torch.zeros(2), torch.zeros(2, 2))
        wheel = geometry(state)
        flight = helper.sample(wheel.link_position_base, wheel.bottom_world_z, course.wheel_ground, wheel.valid)
        ticks.append(RewardTick(state, wheel, torch.zeros(2), course.wheel_ground,
            torch.zeros(2, 21), torch.zeros(2, 2), torch.zeros(2, 3), owned, ~owned, reference,
            native._time + (index + 1) * .005, ids, ids, owned,
            jump_progress=JumpProgress(stage, owned, closed, owner.landed_s, owned,
                owner.peak_rise, owner.active_rise, flight_geometry=flight)))
    queue = torch.zeros(2, 3, 2)
    result = assembly.integrate(state, tuple(ticks), action, memory, queue, legs,
                                torch.zeros(2, 6), torch.ones(2, 6))
    target, correction = legs.decode_target(action, state.joint_position[:, :4])
    support = memory.support
    for index, tick in enumerate(ticks):
        owner.air_ticks.fill_(2 if index < 2 else 0)
        owner.observe_physics()
        course._support_sample = SimpleNamespace(wheel_bottom_world_z_m=tick.geometry.bottom_world_z,
                                                valid=tick.geometry.valid)
        speed = state.joint_velocity[:, 4:6]
        queue = assembly.wheels.push(queue, assembly.wheels.motor_input(
            assembly.wheels.target(action), speed, torch.ones(2, 2)))
        effort = torch.cat((legs.motor_effort(target, state.joint_position, state.joint_velocity,
                                              torch.ones(2, 6)), assembly.wheels.applied(queue, speed)), -1)
        reference, support = assembly.advance_reference(tick, support)
        native.observe(assembly.physics(tick, effort, correction), reference)
    expected = native.finish(terminated=~owned, truncated=~owned,
        physical_success_request_id=torch.full_like(ids, -1), entire_skill_policy_owned=owned)
    for name, value in expected.component_densities.items():
        torch.testing.assert_close(result.components[name], value * .02, atol=1e-12, rtol=1e-12)
    assert (result.components['positive.jump_tuck'] > 0).all()
    assert not result.invalid.any()
