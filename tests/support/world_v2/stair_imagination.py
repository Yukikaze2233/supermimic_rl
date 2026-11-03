"""A stair-capable learner, its recorded start and scan-boundary placement."""

from dataclasses import asdict, replace
import json
import torch
from supermimic_rl.formal_observation import deployed_frame
from supermimic_rl.world_model_v2.features import MotionInputs
from supermimic_rl.world_model_v2.imagination import Critic
from supermimic_rl.world_model_v2.imagined_reward import TerrainLocomotionReward
from supermimic_rl.world_model_v2.learning import Learner, LearningConfig
from supermimic_rl.world_model_v2.model import WorldModelV2
from supermimic_rl.world_model_v2.motion_task import MotionTask, MOTION_TASK_DIM
from supermimic_rl.world_model_v2.policy import EstimatorActor
from supermimic_rl.world_model_v2.records import Memory, RecordSpec, encode_memory, encode_physical, encode_queues, decode_readout, decode_transforms
from supermimic_rl.world_model_v2.sensing import SensorQueues, encoder_imu_sample, gimbal_sample, heading_sample
from supermimic_rl.world_model_v2.stair_source import IdealStairSource, StairSourceState
from supermimic_rl.world_model_v2.stair_task import StairTask, StairTaskConfig, StairTaskState
from supermimic_rl.world_model_v2.termination import PoseContactLimits, PoseContactTermination
from support.world_v2.reward_assembly import setup, ROOT
from support.world_v2.stair_source import source_fixture
from support.world_v2.stair_task import native_stairs
from support.world_v2.terrain import atlas


def fixture(*, sensor_noise=False):
    torch.manual_seed(14)
    _, assembly, geometry, state, legs, reward_memory = setup('rough_dash')
    n, terrain = 2, atlas()
    state = replace(state, position=torch.tensor([[-.113, .03, .35], [-.133, .02, .35]]))
    model = WorldModelV2(MotionInputs(geometry.linkage), assembly.wheels, hidden=(16,), terrain=terrain)
    with torch.no_grad():
        model.motion.net[-1].weight.fill_(1e-4)
        ticks = model.motion.net[-1].bias.view(4, -1)
        ticks[:, 0] = torch.arange(1, 5) * .012
        ticks[:, 2] = torch.arange(1, 5) * .004
        ticks[:, 26:28] = 80.
        ticks[:, 28] = 40.  # Base contact is permitted only after the request condition admits it.
        for i, side in enumerate(geometry.sides):
            table = geometry.linkage.sides[side.name]
            raw = geometry.linkage.unpack(table.values[len(table.values) // 2])
            ticks[:, -4 + i] = raw['passive_joint'][side.passive_knee]
    actor = EstimatorActor(estimator_hidden=(8,), actor_hidden=(16,))
    with torch.no_grad():
        actor.estimator[-1].weight.zero_()
        actor.estimator[-1].bias.zero_()
        actor.estimator[-1].bias[-2:] = 2.
    critic = Critic(Critic.feature_size(16, 65, task_dim=MOTION_TASK_DIM, terrain=True, stairs=True),
                    (16,), task_dim=MOTION_TASK_DIM, terrain=True, stairs=True)
    source_pin = {'asset_manifest_sha256': geometry.linkage.provenance['asset_manifest_sha256'],
        'terrain_sha256': terrain.manifest_sha256,
        'reward_config': {'robot_mass_kg': assembly.parameters.tuning.robot_mass_kg,
            'gamma': assembly.parameters.tuning.gamma, 'profile': 'rough_dash', 'physics_dt_s': .005,
            'available_terms': list(assembly.available_terms)}, 'terrain_support_config': asdict(assembly.support_config)}
    reward = TerrainLocomotionReward(source_pin, geometry, model.wheels, .02, terrain=terrain)
    runtime = json.loads((ROOT/'configs/gpufree/runtime-physx64-16-geometry-audit.json').read_text())
    termination = PoseContactTermination(PoseContactLimits.from_runtime(runtime), .005)
    native = native_stairs(n, assist=True)
    native._terrain_course._stair_course_contact_allowed.zero_()
    stair = StairTaskState.capture(native)
    _, trigger = source_fixture(n)
    source = IdealStairSource(trigger.config, min_height_m=.04, max_height_m=.20, terrain=terrain)
    source_state = StairSourceState.from_trigger(trigger, torch.zeros(n, dtype=torch.int64))
    learner = Learner(model, actor, critic, termination, LearningConfig(critic_epochs=1, critic_minibatches=1),
        reward=reward, stair_task=StairTask(StairTaskConfig.capture(native)), stair_source=source)
    start = {name: torch.zeros(n, *shape) for name, shape in RecordSpec(16, 65).fields().items()}
    start.update(stair.record())
    start.update(source_state.record())
    start['stair_source_valid'] = torch.ones(n, 1)
    command = torch.tensor([[.8, 0., 0., .30]]).expand(n, -1)
    task = MotionTask(command, command, command, torch.tensor([[0., 0., 3., .5, .5]]).expand(n, -1),
        torch.tensor([[0., 3., 1., 0., 0.]]).expand(n, -1), torch.zeros(n, 6),
        torch.tensor([[.08, .38, 90.]]).expand(n, -1), torch.zeros(n, 4), torch.zeros(n, 5))
    start.update(motion_task=task.encode(), task_record_valid=torch.ones(n, 1),
                  terrain_tile=torch.ones(n, 1), task_family=torch.full((n, 1), 3.))
    start['physical'] = encode_physical(state)
    terrain_observation = terrain.observe(state, start['terrain_tile'][:, 0])
    start['height'], start['scan'] = terrain_observation.height_above_ground, terrain_observation.scan_relative
    start['memory'] = encode_memory(Memory(torch.zeros(n, 3, 2), torch.zeros(n, 6), torch.zeros(n, dtype=torch.int64),
                                          torch.zeros(n), torch.zeros(n), torch.zeros(n)))
    start['reward_memory'] = reward_memory.encode()[:, :13]
    start['support_memory'] = reward_memory.encode()[:, 13:]
    start['sim_context'][:, :21] = assembly.parameters.mass_kg[:, None] / 21.
    start['motor_strength'].fill_(1.)
    start['controller_parameters'] = torch.cat((legs.nominal, legs.stiffness, legs.damping,
                                               legs.effort_bounds, torch.zeros(n, 6)), -1)
    start['transforms'] = torch.cat((torch.zeros(n, 9), torch.eye(3).flatten().repeat(n, 1),
                                    torch.ones(n, 2), torch.zeros(n, 2)), -1)
    start['readout'][:, -6:-2] = state.joint_position[:, :4]
    if sensor_noise:
        start['readout'][:, 4] = torch.tensor([0., 1.])
        start['readout'][:, 5:23] = .01
        start['readout'][:, 24] = 1.
    start['reward_context_valid'].fill_(1.)
    start['failure_conditions_valid'].fill_(1.)
    start['horizon'][:] = torch.tensor([1., 1.])
    start['reward_route'][:, 1] = 3
    start['goal'][:, :4] = command * torch.tensor([1., 1., 1., 5.])
    start['goal'][:, 4] = start['goal'][:, 11] = 1.
    transforms = decode_transforms(start['transforms'])
    sample = encoder_imu_sample(state.joint_position[:, :4], state.joint_velocity[:, :6],
                                state.angular_velocity_body, state.gravity_body, transforms)
    gimbal = gimbal_sample(state.joint_position[:, 4:], state.joint_velocity[:, 6:], transforms)
    queues = SensorQueues(sample[:, None].expand(n, 3, -1).clone(), gimbal[:, None].expand(n, 3, -1).clone(),
                          heading_sample(torch.zeros(n))[:, None].expand(n, 3, -1).clone(), torch.full((n,), 3.))
    start['queues'] = encode_queues(queues)
    frame = deployed_frame(queues.values, queues.count, queues.gimbal, start['action'], start['goal'],
                           decode_readout(start['readout']), noise=False)
    history = frame[:, None].expand(n, 5, -1).clone()
    start = learner.imagination.prepare(start, 16)
    return learner, start, history


def move_to_scan_boundary(learner, start, rows):
    from supermimic_rl.world_model_v2.records import decode_physical

    start['physical'][rows, 0] = 4.45
    observed = learner.model.terrain.observe(decode_physical(start['physical']), start['terrain_tile'][:, 0])
    assert observed.valid.all()
    start['height'], start['scan'] = observed.height_above_ground, observed.scan_relative
