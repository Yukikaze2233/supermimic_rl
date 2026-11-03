"""A jump-capable learner with a recorded start for imagination tests."""

from dataclasses import replace
import torch
from supermimic_rl.formal_observation import deployed_frame
from supermimic_rl.world_model_v2.imagination import Critic
from supermimic_rl.world_model_v2.jump_task import JumpTask, JumpTaskConfig, JumpTaskState
from supermimic_rl.world_model_v2.learning import Learner
from supermimic_rl.world_model_v2.motion_task import MOTION_TASK_DIM, MotionTask
from supermimic_rl.world_model_v2.policy import EstimatorActor
from supermimic_rl.world_model_v2.records import decode_physical, decode_queues, decode_readout
from support.world_v2.jump_task import native_jumps
from support.world_v2.stair_imagination import fixture as stair_fixture


def fixture(*, gated=False):
    old, start, _ = stair_fixture()
    native = native_jumps(2)
    if gated:
        from supermimic_rl.course.jump_takeoff import install_jump_takeoff
        install_jump_takeoff(native, {'min_episode_s': .10, 'cooldown_s': .10})
    native._terrain_course.tasks.rise.fill_(.25)
    jump_task = JumpTask(JumpTaskConfig.capture(native))
    start.update(JumpTaskState.capture(native).record(), jump_record_valid=torch.ones(2, 1))
    start['task_family'].fill_(4.)
    start['reward_route'].zero_()
    start['goal'].zero_()
    start['goal'][:, 3] = .35 * 5
    start['goal'][:, 4] = start['goal'][:, 11] = 1
    start['physical'][:, 0] = -1.
    observation = old.model.terrain.observe(decode_physical(start['physical']), start['terrain_tile'][:, 0])
    start['height'], start['scan'] = observation.height_above_ground, observation.scan_relative
    motion = MotionTask.decode(start['motion_task'])
    command = torch.tensor([[0., 0., 0., .35]]).expand(2, -1)
    motion = replace(motion, start=command, end=command, current=command,
                     route=torch.zeros_like(motion.route))
    start['motion_task'] = motion.encode()
    with torch.no_grad():
        ticks = old.model.motion.net[-1].bias.view(4, -1)
        ticks[:, :3] = 0
        ticks[:, 28:52] = 0
        ticks[:, 49:51] = 80.  # Native resultant force, independent of support load.
    actor = EstimatorActor(estimator_hidden=(8,), actor_hidden=(16,), frame_dim=35)
    with torch.no_grad():
        actor.estimator[-1].weight.zero_()
        actor.estimator[-1].bias.zero_()
        actor.estimator[-1].bias[-2:] = 2.
    critic = Critic(Critic.feature_size(16, 65, task_dim=MOTION_TASK_DIM, terrain=True, stairs=True,
                                      jumps=True, jump_trigger=gated),
                    (16,), task_dim=MOTION_TASK_DIM, terrain=True, stairs=True, jumps=True, jump_trigger=gated)
    old.imagination.reward.jump_task = jump_task
    learner = Learner(old.model, actor, critic, old.imagination.termination, old.config,
        reward=old.imagination.reward, stair_task=old.imagination.stair_task,
        stair_source=old.imagination.stair_source, jump_task=jump_task)
    queues = decode_queues(start['queues'])
    frame = deployed_frame(queues.values, queues.count, queues.gimbal, start['action'], start['goal'],
                           decode_readout(start['readout']), noise=False)
    start = learner.imagination.prepare(start, 16)
    return learner, start, frame[:, None].expand(2, 5, -1).clone()
