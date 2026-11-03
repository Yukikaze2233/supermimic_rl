"""A recovery-capable learner with a recorded start for imagination tests."""

from dataclasses import replace
import torch
from supermimic_rl.world_model_v2.imagination import Critic
from supermimic_rl.world_model_v2.learning import Learner
from supermimic_rl.world_model_v2.motion_task import MotionTask, MOTION_TASK_DIM
from supermimic_rl.world_model_v2.recovery_task import RecoveryTask, RecoveryTaskConfig, RecoveryTaskState
from support.world_v2.jump_imagination import fixture as jump_fixture
from support.world_v2.recovery_task import native_recovery


def fixture(*, near_handoff=False, managed=True):
    old, start, frames = jump_fixture()
    native = native_recovery(2, managed=managed)
    native.time.fill_(.48)
    owner = native._terrain_course.recovery
    if near_handoff:
        owner.estimated_stable_s.fill_(.48)
        owner.physical_stable_s.fill_(.48)
    task = RecoveryTask(RecoveryTaskConfig.capture(native))
    start.update(RecoveryTaskState.capture(native).record(), recovery_record_valid=torch.ones(2, 1))
    start['task_family'].fill_(5.)
    start['reward_route'][:, 0] = 3.
    start['goal'][:, 4] = 0.
    start['goal'][:, 7] = 1.
    start['failure_conditions'][:, 0] = 1.
    motion = MotionTask.decode(start['motion_task'])
    route = motion.route.clone()
    route[:, 3] = 3.
    route[:, 4] = float(managed)
    start['motion_task'] = replace(motion, route=route).encode()
    # An already-expired segment must remain known while RECOVERY owns it.
    start['tof_segment_end'].zero_()
    start['stair_clock'][:, 0] = .48
    start['jump_clock'][:, 0] = .48
    critic = Critic(Critic.feature_size(16, 65, task_dim=MOTION_TASK_DIM,
        terrain=True, stairs=True, jumps=True, recovery=True), (16,), task_dim=MOTION_TASK_DIM,
        terrain=True, stairs=True, jumps=True, recovery=True)
    old.imagination.reward.recovery_task = task
    learner = Learner(old.model, old.actor, critic, old.imagination.termination, old.config,
        reward=old.imagination.reward, stair_task=old.imagination.stair_task,
        stair_source=old.imagination.stair_source, jump_task=old.imagination.jump_task, recovery_task=task)
    start = learner.imagination.prepare(start, 16)
    return learner, start, frames
