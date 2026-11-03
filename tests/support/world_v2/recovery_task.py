"""Native recovery lifecycle state built without Isaac."""

from types import SimpleNamespace
import torch
from transformer_rl.formal.control.observation import CommandMode
from transformer_rl.formal.lab_env.course_tasks import LabCourseTasks
from transformer_rl.formal.lab_env.recovery_lifecycle import RecoveryLifecycle, RecoveryLifecycleConfig
from transformer_rl.formal.tasks.command_pool import CommandPool, CommandPoolConfig
from transformer_rl.formal.tasks.curriculum import TaskFamily
from transformer_rl.formal.tasks.gimbal_course import GimbalCourse, GimbalCourseConfig
from transformer_rl.formal.tasks.trajectory import DriveEnvelope


def native_recovery(n=4, *, managed=True):
    tasks = LabCourseTasks(n, DriveEnvelope(.08, .38, 90.), policy_dt_s=.02, seed=93)
    tasks.family.fill_(int(TaskFamily.RECOVERY))
    tasks.base_mode.fill_(int(CommandMode.RECOVERY))
    if managed:
        tasks.command_pool = CommandPool(CommandPoolConfig(
            handoff_s=1.7, handoff_forward_m_s=.6, handoff_yaw_rad_s=1.3))
    lifecycle = RecoveryLifecycle(n, config=RecoveryLifecycleConfig(policy_dt_s=.02))
    lifecycle.reset(torch.arange(n), torch.ones(n, dtype=torch.bool))
    native = SimpleNamespace(physics_dt=.005, time=torch.zeros(n, dtype=torch.float64),
        _terrain_course=SimpleNamespace(tasks=tasks, recovery=lifecycle),
        _gimbal_course=GimbalCourse(GimbalCourseConfig(enabled=True), n))
    native._clock = lambda: native.time
    return native
