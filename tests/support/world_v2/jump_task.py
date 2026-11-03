"""Native jump-request protocol state built without Isaac."""

from types import SimpleNamespace
import torch
from transformer_rl.formal.lab_env.course_tasks import LabCourseTasks
from transformer_rl.formal.tasks.curriculum import TaskFamily
from transformer_rl.formal.tasks.trajectory import DriveEnvelope


def native_jumps(n=1, *, origin_z=0.):
    tasks = LabCourseTasks(n, DriveEnvelope(.08, .38, 90.), policy_dt_s=.02, seed=71)
    tasks.family.fill_(int(TaskFamily.REQUEST))
    tasks.rise.fill_(.06)
    tasks.protocol.last_time_s.fill_(0.)
    tasks._last_supported.fill_(True)
    tasks._last_world_height.fill_(.32 + origin_z)
    origins = torch.zeros(n, 3)
    origins[:, 2] = origin_z
    native = SimpleNamespace(physics_dt=.005, time=torch.zeros(n, dtype=torch.float64),
        _terrain_course=SimpleNamespace(tasks=tasks, router=SimpleNamespace(origins=origins),
                                       _pending_requests={'jump': None}))
    native._clock = lambda: native.time
    return native
