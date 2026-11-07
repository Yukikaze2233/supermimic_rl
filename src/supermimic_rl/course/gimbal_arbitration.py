"""Give jump-owned chassis commands priority over sampled spin-translation."""
from __future__ import annotations

import hashlib
from pathlib import Path


def request_owns_chassis(task_family, physical_request_id):
    """Waiting REQUEST births and unfinished physical requests keep their motion."""
    from transformer_rl.formal.tasks.curriculum import TaskFamily

    return (task_family == int(TaskFamily.REQUEST)) | (physical_request_id >= 0)


def arbitration_receipt():
    return {
        'implementation': 'request_owned_gimbal_exclusion_v1',
        'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'ownership': 'task_family == REQUEST or physical_request_id >= 0',
        'scope': 'exclude chassis spin-translation; independent gimbal servo remains active',
    }


def conflicting_request_starts(storage):
    """Identify old Actor starts without rewriting their physical data or H5.

    The migration may clear only ``task_record_valid`` for these rows. The
    dynamics fitter keeps the original transition and all recorded commands.
    """
    from ..world_model_v2.jump_task import JUMP_ID_NAMES
    from ..world_model_v2.motion_task import MOTION_TASK_DIM, MotionTask

    leading = storage['task_record_valid'].shape[:-1]
    for name, trailing in (('task_family', (1,)), ('jump_ids', (len(JUMP_ID_NAMES),)),
                           ('motion_task', (MOTION_TASK_DIM,)), ('goal', (16,)),
                           ('task_record_valid', (1,))):
        if storage[name].shape != (*leading, *trailing):
            raise ValueError('Request ownership migration has an incompatible recorded ' + name)
    physical_id = storage['jump_ids'][..., JUMP_ID_NAMES.index('physical_id')]
    owned = request_owns_chassis(storage['task_family'][..., 0], physical_id)
    special = MotionTask.decode(storage['motion_task']).gimbal[..., 0] > .5
    return owned & (special | (storage['goal'][..., 12] > .5))


def install_request_gimbal_exclusion(native, *, enabled=False):
    """Filter the instance's existing sampler input; preserve its draws and servo."""
    if type(enabled) is not bool:
        raise ValueError('Request gimbal exclusion must be a boolean')
    if not enabled:
        return None
    course, tasks = native._gimbal_course, native._terrain_course.tasks
    if course is None:
        raise ValueError('Request gimbal exclusion requires an installed gimbal course')
    advance = course.advance

    def advance_owned(dt_s, world_angles, eligible, **kwargs):
        return advance(dt_s, world_angles,
                       eligible & ~request_owns_chassis(tasks.family, tasks.physical_id), **kwargs)

    course.advance = advance_owned
    return arbitration_receipt()
