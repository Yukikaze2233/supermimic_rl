"""Native gimbal boundary and imagined command arbitration share request ownership."""
import ast
from dataclasses import replace
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from transformer_rl.formal.control.observation import CommandMode
from transformer_rl.formal.lab_env.course_tasks import LabCourseTasks
from transformer_rl.formal.tasks.command_pool import SegmentKind
from transformer_rl.formal.tasks.curriculum import TaskFamily
from transformer_rl.formal.tasks.geometry import TerrainKind
from supermimic_rl.course.gimbal_arbitration import (
    arbitration_receipt, conflicting_request_starts, install_request_gimbal_exclusion, request_owns_chassis,
)
from supermimic_rl.world_model_v2.motion_task import MotionTask
from test_world_v2_motion_task import native_tasks, sensors


def native_gimbal_boundary():
    # Execute the actual pinned method while avoiding Isaac imports. No source
    # substitution or separately reimplemented native eligibility predicate.
    path = Path(inspect.getfile(LabCourseTasks)).with_name('env.py')
    root = ast.parse(path.read_text())
    function = next(node for node in ast.walk(root)
                    if isinstance(node, ast.FunctionDef) and node.name == '_update_gimbal_course')
    namespace = dict(torch=torch, TerrainKind=TerrainKind, SegmentKind=SegmentKind, CommandMode=CommandMode)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['_update_gimbal_course']


@pytest.mark.parametrize('enabled', [False, True])
def test_native_command_steps_and_wm_agree_for_waiting_jump_physical_tail_and_ordinary(enabled):
    native = native_tasks(None)
    tasks, gimbal = native._terrain_course.tasks, native._gimbal_course
    tasks.family[:] = torch.tensor([int(TaskFamily.REQUEST)] * 3 + [int(TaskFamily.MOTION)] * 4)
    tasks.next_request_s.fill_(100.)
    tasks.physical_id[:] = torch.tensor([-1, 7, 8, 9, -1, -1, -1])
    gimbal.special.fill_(True)
    native.num_envs, native._device, native.step_dt = 7, torch.device('cpu'), .02
    servo_calls = []
    native._gimbal = SimpleNamespace(controller=SimpleNamespace(set_command=lambda *a, **k: servo_calls.append(k)))
    native.gimbal_measurement = lambda: SimpleNamespace(world_angles=torch.zeros(7, 2))
    native._curriculum_update = lambda: 5000
    receipt = install_request_gimbal_exclusion(native, enabled=enabled)
    assert receipt == (arbitration_receipt() if enabled else None)
    task = MotionTask.decode(MotionTask.capture(native))
    goal = torch.zeros(7, 16)
    goal[:, 4] = goal[:, 11] = 1.
    update = native_gimbal_boundary()
    for tick in range(3):
        queues, readout, _ = sensors(7, tick * 4)
        # This exercises the real native command stream/trajectory before its
        # exact gimbal boundary. Request modes are independently owned FSM
        # outputs; row 3 models a normal-mode external physical tail.
        admission, modes, _, _ = tasks.step(torch.full((7,), (tick + 1) * .02, dtype=torch.float64),
                                            None, torch.zeros(7), yaw_rad=torch.zeros(7))
        modes[1] = int(CommandMode.JUMP)
        tasks.physical_id[:] = torch.tensor([-1, 7, 8, 9, -1, -1, -1])
        # Compare the same admitted command: heading-control sensor parity has
        # independent coverage in test_world_v2_motion_task.
        task = replace(task, current=admission.admitted,
                       timing=torch.cat((tasks.elapsed[:, None], tasks.segment_clock[:, None],
                                         task.timing[:, 2:]), -1))
        native._command, native._command_mode = admission.admitted.clone(), modes
        update(native)
        task, goal = task.finish_command(admission.admitted, goal, mode=modes,
            request_owned=(request_owns_chassis(tasks.family, tasks.physical_id) if enabled else None))
        torch.testing.assert_close(goal[:, :3], native._command[:, :3])
        torch.testing.assert_close(goal[:, 3], native._command[:, 3] * 5)
        torch.testing.assert_close(goal[:, 11:], gimbal.context(torch.zeros(7))[:, :5])
        assert not gimbal.special[1]  # An active JUMP was already protected.
        if enabled:
            assert not gimbal.special[:4].any()
            torch.testing.assert_close(native._command[:4], admission.admitted[:4])
        else:
            assert gimbal.special[[0, 2, 3]].all()  # The original defect is reproducible.
        assert gimbal.special[4:].all()  # Ordinary spin-translation is retained.


def test_native_filter_preserves_independent_gimbal_draws_and_servo_targets():
    original, fixed = native_tasks(None), native_tasks(None)
    for native in (original, fixed):
        native._terrain_course.tasks.family.fill_(int(TaskFamily.REQUEST))
        native._terrain_course.tasks.physical_id.fill_(-1)
    install_request_gimbal_exclusion(fixed, enabled=True)
    kwargs = dict(update=5000, segment_boundary=torch.ones(7, dtype=torch.bool),
                  segment_remaining_s=torch.full((7,), 6.))
    # No special task is selected on these rows in either version; all genuine
    # gimbal motion samples and random draws must remain bitwise unchanged.
    for native in (original, fixed):
        native._gimbal_course.advance(.02, torch.zeros(7, 2), torch.zeros(7, dtype=torch.bool), **kwargs)
    for name in ('mode', 'target', 'rate', 'remaining_s', 'speed', 'direction', 'chassis_rate'):
        torch.testing.assert_close(getattr(original._gimbal_course, name), getattr(fixed._gimbal_course, name),
                                   atol=0, rtol=0)
    assert torch.equal(original._gimbal_course.generator.get_state(), fixed._gimbal_course.generator.get_state())


@pytest.mark.parametrize('enabled', [False, True])
def test_real_imagination_boundary_applies_source_selected_ownership(enabled):
    from support.world_v2.jump_imagination import fixture
    learner, start, frames = fixture()
    learner.imagination.request_gimbal_exclusion = enabled
    # A REQUEST waiting for its next scheduled ID. Use the recorded ordinary
    # segment's fields; do not add an Actor feature or change its H5 layout.
    start['jump_values'][:, 3] = 100.
    task = MotionTask.decode(start['motion_task'])
    task = replace(task, start=torch.tensor([[.5, 0., 0., .35]]).repeat(2, 1),
        end=torch.tensor([[.5, 0., 0., .35]]).repeat(2, 1),
        current=torch.tensor([[.5, 0., 0., .35]]).repeat(2, 1),
        timing=torch.tensor([[6., .1, 10., 0., 0.]]).repeat(2, 1),
        route=torch.tensor([[0., 0., 1., 0., 0.]]).repeat(2, 1),
        gimbal=torch.tensor([[1., .5, 0., 8., 5.]]).repeat(2, 1))
    start['motion_task'] = task.encode()
    # This synthetic old conflicting start is solely an arbitration probe.
    # Production migration excludes it from Actor starts without altering H5.
    result = learner.imagination.rollout(start, frames, 2)
    goals = result.critic_inputs[1:, :, 61:77]
    if enabled:
        torch.testing.assert_close(goals[:, :, 0], torch.full((2, 2), .5))
        assert goals[:, :, 2].eq(0).all() and goals[:, :, 12].eq(0).all()
    else:
        assert goals[:, :, 0].eq(0).all() and goals[:, :, 2].eq(8).all()
        assert goals[:, :, 12].eq(1).all()


def test_source_receipt_is_required_for_changed_command_semantics():
    from dataclasses import asdict
    from support.world_v2.jump_imagination import fixture
    from supermimic_rl.world_model_v2.imagined_reward import OrdinaryFlatReward
    learner, _, _ = fixture()
    old = learner.imagination.reward
    source = dict(asset_manifest_sha256=old.geometry.linkage.provenance['asset_manifest_sha256'],
        reward_config=dict(robot_mass_kg=old.tuning.robot_mass_kg, gamma=old.tuning.gamma,
            profile='rough_dash', physics_dt_s=.005, available_terms=list(old.terms)),
        terrain_support_config=asdict(old.support_config))
    assert not OrdinaryFlatReward(source, old.geometry, old.wheels, .02).request_gimbal_exclusion
    source['request_gimbal_exclusion'] = arbitration_receipt()
    assert OrdinaryFlatReward(source, old.geometry, old.wheels, .02).request_gimbal_exclusion
    source['request_gimbal_exclusion']['ownership'] = 'all rows'
    with pytest.raises(ValueError, match='native source'):
        OrdinaryFlatReward(source, old.geometry, old.wheels, .02)


def test_old_conflicting_starts_are_identified_without_rewriting_motion_or_history():
    from support.world_v2.jump_imagination import fixture
    learner, record, frames = fixture()
    storage = {key: value[None].repeat(3, *([1] * value.ndim)) for key, value in record.items()}
    storage['task_family'].fill_(int(TaskFamily.MOTION))
    storage['jump_ids'][..., 4].fill_(-1)
    storage['motion_task'][..., -5].zero_()
    storage['goal'][..., 12].zero_()
    # Two real ownership conditions; an ordinary spin and ordinary REQUEST
    # start must remain available. Goal-only conflicts are also excluded.
    storage['task_family'][0, 0, 0] = int(TaskFamily.REQUEST)
    storage['motion_task'][0, 0, -5] = 1.
    storage['jump_ids'][0, 1, 4] = 7
    storage['goal'][0, 1, 12] = 1.
    storage['motion_task'][1, 0, -5] = 1.
    storage['task_family'][1, 1, 0] = int(TaskFamily.REQUEST)
    snapshot = {key: value.clone() for key, value in storage.items()}
    mask = conflicting_request_starts(storage)
    assert mask.tolist() == [[True, True], [False, False], [False, False]]
    for key, original in snapshot.items():
        torch.testing.assert_close(storage[key], original, atol=0, rtol=0, equal_nan=True)
    # Only the existing task-start flag is cleared by migration; these flags
    # do not enter eligible physical dynamics windows (CompactReplay).
    storage['task_record_valid'] = storage['task_record_valid'].masked_fill(mask[..., None], 0.)
    assert storage['task_record_valid'].sum() == 4
    for key in ('physical', 'ticks', 'action', 'goal', 'motion_task', 'queues'):
        torch.testing.assert_close(storage[key], snapshot[key], atol=0, rtol=0, equal_nan=True)
