"""Jump take-off task: requests wait for the supported stance, cooldown and episode time."""
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('transformer_rl.formal.tasks.jump')
from transformer_rl.formal.tasks.curriculum import TaskFamily
from transformer_rl.formal.tasks.jump import JumpReferenceProgram

from supermimic_rl.course.jump_takeoff import JumpTakeoffConfig, install_jump_takeoff

REQUEST = int(TaskFamily.REQUEST)


class Tasks:
    """Rows: 0 settled, 1 unsupported, 2 height off, 3 not a request episode."""

    def __init__(self):
        n = 4
        self.policy_dt_s = .02
        self.elapsed = torch.full((n,), 2.)
        self.family = torch.tensor([REQUEST, REQUEST, REQUEST, 0])
        self.next_request_s = torch.full((n,), 2.01)
        self.protocol = SimpleNamespace(active_id=torch.full((n,), -1))
        self._last_supported = torch.tensor([True, False, True, True])
        self._last_world_height = torch.tensor([.32, .32, .36, .32])
        self.stream = SimpleNamespace(current=torch.tensor([[0., 0., 0., .32]]).repeat(n, 1))
        self.reference = JumpReferenceProgram()
        self.seen = []

    def step(self, now, estimate, ground, *, command=None, blocked_by_stair=None, yaw_rad=None):
        self.seen.append(self.next_request_s.clone())
        self.elapsed += self.policy_dt_s


def install(**config):
    tasks = Tasks()
    native = SimpleNamespace(_terrain_course=SimpleNamespace(tasks=tasks))
    return native, tasks, install_jump_takeoff(native, config)


def test_due_requests_wait_for_the_native_supported_stance():
    native, tasks, receipt = install()
    ground, now = torch.zeros(4), torch.full((4,), 10., dtype=torch.float64)
    tasks.step(now, None, ground)
    seen = tasks.seen[-1]
    assert seen[0] == pytest.approx(2.01)                       # settled: due as scheduled
    assert seen[1] == pytest.approx(2.04) and seen[2] == pytest.approx(2.04)   # postponed one interval
    assert seen[3] == pytest.approx(2.01)                       # not a request episode: untouched
    assert int(native.supermimic_jump_trigger.deferred) == 2
    tasks.step(now, None, ground, command=object())             # external requests are not gated
    assert tasks.seen[-1][1] == pytest.approx(2.04)
    assert receipt['reference']['compression_depth_max_m'] == .10
    assert tasks.reference.config.nominal_extension_duration_s == .20


def test_cooldown_and_minimum_episode_time():
    native, tasks, _ = install(cooldown_s=1.)
    ground = torch.zeros(4)
    tasks.protocol.active_id[0] = 3
    tasks.step(torch.full((4,), 10., dtype=torch.float64), None, ground)
    tasks.protocol.active_id[0] = -1          # the request ends inside the next native step
    tasks.next_request_s[0] = 99.
    tasks.step(torch.full((4,), 10.5, dtype=torch.float64), None, ground)
    tasks.next_request_s[0] = 0.
    tasks.step(torch.full((4,), 11., dtype=torch.float64), None, ground)    # 0.5 s after the end
    assert tasks.seen[-1][0] == pytest.approx(float(tasks.elapsed[0]) + tasks.policy_dt_s)
    tasks.next_request_s[0] = 0.
    tasks.step(torch.full((4,), 11.6, dtype=torch.float64), None, ground)
    assert tasks.seen[-1][0] == 0.
    early, early_tasks, _ = install(min_episode_s=3.)
    early_tasks.step(torch.full((4,), 10., dtype=torch.float64), None, ground)
    assert early_tasks.seen[-1][0] == pytest.approx(2.04)
    with pytest.raises(ValueError):
        JumpTakeoffConfig(compression_duration_s=0.)


def test_requests_in_progress_are_never_touched_by_the_gate():
    native, tasks, _ = install()
    tasks.protocol.active_id[:] = torch.tensor([5, 6, 7, -1])
    tasks._last_supported[:] = False                 # pushing: unsettled, as during a real take-off
    tasks.next_request_s[:] = 0.
    tasks.step(torch.full((4,), 10., dtype=torch.float64), None, torch.zeros(4))
    assert (tasks.seen[-1][:3] == 0.).all()          # active rows: scheduling untouched
    assert int(native.supermimic_jump_trigger.deferred) == 0
