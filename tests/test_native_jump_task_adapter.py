"""The shared selector executes at the real native command boundary."""
import pytest
import torch

pytest.importorskip('transformer_rl.formal.lab_env.course_tasks')
from transformer_rl.formal.lab_env.course_tasks import LabCourseTasks, RequestCommand
from transformer_rl.formal.tasks.protocol import CompletedEstimate, RequestResponse

from supermimic_rl.course.jump_takeoff import install_jump_takeoff
from supermimic_rl.course.jump_task_adapter import NativeJumpTaskAdapter
from supermimic_rl.course.reward import install_flight_latch
from tests.support.world_v2.jump_task import native_jumps

pytestmark = pytest.mark.skipif(not hasattr(LabCourseTasks, '_select_request'),
                                reason='Requires the new private native request-selection entry point')


def estimate(now):
    n = len(now)
    return CompletedEstimate(torch.zeros(n), torch.ones(n, 2), torch.zeros(n),
                             torch.zeros(n), now - .02, torch.ones(n, dtype=torch.bool))


def test_owned_jump_keeps_other_rows_automatic_and_preserves_their_gate():
    native = native_jumps(4)
    tasks = native._terrain_course.tasks
    NativeJumpTaskAdapter(tasks)
    install_jump_takeoff(native, {})
    tasks.elapsed.fill_(2.)
    tasks.next_request_s.zero_()
    tasks.trajectory.start[:, 3] = .32
    tasks.trajectory.end[:, 3] = .32
    tasks.stream.current[:, 3] = .32
    tasks._last_supported[:] = torch.tensor([False, True, False, True])
    now = torch.full((4,), 2., dtype=torch.float64)
    tasks.protocol.last_time_s.copy_(now - .02)
    owned = torch.tensor([True, False, False, True])
    packet = RequestCommand(torch.tensor([50, 900, 900, 51]), torch.full((4,), .1))
    expected_rng = torch.Generator().set_state(tasks.generator.get_state())
    torch.rand(4, generator=expected_rng)

    tasks.step(now, estimate(now), torch.zeros(4), command=packet, packet_owned=owned)

    assert tasks.protocol.active_id.tolist() == [50, 0, -1, 51]
    assert tasks.protocol.max_seen_id[1:3].tolist() == [0, -1]
    assert tasks.next_request_s[0] == tasks.next_request_s[3] == 0
    assert tasks.next_request_s[2] == pytest.approx(2.04)
    assert tasks.request_counters[RequestResponse.ACCEPTED] == 3
    assert tasks.external_request_counters[RequestResponse.ACCEPTED] == 2
    assert native.supermimic_jump_trigger.deferred == 1
    assert torch.equal(tasks.generator.get_state(), expected_rng.get_state())


@pytest.mark.parametrize('external', [False, True])
def test_default_adapter_matches_native_state_and_random_draws(external):
    before = native_jumps(3)._terrain_course.tasks
    after = native_jumps(3)._terrain_course.tasks
    NativeJumpTaskAdapter(after)
    for tick in range(1, 8):
        now = torch.full((3,), tick * .02, dtype=torch.float64)
        packet = (RequestCommand(torch.full((3,), tick, dtype=torch.long), torch.full((3,), .1))
                  if external else None)
        old = before.step(now, estimate(now), torch.zeros(3), command=packet)
        new = after.step(now, estimate(now), torch.zeros(3), command=packet)
        for key, value in vars(before).items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(getattr(after, key), value, rtol=0, atol=0)
        for key, value in vars(before.protocol).items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(getattr(after.protocol, key), value, rtol=0, atol=0)
        assert torch.equal(before.generator.get_state(), after.generator.get_state())
        torch.testing.assert_close(new[0].admitted, old[0].admitted, rtol=0, atol=0)
        for expected, actual in zip(old[1:], new[1:]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_shared_physics_runs_once_before_the_existing_first_flight_latch():
    tasks = native_jumps(2)._terrain_course.tasks
    NativeJumpTaskAdapter(tasks)
    latch = install_flight_latch(tasks)
    tasks.physical_id.fill_(7)
    tasks.active_id.fill_(7)
    tasks.latched_height.fill_(.32)
    tasks.ground_anchor.zero_()
    for expected_ticks in (1, 2):
        tasks.observe_physics(torch.zeros(2, 2), torch.ones(2), torch.zeros(2),
                              torch.full((2,), .36), angular_speed_rad_s=torch.zeros(2))
        assert tasks.air_ticks.eq(expected_ticks).all()
    assert latch.lifted.all() and not latch.closed.any()
    tasks.observe_physics(torch.full((2, 2), 100.), torch.zeros(2), torch.zeros(2),
                          torch.full((2,), .32), angular_speed_rad_s=torch.zeros(2))
    assert latch.closed.all()
