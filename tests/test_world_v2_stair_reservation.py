"""Platform caller reservation changes ToF submission, not sensing or packets."""
from dataclasses import fields, is_dataclass

import pytest
import torch

from supermimic_rl.formal_observation import SensorReadout
from supermimic_rl.world_model_v2.motion_task import MotionTask
from supermimic_rl.world_model_v2.sensing import SensorQueues
from supermimic_rl.world_model_v2.stair_source import StairSourceState
from supermimic_rl.world_model_v2.stair_task import StairPacket, StairTask, StairTaskConfig, StairTaskState
from support.world_v2.stair_source import source_fixture
from support.world_v2.stair_task import estimate_at, native_stairs
from support.world_v2.terrain import state as physical_state
from transformer_rl.formal.tasks.protocol import RequestEnd, RequestResponse


def fixture(n=4):
    source, trigger = source_fixture(n)
    native = native_stairs(n, labels=True, assist=True)
    task = StairTask(StairTaskConfig.capture(native))
    command = torch.tensor([[.8, 0., 0., .30]]).expand(n, -1).clone()
    motion = MotionTask(command, command, command,
        torch.tensor([[0., 0., 5., .5, .5]]).expand(n, -1).clone(),
        torch.tensor([[0., 3., 1., 0., 0.]]).expand(n, -1).clone(), torch.zeros(n, 6),
        torch.tensor([[.08, .38, 90.]]).expand(n, -1).clone(), torch.zeros(n, 4), torch.zeros(n, 5))
    goal = torch.zeros(n, 16)
    goal[:, :4] = command * torch.tensor([1., 1., 1., 5.])
    goal[:, 4] = goal[:, 11] = 1.
    queues = SensorQueues(torch.zeros(n, 3, 18), torch.zeros(n, 3, 5),
                          torch.tensor([1., 0.]).expand(n, 3, 2).clone(), torch.full((n,), 3.))
    readout = SensorReadout(torch.zeros(n, 4), torch.zeros(n, 1, dtype=torch.bool),
                            torch.zeros(n, 18), torch.zeros(n, 1), torch.zeros(n, 6))
    source_state = StairSourceState.from_trigger(trigger, torch.zeros(n, dtype=torch.int64))
    stair = StairTaskState.capture(native)
    physical = physical_state(torch.tensor([[-.113, .0, .4]]).expand(n, 3).clone())
    kwargs = dict(task=task, physical=physical, wheel_centres=physical.position[:, None]
        + torch.tensor([[[0., -.2, -.3], [0., .2, -.3]]]), wheel_radius=.08,
        terrain_tile=torch.zeros(n, dtype=torch.int64), queues=queues, readout=readout,
        physics_dt_s=.005, latency=torch.full((n,), 2, dtype=torch.int64), draws=torch.full((n, 7), .5))
    return source, source_state, stair, motion, goal, kwargs


def advance(source, source_state, stair, motion, goal, kwargs, **options):
    result = source.advance(source_state, stair, motion=motion, goal=goal,
        estimate=estimate_at(stair.clock[:, 0], len(goal)), **kwargs, **options)
    return result, (result.source, result.boundary.stair.state, result.boundary.motion, result.boundary.goal)


def same(actual, expected, rows=None):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual if rows is None else actual[rows],
                                   expected if rows is None else expected[rows], rtol=0, atol=0)
    elif is_dataclass(expected):
        for field in fields(expected):
            same(getattr(actual, field.name), getattr(expected, field.name), rows)
    else:
        assert actual == expected


def test_none_and_false_mask_preserve_every_default_source_and_boundary_value():
    source, *states, kwargs = fixture()
    plain, explicit, masked = states, states, states
    for _ in range(12):
        baseline, plain = advance(source, *plain, kwargs)
        none, explicit = advance(source, *explicit, kwargs, reservation=None)
        false, masked = advance(source, *masked, kwargs, reservation=torch.zeros(4, dtype=torch.bool))
        same(none, baseline)
        same(false, baseline)


def test_mixed_reservation_keeps_detection_and_latency_running_then_releases_without_extra_delay():
    source, *states, kwargs = fixture()
    original = {name: value.clone() for name, value in states[0].record().items()}
    reserved = torch.tensor([True, False, True, False])
    masked, baseline = states, states
    for tick in range(3):
        ordinary, baseline = advance(source, *baseline, kwargs)
        actual, masked = advance(source, *masked, kwargs, reservation=reserved)
        same(actual, ordinary, rows=~reserved)
        if tick < 2:
            assert not actual.fired.any()
            assert actual.source.ids[:, 1].tolist() == [1-tick] * 4
            assert not actual.source.flags[:, 0].any()  # Detection already disarmed all rows.
        else:
            assert actual.fired.tolist() == [False, True, False, True]
            assert actual.source.ids[:, 1].tolist() == [0, -1, 0, -1]
            assert actual.boundary.stair.protocol_active_id.tolist() == [-1, 0, -1, 0]
    # Reservation ends after the existing latency has elapsed; no restart or
    # additional waiting period is introduced by caller ownership.
    released, _ = advance(source, *masked, kwargs, reservation=torch.zeros(4, dtype=torch.bool))
    assert released.fired.tolist() == [True, False, True, False]
    assert released.packet.request_id.tolist() == [0, -1, 0, -1]
    for name, value in original.items():
        assert torch.equal(states[0].record()[name], value)


@pytest.mark.parametrize('kind', ['cancel', 'complete'])
def test_reservation_does_not_swallow_external_end_or_stop_rearming(kind):
    source, *states, kwargs = fixture(2)
    for _ in range(3):
        accepted, states = advance(source, *states, kwargs)
    assert accepted.boundary.stair.protocol_active_id.tolist() == [0, 0]
    packet = StairPacket.empty(2, 'cpu')
    getattr(packet, 'cancel_id' if kind == 'cancel' else 'completed_id')[0] = 0
    ended, _ = advance(source, *states, kwargs, reservation=torch.ones(2, dtype=torch.bool), external=packet)
    assert not ended.fired.any()
    assert ended.boundary.stair.finished_id.tolist() == [0, -1]
    expected = RequestEnd.CANCELLED if kind == 'cancel' else RequestEnd.ESTIMATED_COMPLETE
    assert ended.boundary.stair.end_reason.tolist() == [int(expected), int(RequestEnd.NONE)]
    assert ended.source.flags[0, 0]  # The original completion/cancel rearm ran.
    assert ended.source.ids[0, 1] == -1
    assert ended.boundary.stair.protocol_active_id.tolist() == [-1, 0]


def test_reserved_source_still_delivers_an_explicit_external_request():
    source, *states, kwargs = fixture(2)
    packet = StairPacket.empty(2, 'cpu')
    packet.request_id[0], packet.height_m[0] = 17, .12
    actual, _ = advance(source, *states, kwargs, reservation=torch.ones(2, dtype=torch.bool), external=packet)
    assert not actual.fired.any()
    assert actual.packet.request_id.tolist() == [17, -1]
    assert actual.boundary.stair.response.tolist() == [int(RequestResponse.ACCEPTED), int(RequestResponse.NONE)]
    assert actual.boundary.stair.protocol_active_id.tolist() == [17, -1]


@pytest.mark.parametrize('invalid', [torch.zeros(4), torch.zeros(4, 1, dtype=torch.bool),
                                   torch.zeros(3, dtype=torch.bool), [False]*4])
def test_reservation_requires_matching_boolean_rows(invalid):
    source, *states, kwargs = fixture()
    original = {name: value.clone() for name, value in states[0].record().items()}
    with pytest.raises(ValueError, match='reservation'):
        advance(source, *states, kwargs, reservation=invalid)
    for name, value in original.items():
        assert torch.equal(states[0].record()[name], value)
