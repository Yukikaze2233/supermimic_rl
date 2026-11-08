"""Course checkpoint ownership and cold reset route selection stay connected."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.course.adapter import CourseAdapter
from test_platform_sampler import native_owner, reset


def adapter(tmp_path, *, platform=True):
    native, owner = native_owner(tmp_path)
    reset(native, [0, 1, 2], [101, 102, 0], [3, 3, 1])
    owner.caller.state = replace(owner.caller.state,
        phase=torch.tensor([3, 1, 0]), bound_request_id=torch.tensor([7, -1, -1]))
    trace = []
    if platform:
        native.supermimic_platform_course = owner
    native._terrain_course.state_dict = lambda: {'requests': 'saved'}
    native._terrain_course.restore_after_physical_reset = lambda value: trace.append('course')
    native.control_timing_state_dict = lambda: {'control': 'saved'}
    native.restore_control_timing = lambda value: trace.append('control')
    native._sensor_model = SimpleNamespace(
        initialized=torch.ones(3, dtype=torch.bool), count=torch.ones(3, dtype=torch.long),
        _sample='previous', _sample_tick=7,
        checkpoint_state_dict=lambda: {'sensors': 'saved'},
        load_after_explicit_cold_setup=lambda value: trace.append('sensors'))
    result = CourseAdapter.__new__(CourseAdapter)
    result.env = native
    result.curriculum = SimpleNamespace(state_dict=lambda: {'curriculum': 'saved'},
                                       load_state_dict=lambda value: trace.append('curriculum'))

    def cold_reset():
        trace.append('physical_reset')
        if platform:
            # Historical in-flight intent has already been discarded before
            # the real reset picks a different tile for each robot.
            assert owner.caller.state.bound_request_id.tolist() == [-1, -1, -1]
        reset(native, [0, 1, 2], [102, 0, 101], [3, 3, 3])

    result.reset = cold_reset
    return result, owner, trace


def test_saved_accounting_survives_but_old_flight_is_not_resumed(tmp_path):
    course, owner, trace = adapter(tmp_path)
    saved = course.training_state_dict()
    assert saved['platform_course']['caller']['platform_intent_ids'][0].tolist() == [0, 3, 7]
    owner._reset_counts.fill_(9)
    course.load_training_state_dict(saved)
    assert trace == ['curriculum', 'course', 'control', 'sensors', 'physical_reset']
    assert owner._reset_counts.tolist() == [2, 2, 2]
    assert owner.caller.state.target_index.tolist() == [1, -1, 0]
    assert owner.caller.state.bound_request_id.tolist() == [-1, -1, -1]
    assert saved['platform_course']['caller']['platform_intent_ids'][0].tolist() == [0, 3, 7]


def test_legacy_save_layout_stays_unchanged(tmp_path):
    course, _, trace = adapter(tmp_path, platform=False)
    saved = course.training_state_dict()
    assert set(saved) == {'format', 'curriculum', 'course', 'control_timing', 'sensors'}
    course.load_training_state_dict(saved)
    assert trace[-1] == 'physical_reset'


@pytest.mark.parametrize('live_platform', [False, True])
def test_owner_mismatch_is_not_silently_filled_or_discarded(tmp_path, live_platform):
    course, _, trace = adapter(tmp_path, platform=live_platform)
    saved = course.training_state_dict()
    if live_platform:
        del saved['platform_course']
    else:
        saved['platform_course'] = {}
    with pytest.raises(ValueError, match='explicit source migration'):
        course.load_training_state_dict(saved)
    assert trace == []
