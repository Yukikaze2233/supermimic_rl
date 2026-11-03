"""Short failing trajectories must remain visible in one-step model validation."""
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.world_model_v2.model import Ticks
from supermimic_rl.world_model_v2.records import RecordSpec, TICK_LABEL_DIM
from supermimic_rl.world_model_v2.replay import CompactReplay
from supermimic_rl.world_model_v2.validation import validate
from supermimic_rl.world_model_v2.termination import PoseContactLimits, PoseContactTermination


def test_h1_includes_short_failure_excluded_from_h2_and_reports_false_contacts():
    shapes = RecordSpec(16, 65).fields()
    replay = CompactReplay(shapes, 1, 3, 'cpu')
    for step in range(3):
        record = {name: torch.zeros(1, *shape) for name, shape in shapes.items()}
        record['physical'][0, 6] = record['end_physical'][0, 6] = 1.
        record['physical'][0, 2] = record['end_physical'][0, 2] = .3
        record['height'].fill_(.3)
        record['end_height'].fill_(.3)
        record['failure_conditions_valid'].fill_(1.)
        if step == 0:
            record['end_physical'][0, 2] = 1.3
            record['end_height'].fill_(1.3)
        boundary = {name: torch.ones(1, dtype=torch.bool) for name in ('valid', 'policy_owned', 'end_policy_owned')}
        boundary.update(terminated=torch.tensor([step == 0]), truncated=torch.zeros(1, dtype=torch.bool))
        boundary.update({name: torch.tensor([int(step > 0)]) for name in
            ('start_episode_id', 'end_episode_id', 'start_control_epoch', 'end_control_epoch')})
        replay.add(record, boundary)

    class StaticModel:
        def step(self, state, height, scan, memory, context, strength, action):
            labels = torch.zeros(len(action), 4, TICK_LABEL_DIM)
            labels[..., 28] = 2.  # Canonical base body normal load; truth is zero.
            return SimpleNamespace(state=state, height_above_ground=height, terrain_scan=scan,
                                   memory=memory, ticks=Ticks.decode(labels, state))

    termination = PoseContactTermination(PoseContactLimits(
        .1, .05, 1., 1., .04, .04, ('base_link', 'M_link1', 'M_link2')), .005)
    result = validate(SimpleNamespace(model=StaticModel()), replay, [0], 2, 512,
                      torch.Generator().manual_seed(3), 'cpu', contact_threshold_n=1., termination=termination)
    assert result['1_step']['position_m'] > .1
    assert result['1_step']['terminal_endpoint_fraction'] > .1
    assert result['2_step']['position_m'] == 0.
    assert result['2_step']['terminal_endpoint_fraction'] == 0.
    assert result['1_step']['forbidden_contact']['bodies']['base_link']['false_positive'] == 512 * 4
    assert result['1_step']['forbidden_contact']['bodies']['M_link1']['true_negative'] == 512 * 4
    # An instantaneous false contact is not yet a false termination. The exact
    # physical confirmation clock reaches its fixture threshold at H2 only.
    assert result['1_step']['forbidden_contact_latch']['false_positive'] == 0
    assert result['2_step']['forbidden_contact_latch']['false_positive'] == 512
    assert result['2_step']['forbidden_contact_latch']['clock_absolute_error_mean_s'] > .039
    assert (replay.data['failure_memory'] == 0.).all()
    floor = torch.zeros(21)
    floor[0] = 3.
    calibrated = validate(SimpleNamespace(model=StaticModel()), replay, [0], 2, 512,
        torch.Generator().manual_seed(3), 'cpu', contact_threshold_n=1., termination=termination,
        contact_floor=(floor, torch.tensor(3.)))
    assert calibrated['2_step']['forbidden_contact_latch']['false_positive'] == 512
    assert calibrated['2_step']['forbidden_contact_latch_floored']['false_positive'] == 0
    entry = calibrated['1_step']
    assert entry['body_normal_load_n'] == result['1_step']['body_normal_load_n']  # Raw errors stay raw.
    assert entry['forbidden_contact_floored']['bodies']['base_link']['false_positive'] == 0
    assert entry['contact_events']['raw']['base_normal_contact']['false_positive'] == 2048
    no_positive = entry['contact_events']['floored']['base_normal_contact']
    assert no_positive['positive_labels'] == 0 and no_positive['recall'] is None
    assert no_positive['precision'] is None and no_positive['status'] == 'no_positive_labels'
    # Normal support decides native airborne labels. Neither floors nor base
    # collision filtering can change the wheel channel or invent support.
    assert entry['contact_events']['raw']['both_wheels_airborne']['recall'] == 1.
    assert entry['contact_events']['floored']['both_wheels_airborne'] == entry['contact_events']['raw']['both_wheels_airborne']


def test_population_filters_wrapped_windows_and_keeps_held_out_rows():
    shapes = RecordSpec(16, 65).fields()
    replay = CompactReplay(shapes, 2, 5, 'cpu')
    for step in range(9):
        record = {name: torch.zeros(2, *shape) for name, shape in shapes.items()}
        record['physical'][:, 6] = record['end_physical'][:, 6] = 1.
        record['end_physical'][:, 2] = torch.tensor([100., 2.]) if step >= 6 else 1000.
        record['end_height'][:, 0] = record['end_physical'][:, 2]
        boundary = {name: torch.ones(2, dtype=torch.bool) for name in
                    ('valid', 'policy_owned', 'end_policy_owned')}
        boundary.update({name: torch.zeros(2, dtype=torch.bool) for name in ('terminated', 'truncated')})
        boundary.update({name: torch.zeros(2, dtype=torch.int64) for name in
                        ('start_episode_id', 'end_episode_id', 'start_control_epoch', 'end_control_epoch')})
        replay.add(record, boundary)

    class StaticModel:
        def step(self, state, height, scan, memory, context, strength, action):
            labels = torch.zeros(len(action), 4, TICK_LABEL_DIM)
            return SimpleNamespace(state=state, height_above_ground=height, terrain_scan=scan,
                                   memory=memory, ticks=Ticks.decode(labels, state))

    required = torch.zeros(5, 2, dtype=torch.bool)
    required[torch.tensor([6, 7, 8]) % 5] = True
    starts = torch.zeros_like(required)
    starts[6 % 5] = True
    args = (SimpleNamespace(model=StaticModel()), replay, [1], 4, 16,
            torch.Generator().manual_seed(3), 'cpu')
    result = validate(*args, required_rows=required, start_rows=starts)
    assert result['1_step']['eligible_windows'] == 1
    assert result['1_step']['unfiltered_eligible_windows'] == 5
    assert result['1_step']['position_m'] == pytest.approx(2 / (3 ** .5))
    assert result['4_step'] == {'status': 'no_eligible_windows', 'eligible_windows': 0,
                                'unfiltered_eligible_windows': 2, 'sampled_windows': 0}
    # Matching the first row alone must not admit a window whose later rows
    # still lack source metadata. This also checks physical-vs-chronological slots.
    required[7 % 5] = False
    result = validate(args[0], replay, [1], 2, 16, args[5], 'cpu',
                      required_rows=required, start_rows=starts)
    assert result['1_step']['eligible_windows'] == 1
    assert result['2_step']['status'] == 'no_eligible_windows'
    with pytest.raises(ValueError, match='boolean mask'):
        validate(*args, required_rows=required.float())
