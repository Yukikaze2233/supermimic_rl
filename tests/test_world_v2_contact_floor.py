"""Calibrated contact-load floors remove near-zero noise without hiding real contact."""
from types import SimpleNamespace

import torch

from supermimic_rl.world_model_v2.learning import Learner
from supermimic_rl.world_model_v2.model import Ticks
from supermimic_rl.world_model_v2.physical import INCREMENT_DIM
from supermimic_rl.world_model_v2.records import CONTACT_BODIES, RecordSpec, TICK_LABEL_DIM, WHEEL_BODIES, decode_physical
from supermimic_rl.world_model_v2.replay import CompactReplay
from supermimic_rl.world_model_v2.validation import calibrate_contact_floor, validate

BODY = INCREMENT_DIM + 2
BASE = BODY + CONTACT_BODIES.index("base_link")
LEFT_WHEEL = BODY + CONTACT_BODIES.index(WHEEL_BODIES[0])
BASE_RESULTANT = BODY + len(CONTACT_BODIES) + 2


def identity_state(rows):
    physical = torch.zeros(rows, 27)
    physical[:, 6] = 1.
    return decode_physical(physical)


def test_floor_zeroes_only_loads_below_a_positive_floor():
    labels = torch.zeros(1, 4, TICK_LABEL_DIM)
    labels[0, :, BASE] = torch.tensor([.5, 3., 30., 0.])
    labels[0, :, BASE_RESULTANT] = torch.tensor([3., .5, 30., 0.])
    labels[0, :, LEFT_WHEEL] = torch.tensor([-2., .1, 80., 0.])
    ticks = Ticks.decode(labels, identity_state(1))
    body_floor = torch.zeros(len(CONTACT_BODIES))
    body_floor[CONTACT_BODIES.index("base_link")] = 4.
    floored = ticks.with_contact_floor(body_floor, torch.tensor(2.))
    torch.testing.assert_close(floored.body_normal_load[0, :, 0], torch.tensor([0., 0., 30., 0.]))
    torch.testing.assert_close(floored.base_contact_force[0], torch.tensor([3., 0., 30., 0.]))
    # Zero-floor channels (wheels) keep raw values, including negative regression noise.
    torch.testing.assert_close(floored.wheel_body_load, ticks.wheel_body_load)
    torch.testing.assert_close(floored.state.position, ticks.state.position)


def test_calibration_floors_false_contact_and_keeps_true_contact():
    shapes = RecordSpec(16, 65).fields()
    replay = CompactReplay(shapes, 1, 6, "cpu")
    for step in range(6):
        record = {name: torch.zeros(1, *shape) for name, shape in shapes.items()}
        record["physical"][0, 6] = record["end_physical"][0, 6] = 1.
        record["action"][0, 0] = float(step % 2)
        if step % 2:
            record["ticks"][0, :, BASE] = 50.
        boundary = {name: torch.ones(1, dtype=torch.bool) for name in ("valid", "policy_owned", "end_policy_owned")}
        boundary.update(terminated=torch.zeros(1, dtype=torch.bool), truncated=torch.zeros(1, dtype=torch.bool))
        boundary.update({name: torch.zeros(1, dtype=torch.long) for name in
            ("start_episode_id", "end_episode_id", "start_control_epoch", "end_control_epoch")})
        replay.add(record, boundary)

    class NoisyContactModel:
        """Predicts 2 N base noise without contact and the real load with it."""

        def step(self, state, height, scan, memory, context, strength, action):
            labels = torch.zeros(len(action), 4, TICK_LABEL_DIM)
            labels[..., BASE] = torch.where(action[:, :1] > .5, 50., 2.)
            return SimpleNamespace(state=state, height_above_ground=height, terrain_scan=scan,
                                   memory=memory, ticks=Ticks.decode(labels, state))

    result = calibrate_contact_floor(SimpleNamespace(model=NoisyContactModel()), replay, [0], 256,
                                     torch.Generator().manual_seed(5), "cpu", threshold_n=1., horizon=2)
    base = result["report"]["channels"]["base_link"]
    assert base["floor_n"] == 2.
    assert base["false_positive_raw"] == base["negatives"] > 0
    assert base["false_positive_floored"] == 0
    assert base["true_positive_floored"] == base["true_positive_raw"] == base["positives"] > 0
    assert "L_link3" not in result["report"]["channels"]
    assert result["body_floor"][CONTACT_BODIES.index("L_link3")] == 0.

    # Audit a newly sampled set using the saved floors, without fitting them
    # again. Report both lost true contacts and removed false contacts.
    quality = validate(SimpleNamespace(model=NoisyContactModel()), replay, [0], 2, 256,
        torch.Generator().manual_seed(19), 'cpu', contact_threshold_n=1.,
        contact_floor=(result['body_floor'], result['base_floor']))
    raw = quality['2_step']['contact_events']['raw']['base_normal_contact']
    floored = quality['2_step']['contact_events']['floored']['base_normal_contact']
    assert raw['false_positive'] > 0 and floored['false_positive'] == 0
    assert floored['true_positive'] == raw['true_positive'] > 0
    assert floored['recall'] == 1. and floored['precision'] == 1.
    excessive = result['body_floor'].clone()
    excessive[CONTACT_BODIES.index('base_link')] = 100.
    hidden = validate(SimpleNamespace(model=NoisyContactModel()), replay, [0], 2, 256,
        torch.Generator().manual_seed(19), 'cpu', contact_threshold_n=1.,
        contact_floor=(excessive, result['base_floor']))
    missed = hidden['2_step']['contact_events']['floored']['base_normal_contact']
    assert missed['recall'] == 0. and missed['false_negative'] == raw['positive_labels']

    empty = calibrate_contact_floor(SimpleNamespace(model=NoisyContactModel()), replay, [0], 8,
                                    torch.Generator().manual_seed(5), "cpu", threshold_n=1., horizon=7)
    assert empty["report"]["status"] == "no_eligible_windows"
    assert not empty["body_floor"].any() and empty["base_floor"] == 0.


def test_learner_checkpoint_round_trips_floors_and_reads_legacy_state():
    def learner():
        value = Learner.__new__(Learner)
        value.model, value.actor, value.critic = (torch.nn.Linear(1, 1) for _ in range(3))
        value.optimizers, value.anchor = {}, None
        value.platform_contact = None
        value.updates, value.skipped = {"actor": 0}, {"actor": 0}
        value.imagination = SimpleNamespace(contact_floor=None)
        return value

    source = learner()
    assert "contact_floor" not in source.state_dict()
    source.imagination.contact_floor = (torch.arange(len(CONTACT_BODIES), dtype=torch.float), torch.tensor(3.))
    target = learner()
    target.load_state_dict(source.state_dict())
    for restored, saved in zip(target.imagination.contact_floor, source.imagination.contact_floor):
        torch.testing.assert_close(restored, saved)
    target.load_state_dict(learner().state_dict())
    assert target.imagination.contact_floor is None
