"""World-model execution, student and collection follow the native runtime action bounds."""
import json
import math

import pytest
import torch

from supermimic_rl.contracts import ACTION_BOUNDS, Field, StateSchema, action_bounds_from_source
from supermimic_rl.world_model_v2.collection import _schema_action_bounds
from supermimic_rl.world_model_v2.execution import LegExecution
from supermimic_rl.world_model_v2.policy import EstimatorActor

FULL_TURN = (4 * math.pi,) * 4 + (9., 9.)
NOMINAL = torch.tensor([[-.42, .1374, .42, -.1374, 0., 0.]])


def test_source_bounds_default_to_the_legacy_contract():
    assert action_bounds_from_source({}) == ACTION_BOUNDS
    assert action_bounds_from_source({"action_bounds": list(FULL_TURN)}) == pytest.approx(FULL_TURN)
    with pytest.raises(ValueError, match="share one"):
        action_bounds_from_source({"action_bounds": [3., 4., 3., 3., 9., 9.]})


def _legs(bound):
    ones = torch.ones(1, 6)
    return LegExecution(NOMINAL, 160 * ones, 2.5 * ones, 40 * ones, leg_action_bound=bound)


def test_imagined_leg_targets_use_the_runtime_leg_bound():
    action = torch.tensor([[8., 8., 0., 0., 0., 0.]])
    wide = _legs(FULL_TURN[0]).target(action, NOMINAL)
    torch.testing.assert_close(wide[0, :2] - NOMINAL[0, :2], torch.full((2,), 2.))
    legacy = _legs(3.).target(action, NOMINAL)
    torch.testing.assert_close(legacy[0, :2] - NOMINAL[0, :2], torch.full((2,), .75))


def test_student_and_collector_bounds_follow_the_schema_source():
    actor = EstimatorActor(estimator_hidden=(8,), actor_hidden=(16,), frame_dim=35, action_bounds=FULL_TURN)
    assert actor(torch.zeros(2, 5, 35)).abs().max() <= 4 * math.pi
    torch.testing.assert_close(actor.action_bounds, torch.tensor(FULL_TURN, dtype=torch.float32))

    def schema(source):
        return StateSchema((Field("position", (3,)),), (), source=source)

    assert _schema_action_bounds(schema(json.dumps({"action_bounds": list(FULL_TURN)}))) == pytest.approx(FULL_TURN)
    assert _schema_action_bounds(schema("legacy-native-string")) == ACTION_BOUNDS
    assert schema(json.dumps({"runtime_sha256": "r"})).action_bounds == ACTION_BOUNDS
    assert _schema_action_bounds(None) == ACTION_BOUNDS
