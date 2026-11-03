"""Named native field metadata must remain part of replay compatibility."""
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.adapters.packed import PackedEnvironment


def raw_snapshot():
    return SimpleNamespace(
        state={"position": torch.ones(2, 3)}, context={"mass": torch.ones(2, 1)},
        observation=torch.zeros(2, 35), actor_observation=torch.zeros(2, 35), goal=torch.zeros(2, 11),
        episode_id=torch.zeros(2, dtype=torch.int64), control_epoch=torch.zeros(2, dtype=torch.int64),
        policy_owned=torch.ones(2, dtype=torch.bool),
        field_metadata={"state": {"position": {"units": "m", "frame": "world"}},
                        "context": {"mass": {"units": "kg", "frame": "body"}}})


def test_named_native_units_and_frames_are_frozen():
    environment = PackedEnvironment(object(), source="pinned-asset-native-order-sdk")
    raw = raw_snapshot()
    packed = environment._pack(raw)
    assert environment.schema.state_fields[0].units == "m"
    assert environment.schema.context_fields[0].frame == "body"
    raw.state["position"].fill_(9)
    assert packed.state.eq(1).all()
    raw.field_metadata["state"]["position"]["units"] = "mm"
    with pytest.raises(ValueError, match="schema freeze"):
        environment._pack(raw)


def test_source_identity_changes_schema_even_with_identical_shapes():
    one = PackedEnvironment(object(), source="order:left,right")
    two = PackedEnvironment(object(), source="order:right,left")
    one._pack(raw_snapshot())
    two._pack(raw_snapshot())
    assert one.schema.fingerprint != two.schema.fingerprint


def test_formal_widths_and_labels_are_frozen_without_replacing_actor_frame():
    environment = PackedEnvironment(object(), source="formal45")
    raw = raw_snapshot()
    raw.observation = torch.zeros(2, 45)
    raw.actor_observation = torch.ones(2, 45)
    raw.goal = torch.zeros(2, 19)
    raw.estimator_target = torch.tensor([[1., 2., 3., 1., 0.]]).expand(2, -1).clone()
    raw.estimator_valid = torch.tensor([True, False])
    snapshot = environment._pack(raw)
    assert environment.schema.observation_dim == 45 and environment.schema.goal_dim == 19
    assert snapshot.observation.eq(0).all() and snapshot.actor_observation.eq(1).all()
    raw.estimator_target.zero_()
    assert snapshot.estimator_target[0, 0] == 1
    raw.goal = torch.zeros(2, 20)
    with pytest.raises(ValueError, match="schema freeze"):
        environment._pack(raw)


class RawAdapter:
    def __init__(self):
        self.current = raw_snapshot()
        self.invalid_end = False

    def reset(self):
        return self.current

    def step(self, action):
        start = self.current
        end = raw_snapshot()
        end.state["position"].fill_(2)
        if self.invalid_end:
            end.state["position"][0, 0] = float("nan")
        self.current = end
        return SimpleNamespace(start=start, end=end, action=action, reward=torch.zeros(2),
                               terminated=torch.zeros(2, dtype=torch.bool),
                               truncated=torch.zeros(2, dtype=torch.bool), valid=torch.ones(2, dtype=torch.bool))


def test_complete_packed_step_uses_one_host_value_decision_and_keeps_owned_copies(monkeypatch):
    adapter = RawAdapter()
    environment = PackedEnvironment(adapter, source="test")
    before = environment.reset()
    # Same object identity is not permission to reuse mutable tensor contents.
    adapter.current.state["position"].fill_(7)
    decisions = []
    original = torch.Tensor.__bool__

    def counted(value):
        decisions.append(value.shape)
        return original(value)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "__bool__", counted)
        value = environment.step(torch.zeros(2, 6))
    assert len(decisions) == 1
    assert before.state.eq(1).all()
    assert value.start.state.eq(7).all()
    assert value.end.state.eq(2).all()
    adapter.current.state["position"].fill_(99)
    assert environment.current.state.eq(2).all() and value.end.state.eq(2).all()


def test_failed_packed_step_does_not_publish_unvalidated_current_snapshot():
    adapter = RawAdapter()
    environment = PackedEnvironment(adapter, source="test")
    before = environment.reset()
    adapter.invalid_end = True
    with pytest.raises(ValueError, match="Nonfinite snapshot field: position"):
        environment.step(torch.zeros(2, 6))
    assert environment.current is before
