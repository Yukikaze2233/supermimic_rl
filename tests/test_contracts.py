"""Batch boundary value checks without a host decision for each tensor field."""
import pytest
import torch

from supermimic_rl.contracts import Field, Snapshot, StateSchema, Transition


def count_tensor_decisions(monkeypatch, call):
    decisions = []
    original = torch.Tensor.__bool__

    def counted(value):
        decisions.append(value.shape)
        return original(value)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "__bool__", counted)
        result = call()
    return result, decisions


def snapshot(schema, count=2):
    return Snapshot(
        state=torch.ones(count, schema.state_dim), context=torch.ones(count, schema.context_dim),
        observation=torch.zeros(count, schema.observation_dim), goal=torch.ones(count, schema.goal_dim),
        episode_id=torch.zeros(count, dtype=torch.int64), control_epoch=torch.zeros(count, dtype=torch.int64),
        policy_owned=torch.ones(count, dtype=torch.bool), actor_observation=torch.ones(count, schema.observation_dim),
        estimator_target=torch.tensor([[1., 2., 3., 1., 0.]]).expand(count, -1).clone(),
        estimator_valid=torch.ones(count, dtype=torch.bool),
    )


def transition(schema):
    return Transition(snapshot(schema), torch.ones(2, 6), snapshot(schema), torch.ones(2),
                      torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool),
                      torch.ones(2, dtype=torch.bool))


def test_many_named_fields_require_one_host_value_decision(monkeypatch):
    schema = StateSchema(tuple(Field(f"field_{i}", (3,)) for i in range(180)), ())
    values = {field.name: torch.full((2, 3), i, dtype=torch.float64)
              for i, field in enumerate(schema.state_fields)}
    packed, decisions = count_tensor_decisions(monkeypatch, lambda: schema.pack(values))
    assert len(decisions) == 1
    assert packed.shape == (2, 540) and packed.dtype == torch.float32
    assert torch.equal(packed[:, -3:], torch.full((2, 3), 179.))
    values["field_179"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite snapshot field: field_179"):
        schema.pack(values)


def test_float32_pack_overflow_does_not_escape_finite_source_validation():
    schema = StateSchema((Field("double_value", (1,)),), ())
    with pytest.raises(ValueError, match="double_value"):
        schema.pack({"double_value": torch.tensor([[1e300]], dtype=torch.float64)})


def test_snapshot_and_complete_transition_each_use_one_host_decision(monkeypatch):
    schema = StateSchema((Field("state", (2,)),), (Field("context", (3,)),), observation_dim=45)
    _, snapshot_decisions = count_tensor_decisions(monkeypatch, lambda: snapshot(schema).validate(schema))
    _, transition_decisions = count_tensor_decisions(monkeypatch, lambda: transition(schema).validate(schema))
    assert len(snapshot_decisions) == len(transition_decisions) == 1


@pytest.mark.parametrize("field", ["state", "context", "observation", "goal", "actor_observation", "estimator_target"])
def test_every_snapshot_float_group_is_still_checked(field):
    schema = StateSchema((Field("state", (2,)),), (Field("context", (3,)),))
    value = snapshot(schema)
    getattr(value, field)[0, 0] = float("inf")
    with pytest.raises(ValueError, match="Nonfinite|Invalid"):
        value.validate(schema)


@pytest.mark.parametrize("fault,message", [
    ("action", "Invalid executed action"), ("bound", "bounded actions"),
    ("reward", "Invalid reward"), ("terminal", "BEFORE auto-reset"),
    ("contact", "estimator labels"),
])
def test_batched_transition_validation_keeps_bounds_labels_and_terminal_guards(fault, message):
    schema = StateSchema((Field("state", (2,)),), ())
    value = transition(schema)
    if fault == "action":
        value.action[0, 1] = float("nan")
    elif fault == "bound":
        value.action[0, 1] = 3.01
    elif fault == "reward":
        value.reward[0] = float("inf")
    elif fault == "terminal":
        value.end.episode_id[0] = 1
    else:
        value.end.estimator_target[0, 4] = 1.01
    with pytest.raises(ValueError, match=message):
        value.validate(schema)


def test_layout_failure_is_immediate_and_validity_mask_retains_its_original_meaning(monkeypatch):
    schema = StateSchema((Field("state", (2,)),), ())
    value = transition(schema)
    value.end.episode_id[0] = 1
    value.valid[0] = False
    value.validate(schema)
    value.end.context = torch.zeros(2, 1)
    with pytest.raises(ValueError, match="context must be"):
        count_tensor_decisions(monkeypatch, lambda: value.validate(schema))


def test_script_actions_keep_exact_values_and_bounds_use_start_ownership(monkeypatch):
    schema = StateSchema((Field("state", (2,)),), ())
    value = transition(schema)
    value.start.policy_owned[0] = False
    # A handoff at the endpoint cannot retroactively turn a script into actor output.
    value.end.policy_owned[0] = True
    value.action[0] = torch.tensor([5., -5., 4., -4., 12., -12.])
    issued = value.action.clone()
    _, decisions = count_tensor_decisions(monkeypatch, lambda: value.validate(schema))
    assert len(decisions) == 1
    torch.testing.assert_close(value.action, issued)
    # Likewise, losing ownership at the endpoint does not excuse an earlier
    # out-of-bounds policy issuance, even for an invalid training sample.
    value.end.policy_owned[1] = False
    value.valid[1] = False
    value.action[1, 0] = 3.01
    with pytest.raises(ValueError, match="bounded actions"):
        value.validate(schema)


@pytest.mark.parametrize("fault,message", [
    ("nan", "Invalid executed action"), ("inf", "Invalid executed action"),
    ("terminal", "BEFORE auto-reset"),
])
def test_nonpolicy_rows_still_require_finite_actions_and_pre_reset_endpoints(fault, message):
    schema = StateSchema((Field("state", (2,)),), ())
    value = transition(schema)
    value.start.policy_owned[:] = False
    value.end.policy_owned[:] = False
    if fault == "terminal":
        value.end.episode_id[0] += 1
    else:
        value.action[0, 0] = float(fault)
    with pytest.raises(ValueError, match=message):
        value.validate(schema)
