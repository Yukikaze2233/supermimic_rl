"""Numerical checks for imagined gradients, boundaries and learning recovery."""
from copy import deepcopy
from dataclasses import replace
from io import BytesIO
import math
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from supermimic_rl.contracts import Field, Snapshot, StateSchema, Transition, WindowBatch
from supermimic_rl.learning import (
    Learner, LearningConfig, frozen_parameters, lambda_returns,
)
from supermimic_rl.models import Prediction
from supermimic_rl.replay import ReplayBuffer


@pytest.fixture
def formal_learning_case(monkeypatch):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    schema = StateSchema((Field("position", (1,)),), (Field("mass", (1,)),),
                         observation_dim=45, goal_dim=14, source="formal-numerical-fixture")

    def clean(state, goal, context, previous_action, schema):
        observation = state.new_zeros(state.shape[0], 45)
        observation[:, 4] = state[:, 0]
        observation[:, 22:28] = previous_action
        return observation

    def deployed(*args):
        observation = clean(*args)
        observation[:, 4] += 1
        return observation

    def advance(state, goal, context, action, schema):
        goal = goal.clone()
        goal[:, 10] += schema.policy_dt_s
        goal[:, 11] += 1
        return state, goal, context + 2

    monkeypatch.setitem(sys.modules, "supermimic_rl.formal_observation", SimpleNamespace(
        observe=deployed, observe_clean=clean, advance_conditions=advance,
        imagination_horizon=lambda state, goal, context, schema: torch.full((len(state),), 2)))
    data = windows(batch=4, steps=4)
    data.states.zero_()
    data.observations = torch.zeros(4, 5, 45)
    data.actor_observations = data.observations.clone()
    data.actor_observations[..., 4] = 7
    data.goals = torch.zeros(4, 5, 14)
    data.goals[..., 11] = 3
    data.estimator_targets = torch.tensor([1., -.5, .25, 1., 0.]).expand(4, 5, 5).clone()
    data.estimator_valid = torch.ones(4, 5, dtype=torch.bool)
    # Initialization must not depend on how much RNG earlier tests consumed.
    torch.manual_seed(5)
    learner = small_learner(schema, history_length=5, fit_horizon=4, learning_rate=1e-3)
    learner.fit_normalizers(data)
    yield learner, data
    torch.set_num_threads(previous_threads)


def test_formal_clean_supervision_uses_real_actor_history_and_advances_context(formal_learning_case):
    learner, data = formal_learning_case
    _, components = learner.model_loss(data)
    assert components["observation"].item() == 0
    observed_histories = []
    hook = learner.actor.register_forward_pre_hook(
        lambda module, args: observed_histories.append(args[0].detach().clone()))
    try:
        with frozen_parameters(learner.world, learner.critic):
            loss, imagined = learner.imagine(data)
            loss.backward()
    finally:
        hook.remove()
    torch.testing.assert_close(observed_histories[0], data.actor_observations)
    # The model's deployed observer, not the clean supervision, enters H5.
    torch.testing.assert_close(observed_histories[1][:, -1, 4], torch.ones(4))
    assert imagined["goals"].shape[-1] == 14
    torch.testing.assert_close(imagined["goals"][:, 0, 11], torch.tensor([3., 4.]))
    torch.testing.assert_close(imagined["contexts"][:, 0, 0], torch.tensor([1., 3.]))
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in learner.actor.parameters())
    assert all(p.grad is None for p in learner.world.parameters())
    assert all(p.grad is None for p in learner.estimator.parameters())


def test_formal_estimator_learns_real_labels_independently_and_resumes(formal_learning_case):
    learner, data = formal_learning_case
    before = {name: deepcopy(getattr(learner, name).state_dict()) for name in ("actor", "world", "critic")}
    initial_contacts = learner.estimator(data.actor_observations)[:, 6:].detach().sigmoid()
    first = learner.fit_estimator(data)
    losses = [learner.fit_estimator(data)["estimator/loss"] for _ in range(29)]
    assert first["estimator/valid_labels"] == 4
    # Gaussian NLL oscillates while its variance contracts on a constant target;
    # compare a recent average, not one arbitrary final step.
    assert sum(losses[-5:]) / 5 < first["estimator/loss"]
    with torch.no_grad():
        raw = learner.estimator(data.actor_observations)
    assert ((raw[:, :3] - data.estimator_targets[:, -1, :3]).abs().mean() < .25)
    assert (raw[:, 6].sigmoid() > initial_contacts[:, 0]).all()
    assert (raw[:, 7].sigmoid() < initial_contacts[:, 1]).all()
    for name, original in before.items():
        assert_tree_equal(original, getattr(learner, name).state_dict())
    restored = Learner(learner.schema, learner.config)
    restored.load_state_dict(deepcopy(learner.state_dict()))
    assert restored.estimator_updates == 30
    assert_tree_equal(learner.state_dict(), restored.state_dict())
    learner.fit_estimator(data)
    restored.fit_estimator(data)
    assert_tree_equal(learner.state_dict(), restored.state_dict())


@pytest.fixture
def schema():
    return StateSchema((Field("position", (1,), "m", "world"),),
                       (Field("mass", (1,), "kg"),), source="analytic-test")


def windows(*, batch=2, steps=3):
    states = torch.arange(steps + 1, dtype=torch.float32)[None, :, None]
    states = states.expand(batch, -1, -1).clone()
    observations = torch.zeros(batch, steps + 1, 35)
    observations[..., 4] = states[..., 0]
    return WindowBatch(
        states=states, observations=observations,
        goals=torch.zeros(batch, steps + 1, 11),
        contexts=torch.ones(batch, steps + 1, 1),
        actions=torch.zeros(batch, steps, 6),
        rewards=torch.full((batch, steps), 2.0),
        terminated=torch.zeros(batch, steps, dtype=torch.bool),
        truncated=torch.zeros(batch, steps, dtype=torch.bool),
        valid=torch.ones(batch, steps, dtype=torch.bool),
        policy_owned=torch.ones(batch, steps, dtype=torch.bool),
    )


def small_learner(schema, **overrides):
    options = dict(model_hidden=(4,), actor_hidden=(4,), critic_hidden=(4,),
                   history_length=2, fit_horizon=2, imagination_horizon=2,
                   gamma=0.8, gae_lambda=0.6, critic_epochs=1,
                   critic_minibatches=1)
    options.update(overrides)
    return Learner(schema, LearningConfig(**options))


class ScalarActor(nn.Module):
    def __init__(self, value=0.2):
        super().__init__()
        self.command = nn.Parameter(torch.tensor(value))

    def forward(self, history):
        first = self.command.expand(history.shape[0], 1)
        return torch.cat((first, torch.zeros_like(first).expand(-1, 5)), -1)


class LinearValue(nn.Module):
    def __init__(self, slope):
        super().__init__()
        self.slope = nn.Parameter(torch.tensor(float(slope)))

    def forward(self, state, goal, context):
        return self.slope * state[:, 0]


def controlled_learner(schema, monkeypatch, *, reward_slope, value_slope):
    learner = small_learner(schema)
    learner.normalizers_fitted = True
    learner.actor = ScalarActor()
    learner.critic = LinearValue(value_slope)
    learner.actor_optimizer = torch.optim.Adam(learner.actor.parameters(), lr=1e-3)
    learner.critic_optimizer = torch.optim.Adam(learner.critic.parameters(), lr=1e-3)
    with torch.no_grad():
        learner.world.dynamics[-1].bias.fill_(1.5)

    def dynamics(state, action, goal, context):
        gain = learner.world.dynamics[-1].bias[0]
        following = state + gain * action[:, :1]
        return Prediction(following, reward_slope * following[:, 0],
                          torch.zeros_like(following[:, 0]))

    def observe(state, goal, context, previous_action=None):
        result = torch.zeros(state.shape[0], 35, device=state.device)
        result[:, 4] = state[:, 0]
        result[:, 22:28] = previous_action
        return result

    monkeypatch.setattr(learner.world, "step", dynamics)
    monkeypatch.setattr(learner.world, "observe", observe)
    return learner


@pytest.mark.parametrize("reward_slope,value_slope", [(2.0, 0.0), (0.0, 3.0), (2.0, 3.0)])
def test_analytic_actor_gradient_through_frozen_dynamics_and_value(
        schema, monkeypatch, reward_slope, value_slope):
    learner = controlled_learner(schema, monkeypatch,
                                 reward_slope=reward_slope, value_slope=value_slope)
    batch = windows()
    batch.states.requires_grad_()
    before = {name: value.clone() for name, value in learner.world.state_dict().items()}
    with frozen_parameters(learner.world, learner.critic):
        loss, imagined = learner.imagine(batch)
        loss.backward()
        # x0=1, x1=1.3, x2=1.6; gamma*c=0.4 and (gamma*c)^2=0.16.
        expected_return = reward_slope * (1.3 + 0.4 * 1.6) + 0.16 * value_slope * 1.6
        expected_gradient = -(reward_slope * (1.5 + 0.4 * 3.0)
                              + 0.16 * value_slope * 3.0)
        assert loss.item() == pytest.approx(-expected_return, abs=1e-6)
        assert learner.actor.command.grad.item() == pytest.approx(expected_gradient, abs=1e-6)
        assert all(parameter.grad is None for parameter in learner.world.parameters())
        assert learner.critic.slope.grad is None
    assert batch.states.grad is None
    assert all(parameter.requires_grad for parameter in learner.world.parameters())
    assert learner.critic.slope.requires_grad
    assert torch.equal(imagined["states"][:, 0, 0], torch.tensor([1.0, 1.3]))
    assert all(torch.equal(value, before[name]) for name, value in learner.world.state_dict().items())


def test_soft_survival_keeps_the_action_derivative(schema, monkeypatch):
    learner = controlled_learner(schema, monkeypatch, reward_slope=0, value_slope=3)
    learner.config = LearningConfig(**{
        **vars(learner.config), "imagination_horizon": 1,
    })

    def dynamics(state, action, goal, context):
        return Prediction(state, torch.ones_like(state[:, 0]), action[:, 0])

    monkeypatch.setattr(learner.world, "step", dynamics)
    with frozen_parameters(learner.world, learner.critic):
        loss, _ = learner.imagine(windows())
        loss.backward()
    survival = 1 / (1 + math.exp(0.2))
    assert loss.item() == pytest.approx(-(1 + 0.8 * survival * 3), abs=1e-6)
    assert learner.actor.command.grad.item() == pytest.approx(
        0.8 * 3 * survival * (1 - survival), abs=1e-6)


def test_per_row_condition_boundary_bootstraps_without_false_termination(schema, monkeypatch):
    learner = controlled_learner(schema, monkeypatch, reward_slope=2, value_slope=3)
    learner.config = replace(learner.config, imagination_horizon=3)
    monkeypatch.setattr(learner.world, "imagination_horizon",
                        lambda state, goal, context, maximum: torch.tensor([1, 3]))
    with frozen_parameters(learner.world, learner.critic):
        loss, data = learner.imagine(windows())
        loss.backward()
    assert loss.item() == pytest.approx(-(4.16 + 4.8528) / 2, abs=1e-6)
    assert learner.actor.command.grad.item() == pytest.approx(-(4.8 + 7.704) / 2, abs=1e-6)
    torch.testing.assert_close(data["weights"][:, 0], torch.tensor([1., 0., 0.]))
    assert data["continuation"][0, 0] == .5
    targets = lambda_returns(data["rewards"], data["continuation"], data["values"],
                              .8, .6, data["horizon_boundary"])
    assert targets[0, 0].item() == pytest.approx(4.16, abs=1e-6)
    previous = learner.actor.command.detach().clone()
    monkeypatch.setattr(learner.world, "imagination_horizon",
                        lambda state, goal, context, maximum: torch.zeros(2, dtype=torch.long))
    metrics = learner.improve_policy(windows())
    assert metrics["actor/update_skipped"] == 1 and learner.actor_updates == 0
    assert learner.actor.command == previous


def test_feedback_through_predicted_observations_contributes_to_bptt(schema, monkeypatch):
    learner = controlled_learner(schema, monkeypatch, reward_slope=1, value_slope=0)

    def actor(history):
        first = learner.actor.command * history[:, -1, 4:5]
        return torch.cat((first, torch.zeros_like(first).expand(-1, 5)), -1)

    monkeypatch.setattr(learner.actor, "forward", actor)
    with frozen_parameters(learner.world, learner.critic):
        loss, _ = learner.imagine(windows())
        loss.backward()
    # x1=(1+1.5 theta), x2=(1+1.5 theta)^2 at theta=0.2.
    assert loss.item() == pytest.approx(-(1.3 + 0.4 * 1.3**2), abs=1e-6)
    assert learner.actor.command.grad.item() == pytest.approx(
        -(1.5 + 0.4 * 2 * 1.5 * 1.3), abs=1e-6)


def test_world_fit_is_autoregressive_and_does_not_detach_between_steps(schema, monkeypatch):
    learner = controlled_learner(schema, monkeypatch, reward_slope=0, value_slope=0)
    with torch.no_grad():
        learner.world.dynamics[-1].bias.fill_(0.5)
    batch = windows(steps=2)
    batch.actions[:, :, 0] = 1
    _, components = learner.model_loss(batch)
    components["state"].backward()
    # Predictions 0.5,1.0 versus true 1,2; the second error differentiates through both steps.
    assert components["state"].item() == pytest.approx(0.625)
    assert learner.world.dynamics[-1].bias.grad.item() == pytest.approx(-2.5)


def test_freezing_restores_original_flags_even_after_exception():
    module = nn.Linear(1, 1)
    module.bias.requires_grad_(False)
    x = torch.tensor([[2.0]], requires_grad=True)
    with pytest.raises(RuntimeError, match="interrupt"):
        with frozen_parameters(module):
            module(x).sum().backward()
            assert x.grad.item() == pytest.approx(module.weight.item())
            assert module.weight.grad is None
            raise RuntimeError("interrupt")
    assert module.weight.requires_grad
    assert not module.bias.requires_grad


@pytest.mark.parametrize("trace_lambda", [0.0, 0.6, 1.0])
def test_lambda_returns_respects_soft_survival_and_terminal_bootstrap(trace_lambda):
    rewards = torch.tensor([[1.0, 1.0], [2.0, 2.0]], requires_grad=True)
    continuation = torch.tensor([[0.5, 0.5], [0.0, 1.0]])
    values = torch.tensor([[91.0, 91.0], [10.0, 10.0], [20.0, 20.0]], requires_grad=True)
    result = lambda_returns(rewards, continuation, values, 0.8, trace_lambda)
    last = torch.tensor([2.0, 18.0])
    first = 1 + 0.4 * ((1 - trace_lambda) * 10 + trace_lambda * last)
    torch.testing.assert_close(result, torch.stack((first, last)))
    result[0].sum().backward()
    torch.testing.assert_close(rewards.grad[1], torch.full((2,), 0.4 * trace_lambda))
    assert values.grad[-1, 0].item() == 0.0
    assert values.grad[-1, 1].item() == pytest.approx(0.32 * trace_lambda)
    assert torch.count_nonzero(values.grad[0]) == 0


@pytest.mark.parametrize("terminated,expected", [(True, math.log(4 / 3)), (False, math.log(4))])
def test_model_failure_labels_distinguish_termination_from_truncation(
        schema, monkeypatch, terminated, expected):
    learner = small_learner(schema)
    learner.normalizers_fitted = True
    batch = windows(steps=1)
    batch.terminated[:] = terminated
    batch.truncated[:] = not terminated

    def dynamics(state, action, goal, context):
        return Prediction(state + 1, torch.full_like(state[:, 0], 2),
                          torch.full_like(state[:, 0], math.log(3)))

    monkeypatch.setattr(learner.world, "step", dynamics)
    _, losses = learner.model_loss(batch)
    assert losses["termination"].item() == pytest.approx(expected, abs=1e-6)


def test_imagination_history_and_physics_begin_at_the_same_endpoint(schema, monkeypatch):
    learner = controlled_learner(schema, monkeypatch, reward_slope=1, value_slope=0)
    learner.config = LearningConfig(**{
        **vars(learner.config), "history_length": 3, "imagination_horizon": 1,
    })
    histories = []

    def actor(history):
        histories.append(history.detach().clone())
        return torch.cat((history[:, -1, 4:5] * 0.1,
                          torch.zeros(history.shape[0], 5)), -1)

    monkeypatch.setattr(learner.actor, "forward", actor)
    batch = windows(steps=3)
    batch.goals[:, :, 0] = torch.arange(4)
    batch.contexts[:, :, 0] = torch.arange(4) + 10
    loss, data = learner.imagine(batch)
    torch.testing.assert_close(histories[0][:, :, 4], torch.tensor([[0., 1., 2.], [0., 1., 2.]]))
    assert torch.all(data["states"][0, :, 0] == 2)
    assert torch.all(data["goal"][:, 0] == 2)
    assert torch.all(data["context"][:, 0] == 12)
    assert loss.item() == pytest.approx(-2.3, abs=1e-6)


@pytest.mark.parametrize("field", ["valid", "policy_owned"])
def test_imagination_rejects_ineligible_last_history_transition(schema, field):
    learner = small_learner(schema, history_length=3)
    learner.normalizers_fitted = True
    batch = windows()
    getattr(batch, field)[:, learner.config.history_length - 2] = False
    with pytest.raises(ValueError):
        learner.imagine(batch)


def test_replay_history_endpoint_is_policy_owned_and_matches_imagined_state(schema, monkeypatch):
    replay = ReplayBuffer(schema, num_envs=2, capacity=8)

    def snapshot(time, ownership):
        position = torch.tensor([[float(time)], [100.0 + time]])
        observed = torch.zeros(2, 35)
        observed[:, 4] = position[:, 0]
        return Snapshot(position, observed, torch.zeros(2, 11), torch.ones(2, 1),
                        torch.zeros(2, dtype=torch.int64), torch.zeros(2, dtype=torch.int64),
                        torch.tensor(ownership))

    for step in range(2):
        replay.add(Transition(
            snapshot(step, [True, True]), torch.zeros(2, 6),
            snapshot(step + 1, [step == 0, True]), torch.zeros(2),
            torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool),
            torch.ones(2, dtype=torch.bool),
        ))
    assert replay.num_start_candidates(history_length=3) == 1
    batch = replay.sample_starts(5, history_length=3)
    torch.testing.assert_close(batch.states[..., 0],
                               torch.tensor([[100., 101., 102.]]).expand(5, -1))
    learner = controlled_learner(schema, monkeypatch, reward_slope=1, value_slope=0)
    learner.config = LearningConfig(**{
        **vars(learner.config), "history_length": 3, "imagination_horizon": 1,
    })
    loss, data = learner.imagine(batch)
    assert torch.all(data["states"][0, :, 0] == 102)
    assert loss.item() == pytest.approx(-102.3, abs=1e-5)


@pytest.mark.parametrize("field", ["terminated", "truncated"])
def test_imagination_rejects_history_crossing_a_reset_boundary(schema, field):
    learner = small_learner(schema)
    learner.normalizers_fitted = True
    batch = windows()
    getattr(batch, field)[:, 0] = True
    with pytest.raises(ValueError):
        learner.imagine(batch)


def test_final_survival_includes_the_last_predicted_transition(schema, monkeypatch):
    learner = controlled_learner(schema, monkeypatch, reward_slope=1, value_slope=1)
    world_before = {name: value.clone() for name, value in learner.world.state_dict().items()}
    command_before = learner.actor.command.item()
    metrics = learner.improve_policy(windows())
    assert metrics["imagination/final_survival"] == pytest.approx(0.25)
    assert learner.actor.command.item() > command_before
    assert all(torch.equal(world_before[name], value) for name, value in learner.world.state_dict().items())
    assert learner.actor_updates == learner.critic_updates == 1


def test_normalization_stays_fixed_through_world_and_policy_updates(schema):
    torch.manual_seed(3)
    learner = small_learner(schema)
    batch = windows()
    learner.fit_normalizers(batch)
    buffers = {name: value.clone() for name, value in learner.world.named_buffers()}
    torch.testing.assert_close(learner.world.state_normalizer.mean, torch.tensor([1.5]))
    torch.testing.assert_close(learner.world.state_normalizer.scale, torch.tensor([math.sqrt(1.25)]))
    learner.fit_world(batch)
    learner.improve_policy(batch)
    assert all(torch.equal(buffers[name], value) for name, value in learner.world.named_buffers())
    with pytest.raises(RuntimeError, match="frozen"):
        learner.fit_normalizers(batch)


@pytest.fixture
def native_contact_data():
    schema = StateSchema(
        (Field("position", (1,), "m"),
         Field("terrain_contact_force_world_n", (2, 3), "N", "world"),
         Field("unchanged_constant", (1,))),
        (Field("context_before_mass", (1,)), Field("body_mass_kg", (2,), "kg"),
         Field("context_after_mass", (1,))),
        source="native-contact-test",
    )
    batch = windows()
    batch.states = torch.cat((batch.states, torch.zeros(2, 4, 7)), dim=-1)
    masses = torch.tensor([[1., 2.], [3., 4.]])[:, None].expand(-1, 4, -1)
    batch.contexts = torch.cat((torch.full((2, 4, 1), 13.), masses,
                               torch.full((2, 4, 1), 17.)), dim=-1)
    return schema, batch


def test_native_contact_scale_uses_robot_weight_after_zero_contact_warmup(native_contact_data):
    schema, batch = native_contact_data
    before = deepcopy(batch)
    learner = small_learner(schema)
    learner.fit_normalizers(batch)
    normalizer = learner.world.state_normalizer
    # Mean robot mass is (3 + 7) / 2 kg, not mean link mass or total batch mass.
    weight_n = 5 * 9.81
    torch.testing.assert_close(normalizer.scale[1:7], torch.full((6,), weight_n))
    torch.testing.assert_close(normalizer.mean[1:7], torch.zeros(6))
    later_state = batch.states[:, -1].clone()
    later_state[:, 1:7] = 100
    normalized_contact = normalizer(later_state)[:, 1:7]
    torch.testing.assert_close(normalized_contact, torch.full((2, 6), 100 / weight_n))
    # A 100 N contact contributes about 4.16, versus 4,000,000 at a 0.05 N scale.
    assert normalized_contact.square().mean().item() == pytest.approx((100 / weight_n) ** 2)
    torch.testing.assert_close(normalizer.decode(normalizer(later_state)), later_state)
    # Adjacent fields retain their original empirical/default scales.
    assert normalizer.scale[0].item() == pytest.approx(math.sqrt(1.25))
    assert normalizer.scale[7].item() == pytest.approx(0.05)
    for name, value in vars(batch).items():
        torch.testing.assert_close(value, getattr(before, name), rtol=0, atol=0)


def test_native_contact_scale_preserves_larger_empirical_standard_deviation(native_contact_data):
    schema, batch = native_contact_data
    batch.states[:, :, 3] = torch.tensor([-200., -200., 200., 200.])
    learner = small_learner(schema)
    learner.fit_normalizers(batch)
    expected = torch.full((6,), 5 * 9.81)
    expected[2] = 200
    torch.testing.assert_close(learner.world.state_normalizer.scale[1:7], expected)


@pytest.mark.parametrize("mass_field", [
    None,
    Field("body_mass_kg", (1,), "kg"),
    Field("body_mass_kg", (2, 1), "kg"),
    Field("body_mass_kg", (2,), "g"),
])
def test_native_contact_scale_rejects_missing_or_invalid_mass_schema(
        native_contact_data, mass_field):
    schema, batch = native_contact_data
    context_fields = (schema.context_fields[0], schema.context_fields[-1])
    if mass_field is not None:
        context_fields = (context_fields[0], mass_field, context_fields[-1])
    learner = small_learner(replace(schema, context_fields=context_fields))
    with pytest.raises(ValueError, match="body_mass_kg"):
        learner.fit_normalizers(batch)
    assert not learner.normalizers_fitted
    torch.testing.assert_close(learner.world.state_normalizer.scale, torch.ones(schema.state_dim))


@pytest.mark.parametrize("invalid_mass", [0., -1., float("nan"), float("inf")])
def test_native_contact_scale_rejects_invalid_mass_values(native_contact_data, invalid_mass):
    schema, batch = native_contact_data
    batch.contexts[0, 0, 1] = invalid_mass
    learner = small_learner(schema)
    with pytest.raises(ValueError, match="body_mass_kg.*finite positive"):
        learner.fit_normalizers(batch)
    assert not learner.normalizers_fitted


def test_other_force_fields_keep_existing_normalization_without_native_mass(native_contact_data):
    schema, batch = native_contact_data
    other_force = replace(schema.state_fields[1], name="synthetic_contact_force")
    schema = replace(schema, state_fields=(schema.state_fields[0], other_force,
                                          schema.state_fields[-1]),
                     context_fields=(Field("synthetic_context", (4,)),))
    learner = small_learner(schema)
    learner.fit_normalizers(batch)
    torch.testing.assert_close(learner.world.state_normalizer.scale[1:7], torch.full((6,), 0.05))


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


def test_checkpoint_roundtrip_reproduces_the_next_optimizer_update(schema):
    torch.manual_seed(7)
    learner = small_learner(schema)
    batch = windows()
    learner.fit_normalizers(batch)
    learner.fit_world(batch)
    learner.improve_policy(batch)
    stream = BytesIO()
    torch.save(learner.state_dict(), stream)
    stream.seek(0)
    restored = small_learner(schema)
    restored.load_state_dict(torch.load(stream, weights_only=True))
    assert_tree_equal(learner.state_dict(), restored.state_dict())
    for candidate in (learner, restored):
        torch.manual_seed(100)
        candidate.fit_world(batch)
        candidate.improve_policy(batch)
    assert_tree_equal(learner.state_dict(), restored.state_dict())
    assert learner.world_updates == learner.actor_updates == 2
    assert learner.critic_updates == 2


def test_checkpoint_rejects_changed_schema_contents_even_with_old_hash(schema):
    learner = small_learner(schema)
    payload = deepcopy(learner.state_dict())
    payload["schema"]["state_fields"][0]["units"] = "rad"
    with pytest.raises(ValueError, match="schema"):
        learner.load_state_dict(payload)
