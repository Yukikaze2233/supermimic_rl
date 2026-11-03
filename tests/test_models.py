"""World-model observations must preserve the native actor interface."""
import torch

from supermimic_rl.contracts import Field, StateSchema
from supermimic_rl.models import WorldModel


def observation_fixture():
    schema = StateSchema((Field("position", (2,)),), (Field("friction", (1,)),))
    world = WorldModel(schema, hidden=(4,))
    with torch.no_grad():
        world.observation_normalizer.scale.fill_(2)
        world.observation_head[-1].weight.zero_()
        world.observation_head[-1].bias.fill_(75)
        world.observation_head[-1].bias[4:8] = torch.tensor([-60., 60., 12., -12.])
    state, context = torch.zeros(1, 2), torch.ones(1, 1)
    goal = torch.tensor([[0.2, 0., -0.4, 1.5, 1., 0., 0., 0., 0., 0., 5.]])
    previous_action = torch.tensor([[3., -3., 2., -2., 9., -9.]])
    return world, state, context, goal, previous_action


def test_observation_clips_decoded_head_and_preserves_legal_and_analytic_slots():
    world, state, context, goal, action = observation_fixture()
    observed = world.observe(state, goal, context, previous_action=action)
    assert observed.shape == (1, 35)
    assert torch.all(observed.abs() <= 100)
    # Decoding first maps +/-60 to +/-120; clipping the head before decoding is wrong.
    torch.testing.assert_close(observed[:, 4:8], torch.tensor([[-100., 100., 24., -24.]]))
    torch.testing.assert_close(observed[:, :4], goal[:, :4])
    torch.testing.assert_close(observed[:, 28:35], goal[:, 4:11])
    torch.testing.assert_close(observed[:, 22:28], action)
    torch.testing.assert_close(observed[:, 14:16], torch.zeros(1, 2))
    assert observed[0, 16].item() == 100


def test_observation_clipping_preserves_unsaturated_bptt_and_analytic_gradients():
    world, state, context, goal, action = observation_fixture()
    goal.requires_grad_()
    action.requires_grad_()
    world.observe(state, goal, context, previous_action=action).sum().backward()
    gradient = world.observation_head[-1].bias.grad
    torch.testing.assert_close(gradient[4:8], torch.tensor([0., 0., 2., 2.]))
    torch.testing.assert_close(goal.grad, torch.ones_like(goal))
    torch.testing.assert_close(action.grad, torch.ones_like(action))
    assert torch.count_nonzero(gradient[:4]) == 0
    assert torch.count_nonzero(gradient[14:16]) == 0
    assert torch.count_nonzero(gradient[22:35]) == 0


def test_observation_without_explicit_action_clips_the_existing_learned_fallback():
    world, state, context, goal, _ = observation_fixture()
    with torch.no_grad():
        world.observation_head[-1].bias[22:25] = torch.tensor([-75., 75., 1.5])
    observed = world.observe(state, goal, context)
    torch.testing.assert_close(observed[:, 22:25], torch.tensor([[-100., 100., 3.]]))
