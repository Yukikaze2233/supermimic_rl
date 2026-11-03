"""Intermediate critic targets must not retain unused actor graphs."""

from types import MethodType

import pytest
import torch

from supermimic_rl.world_model_v2.imagination import Imagination
from supermimic_rl.world_model_v2.learning import frozen_parameters, lambda_returns
from supermimic_rl.world_model_v2.motion_task import MotionTask

from support.world_v2.stair_imagination import fixture


@pytest.mark.parametrize('early_boundary', [False, True])
def test_detached_critic_targets_preserve_values_and_all_actor_gradients(early_boundary):
    optimized, start, frames = fixture(sensor_noise=True)
    reference, _, _ = fixture(sensor_noise=True)
    if early_boundary:
        task = MotionTask.decode(start['motion_task'])
        task.route[:, 4] = 1.
        task.timing[:, :2] = torch.tensor([[.18, .18], [0., 0.]])
        start['motion_task'] = task.encode()
        start['tof_segment_end'] = torch.full((2, 1), .04)
        start['horizon'][:, 0] = 0.
        start['stair_ids'][:, 1:3] = torch.tensor([7, 3])
        start['stair_clock'][:, 0] = start['stair_clock'][:, 2] = torch.tensor([.18, 0.], dtype=torch.float64)
        start['stair_clock'][:, 1] = 0.
        start['stair_values'][:, 0] = .20
        start['failure_conditions'][:, 1] = 1.

    # Re-enable the former intermediate graph while using identical physical,
    # sensor and request producers. Only target autograd recording differs.
    reference.imagination._critic_targets = MethodType(
        Imagination._critic_targets.__wrapped__, reference.imagination)

    def run(learner):
        saved_bytes = 0

        def pack(tensor):
            nonlocal saved_bytes
            saved_bytes += tensor.numel() * tensor.element_size()
            return tensor

        with frozen_parameters(learner.model, learner.critic):
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                rollout = learner.imagination.rollout(start, frames, 16)
            parameters = tuple(learner.actor.decoder.parameters())
            gradients = torch.autograd.grad(rollout.objective.sum(), parameters, retain_graph=True)
            if early_boundary:
                assert rollout.lengths.tolist() == [1, 10]
                assert rollout.final_survival[0] == 1
                # The shorter trajectory still bootstraps through its own
                # frozen boundary state, not a detached critic target.
                bootstrap = rollout.objective[0] - rollout.rewards[0, 0]
                bootstrap_gradients = torch.autograd.grad(bootstrap, parameters)
                assert sum(gradient.square().sum() for gradient in bootstrap_gradients) > 0
            else:
                bootstrap_gradients = ()
        return rollout, gradients, bootstrap_gradients, saved_bytes

    actual, actual_gradients, actual_bootstrap, actual_bytes = run(optimized)
    expected, expected_gradients, expected_bootstrap, expected_bytes = run(reference)
    for name in ('objective', 'critic_inputs', 'values', 'rewards', 'active', 'boundary', 'lengths'):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
    torch.testing.assert_close(lambda_returns(actual, .99, .95), lambda_returns(expected, .99, .95),
                               rtol=0, atol=0)
    for actual_gradient, expected_gradient in zip(actual_gradients + actual_bootstrap,
                                                 expected_gradients + expected_bootstrap):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
    assert actual_bytes < expected_bytes
