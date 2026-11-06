"""Disturbances remain causal, differentiable and identical across batching."""
import torch

from supermimic_rl.course.randomization import RequestedRandomizationConfig
from supermimic_rl.world_model_v2.disturbance import (
    DISTURBANCE_NOISE_DIM, DisturbanceState, shift_tick_reference,
)
from supermimic_rl.world_model_v2.fitting import WorldModelFit, recorded_rollout
from supermimic_rl.world_model_v2.imagination import Critic
from supermimic_rl.world_model_v2.learning import Learner
from supermimic_rl.world_model_v2.model import Ticks, WorldModelV2
from supermimic_rl.world_model_v2.records import (
    decode_memory, decode_physical, encode_physical,
)
from support.world_v2.offline_continuation import fixture as offline_fixture
from support.world_v2.jump_imagination import fixture as jump_fixture


def extended_model(old):
    model = WorldModelV2(old.inputs, old.wheels, hidden=(16,),
                         terrain=old.terrain, disturbances=True)
    state = old.state_dict()
    state['motion.net.0.weight'] = torch.nn.functional.pad(state['motion.net.0.weight'], (0, 24))
    state['motion.input_norm.mean'] = torch.nn.functional.pad(state['motion.input_norm.mean'], (0, 24))
    state['motion.input_norm.var'] = torch.nn.functional.pad(state['motion.input_norm.var'], (0, 24), value=1.)
    model.load_state_dict(state)
    return model


def test_prepared_draws_preserve_deferred_push_tick_changes_and_row_batching():
    config = RequestedRandomizationConfig(external_force_n=15., external_torque_nm=1.5,
        push_linear_m_s=.5, push_yaw_rad_s=1.)
    state = DisturbanceState(torch.tensor([.01, -.1, .04, .01]),
        torch.tensor([.002, .007, .012, .017]), torch.zeros(4, 6), torch.tensor([1., 1., .5, 0.]))
    noise = torch.rand(4, DISTURBANCE_NOISE_DIM, generator=torch.Generator().manual_seed(14))
    eligible = torch.tensor([True, False, True, True])
    delta, wrench, after = state.advance_prepared(config, eligible, noise)
    torch.testing.assert_close(delta[0], (noise[0, :3] * 2 - 1) * torch.tensor([.5, .5, 1.]))
    assert not delta[1:].any()
    assert after.push_remaining_s[1] < -.1  # Due, but pending during a skill.
    assert not wrench[3].any()  # Nominal episodes still advance their clocks.
    for row in range(4):
        selected = DisturbanceState.decode(state.encode()[row:row + 1])
        one_delta, one_wrench, one_after = selected.advance_prepared(
            config, eligible[row:row + 1], noise[row:row + 1])
        torch.testing.assert_close(one_delta, delta[row:row + 1], rtol=0, atol=0)
        torch.testing.assert_close(one_wrench, wrench[row:row + 1], rtol=0, atol=0)
        torch.testing.assert_close(one_after.encode(), after.encode()[row:row + 1], rtol=0, atol=0)
        assert not wrench[row, :row].any()
        values = noise[row, 4 + row * 7:10 + row * 7]
        expected = (values * 2 - 1) * torch.tensor([15.] * 3 + [1.5] * 3) * state.scale[row]
        torch.testing.assert_close(wrench[row, row:], expected.expand(4 - row, -1))
    # The held force continues without redrawing at the next 20 ms boundary.
    _, held, _ = after.advance_prepared(config, eligible, 1 - noise)
    torch.testing.assert_close(held, wrench[:, -1:, :].expand(-1, 4, -1))


def test_zero_disturbance_extension_keeps_legacy_predictions(tmp_path):
    _, old, replay = offline_fixture(tmp_path)
    model = extended_model(old)
    start = replay.sample_windows(2, 1, env_ids=[0], generator=torch.Generator().manual_seed(2))
    start = {name: value[:, 0] for name, value in start.items()}
    args = (decode_physical(start['physical']), start['height'], start['scan'],
            decode_memory(start['memory']), start['sim_context'], start['motor_strength'][:, 4:6], start['action'])
    expected = old.step(*args)
    actual = model.step(*args, velocity_push=torch.zeros(2, 3), external_wrench=torch.zeros(2, 4, 6))
    torch.testing.assert_close(actual.relative_ticks, expected.relative_ticks, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(encode_physical(actual.state), encode_physical(expected.state), rtol=1e-5, atol=1e-7)


def test_recorded_push_is_applied_once_and_not_learned_as_motion(tmp_path):
    _, old, replay = offline_fixture(tmp_path)
    model = extended_model(old)
    with torch.no_grad():
        model.motion.net[-1].weight.zero_()
        model.motion.net[-1].bias.zero_()
    windows = replay.sample_windows(2, 3, env_ids=[0], generator=torch.Generator().manual_seed(2))
    windows['velocity_push'] = torch.tensor([.1, -.2, .3]).expand(2, 3, -1).clone()
    windows['external_wrench'] = torch.zeros(2, 3, 4, 6)
    windows['sim_context'][..., 41:44] = torch.tensor([.03, .01, .08])
    state = decode_physical(windows['physical'][:, 0])
    for step in range(3):
        windows['physical'][:, step] = encode_physical(state)
        pushed = model.physical_start(state, windows['sim_context'][:, step], windows['velocity_push'][:, step])
        relative = shift_tick_reference(torch.zeros_like(windows['ticks'][:, step]), state, pushed)
        windows['ticks'][:, step] = relative
        windows['end_physical'][:, step] = encode_physical(pushed)
        torch.testing.assert_close(Ticks.decode(relative, state).state.velocity[:, -1], pushed.velocity)
        torch.testing.assert_close(shift_tick_reference(relative, state, pushed, to_pushed=True),
                                   torch.zeros_like(relative), atol=1e-7, rtol=0)
        state = pushed
    predictions = list(recorded_rollout(model, windows))
    assert len(predictions) == 3
    for step, _, prediction, _ in predictions:
        torch.testing.assert_close(prediction.state.velocity, decode_physical(windows['end_physical'][:, step]).velocity)
        torch.testing.assert_close(prediction.relative_ticks, windows['ticks'][:, step], atol=1e-7, rtol=0)
    fit = WorldModelFit(model)
    losses = fit.terms(windows)
    torch.testing.assert_close(losses.state, torch.zeros_like(losses.state), atol=1e-10, rtol=0)
    torch.testing.assert_close(losses.ticks, torch.zeros_like(losses.ticks), atol=1e-10, rtol=0)
    fit.observe(windows)
    torch.testing.assert_close(model.motion.output_norm.mean, torch.zeros_like(model.motion.output_norm.mean), atol=1e-7, rtol=0)
    # Every wrench tick is available to learning and contributes gradients.
    windows['external_wrench'].requires_grad_()
    with torch.no_grad():
        model.motion.net[0].weight[:, -24:].normal_(0, .1)
        model.motion.net[-1].weight.normal_(0, .1)
    loss = WorldModelFit(model).terms(windows)
    gradient = torch.autograd.grad(loss.state + loss.ticks, windows['external_wrench'])[0]
    assert torch.isfinite(gradient).all() and (gradient.abs().sum((0, 1, 3)) > 0).all()


def test_imagination_advances_disturbances_without_changing_initial_actor_history():
    old, start, history = jump_fixture()
    model = extended_model(old.model)
    critic = Critic(old.critic.norm.mean.numel() + 9, (16,), task_dim=old.critic.task_dim,
                    terrain=True, stairs=True, jumps=True, disturbances=True)
    config = RequestedRandomizationConfig(external_force_n=15., external_torque_nm=1.5,
        push_linear_m_s=.5, push_yaw_rad_s=1.)
    learner = Learner(model, old.actor, critic, old.imagination.termination, old.config,
        reward=old.imagination.reward, stair_task=old.imagination.stair_task,
        stair_source=old.imagination.stair_source, jump_task=old.imagination.jump_task,
        disturbance_config=config)
    start['disturbance'] = DisturbanceState(torch.tensor([.01, .03]),
        torch.tensor([.007, .012]), torch.zeros(2, 6), torch.ones(2)).encode()
    start['disturbance_record_valid'] = torch.ones(2, 1)
    start = learner.imagination.prepare(start, 3)
    rng = torch.random.get_rng_state()
    assert learner.imagination.prepare(start, 3)['disturbance_uniform'] is start['disturbance_uniform']
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    actor_inputs, pushes, wrenches = [], [], []
    actor_hook = old.actor.register_forward_pre_hook(lambda _, args: actor_inputs.append(args[0].detach().clone()))
    original_step = model.step

    def capture(*args, **kwargs):
        pushes.append(kwargs['velocity_push'].clone())
        wrenches.append(kwargs['external_wrench'].clone())
        return original_step(*args, **kwargs)

    model.step = capture
    rollout = learner.imagination.rollout(start, history, 3)
    actor_hook.remove()
    expected_push, expected_wrench, next_state = DisturbanceState.decode(start['disturbance']).advance_prepared(
        config, torch.tensor([True, True]), start['disturbance_uniform'][:, 0])
    torch.testing.assert_close(actor_inputs[0], history, rtol=0, atol=0)
    torch.testing.assert_close(pushes[0], expected_push, rtol=0, atol=0)
    torch.testing.assert_close(wrenches[0], expected_wrench, rtol=0, atol=0)
    torch.testing.assert_close(rollout.critic_inputs[0, :, -9:], start['disturbance'], rtol=0, atol=0)
    torch.testing.assert_close(rollout.critic_inputs[1, :, -9:], next_state.encode(), rtol=0, atol=0)
    assert rollout.usable.all()
    gradient = torch.autograd.grad(rollout.objective.sum(), tuple(old.actor.decoder.parameters()))
    assert all(torch.isfinite(value).all() for value in gradient)
    assert sum(value.abs().sum() for value in gradient) > 0
    # A fixed realization must retain the same BPTT objective when rows are
    # evaluated individually; model batching may change only float rounding.
    split = []
    for row in range(2):
        single = learner.imagination.rollout({key: value[row:row + 1] for key, value in start.items()},
                                            history[row:row + 1], 3)
        split.append(single.objective)
    torch.testing.assert_close(torch.cat(split), rollout.objective, rtol=1e-5, atol=1e-6)
    parameter = old.actor.decoder[-1].bias
    baseline = parameter.detach().clone()
    direction = gradient[-1] / gradient[-1].norm()
    epsilon = .001
    with torch.no_grad():
        parameter.copy_(baseline + epsilon * direction)
        positive = learner.imagination.rollout(start, history, 3).objective.sum()
        parameter.copy_(baseline - epsilon * direction)
        negative = learner.imagination.rollout(start, history, 3).objective.sum()
        parameter.copy_(baseline)
    analytic = (gradient[-1] * direction).sum()
    numeric = (positive - negative) / (2 * epsilon)
    assert (numeric - analytic).abs() / analytic.abs().clamp_min(1e-8) < .01
