"""Optional wheel-support regression retains physical labels and rollout masks."""
import pytest
import torch

from supermimic_rl.world_model_v2.fitting import WorldModelFit
from supermimic_rl.world_model_v2.physical import INCREMENT_DIM
from support.world_v2.offline_continuation import fixture


def test_wheel_regression_uses_all_ticks_existing_scales_validity_and_continuous_gradients(tmp_path):
    _, model, replay = fixture(tmp_path)
    windows = replay.sample_windows(2, 8, env_ids=[0], generator=torch.Generator().manual_seed(4))
    # One invalid interval ends this trajectory; subsequent labels must not
    # be scored, but the denominator remains the original B*H as for motion.
    windows['valid'][0, 2] = False
    wheel = slice(INCREMENT_DIM, INCREMENT_DIM + 2)
    with torch.no_grad():
        model.motion.net[-1].weight.zero_()
        model.motion.net[-1].bias.zero_()
        model.motion.output_norm.mean.zero_()
        model.motion.output_norm.var.fill_(1.)
        model.motion.output_norm.var.view(4, -1)[:, wheel] = torch.tensor(
            [[1., 100.], [16., 225.], [625., 36.], [49., 1.]])
        model.motion.net[-1].bias.view(4, -1)[:, wheel] = torch.tensor(
            [[-2., 1.], [-3., 2.], [-4., 3.], [-5., 4.]])
    schema_keys = tuple(model.state_dict())
    baseline = WorldModelFit(model).terms(windows)
    default_gradient = torch.autograd.grad(baseline.state + baseline.ticks, model.motion.net[-1].bias)[0]
    assert baseline.wheel_contact is None
    fit = WorldModelFit(model, wheel_contact_weight=1.)
    actual = fit.terms(windows)
    assert tuple(model.state_dict()) == schema_keys
    torch.testing.assert_close(actual.state, baseline.state, rtol=0, atol=0)
    torch.testing.assert_close(actual.ticks, baseline.ticks, rtol=0, atol=0)
    unchanged_gradient = torch.autograd.grad(actual.state + actual.ticks, model.motion.net[-1].bias,
                                             retain_graph=True)[0]
    torch.testing.assert_close(default_gradient, unchanged_gradient, rtol=0, atol=0)
    std = model.motion.output_norm.std().view(4, -1)[:, wheel]
    raw = model.motion.net[-1].bias.view(4, -1)[:, wheel]
    expected = (raw * std / std.clamp_min(5.)).square().mean() * (10 / 16)
    torch.testing.assert_close(actual.wheel_contact, expected)
    # Two of 56 channels were previously diluted in the tick-group reduction.
    torch.testing.assert_close(actual.wheel_contact, actual.ticks * 28.)
    gradient = torch.autograd.grad(actual.wheel_contact, model.motion.net[-1].bias)[0].view(4, -1)
    expected_gradient = torch.autograd.grad(expected, model.motion.net[-1].bias)[0].view(4, -1)
    torch.testing.assert_close(gradient, expected_gradient)
    assert (gradient[:, INCREMENT_DIM] < 0).all()  # Negative N predictions are penalized too.
    assert (gradient[:, INCREMENT_DIM + 1] > 0).all()
    assert not gradient[:, :INCREMENT_DIM].any() and not gradient[:, INCREMENT_DIM + 2:].any()


@pytest.mark.parametrize('weight', [-1., float('nan'), float('inf')])
def test_wheel_contact_weight_rejects_invalid_values(weight):
    with pytest.raises(ValueError, match='finite and nonnegative'):
        WorldModelFit(None, wheel_contact_weight=weight)
