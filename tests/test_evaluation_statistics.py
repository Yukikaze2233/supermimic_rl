"""Raw and clipped tracking errors retain valid physical failures and exclude bad rows."""
import math
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.ppo.trainer import EpisodeStatistics


def step(errors, *, reward=None, valid=None, numerical=None, terminated=None):
    n = len(errors)
    return SimpleNamespace(
        reward=torch.ones(n) if reward is None else torch.tensor(reward),
        terminated=torch.zeros(n, dtype=torch.bool) if terminated is None else torch.tensor(terminated),
        truncated=torch.zeros(n, dtype=torch.bool),
        valid=None if valid is None else torch.tensor(valid),
        numerical_invalid=None if numerical is None else torch.tensor(numerical),
        tracking={'forward_error_m_s': torch.tensor(errors, dtype=torch.float32)})


def test_mae_keeps_true_large_errors_and_separately_reports_sample_weighted_clipped_mae():
    stats = EpisodeStatistics(3, 'cpu')
    stats.record(step([-.2, 3., float('nan')], reward=[1., 2., float('nan')],
                      valid=[True, True, False], numerical=[False, False, True],
                      terminated=[False, True, True]))
    stats.record(step([-.4, 0., 10.], valid=[True, True, False], numerical=[False, False, True]))
    result = stats.summary()
    assert result['observed_transitions'] == 6
    assert result['valid_tracking_samples'] == 4
    assert result['invalid_transitions'] == result['numerical_invalid_transitions'] == 2
    assert result['nonfinite_measurement_transitions'] == 1
    assert result['terminations'] == 1
    assert result['episode/count'] == 1 and result['episode/return_mean'] == 2.
    assert result['rms/forward_error_m_s'] == pytest.approx(math.sqrt((.04 + 9 + .16) / 4))
    assert result['mae/forward_error_m_s'] == pytest.approx(.9)
    assert result['clipped_mae/forward_error_m_s'] == pytest.approx(.4)
    assert result['clipped_mae_limits']['forward_error_m_s'] == 1.
    assert 'trajectory-weighted' in result['clipped_mae_reference']


def test_invalid_only_is_not_zero_error_or_a_completed_physical_episode():
    stats = EpisodeStatistics(2, 'cpu')
    stats.record(step([2., float('nan')], valid=[True, True], numerical=[True, False],
                      terminated=[True, True]))
    report = stats.summary()
    assert report['valid_tracking_samples'] == report['terminations'] == 0
    assert report['invalid_transitions'] == 2 and report['numerical_invalid_transitions'] == 1
    assert report['rms/forward_error_m_s'] is None
    assert report['mae/forward_error_m_s'] is None
    assert report['clipped_mae/forward_error_m_s'] is None
    assert 'episode/count' not in report
    stats.reset_window()
    stats.record(step([.5, .25], terminated=[True, True]))
    assert stats.summary()['episode/length_mean'] == 1.
    assert stats.summary()['mae/forward_error_m_s'] == .375
