"""Actor-only command features over the unchanged deployed chassis history.

The estimator and replay retain the original support-height command. The
decoder can instead see the visible-clock jump reference in that same channel.
This height-only tensor kernel follows the pinned JumpReferenceProgram; parity
tests cover its stages and support-range-limited extension. It avoids native
input validation/synchronization inside every imagined actor call.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

import torch
from torch import nn

JUMP_REFERENCE_FEATURES = 'jump_reference_height_v1'
ONNX_FEATURE_METADATA = 'supermimic.actor_features'
_REFERENCE_FIELDS = {
    'gravity_m_s2', 'support_height_min_m', 'support_height_max_m', 'jump_min_m', 'jump_max_m',
    'compression_depth_max_m', 'compression_duration_s', 'nominal_extension_duration_s',
}


def actor_feature_config(value: dict | None) -> dict | None:
    if value is None:
        return None
    if (not isinstance(value, dict) or set(value) != {'version', 'reference'}
            or value['version'] != JUMP_REFERENCE_FEATURES
            or not isinstance(value['reference'], dict)
            or set(value['reference']) != _REFERENCE_FIELDS):
        raise ValueError('Actor features require a version and the complete native jump reference')
    # Actor loading/export must work before the native factory adds its pinned
    # source path, and on deployment machines without the training environment.
    c = value['reference']
    if (any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in c.values())
            or c['support_height_min_m'] >= c['support_height_max_m']
            or c['jump_min_m'] > c['jump_max_m']
            or c['compression_depth_max_m'] >= c['support_height_max_m'] - c['support_height_min_m']):
        raise ValueError('Actor jump reference requires positive heights, durations and gravity')
    return {'version': JUMP_REFERENCE_FEATURES, 'reference': dict(c)}


class ActorFeatures(nn.Module):
    def __init__(self, configuration=None):
        super().__init__()
        self.configuration = actor_feature_config(configuration)
        if self.configuration is not None:
            c = self.configuration['reference']
            # Native reference bounds are float32 measurements promoted to double.
            self.register_buffer('_height_limits', torch.tensor(
                [c['support_height_min_m'], c['support_height_max_m']], dtype=torch.float32).double(),
                persistent=False)

    def validate_reference(self, reference):
        if self.configuration is not None and (reference is None
                or asdict(reference) != self.configuration['reference']):
            raise ValueError('Actor reference features and native jump reference differ')

    def forward(self, history):
        if self.configuration is None:
            return history
        if history.shape[-1] != 35:
            raise ValueError('Jump reference features require the 35D chassis frames')
        c = self.configuration['reference']
        frames = history.reshape(-1, 35)
        active = frames[:, 28:33].argmax(-1) == 4
        support, requested = frames[:, 3] / 5., frames[:, 33] / 5.
        elapsed = frames[:, 34].double()
        baseline = torch.where(active, support, c['support_height_min_m']).double()
        rise = torch.where(active, requested, c['jump_min_m']).double()
        depth = (baseline - self._height_limits[0]).clamp(0, c['compression_depth_max_m'])
        low = baseline - depth
        g, nominal_t = c['gravity_m_s2'], c['nominal_extension_duration_s']
        root = ((g * nominal_t)**2 + 8 * g * (rise + depth)).sqrt()
        velocity = 4 * g * (rise + depth) / (root + g * nominal_t)
        travel = velocity * nominal_t / 2
        available = self._height_limits[1] - low
        limited = travel > available
        travel = torch.minimum(travel, available)
        velocity = torch.where(limited, (2 * g * (rise + depth - travel)).clamp_min(0).sqrt(), velocity)
        extension_t = torch.where(limited, 2 * travel / velocity.clamp_min(1e-12), nominal_t)
        launch = c['compression_duration_s'] + extension_t
        takeoff_height = low + travel
        return_time = launch + (velocity + (velocity.square()
            + 2 * g * (takeoff_height - baseline)).clamp_min(0).sqrt()) / g
        u = (elapsed / c['compression_duration_s']).clamp(0, 1)
        compressed = baseline - depth * u.pow(3) * (10 - 15 * u + 6 * u.square())
        u_ext = ((elapsed - c['compression_duration_s']) / extension_t).clamp(0, 1)
        extended = low + velocity * extension_t * (u_ext.pow(3) - .5 * u_ext.pow(4))
        flight_t = (elapsed - launch).clamp_min(0)
        ballistic = takeoff_height + velocity * flight_t - .5 * g * flight_t.square()
        height = torch.where(elapsed < c['compression_duration_s'], compressed,
            torch.where(elapsed < launch, extended, torch.where(elapsed < return_time, ballistic, baseline)))
        feature = torch.where(active, height.float() * 5., frames[:, 3])
        return torch.cat((frames[:, :3], feature[:, None], frames[:, 4:]), -1).reshape_as(history)


def annotate_onnx_features(path: Path, configuration) -> None:
    """The transform is inside ONNX; consumers must feed the original 35D history."""
    import onnx

    model = onnx.load(str(path))
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    metadata[ONNX_FEATURE_METADATA] = json.dumps(actor_feature_config(configuration), sort_keys=True)
    onnx.helper.set_model_props(model, metadata)
    onnx.save(model, str(path))
