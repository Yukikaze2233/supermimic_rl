"""Target-conditioned contact estimates over the unchanged 56D motion output.

The head predicts contact attribution, not fictitious contact patches. Its
boolean evidence is an estimate for the shared task updater, never a native
measurement or a success certificate. Recorded future evidence is accepted
only by ``targets``; forward uses predicted ticks and the current caller.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..course.jump_physics import PlatformLandingEvidence
from .dynamics import RunningNorm, mlp
from .features import physical_features
from .physical import INCREMENT_DIM, apply
from .platform_records import decode_platform_interval
from .records import CONTACT_BODIES, WHEEL_BODIES, decode_physical


# Physical30, knees4, loads26, wheel poses12, target offsets6, target flags6,
# ground columns4, cylinder gaps4, request values2, root-ground validity1.
PLATFORM_CONTACT_FEATURE_DIM = 30 + 4 + 26 + 12 + 6 + 6 + 4 + 4 + 2 + 1
PLATFORM_CONTACT_OUTPUT_DIM = 4
PLATFORM_CONTACT_PRODUCER = 'predicted-platform-contact-evidence-v1'


@dataclass(frozen=True)
class PlatformContactPrediction:
    evidence: PlatformLandingEvidence
    logits: Tensor                       # [N, 4, 3]: target L/R, legitimate support.
    support_height_relative_root: Tensor  # [N, 4]: support ground Z minus predicted root Z.
    features: Tensor                     # [N, 4, 95], before the head's own normalization.
    available: Tensor                    # [N, 4]: source/input availability, not confidence.
    producer: str = PLATFORM_CONTACT_PRODUCER


@dataclass(frozen=True)
class PlatformContactTargets:
    contact: Tensor                      # [N, 4, 3], native booleans converted to floats.
    support_height_relative_root: Tensor  # [N, 4], zero placeholder where height_valid is false.
    valid: Tensor
    height_valid: Tensor


class PlatformContactModel(nn.Module):
    """Four outputs per tick, with independent normalization and explicit width.

    The first three are Bernoulli logits. The fourth is supporting ground
    height relative to the predicted root. All physical thresholds come from
    the existing native skill evidence configuration. A zero-logit classification boundary is not a new
    hardware threshold. Probabilities remain differentiable for future users;
    this module creates no reward and no policy admission rule.
    """

    def __init__(self, wheel_geometry, platform_geometry, clearance, *, hidden: tuple[int, ...]):
        super().__init__()
        from transformer_rl.formal.tasks.skills import SkillEvidenceConfig

        if any(type(width) is not int or width <= 0 for width in hidden):
            raise ValueError('Platform contact hidden widths must be positive integers')
        if clearance.terrain is not platform_geometry.terrain:
            raise ValueError('Platform contacts and cylinder gaps must share the actual atlas')
        self.wheel_geometry, self.geometry, self.clearance = wheel_geometry, platform_geometry, clearance
        self.config = SkillEvidenceConfig()
        self.net = mlp(PLATFORM_CONTACT_FEATURE_DIM, hidden, PLATFORM_CONTACT_OUTPUT_DIM, layer_norm=True)
        self.input_norm = RunningNorm(PLATFORM_CONTACT_FEATURE_DIM)
        self.height_norm = RunningNorm(1, eps=1e-4)
        self.wheel_indices = tuple(CONTACT_BODIES.index(name) for name in WHEEL_BODIES)
        self.nonwheel_indices = tuple(i for i in range(len(CONTACT_BODIES)) if i not in self.wheel_indices)

    @staticmethod
    def _command_ticks(value, n):
        if value.shape == (n,):
            value = value[:, None].expand(-1, 4)
        if value.shape != (n, 4) or not value.is_floating_point():
            raise ValueError('Command height must be the actual admitted height for each row/tick')
        return value

    def _features(self, ticks, intent, terrain_tile, selected):
        """Query only bound requests; ordinary rows do no platform geometry work."""
        n, prototype = len(selected), ticks.state.position
        features = prototype.new_zeros(n, 4, PLATFORM_CONTACT_FEATURE_DIM)
        available = torch.zeros(n, 4, dtype=torch.bool, device=prototype.device)
        gap = prototype.new_full((n, 4, 2), torch.nan)
        gap_valid = torch.zeros_like(gap, dtype=torch.bool)
        ids = selected.nonzero().flatten()
        if not len(ids):
            return features, available, gap, gap_valid
        part = replace(ticks, state=ticks.state.map(lambda value: value[ids]),
            **{name: getattr(ticks, name)[ids] for name in ('wheel_load', 'body_normal_load',
                'resultant_contact_force', 'passive_knee_position', 'passive_knee_velocity')})
        plan = replace(intent, **{name: getattr(intent, name)[ids] for name in intent.__dataclass_fields__})
        values = self._selected_features(part, plan, terrain_tile[ids])
        return tuple(base.index_copy(0, ids, value) for base, value in
                     zip((features, available, gap, gap_valid), values))

    def _selected_features(self, ticks, intent, terrain_tile):
        """Use predicted passive knees, not an ideal closure projection."""
        state, n = ticks.state, len(terrain_tile)
        wheels = self.wheel_geometry(state, passive_knee_position=ticks.passive_knee_position,
                                     passive_knee_velocity=ticks.passive_knee_velocity)
        centres, axes = wheels.centre_world, wheels.cylinder_axis_world
        gap = self.clearance.at(centres.reshape(-1, 2, 3), axes.reshape(-1, 2, 3),
                                terrain_tile[:, None].expand(-1, 4).reshape(-1))
        gap_value = gap.vertical_gap_m.reshape(n, 4, 2)
        gap_valid = (gap.valid & gap.domain_covered & torch.isfinite(gap.vertical_gap_m)).reshape(n, 4, 2)
        root = self.geometry.terrain.height_at(state.position, terrain_tile)
        ground = self.geometry.terrain.height_at(centres, terrain_tile)
        point = centres.reshape(-1, 3)
        tile = terrain_tile[:, None, None].expand(-1, 4, 2).reshape(-1)
        target = intent.target_index[:, None, None].expand(-1, 4, 2).reshape(-1)
        near = self.geometry.nearest(point, tile, target)
        surface = self.geometry.query(point, tile, target)
        source_valid = near.source_valid & near.atlas_valid
        target_offset = torch.where(source_valid[:, None], near.point_local - point, 0.).reshape(n, 4, 2, 3)
        edge = torch.where(source_valid, surface.edge_distance, 0.).reshape(n, 4, 2)
        source_valid, visible = source_valid.reshape(n, 4, 2), near.visible.reshape(n, 4, 2)
        rotate = state.rotation.transpose(-1, -2)[..., None, :, :]
        body = lambda value: (rotate @ value[..., None]).squeeze(-1).flatten(-2)
        loads = torch.cat((ticks.wheel_load, ticks.body_normal_load, ticks.resultant_contact_force), -1)
        height = torch.where(root.valid, state.position[..., 2] - root.height, 0.)
        centre_gap = torch.where(ground.valid, wheels.bottom_world_z - ground.height, 0.)
        plan = torch.stack((intent.rise_m, intent.support_height_m), -1)[:, None].expand(-1, 4, -1)
        features = torch.cat((physical_features(state, height[..., None]),
            ticks.passive_knee_position, ticks.passive_knee_velocity, loads,
            body(centres - state.position[..., None, :]), body(axes), body(target_offset),
            edge, source_valid, visible, centre_gap, ground.valid,
            torch.where(gap_valid, gap_value, 0.), gap_valid, plan, root.valid[..., None]), -1)
        available = source_valid.all(-1) & wheels.valid & torch.isfinite(features).all(-1)
        # Undefined geometry is represented by its availability column. Never
        # feed NaN placeholders to the MLP or include them in normalization.
        return torch.where(torch.isfinite(features), features, 0.), available, gap_value, gap_valid

    def forward(self, ticks, *, intent, jump_state, terrain_tile, command_height):
        return self.predict(ticks, intent=intent, jump_state=jump_state,
                            terrain_tile=terrain_tile, command_height=command_height).evidence

    def predict(self, ticks, *, intent, jump_state, terrain_tile, command_height):
        n, device = len(ticks.state.position), ticks.state.position.device
        intent.validate()
        if (ticks.state.position.shape != (n, 4, 3) or terrain_tile.shape != (n,)
                or terrain_tile.dtype != torch.long or terrain_tile.device != device
                or intent.target_index.device != device or len(intent.target_index) != n
                or jump_state.physical_id.shape != (n,)):
            raise ValueError('Platform prediction requires four ticks and matching current caller/tile rows')
        command = self._command_ticks(command_height, n)
        selected = ((intent.target_index >= 0) & (intent.bound_request_id >= 0)
                    & (intent.bound_request_id == jump_state.physical_id))
        features, available, gap, gap_valid = self._features(ticks, intent, terrain_tile, selected)
        ids = selected.nonzero().flatten()
        predicted = self.net(self.input_norm(features[ids]).to(self.net[0].weight.dtype)).to(features.dtype)
        raw = features.new_zeros(n, 4, PLATFORM_CONTACT_OUTPUT_DIM).index_copy(0, ids, predicted)
        logits = raw[..., :3]
        height = (raw[..., 3:4] * self.height_norm.std() + self.height_norm.mean)[..., 0]
        c = self.config
        wheel_touch = ticks.body_normal_load[..., self.wheel_indices] > c.airborne_load_max_n
        nonwheel = (ticks.body_normal_load[..., self.nonwheel_indices] > c.airborne_load_max_n).any(-1)
        any_contact = wheel_touch.any(-1) | nonwheel
        finite = (torch.isfinite(logits).all(-1) & torch.isfinite(height) & torch.isfinite(command))
        valid = available & finite & (gap_valid.all(-1) | any_contact)
        # Head attribution can describe a rim contact while the centre is still
        # outside the target. Neither centre membership nor a gap witness is a patch.
        target_touch = ((logits[..., :2] > 0) & wheel_touch & (ticks.wheel_load > 0)
                        & valid[..., None])
        supported = (valid & (logits[..., 2] > 0) & gap_valid.all(-1)
                     & (ticks.wheel_load >= c.grounded_load_min_n).all(-1)
                     & (gap >= -c.support_penetration_max_m).all(-1)
                     & (gap <= c.support_gap_max_m).all(-1))
        clear = valid & gap_valid.all(-1) & ~any_contact & (gap > 0).all(-1)
        # Shared imagined physics uses tile-local heights, as do predicted states.
        # The field retains the native API name; no world-origin offset is added.
        evidence = PlatformLandingEvidence(selected[:, None].expand(-1, 4), valid, clear,
            wheel_touch, target_touch, nonwheel, supported,
            torch.where(supported, ticks.state.position[..., 2] + height + command, torch.nan))
        return PlatformContactPrediction(evidence, logits, height, features, available)

    def targets(self, record, *, command_height):
        """Decode v2 truth; command height must match native ``_command[:, 3]``.

        A policy-start goal, requested support height, or anchor is not a general
        substitute for this admitted per-tick value across a command boundary.
        World Z is converted through the record's actual atlas tile origins,
        then made relative to the actual recorded physical tick's root Z.
        Invalid/inactive legacy records cannot silently become negative labels.
        """
        interval = decode_platform_interval(record)
        truth, n = interval.evidence, len(interval.evidence.selected)
        command = self._command_ticks(command_height, n)
        tile = record['terrain_tile'][:, 0]
        if (not torch.isfinite(tile).all() or not (tile == tile.round()).all()
                or (tile < 0).any() or (tile >= len(self.geometry.terrain.origins)).any()):
            raise ValueError('Platform target labels require recorded atlas tile origins')
        valid = (record['valid'][:, None] & record['platform_record_valid']
                 & record['contact_valid_ticks'][..., 0] & truth.selected & truth.measurement_valid)
        height_valid = valid & truth.landing_support_valid
        tick_state = apply(decode_physical(record['physical']).with_time_axis(),
                           record['ticks'][..., :INCREMENT_DIM])
        local = (truth.landing_world_height - self.geometry.terrain.origins[tile.long(), 2, None]
                 - command - tick_state.position[..., 2])
        if (height_valid & ~torch.isfinite(local)).any():
            raise ValueError('A supported native platform label must have finite height and command')
        return PlatformContactTargets(torch.cat((truth.target_wheel_touch,
            truth.landing_support_valid[..., None]), -1).to(local.dtype),
            torch.where(height_valid, local, 0.), valid, height_valid)

    @torch.no_grad()
    def observe(self, prediction: PlatformContactPrediction, targets: PlatformContactTargets):
        """Explicit fit-only update from real labeled examples; never in forward."""
        valid = prediction.available & targets.valid
        self.input_norm.update(prediction.features[valid])
        self.height_norm.update(targets.support_height_relative_root[valid & targets.height_valid][..., None])

    def losses(self, prediction: PlatformContactPrediction, targets: PlatformContactTargets):
        """Independent supervised terms, with no policy reward or implied recall."""
        valid = prediction.available & targets.valid
        height_valid = valid & targets.height_valid
        contact = F.binary_cross_entropy_with_logits(prediction.logits, targets.contact, reduction='none')
        average = lambda values, mask: values[mask].sum() / mask.sum().clamp_min(1)
        height_error = ((prediction.support_height_relative_root - targets.support_height_relative_root)
                        / self.height_norm.std()[0]).square()
        return {'target_contact': average(contact[..., :2].mean(-1), valid),
                'support_contact': average(contact[..., 2], valid),
                'support_height': average(height_error, height_valid)}
