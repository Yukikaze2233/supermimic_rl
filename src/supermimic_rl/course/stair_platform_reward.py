"""Bind ordinary stair reward targets to the already accepted native ToF beam."""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import torch

from ..reward_kernel import StairPlatformGeometry, stair_support_masks


class NativeStairPlatformReward:
    """Observe request ownership without resampling sensors or changing the FSM.

    Manual requests carry no verified platform world height and retain the
    original objective. Binding is per request and episode, never inferred
    from a subsequent root ray, a height command or the protocol's completion.
    """

    def __init__(self, native):
        self.native, self.course = native, native._terrain_course
        self.trigger, self.protocol = self.course.tof_trigger, self.course.stair_requests
        self.request_id = torch.full_like(self.protocol.active_id, -1)
        self.episode_id = torch.full_like(native._episode_id, -1)
        self.top = torch.zeros_like(self.protocol.height_m)
        self.lower = torch.zeros_like(self.top)
        self.transfer_started = torch.zeros_like(self.request_id, dtype=torch.bool)
        self._originals = None

    def install(self):
        if self._originals is not None:
            return self
        record, reset, clear = self.trigger.record_responses, self.protocol.reset, self.protocol.clear

        def record_responses(fired, accepted, *, active_id=None):
            result = record(fired, accepted, active_id=active_id)
            # An accepted external replacement must not inherit a ToF target.
            self.request_id.masked_fill_(accepted, -1)
            self.transfer_started.masked_fill_(accepted, False)
            beam = self.course._last_tof_beam
            if active_id is not None and beam is not None:
                # detect() is pure. Reuse the current, already queried beam;
                # step() has separately consumed its original sensor draws.
                found, _, rise = self.trigger.detect(beam.height_m, beam.valid)
                lower = beam.height_m[:, 0]
                top = lower + rise
                bound = (fired & accepted & (active_id >= 0) & found & torch.isfinite(lower)
                         & torch.isfinite(top) & (rise > 0))
                self.request_id.copy_(torch.where(bound, active_id, self.request_id))
                self.episode_id.copy_(torch.where(bound, self.native._episode_id, self.episode_id))
                self.top.copy_(torch.where(bound, top, self.top))
                self.lower.copy_(torch.where(bound, lower, self.lower))
            if active_id is not None:
                self.transfer_started &= (self.request_id >= 0) & (self.request_id == active_id)
            return result

        def reset_rows(ids, now_s):
            result = reset(ids, now_s)
            self.request_id[ids] = -1
            self.transfer_started[ids] = False
            return result

        def clear_rows(mask):
            result = clear(mask)
            self.request_id.masked_fill_(mask, -1)
            self.transfer_started.masked_fill_(mask, False)
            return result

        self._originals = record, reset, clear
        self.trigger.record_responses = record_responses
        self.protocol.reset, self.protocol.clear = reset_rows, clear_rows
        return self

    def close(self):
        if self._originals is not None:
            self.trigger.record_responses, self.protocol.reset, self.protocol.clear = self._originals
            self._originals = None

    def __call__(self, sample):
        support = self.course._support_sample
        bound = ((self.request_id >= 0) & (self.request_id == self.protocol.active_id)
                 & (self.episode_id == sample.episode_id)
                 & (self.episode_id == self.native._episode_id))
        if support is None:
            # Ordinary invalid/initial terrain samples have no current wheel
            # geometry; absence is not a guessed platform or a task failure.
            bottoms = torch.zeros((*self.top.shape, 2), device=self.top.device, dtype=self.top.dtype)
            valid = torch.zeros_like(bound)
        else:
            bottoms = support.wheel_bottom_world_z_m
            valid = bound & support.valid & ~self.course.query_invalid
        return StairPlatformGeometry(bottoms, self.top, self.lower, valid, self.transfer_started & bound)

    @torch.no_grad()
    def observe(self, sample, reference, base_net_upward_n, *, input_valid):
        """Advance once at this pre-reset physical endpoint, then score that tick.

        The composer supplies its complete identity/timestamp validation and
        the force scoped by install_stair_support_reference to this observe.
        Missing support geometry cannot start the phase, but unloading after
        the first support cannot return the same request to its approach reward.
        """
        geometry = self(sample)
        current = ((self.request_id >= 0) & (self.request_id == self.protocol.active_id)
                   & (self.episode_id == self.native._episode_id))
        self.transfer_started &= current
        _, supported, _ = stair_support_masks(reference, base_net_upward_n)
        same_control = sample.control_epoch == self.native._control_epoch
        finite_geometry = (torch.isfinite(geometry.wheel_bottom_world_z_m).all(-1)
                           & torch.isfinite(self.top) & torch.isfinite(self.lower) & (self.top > self.lower))
        self.transfer_started |= geometry.valid & finite_geometry & input_valid & same_control & supported
        return self(sample)

    @property
    def receipt(self):
        def source_sha(value):
            return hashlib.sha256(Path(inspect.getfile(value)).read_bytes()).hexdigest()

        return {
            'format': 'ordinary-stair-accepted-platform-height-transfer-v2',
            'source_sha256': source_sha(type(self)),
            'native_trigger_source_sha256': source_sha(type(self.trigger)),
            'native_protocol_source_sha256': source_sha(type(self.protocol)),
            'capture': 'record_responses: fired & accepted, current existing ToF beam detect()',
            'target': 'lower=beam.height_m[:,0]; top=lower+detected_rise; actual world metres',
            'wheel_bottom': 'same-tick TerrainSupportSample.wheel_bottom_world_z_m',
            'lifetime': 'accepted request ID and episode; reset/clear invalidate; no external-request target',
            'phase': 'first valid same-tick stair_base_supported latches transfer_started until request end/reset/replace',
            'phase_identity': 'composer pre-reset timestamp/episode/control validation plus current native episode/control',
            'score': 'mean_wheels(clamp(1-relu(top-wheel_bottom)/(top-lower),0,1))',
            'budget': 'replace existing positive.height, maximum 1 per second; all costs unchanged',
            'scope': 'ordinary STAIR, not special gimbal, valid bound target and latched transfer_started',
            'fallback': 'original height objective before first support or when target unavailable; not completion',
            'boundary': 'first phase entry and request end can still change reward discontinuously; not smoothed',
            'sensor': 'no extra terrain query or random draw; ideal simulator geometry, not hardware ToF',
            'world_model': 'unsupported: accepted target and transfer phase are not recorded or advanced in imagination',
        }
