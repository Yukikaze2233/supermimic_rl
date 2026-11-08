"""Bind ordinary stair reward targets to the already accepted native ToF beam."""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import torch

from ..reward_kernel import StairPlatformGeometry


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
        self._originals = None

    def install(self):
        if self._originals is not None:
            return self
        record, reset, clear = self.trigger.record_responses, self.protocol.reset, self.protocol.clear

        def record_responses(fired, accepted, *, active_id=None):
            result = record(fired, accepted, active_id=active_id)
            # An accepted external replacement must not inherit a ToF target.
            self.request_id.masked_fill_(accepted, -1)
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
            return result

        def reset_rows(ids, now_s):
            result = reset(ids, now_s)
            self.request_id[ids] = -1
            return result

        def clear_rows(mask):
            result = clear(mask)
            self.request_id.masked_fill_(mask, -1)
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
        return StairPlatformGeometry(bottoms, self.top, self.lower, valid)

    @property
    def receipt(self):
        def source_sha(value):
            return hashlib.sha256(Path(inspect.getfile(value)).read_bytes()).hexdigest()

        return {
            'format': 'ordinary-stair-accepted-platform-height-v1',
            'source_sha256': source_sha(type(self)),
            'native_trigger_source_sha256': source_sha(type(self.trigger)),
            'native_protocol_source_sha256': source_sha(type(self.protocol)),
            'capture': 'record_responses: fired & accepted, current existing ToF beam detect()',
            'target': 'lower=beam.height_m[:,0]; top=lower+detected_rise; actual world metres',
            'wheel_bottom': 'same-tick TerrainSupportSample.wheel_bottom_world_z_m',
            'lifetime': 'accepted request ID and episode; reset/clear invalidate; no external-request target',
            'score': 'mean_wheels(clamp(1-relu(top-wheel_bottom)/(top-lower),0,1))',
            'budget': 'replace existing positive.height, maximum 1 per second; all costs unchanged',
            'scope': 'existing stair_base_supported gate and a valid bound target',
            'fallback': 'original height objective when unsupported or target unavailable; not completion',
            'sensor': 'no extra terrain query or random draw; ideal simulator geometry, not hardware ToF',
            'world_model': 'unsupported: accepted target context is not recorded or advanced in imagination',
        }
