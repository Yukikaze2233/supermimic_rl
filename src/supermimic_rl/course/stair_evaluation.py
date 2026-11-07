"""Evaluation-only stair request boundaries and same-riser retry provenance."""
from __future__ import annotations

from copy import deepcopy
import math

import torch


class StairRequestEvaluation:
    """Keep raw protocol events separate from physical whole-course success.

    An estimated completion is not a traversal certificate. These records
    identify retries; the existing authored-terrain observer judges passage.
    No hook draws randomness or changes a native return value.
    """

    def __init__(self, env, row_names):
        self.env, self.course = env, env._terrain_course
        self.row_names = tuple(row_names)
        self.active = [None] * len(row_names)
        self.previous = [None] * len(row_names)
        self.records, self._restore = [], []
        self.fired = torch.zeros_like(self.course.stair_requests.active_id, dtype=torch.bool)
        if len(row_names) != len(self.fired):
            raise ValueError('Stair evaluation needs one scenario per environment')

    def install(self):
        from transformer_rl.formal.tasks.protocol import RequestResponse

        if self._restore:
            raise RuntimeError('Stair evaluation is already installed')
        course, protocol = self.course, self.course.stair_requests
        tof, step, record_end = course._tof_command, protocol.step, course._record_stair_end

        def observed_tof(*args, **kwargs):
            result = tof(*args, **kwargs)
            self.fired.copy_(result[2])
            return result

        def observed_step(*args, **kwargs):
            result = step(*args, **kwargs)
            # The protocol returns the ended ID even after clearing active_id.
            for row in (result.finished_id >= 0).nonzero().flatten().tolist():
                self._end(row, int(result.end_reason[row]), int(result.finished_id[row]))
            for row in (result.response == int(RequestResponse.ACCEPTED)).nonzero().flatten().tolist():
                self._begin(row, int(result.active_id[row]), float(result.height_m[row]))
            return result

        def observed_end(ended, reason):
            # Reset or FSM safety exits can bypass protocol.step. Native end
            # bookkeeping may be called twice; a record is closed only once.
            for row in ended.nonzero().flatten().tolist():
                self._end(row, int(reason[row]))
            return record_end(ended, reason)

        for owner, name, callback in ((course, '_tof_command', observed_tof),
                                      (protocol, 'step', observed_step),
                                      (course, '_record_stair_end', observed_end)):
            self._restore.append((owner, name, owner.__dict__.get(name), name in owner.__dict__))
            setattr(owner, name, callback)

    def _begin(self, row, request_id, height):
        from transformer_rl.formal.tasks.protocol import RequestEnd

        episode = int(self.env._episode_id[row])
        tile = int(self.course.router.tile[row])
        tof = bool(self.fired[row])
        riser = self.course.tof_trigger.last_riser_xy[row].detach().cpu().tolist() if tof else None
        if riser is not None and not all(math.isfinite(value) for value in riser):
            riser = None
        previous = self.previous[row]
        distance = None
        retry_of = None
        if (previous is not None and previous['episode_id'] == episode
                and previous['terrain_tile'] == tile and previous['riser_world_xy_m'] is not None
                and riser is not None):
            distance = math.dist(riser, previous['riser_world_xy_m'])
            if (previous['end_reason'] == RequestEnd.TIMEOUT.name.lower()
                    and distance < self.course.tof_trigger.config.riser_identity_m):
                retry_of = previous['request_id']
        record = {'env_id': row, 'scenario': self.row_names[row], 'episode_id': episode,
                  'terrain_tile': tile, 'request_id': request_id, 'height_m': height,
                  'accepted_at_s': float(self.env._clock()[row]), 'source': 'simulated_tof' if tof else 'external',
                  'riser_world_xy_m': riser, 'previous_riser_distance_m': distance,
                  'retry_of_timed_out_request_id': retry_of, 'end_reason': None, 'ended_at_s': None}
        self.records.append(record)
        self.active[row] = record

    def _end(self, row, reason, request_id=None):
        from transformer_rl.formal.tasks.protocol import RequestEnd

        record = self.active[row]
        if record is None or request_id is not None and record['request_id'] != request_id:
            return
        record['end_reason'] = RequestEnd(reason).name.lower()
        record['ended_at_s'] = float(self.env._clock()[row])
        self.previous[row], self.active[row] = record, None

    def summary(self):
        records = deepcopy(self.records)
        for record in records:
            if record['end_reason'] is None:
                record['end_reason'] = 'active_at_evaluation_end'
        return {'format': 'native-stair-request-boundaries-v1', 'requests': records,
                'same_riser_tolerance_m': self.course.tof_trigger.config.riser_identity_m,
                'scope': 'Raw native protocol boundaries, including invalid/reset intervals. '
                         'A timeout retry is a new simulated-ToF ID in the same episode/tile '
                         'within the native riser identity tolerance. Estimated completion '
                         'is not physical passage; use authored whole-course evidence separately.'}

    def close(self):
        for owner, name, previous, existed in reversed(self._restore):
            if existed:
                setattr(owner, name, previous)
            else:
                delattr(owner, name)
        self._restore.clear()
