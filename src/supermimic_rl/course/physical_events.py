"""Evaluation attribution of native physical certifications across auto-reset."""
from __future__ import annotations


class PhysicalSuccessEvents:
    """Keep native event identities until their complete transition is available.

    This ledger never evaluates a success predicate. Its producers are the
    existing jump/recovery observers immediately after native certification.
    """

    def __init__(self, row_names):
        self.row_names = tuple(row_names)
        self.events = {}
        self.pending = set()

    def record(self, row, episode, *, request=None, prior_invalid=False,
               nonfinite_physical_ticks=0):
        key = (row, episode, request)
        if key in self.events:
            return
        self.events[key] = {
            'env_id': row, 'episode_id': episode, 'request_id': request,
            'scenario': self.row_names[row], 'prior_invalid': bool(prior_invalid),
            'nonfinite_physical_ticks': int(nonfinite_physical_ticks),
            'interval_matches_episode': None, 'boundary_valid': None,
            'validated_event': None}
        self.pending.add(key)

    def observe_transition(self, transition, valid):
        if not self.pending:
            return []
        starts = transition.start.episode_id.cpu().tolist()
        ends = transition.end.episode_id.cpu().tolist()
        events = []
        for key in sorted(self.pending):
            event = self.events[key]
            row, episode, _ = key
            matches = starts[row] == episode
            boundary_valid = matches and ends[row] == episode and bool(valid[row])
            event.update(interval_matches_episode=matches, boundary_valid=boundary_valid,
                         validated_event=boundary_valid and not event['prior_invalid']
                         and not event['nonfinite_physical_ticks'])
            events.append(dict(event))
        # An unmatched event is retained as excluded/unattributed, never carried
        # into a later interval with a different phase or a recycled request ID.
        self.pending.clear()
        return events

    def summary(self):
        return [dict(event) for event in self.events.values()]
