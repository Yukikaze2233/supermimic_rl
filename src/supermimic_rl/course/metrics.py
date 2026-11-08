"""Native skill counters shared by PPO and world-model online training."""

import torch

from .jump_assist import RISE_BIN_EDGES, RISE_BINS


class JumpOutcomeMetrics:
    """Count unassisted request outcomes without touching forces, actions or RNG.

    World-model collection does not install JumpAssist. Observe the native
    success decision at every physics tick, before a reset can erase it, and
    retain the requested height until that same request ends. Metric names
    match the existing PPO height ladder; no assist controller is installed.
    """

    def __init__(self, tasks):
        self.tasks = tasks
        self.request = torch.full_like(tasks.physical_id, -1, dtype=torch.long)
        self.rise = torch.zeros_like(tasks.active_rise)
        self.edges = self.rise.new_tensor(RISE_BIN_EDGES)
        self.counts = torch.zeros(2, len(RISE_BINS), dtype=torch.long, device=self.rise.device)

    def _count(self, outcome, mask):
        # Match JumpAssist's upper-inclusive curriculum height bins.
        bins = torch.bucketize(self.rise, self.edges, right=False)
        self.counts[outcome].scatter_add_(0, bins, mask.long())

    def _track(self):
        current = self.tasks.physical_id.long()
        changed = current != self.request
        self._count(0, changed & (self.request >= 0))
        self.rise = torch.where(changed & (current >= 0), self.tasks.active_rise, self.rise)
        self.request = current.clone()

    def install(self):
        original = self.tasks.observe_physics

        def observe(*args, **kwargs):
            self._track()
            before = self.request.clone()
            result = original(*args, **kwargs)
            success = ((before >= 0) & (self.tasks.physical_id < 0)
                       & (self.tasks.success_id == before))
            self._count(1, success)
            self._track()
            return result

        self.tasks.observe_physics = observe

    def metrics(self):
        values = self.counts.tolist()
        counters = {f'course/jump_assist/unassisted_{kind}_rise_{name}': value
                    for kind, row in zip(('ended', 'success'), values)
                    for name, value in zip(RISE_BINS, row)}
        counters.update({f'course/jump_assist/unassisted_{kind}': sum(row)
                         for kind, row in zip(('ended', 'success'), values)})
        return counters, {'course/jump_assist/level': 0.}


def install_jump_outcome_metrics(native):
    """Reuse existing outcome accounting, or add a read-only native observer."""
    if getattr(native, 'supermimic_jump_assist', None) is not None:
        return
    if getattr(native, 'supermimic_jump_outcomes', None) is None:
        monitor = JumpOutcomeMetrics(native._terrain_course.tasks)
        monitor.install()
        native.supermimic_jump_outcomes = monitor


def course_progress(native, curriculum):
    """Read cumulative outcome counters and current difficulty gauges."""
    metrics = native._terrain_course.metrics()
    counters = {f"course/jump/{name}": value
                for name, value in metrics["physical_request_outcomes"].items()}
    if 'request_responses' in metrics:
        from transformer_rl.formal.tasks.protocol import RequestResponse

        # Already materialized by native.metrics(); these count protocol
        # responses, not independently successful physical jumps.
        for field, prefix in (('request_responses', 'request_response'),
                              ('external_request_responses', 'external_request_response')):
            if field in metrics:
                counters.update({f'course/jump/{prefix}/{response.name.lower()}': metrics[field][int(response)]
                                 for response in RequestResponse})
    counters.update({f'course/jump/request_end/{name}': value
                     for name, value in metrics.get('request_end_counts', {}).items()})
    gate = getattr(native, 'supermimic_jump_trigger', None)
    if gate is not None:
        # One cumulative scalar at the existing progress/log boundary. A row
        # waiting for several ticks contributes repeatedly; it is not a request.
        counters['course/jump/gate_deferred_row_ticks'] = int(gate.deferred)
    counters.update({f"course/stair/{name}": value for name, value in metrics["stair_closed_loop"].items()})
    authored = metrics["authored_stair_completion"]
    for name in ("ended", "ended_complete"):
        counters[f"course/stair/authored_{name}"] = sum(authored[name + "_by_tile"])
    # Already-materialized native counts: reset placements are exposure, never
    # successful traversals. Older routers do not publish these named groups.
    coverage = metrics.get("stair_coverage", {})
    for source, label in (("authored_reset_counts", "authored_birth_placements"),
                          ("foundation_rehearsal_reset_counts", "foundation_rehearsal_birth_placements")):
        counters.update({f"course/terrain/{label}/{asset}": count
                         for asset, count in coverage.get(source, {}).items()})
    counters.update({f"course/recovery/{name}": value
                     for name, value in metrics["recovery"]["counts"].items()})
    recovery_samples = getattr(curriculum, "recovery_birth_samples", None)
    if callable(recovery_samples):
        # The sampler already counted actual reset draws on the host. Bridge
        # labels must not be merged into their reused native 90-degree IDs.
        counters.update({f"course/recovery/birth_samples/{name}": count
                         for name, count in recovery_samples().items()})
    counters.update({f"course/births/{name}": value
                     for name, value in metrics["episode_birth_family_counts"].items()})
    levels = metrics["level_histogram"]
    rows = sum(levels)
    gauges = {"course/terrain/level_fraction": (
        sum(level * count for level, count in enumerate(levels)) / (rows * max(len(levels) - 1, 1))
        if rows else 0.)}
    gauges["course/jump/height_cap_m"] = float(curriculum.command_context().domain.jump_m[1])
    level = getattr(curriculum, "jump_height_level", lambda: None)()
    if level is not None:
        gauges["course/jump/height_level"] = float(level)
    monitor = (getattr(native, "supermimic_jump_assist", None)
               or getattr(native, "supermimic_jump_outcomes", None))
    if monitor is not None:
        assist_counters, assist_gauges = monitor.metrics()
        counters.update(assist_counters)
        gauges.update(assist_gauges)
    return counters, gauges
