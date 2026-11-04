"""Native regression guard for online student updates.

An imagined objective cannot reveal that the student is exploiting world-model
errors; its own native return can. The guard judges the median student reward
of the last ``window`` collections against the best such median. A single
collection also reflects population events: after a synchronized timeout most
rows are reborn together and their release interval earns little, which is not
a policy regression. A median over an odd window ignores dips shorter than half
the window, whereas a mean would count one dip in ``window`` judgments.

``restore`` mode keeps the complete actor and the Adam states of its decoder and
optional estimator, restores them after ``patience`` consecutive judgments more
than ``drop_fraction`` below the best, and halves the actor learning rate on
every restoration, down to ``min_rate_fraction`` of the base rate; after
``recover_after`` consecutive judgments within the tolerance one halving is
undone. The window restarts after a restore, so the restored actor is judged
on its own collections. ``report`` mode only reports the same judgment for
review: it never restores and keeps the configured learning rate.
"""
from __future__ import annotations

from collections import deque
import copy
import statistics

GUARD_MODES = ('restore', 'report')


class NativeRegressionGuard:
    def __init__(self, drop_fraction: float, patience: int, base_learning_rate: float, *,
                 window: int = 1, mode: str = 'restore', min_rate_fraction: float = 0., recover_after: int = 0):
        if (not 0 < drop_fraction < 1 or type(patience) is not int or patience < 1 or not base_learning_rate > 0
                or type(window) is not int or window < 1 or mode not in GUARD_MODES
                or not 0 <= min_rate_fraction <= 1 or type(recover_after) is not int or recover_after < 0):
            raise ValueError('Guard needs a drop fraction in (0, 1), positive patience, learning rate and window, '
                             f'a mode in {GUARD_MODES}, a rate floor in [0, 1] and nonnegative recovery')
        self.drop_fraction, self.patience, self.base_learning_rate = drop_fraction, patience, base_learning_rate
        self.window, self.mode = window, mode
        self.min_rate_fraction, self.recover_after = min_rate_fraction, recover_after
        self.best_reward, self.best, self.bad_cycles, self.restorations = None, None, 0, 0
        self.halvings = self.good_judgments = 0   # restorations stay the record; halvings can be undone
        self.recent = deque(maxlen=window)

    @staticmethod
    def snapshot(actor, optimizer, *, estimator_optimizer=None) -> dict:
        """The actor that is about to collect; call before the cycle's collection."""
        return {'actor': {name: value.detach().cpu().clone() for name, value in actor.state_dict().items()},
                'optimizer': copy.deepcopy(optimizer.state_dict()),
                **({} if estimator_optimizer is None else
                   {'estimator_optimizer': copy.deepcopy(estimator_optimizer.state_dict())})}

    @property
    def restores(self) -> bool:
        return self.mode == 'restore'

    def learning_rate(self) -> float:
        if not self.restores:
            return self.base_learning_rate
        return self.base_learning_rate * max(.5 ** self.halvings, self.min_rate_fraction)

    def observe(self, reward: float | None, snapshot: dict | None, actor, optimizer, *,
                estimator_optimizer=None) -> dict:
        """Judge the collection made by ``snapshot`` (None in report mode); restore the best actor if it regressed.

        With a window, the stored best is the actor that collected the last
        cycle of the best-scoring window.
        """
        restored = legacy_actor_reset = legacy_estimator_reset = False
        judged = None
        if reward is not None:
            self.recent.append(float(reward))
        if reward is not None and len(self.recent) == self.window:
            judged = statistics.median(self.recent)
            if self.best_reward is None or judged > self.best_reward:
                self.best_reward, self.best, self.bad_cycles = judged, snapshot, 0
                self._recover()
            elif judged < self.best_reward - self.drop_fraction * abs(self.best_reward):
                self.bad_cycles, self.good_judgments = self.bad_cycles + 1, 0
                if self.restores and self.bad_cycles >= self.patience:
                    actor.load_state_dict(self.best['actor'])
                    if estimator_optimizer is not None and 'estimator_optimizer' not in self.best:
                        # Old joint snapshots lack estimator moments, and their
                        # decoder moments could alias live Adam after a prior
                        # restore. Neither history can be paired reliably now.
                        optimizer.state.clear()
                        estimator_optimizer.state.clear()
                        legacy_actor_reset = legacy_estimator_reset = True
                    else:
                        # Adam may reuse loaded moment tensors on the same device.
                        # Subsequent updates must not mutate the saved best state.
                        optimizer.load_state_dict(copy.deepcopy(self.best['optimizer']))
                        if estimator_optimizer is not None:
                            estimator_optimizer.load_state_dict(copy.deepcopy(self.best['estimator_optimizer']))
                    self.restorations += 1
                    self.halvings += 1
                    self.bad_cycles, restored = 0, True
                    self.recent.clear()
            else:
                self.bad_cycles = 0
                self._recover()
        for group in optimizer.param_groups:
            group['lr'] = self.learning_rate()
        return {'student_reward': reward, 'judged_student_reward': judged, 'best_student_reward': self.best_reward,
                'bad_cycles': self.bad_cycles, 'mode': self.mode, 'window': self.window,
                'restored': restored, 'restorations': self.restorations, 'actor_learning_rate': self.learning_rate(),
                **({} if estimator_optimizer is None else
                   {'actor_optimizer_legacy_reset': legacy_actor_reset,
                    'estimator_optimizer_legacy_reset': legacy_estimator_reset})}

    def _recover(self) -> None:
        self.good_judgments += 1
        if self.recover_after and self.halvings and self.good_judgments >= self.recover_after:
            self.halvings, self.good_judgments = self.halvings - 1, 0

    def state_dict(self) -> dict:
        return {'best_reward': self.best_reward, 'best': self.best, 'bad_cycles': self.bad_cycles,
                'restorations': self.restorations, 'window': self.window, 'mode': self.mode,
                'recent': list(self.recent), 'halvings': self.halvings, 'good_judgments': self.good_judgments}

    def load_state_dict(self, state: dict) -> bool:
        """Restore; returns False when the judgment restarted because window or mode changed.

        A best of another averaging window or mode is not comparable, so the
        judgment starts afresh; past restorations stay counted for the record.
        """
        self.restorations = int(state['restorations'])
        self.halvings = int(state.get('halvings', self.restorations))
        self.good_judgments = int(state.get('good_judgments', 0))
        if (state.get('window', 1), state.get('mode', 'restore')) != (self.window, self.mode):
            return False
        self.best_reward, self.best = state['best_reward'], state['best']
        self.bad_cycles = int(state['bad_cycles'])
        self.recent.extend(state.get('recent', []))
        return True

