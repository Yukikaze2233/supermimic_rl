"""Native regression guard for online student updates.

An imagined objective cannot reveal that the student is exploiting world-model
errors; its own native return can. The guard keeps the complete actor and the
Adam states of its decoder and optional estimator, restores them after
``patience`` consecutive cycles that are more than ``drop_fraction`` below that
best, and halves the actor learning rate on every restoration.
"""
from __future__ import annotations

import copy


class NativeRegressionGuard:
    def __init__(self, drop_fraction: float, patience: int, base_learning_rate: float):
        if not 0 < drop_fraction < 1 or type(patience) is not int or patience < 1 or not base_learning_rate > 0:
            raise ValueError('Guard needs a drop fraction in (0, 1), positive patience and learning rate')
        self.drop_fraction, self.patience, self.base_learning_rate = drop_fraction, patience, base_learning_rate
        self.best_reward, self.best, self.bad_cycles, self.restorations = None, None, 0, 0

    @staticmethod
    def snapshot(actor, optimizer, *, estimator_optimizer=None) -> dict:
        """The actor that is about to collect; call before the cycle's collection."""
        return {'actor': {name: value.detach().cpu().clone() for name, value in actor.state_dict().items()},
                'optimizer': copy.deepcopy(optimizer.state_dict()),
                **({} if estimator_optimizer is None else
                   {'estimator_optimizer': copy.deepcopy(estimator_optimizer.state_dict())})}

    def learning_rate(self) -> float:
        return self.base_learning_rate * .5 ** self.restorations

    def observe(self, reward: float | None, snapshot: dict, actor, optimizer, *, estimator_optimizer=None) -> dict:
        """Judge one collection made by ``snapshot``; restore the best actor if it regressed."""
        restored = legacy_estimator_reset = False
        if reward is not None:
            if self.best_reward is None or reward > self.best_reward:
                self.best_reward, self.best, self.bad_cycles = reward, snapshot, 0
            elif reward < self.best_reward - self.drop_fraction * abs(self.best_reward):
                self.bad_cycles += 1
                if self.bad_cycles >= self.patience:
                    actor.load_state_dict(self.best['actor'])
                    # Adam may reuse loaded moment tensors on the same device.
                    # Subsequent updates must not mutate the saved best state.
                    optimizer.load_state_dict(copy.deepcopy(self.best['optimizer']))
                    if estimator_optimizer is not None:
                        if 'estimator_optimizer' in self.best:
                            estimator_optimizer.load_state_dict(copy.deepcopy(self.best['estimator_optimizer']))
                        else:
                            # Legacy snapshots lack matching estimator moments.
                            # Reset only on restoration, retaining its groups/LR.
                            estimator_optimizer.state.clear()
                            legacy_estimator_reset = True
                    self.restorations += 1
                    self.bad_cycles, restored = 0, True
            else:
                self.bad_cycles = 0
        for group in optimizer.param_groups:
            group['lr'] = self.learning_rate()
        return {'student_reward': reward, 'best_student_reward': self.best_reward, 'bad_cycles': self.bad_cycles,
                'restored': restored, 'restorations': self.restorations, 'actor_learning_rate': self.learning_rate(),
                **({} if estimator_optimizer is None else
                   {'estimator_optimizer_legacy_reset': legacy_estimator_reset})}

    def state_dict(self) -> dict:
        return {'best_reward': self.best_reward, 'best': self.best, 'bad_cycles': self.bad_cycles,
                'restorations': self.restorations}

    def load_state_dict(self, state: dict) -> None:
        self.best_reward, self.best = state['best_reward'], state['best']
        self.bad_cycles, self.restorations = int(state['bad_cycles']), int(state['restorations'])

