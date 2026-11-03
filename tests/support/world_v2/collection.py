"""A two-row native environment stub for collector tests."""

from types import SimpleNamespace
import torch


class Environment:
    num_envs, device, schema = 2, 'cpu', None

    def __init__(self):
        self.tick = 0
        self.current = self.snapshot()
        self.estimates, self.progress = [], []

    def snapshot(self):
        return SimpleNamespace(actor_observation=torch.full((2, 45), float(self.tick)),
            episode_id=torch.tensor([0, int(self.tick >= 3)]), control_epoch=torch.tensor([int(self.tick >= 2), 0]),
            policy_owned=torch.tensor([self.tick >= 2, True]), estimator_target=torch.full((2, 5), 999.))

    def set_training_progress(self, **record):
        self.progress.append(record)

    def set_policy_estimate(self, frame, raw):
        self.estimates.append(raw.clone())
        torch.testing.assert_close(frame, self.current.actor_observation)

    def step(self, action):
        before = self.current
        self.tick += 1
        self.current = self.snapshot()
        end = SimpleNamespace(episode_id=before.episode_id.clone(), control_epoch=before.control_epoch.clone(),
                              policy_owned=before.policy_owned.clone())
        return SimpleNamespace(start=before, end=end, action=action, reward=torch.ones(2),
            valid=torch.ones(2, dtype=torch.bool), terminated=torch.tensor([False, self.tick == 3]),
            truncated=torch.zeros(2, dtype=torch.bool))
