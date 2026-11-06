"""Retain independent native skill data alongside the live dynamics replay."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import torch

from ..contracts import StateSchema
from .fitting import WorldModelFit
from .records import MODEL_CONTRACT
from .replay import CompactReplay
from .validation import validate


@dataclass(frozen=True)
class WorldDatasetConfig:
    name: str
    path: str
    sha256: str
    fraction: float
    train_env_ids: tuple[int, ...]
    validation_env_ids: tuple[int, ...]

    def __post_init__(self):
        for key in ('train_env_ids', 'validation_env_ids'):
            ids = tuple(getattr(self, key))
            if not ids or any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
                raise ValueError('Auxiliary world data needs distinct nonnegative environment IDs')
            object.__setattr__(self, key, ids)
        if set(self.train_env_ids) & set(self.validation_env_ids):
            raise ValueError('Auxiliary world training and validation rows overlap')
        if (not self.name or not self.path or len(self.sha256) != 64
                or any(c not in '0123456789abcdef' for c in self.sha256)):
            raise ValueError('Auxiliary world data needs a name, path and SHA256')
        if not math.isfinite(self.fraction) or not 0 < self.fraction < 1:
            raise ValueError('Auxiliary world sampling fraction must be between zero and one')


class AuxiliaryWorldData:
    """Mix complete motion windows; never join streams or expose held-out rows.

    The live sampler still draws its usual full batch. A separate CPU RNG
    selects which rows to replace, preserving the live Actor/start sampler's
    random stream. Auxiliary files remain immutable and outside the online
    recovery ring; checkpoints retain their pins, splits and sampling state.
    """

    def __init__(self, configs, schema, live_replay, horizon, *, seed):
        self.generator = torch.Generator().manual_seed(seed)
        self.validation_generator = torch.Generator().manual_seed(seed + 1)
        self.sources = []
        self.sampled = {}
        self.fields = WorldModelFit.RECORD_FIELDS + tuple(
            name for name in WorldModelFit.OPTIONAL_RECORD_FIELDS if name in live_replay.data)
        for config in configs:
            path = Path(config.path)
            with path.open('rb') as stream:
                checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
            if checksum != config.sha256:
                raise ValueError('Auxiliary world dataset SHA256 differs: ' + config.name)
            payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            if (payload.get('format') != 'supermimic-world-v2-dataset-v1'
                    or payload.get('model_contract') != MODEL_CONTRACT
                    or payload.get('schema_fingerprint') != schema.fingerprint
                    or StateSchema.from_dict(payload['schema']).fingerprint != schema.fingerprint):
                raise ValueError('Auxiliary world dataset contract differs: ' + config.name)
            replay = CompactReplay.from_state_dict(payload['replay'], 'cpu', read_only=True)
            if max((*config.train_env_ids, *config.validation_env_ids)) >= replay.num_envs:
                raise ValueError('Auxiliary world environment ID exceeds the dataset')
            for name in self.fields:
                expected = live_replay.data[name]
                actual = replay.data.get(name)
                if (actual is None or actual.shape[2:] != expected.shape[2:]
                        or actual.dtype != expected.dtype):
                    raise ValueError('Auxiliary world field differs: ' + name)
            candidates = replay.window_candidates(horizon, config.train_env_ids)
            if not len(candidates) or not len(replay.window_candidates(1, config.validation_env_ids)):
                raise ValueError('Auxiliary world dataset has no eligible training/validation windows')
            self.sources.append((config, replay))
            self.sampled[config.name] = 0

    def mix(self, windows):
        batch, horizon = windows['action'].shape[:2]
        choice = torch.rand(batch, generator=self.generator)
        lower = 0.
        for config, replay in self.sources:
            upper = lower + config.fraction
            rows = ((choice >= lower) & (choice < upper)).nonzero().flatten()
            lower = upper
            if not len(rows):
                continue
            sample = replay.sample_windows(len(rows), horizon, generator=self.generator,
                env_ids=config.train_env_ids, field_names=self.fields)
            device_rows = rows.to(windows['action'].device)
            for name, value in sample.items():
                windows[name].index_copy_(0, device_rows, value.to(windows[name].device))
            self.sampled[config.name] += len(rows)
        return windows

    def state_dict(self):
        return {'sample_rng': self.generator.get_state(),
                'validation_rng': self.validation_generator.get_state(),
                'sampled_windows': self.sampled.copy()}

    def load_state_dict(self, state):
        self.generator.set_state(state['sample_rng'])
        self.validation_generator.set_state(state['validation_rng'])
        for name in self.sampled:
            self.sampled[name] = state['sampled_windows'].get(name, 0)

    @torch.no_grad()
    def validate(self, learner, config, device):
        report = {}
        for source, replay in self.sources:
            try:
                prediction = validate(learner.fit, replay, source.validation_env_ids,
                    config.validation_horizon, config.validation_batch_size,
                    self.validation_generator, device,
                    contact_threshold_n=learner.imagination.termination.limits.forbidden_body_load_n,
                    termination=learner.imagination.termination,
                    contact_floor=learner.imagination.contact_floor)
            except ValueError as error:
                # A diagnostic has its own error record; it must not stop an
                # otherwise valid online optimizer/collector cycle.
                prediction = {'error': str(error)}
            report[source.name] = {'sha256': source.sha256,
                'validation_env_ids': list(source.validation_env_ids),
                'sampled_training_windows': self.sampled[source.name], 'prediction': prediction}
        return report
