"""Freeze named raw simulator tensors into a versioned learning schema."""
from __future__ import annotations

import torch

from ..contracts import Field, Snapshot, StateSchema, Transition, _TensorValidation


class PackedEnvironment:
    def __init__(self, adapter, *, source: str, policy_dt_s: float = 0.02):
        self.adapter = adapter
        self.source = source
        self.policy_dt_s = policy_dt_s
        self.schema = None
        self.current = None

    def _pack(self, raw, *, _checks=None) -> Snapshot:
        checks = _TensorValidation() if _checks is None else _checks
        metadata = getattr(raw, "field_metadata", None) or {}
        groups = []
        for group in ("state", "context"):
            groups.append(tuple(Field(name, tuple(value.shape[1:]),
                                      **metadata.get(group, {}).get(name, {}))
                                for name, value in sorted(getattr(raw, group).items())))
        # Substep truth exists only inside a step, so its fields are declared
        # statically by the adapter and frozen together with the snapshot schema.
        substep_fields = tuple(getattr(self.adapter, "substep_fields", ()))
        candidate = StateSchema(*groups, policy_dt_s=self.policy_dt_s, source=self.source,
                                observation_dim=raw.observation.shape[-1], goal_dim=raw.goal.shape[-1],
                                substep_fields=substep_fields,
                                substeps_per_step=(getattr(self.adapter, "substeps_per_step", 0)
                                                   if substep_fields else 0))
        if self.schema is None:
            self.schema = candidate
        elif candidate != self.schema:
            raise ValueError("Native field shapes, units or frames changed after schema freeze")
        context = (self.schema.pack(raw.context, context=True, _checks=checks) if self.schema.context_fields
                   else raw.observation.new_empty((raw.observation.shape[0], 0)))
        result = Snapshot(self.schema.pack(raw.state, _checks=checks), raw.observation.float().clone(),
                          raw.goal.float().clone(), context, raw.episode_id.clone(),
                          raw.control_epoch.clone(), raw.policy_owned.clone(),
                          (None if getattr(raw, "actor_observation", None) is None
                           else raw.actor_observation.float().clone()),
                          (None if getattr(raw, "estimator_target", None) is None
                           else raw.estimator_target.float().clone()),
                          (None if getattr(raw, "estimator_valid", None) is None
                           else raw.estimator_valid.clone()))
        result.validate(self.schema, _checks=checks)
        if _checks is None:
            checks.finish()
        return result

    def reset(self):
        self.current = self._pack(self.adapter.reset())
        self.num_envs = self.current.state.shape[0]
        self.device = self.current.state.device
        return self.current

    def step(self, action):
        raw = self.adapter.step(action)
        checks = _TensorValidation()
        raw_substeps = getattr(raw, "substeps", None)
        if self.schema.substep_fields and raw_substeps is None:
            raise ValueError("The frozen schema requires substep truth for every transition")
        substeps = (None if not self.schema.substep_fields
                    else self.schema.pack_substeps(raw_substeps, _checks=checks))
        result = Transition(self._pack(raw.start, _checks=checks), raw.action.float().clone(),
                            self._pack(raw.end, _checks=checks),
                            raw.reward.float().clone(), raw.terminated.clone(), raw.truncated.clone(),
                            raw.valid.clone(), substeps)
        result.validate(self.schema, _checks=checks)
        current = self._pack(self.adapter.current, _checks=checks)
        checks.finish()
        self.current = current
        return result

    def close(self):
        self.adapter.close()

    def set_policy_estimate(self, frame, raw_state):
        hook = getattr(self.adapter, "set_policy_estimate", None)
        if callable(hook):
            hook(frame, raw_state)

    def set_training_progress(self, real_transitions, actor_updates):
        hook = getattr(self.adapter, "set_training_progress", None)
        if callable(hook):
            hook(real_transitions=real_transitions, actor_updates=actor_updates)

    def training_state_dict(self):
        hook = getattr(self.adapter, "training_state_dict", None)
        return hook() if callable(hook) else None

    def load_training_state_dict(self, state):
        hook = getattr(self.adapter, "load_training_state_dict", None)
        if state is not None:
            if not callable(hook):
                raise ValueError("Environment cannot restore its saved training state")
            hook(state)
            if self.adapter.current is not None:
                self.current = self._pack(self.adapter.current)
