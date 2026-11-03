"""Kamino constraint-stabilization settings layered over the pinned physics builder.

The pinned source builds Kamino with IsaacLab's ``KaminoConstraintsCfg``
defaults (bilateral alpha 0.1, limit beta 0.01, contact gamma 0.01). Its
runtime JSON owns iteration counts and contact capacity; these Baumgarte
factors have no runtime field, so they are an explicit extension here. The
override applies to every physics build inside its scope and becomes part of
the backend identity, so checkpoints stay bound to the solver they saw.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class KaminoConstraintSettings:
    """``None`` keeps the IsaacLab default for that factor."""

    joint_alpha: float | None = None
    limit_beta: float | None = None
    contact_gamma: float | None = None

    def __post_init__(self):
        for name, value in asdict(self).items():
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)
                                      or not 0 <= value <= 1):
                raise ValueError(f"Kamino {name} must be None or within [0, 1]")

    @classmethod
    def from_dict(cls, value) -> KaminoConstraintSettings:
        if value is None:
            return cls()
        if not isinstance(value, dict) or not set(value) <= {"joint_alpha", "limit_beta", "contact_gamma"}:
            raise ValueError("kamino_constraints accepts joint_alpha, limit_beta and contact_gamma")
        return cls(**value)

    def identity(self) -> dict:
        """Only explicit overrides; an empty identity keeps legacy schema sources unchanged."""
        return {name: float(value) for name, value in asdict(self).items() if value is not None}

    def apply(self, physics_configuration) -> None:
        constraints = physics_configuration.solver_cfg.constraints
        for field, attribute in (("joint_alpha", "alpha"), ("limit_beta", "beta"), ("contact_gamma", "gamma")):
            value = getattr(self, field)
            if value is not None:
                setattr(constraints, attribute, float(value))


@contextmanager
def kamino_constraint_scope(settings: KaminoConstraintSettings, builder_module):
    """Wrap ``builder_module.build_kamino_physics`` for builds made inside this scope."""
    if not settings.identity():
        yield
        return
    original = builder_module.build_kamino_physics

    def build_with_constraints(backend):
        configuration = original(backend)
        settings.apply(configuration)
        return configuration

    builder_module.build_kamino_physics = build_with_constraints
    try:
        yield
    finally:
        builder_module.build_kamino_physics = original
