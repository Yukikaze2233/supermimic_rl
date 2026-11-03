"""Constraint-stabilization overrides wrap the pinned builder only inside their scope."""
from types import SimpleNamespace

import pytest

from supermimic_rl.course.kamino_solver import KaminoConstraintSettings, kamino_constraint_scope


def builder_module():
    def build_kamino_physics(backend):
        constraints = SimpleNamespace(alpha=.1, beta=.01, gamma=.01, delta=1e-6)
        return SimpleNamespace(solver_cfg=SimpleNamespace(constraints=constraints), backend=backend)
    return SimpleNamespace(build_kamino_physics=build_kamino_physics)


def test_overrides_apply_to_every_build_in_scope_and_restore_the_builder():
    module = builder_module()
    original = module.build_kamino_physics
    settings = KaminoConstraintSettings.from_dict({"joint_alpha": .3, "contact_gamma": .05})
    with kamino_constraint_scope(settings, module):
        first, second = module.build_kamino_physics("a"), module.build_kamino_physics("b")
    for configuration in (first, second):
        constraints = configuration.solver_cfg.constraints
        assert (constraints.alpha, constraints.beta, constraints.gamma) == (.3, .01, .05)
    assert module.build_kamino_physics is original
    assert settings.identity() == {"joint_alpha": .3, "contact_gamma": .05}


def test_empty_settings_leave_builder_and_identity_untouched():
    module = builder_module()
    original = module.build_kamino_physics
    settings = KaminoConstraintSettings.from_dict(None)
    with kamino_constraint_scope(settings, module):
        assert module.build_kamino_physics is original
    assert settings.identity() == {}


def test_scope_restores_builder_after_failure():
    module = builder_module()
    original = module.build_kamino_physics
    with pytest.raises(RuntimeError):
        with kamino_constraint_scope(KaminoConstraintSettings(joint_alpha=.2), module):
            raise RuntimeError("construction failed")
    assert module.build_kamino_physics is original


@pytest.mark.parametrize("value", [{"joint_alpha": 1.5}, {"joint_alpha": float("nan")}, {"alpha": .1}, [0.1]])
def test_invalid_settings_are_rejected(value):
    with pytest.raises((ValueError, TypeError)):
        KaminoConstraintSettings.from_dict(value)
