"""The standalone probe reads pre-reset physical truth and leaves control intact."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


spec = importlib.util.spec_from_file_location(
    'diagnose_native_posture', Path(__file__).parents[1] / 'scripts/diagnose_native_posture.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Controller:
    def effort(self, targets, q, dq, *_):
        return SimpleNamespace(requested_effort=q + 2., capped_effort=q + 1.,
                               saturated=torch.ones_like(q, dtype=torch.bool))


class Native:
    num_envs, device = 2, 'cpu'

    def __init__(self):
        self._binding = SimpleNamespace(
            native_joint_names=('L_joint1', 'LL_joint1', 'R_joint1', 'RR_joint1',
                                'L_joint3', 'R_joint3', 'L_joint2', 'R_joint2'),
            native_body_names=('base_link', 'L_link3', 'R_link3', 'L_link2'))
        self._active_indices = torch.arange(6)
        self._control = SimpleNamespace(controller=Controller())
        self._formal_reward = SimpleNamespace(
            _passive_knee_indices=[6, 7], _knee_inner_at_lower=torch.tensor([1., 1.]),
            _knee_inner_direction=torch.tensor([1., -1.]), _knee_native_lower=torch.zeros(2),
            config=SimpleNamespace(normal_load_source='native_contact_patch_normal_forces'))
        self._command = torch.tensor([[0., 0., 0., .32], [0., 0., 0., .32]])
        self._issued_action = torch.zeros(2, 6)
        self._last_effort = torch.full((2, 6), 5.)
        self._last_native_effort = torch.full((2, 6), 4.)
        self._last_sample = SimpleNamespace(
            joint_position_rad=torch.tensor([[-.2, .4, -.3, .2, 0., 0., .1, -.2]]).repeat(2, 1),
            terrain_contact_force_world_n=torch.zeros(2, 4, 3),
            body_normal_load_n=torch.tensor([[0., 100., 100., 7.], [0., 100., 100., 0.]]))
        self._last_state = SimpleNamespace(
            gravity_body=torch.tensor([[0., 0., -1.]]).repeat(2, 1),
            velocity_origin_body_m_s=torch.zeros(2, 3), angular_velocity_body_rad_s=torch.zeros(2, 3),
            height_above_ground_m=torch.full((2,), .25))
        self.step_count = self.ticks = 0

    def _observe_substep(self):
        self.ticks += 1

    def step(self, action):
        self._issued_action.copy_(action)
        self._command[:, 3] = .32
        self._last_state.height_above_ground_m.fill_(.25)
        self._last_sample.joint_position_rad[:] = torch.tensor([-.2, .4, -.3, .2, 0., 0., .1, -.2])
        for tick in range(4):
            target = SimpleNamespace(admitted_leg_position_rad=torch.full((2, 4), 1. + tick),
                                     wheel_velocity_rad_s=torch.full((2, 2), 10. + tick),
                                     pair_projected=torch.tensor([[True, False], [False, False]]))
            self._control.controller.effort(target, torch.zeros(2, 6), torch.ones(2, 6), 0.)
            self._observe_substep()
        self._formal_result = SimpleNamespace(
            fully_policy_owned=torch.tensor([True, False]),
            component_densities={'positive.height': torch.tensor([.1, 0.])},
            event_rewards={'failure': torch.tensor([-4. if self.step_count == 1 else 0., 0.])},
            gate_fractions={'velocity_height_gate': torch.tensor([.6, 0.])})
        terminated = torch.tensor([self.step_count == 1, False])
        self.step_count += 1
        # Auto-reset mutates live state/command after the physical interval.
        self._command.fill_(99.)
        self._last_sample.joint_position_rad.fill_(99.)
        self._last_state.height_above_ground_m.fill_(99.)
        return (None, torch.tensor([.2, 0.]), terminated, torch.zeros(2, dtype=torch.bool),
                {'interval_observation_valid': torch.ones(2, dtype=torch.bool),
                 'numerical_invalid': torch.zeros(2, dtype=torch.bool)})


def test_probe_preserves_control_and_records_last_tick_before_reset():
    native = Native()
    original_step, original_observe, original_effort = native.step, native._observe_substep, native._control.controller.effort
    probe = module.NativePostureProbe(native, steps=3, split_step=1, tail_steps=1)
    raw = torch.tensor([[4., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.]])
    with probe:
        for _ in range(3):
            native._command.copy_(torch.tensor([[0., 0., 0., .32], [0., 0., 0., .32]]))
            probe.requested_action(raw)
            result = native.step(raw.clamp(-3., 3.))
            torch.testing.assert_close(result[1], torch.tensor([.2, 0.]))
    assert native.ticks == 12
    assert native.step == original_step and native._observe_substep == original_observe
    assert native._control.controller.effort == original_effort
    report = probe.summary()
    all_rows = report['segments']['all']
    values = all_rows['tracking_diagnostics']
    assert all_rows['valid_tracking_samples'] == 3
    assert all_rows['terminations'] == 1
    assert values['height_m']['mean'] == .25
    assert abs(values['phi_left_rad']['mean'] - .6) < 1e-6
    assert abs(values['phi_right_rad']['mean'] - .5) < 1e-6
    assert values['target_q_rad/L_joint1']['mean'] == 4.
    assert values['target_wheel_rad_s/L_joint3']['mean'] == 13.
    assert values['issued_action/L_joint1']['mean'] == 3.
    assert values['policy_output/L_joint1']['mean'] == 4.
    assert values['action_clipped/L_joint1']['mean'] == 1.
    assert values['pair_projected/left']['mean'] == 1.
    assert values['body_load_n/L_link2']['mean'] == 7.
    assert values['body_contact_over_3n/L_link2']['mean'] == 1.
    assert values['body_resultant_n/L_link2']['mean'] == 0.
    assert abs(values['passive_inner_rad/L_joint2']['mean'] - 1.1) < 1e-6
    assert abs(values['passive_inner_rad/R_joint2']['mean'] - 1.2) < 1e-6
    assert report['segments']['early']['valid_tracking_samples'] == 1
    assert report['segments']['late']['valid_tracking_samples'] == 2
    assert report['segments']['tail']['valid_tracking_samples'] == 1


def test_probe_restores_hooks_after_error():
    native = Native()
    original = native.step
    with pytest.raises(RuntimeError, match='Record the policy output'):
        with module.NativePostureProbe(native, steps=3, split_step=1, tail_steps=1):
            native.step(torch.zeros(2, 6))
    assert native.step == original
