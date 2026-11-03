"""Long physical-clock parity, including riser rejection and independently reset rows."""
from dataclasses import fields

import pytest
import torch

pytest.importorskip('transformer_rl.formal.lab_env.terrain_support')
from transformer_rl.formal.lab_env.terrain_support import TerrainSupportConfig, TerrainSupportReference
from transformer_rl.formal.simulation.ground import WheelCollisionCentres
from supermimic_rl.world_model_v2.support import SupportState, SupportInputs, advance_support


@pytest.mark.parametrize('profile', ['legacy', 'rough_dash'])
def test_support_clocks_match_native_across_flight_landing_base_contact_and_timeout(profile):
    wheels = WheelCollisionCentres(('left', 'right'), ((0., 0., 0.),) * 2,
                                   (.06, .06), (.025, .025), ((0., 1., 0.),) * 2)
    config = TerrainSupportConfig(profile=profile)
    native = TerrainSupportReference(4, wheels, physics_dt_s=.005, config=config)
    state = SupportState.from_native(native)
    for tick in range(450):
        root = torch.tensor([[0., 0., .43]]).repeat(4, 1)
        pose = torch.zeros(4, 2, 7); pose[..., 6] = 1; pose[..., 2] = .19
        support = torch.zeros(4, 2)
        force = torch.zeros(4, 2)
        base = torch.zeros(4)
        # Row 0 lands after flight; row 1 keeps hitting a vertical riser, with no support.
        if tick % 100 >= 12:
            pose[0, :, 2] = .06; support[0] = 21.; force[0] = 22.
        force[1] = 22.
        # Row 2 times out after sustained flight; row 3 lands on the base.
        if tick > 10:
            base[3] = 10.
        valid, eligible = torch.ones(4, dtype=torch.bool), torch.ones(4, dtype=torch.bool)
        if 250 <= tick < 253:
            valid[2] = False
        if 270 <= tick < 280:
            eligible[0] = False
        if tick == 310:
            native.reset(torch.tensor([0, 2]))
            flat = state.encode().clone(); flat[[0, 2]] = 0
            state = SupportState.decode(flat)
        ground, wheel_ground = torch.zeros(4), torch.zeros(4, 2)
        expected = native.observe(root, pose, support, valid,
            root_ground_world_z_m=ground, wheel_ground_world_z_m=wheel_ground,
            wheel_contact_force_n=force, base_contact_force_n=base, eligible=eligible)
        inputs = SupportInputs(valid, eligible, root[:, 2], pose[..., 2], pose[..., 2] - .06,
                               ground, wheel_ground, support, force, base)
        result = advance_support(state, inputs, config, .005)
        state = result.state
        actual_state = SupportState.from_native(native)
        for field in fields(state):
            torch.testing.assert_close(getattr(state, field.name), getattr(actual_state, field.name), rtol=0, atol=0)
        for name in ('airborne', 'landing', 'just_landed'):
            torch.testing.assert_close(getattr(result, name), getattr(expected, name), rtol=0, atol=0)
        torch.testing.assert_close(SupportState.decode(state.encode()).encode(), state.encode())
        if profile == 'rough_dash':
            assert not state.phase_active[1]  # A riser strike is never flight evidence.
