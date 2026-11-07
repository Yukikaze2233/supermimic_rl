"""Ordinary stair support transfer keeps tracking without prescribing a jump."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('transformer_rl.formal.tasks.reward')
from transformer_rl.formal.control.observation import CommandMode, MeasuredState, NativePhysicalSample
from transformer_rl.formal.lab_env.formal_reward import FormalRewardConfig, LabFormalReward
from transformer_rl.formal.tasks.reward import RewardPhase, TerrainRewardReference
from transformer_rl.formal.tasks.gimbal_course import GimbalRewardReference
from supermimic_rl.course.reward import install_shared_reward, install_stair_support_reference
from supermimic_rl.reward_kernel import RewardExtensions, RewardHistory, RewardParameters, physical_reward_densities
from supermimic_rl.world_model_v2.imagined_reward import OrdinaryFlatReward
from test_reward_kernel import begin, sample, setup


def inputs(*, extension=None, mode=CommandMode.STAIR, allowed=(True, True), airborne=False):
    composer, reference = setup('air_landing')
    command = torch.tensor([[.5, 0., .5, .4]]).repeat(2, 1)
    extension = torch.tensor([.2, .45]) if extension is None else extension
    phase = (RewardPhase.FLIGHT if mode == CommandMode.JUMP else
             RewardPhase.RECOVERY if mode == CommandMode.RECOVERY else RewardPhase.SUPPORT)
    reference = replace(reference, admitted_body_command=command, mode=torch.full((2,), int(mode)),
        phase=torch.full((2,), int(phase)), intentional_base_contact=torch.tensor(allowed),
        terrain=TerrainRewardReference(torch.ones(2, dtype=torch.bool), extension,
                                      torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)),
        rough_dash=replace(reference.rough_dash, active=torch.full((2,), airborne),
                           body_extension_m=extension))
    begin(composer, reference, torch.zeros(2, 6))
    physics = replace(sample(composer, 1), body_normal_load_n=torch.tensor([[4., 50., 50., 0., 0.]]).repeat(2, 1),
                      velocity_origin_body_m_s=torch.zeros(2, 3), velocity_origin_world_m_s=torch.zeros(2, 3))
    history = RewardHistory(composer._window['dq'], torch.zeros(2), torch.zeros(2), torch.zeros(2))
    return physics, reference, history, RewardParameters.from_composer(composer)


def score(values, support, enabled=True):
    physics, reference, history, parameters = values
    parameters = replace(parameters, extensions=RewardExtensions(stair_support_transfer=enabled))
    return physical_reward_densities(physics, reference, history, parameters, stair_base_net_upward_n=support)


def test_supported_stair_releases_retraction_but_retains_above_target_cost_and_raw_error():
    values = inputs()
    before = score(values, None, False)
    after = score(values, torch.tensor([4., 4.]))
    assert before.gates['velocity_height_gate'][0] == 0
    torch.testing.assert_close(after.gates['velocity_height_gate'], torch.ones(2))
    assert after.positive['height'][0] == 1 and after.cost['height'][0] == 0
    torch.testing.assert_close(after.positive['height'][1], before.positive['height'][1])
    torch.testing.assert_close(after.cost['height'][1], before.cost['height'][1])
    torch.testing.assert_close(after.errors['terrain_height_error_m'], torch.tensor([-.2, .05]))
    torch.testing.assert_close(after.errors['stair_objective_height_error_m'], torch.tensor([0., .05]))
    assert after.cost['harmful_contact'].tolist() == [0., 0.]
    assert not after.invalid.any()


def test_riser_contact_keeps_height_shaping_while_forward_signal_survives():
    values = inputs()
    before = score(values, None, False)
    after = score(values, torch.tensor([0., 3.]))
    assert not after.gates['stair_base_supported'].any()
    for key in ('height', 'harmful_contact'):
        torch.testing.assert_close(after.cost[key], before.cost[key])
    torch.testing.assert_close(after.positive['height'], before.positive['height'])
    assert after.positive['forward'][0] > 0 and after.cost['forward'][0] > 0
    assert before.positive['forward'][0] == 0


@pytest.mark.parametrize('airborne', [False, True])
def test_other_body_contacts_keep_their_full_cost_and_existing_rough_multiplier(airborne):
    values = list(inputs(airborne=airborne))
    loads = values[0].body_normal_load_n.clone()
    loads[1, 3] = 4.
    values[0] = replace(values[0], body_normal_load_n=loads)
    result = score(values, torch.tensor([5., 5.]))
    assert result.cost['harmful_contact'].tolist() == [0., 50. if airborne else 2.]


@pytest.mark.parametrize('mode', [CommandMode.NORMAL, CommandMode.SLOPE, CommandMode.JUMP, CommandMode.RECOVERY])
def test_non_stair_tasks_have_exactly_the_same_components(mode):
    values = inputs(mode=mode)
    before, after = score(values, None, False), score(values, torch.tensor([100., 100.]))
    for key in before.components:
        torch.testing.assert_close(after.components[key], before.components[key], atol=0, rtol=0)
    assert not after.gates['stair_support_transfer'].any()


def test_absent_permission_preserves_all_components():
    values = inputs(allowed=(False, False))
    before, after = score(values, None, False), score(values, torch.tensor([100., 100.]))
    for key in before.components:
        torch.testing.assert_close(after.components[key], before.components[key], atol=0, rtol=0)


def test_special_gimbal_motion_keeps_its_own_tracking_objective():
    values = list(inputs())
    values[1] = replace(values[1], gimbal=GimbalRewardReference(
        torch.ones(2, dtype=torch.bool), torch.zeros(2), torch.ones(2), torch.zeros(2)))
    before, after = score(values, None, False), score(values, torch.tensor([100., 100.]))
    for key in before.components:
        torch.testing.assert_close(after.components[key], before.components[key], atol=0, rtol=0)
    assert not after.gates['stair_support_transfer'].any()


def test_height_and_velocity_gradients_match_directional_difference():
    extension = torch.tensor([.2, .45], dtype=torch.float64, requires_grad=True)
    velocity = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)

    def objective(length, speed):
        values = list(inputs(extension=length))
        values[0] = replace(values[0], velocity_origin_body_m_s=speed, velocity_origin_world_m_s=speed)
        result = score(values, torch.tensor([4., 4.]))
        return (result.positive['height'] - result.cost['height']
                + result.positive['forward'] - result.cost['forward']).sum()

    gradients = torch.autograd.grad(objective(extension, velocity), (extension, velocity))
    assert gradients[0][0] == 0 and gradients[0][1] < 0
    assert (gradients[1][:, 0] > 0).all()
    directions = (torch.tensor([.1, -.2], dtype=torch.float64), torch.full((2, 3), .1, dtype=torch.float64))
    epsilon = 1e-6
    numeric = (objective(extension + epsilon * directions[0], velocity + epsilon * directions[1])
               - objective(extension - epsilon * directions[0], velocity - epsilon * directions[1])) / (2 * epsilon)
    analytic = sum((g * d).sum() for g, d in zip(gradients, directions))
    torch.testing.assert_close(analytic, numeric, atol=1e-7, rtol=1e-6)


def test_contract_does_not_substitute_scalar_norms_or_invalidate_unrelated_rows():
    with pytest.raises(ValueError, match='same-tick base net upward'):
        score(inputs(), None)
    result = score(inputs(allowed=(False, True)), torch.tensor([float('nan'), 4.]))
    assert not result.invalid.any() and torch.isfinite(result.total).all()
    result = score(inputs(), torch.tensor([float('nan'), 4.]))
    assert result.invalid.tolist() == [True, False] and torch.isfinite(result.total).all()
    with pytest.raises(ValueError, match='boolean'):
        RewardExtensions(stair_support_transfer=1)
    receipt = install_shared_reward(setup('ordinary')[0], extensions=RewardExtensions(recovery_quiet=1.))
    assert 'stair_support_transfer' not in receipt['extensions']


def test_actual_lab_observer_uses_this_ticks_direction_and_clears_it_afterwards():
    binding = setup('ordinary')[0].binding
    reward = LabFormalReward(binding, 2, FormalRewardConfig(21.44, normal_load_source='contact_force_norm'))
    ids, owned = torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.bool)
    reward.reset(owned, torch.zeros(2, dtype=torch.float64), ids, ids)
    install_shared_reward(reward.composer, extensions=RewardExtensions(stair_support_transfer=True))
    native = SimpleNamespace(_formal_reward=reward, _base_body_index=0)
    receipt = install_stair_support_reference(native)
    assert receipt['measure'] == 'positive_world_z_component_of_base_net_contact_force'
    command = torch.tensor([[.5, 0., 0., .4]]).repeat(2, 1)
    mode = torch.full((2,), int(CommandMode.STAIR))
    reward.begin(torch.zeros(2, 6), torch.zeros(2, 6), now_s=torch.zeros(2, dtype=torch.float64),
                 episode_id=ids, control_epoch=ids, command=command, mode=mode)
    for tick, base_z in enumerate([0., 10., -10., 10.], start=1):
        force = torch.zeros(2, 5, 3)
        force[:, 0, 0], force[:, 0, 2], force[:, 1:3, 2] = 100., base_z, 50.
        time = torch.full((2,), tick * .005, dtype=torch.float64)
        position = torch.tensor([[0., 0., .2]]).repeat(2, 1)
        physical = NativePhysicalSample(torch.zeros(2, 6), torch.zeros(2, 6), position,
            torch.tensor([[0., 0., 0., 1.]]).repeat(2, 1), position, torch.zeros(2, 3),
            torch.zeros(2, 3), force, torch.zeros(2), torch.zeros(2, 2), torch.zeros(2, 3),
            torch.ones(2, 2), time)
        state = MeasuredState(torch.zeros(2, 6), torch.zeros(2, 6), torch.zeros(2, 6), torch.zeros(2, 6),
            torch.zeros(2, 3), torch.zeros(2, 3), torch.tensor([[0., 0., -1.]]).repeat(2, 1),
            torch.full((2,), .2), torch.full((2, 2), 50.), torch.ones(2, 2), torch.zeros(2, 2), time)
        reward.observe(physical, state, effort_nm=torch.zeros(2, 6), episode_id=ids, control_epoch=ids,
            command=command, mode=mode, body_contact_allowed=torch.tensor([True, False]),
            wheel_position_base_m=torch.tensor([[[0., .2, -.2], [0., -.2, -.2]]]).repeat(2, 1, 1),
            terrain_reference=TerrainRewardReference(owned, torch.full((2,), .2), ~owned, ~owned))
        assert reward.composer.supermimic_stair_base_net_upward_n is None
    window = reward.composer._window
    torch.testing.assert_close(window['gates']['stair_base_supported'], torch.tensor([.01, 0.], dtype=torch.float64))
    # The negative/zero vertical-force ticks still pay the original height
    # cost; horizontal contact alone never opens the supported-height path.
    torch.testing.assert_close(window['components']['cost.height'], torch.tensor([-.04, -.08], dtype=torch.float64))


def test_world_model_refuses_new_flag_without_a_direction_label_producer():
    source = {'asset_manifest_sha256': 'test', 'reward_config': {'physics_dt_s': .005},
              'reward_kernel': {'extensions': {'stair_support_transfer': True}}}
    geometry = SimpleNamespace(linkage=SimpleNamespace(provenance={'asset_manifest_sha256': 'test'}))
    with pytest.raises(ValueError, match='base net force direction is not recorded or predicted'):
        OrdinaryFlatReward(source, geometry, None, .02)
