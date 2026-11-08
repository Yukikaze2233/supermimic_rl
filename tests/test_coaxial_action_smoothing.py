"""Common-winding action costs preserve paired motion and native lifecycle."""
from dataclasses import replace
import math

import pytest
import torch

pytest.importorskip('transformer_rl.formal.tasks.reward')
from transformer_rl.formal.control.actuation import OutputFeedbackController
from transformer_rl.formal.control.ownership import ControlSource
from transformer_rl.formal.tasks import reward as native_reward
from transformer_rl.formal.tasks.reward import RewardPhase, RewardReference
from supermimic_rl.course.action_contract import _bounded_validator
from supermimic_rl.course.reward import install_shared_reward
from supermimic_rl.reward_kernel import RewardExtensions, RewardHistory
from supermimic_rl.world_model_v2.reward_assembly import RewardTick
from support.world_v2.reward_assembly import setup


def history(action, previous, previous2=None, *, count=2, enabled=True):
    previous2 = previous if previous2 is None else previous2
    return RewardHistory.from_actions(torch.zeros_like(action), action, previous, previous2,
        torch.full(action.shape[:1], count, device=action.device), coaxial_action_smoothing=enabled)


def test_disabled_extension_preserves_original_receipt_and_exact_costs():
    assert 'coaxial_action_smoothing' not in RewardExtensions().receipt()
    assert not RewardExtensions().active
    with pytest.raises(ValueError, match='boolean'):
        RewardExtensions(coaxial_action_smoothing=1)
    generator = torch.Generator().manual_seed(763)
    a, p, p2 = [torch.randn(11, 6, generator=generator) for _ in range(3)]
    actual = history(a, p, p2, enabled=False)
    first, second = (a-p).square(), (a-2*p+p2).square()
    assert torch.equal(actual.action_cost, .0025*first[:,:4].sum(-1)+.01*first[:,4:].sum(-1))
    assert torch.equal(actual.action2_cost, .0125*second[:,:4].sum(-1)+.01*second[:,4:].sum(-1))


@pytest.mark.parametrize('pair', [slice(0, 2), slice(2, 4), slice(0, 4)])
@pytest.mark.parametrize('epsilon', [0., .04])
def test_equivalent_or_near_cut_targets_match_native_common_winding(pair, epsilon):
    p = torch.zeros(1, 6)
    p[:, pair] = 4*math.pi - epsilon
    a = p.clone()
    a[:, pair] = -4*math.pi + epsilon
    controller = OutputFeedbackController((0.,)*6, 1, pair_domain=None)
    controller.action_bounds[:4] = 4*math.pi
    feedback = torch.zeros(1,6)
    feedback[:,pair] = math.pi
    clock = torch.zeros(1,dtype=torch.float64)
    old = controller.decode(p, feedback, clock)
    new = controller.decode(a, feedback, clock)
    delta = new.admitted_leg_position_rad - old.admitted_leg_position_rad
    # Neither pair's relative target is changed by a common winding.
    torch.testing.assert_close(old.admitted_leg_position_rad.unflatten(-1,(2,2)).diff(dim=-1),
                               new.admitted_leg_position_rad.unflatten(-1,(2,2)).diff(dim=-1), atol=1e-6,rtol=0)
    actual, raw = history(a,p), history(a,p,enabled=False)
    torch.testing.assert_close(actual.action_cost, .0025*(delta/.25).square().sum(-1),atol=1e-8,rtol=1e-4)
    torch.testing.assert_close(actual.action2_cost, .0125*(delta/.25).square().sum(-1),atol=1e-8,rtol=1e-4)
    assert raw.action_cost.item() > 3.
    assert actual.action_cost.item() < .0001


def test_relative_pair_motion_is_not_independently_wrapped_and_wheels_never_wrap():
    p = torch.tensor([[4*math.pi,-4*math.pi,0.,0.,-9.,9.]])
    a = -p
    actual, raw = history(a,p), history(a,p,enabled=False)
    # Pair center is unchanged; its large relative phase change is real.
    torch.testing.assert_close(actual.action_cost,raw.action_cost,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(actual.action2_cost,raw.action2_cost,atol=1e-5,rtol=1e-6)
    p[:,:4] = 0
    a[:,:4] = 0
    actual,raw = history(a,p),history(a,p,enabled=False)
    assert torch.equal(actual.action_cost,raw.action_cost)
    assert torch.equal(actual.action2_cost,raw.action2_cost)


def test_second_difference_uses_continuous_increments_without_wrapping_acceleration():
    p2 = torch.zeros(1,6,dtype=torch.float64)
    p2[:,:2] = 4*math.pi-.16
    p = p2.clone(); p[:,:2] += .08
    a = p.clone(); a[:,:2] = -4*math.pi
    actual = history(a,p,p2)
    assert actual.action2_cost.item() < 1e-26
    # Opposite near-pi increments must retain their large acceleration;
    # wrapping the second difference itself would incorrectly erase it.
    p2[:,:2] = 0
    p[:,:2] = 4*math.pi-.1
    a[:,:2] = 0
    actual,raw = history(a,p,p2),history(a,p,p2,enabled=False)
    torch.testing.assert_close(actual.action2_cost,raw.action2_cost,atol=1e-12,rtol=1e-12)
    assert actual.action2_cost.item() > 15.


def test_gradients_match_finite_difference_away_from_the_common_branch_cut():
    p = torch.tensor([[4*math.pi-.2,4*math.pi-.1,.3,-.2,1.,-1.]],dtype=torch.float64)
    p2 = p-.05
    a = torch.tensor([[-4*math.pi+.1,-4*math.pi+.3,.2,-.1,-1.,1.]],dtype=torch.float64,requires_grad=True)
    def objective(value):
        h = history(value,p,p2)
        return (h.action_cost+h.action2_cost).sum()
    gradient, = torch.autograd.grad(objective(a),a)
    assert torch.isfinite(gradient).all()
    numeric = torch.empty_like(a)
    for j in range(6):
        upper,lower = a.detach().clone(),a.detach().clone()
        upper[0,j] += 1e-6; lower[0,j] -= 1e-6
        numeric[0,j] = (objective(upper)-objective(lower))/(2e-6)
    torch.testing.assert_close(gradient,numeric,atol=1e-8,rtol=1e-7)


def test_native_begin_imagination_and_reset_share_raw_history_and_optional_costs(monkeypatch):
    native,assembly,geometry,state,legs,memory = setup('legacy')
    extension = RewardExtensions(coaxial_action_smoothing=True)
    receipt = install_shared_reward(native,extensions=extension)
    assert receipt['coaxial_action_smoothing']['contract'] == 'coaxial-action-smoothing-v1'
    assert receipt['coaxial_action_smoothing']['pairs'] == [['L_joint1','LL_joint1'],['R_joint1','RR_joint1']]
    original = native_reward.validate_issued_control
    original = getattr(original,'supermimic_original',original)
    monkeypatch.setattr(native_reward,'validate_issued_control',_bounded_validator(
        original,(4*math.pi,)*4+(9.,9.),int(ControlSource.PREPARATION_HOLD)))
    assembly.parameters = replace(assembly.parameters,extensions=extension)
    legs = replace(legs,leg_action_bound=4*math.pi)
    ids = torch.zeros(2,dtype=torch.long)
    owned = torch.ones(2,dtype=torch.bool)
    reference = RewardReference(torch.tensor([[0.,0.,0.,.32]]).repeat(2,1),ids,
        torch.full((2,),int(RewardPhase.SUPPORT)),torch.zeros(2),torch.zeros(2),torch.zeros(2,2))
    queue,bias,strength = torch.zeros(2,3,2),torch.zeros(2,6),torch.ones(2,6)
    for interval in range(3):
        action = torch.zeros(2,6)
        action[:,:2] = 4*math.pi-.04 if interval < 2 else -4*math.pi+.04
        native.begin(action,state.joint_velocity[:,:6],now_s=native._time,episode_id=ids,control_epoch=ids,
            policy_owned=owned,admitted_body_command=reference.admitted_body_command,command_mode=ids)
        if interval == 0:
            assert not native._window['action_cost'].any()
        if interval < 2:
            assert not native._window['action2_cost'].any()
        samples = tuple(RewardTick(state,geometry(state),torch.zeros(2),torch.zeros(2,2),torch.zeros(2,21),
            torch.zeros(2,2),torch.zeros(2,3),owned,~owned,reference,native._time+(tick+1)*.005,ids,ids,owned)
            for tick in range(4))
        result = assembly.integrate(state,samples,action,memory,queue,legs,bias,strength)
        target,correction = legs.decode_target(action,state.joint_position[:,:4])
        saved_queue,support = queue,memory.support
        for tick in samples:
            speed = state.joint_velocity[:,4:6]
            saved_queue = assembly.wheels.push(saved_queue,assembly.wheels.motor_input(
                assembly.wheels.target(action),speed,strength[:,4:6]))
            effort = torch.cat((legs.motor_effort(target,state.joint_position,state.joint_velocity,strength),
                                assembly.wheels.applied(saved_queue,speed)),-1)
            physics = assembly.physics(tick,effort,correction)
            tick_reference,support = assembly.advance_reference(tick,support)
            native.observe(physics,tick_reference)
        expected = native.finish(terminated=~owned,truncated=~owned,
            physical_success_request_id=torch.full((2,),-1),entire_skill_policy_owned=owned)
        for term in ('cost.issued_first_difference','cost.issued_second_difference'):
            torch.testing.assert_close(result.components[term],expected.component_densities[term]*.02,atol=1e-12,rtol=1e-12)
        assert torch.equal(native._previous,action)
        assert torch.equal(result.memory.previous_action,action)
        queue,memory = result.wheel_torque_history,result.memory
    assert native._window is None
    native.reset(owned,native._time,ids+1,ids)
    native.begin(action,state.joint_velocity[:,:6],now_s=native._time,episode_id=ids+1,control_epoch=ids,
        policy_owned=owned,admitted_body_command=reference.admitted_body_command,command_mode=ids)
    assert not native._window['action_cost'].any()
    assert not native._window['action2_cost'].any()
