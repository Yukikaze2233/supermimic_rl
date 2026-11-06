"""Equal-effort relabeling is checked against the deployed controller chain."""

from dataclasses import replace
from functools import cache
import hashlib
import json
import math
import sys
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.contracts import Field, StateSchema
from supermimic_rl.course.identified_execution import IdentifiedExecution
from supermimic_rl.course.action_contract import install_action_bounds
from supermimic_rl.course.randomization import install_identified_feedback
from supermimic_rl.world_model_v2.action_equivalence import FirstStepLegActionEquivalence
from supermimic_rl.world_model_v2.model import Ticks
from supermimic_rl.world_model_v2.physical import PhysicalState, increments
from supermimic_rl.world_model_v2.records import (
    INCREMENT_DIM, RecordSpec, decode_memory, decode_physical, encode_physical,
)
from support import REPO_ROOT


RUNTIME = REPO_ROOT / 'configs/gpufree/runtime-physx64-16-fullturn-knee40-107-push-tof.json'


@pytest.fixture(autouse=True)
def native_action_domain(monkeypatch):
    upstream = pytest.importorskip('transformer_rl.formal.control.actuation')
    # The production factory installs the runtime domain in both the decoder
    # and native ownership validator. Restore its module bindings after each test.
    for name, module in tuple(sys.modules.items()):
        if name.startswith('transformer_rl.') and hasattr(module, 'validate_issued_control'):
            monkeypatch.setattr(module, 'validate_issued_control', module.validate_issued_control)
    bounds = json.loads(RUNTIME.read_text())['task']['action_bounds']
    controller = upstream.OutputFeedbackController([0.] * 6, 1, pair_domain=None)
    install_action_bounds(SimpleNamespace(_action_bounds=torch.tensor(bounds),
        _control=SimpleNamespace(controller=controller)), bounds)


def native_execution(batch):
    return IdentifiedExecution(batch, 'cpu', .005,
        asset_directory=REPO_ROOT / 'assets/full_vehicle_v2p0p1_source_audited_v2',
        bundle_directory=REPO_ROOT / 'configs/identified/20261031',
        friction_mode='software_dynamic')


@cache
def execution_receipt():
    return native_execution(1).receipt()


def runtime_schema(runtime=None):
    runtime = json.loads(RUNTIME.read_text()) if runtime is None else runtime
    encoded = json.dumps(runtime, sort_keys=True).encode()
    source = {'runtime_sha256': hashlib.sha256(encoded).hexdigest(),
              'action_bounds': runtime['task']['action_bounds'],
              'execution': execution_receipt()}
    schema = StateSchema((Field('fixture', (1,)),), (),
                         substep_fields=(Field('ticks', (1,)),), substeps_per_step=4,
                         source=json.dumps(source))
    return schema, encoded


def windows(batch=2, horizon=8, *, both_pairs=False):
    record = {name: torch.zeros(batch, horizon, *shape)
              for name, shape in RecordSpec(16, 65).fields().items()}
    q, velocity = torch.zeros(batch, 6), torch.zeros(batch, 8)
    velocity[:, :2] = torch.tensor([.3, -.2])
    velocity[:, 4:6] = torch.tensor([55., -62.])
    state = PhysicalState(torch.tensor([[0., 0., .35]]).repeat(batch, 1),
                          torch.eye(3).repeat(batch, 1, 1), torch.zeros(batch, 3),
                          torch.zeros(batch, 3), q, velocity)
    record['physical'] = encode_physical(state)[:, None].repeat(1, horizon, 1)
    record['end_physical'] = record['physical'].clone()
    record['height'].fill_(.35)
    record['end_height'].fill_(.35)
    record['scan'].fill_(-.35)
    record['motor_strength'].fill_(1.)
    record['action'][..., :2] = torch.tensor([4., -4.])
    record['action'][..., 4:] = torch.tensor([.7, -.9])
    if both_pairs:
        record['action'][..., 2:4] = torch.tensor([-4., 4.])
    record['memory'][:, 1:, 6:12] = record['action'][:, :-1]
    record['memory'][:, 0, 6:12] = .125
    nominal = torch.zeros(batch, horizon, 6)
    stiffness = torch.tensor([160.] * 4 + [0., 0.]).expand_as(nominal)
    damping = torch.tensor([2.5] * 4 + [.5, .5]).expand_as(nominal)
    bounds = torch.tensor([40.] * 4 + [4.5, 4.5]).expand_as(nominal)
    bias = torch.tensor([.017, -.013, 0., 0., 0., 0.]).expand_as(nominal)
    record['controller_parameters'] = torch.cat((nominal, stiffness, damping, bounds, bias), -1)
    for tick, offset in enumerate((.02, .04, .06, 0.)):
        endpoint = replace(state, joint_position=state.joint_position + offset)
        record['ticks'][:, :, tick, :INCREMENT_DIM] = increments(state, endpoint)[:, None]
    # Contact/knee labels and task/observation values must not be relabeled.
    record['ticks'][..., INCREMENT_DIM:] = torch.arange(
        record['ticks'].shape[-1] - INCREMENT_DIM).float()
    record['frame'].fill_(.37)
    record['reward'].fill_(.9)
    record['reward_context_valid'].fill_(1.)
    record['valid'] = torch.ones(batch, horizon, dtype=torch.bool)
    record['policy_owned'] = torch.ones(batch, horizon, dtype=torch.bool)
    record['end_policy_owned'] = torch.ones(batch, horizon, dtype=torch.bool)
    return record


def native_trace(record, action):
    """Evaluate native held targets at the four *start* states, including losses."""
    upstream = pytest.importorskip('transformer_rl.formal.control.actuation')
    batch = len(action)
    parameters = record['controller_parameters'][:, 0]
    nominal, _, _, _, bias = parameters.split(6, -1)
    assert torch.equal(nominal, nominal[0].expand_as(nominal))
    controller = upstream.OutputFeedbackController(nominal[0].tolist(), batch, pair_domain=None)
    install_identified_feedback(controller)
    controller.action_bounds = torch.tensor(json.loads(RUNTIME.read_text())['task']['action_bounds'])
    execution = native_execution(batch)
    state = decode_physical(record['physical'][:, 0])
    truth = Ticks.decode(record['ticks'][:, 0], state)
    issued = torch.zeros(batch, dtype=torch.float64)
    targets = controller.decode(action, state.joint_position + bias, issued)
    traces = []
    for tick in range(4):
        feedback = state if tick == 0 else truth.state.tick(tick - 1)
        effort = controller.effort(targets, feedback.joint_position + bias,
            feedback.joint_velocity[:, :6], issued + tick * .005,
            strength=record['motor_strength'][:, 0]).capped_effort
        traces.append(execution.step(effort, feedback.joint_velocity[:, :6]).native_effort_nm)
    return torch.stack(traces, 1), targets.common_turns


def test_only_first_pair_action_and_following_previous_action_change():
    record = windows(both_pairs=True)
    original = {name: value.clone() for name, value in record.items()}
    augmenter = FirstStepLegActionEquivalence.from_runtime(*runtime_schema())
    result = augmenter.augment(record, torch.tensor([[.2, .8], [.8, .2]]))
    assert result.changed_rows.all()
    assert result.eligible_pairs.all()
    assert result.selected_pair.tolist() == [0, 1]
    assert (result.delta_phi != 0).all()
    assert not result.max_effort_difference.any()
    for name in record:
        assert torch.equal(record[name], original[name]), name
        if name not in ('action', 'memory'):
            assert torch.equal(result.windows[name], original[name]), name
    changed = result.windows['action'][:, 0] != original['action'][:, 0]
    assert changed.sum(-1).tolist() == [2, 2]
    assert changed[:, 4:].sum() == 0
    assert torch.equal(result.windows['action'][:, 1:], original['action'][:, 1:])
    expected_memory = original['memory'].clone()
    expected_memory[:, 1, 6:12] = result.windows['action'][:, 0]
    assert torch.equal(result.windows['memory'], expected_memory)
    assert torch.equal(decode_memory(result.windows['memory'][:, 0]).previous_action,
                       decode_memory(original['memory'][:, 0]).previous_action)
    before, turns_before = native_trace(record, original['action'][:, 0])
    after, turns_after = native_trace(record, result.windows['action'][:, 0])
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    assert torch.equal(turns_after, turns_before)


@pytest.mark.parametrize('strength', [.84, 1., 1.16])
def test_both_native_saturation_stages_bias_and_friction_preserve_exact_effort(strength):
    record = windows(batch=17)
    record['motor_strength'][..., :4] = strength
    samples = torch.stack((torch.full((17,), .2), torch.arange(17).float() / 17), -1)
    result = FirstStepLegActionEquivalence.from_runtime(*runtime_schema()).augment(record, samples)
    assert result.changed_rows.any()
    native_before, turns_before = native_trace(record, record['action'][:, 0])
    native_after, turns_after = native_trace(record, result.windows['action'][:, 0])
    torch.testing.assert_close(native_after, native_before, rtol=0, atol=0)
    assert torch.equal(turns_after, turns_before)
    bounds = torch.tensor(json.loads(RUNTIME.read_text())['task']['action_bounds'])
    assert (result.windows['action'].abs() <= bounds).all()


@pytest.mark.parametrize('control_tick', [0, 1, 2, 3])
def test_any_unsaturated_control_start_prevents_the_pair_relabel(control_tick):
    record = windows(batch=1)
    bias = record['controller_parameters'][0, 0, 24:30]
    state = decode_physical(record['physical'][:, 0])
    velocity = state.joint_velocity.clone()
    velocity[:, :2] = 0.
    q = state.joint_position.clone()
    q[:, :2] = record['action'][:, 0, :2] * .25 - bias[:2]
    unsaturated = replace(state, joint_position=q, joint_velocity=velocity)
    if control_tick == 0:
        # Keep all recorded endpoints unchanged while changing the interval start.
        endpoints = Ticks.decode(record['ticks'][:, 0], state).state
        record['physical'][:, 0] = encode_physical(unsaturated)
        for tick in range(4):
            record['ticks'][:, 0, tick, :INCREMENT_DIM] = increments(unsaturated, endpoints.tick(tick))
    else:
        record['ticks'][:, 0, control_tick - 1, :INCREMENT_DIM] = increments(state, unsaturated)
    result = FirstStepLegActionEquivalence.from_runtime(*runtime_schema()).augment(
        record, torch.tensor([[.2, .8]]))
    assert not result.eligible_pairs.any()
    assert not result.changed_rows.any()
    assert result.selected_pair.tolist() == [-1]
    for name in record:
        assert torch.equal(result.windows[name], record[name]), name


def test_final_endpoint_is_not_a_fifth_control_start():
    record = windows(batch=1)
    state = decode_physical(record['physical'][:, 0])
    q = state.joint_position.clone()
    q[:, :2] = record['action'][:, 0, :2] * .25
    endpoint = replace(state, joint_position=q)
    record['ticks'][:, 0, 3, :INCREMENT_DIM] = increments(state, endpoint)
    result = FirstStepLegActionEquivalence.from_runtime(*runtime_schema()).augment(
        record, torch.tensor([[.2, .8]]))
    assert result.changed_rows.item()
    before, _ = native_trace(record, record['action'][:, 0])
    after, _ = native_trace(record, result.windows['action'][:, 0])
    torch.testing.assert_close(after, before, rtol=0, atol=0)


def test_strength_above_one_can_saturate_before_the_nominal_pd_limit():
    record = windows(batch=1)
    record['motor_strength'][..., :4] = 1.16
    record['action'][..., :2] = torch.tensor([.9, -.9])
    record['controller_parameters'][..., 24:30] = 0.
    record['physical'][..., -8:] = 0.
    record['ticks'][..., :INCREMENT_DIM] = 0.
    record['memory'][:, 1:, 6:12] = record['action'][:, :-1]
    # Native requested PD is +/-36 Nm; the second cap makes it +/-40 Nm.
    result = FirstStepLegActionEquivalence.from_runtime(*runtime_schema()).augment(
        record, torch.tensor([[.2, .9]]))
    assert result.changed_rows.item()
    before, _ = native_trace(record, record['action'][:, 0])
    after, _ = native_trace(record, result.windows['action'][:, 0])
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    torch.testing.assert_close(before[..., :2], torch.tensor([40., -40.]).expand(1, 4, 2),
                               rtol=0, atol=0)


def test_near_common_turn_cut_keeps_the_original_decode_branch():
    record = windows(batch=33)
    record['action'][..., :2] = torch.tensor([4.1, -3.9])
    record['memory'][:, 1:, 6:12] = record['action'][:, :-1]
    record['controller_parameters'][..., 24:30] = 0.
    # At this cut a rounded pair-center change can change the common turn.
    # Large damping makes effort alone insufficient to detect that branch swap.
    record['physical'][..., 13:15] = math.pi + .025
    record['physical'][..., 19:21] = 1e3
    record['ticks'][..., :INCREMENT_DIM] = 0.
    samples = torch.stack((torch.full((33,), .2), torch.arange(33).float() / 33), -1)
    result = FirstStepLegActionEquivalence.from_runtime(*runtime_schema()).augment(record, samples)
    assert result.changed_rows.any()
    before, branch_before = native_trace(record, record['action'][:, 0])
    after, branch_after = native_trace(record, result.windows['action'][:, 0])
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    assert torch.equal(branch_after, branch_before)


def test_single_interval_is_outside_the_multistep_memory_contract():
    record = windows(batch=1, horizon=1)
    with pytest.raises(ValueError, match='multi-step'):
        FirstStepLegActionEquivalence.from_runtime(*runtime_schema()).augment(
            record, torch.tensor([[.2, .8]]))


@pytest.mark.parametrize('evidence', ['controller_parameters', 'policy_owned',
                                    'end_policy_owned', 'reward_context_valid'])
def test_missing_control_evidence_preserves_the_training_batch(evidence):
    record = windows()
    record.pop(evidence)
    result = FirstStepLegActionEquivalence.from_runtime(*runtime_schema()).augment(
        record, torch.tensor([[.2, .8], [.2, .8]]))
    assert not result.changed_rows.any()
    assert not result.eligible_pairs.any()
    for name in record:
        assert torch.equal(result.windows[name], record[name]), name


@pytest.mark.parametrize('evidence', ['valid', 'policy_owned', 'end_policy_owned', 'reward_context_valid'])
def test_script_owned_release_or_invalid_context_rows_are_not_relabelled(evidence):
    record = windows()
    record[evidence][0, 0] = 0
    result = FirstStepLegActionEquivalence.from_runtime(*runtime_schema()).augment(
        record, torch.tensor([[.2, .8], [.2, .8]]))
    assert result.changed_rows.tolist() == [False, True]
    for name in record:
        assert torch.equal(result.windows[name][0], record[name][0]), name


@pytest.mark.parametrize('field', ['action_transport', 'action_latch', 'communication'])
def test_unsupported_delayed_control_is_rejected_before_reusing_labels(field):
    runtime = json.loads(RUNTIME.read_text())
    runtime[field]['enabled'] = True
    with pytest.raises(ValueError):
        FirstStepLegActionEquivalence.from_runtime(*runtime_schema(runtime))


@pytest.mark.parametrize('field', ['pair_domain', 'motor_envelope'])
def test_unverified_decoder_or_motor_curve_cannot_claim_equivalence(field):
    runtime = json.loads(RUNTIME.read_text())
    runtime[field] = {'different': True}
    with pytest.raises(ValueError):
        FirstStepLegActionEquivalence.from_runtime(*runtime_schema(runtime))


def test_runtime_identity_and_action_domain_must_match_the_schema():
    schema, encoded = runtime_schema()
    with pytest.raises(ValueError):
        FirstStepLegActionEquivalence.from_runtime(schema, encoded + b' ')
    source = json.loads(schema.source)
    source['action_bounds'] = [3.] * 4 + [9.] * 2
    with pytest.raises(ValueError):
        FirstStepLegActionEquivalence.from_runtime(replace(schema, source=json.dumps(source)), encoded)


@pytest.mark.parametrize('field,value', [
    ('coordinate', 'motor_rotor'),
    ('encoder_holding_compensation', True),
    ('action_order', ['L_joint1'] * 6),
    ('model_pins', {}),
])
def test_different_identified_execution_cannot_reuse_equal_pd_labels(field, value):
    schema, encoded = runtime_schema()
    source = json.loads(schema.source)
    source['execution'][field] = value
    with pytest.raises(ValueError, match='pinned identified'):
        FirstStepLegActionEquivalence.from_runtime(replace(schema, source=json.dumps(source)), encoded)
