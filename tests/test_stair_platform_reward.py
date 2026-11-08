"""Accepted ordinary-riser targets and the unchanged one-unit height budget."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from transformer_rl.formal.control.observation import CommandMode
from transformer_rl.formal.lab_env.course_tasks import RequestCommand, StairRequestProtocol
from transformer_rl.formal.lab_env.stair_trigger import TofBeam
from transformer_rl.formal.tasks.gimbal_course import GimbalRewardReference
from transformer_rl.formal.tasks.protocol import RequestResponse
from supermimic_rl.course.reward import install_shared_reward
from supermimic_rl.course.stair_platform_reward import NativeStairPlatformReward
from supermimic_rl.reward_kernel import (
    RewardExtensions, StairPlatformGeometry, physical_reward_densities,
)
from supermimic_rl.world_model_v2.imagined_reward import OrdinaryFlatReward
from supermimic_rl.world_model_v2.stair_source import StairSourceState
from supermimic_rl.world_model_v2.stair_task import StairPacket, StairTaskState
from support.world_v2.stair_source import source_fixture
from support.world_v2.stair_task import estimate_at
from test_reward_kernel import begin, sample, setup
from test_stair_support_transfer import inputs
from test_stair_timeout_retry import timeline


def geometry(bottoms, *, valid=(True, True), started=(True, True), top=3.2, lower=3.):
    return StairPlatformGeometry(torch.as_tensor(bottoms), torch.full((2,), top),
                                 torch.full((2,), lower), torch.tensor(valid), torch.tensor(started))


def score(values, target, *, enabled=True, support=None):
    physics, reference, history, parameters = values
    parameters = replace(parameters, extensions=RewardExtensions(
        stair_support_transfer=True, stair_platform_height=enabled))
    return physical_reward_densities(physics, reference, history, parameters,
        stair_base_net_upward_n=torch.full((2,), 4.) if support is None else support,
        stair_platform_geometry=target)


def test_height_budget_is_linear_per_wheel_and_does_not_change_other_terms():
    values = inputs()
    before = score(values, None, enabled=False)
    for bottoms, expected in (([[3., 3.], [3.1, 3.1]], [0., .5]),
                              ([[3., 3.2], [3.3, 3.2]], [.5, 1.]),
                              ([[2., 2.], [4., 4.]], [0., 1.])):
        result = score(values, geometry(bottoms))
        torch.testing.assert_close(result.positive['height'], torch.tensor(expected), atol=2e-6, rtol=0)
        assert result.components.keys() == before.components.keys()
        for key in before.components.keys() - {'positive.height'}:
            torch.testing.assert_close(result.components[key], before.components[key], atol=0, rtol=0)
        assert not result.invalid.any()


@pytest.mark.parametrize('mode', [CommandMode.NORMAL, CommandMode.SLOPE, CommandMode.JUMP, CommandMode.RECOVERY])
def test_non_stair_components_remain_bitwise_identical(mode):
    values = inputs(mode=mode)
    before = score(values, None, enabled=False)
    result = score(values, geometry([[float('nan')] * 2] * 2))
    for key in before.components:
        torch.testing.assert_close(result.components[key], before.components[key], atol=0, rtol=0)
    assert not result.invalid.any() and not result.gates['stair_platform_height_active'].any()


@pytest.mark.parametrize('case', ['unsupported', 'unbound', 'not_permitted', 'gimbal'])
def test_approach_and_unavailable_targets_preserve_old_height(case):
    values = list(inputs(allowed=(False, False) if case == 'not_permitted' else (True, True)))
    if case == 'gimbal':
        values[1] = replace(values[1], gimbal=GimbalRewardReference(
            torch.ones(2, dtype=torch.bool), torch.zeros(2), torch.ones(2), torch.zeros(2)))
    support = torch.zeros(2) if case == 'unsupported' else torch.full((2,), 4.)
    target = geometry([[3., 3.]] * 2, valid=(case != 'unbound',) * 2,
                      started=(case not in ('unsupported', 'not_permitted'),) * 2)
    before, result = score(values, None, enabled=False, support=support), score(values, target, support=support)
    for key in before.components:
        torch.testing.assert_close(result.components[key], before.components[key], atol=0, rtol=0)
    assert not result.gates['stair_platform_height_active'].any()


def test_inactive_bad_geometry_is_safe_and_bound_bad_geometry_is_reported():
    target = geometry([[float('nan')] * 2, [3., 3.]], valid=(False, True))
    assert not score(inputs(), target).invalid.any()
    target = replace(target, valid=torch.ones(2, dtype=torch.bool))
    result = score(inputs(), target)
    assert result.invalid.tolist() == [True, False] and torch.isfinite(result.total).all()
    result = score(inputs(), replace(target, platform_lower_world_z_m=target.platform_top_world_z_m))
    assert result.invalid.all() and torch.isfinite(result.total).all()
    with pytest.raises(ValueError, match='accepted-riser geometry producer'):
        score(inputs(), None)


def test_wheel_progress_gradient_and_unavailable_term_filter():
    bottoms = torch.tensor([[3.04, 3.06], [3.12, 3.16]], dtype=torch.float64, requires_grad=True)
    target = geometry(bottoms)
    objective = lambda b: score(inputs(), replace(target, wheel_bottom_world_z_m=b)).positive['height'].sum()
    grad, = torch.autograd.grad(objective(bottoms), (bottoms,))
    direction = torch.tensor([[.1, -.2], [.3, .4]], dtype=torch.float64)
    eps = 1e-6
    numeric = (objective(bottoms + eps * direction) - objective(bottoms - eps * direction)) / (2 * eps)
    torch.testing.assert_close((grad * direction).sum(), numeric, atol=1e-8, rtol=1e-6)
    assert torch.isfinite(grad).all() and (grad > 0).all()
    values = list(inputs())
    values[0] = replace(values[0], available_terms=('forward',))
    assert not score(values, target).positive['height'].any()


def native_fixture():
    _, trigger = source_fixture(2)
    protocol = StairRequestProtocol(2)
    native = SimpleNamespace(_episode_id=torch.zeros(2, dtype=torch.long),
                             _control_epoch=torch.zeros(2, dtype=torch.long))
    course = SimpleNamespace(tof_trigger=trigger, stair_requests=protocol, _last_tof_beam=None,
        _support_sample=SimpleNamespace(wheel_bottom_world_z_m=torch.full((2, 2), 3.),
                                        valid=torch.ones(2, dtype=torch.bool)),
        query_invalid=torch.zeros(2, dtype=torch.bool))
    native._terrain_course = course
    owner = NativeStairPlatformReward(native).install()
    return native, owner


def beam(trigger, *, lower=3., rise=.2):
    heights = (lower + rise * (trigger.distances_m >= .15)).expand(2, -1).clone()
    return TofBeam(heights, torch.ones_like(heights, dtype=torch.bool), torch.zeros(2, 2),
                   torch.tensor([[1., 0.]]).repeat(2, 1), torch.ones(2, dtype=torch.bool))


def request(native, *, now=.02, request_id=0, fired=True, cancel=-1, completed=-1, ready=True):
    course = native._terrain_course
    # Intentionally differ from actual rise: reward must not use noisy packet height.
    condition = course.stair_requests.step(now_s=torch.full((2,), now, dtype=torch.float64),
        request_id=torch.full((2,), request_id, dtype=torch.long), height_m=torch.full((2,), .15),
        cancel_id=torch.full((2,), cancel, dtype=torch.long), jump_active=torch.zeros(2, dtype=torch.bool),
        completed_id=torch.full((2,), completed, dtype=torch.long), ready=torch.full((2,), ready))
    accepted = condition.response == int(RequestResponse.ACCEPTED)
    course.tof_trigger.record_responses(torch.full((2,), fired), accepted, active_id=condition.active_id)
    return condition


def capture(native, owner):
    return owner(SimpleNamespace(episode_id=native._episode_id, control_epoch=native._control_epoch))


def start_transfer(native, owner):
    physics, reference, *_ = inputs()
    return owner.observe(replace(physics, episode_id=native._episode_id, control_epoch=native._control_epoch),
                         reference, torch.full((2,), 4.), input_valid=torch.ones(2, dtype=torch.bool))


def test_accepted_actual_target_is_fixed_and_busy_request_cannot_replace_it():
    native, owner = native_fixture()
    course = native._terrain_course
    course._last_tof_beam = beam(course.tof_trigger)
    rng = course.tof_trigger.generator.get_state().clone()
    request(native)
    target = capture(native, owner)
    assert target.valid.all()
    assert not target.transfer_started.any()
    assert start_transfer(native, owner).transfer_started.all()
    torch.testing.assert_close(target.platform_lower_world_z_m, torch.full((2,), 3.))
    torch.testing.assert_close(target.platform_top_world_z_m, torch.full((2,), 3.2))
    assert torch.equal(rng, course.tof_trigger.generator.get_state())
    course._last_tof_beam = beam(course.tof_trigger, lower=3.2, rise=.15)
    assert (request(native, now=.04, request_id=1).response == int(RequestResponse.BUSY)).all()
    assert capture(native, owner).valid.all()
    assert capture(native, owner).transfer_started.all()  # BUSY cannot reset the current request's phase.
    torch.testing.assert_close(owner.top, torch.full((2,), 3.2))
    # Clear after record_responses, as deployment FSM exit can do.
    course.stair_requests.clear(torch.tensor([True, False]))
    assert capture(native, owner).valid.tolist() == [False, True]
    assert owner.transfer_started.tolist() == [False, True]


@pytest.mark.parametrize('end', ['timeout', 'cancel', 'complete', 'abort', 'reset', 'episode'])
def test_request_end_invalidates_old_target_and_new_second_step_rebinds(end):
    native, owner = native_fixture()
    course = native._terrain_course
    course._last_tof_beam = beam(course.tof_trigger)
    request(native)
    assert start_transfer(native, owner).transfer_started.all()
    now = .04
    if end == 'reset':
        course.stair_requests.reset(torch.arange(2), torch.full((2,), now, dtype=torch.float64))
    elif end == 'episode':
        native._episode_id.add_(1)
    else:
        kwargs = {'request_id': -1, 'fired': False, 'now': now}
        if end == 'timeout':
            now = course.stair_requests.config.timeout_s + .03
            kwargs['now'] = now
        if end == 'cancel':
            kwargs['cancel'] = 0
        if end == 'complete':
            kwargs['completed'] = 0
        if end == 'abort':
            kwargs['ready'] = False
        request(native, **kwargs)
    assert not capture(native, owner).valid.any()
    assert not capture(native, owner).transfer_started.any()
    course.stair_requests.clear(torch.ones(2, dtype=torch.bool))
    course._last_tof_beam = beam(course.tof_trigger, lower=3.2, rise=.15)
    request(native, now=now + .02, request_id=1)
    assert capture(native, owner).valid.all()
    assert not capture(native, owner).transfer_started.any()  # Second riser starts in approach again.
    torch.testing.assert_close(owner.top, torch.full((2,), 3.35))
    torch.testing.assert_close(owner.lower, torch.full((2,), 3.2))


def test_external_accepted_request_and_invalid_terrain_have_no_guessed_target():
    native, owner = native_fixture()
    course = native._terrain_course
    course._last_tof_beam = beam(course.tof_trigger)
    request(native)
    start_transfer(native, owner)
    course.query_invalid[0] = True
    assert capture(native, owner).valid.tolist() == [False, True]
    course._support_sample = None
    assert not capture(native, owner).valid.any()
    course.stair_requests.clear(torch.ones(2, dtype=torch.bool))
    request(native, now=.04, request_id=1, fired=False)
    assert (owner.request_id == -1).all()
    assert not owner.transfer_started.any()
    original = owner._originals
    owner.close()
    assert (course.tof_trigger.record_responses, course.stair_requests.reset, course.stair_requests.clear) == original


def test_actual_advance_skill_preserves_rng_commands_protocol_and_timeout_retries():
    plain, *_ = timeline(retry=True)
    observed, *_ = timeline(retry=True)
    for native in (plain, observed):
        native._episode_id = torch.zeros(native.num_envs, dtype=torch.long)
    owner = NativeStairPlatformReward(observed).install()
    bindings = set()
    for step in range(55):
        for native in (plain, observed):
            course = native._terrain_course
            previous = native.time.clone()
            native.time = torch.full((native.num_envs,), (step + 1) * .02, dtype=torch.float64)
            native._completed_estimate = estimate_at(previous, native.num_envs)
            packet = StairPacket.empty(native.num_envs, 'cpu')
            if step == 6:
                packet.cancel_id[1] = course.stair_requests.active_id[1]
                packet.completed_id[2] = course.stair_requests.active_id[2]
                packet.safety_abort[3] = True
            course._pending_requests['stair'] = RequestCommand(packet.request_id, packet.height_m,
                packet.cancel_id, packet.safety_abort, completed_id=packet.completed_id)
            course.advance_skill()
        for capture_state in (StairTaskState.capture, StairSourceState.capture):
            left, right = capture_state(plain).record(), capture_state(observed).record()
            for key in left:
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        torch.testing.assert_close(plain._command, observed._command, rtol=0, atol=0)
        assert torch.equal(plain._terrain_course.tof_trigger.generator.get_state(),
                           observed._terrain_course.tof_trigger.generator.get_state())
        active = observed._terrain_course.stair_requests.active_id >= 0
        assert ((owner.request_id >= 0) | ~active).all()
        if active[0]:
            bindings.add(int(owner.request_id[0]))
            assert owner.top[0] == pytest.approx(3.1)
            assert owner.lower[0] == pytest.approx(3.)
    assert len(bindings) >= 3  # New accepted IDs rebind the same riser after actual timeouts.


def test_shared_composer_approach_first_support_and_unloading_use_this_ticks_geometry():
    composer, reference = setup('ordinary')
    native, owner = native_fixture()
    course = native._terrain_course
    course._last_tof_beam = beam(course.tof_trigger)
    request(native)
    extensions = RewardExtensions(stair_support_transfer=True, stair_platform_height=True)
    receipt = install_shared_reward(composer, extensions=extensions, stair_platform_geometry=owner)
    reference = replace(reference, mode=torch.full((2,), int(CommandMode.STAIR)),
                        intentional_base_contact=torch.ones(2, dtype=torch.bool))
    begin(composer, reference, torch.zeros(2, 6))
    for tick, (z, support) in enumerate(zip([3., 3.05, 3.1, 3.2], [0., 4., 0., 0.]), 1):
        course._support_sample.wheel_bottom_world_z_m.fill_(z)
        composer.supermimic_stair_base_net_upward_n = torch.full((2,), support)
        physics = replace(sample(composer, tick), height_above_ground_m=reference.admitted_body_command[:, 3])
        composer.observe(physics, reference)
        assert owner.transfer_started.tolist() == [tick >= 2] * 2
    torch.testing.assert_close(composer._window['components']['positive.height'],
                               torch.full((2,), .005 * (1 + .25 + .5 + 1), dtype=torch.float64), atol=1e-8, rtol=0)
    torch.testing.assert_close(composer._window['gates']['stair_base_supported'],
                               torch.full((2,), .005, dtype=torch.float64), atol=0, rtol=0)
    torch.testing.assert_close(composer._window['gates']['stair_platform_height_active'],
                               torch.full((2,), .015, dtype=torch.float64), atol=0, rtol=0)
    assert receipt['extensions']['stair_platform_height'] is True
    assert receipt['stair_platform_height']['format'] == 'ordinary-stair-accepted-platform-height-transfer-v2'


@pytest.mark.parametrize('invalid_field', ['timestamp_s', 'episode_id', 'control_epoch'])
def test_invalid_pre_reset_identity_cannot_start_transfer(invalid_field):
    composer, reference = setup('ordinary')
    native, owner = native_fixture()
    native._terrain_course._last_tof_beam = beam(native._terrain_course.tof_trigger)
    request(native)
    install_shared_reward(composer, extensions=RewardExtensions(
        stair_support_transfer=True, stair_platform_height=True), stair_platform_geometry=owner)
    reference = replace(reference, mode=torch.full((2,), int(CommandMode.STAIR)),
                        intentional_base_contact=torch.ones(2, dtype=torch.bool))
    begin(composer, reference, torch.zeros(2, 6))
    composer.supermimic_stair_base_net_upward_n = torch.full((2,), 4.)
    physics = sample(composer, 1)
    physics = replace(physics, **{invalid_field: getattr(physics, invalid_field) + 1})
    with pytest.raises(ValueError, match='PRE-reset identity'):
        composer.observe(physics, reference)
    assert not owner.transfer_started.any()


@pytest.mark.parametrize('case', ['no_permission', 'gimbal', 'normal', 'zero', 'threshold', 'nan',
                                  'unbound', 'missing_geometry', 'wrong_native_control'])
def test_phase_starts_only_with_existing_eligible_support_and_complete_identity(case):
    native, owner = native_fixture()
    native._terrain_course._last_tof_beam = beam(native._terrain_course.tof_trigger)
    request(native, fired=case != 'unbound')
    physics, reference, *_ = inputs(mode=CommandMode.NORMAL if case == 'normal' else CommandMode.STAIR,
                                   allowed=(case != 'no_permission',) * 2)
    if case == 'gimbal':
        reference = replace(reference, gimbal=GimbalRewardReference(
            torch.ones(2, dtype=torch.bool), torch.zeros(2), torch.ones(2), torch.zeros(2)))
    if case == 'missing_geometry':
        native._terrain_course._support_sample = None
    if case == 'wrong_native_control':
        native._control_epoch.add_(1)
    force = {'zero': 0., 'threshold': 3., 'nan': float('nan')}.get(case, 4.)
    result = owner.observe(physics, reference, torch.full((2,), force),
                           input_valid=torch.ones(2, dtype=torch.bool))
    assert not result.transfer_started.any() and not owner.transfer_started.any()


def test_once_started_support_permission_loss_and_unloading_do_not_restore_approach_height():
    native, owner = native_fixture()
    native._terrain_course._last_tof_beam = beam(native._terrain_course.tof_trigger)
    request(native)
    start_transfer(native, owner)
    values = inputs(allowed=(False, False))
    geometry = owner.observe(values[0], values[1], torch.zeros(2), input_valid=torch.ones(2, dtype=torch.bool))
    assert geometry.transfer_started.all()
    result = score(values, geometry, support=torch.zeros(2))
    assert not result.positive['height'].any() and result.gates['stair_platform_height_active'].all()
    assert not result.gates['stair_base_supported'].any()


def test_opt_in_contract_default_identity_and_world_model_boundary():
    assert 'stair_platform_height' not in RewardExtensions().receipt()
    for value in (1, 'true', None):
        with pytest.raises(ValueError, match='boolean'):
            RewardExtensions(stair_support_transfer=True, stair_platform_height=value)
    with pytest.raises(ValueError, match='stair_support_transfer'):
        RewardExtensions(stair_platform_height=True)
    with pytest.raises(ValueError, match='geometry producer must match'):
        install_shared_reward(setup('ordinary')[0], extensions=RewardExtensions(
            stair_support_transfer=True, stair_platform_height=True))
    source = {'asset_manifest_sha256': 'test', 'reward_config': {'physics_dt_s': .005},
              'reward_kernel': {'extensions': {'stair_support_transfer': True, 'stair_platform_height': True}}}
    geometry_source = SimpleNamespace(linkage=SimpleNamespace(provenance={'asset_manifest_sha256': 'test'}))
    with pytest.raises(ValueError, match='accepted-riser top/lower target context'):
        OrdinaryFlatReward(source, geometry_source, None, .02)
