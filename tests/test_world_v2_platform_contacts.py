"""Predicted attribution is separate from true patches and motion labels."""
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.course.atlas_cylinder_gap import AtlasCylinderTerrainGap
from supermimic_rl.course.jump_physics import PlatformLandingEvidence, PlatformLandingState
from supermimic_rl.course.native_platform_landing import PlatformLandingInterval
from supermimic_rl.course.platform_intent import PlatformIntentPhase, PlatformIntentState
from supermimic_rl.world_model_v2.model import Ticks
from supermimic_rl.world_model_v2.platform_contacts import (
    PLATFORM_CONTACT_FEATURE_DIM, PLATFORM_CONTACT_PRODUCER, PlatformContactModel,
)
from supermimic_rl.world_model_v2.platform_records import encode_platform_interval
from supermimic_rl.world_model_v2.records import CONTACT_BODIES, WHEEL_BODIES, TICK_LABEL_DIM, encode_physical
from supermimic_rl.world_model_v2.physical import increments
from support.world_v2.reward_assembly import setup
from test_atlas_cylinder_gap import terrain, rectangle
from test_platform_geometry import fixture


def inputs(tmp_path, *, origin=(64., 128., 2.), x=-.02, ground=.2, z_offset=0.):
    _, _, wheel_geometry, state, _, _ = setup('legacy')
    mesh = rectangle(-3, -3, 0, 3, z_offset) + rectangle(0, -3, 3, 3, .2 + z_offset)
    atlas = terrain([mesh, mesh], origins=(origin, (-64., -128., -3.)))
    atlas.manifest_sha256 = 'atlas-pin'

    def height_at(points, tile):
        height = torch.where(points[..., 0] >= 0, ground + z_offset, z_offset)
        return SimpleNamespace(height=height, valid=torch.ones_like(height, dtype=torch.bool))

    atlas.height_at = height_at
    platform = fixture(tmp_path, rectangle(0, -3, 3, 3, .2 + z_offset), terrain=atlas)
    radius = tuple(float(side.radius) for side in wheel_geometry.sides)
    length = tuple(float(side.half_length) * 2 for side in wheel_geometry.sides)
    gap = AtlasCylinderTerrainGap(atlas, radius, length, tile_ids=(0, 1))
    knees = torch.stack([wheel_geometry.linkage.evaluate(side.name,
        state.joint_position[:, side.secondary] - state.joint_position[:, side.primary])[0]
        ['passive_joint'][:, side.passive_knee] for side in wheel_geometry.sides], -1)
    wheels = wheel_geometry(state, passive_knee_position=knees, passive_knee_velocity=torch.zeros_like(knees))
    pos = state.position.clone()
    pos[:, 0] += x - wheels.centre_world[:, 0, 0]
    pos[:, 2] += radius[0] + .2 + z_offset - wheels.centre_world[:, 0, 2]
    state = replace(state, position=pos)
    state = state.with_time_axis().map(lambda value: value.expand(-1, 4, *value.shape[2:]).clone())
    body = torch.zeros(2, 4, len(CONTACT_BODIES))
    indices = [CONTACT_BODIES.index(name) for name in WHEEL_BODIES]
    body[..., indices] = 20.
    ticks = Ticks(state, torch.full((2, 4, 2), 20.), body, torch.zeros(2, 4, 3),
                  knees[:, None].expand(-1, 4, -1), torch.zeros(2, 4, 2))
    intent = replace(PlatformIntentState.plan(target_index=torch.zeros(2, dtype=torch.long),
        earliest_episode_s=torch.zeros(2, dtype=torch.float64), rise_m=torch.full((2,), .2),
        support_height_m=torch.full((2,), .4)),
        phase=torch.full((2,), int(PlatformIntentPhase.ACCEPTED)),
        bound_request_id=torch.full((2,), 7, dtype=torch.long))
    model = PlatformContactModel(wheel_geometry, platform, gap, hidden=())
    set_output(model, height=.2 + z_offset - state.position[0, 0, 2].item())
    kwargs = dict(intent=intent, jump_state=SimpleNamespace(physical_id=torch.full((2,), 7, dtype=torch.long)),
                  terrain_tile=torch.zeros(2, dtype=torch.long), command_height=torch.full((2,), .4))
    return model, ticks, kwargs


def set_output(model, logits=(8., 8., 8.), height=None):
    with torch.no_grad():
        model.net[-1].weight.zero_()
        model.net[-1].bias[:3].copy_(torch.tensor(logits))
        if height is not None:
            model.net[-1].bias[3] = height / model.height_norm.std().item()


def record(model, prediction, kwargs, *, ticks, command=.4, ground=.2):
    truth = replace(prediction.evidence, landing_world_height=torch.where(
        prediction.evidence.landing_support_valid,
        torch.full((2, 4), ground + command) + model.geometry.terrain.origins[kwargs['terrain_tile'], 2, None],
        torch.nan))
    state = PlatformLandingState(torch.full((2, 4), 7), torch.zeros(2, 4, dtype=torch.long),
                                 torch.zeros(2, 4, dtype=torch.long))
    return {**encode_platform_interval(PlatformLandingInterval(truth, state), count=2, device='cpu'),
        'contact_valid_ticks': torch.ones(2, 4, 1, dtype=torch.bool),
        'platform_record_valid': torch.ones(2, 1, dtype=torch.bool),
        'terrain_tile': kwargs['terrain_tile'][:, None].float(), 'valid': torch.ones(2, dtype=torch.bool),
        'physical': encode_physical(ticks.state.tick(0)),
        'ticks': torch.cat((increments(ticks.state.tick(0).with_time_axis(), ticks.state),
            ticks.wheel_load, ticks.body_normal_load, ticks.resultant_contact_force,
            ticks.passive_knee_position, ticks.passive_knee_velocity), -1)}


def test_rim_outside_centre_can_predict_target_without_fabricating_patches(tmp_path):
    model, ticks, kwargs = inputs(tmp_path)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    output = model.predict(ticks, **kwargs)
    assert output.features.shape == (2, 4, PLATFORM_CONTACT_FEATURE_DIM)
    assert output.logits.shape == (2, 4, 3) and output.available.all()
    assert output.producer == PLATFORM_CONTACT_PRODUCER
    assert output.evidence.target_wheel_touch.all() and output.evidence.landing_support_valid.all()
    assert not output.evidence.full_body_clear.any()
    torch.testing.assert_close(output.evidence.landing_world_height, torch.full((2, 4), .6))
    assert (model.wheel_geometry(ticks.state).centre_world[..., 0] < 0).all()
    direct = model(ticks, **kwargs)
    for field in fields(direct):
        torch.testing.assert_close(getattr(direct, field.name), getattr(output.evidence, field.name), equal_nan=True)
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])  # Forward never observes or fits.
    assert TICK_LABEL_DIM == 56


def test_wall_and_nonwheel_contact_are_not_landing_or_full_air(tmp_path):
    model, ticks, kwargs = inputs(tmp_path)
    wall = replace(ticks, wheel_load=torch.zeros_like(ticks.wheel_load))
    result = model(wall, **kwargs)
    assert result.wheel_touch.all() and not result.target_wheel_touch.any()
    assert not result.landing_support_valid.any() and not result.full_body_clear.any()
    air_state = replace(ticks.state, position=ticks.state.position + torch.tensor([0., 0., .3]))
    air = replace(ticks, state=air_state, wheel_load=torch.zeros_like(ticks.wheel_load),
                  body_normal_load=torch.zeros_like(ticks.body_normal_load))
    assert model(air, **kwargs).full_body_clear.all()
    # An auxiliary linkage, not just base/gimbal, supports the robot.
    body = air.body_normal_load.clone()
    body[..., CONTACT_BODIES.index('LL_link3')] = 3.
    supported = model(replace(air, body_normal_load=body), **kwargs)
    assert supported.nonwheel_contact.all() and not supported.full_body_clear.any()


def test_single_target_wheel_and_mixed_wall_touch_remain_distinct(tmp_path):
    model, ticks, kwargs = inputs(tmp_path)
    set_output(model, logits=(-8., 8., 8.))
    result = model(ticks, **kwargs)
    assert result.target_wheel_touch.tolist() == [[[False, True]] * 4] * 2
    assert result.wheel_touch.all() and result.landing_support_valid.all()
    # No rule reconstructs target attribution from the aggregate support load.
    set_output(model, logits=(-8., -8., 8.))
    assert not model(ticks, **kwargs).target_wheel_touch.any()


def test_existing_load_and_gap_bands_are_unchanged(tmp_path):
    model, ticks, kwargs = inputs(tmp_path)
    low = replace(ticks, wheel_load=torch.full_like(ticks.wheel_load, 4.99))
    assert not model(low, **kwargs).landing_support_valid.any()
    assert model(replace(ticks, wheel_load=torch.full_like(ticks.wheel_load, 5.)), **kwargs).landing_support_valid.all()
    for displacement in (.02, -.02):
        moved = replace(ticks, state=replace(ticks.state,
            position=ticks.state.position + torch.tensor([0., 0., displacement])))
        assert not model(moved, **kwargs).landing_support_valid.any()


def test_inactive_rows_do_not_query_geometry_or_consume_future_records(tmp_path, monkeypatch):
    model, ticks, kwargs = inputs(tmp_path)
    inactive = PlatformIntentState.inactive(2)
    def forbidden(*args, **kwargs):
        raise AssertionError('An inactive caller must not query platform geometry')
    monkeypatch.setattr(model.clearance, 'at', forbidden)
    monkeypatch.setattr(model.geometry, 'nearest', forbidden)
    monkeypatch.setattr(model.geometry.terrain, 'height_at', forbidden)
    # A future-label-looking attribute is not part of the forward interface.
    object.__setattr__(ticks, 'platform_evidence', object())
    result = model(ticks, **{**kwargs, 'intent': inactive})
    assert not result.selected.any() and not result.target_wheel_touch.any()
    assert result.wheel_touch.all() and torch.isnan(result.landing_world_height).all()


def test_mixed_binding_only_queries_selected_rows_and_unknown_tile_is_not_positive(tmp_path, monkeypatch):
    model, ticks, kwargs = inputs(tmp_path)
    calls = []
    original = model.clearance.at
    def counted(centres, axes, tile):
        calls.append(len(centres))
        return original(centres, axes, tile)
    monkeypatch.setattr(model.clearance, 'at', counted)
    intent = replace(kwargs['intent'], bound_request_id=torch.tensor([7, 6]))
    output = model.predict(ticks, **{**kwargs, 'intent': intent})
    assert calls == [4] and output.evidence.selected[:, 0].tolist() == [True, False]
    wrong = model.predict(ticks, **{**kwargs, 'terrain_tile': torch.ones(2, dtype=torch.long)})
    assert not wrong.available.any() and not wrong.evidence.target_wheel_touch.any()


def test_occluded_nearest_surface_is_a_feature_not_a_false_patch_certificate(tmp_path):
    model, ticks, kwargs = inputs(tmp_path, x=.3, ground=.4)
    output = model.predict(ticks, **kwargs)
    # Offset 82:84 is nearest target visibility; the high atlas overlay hides it.
    assert not output.features[..., 82:84].bool().any()
    assert output.available.all()  # Source is known; visibility is not missing data.
    set_output(model, logits=(-8., -8., -8.))
    assert not model(ticks, **kwargs).target_wheel_touch.any()


def test_labels_remove_actual_nonzero_origin_and_actual_command(tmp_path):
    model, ticks, kwargs = inputs(tmp_path)
    output = model.predict(ticks, **kwargs)
    data = record(model, output, kwargs, ticks=ticks, command=.37)
    target = model.targets(data, command_height=torch.full((2, 4), .37))
    assert target.valid.all() and target.height_valid.all()
    torch.testing.assert_close(target.support_height_relative_root, .2 - ticks.state.position[..., 2], atol=2e-7, rtol=0)
    assert kwargs['intent'].support_height_m[0] == .4  # Deliberately not the label command.
    original = {k: v.clone() for k, v in data.items()}
    model.observe(output, target)
    assert model.input_norm.count.item() == 8 and model.height_norm.count.item() == 8
    for name, value in data.items():
        torch.testing.assert_close(value, original[name], equal_nan=True)


def test_unknown_inactive_and_invalid_labels_are_not_negative_training_examples(tmp_path):
    model, ticks, kwargs = inputs(tmp_path)
    output = model.predict(ticks, **kwargs)
    data = record(model, output, kwargs, ticks=ticks)
    data['platform_record_valid'][0] = False
    data['contact_valid_ticks'][1, 0] = False
    data['platform_evidence_ticks'][1, 1, 0] = False
    data['platform_evidence_ticks'][1, 2, 8] = False
    data['platform_landing_world_height_ticks'][1, 2] = torch.nan
    target = model.targets(data, command_height=kwargs['command_height'])
    assert target.valid.sum().item() == 2 and target.height_valid.sum().item() == 1
    model.observe(output, target)
    assert model.input_norm.count.item() == 2 and model.height_norm.count.item() == 1
    with pytest.raises(ValueError, match='Missing platform tick evidence'):
        model.targets({}, command_height=kwargs['command_height'])
    data['platform_landing_world_height_ticks'][1, 3] = torch.nan
    with pytest.raises(ValueError, match='finite height'):
        model.targets(data, command_height=kwargs['command_height'])


def test_logits_and_support_height_keep_gradients_and_loss_is_supervised_only(tmp_path):
    model, ticks, kwargs = inputs(tmp_path)
    # An explicit nonconstant head proves geometry/state derivatives remain live.
    with torch.no_grad():
        model.net[-1].weight.normal_(0., .02)
    pos = ticks.state.position.clone().requires_grad_()
    prediction = model.predict(replace(ticks, state=replace(ticks.state, position=pos)), **kwargs)
    grad = torch.autograd.grad(prediction.logits.sum() + prediction.support_height_relative_root.sum(), pos,
                               retain_graph=True)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    # Native future labels live solely on this separate supervised path.
    reference = record(model, model.predict(ticks, **kwargs), kwargs, ticks=ticks)
    reference['platform_evidence_ticks'][..., 5:7] = True
    reference['platform_evidence_ticks'][..., 8] = True
    reference['platform_landing_world_height_ticks'].fill_(2.6)
    target = model.targets(reference, command_height=kwargs['command_height'])
    losses = model.losses(prediction, target)
    assert set(losses) == {'target_contact', 'support_contact', 'support_height'}
    sum(losses.values()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert not prediction.evidence.target_wheel_touch.requires_grad
    assert prediction.logits.requires_grad


def test_empty_supervision_has_zero_loss_and_does_not_update_normalizers(tmp_path):
    model, ticks, kwargs = inputs(tmp_path)
    output = model.predict(ticks, **kwargs)
    data = record(model, output, kwargs, ticks=ticks)
    data['valid'].fill_(False)
    labels = model.targets(data, command_height=kwargs['command_height'])
    model.observe(output, labels)
    assert model.input_norm.count.item() == model.height_norm.count.item() == 0
    assert all(value.item() == 0 for value in model.losses(output, labels).values())


def test_checkpoint_restores_only_independent_head_and_its_norms(tmp_path):
    model, ticks, kwargs = inputs(tmp_path)
    output = model.predict(ticks, **kwargs)
    model.observe(output, model.targets(record(model, output, kwargs, ticks=ticks), command_height=kwargs['command_height']))
    clone = PlatformContactModel(model.wheel_geometry, model.geometry, model.clearance, hidden=())
    clone.load_state_dict(model.state_dict())
    first, second = model.predict(ticks, **kwargs), clone.predict(ticks, **kwargs)
    torch.testing.assert_close(first.logits, second.logits, rtol=0, atol=0)
    torch.testing.assert_close(first.support_height_relative_root, second.support_height_relative_root, rtol=0, atol=0)
    assert all(not key.startswith('motion.') for key in clone.state_dict())


def test_scene_z_translation_preserves_features_and_relative_height_labels(tmp_path):
    first_dir, second_dir = tmp_path / 'first', tmp_path / 'second'
    first_dir.mkdir()
    second_dir.mkdir()
    model, ticks, kwargs = inputs(first_dir)
    shifted, moved, shifted_kwargs = inputs(second_dir, z_offset=.3)
    shifted.load_state_dict(model.state_dict())
    a, b = model.predict(ticks, **kwargs), shifted.predict(moved, **shifted_kwargs)
    torch.testing.assert_close(a.features, b.features, atol=1e-5, rtol=1e-6)
    torch.testing.assert_close(a.support_height_relative_root, b.support_height_relative_root)
    torch.testing.assert_close(b.evidence.landing_world_height, a.evidence.landing_world_height + .3)
    a_labels = model.targets(record(model, a, kwargs, ticks=ticks), command_height=kwargs['command_height'])
    b_labels = shifted.targets(record(shifted, b, shifted_kwargs, ticks=moved, ground=.5),
                               command_height=shifted_kwargs['command_height'])
    torch.testing.assert_close(a_labels.support_height_relative_root, b_labels.support_height_relative_root,
                               atol=3e-7, rtol=0)


def test_true_patch_labels_can_disagree_with_all_negative_predicted_logits(tmp_path):
    from supermimic_rl.course.native_platform_landing import (
        PlatformContactPatches, PlatformLandingEvidenceBuilder,
    )
    model, ticks, kwargs = inputs(tmp_path)
    set_output(model, logits=(-8., -8., -8.))
    prediction = model.predict(ticks, **kwargs)
    assert not prediction.evidence.target_wheel_touch.any()
    real_geometry = model.wheel_geometry(ticks.state, passive_knee_position=ticks.passive_knee_position,
                                        passive_knee_velocity=ticks.passive_knee_velocity)
    true_tick = []
    origins = model.geometry.terrain.origins[kwargs['terrain_tile']]
    for tick in range(4):
        point = real_geometry.centre_world[:, tick].clone()
        point[..., 0], point[..., 2] = .001, .2
        point += origins[:, None]
        patches = PlatformContactPatches(torch.tensor([0, 0, 1, 1]),
            torch.tensor(model.wheel_indices).repeat(2), point.reshape(-1, 3),
            torch.tensor([[0., 0., 1.]]).expand(4, -1), torch.full((4,), 20.))
        gap = model.clearance.at(real_geometry.centre_world[:, tick],
                                real_geometry.cylinder_axis_world[:, tick], kwargs['terrain_tile'])
        true_tick.append(PlatformLandingEvidenceBuilder(model.geometry, model.wheel_indices).build(
            patches=patches, body_load_n=ticks.body_normal_load[:, tick],
            contact_valid=torch.ones(2, dtype=torch.bool), tile=kwargs['terrain_tile'],
            selected=torch.ones(2, dtype=torch.bool), target_index=kwargs['intent'].target_index,
            command_height_m=kwargs['command_height'], clearance=gap))
    native_evidence = PlatformLandingEvidence(**{field.name: torch.stack(
        [getattr(item, field.name) for item in true_tick], 1) for field in fields(true_tick[0])})
    data = record(model, prediction, kwargs, ticks=ticks)
    interval = PlatformLandingInterval(native_evidence, PlatformLandingState(
        torch.full((2, 4), 7), torch.zeros(2, 4, dtype=torch.long), torch.zeros(2, 4, dtype=torch.long)))
    data.update(encode_platform_interval(interval, count=2, device='cpu'))
    targets = model.targets(data, command_height=kwargs['command_height'])
    assert targets.contact.bool().all() and targets.valid.all() and targets.height_valid.all()
    torch.testing.assert_close(targets.support_height_relative_root,
                               .2 - ticks.state.position[..., 2], atol=2e-7, rtol=0)
    loss = sum(model.losses(prediction, targets).values())
    loss.backward()
    assert (model.net[-1].bias.grad[:3] < 0).all()  # Positive truth pushes negative logits upward.
    again = model.predict(ticks, **kwargs)
    torch.testing.assert_close(again.logits, prediction.logits, rtol=0, atol=0)


def test_tick_height_targets_follow_actual_four_substeps_not_policy_endpoint(tmp_path):
    model, ticks, kwargs = inputs(tmp_path)
    rising = ticks.state.position.clone()
    rising[..., 2] += torch.tensor([0., .01, .02, .03])
    actual = replace(ticks, state=replace(ticks.state, position=rising))
    data = record(model, model.predict(ticks, **kwargs), kwargs, ticks=actual)
    target = model.targets(data, command_height=kwargs['command_height'])
    torch.testing.assert_close(target.support_height_relative_root,
                               .2 - rising[..., 2], atol=2e-7, rtol=0)
    assert (target.support_height_relative_root[:, 0] != target.support_height_relative_root[:, 3]).all()
