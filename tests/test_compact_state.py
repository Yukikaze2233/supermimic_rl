"""Compact physical projection and observer checks, without starting Isaac."""

from dataclasses import replace
import math

import pytest
import torch

from supermimic_rl.adapters.isaac import RawSnapshot
from supermimic_rl.adapters.packed import PackedEnvironment
from supermimic_rl.compact_state import (
    CompactStateProjector, analytic_actor_observation, named_field,
)
from supermimic_rl.contracts import encoded_goal


BODY_NAMES = tuple("base_link" if index == 7 else f"body_{index}" for index in range(19))
JOINT_NAMES = tuple(f"joint_{index}" for index in range(18))
POLICY_INDICES = (5, 1, 12, 3, 17, 0)


@pytest.fixture
def raw():
    n = 3
    pose = torch.zeros(n, 19, 7)
    pose[..., 6] = 1
    axis = torch.tensor([[0., 0., 1.], [1., 0., 0.], [1., 2., 3.]])
    axis = axis / axis.norm(dim=-1, keepdim=True)
    angle = torch.tensor([math.pi / 2, -0.7, 1.1])[:, None]
    pose[:, 7, 3:] = torch.cat((axis * (angle / 2).sin(), (angle / 2).cos()), -1)
    pose[:, 7, :3] = torch.tensor([[100., 200., .4], [-300., 100., 1.6], [200., 400., 2.5]])
    com = pose.clone()
    offset = torch.tensor([[.1, .2, .3], [-.2, .3, -.1], [.4, -.1, .2]])
    com[:, 7, :3] += offset
    motion = torch.zeros(n, 19, 6)
    omega = torch.tensor([[1., 2., 3.], [2., -.5, 1.], [-.2, -.4, .6]])
    velocity = torch.tensor([[.2, .3, .1], [-.2, .1, .4], [.6, -.4, .2]])
    motion[:, 7, :3] = velocity + torch.linalg.cross(omega, offset)
    motion[:, 7, 3:] = omega
    nominal = torch.tensor([.2, -.3, .4, -.5, 0., 0.]).expand(n, -1).clone()
    relative = torch.tensor([
        [2 * math.pi + .2, 2 * math.pi + .5, -4 * math.pi - .4, -4 * math.pi + .6, 123., -88.],
        [3.3, 2.9, -3.2, -2.9, 200., -300.],
        [3.4, -2.8, .4, -.2, 400., -500.],
    ])
    q = torch.arange(n * 18).reshape(n, 18).float() * .01
    q[:, list(POLICY_INDICES)] = nominal + relative
    dq = torch.arange(n * 18).reshape(n, 18).float() * .1
    previous = torch.tensor([[.2, -.1, .4, -.3, 2., -3.]]).expand(n, -1).clone()
    state = {
        "body_link_pose_world_xyzw": pose,
        "body_com_pose_world_xyzw": com,
        "body_com_velocity_world": motion,
        "joint_position_native": q.remainder(4 * math.pi),
        "joint_position_continuous": q,
        "joint_velocity_native": dq,
        "continuous._previous_raw": q.clone(),
        "continuous._position": q.clone(),
        "continuous._turns": torch.zeros(n, 4),
        "terrain_contact_force_world_n": torch.ones(n, 19, 3),
        "clean_observation": torch.zeros(n, 35),
        "control._previous_action": previous,
        "control._last_policy_velocity": dq[:, list(POLICY_INDICES)].clone(),
        "target.admitted_leg_position_rad": q[:, list(POLICY_INDICES[:4])].clone(),
        "drive.state": torch.ones(n, 6),
        "transport.history": previous[:, None].repeat(1, 2, 1),
        "communication.held_effort": torch.full((n, 6), 2.),
        "latch.held": previous.clone(),
        "sensor.values": torch.full((n, 3, 18), 3.),
        "sensor.count": torch.full((n,), 3, dtype=torch.long),
        "reward._previous2": previous.clone(),
        "failure.contact_time_s": torch.tensor([0., .01, .02]),
        "failure.failure_reasons": torch.tensor([0, 0, 4]),
    }
    context = {
        "ground_at_root_world_z_m": torch.tensor([0., 1., 2.]),
        "environment_origin_world_m": pose[:, 7, :3].clone(),
        "controller.nominal": nominal,
        "policy_joint_indices": torch.tensor(POLICY_INDICES).expand(n, -1).float().clone(),
        "sensor.noise_halfwidth": torch.ones(n, 18),
        "sensor.lags": torch.ones(n, 4),
        "body_mass_kg": torch.ones(n, 19),
    }
    goal = torch.tensor([[.5, 0., .3, 1.5, 1., 0., 0., 0., 0., 0., .4]]).repeat(n, 1)
    return RawSnapshot(
        state, context, torch.zeros(n, 35), torch.full((n, 35), .75), goal,
        torch.arange(n), torch.arange(n) + 1, torch.ones(n, dtype=torch.bool),
        timestamp_s=torch.ones(n, dtype=torch.float64),
        field_metadata={"state": {"joint_velocity_native": {"units": "mixed", "frame": "native order"}},
                        "context": {"body_mass_kg": {"units": "kg", "frame": "native order"}}},
    )


def packed(raw):
    projected = CompactStateProjector(BODY_NAMES, JOINT_NAMES).project(raw)
    environment = PackedEnvironment(None, source="compact-test")
    snapshot = environment._pack(projected)
    return projected, snapshot, environment.schema


def observe(snapshot, schema, **kwargs):
    action = named_field(snapshot.state, schema, "control._previous_action")
    return analytic_actor_observation(snapshot.state, snapshot.context, snapshot.goal, action, schema, **kwargs)


def test_projection_preserves_memory_but_removes_redundant_body_and_observer_fields(raw):
    projected, snapshot, schema = packed(raw)
    assert all(name not in projected.state for name in CompactStateProjector._REMOVED)
    kept = set(raw.state) - CompactStateProjector._REMOVED
    for name in kept:
        torch.testing.assert_close(projected.state[name], raw.state[name])
        assert projected.state[name].data_ptr() != raw.state[name].data_ptr()
    assert set(projected.context) == set(raw.context)
    for name in raw.context:
        torch.testing.assert_close(projected.context[name], raw.context[name])
    for name in ("observation", "actor_observation", "goal", "episode_id", "control_epoch", "policy_owned", "timestamp_s"):
        torch.testing.assert_close(getattr(projected, name), getattr(raw, name))
    assert snapshot.state.shape[-1] < sum(value[0].numel() for value in raw.state.values()) - 400
    assert set(projected.field_metadata["state"]) == set(projected.state)
    assert projected.field_metadata["state"]["joint_velocity_native"]["units"] == "mixed"
    assert next(field for field in schema.state_fields if field.name == "root_orientation_xyzw").shape == (4,)
    assert "clean_observation" in raw.state


def test_root_motion_uses_named_link_origin_not_com_or_body_zero(raw):
    projected = CompactStateProjector(BODY_NAMES, JOINT_NAMES).project(raw)
    torch.testing.assert_close(projected.state["root_height_above_ground_m"], torch.tensor([[.4], [.6], [.5]]))
    # First row is a 90-degree yaw: world [x,y,z] becomes body [y,-x,z].
    torch.testing.assert_close(projected.state["root_linear_velocity_body_m_s"][0], torch.tensor([.3, -.2, .1]))
    torch.testing.assert_close(projected.state["root_angular_velocity_body_rad_s"][0], torch.tensor([2., -1., 3.]))
    shifted_state = {name: value.clone() for name, value in raw.state.items()}
    for name in ("body_link_pose_world_xyzw", "body_com_pose_world_xyzw"):
        shifted_state[name][..., :3] += torch.tensor([800., -900., 5.])
    shifted_context = dict(raw.context)
    shifted_context["ground_at_root_world_z_m"] = raw.context["ground_at_root_world_z_m"] + 5
    shifted = CompactStateProjector(BODY_NAMES, JOINT_NAMES).project(replace(raw, state=shifted_state, context=shifted_context))
    for name in CompactStateProjector._ROOT_METADATA:
        torch.testing.assert_close(shifted.state[name], projected.state[name], atol=1e-4, rtol=1e-4)


def test_full_vehicle_auxiliary_axes_and_exogenous_commands_are_not_fabricated(raw):
    assert not any(name.startswith("gimbal.") for name in packed(raw)[0].state)
    for name, value in (("position_rad", 1.), ("velocity_rad_s", 2.), ("effort_nm", 3.)):
        raw.state["gimbal." + name] = torch.full((3, 2), value)
    raw.context["gimbal.command"] = torch.tensor([[.5, -.2]]).repeat(3, 1)
    projected, _, _ = packed(raw)
    for name in ("gimbal.velocity_rad_s", "gimbal.effort_nm"):
        torch.testing.assert_close(projected.state[name], raw.state[name])
    torch.testing.assert_close(projected.state["gimbal.pitch_position_rad"], raw.state["gimbal.position_rad"][:, 1:])
    phase = projected.state["gimbal.yaw_sin_cos"]
    torch.testing.assert_close(phase.norm(dim=-1), torch.ones(3))
    raw.state["gimbal.position_rad"][:, 0] += 4 * math.pi
    wrapped, _, _ = packed(raw)
    torch.testing.assert_close(wrapped.state["gimbal.yaw_sin_cos"], phase, atol=1e-6, rtol=0)
    torch.testing.assert_close(projected.context["gimbal.command"], raw.context["gimbal.command"])
    assert projected.state["joint_position_continuous"].shape == (3, 18)


@pytest.mark.parametrize("fault", ["ground", "joint", "quaternion", "root_name", "duplicate_names"])
def test_invalid_physical_source_is_explicitly_rejected(raw, fault):
    names = BODY_NAMES
    if fault == "ground":
        del raw.context["ground_at_root_world_z_m"]
    elif fault == "joint":
        raw.state["joint_position_continuous"] = torch.zeros(3, 17)
    elif fault == "quaternion":
        raw.state["body_link_pose_world_xyzw"][:, 7, 3:] = 0
    elif fault == "root_name":
        names = tuple(f"other_{i}" for i in range(19))
    else:
        names = ("base_link",) * 19
    with pytest.raises(ValueError):
        CompactStateProjector(names, JOINT_NAMES).project(raw)


def test_clean_observer_matches_real_upstream_measure_and_actor(raw):
    upstream = pytest.importorskip("transformer_rl.formal.control.observation")
    binding = upstream.MechanicalBinding(
        JOINT_NAMES, BODY_NAMES, JOINT_NAMES, POLICY_INDICES, tuple(range(18)),
        (0, 1), (17, 0), tuple(raw.context["controller.nominal"][0].tolist()),
    )
    assembler = upstream.ObservationAssembler(binding, 3)
    sample = upstream.NativePhysicalSample(
        raw.state["joint_position_continuous"], raw.state["joint_velocity_native"],
        raw.state["body_link_pose_world_xyzw"][:, 7, :3],
        raw.state["body_link_pose_world_xyzw"][:, 7, 3:],
        raw.state["body_com_pose_world_xyzw"][:, 7, :3],
        raw.state["body_com_velocity_world"][:, 7, :3],
        raw.state["body_com_velocity_world"][:, 7, 3:],
        raw.state["terrain_contact_force_world_n"], raw.context["ground_at_root_world_z_m"],
        torch.zeros(3, 2), raw.context["environment_origin_world_m"], torch.ones(3, 2), raw.timestamp_s,
    )
    commands = upstream.CommandFeatures(
        torch.tensor([[.5, 0., .3, .3]]).repeat(3, 1), torch.tensor([0, 4, 2]),
        torch.tensor([0., .2, 0.]), torch.tensor([.4, 1., 8.]), raw.state["control._previous_action"],
    )
    measured = assembler.measure(sample)
    expected = assembler.actor(assembler.clean_sensors(measured), commands, now_s=raw.timestamp_s)
    raw = replace(raw, observation=expected, goal=encoded_goal(expected))
    _, snapshot, schema = packed(raw)
    torch.testing.assert_close(observe(snapshot, schema), expected)
    torch.testing.assert_close(observe(snapshot, schema, policy_joint_indices=POLICY_INDICES,
                                       nominal=raw.context["controller.nominal"][0]), expected)
    assert not torch.equal(expected, snapshot.actor_observation)


def test_common_pair_winding_preserves_relative_phase_and_zeros_wheel_position(raw):
    _, snapshot, schema = packed(raw)
    frame = observe(snapshot, schema)
    torch.testing.assert_close(frame[0, 10:14], torch.tensor([.2, .5, -.4, .6]), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(frame[2, 10:14], torch.tensor([3.4, -2.8, .4, -.2]))
    assert torch.equal(frame[:, 14:16], torch.zeros(3, 2))
    torch.testing.assert_close(frame[:, :4], snapshot.goal[:, :4])
    torch.testing.assert_close(frame[:, 28:], snapshot.goal[:, 4:])
    torch.testing.assert_close(frame[:, 22:28], raw.state["control._previous_action"])


def test_missing_previous_action_uses_native_issued_action_memory(raw):
    _, snapshot, schema = packed(raw)
    expected = observe(snapshot, schema)
    actual = analytic_actor_observation(snapshot.state, snapshot.context, snapshot.goal, None, schema)
    torch.testing.assert_close(actual, expected)
    replacement = torch.full((3, 6), .5)
    explicit = analytic_actor_observation(snapshot.state, snapshot.context, snapshot.goal, replacement, schema)
    torch.testing.assert_close(explicit[:, 22:28], replacement)
    torch.testing.assert_close(actual[:, 22:28], raw.state["control._previous_action"])


def test_observer_keeps_autograd_through_physics_goal_and_previous_action(raw):
    _, snapshot, schema = packed(raw)
    state = snapshot.state.double().requires_grad_()
    context = snapshot.context.double()
    goal = snapshot.goal.double().requires_grad_()
    action = raw.state["control._previous_action"].double().requires_grad_()
    result = analytic_actor_observation(state, context, goal, action, schema)
    result.square().sum().backward()
    assert torch.isfinite(state.grad).all()
    for name in ("root_orientation_xyzw", "root_angular_velocity_body_rad_s",
                 "joint_position_continuous", "joint_velocity_native"):
        assert named_field(state.grad, schema, name).abs().sum() > 0
    torch.testing.assert_close(named_field(state.grad, schema, "joint_position_continuous")[:, list(POLICY_INDICES[4:])],
                               torch.zeros(3, 2, dtype=torch.float64))
    assert goal.grad.abs().sum() > 0 and action.grad.abs().sum() > 0
    assert torch.autograd.gradcheck(
        lambda s, a: analytic_actor_observation(s, context[:1], goal[:1].detach(), a, schema),
        (state[:1].detach().requires_grad_(), action[:1].detach().requires_grad_()),
        fast_mode=True,
    )


def test_observer_supports_sequence_batches_and_per_row_native_orders(raw):
    _, snapshot, schema = packed(raw)
    state = snapshot.state[:, None].repeat(1, 2, 1)
    context = snapshot.context[:, None].repeat(1, 2, 1)
    goal = snapshot.goal[:, None].repeat(1, 2, 1)
    action = raw.state["control._previous_action"][:, None].repeat(1, 2, 1)
    result = analytic_actor_observation(state, context, goal, action, schema)
    torch.testing.assert_close(result, observe(snapshot, schema)[:, None].repeat(1, 2, 1))
    swapped = raw.context["policy_joint_indices"].clone()
    swapped[1] = swapped[1].flip(0)
    result = analytic_actor_observation(snapshot.state, snapshot.context, snapshot.goal,
                                      raw.state["control._previous_action"], schema, swapped)
    expected_dq = raw.state["joint_velocity_native"].gather(-1, swapped.long()) * .1
    torch.testing.assert_close(result[:, 16:22], expected_dq)


def test_compact_world_model_uses_last_axis_for_quaternion_and_observer(raw):
    from supermimic_rl.models import WorldModel

    _, snapshot, schema = packed(raw)
    model = WorldModel(schema, hidden=(8, 8))
    assert model.observation_head is None
    model.state_normalizer.fit(snapshot.state)
    model.context_normalizer.fit(snapshot.context)
    model.observation_normalizer.fit(observe(snapshot, schema))
    with torch.no_grad():
        # The intentional zero residual initialization has no action Jacobian.
        model.dynamics[-1].weight.normal_(0., .002, generator=torch.Generator().manual_seed(23))
    state = snapshot.state[:, None].repeat(1, 2, 1)
    named_field(state, schema, "root_orientation_xyzw").mul_(2)
    state.requires_grad_()
    context = snapshot.context[:, None].repeat(1, 2, 1)
    goal = snapshot.goal[:, None].repeat(1, 2, 1)
    action = torch.zeros(3, 2, 6, requires_grad=True)
    torch.testing.assert_close(model.observe(state, goal, context),
                               observe(snapshot, schema)[:, None].repeat(1, 2, 1))
    predicted = model.step(state, action, goal, context)
    quaternion = named_field(predicted.state, schema, "root_orientation_xyzw")
    torch.testing.assert_close(quaternion.norm(dim=-1), torch.ones(3, 2))
    observed = model.observe(predicted.state, goal, context)
    assert observed.shape == (3, 2, 35)
    observed.square().mean().backward()
    assert torch.isfinite(state.grad).all() and torch.isfinite(action.grad).all()
    assert state.grad.abs().sum() > 0 and action.grad.abs().sum() > 0


@pytest.mark.parametrize("indices", [(0, 0, 1, 2, 3, 4), (0, 1, 2, 3, 4, 18),
                                      (0, 1, 2, 3, 4, -1), (0, 1, 2, 3, 4, 5.5)])
def test_invalid_policy_binding_is_rejected(raw, indices):
    _, snapshot, schema = packed(raw)
    with pytest.raises(ValueError, match="policy_joint_indices"):
        observe(snapshot, schema, policy_joint_indices=indices)


def test_nonfinite_values_are_rejected_before_clamp_and_legal_outputs_are_retained(raw):
    _, snapshot, schema = packed(raw)
    omega = named_field(snapshot.state, schema, "root_angular_velocity_body_rad_s")
    omega[0] = torch.tensor([300., -300., 48.])
    frame = observe(snapshot, schema)
    torch.testing.assert_close(frame[0, 4:7], torch.tensor([100., -100., 24.]))
    omega[0, 0] = float("inf")
    with pytest.raises(ValueError, match="nonfinite"):
        observe(snapshot, schema)


def test_required_named_fields_do_not_fall_back_to_opaque_offsets(raw):
    _, snapshot, schema = packed(raw)
    missing = replace(schema, context_fields=tuple(field for field in schema.context_fields
                                                  if field.name != "policy_joint_indices"))
    context = missing.pack({name: value for name, value in raw.context.items()
                            if name != "policy_joint_indices"}, context=True)
    with pytest.raises(ValueError, match="policy_joint_indices"):
        analytic_actor_observation(snapshot.state, context, snapshot.goal,
                                   raw.state["control._previous_action"], missing)
    with pytest.raises(ValueError, match="missing"):
        named_field(snapshot.state, schema, "unknown")
    with pytest.raises(ValueError, match="width"):
        named_field(snapshot.state[:, :-1], schema, "root_orientation_xyzw")
