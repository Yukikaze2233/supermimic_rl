"""Test core/aux separation with reordered synthetic native buffers, without Kit."""


import pytest
import torch

from supermimic_rl.adapters.core_projection import CoreArticulationProjection
from support.core_projection import (
    CORE_BODIES, CORE_JOINTS, NativeArray, NativeArticulation, NativeView, NewtonNativeArticulation, to_native,
    to_tensor,
)


def project(native):
    return CoreArticulationProjection(
        native, CORE_BODIES, CORE_JOINTS,
        native_to_tensor=to_tensor, tensor_to_native=to_native)


@pytest.fixture
def pair():
    native = NativeArticulation()
    return native, project(native)


def test_construction_is_lazy_and_public_and_backend_orders_are_distinct():
    native = NativeArticulation(ready=False)
    core = project(native)
    view = core.root_view
    data = core.data
    assert core.native is native
    assert core.num_bodies == core.num_joints == 3
    assert not native.shape_requests
    with pytest.raises(AssertionError, match="not initialized"):
        _ = core.body_names
    native.ready = True
    assert core.root_view is view and core.data is data
    assert core.body_names == core.backend_body_names == CORE_BODIES
    assert core.joint_names == CORE_JOINTS
    assert core.num_instances == 3 and core.device == torch.device("cpu")
    assert core._physics_sim_view is native._physics_sim_view
    assert view.link_paths == [[f"/env_{row}/{name}" for name in CORE_BODIES]
                               for row in range(3)]
    assert torch.equal(data.body_link_pose_w, native.data.body_link_pose_w[:, [2, 4, 1]])
    assert torch.equal(data.joint_pos, native.data.joint_pos[:, [2, 4, 0]])
    mass = view.get_masses()
    assert isinstance(mass, NativeArray) and mass.shape == (3, 3)
    assert torch.equal(mass.tensor, native.root_view.buffers["masses"][:, [2, 0, 3]])


@pytest.mark.parametrize("name", [
    "body_link_pose_w", "body_com_pose_w", "body_com_vel_w", "body_lin_vel_w",
    "joint_pos", "joint_vel", "default_joint_pos",
])
def test_data_reads_refresh_from_native_buffers(pair, name):
    native, core = pair
    before = getattr(core.data, name).clone()
    getattr(native.data, name).add_(123)
    assert torch.equal(getattr(core.data, name), before + 123)


def test_com_torch_accessor_and_root_identity(pair):
    native, core = pair
    com = core.data.body_com_pose_b.torch
    assert torch.equal(com, native.root_view.buffers["coms"][:, [2, 0, 3]])
    for name in ("root_link_pos_w", "root_link_quat_w", "root_com_vel_w"):
        assert getattr(core.data, name) is getattr(native.data, name)
    with pytest.raises(AttributeError, match="does not expose"):
        _ = core.data.unlisted_axis_dependent_field


@pytest.mark.parametrize("name,tail", [("masses", ()), ("coms", (7,)), ("inertias", (9,))])
@pytest.mark.parametrize("full_data", [False, True])
def test_public_property_writes_select_rows_and_core_columns_only(pair, name, tail, full_data):
    native, core = pair
    rows, columns = torch.tensor([2, 0]), torch.tensor([2, 0])
    shape = ((3, 3) if full_data else (2, 2)) + tail
    value = torch.arange(torch.tensor(shape).prod()).float().reshape(shape) + 1000
    selected = value[rows[:, None], columns[None, :]] if full_data else value
    expected = native.root_view.buffers[name].clone()
    expected[rows[:, None], torch.tensor([3, 2])[None, :]] = selected
    getattr(core, f"set_{name}_index")(
        **{name: value}, body_ids=columns, env_ids=rows, full_data=full_data)
    assert torch.equal(native.root_view.buffers[name], expected)
    assert native.public_calls[-1][1].tolist() == [1, 2]
    assert native.public_calls[-1][3] is False
    assert not native.root_view.writes
    if name == "coms":
        assert torch.equal(core.data.body_com_pose_b.torch, expected[:, [2, 0, 3]])


@pytest.mark.parametrize("name,tail", [("masses", ()), ("coms", (7,)), ("inertias", (9,))])
def test_raw_property_round_trip_preserves_aux_and_other_rows(pair, name, tail):
    native, core = pair
    value = torch.full((3, 3) + tail, 1000.)
    expected = native.root_view.buffers[name].clone()
    expected[1, [2, 0, 3]] = 1000
    getattr(core.root_view, f"set_{name}")(NativeArray(value), NativeArray(torch.tensor([1])))
    assert torch.equal(native.root_view.buffers[name], expected)
    result = getattr(core.root_view, f"get_{name}")()
    assert result.shape == (3, 3) + tail
    assert torch.equal(result.tensor, expected[:, [2, 0, 3]])


def test_material_columns_use_true_per_link_shape_counts(pair):
    native, _ = pair
    core = CoreArticulationProjection(
        native, ("leg", "base", "wheel"), CORE_JOINTS,
        native_to_tensor=to_tensor, tensor_to_native=to_native)
    view = core.root_view
    assert view.max_shapes == 5
    assert len(native.shape_requests) == 5
    # Full backend: wheel(2), auxYaw(1), base(0), leg(3), auxPitch(2).
    expected_columns = [3, 4, 5, 0, 1]
    source = native.root_view.buffers["material_properties"].clone()
    assert torch.equal(view.get_material_properties().tensor, source[:, expected_columns])
    assert len(native.shape_requests) == 5
    value = view.get_material_properties().tensor.clone()
    value[2] += 1000
    view.set_material_properties(NativeArray(value), NativeArray(torch.tensor([2])))
    expected = source.clone()
    expected[2, expected_columns] += 1000
    assert torch.equal(native.root_view.buffers["material_properties"], expected)


def test_material_topology_inconsistency_is_rejected(pair):
    native, core = pair
    native.root_view.max_shapes += 1
    with pytest.raises(ValueError, match="shape counts"):
        _ = core.root_view.max_shapes


@pytest.mark.parametrize("name,field", [("position", "joint_pos"), ("velocity", "joint_vel")])
@pytest.mark.parametrize("full_data", [False, True])
def test_joint_reset_writes_preserve_aux_and_unselected_core(pair, name, field, full_data):
    native, core = pair
    rows, columns = torch.tensor([2, 0]), torch.tensor([2, 1])
    value = (torch.arange(9).float().reshape(3, 3) if full_data
             else torch.arange(4).float().reshape(2, 2)) + 1000
    selected = value[rows[:, None], columns[None, :]] if full_data else value
    expected = getattr(native.data, field).clone()
    expected[rows[:, None], torch.tensor([0, 4])[None, :]] = selected
    getattr(core, f"write_joint_{name}_to_sim_index")(
        **{name: value}, env_ids=rows, joint_ids=columns, full_data=full_data)
    assert torch.equal(getattr(native.data, field), expected)
    assert native.public_calls[-1][3] is False


def test_default_joint_target_and_reset_keep_aux_commands(pair):
    native, core = pair
    previous = native.effort.clone()
    core.set_joint_effort_target(torch.full((3, 3), 200.))
    expected = previous.clone()
    expected[:, [2, 4, 0]] = 200
    assert torch.equal(native.effort, expected)
    core.set_joint_effort_target(torch.tensor([[300.]]), joint_ids=[1], env_ids=[2])
    expected[2, 4] = 300
    assert torch.equal(native.effort, expected)
    before = native.data.joint_pos.clone()
    core.write_joint_position_to_sim_index(position=torch.zeros(1, 3), env_ids=[1])
    before[1, [2, 4, 0]] = 0
    assert torch.equal(native.data.joint_pos, before)


def test_wrench_addition_maps_public_bodies_without_clearing_aux(pair):
    native, core = pair
    wrench = native.instantaneous_wrench_composer
    previous_forces, previous_torques = wrench.forces.clone(), wrench.torques.clone()
    force, torque, positions = torch.full((2, 2, 3), 3.), torch.full((2, 2, 3), 4.), torch.ones(2, 2, 3)
    core.instantaneous_wrench_composer.add_forces_and_torques_index(
        forces=force, torques=torque, positions=positions, body_ids=[2, 0],
        env_ids=[2, 0], is_global=True)
    rows, columns = torch.tensor([2, 0]), torch.tensor([1, 2])
    previous_forces[rows[:, None], columns[None, :]] += 3
    previous_torques[rows[:, None], columns[None, :]] += 4
    assert torch.equal(wrench.forces, previous_forces)
    assert torch.equal(wrench.torques, previous_torques)
    assert torch.equal(wrench.calls[-1]["positions"], positions)
    assert wrench.calls[-1]["is_global"] is True
    core.instantaneous_wrench_composer.add_forces_and_torques_index(
        torques=torch.ones(3, 3, 3))
    previous_torques[:, [2, 4, 1]] += 1
    assert torch.equal(wrench.torques, previous_torques)
    assert torch.equal(wrench.forces, previous_forces)


def test_root_writes_are_forwarded_without_reinterpreting_root_com(pair):
    native, core = pair
    pose, velocity, ids = torch.ones(1, 7), torch.ones(1, 6), torch.tensor([2])
    core.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=ids)
    core.write_root_com_velocity_to_sim_index(root_velocity=velocity, env_ids=ids)
    assert native.public_calls[0][1]["root_pose"] is pose
    assert native.public_calls[1][1]["root_velocity"] is velocity
    assert native.public_calls[1][1]["env_ids"] is ids


@pytest.mark.parametrize("ids", [[-1], [3], [0, 0], [1.5], torch.tensor([True])])
def test_invalid_projected_indices_fail_before_native_mutation(pair, ids):
    native, core = pair
    with pytest.raises(ValueError, match="indices"):
        core.set_masses_index(masses=torch.ones(3, len(ids)), body_ids=ids)
    assert not native.public_calls and not native.root_view.writes


def test_wrong_full_data_shape_and_dtype_fail_before_native_mutation(pair):
    native, core = pair
    with pytest.raises(ValueError, match="shape"):
        core.set_masses_index(masses=torch.ones(1, 3), env_ids=[0], full_data=True)
    with pytest.raises(ValueError, match="float32"):
        core.set_joint_effort_target(torch.zeros(3, 3, dtype=torch.float64))
    assert not native.public_calls


def test_empty_selections_are_valid_and_keep_every_native_value(pair):
    native, core = pair
    before = native.root_view.buffers["masses"].clone()
    core.set_masses_index(masses=torch.empty(0, 3), env_ids=[])
    assert torch.equal(native.root_view.buffers["masses"], before)


def test_replaced_native_view_is_not_silently_rebound(pair):
    native, core = pair
    _ = core.body_names
    native._root_view = NativeView(native)
    with pytest.raises(RuntimeError, match="view changed"):
        _ = core.data.joint_pos


def test_missing_duplicate_or_inconsistent_names_fail_explicitly():
    native = NativeArticulation()
    with pytest.raises(ValueError, match="unique"):
        CoreArticulationProjection(native, ["base", "base"], CORE_JOINTS)
    core = CoreArticulationProjection(native, ["missing"], CORE_JOINTS)
    with pytest.raises(ValueError, match="absent"):
        _ = core.body_names
    core = project(native)
    native._root_view.link_paths[1] = list(reversed(native._root_view.link_paths[1]))
    with pytest.raises(ValueError, match="disagree"):
        _ = core.body_names


def test_newton_projection_uses_public_names_for_mechanics_reset_effort_and_wrenches():
    native = NewtonNativeArticulation()
    core = CoreArticulationProjection(native, CORE_BODIES, CORE_JOINTS, mechanics_backend="newton")
    assert core.body_names == core.backend_body_names == CORE_BODIES
    assert core.joint_names == CORE_JOINTS
    assert core.num_instances == 3
    for name in ("body_mass", "body_com_pos_b", "body_inertia"):
        assert torch.equal(getattr(core.data, name), getattr(native.data, name)[:, [2, 4, 1]])
    assert torch.equal(core.data.body_com_pose_b.torch, native.data.body_com_pose_b.torch[:, [2, 4, 1]])
    q_before, effort_before = native.data.joint_pos.clone(), native.effort.clone()
    core.write_joint_position_to_sim_index(position=torch.tensor([[123.]]), joint_ids=[1], env_ids=[2])
    q_before[2, 4] = 123
    assert torch.equal(native.data.joint_pos, q_before)
    core.set_joint_effort_target(torch.full((3, 3), 9.))
    effort_before[:, [2, 4, 0]] = 9
    assert torch.equal(native.effort, effort_before)
    before = native.instantaneous_wrench_composer.forces.clone()
    core.instantaneous_wrench_composer.add_forces_and_torques_index(
        forces=torch.ones(1, 2, 3), body_ids=[2, 0], env_ids=[1], is_global=True)
    before[1, [1, 2]] += 1
    assert torch.equal(native.instantaneous_wrench_composer.forces, before)


def test_newton_projection_rejects_physx_properties_and_mechanical_randomization():
    core = CoreArticulationProjection(
        NewtonNativeArticulation(), CORE_BODIES, CORE_JOINTS, mechanics_backend="newton")
    with pytest.raises(NotImplementedError, match="link_paths"):
        _ = core.root_view.link_paths
    with pytest.raises(NotImplementedError, match="Articulation.data"):
        core.root_view.get_masses()
    with pytest.raises(NotImplementedError, match="randomization"):
        core.set_masses_index(masses=torch.ones(3, 3))
    with pytest.raises(NotImplementedError, match="material"):
        _ = core.root_view.max_shapes


@pytest.mark.parametrize("name,field", [("position", "joint_pos"), ("velocity", "joint_vel")])
@pytest.mark.parametrize("full_data", [False, True])
@pytest.mark.parametrize("skip_forward", [False, True])
def test_newton_joint_writes_follow_partial_only_sdk_signature(name, field, full_data, skip_forward):
    native = NewtonNativeArticulation()
    core = CoreArticulationProjection(native, CORE_BODIES, CORE_JOINTS, mechanics_backend="newton")
    rows, columns = torch.tensor([2, 0]), torch.tensor([2, 1])
    value = (torch.arange(9).float().reshape(3, 3) if full_data
             else torch.arange(4).float().reshape(2, 2)) + 1000
    selected = value[rows[:, None], columns[None, :]] if full_data else value
    expected = getattr(native.data, field).clone()
    expected[rows[:, None], torch.tensor([0, 4])[None, :]] = selected
    getattr(core, f"write_joint_{name}_to_sim_index")(
        **{name: value}, env_ids=rows, joint_ids=columns,
        full_data=full_data, skip_forward=skip_forward)
    assert torch.equal(getattr(native.data, field), expected)
    assert native.public_calls[-1][3] is skip_forward


@pytest.mark.parametrize("skip_forward", [False, True])
def test_newton_root_writes_preserve_native_partial_rows_and_cache_flag(skip_forward):
    native = NewtonNativeArticulation()
    core = CoreArticulationProjection(native, CORE_BODIES, CORE_JOINTS, mechanics_backend="newton")
    pose, velocity, ids = torch.ones(1, 7), torch.ones(1, 6), torch.tensor([2])
    core.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=ids, skip_forward=skip_forward)
    core.write_root_com_velocity_to_sim_index(root_velocity=velocity, env_ids=ids,
                                            skip_forward=skip_forward)
    assert native.public_calls[0][1]["root_pose"] is pose
    assert native.public_calls[1][1]["root_velocity"] is velocity
    for _, kwargs in native.public_calls:
        assert kwargs["env_ids"] is ids
        assert kwargs["skip_forward"] is skip_forward
