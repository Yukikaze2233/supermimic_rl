"""Parity with frozen native-training algorithms and selected-asset geometry."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest
import torch

from supermimic_rl.course.identified_execution import ACTION_ORDER, IdentifiedExecution


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "configs/identified/20261031"
ASSET = ROOT / "assets/full_vehicle_v2p0p1_source_audited_v2"


def response(count=2, dt=.005, mode="native_floor", dtype=torch.float32, **kwargs):
    return IdentifiedExecution(count, "cpu", dt, asset_directory=ASSET,
                               bundle_directory=BUNDLE, friction_mode=mode, dtype=dtype, **kwargs)


def reference_module(name):
    spec = importlib.util.spec_from_file_location(name, BUNDLE / "reference_source" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dt", [.001, .0025, .005, .01])
def test_wheel_delay_envelope_and_residual_match_frozen_training_source(dt):
    migrated = response(count=7, dt=dt)
    original = reference_module("identified_wheel_response").IdentifiedWheelResponse(
        7, "cpu", dt, dict(enabled=True, model_path="wheel_model.json",
                          model_sha256=migrated.receipt()["model_pins"]["wheel_model.json"],
                          apply_aggregate_delay=True), root=BUNDLE,
    )
    generator = torch.Generator().manual_seed(31)
    # Source is hip, auxiliary, wheel per side; formal order is four legs then wheels.
    source_order = [0, 1, 4, 2, 3, 5]
    inverse = [0, 1, 3, 4, 2, 5]
    for step in range(40):
        torque = torch.randn(7, 6, generator=generator) * 10
        velocity = torch.randn(7, 6, generator=generator) * 50
        if step == 17:
            migrated.reset(torch.tensor([1, 5]))
            original.reset(torch.tensor([1, 5]))
        actual = migrated.step(torque, velocity)
        expected = original.apply_torque(torque[:, source_order], velocity[:, source_order])[:, inverse]
        torch.testing.assert_close(actual.motor_effort_nm[:, 4:], expected[:, 4:], rtol=0, atol=0)
        torch.testing.assert_close(actual.passive_effort_nm[:, 4:], original.passive_effort, rtol=0, atol=0)
        torch.testing.assert_close(actual.wheel_motor_bound_nm, original.drive_bound, rtol=0, atol=0)


def test_fractional_delay_is_exactly_one_total_delay_not_two_lags():
    migrated = response(count=1)
    effort = torch.zeros(1, 6)
    effort[:, 4:] = 1
    zero = torch.zeros_like(effort)
    values = [migrated.step(effort if i == 0 else zero, zero).motor_effort_nm[0, 4:]
              for i in range(4)]
    torch.testing.assert_close(torch.stack(values), torch.tensor([[0., .3], [.9, .7], [.1, 0.], [0., 0.]]))


def test_software_total_dynamic_mode_is_passive_and_explicitly_has_no_static_model():
    migrated = response(count=3, mode="software_dynamic")
    velocity = torch.tensor([[.001, -.001, .1, -.1, 20., -20.],
                             [.05, -.05, 4., -4., -6., 6.], [0.] * 6])
    output = migrated.step(torch.zeros_like(velocity), velocity)
    assert torch.all(output.passive_effort_nm * velocity <= 0)
    assert torch.equal(output.passive_effort_nm[-1], torch.zeros(6))
    assert output.passive_effort_nm[0, 4] == pytest.approx(-(.055020881181436186 + .000345880302504356))
    assert output.passive_effort_nm[0, 5] == pytest.approx(.21075946885507976 - .007037539135300832, rel=1e-4)
    assert migrated.receipt()["static_breakaway_reproduced"] is False
    for row in migrated.native_parameters().values():
        assert row["friction"] == row["dynamic_friction"] == row["viscous_friction"] == 0


def test_braking_cap_remains_available_above_motoring_zero_speed():
    migrated = response(count=2)
    torque = torch.tensor([[0., 0., 0., 0., 4.5, -4.5], [0., 0., 0., 0., -4.5, 4.5]])
    velocity = torch.tensor([[0., 0., 0., 0., 80., -80.], [0., 0., 0., 0., 80., -80.]])
    for _ in range(4):
        output = migrated.step(torque, velocity)
    assert torch.equal(output.motor_effort_nm[0, 4:], torch.zeros(2))
    torch.testing.assert_close(output.motor_effort_nm[1, 4:], torch.tensor([-4.5, 4.5]))


def test_disable_clears_queued_motor_commands_but_preserves_passive_loss():
    migrated = response(count=2)
    effort, speed = torch.ones(2, 6), torch.ones(2, 6)
    for _ in range(4):
        migrated.step(effort, speed)
    enabled = torch.tensor([False, True])
    output = migrated.step(effort, speed, motor_enabled=enabled)
    assert not output.motor_effort_nm[0].any()
    assert (output.passive_effort_nm[0] < 0).all()
    assert not migrated.state_memory()["wheel_torque_history_nm"][0].any()
    assert migrated.state_memory()["wheel_torque_history_nm"][1].all()
    output = migrated.step(effort, speed)
    torch.testing.assert_close(output.motor_effort_nm[0, 4:], torch.tensor([0., .3]))
    snapshot = migrated.state_memory()["wheel_torque_history_nm"]
    migrated.reset()
    assert snapshot.any() and not migrated.state_memory()["wheel_torque_history_nm"].any()


def test_leg_friction_matches_frozen_source_without_legacy_lag_or_viscosity(tmp_path):
    original = reference_module("leg_motor_friction")
    for source, destination in (("leg_model.json", "model.json"),
                                ("leg_parameters.json", "parameters.json"),
                                ("leg_README.md", "README.md")):
        shutil.copyfile(BUNDLE / source, tmp_path / destination)
    settings = dict(enabled=True, model_path="model.json", model_sha256=original.digest(tmp_path / "model.json"))
    reference = original.LegMotorFriction(4, "cpu", settings, root=tmp_path)
    migrated = response(count=4)
    velocity = torch.tensor([[.01, -.01, .04, -.04], [.05, -.05, 2., -2.],
                             [0.] * 4, [-4., 4., -3., 3.]])
    six = torch.cat((velocity, torch.zeros(4, 2)), -1)
    output = migrated.step(torch.zeros_like(six), six)
    torch.testing.assert_close(output.passive_effort_nm[:, :4], reference.effort(velocity), rtol=0, atol=0)
    assert list(migrated.native_parameters()) == list(ACTION_ORDER)
    for name in ACTION_ORDER[:4]:
        assert all(value == 0 for value in migrated.native_parameters()[name].values())


def test_selected_wheel_axis_inertia_is_rebound_to_identified_total():
    migrated = response(dtype=torch.float64)
    receipt = migrated.receipt()
    for k, name in enumerate(ACTION_ORDER[4:]):
        assert receipt["wheel_cad_axis_inertia_kg_m2"][k] == pytest.approx(.00058325, abs=1e-10)
        assert receipt["native_parameters"][name]["armature"] + receipt["wheel_cad_axis_inertia_kg_m2"][k] == pytest.approx(
            migrated.wheel_rows[k]["effective_total_inertia_kg_m2"], abs=1e-14)
        assert receipt["native_parameters"][name]["armature"] != pytest.approx(
            .003694571255748743, abs=1e-8)


def test_gas_curve_matches_original_measured_segments_and_endpoint_prior():
    migrated = response(dtype=torch.float64)
    module = reference_module("gas_spring_compensation")
    reference = object.__new__(module.GasSpringCompensation)
    # The source constructor binds a canonical encoder table. Physical force
    # depends only on these frozen buffers, independent of that table's frame.
    reference.model = migrated.spring_model
    reference.compression = migrated.spring_compression
    reference.force_n = migrated.spring_force
    reference.geometry_compression_min, reference.geometry_compression_max = migrated.geometry_compression_bounds
    compression = torch.cat((torch.linspace(-.02, .12, 200, dtype=torch.float64), migrated.spring_compression))
    expected, outside = reference._evaluate_force(compression)
    actual = migrated.physical_spring_force(compression)
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-12)
    torch.testing.assert_close(migrated.physical_spring_force(migrated.spring_compression), migrated.spring_force)
    assert (actual > 0).all() and outside.any()
    q = migrated.spring_zero.expand(2, -1) - torch.tensor([[.03, .04], [.05, .06]], dtype=torch.float64)
    torch.testing.assert_close(migrated.slider_effort(q), migrated.physical_spring_force(migrated.spring_zero - q))


def test_actual_mount_wrenches_conserve_net_force_and_give_correct_virtual_work():
    migrated = response(count=1, dtype=torch.float64)
    body_names = tuple(reversed(list(migrated._bodies)))
    spring = migrated.mount_force_element(body_names)
    pose = torch.zeros(1, len(body_names), 7, dtype=torch.float64)
    pose[..., 6] = 1
    # Specify a known finite mounting distance, accounting for source local points.
    target_mounts = torch.tensor([[.18, 0., 0.], [0., 0., 0.],
                                  [.2, .4, 0.], [0., .4, 0.]], dtype=torch.float64)
    for index, local, target in zip(spring.body_ids, spring._points, target_mounts):
        pose[0, index, :3] = target - local
    com = pose[..., :3].clone()
    result = spring.evaluate(pose, com, torch.ones(1, 1, dtype=torch.float64))
    torch.testing.assert_close(result.forces_world.sum(1), torch.zeros(1, 3, dtype=torch.float64))
    net_torque = (result.torques_about_com_world
                  + torch.linalg.cross(com[:, spring.body_ids], result.forces_world)).sum(1)
    torch.testing.assert_close(net_torque, torch.zeros_like(net_torque), atol=1e-12, rtol=0)
    upper = spring.body_ids[0]
    epsilon = 1e-7
    altered = pose.clone()
    altered[0, upper, 3:7] = torch.tensor([0., 0., torch.sin(torch.tensor(epsilon / 2)).item(),
                                          torch.cos(torch.tensor(epsilon / 2)).item()], dtype=torch.float64)
    changed = spring.evaluate(altered, com, torch.ones(1, 1, dtype=torch.float64))
    numerical_jacobian = (changed.mount_distance_m[0, 0] - result.mount_distance_m[0, 0]) / epsilon
    assert result.torques_about_com_world[0, 0, 2] == pytest.approx(
        float(result.pressure_force_n[0, 0] * numerical_jacobian), abs=2e-5)
    # The physical torque extends the spring; holding compensation has opposite sign.
    assert migrated.receipt()["encoder_holding_compensation"] is False


def test_changed_model_pin_and_nonfinite_input_reject_before_history_mutation(tmp_path):
    shutil.copytree(BUNDLE, tmp_path / "bundle")
    wheel = tmp_path / "bundle/wheel_model.json"
    wheel.write_text(wheel.read_text() + " ")
    with pytest.raises(ValueError, match="source differs"):
        IdentifiedExecution(1, "cpu", .005, asset_directory=ASSET,
                            bundle_directory=tmp_path / "bundle", friction_mode="native_floor")
    migrated = response()
    effort = torch.zeros(2, 6)
    effort[0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        migrated.step(effort, torch.zeros_like(effort))
    assert migrated.head == 0 and not migrated.history.any()


def test_receipt_is_json_serializable_and_distinguishes_unidentified_parameters():
    receipt = response().receipt()
    json.dumps(receipt, allow_nan=False)
    assert receipt["spring_pressure_pa"] is None
    assert receipt["encoder_holding_compensation"] is False
    assert receipt["wheel_velocity_p_nm_per_rad_s"] == .5
    assert receipt["wheel_aggregate_delay_s"] == [.0055, .0035]


def test_same_native_inertia_draw_controls_software_bound_and_is_explicit_context():
    migrated = response(count=2, mode="software_dynamic", dtype=torch.float64)
    scale = torch.tensor([[.9, 1.1], [1.2, .8]], dtype=torch.float64)
    migrated.set_wheel_inertia_scale(scale)
    velocity = torch.zeros(2, 6, dtype=torch.float64)
    velocity[:, 4:] = 1e-4
    result = migrated.step(torch.zeros_like(velocity), velocity)
    torch.testing.assert_close(result.passive_effort_nm[:, 4:], -migrated.inertia * scale * 1e-4 / migrated.dt)
    context = migrated.context_parameters()
    torch.testing.assert_close(context["wheel_total_inertia_kg_m2"], migrated.inertia * scale)
    torch.testing.assert_close(context["wheel_additional_armature_kg_m2"], migrated.armature * scale)
    migrated.reset()
    torch.testing.assert_close(migrated.wheel_inertia_scale, scale)
