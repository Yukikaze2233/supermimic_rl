"""Terrain gradients and routing preserve physical steps, slopes and query misses."""
from dataclasses import replace

import pytest
import torch
from support.world_v2.terrain import atlas, state


def test_native_height_and_facet_gradient_use_tile_frame_and_ceiling():
    terrain = atlas()
    points = torch.tensor([[[.3, .1, -100.], [.8, .2, 20.]],
                           [[-.2, 0., 100.], [.2, 0., -100.]]], requires_grad=True)
    sample = terrain.height_at(points, torch.tensor([0., 1.]))
    torch.testing.assert_close(sample.height, torch.tensor([[.05, .14], [0., .2]]), atol=3e-6, rtol=0)
    assert sample.valid.all()
    sample.height.sum().backward()
    torch.testing.assert_close(points.grad[0], torch.tensor([[.2, -.1, 0.], [.2, -.1, 0.]]))
    assert (points.grad[1] == 0).all()  # No invented derivative across the discontinuous riser.


def test_predicted_motion_refreshes_scan_across_step_instead_of_freezing_it():
    terrain = atlas()
    start = state(torch.tensor([[-.6, 0., .3]]))
    end = state(torch.tensor([[[.1, 0., .4], [.2, 0., .5]]]))
    before = terrain.observe(start, torch.tensor([1]))
    height, scan = terrain.advance(before.height_above_ground[:, None], before.scan_relative[:, None],
                                   start.with_time_axis(), end, tile=torch.tensor([1]))
    torch.testing.assert_close(height[..., 0], torch.tensor([[.2, .3]]), atol=1e-6, rtol=0)
    assert before.scan_relative.max() == -.3
    torch.testing.assert_close(scan[0, :, 38], torch.tensor([-.2, -.3]), atol=1e-6, rtol=0)
    assert (scan[0, 0] > -.3).any() and (scan[0, 0] < -.3).any()


def test_scan_rotation_and_height_gradient_match_finite_difference_inside_facet():
    terrain = atlas()
    position = torch.tensor([[.3, -.2, .5]], requires_grad=True)
    yaw = torch.tensor([.7], requires_grad=True)

    def value(p, y):
        observed = terrain.observe(state(p, y), torch.tensor([0]))
        return observed.height_above_ground.sum() + observed.scan_relative[:, 0].sum()

    value(position, yaw).backward()
    # Native geometry is float32 at translated atlas coordinates. A 2 mm/rad
    # finite difference stays inside this plane and above its quantization.
    epsilon = .002
    gradient = []
    for column in range(3):
        delta = torch.zeros_like(position)
        delta[:, column] = epsilon
        gradient.append((value(position + delta, yaw) - value(position - delta, yaw)) / (2 * epsilon))
    numeric_yaw = (value(position, yaw + epsilon) - value(position, yaw - epsilon)) / (2 * epsilon)
    torch.testing.assert_close(position.grad.flatten(), torch.stack(gradient), atol=7e-4, rtol=0)
    torch.testing.assert_close(yaw.grad[0], numeric_yaw, atol=7e-4, rtol=0)


def test_query_misses_remain_invalid_and_tile_ids_are_not_interpolated():
    terrain = atlas()
    sample = terrain.height_at(torch.tensor([[6., 0., .3]]), torch.tensor([0]))
    assert not sample.valid.item() and torch.isnan(sample.height).all()
    with pytest.raises(ValueError, match='integer routes'):
        terrain.height_at(torch.zeros(1, 3), torch.tensor([.2]))
    with pytest.raises(ValueError, match='recorded terrain tile'):
        physical = state(torch.zeros(1, 3))
        terrain.advance(None, None, physical, physical)


def test_world_model_queries_all_four_predicted_ticks_and_fitting_uses_tile_routes():
    from pathlib import Path
    from supermimic_rl.world_model_v2.execution import WheelExecution
    from supermimic_rl.world_model_v2.features import MotionInputs
    from supermimic_rl.world_model_v2.fitting import WorldModelFit
    from supermimic_rl.world_model_v2.leg_linkage import LegLinkage
    from supermimic_rl.world_model_v2.model import WorldModelV2
    from supermimic_rl.world_model_v2.records import Memory, encode_memory, encode_physical

    root = Path(__file__).resolve().parents[1]
    linkage = LegLinkage.load(root / 'configs/linkage/leg_linkage_e726d23c.json')
    model = WorldModelV2(MotionInputs(linkage), WheelExecution.from_bundle(root / 'configs/identified/20261031', .005),
                         hidden=(16,), terrain=atlas())
    physical = state(torch.tensor([[-.1, 0., .3]]))
    q = torch.zeros(1, 6)
    q[:, 1] = linkage.sides['left'].phi_min + .1
    q[:, 3] = linkage.sides['right'].phi_min + .1
    physical = replace(physical, joint_position=q)
    memory = Memory(torch.zeros(1, 3, 2), torch.zeros(1, 6), torch.zeros(1, dtype=torch.long),
                    torch.zeros(1), torch.zeros(1), torch.zeros(1))
    with torch.no_grad():
        ticks = model.motion.net[-1].bias.view(4, -1)
        ticks[:, 0] = torch.tensor([.04, .08, .12, .16])
    before = model.terrain.observe(physical, torch.tensor([1]))
    prediction = model.step(physical, before.height_above_ground, before.scan_relative, memory,
                            torch.zeros(1, 65), torch.ones(1, 2), torch.zeros(1, 6), terrain_tile=torch.tensor([1]))
    torch.testing.assert_close(prediction.tick_height_above_ground.flatten(), torch.tensor([.3, .3, .1, .1]),
                               atol=1e-6, rtol=0)
    windows = {name: value[:, None] for name, value in {
        'physical': encode_physical(physical), 'end_physical': encode_physical(prediction.state),
        'memory': encode_memory(memory), 'height': before.height_above_ground, 'scan': before.scan_relative,
        'action': torch.zeros(1, 6), 'sim_context': torch.zeros(1, 65), 'motor_strength': torch.ones(1, 6),
        'ticks': prediction.relative_ticks.detach(), 'terrain_tile': torch.ones(1, 1),
        'valid': torch.ones(1, dtype=torch.bool), 'reward_route': torch.tensor([[0., 3., 0.]])}.items()}
    fit = WorldModelFit(model)
    assert fit.terms(windows).state < 1e-10
    del windows['terrain_tile']
    with pytest.raises(ValueError, match='recorded terrain tile'):
        fit.terms(windows)


@pytest.mark.parametrize('wheel_contact_weight', [0., 1.])
def test_scan_domain_exit_keeps_boundary_gradient_and_reports_incomplete_predictions(wheel_contact_weight):
    from pathlib import Path
    from supermimic_rl.world_model_v2.execution import WheelExecution
    from supermimic_rl.world_model_v2.features import MotionInputs
    from supermimic_rl.world_model_v2.fitting import WorldModelFit, recorded_rollout
    from supermimic_rl.world_model_v2.leg_linkage import LegLinkage
    from supermimic_rl.world_model_v2.model import WorldModelV2
    from supermimic_rl.world_model_v2.records import Memory, RecordSpec, encode_memory, encode_physical
    from supermimic_rl.world_model_v2.replay import CompactReplay
    from supermimic_rl.world_model_v2.validation import validate

    root = Path(__file__).resolve().parents[1]
    linkage = LegLinkage.load(root / 'configs/linkage/leg_linkage_e726d23c.json')
    model = WorldModelV2(MotionInputs(linkage), WheelExecution.from_bundle(root / 'configs/identified/20261031', .005),
                         hidden=(16,), terrain=atlas())
    physical = state(torch.tensor([[4.45, 0., .3], [0., 0., .3]]))
    physical.joint_position[:, 1] = linkage.sides['left'].phi_min + .1
    physical.joint_position[:, 3] = linkage.sides['right'].phi_min + .1
    memory = Memory(torch.zeros(2, 3, 2), torch.zeros(2, 6), torch.zeros(2, dtype=torch.long),
                    torch.zeros(2), torch.zeros(2), torch.zeros(2))
    with torch.no_grad():
        model.motion.net[-1].bias.view(4, -1)[:, 0] = torch.tensor([.04, .08, .12, .16])
        model.motion.net[-1].bias.view(4, -1)[:, 26:28] = 10.
    before = model.terrain.observe(physical, torch.zeros(2))
    assert before.valid.all()
    shapes = RecordSpec(16, 65).fields(record_terrain=True)
    record = {name: torch.zeros(2, *shape) for name, shape in shapes.items()}
    record.update(physical=encode_physical(physical), end_physical=encode_physical(physical),
                  memory=encode_memory(memory), height=before.height_above_ground,
                  end_height=before.height_above_ground, scan=before.scan_relative,
                  motor_strength=torch.ones(2, 6), reward_route=torch.tensor([[0., 1., 0.]]).repeat(2, 1))
    boundary = {name: torch.ones(2, dtype=torch.bool) for name in ('valid', 'policy_owned', 'end_policy_owned')}
    boundary.update({name: torch.zeros(2, dtype=torch.bool) for name in ('terminated', 'truncated')})
    boundary.update({name: torch.zeros(2, dtype=torch.int64) for name in
                     ('start_episode_id', 'end_episode_id', 'start_control_epoch', 'end_control_epoch')})
    replay = CompactReplay(shapes, 2, 4, 'cpu')
    for _ in range(4):
        replay.add(record, boundary)
    windows = {name: value.transpose(0, 1) for name, value in replay.data.items()}
    fit = WorldModelFit(model, wheel_contact_weight=wheel_contact_weight)
    trace = list(recorded_rollout(model, windows))
    assert [rows.tolist() for _, rows, _, _ in trace] == [[0, 1], [1], [1], [1]]
    assert torch.isnan(trace[0][2].terrain_scan[0]).any()
    terms = fit.terms(windows)
    assert terms.terrain_truncated_fraction == .5
    assert terms.per_step_state[0, 0] > 0 and (terms.per_step_state[0, 1:] == 0).all()
    single = fit.terms({name: value[:1, :1] for name, value in windows.items()})
    complete = fit.terms({name: value[1:] for name, value in windows.items()})
    expected = (single.state + single.ticks) / 8 + (complete.state + complete.ticks) / 2
    loss = terms.state + terms.ticks
    if wheel_contact_weight:
        expected_wheel = single.wheel_contact / 8 + complete.wheel_contact / 2
        torch.testing.assert_close(terms.wheel_contact, expected_wheel)
        loss = loss + wheel_contact_weight * terms.wheel_contact
        expected = expected + wheel_contact_weight * expected_wheel
    torch.testing.assert_close(loss, expected)
    parameter = model.motion.net[-1].bias
    gradient = torch.autograd.grad(loss, parameter)[0]
    torch.testing.assert_close(gradient, torch.autograd.grad(expected, parameter)[0])
    assert torch.isfinite(gradient).all() and (gradient.view(4, -1)[:, 0] > 0).all()
    report = validate(fit, replay, [0, 1], 4, 128, torch.Generator().manual_seed(3), 'cpu')
    for value in report.values():
        assert 0 < value['terrain_truncated_fraction'] < 1
        assert 0 < value['complete_prediction_windows'] < value['sampled_windows']
        assert value['nonfinite_motion_fraction'] == 0
        assert value['position_m'] > 0
