"""Finite PhysX decoder failures invalidate/reset only attributable native rows."""
import math
import json
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.course.execution import make_environment

coordinates = pytest.importorskip('transformer_rl.formal.lab_env.joint_coordinates')
ContinuousJointCoordinates = coordinates.ContinuousJointCoordinates


def environment():
    class Base:
        def _joint_positions(self):
            return self._joint_coordinates.read(self.robot.data.joint_pos,
                self.robot.data.joint_vel, self._physical_tick)

        def _pre_physics_step(self, action):
            self._joint_positions()

        def _reset_idx(self, ids):
            self.resets.append(ids.clone())
            self.robot.data.joint_pos[ids] = 0.
            self.robot.data.joint_vel[ids] = 0.
            self._joint_coordinates.reset_rows(ids.long(), self.robot.data.joint_pos[ids])
            self._terminated[ids] = False

        def step(self, action):
            self._pre_physics_step(action)
            self._terminated.zero_()
            self._physical_tick += 1
            self.robot.data.joint_pos.copy_(self.next_position)
            self.endpoint = self._joint_positions().clone()
            terminated = self._terminated.clone()
            self.extras['interval_observation_valid'] = torch.ones(3, dtype=torch.bool)
            self.extras['termination_counts'] = torch.tensor([int(terminated.sum()), 0, 0, 0])
            ids = terminated.nonzero().flatten()
            if len(ids):
                self._reset_idx(ids)
            return {}, torch.ones(3), terminated, torch.zeros(3, dtype=torch.bool), self.extras

    execution = SimpleNamespace(reset=lambda ids: None)
    instance = object.__new__(make_environment(Base, execution))
    initial = torch.tensor([[0., .3], [6.2, .4], [-6.2, .5]])
    instance.num_envs, instance._device = 3, torch.device('cpu')
    instance._identified_execution = execution
    instance._as_torch = lambda value: value
    pose = torch.zeros(3, 7)
    pose[:, 2], pose[:, 6] = .3, 1.
    data = SimpleNamespace(joint_pos=initial.clone(), joint_vel=torch.zeros_like(initial),
                           root_pose=pose, root_velocity=torch.zeros(3, 6))
    writes = []

    def write(field, value, ids):
        writes.append((field, ids.clone()))
        getattr(data, field)[ids] = value

    instance.robot = SimpleNamespace(data=data,
        write_root_link_pose_to_sim_index=lambda root_pose, env_ids: write('root_pose', root_pose, env_ids),
        write_root_com_velocity_to_sim_index=lambda root_velocity, env_ids: write('root_velocity', root_velocity, env_ids),
        write_joint_position_to_sim_index=lambda position, env_ids: write('joint_pos', position, env_ids),
        write_joint_velocity_to_sim_index=lambda velocity, env_ids: write('joint_vel', velocity, env_ids))
    instance._root_link_position = lambda: data.root_pose[:, :3]
    instance._root_link_quat = lambda: data.root_pose[:, 3:]
    instance.scene = SimpleNamespace(write_data_to_sim=lambda: None, update=lambda dt: None)
    instance.sim = SimpleNamespace(forward=lambda: None)
    instance.physics_writes = writes
    instance._numerical_start_pose = pose.clone()
    instance._numerical_start_position = initial.clone()
    instance._binding = SimpleNamespace(native_joint_names=('wheel', 'bounded'))
    instance._joint_coordinates = ContinuousJointCoordinates(initial, ('wheel', 'bounded'), ('wheel',), .005)
    instance._physical_tick = 0
    instance._joint_coordinates.read(initial, torch.zeros_like(initial), 0)
    instance._terminated = torch.zeros(3, dtype=torch.bool)
    instance._numerical_interval_active = False
    instance._numerical_invalid = torch.zeros(3, dtype=torch.bool)
    instance._numerical_had_event = False
    instance._numerical_invalid_total = 0
    instance.extras, instance.resets = {}, []
    return instance


def test_bad_row_resets_at_interval_end_healthy_native_four_pi_wraps_stay_exact():
    env = environment()
    env.next_position = torch.tensor([[3.7, .31], [-6.2, .41], [6.2, .51]])
    healthy = ContinuousJointCoordinates(env.robot.data.joint_pos[1:].clone(),
        ('wheel', 'bounded'), ('wheel',), .005)
    healthy.read(env.robot.data.joint_pos[1:], env.robot.data.joint_vel[1:], 0)
    expected = healthy.read(env.next_position[1:], env.robot.data.joint_vel[1:], 1)
    _, _, terminated, _, extras = env.step(None)
    torch.testing.assert_close(env.endpoint[1:], expected, rtol=0, atol=0)
    assert env.endpoint[0, 0] == 0.  # Park immediately; reset the episode at the boundary.
    assert env._joint_coordinates._turns[:, 0].tolist() == [0, 1, -1]
    assert terminated.tolist() == [True, False, False]
    assert [rows.tolist() for rows in env.resets] == [[0]]
    assert extras['numerical_invalid'].tolist() == [True, False, False]
    assert extras['interval_observation_valid'].tolist() == [False, True, True]
    assert extras['termination_counts'][0] == 0
    assert extras['numerical_rejected_intervals_total'] == 1
    previous_mask = extras['numerical_invalid']
    env.next_position = torch.tensor([[.1, .1], [-6.1, .42], [6.1, .52]])
    _, _, terminated, _, extras = env.step(None)
    assert not terminated.any() and extras['interval_observation_valid'].all()
    assert not extras['numerical_invalid'].any()
    assert previous_mask.tolist() == [True, False, False]  # Returned mask owns its interval.
    assert extras['numerical_rejected_intervals_total'] == 1


def test_parking_modifies_only_rejected_rows_and_counts_a_repeated_row_once():
    env = environment()
    env._numerical_interval_active = True
    for tick, value in ((1, 3.7), (2, 3.7)):
        env._physical_tick = tick
        env.robot.data.joint_pos[0, 0] = value
        before = env.robot.data.joint_pos.clone()
        env._joint_positions()
        assert torch.equal(env.robot.data.joint_pos[1:], before[1:])
        assert env.robot.data.joint_pos[0, 0] == 0.
    assert env._numerical_invalid_total == 1
    assert not env.resets  # Task/episode reset still belongs to the policy boundary.
    assert all(rows.tolist() == [0] for _, rows in env.physics_writes)


@pytest.mark.parametrize('failure', ['nan_position', 'nan_velocity', 'lost_tick', 'outside_step'])
def test_nonfinite_missing_samples_and_unhandled_sampling_failures_still_raise(failure, capsys):
    env = environment()
    env._numerical_interval_active = failure != 'outside_step'
    env._physical_tick = 2 if failure == 'lost_tick' else 1
    env.robot.data.joint_pos[0, 0] = 3.7
    if failure == 'nan_position':
        env.robot.data.joint_pos[1, 0] = float('nan')
    elif failure == 'nan_velocity':
        env.robot.data.joint_vel[1, 0] = float('nan')
    with pytest.raises(RuntimeError):
        env._joint_positions()
    assert not env._numerical_invalid.any()
    assert env._numerical_invalid_total == 0
    assert env._joint_coordinates._tick == 0
    if failure.startswith('nan_'):
        diagnostic = json.loads(capsys.readouterr().out)
        assert diagnostic['event'] == 'nonfinite_native_joint_state'
        assert diagnostic['action'] == 'raise_original_error'
        assert diagnostic['samples'] == [{
            'field': failure.removeprefix('nan_'), 'row': 1, 'joint': 'wheel', 'value': 'nan'}]


def test_finite_speed_isolation_preserves_healthy_coordinates_and_parks_invalid_rows():
    env = environment()
    env._numerical_interval_active = True
    healthy = ContinuousJointCoordinates(env.robot.data.joint_pos[2:].clone(),
        ('wheel', 'bounded'), ('wheel',), .005)
    healthy.read(env.robot.data.joint_pos[2:], env.robot.data.joint_vel[2:], 0)
    for tick in (1, 2):
        env._physical_tick = tick
        env.robot.data.joint_pos[0, 0] = tick * .1
        env.robot.data.joint_pos[2, 0] += .1
        env.robot.data.joint_vel[0, 0] = math.pi / .005 + 1.
        if tick == 2:
            env.robot.data.joint_vel[1, 0] = -(math.pi / .005 + 20.)
        original_position = env.robot.data.joint_pos.clone()
        original_velocity = env.robot.data.joint_vel.clone()
        expected = healthy.read(original_position[2:], original_velocity[2:], tick)
        actual = env._joint_positions()
        assert torch.equal(env.robot.data.joint_pos[2:], original_position[2:])
        assert torch.equal(env.robot.data.joint_vel[2:], original_velocity[2:])
        assert not env.robot.data.joint_vel[env._numerical_invalid].any()
        torch.testing.assert_close(actual[2:], expected, rtol=0, atol=0)
        assert env._numerical_invalid.tolist() == [True, tick == 2, False]
        assert env._numerical_invalid_total == tick
    assert not env.resets
    assert all(env._numerical_last_event['ambiguous_speed'])


def test_speed_failure_marks_whole_interval_invalid_and_uses_normal_row_reset():
    env = environment()
    env.next_position = env.robot.data.joint_pos.clone()
    env.next_position[0, 0] += .1
    env.robot.data.joint_vel[0, 0] = math.pi / .005 + 1.
    native_velocity = env.robot.data.joint_vel.clone()
    _, _, terminated, _, extras = env.step(None)
    assert terminated.tolist() == [True, False, False]
    assert [rows.tolist() for rows in env.resets] == [[0]]
    assert extras['numerical_invalid'].tolist() == [True, False, False]
    assert extras['interval_observation_valid'].tolist() == [False, True, True]
    assert extras['termination_counts'][0] == 0
    assert torch.equal(env.robot.data.joint_vel[1:], native_velocity[1:])
    assert not env.robot.data.joint_vel[0].any()


def test_failed_first_substep_is_parked_before_terrain_and_later_physics_ticks():
    env = environment()
    env._numerical_interval_active = True
    env._pre_physics_step(None)
    healthy_initial = env.robot.data.joint_pos[1:].clone()
    for tick in range(1, 5):
        env._physical_tick = tick
        if tick == 1:
            env.robot.data.joint_vel[0, 0] = 838.
        elif env.robot.data.joint_vel[0, 0] > 800:
            # The previous implementation reached NaNs before end-of-step reset.
            env.robot.data.root_pose[0, :3] = float('nan')
        env.robot.data.joint_pos[1:, 0] += .01
        env._joint_positions()
        assert torch.isfinite(env.robot.data.root_pose).all()
    torch.testing.assert_close(env.robot.data.joint_pos[1:, 0], healthy_initial[:, 0] + .04)
    assert env._numerical_invalid.tolist() == [True, False, False]
    assert env._numerical_invalid_total == 1
