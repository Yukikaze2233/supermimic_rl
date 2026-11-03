"""Collect native transitions directly into compact replay, without an unused learner.

The decision callable supplies deployed actions and velocity/contact estimates.
Privileged estimator labels remain replay targets; they never drive the task FSM.
"""
from __future__ import annotations

import torch

from ..contracts import ACTION_BOUNDS
from .layout import SnapshotLayout
from .replay import CompactReplay
from .motion_task import MotionTask
from .stair_task import STAIR_RECORD_FIELDS, StairTaskState
from .stair_source import TOF_RECORD_FIELDS, StairSourceState
from .jump_task import JUMP_RECORD_FIELDS, JumpTaskState
from .recovery_task import RECOVERY_RECORD_FIELDS, RecoveryTaskState


def _schema_action_bounds(schema) -> tuple[float, ...]:
    """Native JSON sources pin non-legacy bounds; schema-less test environments keep the legacy contract."""
    return ACTION_BOUNDS if schema is None else schema.action_bounds


class Collector:
    def __init__(self, environment, decision, *, capacity_steps: int, history_length: int = 5,
                 seed: int = 7, action_noise_std: float = 0., curriculum_transition_cap: int | None = None,
                 replay: CompactReplay | None = None, advance_training_progress: bool = True,
                 record_terrain: bool = False, record_tasks: bool = False, record_stairs: bool = False,
                 record_stair_source: bool = False, record_jumps: bool = False, record_recovery: bool = False,
                 course_coordinate: int | None = None):
        if environment.current is None:
            raise ValueError('Reset the native environment before constructing its collector')
        if action_noise_std < 0 or history_length < 1 or capacity_steps < 1:
            raise ValueError('Invalid collection history or action noise')
        if course_coordinate is not None and (course_coordinate < 0 or not advance_training_progress):
            raise ValueError('A fixed course coordinate must be nonnegative and explicitly applied')
        self.environment, self.decision = environment, decision
        self.layout = (SnapshotLayout(environment.schema, record_terrain=True) if record_terrain
                       else SnapshotLayout(environment.schema))
        if replay is not None and ('terrain_tile' in replay.data) != record_terrain:
            raise ValueError('Collector terrain record layout differs from its replay')
        if replay is not None and record_terrain and 'task_family' not in replay.data:
            raise ValueError('Terrain collection needs the recorded task family; re-encode packed data or recollect')
        if replay is not None and ('motion_task' in replay.data) != record_tasks:
            raise ValueError('Collector motion-task record layout differs from its replay')
        if replay is not None and record_tasks and 'task_record_valid' not in replay.data:
            raise ValueError('Motion-task collection needs explicit recorded-state validity')
        if replay is not None and any((name in replay.data) != record_stairs for name in STAIR_RECORD_FIELDS):
            raise ValueError('Collector stair record layout differs from its replay')
        if record_stair_source and not (record_terrain and record_tasks and record_stairs):
            raise ValueError('Ideal stair-source collection requires terrain, command and stair task records')
        if replay is not None and any((name in replay.data) != record_stair_source for name in TOF_RECORD_FIELDS):
            raise ValueError('Collector ideal stair-source record layout differs from its replay')
        if replay is not None and ('task_estimate' in replay.data) != record_stair_source:
            raise ValueError('Ideal stair-source replay requires the actual deployed estimate')
        if replay is not None and ('stair_source_valid' in replay.data) != record_stair_source:
            raise ValueError('Ideal stair-source replay requires recorded-state validity')
        if record_jumps and not record_stair_source:
            raise ValueError('Jump collection requires recorded command, terrain and stair arbitration state')
        if replay is not None and any((name in replay.data) != record_jumps
                                     for name in (*JUMP_RECORD_FIELDS, 'jump_record_valid')):
            raise ValueError('Collector jump record layout differs from its replay')
        if record_recovery and not record_jumps:
            raise ValueError('Recovery collection requires complete jump and stair arbitration state')
        if replay is not None and any((name in replay.data) != record_recovery
                                     for name in (*RECOVERY_RECORD_FIELDS, 'recovery_record_valid')):
            raise ValueError('Collector recovery record layout differs from its replay')
        self.record_recovery = record_recovery
        self.record_tasks = record_tasks
        self.record_stairs = record_stairs
        self.record_stair_source = record_stair_source
        self.record_jumps = record_jumps
        if replay is not None and (replay.num_envs != environment.num_envs or replay.capacity != capacity_steps):
            raise ValueError('Collector replay size differs from the live environment/configuration')
        self.capacity, self.replay = capacity_steps, replay
        # A restarted physics engine is a new episode namespace, even if its
        # local counters happen to equal the last saved/teacher episode IDs.
        self.episode_offset = (int(torch.maximum(replay.data['start_episode_id'].max(),
                                                replay.data['end_episode_id'].max())) + 1
                               if replay is not None and replay.size else 0)
        self.history_length, self.action_noise_std = history_length, action_noise_std
        self.curriculum_transition_cap = curriculum_transition_cap
        self.course_coordinate = course_coordinate
        self.advance_training_progress = advance_training_progress
        self.generator = torch.Generator(device=environment.device).manual_seed(seed)
        frame = environment.current.actor_observation
        self.history = frame[:, None].expand(-1, history_length, -1).clone()
        self.action_bounds = frame.new_tensor(_schema_action_bounds(environment.schema))
        self.steps = 0

    @torch.no_grad()
    def collect(self, steps: int, *, actor_updates: int = 0, on_progress=None, on_transition=None,
                on_source_rng=None) -> dict:
        if steps < 1:
            raise ValueError('Collection needs positive physics exposure')
        if on_source_rng is not None and not self.record_stair_source:
            raise ValueError('Source RNG capture requires the recorded stair source')
        reward_sum = terminations = truncations = 0
        # Mixed collection: per-population totals separate the student's own
        # native return from the teacher rows sharing the batch.
        populations = {name: {'reward': 0., 'transitions': 0, 'terminations': 0} for name in ('teacher', 'student')}
        for _ in range(steps):
            env, before = self.environment, self.environment.current
            real_transitions = self.steps * env.num_envs
            coordinate = (real_transitions if self.curriculum_transition_cap is None else
                          min(real_transitions, self.curriculum_transition_cap))
            if self.course_coordinate is not None:
                coordinate = self.course_coordinate
            if self.advance_training_progress:
                env.set_training_progress(real_transitions=coordinate, actor_updates=actor_updates)
            task = MotionTask.capture(env.adapter.env) if self.record_tasks else None
            stair = StairTaskState.capture(env.adapter.env) if self.record_stairs else None
            source = StairSourceState.capture(env.adapter.env) if self.record_stair_source else None
            jump = JumpTaskState.capture(env.adapter.env) if self.record_jumps else None
            recovery = RecoveryTaskState.capture(env.adapter.env) if self.record_recovery else None
            action, estimate = self.decision(self.history)
            teacher_rows = getattr(self.decision, 'teacher_rows', None)
            if action.shape != (env.num_envs, 6) or estimate.shape != (env.num_envs, 5):
                raise ValueError('Collection decisions require six actions and five deployed estimates')
            raw = torch.cat((estimate[:, :3], torch.zeros_like(estimate[:, :3]), estimate[:, 3:5]), -1)
            env.set_policy_estimate(before.actor_observation, raw)
            task_estimate = estimate.clone() if self.record_stair_source else None
            noise = torch.randn(action.shape, device=action.device, generator=self.generator) * self.action_noise_std
            if on_source_rng is not None:
                # One generator state belongs to the entire native vector step.
                # Keeping it outside replay avoids duplicating it per environment.
                generator = env.adapter.env._terrain_course.tof_trigger.generator
                on_source_rng(self.steps, generator.get_state().clone())
            transition = env.step((action + noise).clamp(-self.action_bounds, self.action_bounds))
            record = self.layout.encode(transition)
            if task is not None:
                record['motion_task'] = task
                record['task_record_valid'] = task.new_ones(env.num_envs, 1)
            if stair is not None:
                record.update(stair.record())
            if source is not None:
                record.update(source.record())
                record['task_estimate'] = task_estimate
                record['stair_source_valid'] = task_estimate.new_ones(env.num_envs, 1)
            if jump is not None:
                record.update(jump.record())
                record['jump_record_valid'] = task_estimate.new_ones(env.num_envs, 1)
            if recovery is not None:
                record.update(recovery.record())
                record['recovery_record_valid'] = task_estimate.new_ones(env.num_envs, 1)
            if self.replay is None:
                self.replay = CompactReplay({name: tuple(value.shape[1:]) for name, value in record.items()},
                                             env.num_envs, self.capacity, env.device)
            boundary = {name: getattr(transition, name) for name in ('valid', 'terminated', 'truncated')}
            boundary.update(policy_owned=transition.start.policy_owned, end_policy_owned=transition.end.policy_owned)
            for endpoint in ('start', 'end'):
                for name in ('episode_id', 'control_epoch'):
                    boundary[endpoint + '_' + name] = getattr(getattr(transition, endpoint), name)
                boundary[endpoint + '_episode_id'] = boundary[endpoint + '_episode_id'] + self.episode_offset
            if self.replay.device != torch.device(env.device):
                # Copy both record and boundary before the ring computes its
                # episode/history continuity. CPU replay frees GPU capacity
                # for full policy batches without reducing physical coverage.
                record = {name: value.to(self.replay.device) for name, value in record.items()}
                boundary = {name: value.to(self.replay.device) for name, value in boundary.items()}
            self.replay.add(record, boundary)
            if on_transition is not None:
                on_transition(transition)
            after = env.current
            frame = after.actor_observation
            history = torch.cat((self.history[:, 1:], frame[:, None]), 1)
            reset = (~transition.valid | transition.terminated | transition.truncated
                     | ~transition.start.policy_owned | ~transition.end.policy_owned
                     | (before.episode_id != after.episode_id) | (before.control_epoch != after.control_epoch))
            self.history = torch.where(reset[:, None, None], frame[:, None], history)
            reward_sum += float(transition.reward.sum())
            terminations += int(transition.terminated.sum())
            if teacher_rows is not None:
                for name, rows in (('teacher', teacher_rows), ('student', ~teacher_rows)):
                    populations[name]['reward'] += float(transition.reward[rows].sum())
                    populations[name]['transitions'] += int(rows.sum())
                    populations[name]['terminations'] += int((transition.terminated & rows).sum())
            truncations += int(transition.truncated.sum())
            self.steps += 1
            if on_progress is not None:
                on_progress({'steps': self.steps, 'transitions': self.steps * env.num_envs,
                              'terminations': terminations, 'truncations': truncations})
        return {'transitions': self.steps * env.num_envs, 'stored_transitions': len(self.replay),
                'reward_mean': reward_sum / (steps * self.environment.num_envs),
                'terminations': terminations, 'truncations': truncations,
                'course_coordinate': coordinate, 'fixed_course_coordinate': self.course_coordinate,
                **{f'{name}_{key}': value for name, totals in populations.items() if totals['transitions']
                   for key, value in (('reward_mean', totals['reward'] / totals['transitions']),
                                      ('transitions', totals['transitions']),
                                      ('terminations', totals['terminations']))}}
