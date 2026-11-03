#!/usr/bin/env python3
"""Read-only native stand probe for own PPO, V2 student, or zero action.

Run each policy in a separate native process. Fixed scheduled stand targets do
not make birth, handoff, DR or policy-specific history RNG sequences identical.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import traceback

import torch

from supermimic_rl.contracts import ACTION_BOUNDS
from supermimic_rl.ppo.trainer import EpisodeStatistics


class NativePostureProbe:
    """Observe completed physical intervals without changing control or resets."""

    def __init__(self, native, *, steps, split_step, tail_steps=50):
        if not 0 < split_step < steps or not 0 < tail_steps <= steps:
            raise ValueError('Probe needs an interior split and a positive final window')
        self.native, self.steps, self.split_step, self.tail_steps = native, steps, split_step, tail_steps
        self.index, self.requested, self.endpoint, self.feedback = 0, None, None, None
        self._active_interval = False
        self.bounds = torch.tensor(ACTION_BOUNDS, device=native.device)
        self.joints = tuple(native._binding.native_joint_names)
        self.bodies = tuple(native._binding.native_body_names)
        self.active = tuple(self.joints[index] for index in native._active_indices.tolist())
        self.knee = tuple(native._formal_reward._passive_knee_indices)
        self.statistics = {name: EpisodeStatistics(native.num_envs, native.device)
                           for name in ('all', 'early', 'late', 'tail')}
        self._restore = []

    def requested_action(self, action):
        self.requested = action.detach().clone()

    def _wrap(self, owner, name, replacement):
        self._restore.append((owner, name, owner.__dict__.get(name), name in owner.__dict__))
        setattr(owner, name, replacement)

    def __enter__(self):
        native = self.native
        controller = native._control.controller
        effort, observe, step = controller.effort, native._observe_substep, native.step
        self.feedback_method = {'module': effort.__module__, 'qualname': effort.__qualname__,
                                'source': inspect.getsourcefile(effort)}

        def measured_effort(targets, q, dq, *args, **kwargs):
            result = effort(targets, q, dq, *args, **kwargs)
            if self._active_interval:
                self.feedback = {
                    'target': torch.cat((targets.admitted_leg_position_rad, targets.wheel_velocity_rad_s), -1).clone(),
                    'feedback_q': q.clone(), 'feedback_dq': dq.clone(),
                    'pair_projected': targets.pair_projected.clone(),
                    'requested_effort': result.requested_effort.clone(),
                    'capped_effort': result.capped_effort.clone(), 'saturated': result.saturated.clone()}
            return result

        def measured_observe(*args, **kwargs):
            result = observe(*args, **kwargs)
            # Snapshot before the policy boundary can reset native tensors.
            if self._active_interval:
                self.endpoint = self._endpoint()
            return result

        def measured_step(action):
            if self.requested is None:
                raise RuntimeError('Record the policy output before the native step')
            command = native._command.clone()
            self.endpoint = self.feedback = None
            self._active_interval = True
            try:
                result = step(action)
            finally:
                self._active_interval = False
            self._record(command, result)
            return result

        self._wrap(controller, 'effort', measured_effort)
        self._wrap(native, '_observe_substep', measured_observe)
        self._wrap(native, 'step', measured_step)
        return self

    def __exit__(self, *_):
        for owner, name, value, existed in reversed(self._restore):
            if existed:
                setattr(owner, name, value)
            else:
                delattr(owner, name)
        self._restore.clear()

    def _endpoint(self):
        native, feedback = self.native, self.feedback
        if feedback is None:
            raise RuntimeError('Physical endpoint lacks its actual controller input')
        sample, state = native._last_sample, native._last_state
        q = sample.joint_position_rad
        values = {'height_m': state.height_above_ground_m.clone(),
                  'phi_left_rad': (q[:, self.joints.index('LL_joint1')] - q[:, self.joints.index('L_joint1')]).clone(),
                  'phi_right_rad': (q[:, self.joints.index('RR_joint1')] - q[:, self.joints.index('R_joint1')]).clone()}
        for i, name in enumerate(self.active):
            values['joint_rad/' + name] = q[:, native._active_indices[i]].clone()
            values['issued_action/' + name] = native._issued_action[:, i].clone()
            values['actual_motor_effort_nm/' + name] = native._last_effort[:, i].clone()
            values['native_effort_nm/' + name] = native._last_native_effort[:, i].clone()
            for label, field in (('feedback_q_rad', 'feedback_q'), ('feedback_dq_rad_s', 'feedback_dq'),
                                 ('controller_requested_nm', 'requested_effort'), ('controller_capped_nm', 'capped_effort'),
                                 ('controller_saturated', 'saturated')):
                values[label + '/' + name] = feedback[field][:, i].float().clone()
            units = 'target_q_rad/' if i < 4 else 'target_wheel_rad_s/'
            values[units + name] = feedback['target'][:, i].clone()
        for i, side in enumerate(('left', 'right')):
            values['pair_projected/' + side] = feedback['pair_projected'][:, i].float().clone()
        reward = native._formal_reward
        knee = q[:, self.knee]
        inner = reward._knee_inner_at_lower + reward._knee_inner_direction * (knee - reward._knee_native_lower)
        for i, index in enumerate(self.knee):
            values['passive_native_rad/' + self.joints[index]] = knee[:, i].clone()
            values['passive_inner_rad/' + self.joints[index]] = inner[:, i].clone()
        force = sample.terrain_contact_force_world_n
        normal_source = reward.config.normal_load_source
        if normal_source == 'native_contact_patch_normal_forces':
            load = sample.body_normal_load_n
        elif normal_source == 'contact_force_norm':
            load = force.norm(dim=-1)
        elif normal_source == 'contact_force_world_z_abs':
            load = force[..., 2].abs()
        else:
            raise ValueError('Unknown native contact load source')
        for i, name in enumerate(self.bodies):
            values['body_load_n/' + name] = load[:, i].clone()
            values['body_resultant_n/' + name] = force[:, i].norm(dim=-1).clone()
            values['body_contact_over_3n/' + name] = (load[:, i] > 3.).float()
        up = -state.gravity_body
        forward = torch.zeros_like(up)
        forward[:, 0] = 1.
        forward -= (forward * up).sum(-1, keepdim=True) * up
        forward /= forward.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        values['forward_m_s'] = (state.velocity_origin_body_m_s * forward).sum(-1).clone()
        values['yaw_rate_rad_s'] = (state.angular_velocity_body_rad_s * up).sum(-1).clone()
        values['tilt_rad'] = torch.atan2(state.gravity_body[:, :2].norm(dim=-1), -state.gravity_body[:, 2]).clone()
        return values

    def _record(self, command, step):
        if self.endpoint is None:
            raise RuntimeError('Native interval produced no physical endpoint')
        native, values = self.native, dict(self.endpoint)
        reward = native._formal_result
        valid = step[4]['interval_observation_valid'] & reward.fully_policy_owned
        numerical = step[4].get('numerical_invalid', torch.zeros_like(valid))
        values.update(command_forward_m_s=command[:, 0], command_yaw_rate_rad_s=command[:, 2],
                      command_height_m=command[:, 3])
        for i, name in enumerate(self.active):
            values['policy_output/' + name] = self.requested[:, i]
            values['action_clipped/' + name] = (self.requested[:, i].abs() > self.bounds[i]).float()
            values['action_near_bound/' + name] = (self.requested[:, i].abs() >= self.bounds[i] * (1 - 1e-5)).float()
        for prefix, attribute in (('reward_per_s', 'component_densities'),
                                  ('reward_event', 'event_rewards'), ('reward_gate', 'gate_fractions')):
            values.update({prefix + '/' + name: value.clone() for name, value in getattr(reward, attribute).items()})
        tracking = {'forward_error_m_s': values['forward_m_s'] - command[:, 0],
                    'yaw_rate_error_rad_s': values['yaw_rate_rad_s'] - command[:, 2],
                    'height_error_m': values['height_m'] - command[:, 3], 'tilt_rad': values['tilt_rad']}
        record = SimpleNamespace(reward=step[1], terminated=step[2], truncated=step[3],
                                 valid=valid, numerical_invalid=numerical, tracking=tracking, diagnostics=values)
        names = ['all', 'early' if self.index < self.split_step else 'late']
        if self.index >= self.steps - self.tail_steps:
            names.append('tail')
        for name in names:
            self.statistics[name].record(record)
        self.index += 1

    def summary(self):
        return {'completed_steps': self.index,
                'segments': {name: statistics.summary() for name, statistics in self.statistics.items()},
                'windows': {'early': [0, self.split_step], 'late': [self.split_step, self.steps],
                            'tail': [self.steps - self.tail_steps, self.steps]},
                'active_joint_order': self.active, 'body_order': self.bodies,
                'controller_method': self.feedback_method,
                'sampling': 'physical pose, targets, torque and per-body contacts at final 5ms tick before reset; reward densities/gates averaged over four ticks',
                'selection': 'finite/valid intervals fully owned by policy, including their terminal endpoint; excluded script/preparation intervals are not physical failures',
                'contact_load_source': self.native._formal_reward.config.normal_load_source,
                'contact_threshold_n': 3.,
                'action_clipping': 'policy output outside hard action bounds; near_bound uses 1e-5 relative tolerance; student output is already tanh-bounded',
                'reward_units': 'reward_per_s already weighted signed per-second mean; reward_event once-per-interval amount'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--factory-config', type=Path, required=True)
    parser.add_argument('--policy', choices=('ppo', 'student', 'zero'), required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--split-step', type=int, default=100)
    parser.add_argument('--tail-steps', type=int, default=50)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (args.policy == 'zero') != (args.checkpoint is None):
        parser.error('Only PPO/student take a checkpoint; zero is a control probe')
    factory = json.loads(args.factory_config.read_text())
    if factory.get('evaluation', {}).get('scenarios') != ['stand']:
        parser.error('Use the existing fixed stand scenario, not a random course')
    factory['record_substeps'] = True
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new posture diagnostic output directory')
    output.mkdir(parents=True, exist_ok=True)
    factory['native_log_directory'] = str(output / 'native')
    (output / 'factory-config.json').write_text(json.dumps(factory, indent=2) + '\n')
    from supermimic_rl.course.factory import create_course
    from supermimic_rl.adapters.packed import PackedEnvironment
    from supermimic_rl.cli import _native_schema_source, native_logs
    from supermimic_rl.world_model_v2.teacher import file_sha256
    with native_logs(output / 'native'):
        adapter = create_course(factory)
        packed = PackedEnvironment(adapter, source=_native_schema_source(adapter.provenance), policy_dt_s=adapter.policy_dt_s)
        try:
            packed.reset()
            probe = NativePostureProbe(adapter.env, steps=args.steps, split_step=args.split_step,
                                       tail_steps=args.tail_steps)
            checkpoint_sha = None if args.checkpoint is None else file_sha256(args.checkpoint)

            def progress(record):
                with (output / 'progress.jsonl').open('a') as stream:
                    stream.write(json.dumps(record) + '\n')

            with torch.no_grad(), probe:
                if args.policy == 'student':
                    from train_world_v2 import read_actor
                    from supermimic_rl.world_model_v2.evaluation import evaluate_native
                    actor, saved = read_actor(args.checkpoint, packed.device)
                    if saved['schema_fingerprint'] != packed.schema.fingerprint:
                        raise ValueError('Student/native physical source differs')

                    class ObservedActor:
                        def eval(self):
                            actor.eval()
                            return self

                        def __call__(self, history):
                            action = actor(history)
                            probe.requested_action(action)
                            return action

                        def estimate(self, history):
                            return actor.estimate(history)

                    evaluation = evaluate_native(packed, ObservedActor(), args.steps, on_progress=progress)
                    interface = 'V2 evaluate_native/Collector history, reset rules and estimator; tanh-bounded output'
                else:
                    from supermimic_rl.ppo.environment import CourseEnvironment
                    from supermimic_rl.ppo.evaluation import evaluate
                    from supermimic_rl.ppo.trainer import load_actor
                    adapter.evaluation_begin(packed.schema)
                    environment = CourseEnvironment(adapter)
                    actor = None if args.policy == 'zero' else load_actor(args.checkpoint, packed.device)

                    def decision(history):
                        if actor is None:
                            action, estimate = history.new_zeros(history.shape[0], 6), history.new_zeros(history.shape[0], 5)
                            probe.requested_action(action)
                            return action, estimate
                        estimate, action = actor.estimator(history), actor(history)
                        probe.requested_action(action)
                        return actor.clip(action), estimate

                    evaluation = evaluate(environment, actor, args.steps, decision=decision, on_progress=progress)
                    interface = 'PPO evaluate/CourseEnvironment native 5x35 history and estimator; raw mean then hard clip'
                    if actor is None:
                        interface = 'zero six chassis actions and zero deployment estimate; existing native control/preparation remains active'
                report = {**probe.summary(), 'policy': args.policy, 'policy_interface': interface,
                          'checkpoint': None if args.checkpoint is None else str(args.checkpoint.resolve()),
                          'checkpoint_sha256': checkpoint_sha, 'evaluation': evaluation,
                          'fixed_scenario': adapter._course_evaluator.plan.description(),
                          'comparison_scope': 'fixed scheduled stand target; independent cold starts with native births/handoffs/DR, not matched RNG trajectories'}
                if args.checkpoint is not None and file_sha256(args.checkpoint) != checkpoint_sha:
                    raise ValueError('Checkpoint changed during the diagnostic')
                (output / 'posture.json').write_text(json.dumps(report, indent=2) + '\n')
                print(json.dumps({'completed_steps': probe.index, 'policy': args.policy}), flush=True)
        except BaseException:
            (output / 'failure-traceback.txt').write_text(traceback.format_exc())
            raise
        finally:
            finish = getattr(adapter, 'evaluation_end', None)
            if callable(finish):
                finish()
            packed.close()


if __name__ == '__main__':
    main()
