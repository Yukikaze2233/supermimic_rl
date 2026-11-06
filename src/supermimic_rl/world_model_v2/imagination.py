"""Imagined rollouts for BPTT: the actor sees reassembled deployed frames, never clean state.

Each step: actor action from the 5x45 history -> world-model step -> four ticks
pushed through the sensor queues -> delayed noisy 45D frame -> analytic reward
and physical-clock termination on the interval. Recorded command trajectories
advance from delayed heading measurements after the interval's reward. Rollouts
stop before unknown resampling boundaries; frozen rows bootstrap there.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch
from torch import Tensor, nn

from ..formal_observation import advance_goal, deployed_frame, known_horizon
from .dynamics import RunningNorm
from .features import linkage_features, memory_features, physical_features
from .model import Prediction, Ticks, WorldModelV2
from .motion_task import MotionTask, MOTION_TASK_DIM
from .policy import FRAME_DIM, EstimatorActor, elu_mlp
from .records import decode_memory, decode_physical, decode_queues, decode_readout, decode_transforms
from .sensing import encoder_imu_sample, gimbal_sample, heading_sample
from .termination import PoseContactTermination
from .stair_task import StairTaskState, STAIR_FEATURE_DIM
from .stair_source import StairSourceState, TOF_FEATURE_DIM
from .jump_task import JumpTaskState, JUMP_FEATURE_DIM, JUMP_TRIGGER_FEATURE_DIM
from .recovery_task import RecoveryTaskState, RECOVERY_FEATURE_DIM
from .disturbance import DisturbanceState, DISTURBANCE_NOISE_DIM

POLICY_DT_S = .02


class InvalidImagination(RuntimeError):
    """Undefined model/reference rows, detected before advancing a task protocol."""
    def __init__(self, rows, step, reason, *, rejection_reasons=None):
        super().__init__(f'{int(rows.sum())} invalid imagined rows at step {step}: {reason}')
        self.invalid_rows, self.step, self.reason = rows.detach(), step, reason
        self.rejection_reasons = rejection_reasons or {reason: rows.detach()}


@dataclass
class Imagined:
    objective: Tensor
    critic_inputs: Tensor     # [H + 1, B, D] detached
    rewards: Tensor           # [H, B]
    continuation: Tensor      # [H, B]
    values: Tensor            # [H + 1, B]
    active: Tensor            # [H, B]
    boundary: Tensor          # [H, B] bool, last active step of each row
    lengths: Tensor           # [B]
    final_survival: Tensor    # [B]
    reward_terms: dict
    invalid: Tensor           # [B], invalid physical/reference prediction in an active step
    rejection_reasons: dict[str, Tensor]  # masks in the original input batch
    source_rows: Tensor | None = None  # retained indices in the original input batch

    @property
    def usable(self) -> Tensor:
        """Whole trajectories whose actor objective and critic data are defined."""
        return (~self.invalid & torch.isfinite(self.objective)
                & torch.isfinite(self.values).all(0) & torch.isfinite(self.rewards).all(0)
                & torch.isfinite(self.continuation).all(0)
                & torch.isfinite(self.critic_inputs).all(-1).all(0))


class Critic(nn.Module):
    @staticmethod
    def feature_size(goal_dim: int, sim_context_dim: int, *, task_dim: int = 0,
                     terrain: bool = False, stairs: bool = False, jumps: bool = False, recovery: bool = False,
                     jump_trigger: bool = False, disturbances: bool = False) -> int:
        return (30 + 12 + 19 + goal_dim + sim_context_dim + 77 + 3 + 2 + 23 + task_dim
                + 14 * terrain + (STAIR_FEATURE_DIM + TOF_FEATURE_DIM) * stairs + JUMP_FEATURE_DIM * jumps
                + RECOVERY_FEATURE_DIM * recovery + JUMP_TRIGGER_FEATURE_DIM * jump_trigger + 9 * disturbances)

    def __init__(self, input_dim: int, hidden: tuple[int, ...] = (256, 128), *, task_dim: int = 0,
                 terrain: bool = False, stairs: bool = False, jumps: bool = False, recovery: bool = False,
                 input_clip: float | None = None, jump_trigger: bool = False, disturbances: bool = False):
        super().__init__()
        if input_clip is not None and not input_clip > 0:
            raise ValueError('Critic input clip must be positive or None')
        self.task_dim, self.input_clip = task_dim, input_clip
        if stairs and (not terrain or task_dim != MOTION_TASK_DIM):
            raise ValueError('Stair critic requires terrain routes and command task state')
        self.terrain, self.stairs = terrain, stairs
        if jumps and not stairs:
            raise ValueError('Jump critic requires the complete terrain, command and stair arbitration state')
        self.jumps = jumps
        if jump_trigger and not jumps:
            raise ValueError('Jump trigger critic requires the jump task')
        self.jump_trigger = jump_trigger
        if recovery and not jumps:
            raise ValueError('Recovery critic requires complete jump and stair arbitration state')
        self.recovery = recovery
        self.disturbances = disturbances
        self.norm = RunningNorm(input_dim)
        self.net = elu_mlp(input_dim, hidden, 1)

    def forward(self, features: Tensor) -> Tensor:
        normalized = self.norm(features)
        if self.input_clip is not None:
            # Imagined failure clocks and linkage flags leave the start-state
            # statistics by up to 200 sigma; unbounded inputs let bootstrapped
            # values extrapolate and feed back into their own targets.
            normalized = normalized.clamp(-self.input_clip, self.input_clip)
        return self.net(normalized).squeeze(-1)


def _select(active: Tensor, new, old):
    """Row-wise choice between two dataclasses of batched tensors; frozen rows keep ``old``."""
    def pick(a, b):
        return torch.where(active.reshape(-1, *([1] * (a.ndim - 1))), a, b)
    return type(new)(**{f.name: pick(getattr(new, f.name), getattr(old, f.name)) for f in fields(new)})


def _scatter_prediction(rows, current, previous):
    """Replace active predictions; ended rows retain their last defined interval.

    Those held intervals are padding for batched analytic modules only. Their
    rewards, task updates and state updates are excluded by the active mask.
    They never request a new motion prediction beyond an ended trajectory.
    """
    if isinstance(current, Tensor):
        return previous.index_copy(0, rows, current)
    if current is None:
        return None
    return type(current)(**{field.name: _scatter_prediction(rows, getattr(current, field.name),
        getattr(previous, field.name)) for field in fields(current)})


def _take_rows(value, rows):
    """Select the same rows of a tensor-only, possibly nested runtime record."""
    if isinstance(value, Tensor):
        return value[rows]
    if value is None:
        return None
    return type(value)(**{field.name: _take_rows(getattr(value, field.name), rows) for field in fields(value)})


class Imagination:
    def __init__(self, model: WorldModelV2, actor: EstimatorActor, critic: Critic,
                 termination: PoseContactTermination, *, reward, gamma: float = .99,
                 stair_task=None, stair_source=None, jump_task=None, recovery_task=None,
                 bootstrap_gradient: bool = True, disturbance_config=None):
        self.model, self.actor, self.critic, self.gamma = model, actor, critic, gamma
        self.bootstrap_gradient = bootstrap_gradient
        self.termination, self.reward = termination, reward
        if (stair_task is None) != (stair_source is None) or critic.stairs != (stair_task is not None):
            raise ValueError('Stair imagination needs its source, protocol and critic state together')
        if stair_source is not None and stair_source.terrain is not model.terrain:
            raise ValueError('Stair sensor and motion must query the same atlas')
        self.stair_task, self.stair_source = stair_task, stair_source
        if critic.jumps != (jump_task is not None) or jump_task is not None and stair_source is None:
            raise ValueError('Jump imagination needs its physical task, source arbitration and critic state together')
        self.jump_task = jump_task
        if critic.jump_trigger != (jump_task is not None and jump_task.config.trigger is not None):
            raise ValueError('Jump gate rollout and critic causal features must be enabled together')
        if critic.recovery != (recovery_task is not None) or recovery_task is not None and jump_task is None:
            raise ValueError('Recovery imagination needs its lifecycle and complete skill arbitration state')
        self.recovery_task = recovery_task
        self.disturbance_config = disturbance_config
        if (getattr(model, 'disturbances', False) != (disturbance_config is not None)
                or critic.disturbances != (disturbance_config is not None)):
            raise ValueError('Disturbance motion, critic and analytic schedule must be configured together')
        # (body_floor [bodies], base_floor []) from held-out calibration; None leaves loads raw.
        self.contact_floor = None

    def prepare(self, start, horizon):
        """Draw source and sensor noise once, before microbatch splitting/retries."""
        additions = {}
        if self.disturbance_config is not None:
            if 'disturbance_uniform' not in start:
                additions['disturbance_uniform'] = torch.rand(
                    len(start['goal']), horizon, DISTURBANCE_NOISE_DIM, device=start['goal'].device)
            elif (start['disturbance_uniform'].shape[0] != len(start['goal'])
                  or start['disturbance_uniform'].shape[1] < horizon
                  or start['disturbance_uniform'].shape[2:] != (DISTURBANCE_NOISE_DIM,)):
                raise ValueError('Disturbance realization does not cover the imagined horizon')
        if 'observation_noise' not in start:
            like = start['goal'].new_empty(len(start['goal']), horizon, 18)
            additions['observation_noise'] = decode_readout(start['readout']).draw_noise(like)
        elif (start['observation_noise'].shape[0] != len(start['goal'])
              or start['observation_noise'].shape[1] < horizon
              or start['observation_noise'].shape[2:] != (18,)):
            raise ValueError('Recorded sensor realization does not cover the imagined horizon')
        if self.stair_source is not None:
            if ('tof_latency' in start) != ('tof_uniform' in start):
                raise ValueError('Incomplete imagined source realization')
            if 'tof_latency' not in start:
                additions.update(self.stair_source.draw(len(start['goal']), horizon, device=start['goal'].device))
        if self.jump_task is not None:
            if 'jump_uniform' not in start:
                additions['jump_uniform'] = torch.rand(horizon, len(start['goal']),
                                                       device=start['goal'].device).T.contiguous()
            elif (start['jump_uniform'].ndim != 2 or start['jump_uniform'].shape[0] != len(start['goal'])
                  or start['jump_uniform'].shape[1] < horizon):
                raise ValueError('Jump request realization does not cover the imagined horizon')
        if self.recovery_task is not None and self.recovery_task.config.handoff is not None:
            if 'recovery_uniform' not in start:
                additions['recovery_uniform'] = torch.rand(
                    horizon, len(start['goal']), 2, device=start['goal'].device).transpose(0, 1).contiguous()
            elif (start['recovery_uniform'].shape[0] != len(start['goal'])
                  or start['recovery_uniform'].shape[1] < horizon
                  or start['recovery_uniform'].shape[2:] != (2,)):
                raise ValueError('Recovery handoff realization does not cover the imagined horizon')
        return {**start, **additions} if additions else start

    def critic_features(self, state, height, scan, memory, goal, sim_context, failure_memory,
                        conditions, reward_memory, motion_task=None, *, terrain_route=None,
                        stair_state=None, stair_source=None, jump_state=None, recovery_state=None,
                        disturbance_state=None) -> Tensor:
        linkage, _ = linkage_features(self.model.inputs.linkage, state)
        features = (physical_features(state, height), linkage, memory_features(memory, state), goal,
                    sim_context, scan, failure_memory, conditions, reward_memory)
        if self.critic.task_dim:
            if motion_task is None or self.critic.task_dim != MOTION_TASK_DIM:
                raise ValueError('The task-aware critic needs the complete recorded command state')
            features += (MotionTask.decode(motion_task).critic_features(state),)
        elif motion_task is not None:
            raise ValueError('Dynamic task rollouts require a task-aware critic')
        if self.critic.terrain:
            if terrain_route is None:
                raise ValueError('Terrain critic requires native family and kind')
            features += (terrain_route,)
        if self.critic.stairs:
            features += (stair_state.critic_features(),
                         stair_source.critic_features(state, stair_state, MotionTask.decode(motion_task)))
        if self.critic.jumps:
            if jump_state is None:
                raise ValueError('Jump critic needs the recorded native request and physical evidence')
            features += (jump_state.critic_features(state, MotionTask.decode(motion_task), self.jump_task.config),)
        if self.critic.recovery:
            if recovery_state is None:
                raise ValueError('Recovery critic needs the recorded native lifecycle')
            features += (recovery_state.critic_features(self.recovery_task.config),)
        if self.critic.jump_trigger:
            features += (jump_state.trigger_critic_features(
                MotionTask.decode(motion_task), self.jump_task.config),)
        if self.critic.disturbances:
            if disturbance_state is None:
                raise ValueError('Disturbance critic needs current held wrench and scheduler state')
            features += (disturbance_state.encode(),)
        return torch.cat(features, -1)

    @staticmethod
    def task_features(start):
        result = {}
        if 'task_family' in start:
            result['terrain_route'] = torch.cat((
                torch.nn.functional.one_hot(start['task_family'][:, 0].long(), 6),
                torch.nn.functional.one_hot(start['reward_route'][:, 1].long(), 8)), -1).to(start['goal'])
        if 'tof_ids' in start:
            result.update(stair_state=StairTaskState.decode(start), stair_source=StairSourceState.decode(start))
        if 'jump_ids' in start:
            result['jump_state'] = JumpTaskState.decode(start)
        if 'recovery_flags' in start:
            result['recovery_state'] = RecoveryTaskState.decode(start)
        if 'disturbance' in start:
            result['disturbance_state'] = DisturbanceState.decode(start['disturbance'])
        return result

    @torch.no_grad()
    def _critic_targets(self, *args, **kwargs):
        """Intermediate values train the critic; only the final bootstrap trains the actor.

        Ended rows retain their differentiable boundary state until the final
        critic call, so this also preserves the actor's early-boundary gradient.
        """
        features = self.critic_features(*args, **kwargs)
        return features, self.critic(features)

    def rollout(self, start: dict[str, Tensor], frames: Tensor, horizon: int, *,
                screen_invalid: bool = False, isolate_terrain: bool = False) -> Imagined:
        """Advance known intervals; optionally retain only atlas-defined trajectories.

        ``isolate_terrain`` compacts every state/history field before analytic
        processing and returns original indices in ``source_rows``. It is not
        a recovery path for nonfinite dynamics or unknown reward references.
        """
        from .reward_assembly import RewardMemory

        if screen_invalid and isolate_terrain:
            raise ValueError('Screening and differentiable terrain isolation are separate paths')
        if screen_invalid and torch.is_grad_enabled():
            raise ValueError('Invalid-trajectory screening requires torch.no_grad()')
        start = self.prepare(start, horizon)
        model, actor = self.model, self.actor
        task_features = self.task_features(start)
        stair = task_features.get('stair_state')
        source = task_features.get('stair_source')
        jump = task_features.get('jump_state')
        recovery = task_features.get('recovery_state')
        disturbance = task_features.get('disturbance_state')
        if (disturbance is None) != (self.disturbance_config is None):
            raise ValueError('Imagined disturbance state and configured schedule differ')
        if disturbance is not None and not (start['disturbance_record_valid'] > .5).all():
            raise ValueError('Imagination requires recorded disturbance clocks, not migrated placeholders')
        if (recovery is None) != (self.recovery_task is None):
            raise ValueError('Imagined recovery state and configured lifecycle differ')
        if (jump is None) != (self.jump_task is None):
            raise ValueError('Imagined jump state and configured task differ')
        if (source is None) != (self.stair_source is None):
            raise ValueError('Imagined source state and configured source differ')
        if source is not None and ('tof_latency' not in start or 'tof_uniform' not in start):
            raise ValueError('Prepare one source realization before microbatching imagined starts')
        if 'task_record_valid' in start and not (start['task_record_valid'] > .5).all():
            raise ValueError('Imagination cannot invent unrecorded command-controller state')
        if 'stair_source_valid' in start and not (start['stair_source_valid'] > .5).all():
            raise ValueError('Imagination cannot invent unrecorded stair-source state')
        if jump is not None and ('jump_record_valid' not in start or not (start['jump_record_valid'] > .5).all()):
            raise ValueError('Imagination cannot invent unrecorded jump state')
        if recovery is not None and ('recovery_record_valid' not in start
                or not (start['recovery_record_valid'] > .5).all()):
            raise ValueError('Imagination cannot invent unrecorded recovery state')
        history = frames[..., :FRAME_DIM].detach()
        state, memory = decode_physical(start["physical"]), decode_memory(start["memory"])
        reward_context = self.reward.bind(start, state)
        reward_memory = RewardMemory.decode(torch.cat((start['reward_memory'], start['support_memory']), -1))
        height, scan, goal = start["height"], start["scan"], start["goal"]
        task = MotionTask.decode(start['motion_task']) if 'motion_task' in start else None
        sim_context, strength = start["sim_context"], start["motor_strength"][..., 4:6]
        queues, transforms = decode_queues(start["queues"]), decode_transforms(start["transforms"])
        readout = decode_readout(start["readout"])
        failure_memory, conditions = start["failure_memory"], start["failure_conditions"]
        if not (start["failure_conditions_valid"] > .5).all():
            raise ValueError("Imagination requires recorded recovery/contact-exception semantics")
        dynamic = start['horizon'][..., 1] if task is None else torch.zeros_like(start['horizon'][..., 1])
        lengths = known_horizon(start["horizon"][..., 0], dynamic, goal,
                                POLICY_DT_S).clamp_max(horizon)
        if source is not None:
            known = source.boundary_known(task, stair, POLICY_DT_S, jump)
            lengths = torch.where(known, horizon, 0).long()
        if horizon < 1 or (lengths < 1).any():
            raise ValueError("Imagination starts must precede a known complete control interval")
        steps = int(lengths.max())
        survival = torch.ones(goal.shape[0], device=goal.device)
        objective = torch.zeros_like(survival)
        invalid = torch.zeros_like(survival, dtype=torch.bool)
        rejection_reasons = {}
        input_count = len(goal)
        source_rows = torch.arange(input_count, device=goal.device)

        def record_rejection(rows, reason):
            original = torch.zeros(input_count, dtype=torch.bool, device=goal.device)
            original[source_rows[rows]] = True
            rejection_reasons[reason] = rejection_reasons.get(reason, torch.zeros_like(original)) | original

        def reject(rows, step, reason):
            if rows.any():
                record_rejection(rows, reason)
                if not screen_invalid:
                    # A later nonfinite/reference failure may follow an earlier
                    # terrain removal. A retry must also omit those removed rows.
                    rejected = torch.stack(tuple(rejection_reasons.values())).any(0)
                    raise InvalidImagination(rejected, step, reason, rejection_reasons=rejection_reasons)
                invalid.logical_or_(rows)

        previous_prediction = None
        inputs, rewards, continues, values, actives, boundaries, terms_log = [], [], [], [], [], [], {}
        for step in range(steps):
            active = (step < lengths) & (survival > 0) & ~invalid
            if not active.any():
                break
            features, value = self._critic_targets(
                state, height, scan, memory, goal, sim_context, failure_memory,
                conditions, reward_memory.encode(), None if task is None else task.encode(), **task_features)
            inputs.append(features.detach())
            values.append(value)
            action = actor(history)
            terrain = ({'terrain_tile': start['terrain_tile'][:, 0]} if 'terrain_tile' in start else {})
            next_disturbance = None
            if disturbance is not None:
                # Imagined starts are policy-owned: recovery preparation has
                # ended. Goal flags retain the admitted native command mode.
                eligible = (goal[:, 4:7] > .5).any(-1) & active
                push, wrench, next_disturbance = disturbance.advance_prepared(
                    self.disturbance_config, eligible, start['disturbance_uniform'][:, step],
                    policy_dt_s=POLICY_DT_S)
                terrain.update(velocity_push=push, external_wrench=wrench)
            if active.all():
                prediction = model.step(state, height, scan, memory, sim_context, strength, action, **terrain)
            else:
                rows = active.nonzero().flatten()
                selected_memory = type(memory)(**{f.name: getattr(memory, f.name)[rows] for f in fields(memory)})
                prediction = model.step(state.map(lambda value: value[rows]), height[rows], scan[rows],
                    selected_memory, sim_context[rows], strength[rows], action[rows],
                    **{name: value[rows] for name, value in terrain.items()})
                prediction = _scatter_prediction(rows, prediction, previous_prediction)
            finite = torch.isfinite(prediction.relative_ticks).flatten(1).all(-1)
            for field in fields(prediction.ticks.state):
                finite &= torch.isfinite(getattr(prediction.ticks.state, field.name)).flatten(1).all(-1)
            reject(active & ~finite, step, 'nonfinite_prediction')
            if prediction.tick_terrain_valid is not None:
                missing = active & ~invalid & ~prediction.tick_terrain_valid.all(-1)
                if isolate_terrain and missing.any():
                    # Motion is finite here; only the atlas has no surface. Drop
                    # the entire trajectory before NaN heights/scans can enter
                    # rewards, sensor history or a subsequent model invocation.
                    # Indexing keeps the healthy prefix's graph, unlike replay.
                    record_rejection(missing, 'terrain_query')
                    keep = (~missing).nonzero().flatten()
                    source_rows = source_rows[keep]
                    if not len(keep):
                        empty = goal.new_empty(0)
                        return Imagined(empty, features.new_empty(1, 0, features.shape[-1]),
                            empty.reshape(0, 0), empty.reshape(0, 0), empty.reshape(1, 0),
                            empty.reshape(0, 0).bool(), empty.reshape(0, 0).bool(), lengths[:0],
                            empty, {}, invalid[:0], rejection_reasons, source_rows)
                    start = {name: value[keep] for name, value in start.items()}
                    (history, height, scan, goal, sim_context, strength, failure_memory, conditions,
                     lengths, survival, objective, invalid, active, action) = (
                        value[keep] for value in (history, height, scan, goal, sim_context, strength,
                            failure_memory, conditions, lengths, survival, objective, invalid, active, action))
                    (state, memory, reward_memory, queues, transforms, task, stair, source, jump, recovery,
                     disturbance, next_disturbance,
                     prediction) = (_take_rows(value, keep) for value in (
                        state, memory, reward_memory, queues, transforms, task, stair, source, jump, recovery,
                        disturbance, next_disturbance,
                        prediction))
                    readout = decode_readout(start['readout'])
                    task_features = {name: _take_rows(value, keep) for name, value in task_features.items()}
                    # Bind from the original physical start so fixed reference
                    # ground is unchanged; restore the current protocol conditions.
                    reward_context = self.reward.bind(
                        start, decode_physical(start['physical'])).with_conditions(conditions)
                    inputs = [value[keep] for value in inputs]
                    values = [value[keep] for value in values]
                    rewards = [value[keep] for value in rewards]
                    continues = [value[keep] for value in continues]
                    actives = [value[keep] for value in actives]
                    boundaries = [value[keep] for value in boundaries]
                    terms_log = {name: value[keep] for name, value in terms_log.items()}
                else:
                    reject(missing, step, 'terrain_query')
            if screen_invalid and (active & invalid).any():
                # Undefined rows cannot enter queues or task protocols. These
                # finite held intervals are padding exclusively for no-grad
                # screening; the entire rejected trajectory is discarded.
                relative = torch.zeros_like(prediction.relative_ticks)
                held = Prediction(state, Ticks.decode(relative, state), memory, height, scan,
                    torch.zeros_like(prediction.linkage_inside), torch.zeros_like(prediction.increment),
                    relative, height[:, None].expand(-1, relative.shape[1], -1),
                    torch.ones_like(prediction.tick_terrain_valid) if prediction.tick_terrain_valid is not None else None)
                rows = (~invalid).nonzero().flatten()
                prediction = _scatter_prediction(rows, _take_rows(prediction, rows), held)
            active = active & ~invalid
            if self.contact_floor is not None:
                prediction = replace(prediction, ticks=prediction.ticks.with_contact_floor(*self.contact_floor))
            previous_prediction = prediction
            ticks, end = prediction.ticks, prediction.state
            count = ticks.leg_position.shape[-2]
            angular_velocity_body, yaw = ticks.angular_velocity_body, ticks.yaw
            for tick in range(count):
                leg_q = ticks.leg_position[..., tick, :]
                velocity = ticks.joint_velocity[..., tick, :]
                gyro = angular_velocity_body[..., tick, :]
                gravity = ticks.gravity_body[..., tick, :]
                gimbal_q = ticks.gimbal_position[..., tick, :]
                queues = queues.push(encoder_imu_sample(leg_q, velocity[..., :6], gyro, gravity, transforms),
                                     gimbal_sample(gimbal_q, velocity[..., 6:8], transforms),
                                     heading_sample(yaw[..., tick]))
            estimate = None
            if source is not None and active.any():
                # The interval is complete, but the controller uses the prior
                # noisy actor observation, not its newly predicted physical state.
                with torch.no_grad():
                    estimate = self.stair_task.completed_estimate(history, actor.estimate(history), stair.clock[:, 0])
            physical = reward_context.integrate(state, prediction, action, reward_memory, memory, goal, step,
                **({} if jump is None else {'jump_state': jump, 'base_mode': task.route[:, 3].long()}))
            reject(active & physical.invalid, step, 'reward_reference')
            active = active & ~invalid
            termination_height = (prediction.tick_height_above_ground if physical.termination_height is None
                                  else physical.termination_height)
            observed_recovery = None
            if recovery is not None:
                observed_recovery = self.recovery_task.observe_interval(recovery, physical=ticks.state,
                    wheel_support_load=ticks.wheel_load.clamp_min(0),
                    base_normal_load=ticks.body_normal_load[..., 0].clamp_min(0),
                    base_com_offset=start['sim_context'][:, 41:44], command_height=goal[:, 3] / 5.,
                    ground_z=ticks.state.position[..., 2] - prediction.tick_height_above_ground[..., 0],
                    policy_owned=active)
            interval = self.termination.advance(failure_memory, termination_height,
                                                ticks.gravity_body, ticks.forbidden_load,
                                                recovery=conditions[:, 0] > .5,
                                                body_contact_allowed=conditions[:, 1] > .5)
            recovery_boundary = None
            if recovery is not None:
                recovery_boundary = self.recovery_task.finish_interval(observed_recovery,
                    now_s=self.recovery_task.boundary_time(recovery), estimate=estimate,
                    policy_owned=active, failed=interval.failed)
                recovery = _select(active, recovery_boundary.state, recovery)
                task_features['recovery_state'] = recovery
                # Native recovery timeout is a time-limit endpoint. Keep its
                # next-decision bootstrap and do not invent a failure payment.
                lengths = torch.where(active & recovery_boundary.timeout, step + 1, lengths)
            success = torch.zeros_like(active) if physical.success_credited is None else physical.success_credited
            reward = physical.finish(interval.first_failure, success, reward_context.assembly.parameters)
            failure = interval.failed.to(reward.dtype)
            # One failure event at the failing step; survival removes everything after it.
            reward = torch.where(active, reward, 0.)
            continuation = torch.where(active, 1 - failure, 1.)
            objective = objective + self.gamma ** step * survival * reward
            survival = survival * continuation
            rewards.append(reward)
            continues.append(continuation)
            actives.append(active)
            for name, value in physical.components.items():
                terms_log[name] = terms_log.get(name, 0.) + value.detach() / steps
            if source is not None and active.any():
                if recovery_boundary is not None:
                    handoff = self.recovery_task.apply_handoff(recovery_boundary, task,
                        task_family=start['task_family'][:, 0].long(), segment_end=source.segment_end,
                        handoff_uniform=(start['recovery_uniform'][:, step]
                                         if 'recovery_uniform' in start else None))
                    task = handoff.motion
                    source = replace(source, segment_end=handoff.segment_end)
                    start = {**start, 'task_family': handoff.task_family[:, None].to(start['task_family'])}
                    reward_context = replace(reward_context, task_family=start['task_family'][:, 0])
                    task_features['terrain_route'] = torch.cat((
                        torch.nn.functional.one_hot(handoff.task_family, 6),
                        torch.nn.functional.one_hot(start['reward_route'][:, 1].long(), 8)), -1).to(goal)
                wheel = reward_context.geometry(end,
                    passive_knee_position=ticks.passive_knee_position[:, -1],
                    passive_knee_velocity=ticks.passive_knee_velocity[:, -1])
                radius = goal.new_tensor([side.radius for side in reward_context.geometry.sides])
                rows = active.nonzero().flatten()
                jump_inputs = {}
                if jump is not None:
                    following, admitted = _take_rows(task, rows).advance_admission(
                        _take_rows(queues, rows), decode_readout(start['readout'][rows]), POLICY_DT_S)
                    physical_jump = _take_rows(physical.jump_state, rows)
                    ground = end.position[rows, 2] - prediction.height_above_ground[rows, 0]
                    jumped = self.jump_task.advance(physical_jump, now_s=self.jump_task.boundary_time(physical_jump),
                        estimate=_take_rows(estimate, rows), admitted=admitted, base_mode=following.route[:, 3].long(),
                        task_family=start['task_family'][rows, 0].long(), elapsed_s=following.timing[:, 0],
                        ground_z=ground, blocked_by_stair=stair.ids[rows, 1] >= 0,
                        interval_uniform=start['jump_uniform'][rows, step],
                        previous_command_height=task.current[rows, 3], previous_elapsed_s=task.timing[rows, 0])
                    # Native tasks latch the accepted baseline into the command
                    # stream before stair assistance and gimbal eligibility.
                    following = replace(following, current=jumped.command)
                    jump_inputs = {'admission': (following, jumped.command), 'base_mode': jumped.mode,
                        'jump_active': jumped.state.active_id >= 0, 'request_height': jumped.request_height,
                        'request_clock': jumped.request_clock}
                    jump = _scatter_prediction(rows, jumped.state, jump)
                advanced = self.stair_source.advance(_take_rows(source, rows), _take_rows(stair, rows),
                    task=self.stair_task, motion=_take_rows(task, rows), physical=_take_rows(end, rows),
                    wheel_centres=wheel.centre_world[rows], wheel_radius=radius,
                    terrain_tile=start['terrain_tile'][rows, 0], queues=_take_rows(queues, rows),
                    readout=decode_readout(start['readout'][rows]), goal=goal[rows], estimate=_take_rows(estimate, rows),
                    physics_dt_s=model.physics_dt_s, latency=start['tof_latency'][rows, step],
                    draws=start['tof_uniform'][rows, step], **jump_inputs)
                boundary = advanced.boundary
                task = _scatter_prediction(rows, boundary.motion, task)
                stair = _scatter_prediction(rows, boundary.stair.state, stair)
                source = _scatter_prediction(rows, advanced.source, source)
                conditions = conditions.index_copy(0, rows, boundary.failure_conditions)
                reward_context = reward_context.with_conditions(conditions)
                next_goal = goal.index_copy(0, rows, boundary.goal)
                task_features.update(stair_state=stair, stair_source=source)
                if jump is not None:
                    task_features['jump_state'] = jump
                known = source.boundary_known(task, stair, POLICY_DT_S, jump)
                lengths = torch.where(active & ~known, step + 1, lengths)
            elif source is not None:
                next_goal = goal
            elif task is None:
                next_goal = advance_goal(goal, POLICY_DT_S)
            else:
                following_task, next_goal = task.advance(queues, readout, goal, POLICY_DT_S)
                task = MotionTask.decode(torch.where(active[:, None], following_task.encode(), task.encode()))
            boundaries.append(step + 1 == lengths)
            frame = deployed_frame(queues.values, queues.count, queues.gimbal, action, next_goal, readout,
                                   noise_draw=start['observation_noise'][:, step])[..., :FRAME_DIM]
            keep = active[:, None]
            history = torch.where(keep[..., None], torch.cat((history[:, 1:], frame[:, None]), 1), history)
            reward_memory = RewardMemory.decode(torch.where(keep, physical.memory.encode(), reward_memory.encode()))
            state = _select(active, end, state)
            height = torch.where(keep, prediction.height_above_ground, height)
            scan = torch.where(keep, prediction.terrain_scan, scan)
            memory = _select(active, prediction.memory, memory)
            failure_memory = torch.where(keep, interval.memory, failure_memory)
            goal = torch.where(keep, next_goal, goal)
            if disturbance is not None:
                disturbance = _select(active, next_disturbance, disturbance)
                task_features['disturbance_state'] = disturbance
            if source is not None and not ((step + 1 < lengths) & (survival > 0) & ~invalid).any():
                break
        final = self.critic_features(state, height, scan, memory, goal, sim_context, failure_memory,
                                     conditions, reward_memory.encode(), None if task is None else task.encode(), **task_features)
        inputs.append(final.detach())
        # The critic fits values, not input gradients; on clock/flag features
        # with near-constant start statistics its gradient misdirects the actor.
        values.append(self.critic(final if self.bootstrap_gradient else final.detach()))
        objective = objective + self.gamma ** lengths * survival * values[-1]
        objective = torch.where(invalid, torch.full_like(objective, float('nan')), objective)
        return Imagined(objective, torch.stack(inputs), torch.stack(rewards), torch.stack(continues),
                        torch.stack(values), torch.stack(actives), torch.stack(boundaries), lengths,
                        survival, {name: value.mean() for name, value in terms_log.items()},
                        invalid, rejection_reasons, source_rows)
