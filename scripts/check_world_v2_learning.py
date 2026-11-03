#!/usr/bin/env python3
"""Exercise V2 fitting, BPTT gradients and optimizer restoration on native replay.

The small networks and few updates here validate the learning path, not model
accuracy or robot skill. Complete finite differences include the sensor queues,
closed-chain table, estimator, critic and four-substep motion model.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch

from supermimic_rl.formal_observation import deployed_frame, known_horizon
from supermimic_rl.world_model_v2.execution import WheelExecution
from supermimic_rl.world_model_v2.features import MotionInputs
from supermimic_rl.world_model_v2.imagination import Critic
from supermimic_rl.world_model_v2.imagined_reward import OrdinaryFlatReward
from supermimic_rl.world_model_v2.geometry import RewardGeometry
from supermimic_rl.world_model_v2.ingest import load_recorded_replay
from supermimic_rl.world_model_v2.learning import Learner, LearningConfig, frozen_parameters
from supermimic_rl.world_model_v2.leg_linkage import LegLinkage
from supermimic_rl.world_model_v2.model import SIM_CONTEXT_DIM, WorldModelV2
from supermimic_rl.world_model_v2.motion_task import MotionTask, MOTION_TASK_DIM
from supermimic_rl.world_model_v2.policy import EstimatorActor
from supermimic_rl.world_model_v2.records import MODEL_CONTRACT, decode_physical, decode_queues, decode_readout
from supermimic_rl.world_model_v2.termination import PoseContactLimits, PoseContactTermination
from supermimic_rl.world_model_v2 import so3


def make_learner(runtime, schema, device, *, motion_task=False):
    model = WorldModelV2(MotionInputs(LegLinkage.load("configs/linkage/leg_linkage_e726d23c.json")),
                         WheelExecution.from_bundle("configs/identified/20261031", .005), hidden=(64, 64)).to(device)
    actor = EstimatorActor(estimator_hidden=(32,), actor_hidden=(64, 32)).to(device)
    task_dim = MOTION_TASK_DIM if motion_task else 0
    critic = Critic(Critic.feature_size(16, SIM_CONTEXT_DIM, task_dim=task_dim), (64, 32),
                    task_dim=task_dim).to(device)
    termination = PoseContactTermination(PoseContactLimits.from_runtime(runtime), .005)
    reward = OrdinaryFlatReward(json.loads(schema.source),
        RewardGeometry(model.inputs.linkage, 'assets/full_vehicle_v2p0p1_source_audited_v2'),
        model.wheels, schema.policy_dt_s)
    return Learner(model, actor, critic, termination, LearningConfig(critic_epochs=1, critic_minibatches=2), reward=reward)


def with_motion_fixture(start):
    """Synthetic command cases on native physical states, never training data/DR.

    Native task parity is checked separately against all seven trajectories.
    Here a mixed batch exercises their composition with sensors, reward and BPTT.
    The fixture envelope and rates are test inputs, not claimed hardware values.
    """
    start = {name: value.clone() for name, value in start.items()}
    goal, n = start['goal'], len(start['goal'])
    rows = torch.arange(n, device=goal.device)
    command = goal[:, :4].clone()
    command[:, 3] /= 5.
    special = rows >= n // 2
    route = torch.stack((rows % 7, rows * 0, rows * 0 + 1, rows * 0, rows * 0 + 1), -1).to(goal)
    timing = goal.new_tensor([6., 0., 2., .5, 1.]).expand(n, -1)
    heading = goal.new_zeros(n, 6)
    heading[:, 0] = (rows % 2 == 1).to(goal)
    heading[:, 1] = 1.
    heading[:, 2] = so3.yaw(decode_physical(start['physical']).rotation) + .1
    heading[:, 4:] = goal.new_tensor([1.3, 2.5])
    gimbal = goal.new_tensor([0., .3, .25, .4, 5.]).repeat(n, 1)
    gimbal[:, 0] = special.to(goal)
    task = MotionTask(command, command + goal.new_tensor([.3, 0., .2, .01]), command, timing, route,
                      heading, goal.new_tensor([.08, .38, 90.]).expand(n, -1),
                      goal.new_zeros(n, 4), gimbal)
    goal[:, 4:11] = 0.
    goal[:, 4] = 1.
    goal[special, :2] = 0.
    goal[special, 2] = gimbal[special, 3]
    goal[:, 11:16] = torch.stack((~special, special, special * .3,
                                  special * gimbal[:, 2].sin(), special * gimbal[:, 2].cos()), -1)
    start['reward_route'][:, 2] = special.to(goal)
    start['horizon'][:, 1] = 1.  # Would be capped to H1 without recorded task state.
    start['motion_task'] = task.encode()
    queues = decode_queues(start['queues'])
    frame = deployed_frame(queues.values, queues.count, queues.gimbal, start['action'], goal,
                           decode_readout(start['readout']), noise=False)
    return start, frame[:, None].expand(-1, 5, -1).clone()


def gradient_check(learner, start, frames):
    modules = (learner.model, learner.actor, learner.critic)
    for module in modules:
        module.double()
    start = {name: value.double() if value.is_floating_point() else value for name, value in start.items()}
    frames = frames.double()
    parameter = learner.actor.decoder[-1].bias
    baseline = parameter.detach().clone()

    def objective():
        torch.manual_seed(731)
        return learner.imagination.rollout(start, frames, 16).objective.mean()

    try:
        with frozen_parameters(learner.model, learner.critic):
            gradient, = torch.autograd.grad(objective(), parameter)
            direction = gradient.detach() / gradient.norm().clamp_min(1e-12)
            analytic = (gradient * direction).sum()
            eps = 1e-5
            with torch.no_grad():
                parameter.copy_(baseline + eps * direction)
                positive = objective()
                parameter.copy_(baseline - eps * direction)
                negative = objective()
                numerical = (positive - negative) / (2 * eps)
            relative = (analytic - numerical).abs() / torch.maximum(analytic.abs(), numerical.abs()).clamp_min(1e-10)
            result = {"analytic": float(analytic), "finite_difference": float(numerical),
                      "relative_error": float(relative), "horizon": 16, "epsilon": eps}
            if not torch.isfinite(relative) or analytic.abs() < 1e-8 or relative > 1e-3:
                raise AssertionError(f"BPTT finite difference failed: {result}")
            return result
    finally:
        with torch.no_grad():
            parameter.copy_(baseline)
        for module in modules:
            module.float()


def microbatch_check(learner, start, frames):
    whole, chunked = deepcopy(learner), deepcopy(learner)
    start = {name: value.clone() for name, value in start.items()}
    # Preserve one actual sensor realization across both accumulation orders.
    start = learner.imagination.prepare(start, learner.config.imagination_horizon)
    full = whole.update_policy(start, frames)
    micro = chunked.update_policy(start, frames, microbatch_size=3)
    difference = 0.
    for a, b in zip(whole.actor.decoder.parameters(), chunked.actor.decoder.parameters()):
        torch.testing.assert_close(a, b, atol=2e-7, rtol=1e-5)
        difference = max(difference, float((a - b).abs().max().detach()))
    return {'batch': len(frames), 'microbatch': 3, 'actor_max_parameter_difference': difference,
            'loss_difference': abs(full['actor/loss'] - micro['actor/loss']),
            'whole_updates': whole.updates['actor'], 'microbatch_updates': chunked.updates['actor']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--runtime", type=Path, default=Path("configs/gpufree/runtime-physx64-16-geometry-audit.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--motion-task-fixture", action="store_true",
                        help="Check H16 dynamic/heading/gimbal composition using explicit synthetic task inputs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(7)
    replay, schema = load_recorded_replay(args.replay, args.device)
    runtime = json.loads(args.runtime.read_text())
    learner = make_learner(runtime, schema, args.device, motion_task=args.motion_task_fixture)
    fitted = [learner.fit_model(replay.sample_windows(32, 4)) for _ in range(4)]
    start, frames = replay.sample_starts(256, 5)
    keep = known_horizon(start["horizon"][:, 0], start["horizon"][:, 1], start["goal"], schema.policy_dt_s) >= 16
    keep &= (start['reward_context_valid'][:, 0] > .5) & (start['reward_route'] == 0).all(-1)
    start, frames = {name: value[keep][:8] for name, value in start.items()}, frames[keep][:8]
    if len(frames) != 8:
        raise ValueError("Need at least eight known 16-step start samples")
    task_report = None
    if args.motion_task_fixture:
        start, frames = with_motion_fixture(start)
        with torch.no_grad():
            imagined = learner.imagination.rollout(start, frames, 16)
        if not (imagined.lengths == 16).all():
            raise AssertionError('Known dynamic trajectories were incorrectly truncated to H1')
        goal_offset = 30 + 12 + 19
        changes = (imagined.critic_inputs[1, :, goal_offset:goal_offset + 16] - start['goal']).abs().amax(-1)
        if not (changes > 0).any():
            raise AssertionError('Recorded trajectories did not advance inside the rollout')
        task_report = {'scope': 'synthetic task inputs on native physical states; not a native dynamic-course evaluation',
                       'record_dim': MOTION_TASK_DIM, 'critic_dim': learner.critic.norm.mean.numel(),
                       'requested_lengths': imagined.lengths.tolist(), 'rows_with_changed_next_command': int((changes > 0).sum())}
    policy = learner.update_policy(start, frames)
    gradients = gradient_check(learner, start, frames)
    accumulation = microbatch_check(learner, start, frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output.with_suffix(".pt")
    torch.save(learner.state_dict(), checkpoint)
    restored = make_learner(runtime, schema, args.device, motion_task=args.motion_task_fixture)
    restored.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=True))
    for name in ("model", "actor", "critic"):
        for key, value in getattr(learner, name).state_dict().items():
            torch.testing.assert_close(value, getattr(restored, name).state_dict()[key])
    resumed = restored.update_policy(start, frames)
    report = {"scope": "learning-path correctness on recorded native data; not skill or world-model accuracy",
              "model_contract": {"version": MODEL_CONTRACT,
                                 "input_dim": learner.model.motion.input_norm.mean.numel(),
                                 "output_dim": learner.model.motion.output_dim},
              "fitting": fitted, "policy": policy, "gradient": gradients, "microbatch": accumulation,
              "resumed_policy": resumed,
              "optimizer_updates": restored.updates, "skipped": restored.skipped}
    if task_report is not None:
        report['motion_task'] = task_report
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
