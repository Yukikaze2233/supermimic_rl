"""Short deterministic-policy rollouts in the checkpoint's own environment."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .adapters.packed import PackedEnvironment
from .learning import Learner, LearningConfig


@dataclass(frozen=True)
class NativeTrackingSample:
    actual: torch.Tensor
    target: torch.Tensor
    gimbal: dict[str, torch.Tensor] | None = None
    course: dict[str, torch.Tensor] | None = None

    def __iter__(self):
        # Preserve the two-value tracking interface for ordinary environments.
        return iter((self.actual, self.target))


def terminal_failure_masks(schema, transition):
    """Count packed pre-reset failure bitmasks, preserving combined reasons."""
    offset = 0
    for field in schema.state_fields:
        if field.name == "failure.failure_reasons":
            if field.size != 1:
                raise ValueError("failure.failure_reasons must be a scalar bitmask")
            values = transition.end.state[transition.terminated, offset]
            if (not torch.isfinite(values).all() or (values < 0).any()
                    or (values != values.round()).any()):
                raise ValueError("Invalid packed terminal failure bitmask")
            masks, counts = values.to(torch.int64).unique(return_counts=True)
            return {str(mask): count for mask, count in
                    zip(masks.cpu().tolist(), counts.cpu().tolist())}
        offset += field.size
    return None


def native_tracking_sample(environment):
    """Read named SI measurements using the pinned native observation assembler.

    The heading axes match upstream formal/tasks/reward.py: body x projected
    onto the horizontal plane, and yaw about world up. Measurement is BEFORE
    the action, since the environment may auto-reset before step() returns.
    """
    if not isinstance(environment, PackedEnvironment):
        return None
    native = getattr(environment.adapter, "env", None)
    if native is None:
        raise ValueError("Native tracking requires the adapter's audited environment")
    sample = native._sample()
    measured = native._observations.assembler.measure(sample)
    command = native._commands().admitted_body_command
    velocity = measured.velocity_origin_body_m_s
    angular = measured.angular_velocity_body_rad_s
    up = -measured.gravity_body
    height = measured.height_above_ground_m
    count = environment.num_envs
    for name, value, shape in (
        ("origin velocity", velocity, (count, 3)),
        ("angular velocity", angular, (count, 3)),
        ("gravity", up, (count, 3)), ("height", height, (count,)),
        ("admitted command", command, (count, 4)),
    ):
        if not isinstance(value, torch.Tensor) or value.shape != shape:
            raise ValueError(f"Unexpected named native {name} shape")
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite native {name}")
    if (up.norm(dim=-1) - 1).abs().gt(2e-4).any():
        raise ValueError("Native tracking requires the normalized measured gravity")
    forward = torch.zeros_like(up)
    forward[:, 0] = 1
    forward -= (forward * up).sum(-1, keepdim=True) * up
    forward /= forward.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    actual = torch.stack(((velocity * forward).sum(-1),
                          (angular * up).sum(-1), height), dim=-1)
    # These are named native command fields in SI units, not actor feature slots.
    target = command[:, [0, 2, 3]]
    gimbal = None
    course = getattr(native, "_gimbal_course", None)
    if course is not None:
        measurement = native.gimbal_measurement()
        # Same root-origin velocity and yaw frame as native gimbal reward;
        # the view ray's projected heading also contains pitch/tilt and differs.
        quaternion = sample.root_link_orientation_xyzw
        twice_cross = 2 * torch.linalg.cross(quaternion[:, :3], velocity)
        world_velocity = (velocity + quaternion[:, 3:] * twice_cross
                          + torch.linalg.cross(quaternion[:, :3], twice_cross))
        yaw = measurement.task_heading
        vx, vy = world_velocity[:, 0], world_velocity[:, 1]
        gimbal = {
            "native_position_rad": measurement.native_position,
            "native_velocity_rad_s": measurement.native_velocity,
            "controller_position_rad": measurement.joint_angles,
            "controller_velocity_rad_s": measurement.native_velocity * native._gimbal._sign,
            "view_world_angles_rad": measurement.world_angles,
            "view_relative_yaw_rad": measurement.relative_yaw[:, None],
            "task_heading_rad": yaw[:, None],
            "pitch_command_rad": course.target[:, 1:2],
            "yaw_rate_command_rad_s": course.rate[:, None],
            "chassis_yaw_command_rad_s": course.chassis_rate[:, None],
            "translation_actual_m_s": torch.stack((yaw.cos() * vx + yaw.sin() * vy,
                                                     -yaw.sin() * vx + yaw.cos() * vy), -1),
            "translation_target_m_s": course.speed[:, None] * torch.stack(
                (course.direction.cos(), course.direction.sin()), -1),
            "translation_active": course.special,
        }
        for name, value in gimbal.items():
            width = 2 if name in {
                "native_position_rad", "native_velocity_rad_s", "controller_position_rad",
                "controller_velocity_rad_s", "view_world_angles_rad", "translation_actual_m_s", "translation_target_m_s",
            } else 1
            shape = (count,) if name == "translation_active" else (count, width)
            if value.shape != shape or not torch.isfinite(value).all():
                raise ValueError("Invalid pre-action gimbal tracking: " + name)
        gimbal = {name: value.detach().clone() for name, value in gimbal.items()}
    sampler = getattr(environment.adapter, 'evaluation_sample', None)
    course_sample = sampler() if callable(sampler) else None
    return NativeTrackingSample(actual.detach().clone(), target.detach().clone(), gimbal, course_sample)


@torch.no_grad()
def evaluate(environment, checkpoint: str | Path, output: str | Path, *, steps: int):
    """Evaluate a trained actor without exploration noise or optimizer updates.

    Native physics and commands retain the supplied factory configuration.
    Deterministic actor inference does not assert deterministic GPU physics.
    The caller owns environment cleanup.
    """
    if type(steps) is not int or steps < 1:
        raise ValueError("Evaluation steps must be a positive integer")
    output, checkpoint = Path(output), Path(checkpoint)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Evaluation requires a new output directory")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format") != "supermimic-rl-checkpoint-v1":
        raise ValueError("Evaluation requires an online training checkpoint")
    current = environment.reset()
    schema = environment.schema
    config = LearningConfig.from_dict(payload["learner"]["config"])
    learner = Learner(schema, config, device=environment.device)
    learner.load_state_dict(payload["learner"])
    evaluation_owner = getattr(environment, "adapter", environment)
    begin = getattr(evaluation_owner, "evaluation_begin", None)
    environment_state = payload.get("environment_training_state")
    # Fixed suites own their task distribution and fresh native rows. Restoring
    # training RNG/counters would both import the old task exposure and require
    # its original batch size. Model/schema checks above remain strict.
    if environment_state is not None and not callable(begin):
        restore = getattr(environment, "load_training_state_dict", None)
        if not callable(restore):
            raise ValueError("Evaluation environment cannot restore the saved task state")
        restore(environment_state)
        current = environment.current
    if not learner.normalizers_fitted or learner.actor_updates < 1:
        raise ValueError("Evaluation requires a trained actor and fitted normalization")
    for module in (learner.world, learner.actor, learner.critic):
        module.eval()
        module.requires_grad_(False)
    if learner.estimator is not None:
        learner.estimator.eval().requires_grad_(False)
    if callable(begin):
        begin(schema)
        # Repack the fresh physical episodes through the environment boundary.
        if isinstance(environment, PackedEnvironment):
            environment.current = environment._pack(evaluation_owner.current)
        current = environment.current
    output.mkdir(parents=True, exist_ok=True)

    def actor_frame(snapshot):
        return (snapshot.observation if snapshot.actor_observation is None
                else snapshot.actor_observation)

    history = actor_frame(current)[:, None].expand(-1, config.history_length, -1).clone()
    count, device = environment.num_envs, environment.device
    total_reward = torch.zeros(count, dtype=torch.float64, device=device)
    episode_reward = torch.zeros_like(total_reward)
    completed_reward = torch.zeros_like(total_reward)
    failures = torch.zeros(count, dtype=torch.int64, device=device)
    truncations = torch.zeros_like(failures)
    completed = torch.zeros_like(failures)
    episode_steps = torch.zeros_like(failures)
    longest_episode = torch.zeros_like(failures)
    first_failure = torch.full_like(failures, steps)
    failed = torch.zeros(count, dtype=torch.bool, device=device)
    tracking_sum = torch.zeros(5, 3, dtype=torch.float64, device=device)
    command_min = torch.full((3,), float("inf"), device=device)
    command_max = torch.full((3,), -float("inf"), device=device)
    tracking_count = 0
    failure_masks = None

    try:
        with (output / "steps.jsonl").open("w") as stream:
            for step in range(1, steps + 1):
                current = environment.current
                estimate_hook = getattr(environment, "set_policy_estimate", None)
                if learner.estimator is not None and callable(estimate_hook):
                    estimate_hook(actor_frame(current), learner.estimator(history))
                tracking = native_tracking_sample(environment)
                action = learner.actor(history)
                transition = environment.step(action)
                transition.validate(schema)
                if not transition.valid.all():
                    raise RuntimeError(
                        "Invalid native transitions cannot count as evaluation evidence")
                observe = getattr(evaluation_owner, "evaluation_step", None)
                if callable(observe):
                    observe(transition, tracking)
                step_masks = terminal_failure_masks(schema, transition)
                if step_masks is not None:
                    if failure_masks is None:
                        failure_masks = {}
                    for mask, occurrences in step_masks.items():
                        failure_masks[mask] = failure_masks.get(mask, 0) + occurrences
                done = transition.terminated | transition.truncated
                reward = transition.reward.to(torch.float64)
                total_reward += reward
                episode_reward += reward
                episode_steps += 1
                longest_episode = torch.maximum(longest_episode, episode_steps)
                first_failure[transition.terminated & ~failed] = step
                failed |= transition.terminated
                failures += transition.terminated
                truncations += transition.truncated
                completed += done
                completed_reward += torch.where(done, episode_reward, 0)
                episode_reward[done] = 0
                episode_steps[done] = 0
    
                following = environment.current
                frame = actor_frame(following)
                history = torch.cat((history[:, 1:], frame[:, None]), dim=1)
                reset = (done | (current.episode_id != following.episode_id)
                         | (current.control_epoch != following.control_epoch))
                history[reset] = frame[reset, None].expand(-1, config.history_length, -1)
                record = {"step": step, "reward_mean": float(reward.mean()),
                          "terminated": int(transition.terminated.sum()),
                          "truncated": int(transition.truncated.sum())}
                if tracking is not None:
                    actual, target = tracking
                    error = (actual - target).to(torch.float64)
                    tracking_sum += torch.stack((
                        actual.double().sum(0), target.double().sum(0), error.sum(0),
                        error.abs().sum(0), error.square().sum(0),
                    ))
                    command_min = torch.minimum(command_min, target.amin(0))
                    command_max = torch.maximum(command_max, target.amax(0))
                    tracking_count += count
                    record["tracking_pre_action"] = {
                        name: {"actual_mean": float(actual[:, index].mean()),
                               "command_mean": float(target[:, index].mean())}
                        for index, name in enumerate(("forward_m_s", "yaw_rad_s", "height_m"))
                    }
                stream.write(json.dumps(record, allow_nan=False) + "\n")
    finally:
        finish = getattr(evaluation_owner, "evaluation_end", None)
        if callable(finish):
            finish()

    tracking_report = {"available": tracking_count > 0}
    if tracking_count:
        tracking_report.update({
            "sampling": "pre-action native state and admitted SI commands",
            "coordinates": ("base_link origin; horizontal heading forward; "
                            "yaw about world up; height above ground"),
            "command_deadband": "none; errors use the admitted command directly",
            "samples": tracking_count,
        })
        for index, name in enumerate(("forward_m_s", "yaw_rad_s", "height_m")):
            values = tracking_sum[:, index] / tracking_count
            tracking_report[name] = {
                "actual_mean": float(values[0]), "command_mean": float(values[1]),
                "bias": float(values[2]), "mae": float(values[3]),
                "rmse": float(values[4].sqrt()),
                "command_min": float(command_min[index]),
                "command_max": float(command_max[index]),
            }
    else:
        tracking_report["reason"] = "Synthetic/unclassified environment; no robot physical metrics"

    report = {
        "format": "supermimic-rl-evaluation-v1",
        "checkpoint": str(checkpoint.resolve()), "schema_hash": schema.fingerprint,
        "source": schema.source, "checkpoint_updates": payload["completed_updates"],
        "actor_updates": learner.actor_updates, "deterministic_actor": True,
        "environment_initialization": ("fresh fixed evaluation suite; training environment state not restored"
                                       if callable(begin) else "checkpoint environment state when available"),
        "action_noise_std": 0.0, "steps_per_environment": steps, "num_envs": count,
        "policy_dt_s": schema.policy_dt_s, "seconds_per_environment": steps * schema.policy_dt_s,
        "total_reward": float(total_reward.sum()),
        "mean_step_reward": float(total_reward.sum() / (steps * count)),
        "terminated_episodes": int(failures.sum()), "truncated_episodes": int(truncations.sum()),
        "completed_episodes": int(completed.sum()),
        "failure_free_environments": int((~failed).sum()),
        "tracking": tracking_report,
        "per_environment": [
            {"env_id": index, "return_sum": float(total_reward[index]),
             "mean_step_reward": float(total_reward[index] / steps),
             "terminated_episodes": int(failures[index]),
             "truncated_episodes": int(truncations[index]),
             "completed_episodes": int(completed[index]),
             "episodes_started_including_final_autoreset": int(completed[index]) + 1,
             "steps_until_first_failure_or_horizon": int(first_failure[index]),
             "failed_during_evaluation": bool(failed[index]),
             "longest_episode_steps": int(longest_episode[index]),
             "current_episode_steps": int(episode_steps[index]),
             "completed_episode_return_mean": (
                 float(completed_reward[index] / completed[index])
                 if completed[index] else None)}
            for index in range(count)
        ],
    }
    if failure_masks is not None:
        report["failure_reason_masks"] = {
            "source": "transition.end.failure.failure_reasons before auto-reset",
            "encoding": "numeric multi-reason bitmask; combined bits are not separate failures",
            "terminated_mask_histogram": failure_masks,
        }
    summary = getattr(evaluation_owner, "evaluation_summary", None)
    if callable(summary):
        report["course"] = summary()
    (output / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return report
