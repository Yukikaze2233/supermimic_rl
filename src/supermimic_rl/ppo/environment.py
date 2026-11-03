"""PPO-facing view of the native course built by ``course_factory.create_course``.

PPO needs no 441D snapshots, so this view steps the native environment directly
and reads what it already produces: the deployed actor history (``[N, 5, 42]``,
oldest first, reset rows refilled with their first frame), the 182D privileged
critic features and their pre-reset terminal copy for truncation bootstrap.
Curriculum progress and the deployed request estimate go through the course
adapter's public hooks; native physics health keeps being checked every step.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .policy import FRAME_DIM, HISTORY_LENGTH

NATIVE_FRAME_DIM = 42
WHEEL_CONTACT_N = 2.


@dataclass(frozen=True)
class Observation:
    history: Tensor       # [N, 5, 35] deployed frames, oldest first
    critic: Tensor        # [N, C] privileged critic features
    label: Tensor         # [N, 5] true body velocity xyz and wheel contact (0/1) for ``history``
    label_valid: Tensor   # [N] False right after a reset: the newest frame belongs to a new episode


@dataclass(frozen=True)
class Step:
    observation: Observation      # next decision endpoint (post-reset for finished rows)
    reward: Tensor
    terminated: Tensor
    truncated: Tensor
    terminal_critic: Tensor       # pre-reset endpoint critic features
    tracking: dict[str, Tensor]   # per-row physical errors at the pre-reset endpoint
    valid: Tensor | None = None   # False intervals supply no PPO or estimator loss.
    numerical_invalid: Tensor | None = None  # Separate from genuine mechanical/task failures.
    diagnostics: dict[str, Tensor] | None = None  # Raw physical endpoints/interval targets, never losses.


class CourseEnvironment:
    def __init__(self, adapter):
        self.adapter, self.env = adapter, adapter.env
        self.num_envs, self.device = self.env.num_envs, torch.device(self.env.device)
        # The adapter owns the native health monitor; PPO keeps its per-step checks.
        self._health = getattr(adapter, "_health")
        self._label = torch.zeros(self.num_envs, 5, device=self.device)
        self._label_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def action_bounds(self) -> tuple[float, ...]:
        """Native execution clip; the actor's bounds must match it exactly."""
        return tuple(self.env._action_bounds.tolist())

    def _observation(self) -> Observation:
        frames = self.env.obs_buf["policy"].reshape(self.num_envs, HISTORY_LENGTH, NATIVE_FRAME_DIM)
        return Observation(frames[..., :FRAME_DIM].clone(), self.env.obs_buf["critic"].clone(),
                           self._label.clone(), self._label_valid.clone())

    def reset(self) -> Observation:
        self.adapter.reset()
        self._label_valid.zero_()
        return self._observation()

    def set_progress(self, real_transitions: int, actor_updates: int) -> None:
        self.adapter.set_training_progress(real_transitions, actor_updates)

    def set_estimate(self, history: Tensor, estimate: Tensor) -> None:
        """Feed the deployed request protocol with velocity and contact logits (8D layout)."""
        raw = torch.cat((estimate[:, :3], torch.zeros_like(estimate[:, :3]), estimate[:, 3:5]), -1)
        self.adapter.set_policy_estimate(history[:, -1], raw)

    @torch.no_grad()
    def step(self, action: Tensor) -> Step:
        self._health.check()
        # The native step advances task commands and may reset rows after its
        # last physical sample. Preserve the target used by this interval.
        command = self.env._command.clone()
        _, reward, terminated, truncated, extras = self.env.step(action)
        self._health.sample_policy()
        state = self.env._last_state
        done = terminated | truncated
        valid = torch.isfinite(state.velocity_origin_body_m_s).all(-1)
        if getattr(state, "contact_measurement_valid", None) is not None:
            valid &= state.contact_measurement_valid
        self._label = torch.cat((state.velocity_origin_body_m_s,
                                 (state.wheel_ground_force_n > WHEEL_CONTACT_N).float()), -1)
        self._label_valid = valid & ~done
        tracking, diagnostics = self._tracking(state, command)
        result = getattr(self.env, "_formal_result", None)
        if result is not None:
            if "velocity_height_gate" in result.gate_fractions:
                diagnostics["velocity_height_gate"] = result.gate_fractions["velocity_height_gate"].clone()
            # The native producer already divides its four-tick integral by
            # policy_dt. These are signed, weighted densities, not step rewards.
            # Events remain once-per-interval amounts; never rescale them by dt.
            for prefix, attribute in (("reward_per_s", "component_densities"),
                                      ("reward_event", "event_rewards"),
                                      ("reward_gate", "gate_fractions")):
                diagnostics.update({f"{prefix}/{name}": value.clone()
                                    for name, value in getattr(result, attribute, {}).items()})
            for name in ("physical_step_reward", "learning_reward"):
                if hasattr(result, name):
                    diagnostics[f"reward_step/{name}"] = getattr(result, name).clone()
        return Step(self._observation(), reward.float(), terminated.bool(), truncated.bool(),
                    self.env._terminal_critic_182.clone(), tracking,
                    extras["interval_observation_valid"].clone(),
                    extras.get("numerical_invalid", torch.zeros_like(done)).clone(), diagnostics)

    def _tracking(self, state, command) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        up = -state.gravity_body
        forward = torch.zeros_like(up)
        forward[:, 0] = 1
        forward = forward - (forward * up).sum(-1, keepdim=True) * up
        forward = forward / forward.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        v_forward = (state.velocity_origin_body_m_s * forward).sum(-1)
        yaw_rate = (state.angular_velocity_body_rad_s * up).sum(-1)
        height = state.height_above_ground_m.reshape(-1)
        return ({"forward_error_m_s": v_forward - command[:, 0], "yaw_rate_error_rad_s": yaw_rate - command[:, 2],
                 "height_error_m": height - command[:, 3],
                 "tilt_rad": torch.atan2(state.gravity_body[:, :2].norm(dim=-1), -state.gravity_body[:, 2])},
                {"command_forward_m_s": command[:, 0], "actual_forward_m_s": v_forward,
                 "command_yaw_rate_rad_s": command[:, 2], "actual_yaw_rate_rad_s": yaw_rate,
                 "command_height_m": command[:, 3], "actual_height_m": height})

    def course_stage_starts(self) -> dict[str, int]:
        """First real-transition coordinate of each native course stage."""
        curriculum = self.adapter.curriculum
        starts, total = {}, 0
        for name, budget in zip(curriculum.stage_names, curriculum.stage_budgets):
            starts[name] = total
            total += budget
        return starts

    def course_progress(self) -> tuple[dict[str, float], dict[str, float]]:
        """Native cumulative skill counters (since cold setup) and current course gauges.

        Jump outcomes are physically verified request results; stair completion
        is the read-only authored whole-course evidence; recovery counts are the
        native lifecycle. None of these is a policy input.
        """
        metrics = self.env._terrain_course.metrics()
        counters = {f"course/jump/{name}": value
                    for name, value in metrics["physical_request_outcomes"].items()}
        counters.update({f"course/stair/{name}": value for name, value in metrics["stair_closed_loop"].items()})
        authored = metrics["authored_stair_completion"]
        for name in ("ended", "ended_complete"):
            counters[f"course/stair/authored_{name}"] = sum(authored[name + "_by_tile"])
        counters.update({f"course/recovery/{name}": value
                         for name, value in metrics["recovery"]["counts"].items()})
        counters.update({f"course/births/{name}": value
                         for name, value in metrics["episode_birth_family_counts"].items()})
        levels = metrics["level_histogram"]
        rows = sum(levels)
        gauges = {"course/terrain/level_fraction": (
            sum(level * count for level, count in enumerate(levels)) / (rows * max(len(levels) - 1, 1))
            if rows else 0.)}
        assist = getattr(self.env, "supermimic_jump_assist", None)
        if assist is not None:
            assist_counters, assist_gauges = assist.metrics()
            counters.update(assist_counters)
            gauges.update(assist_gauges)
        return counters, gauges

    def termination_reasons(self) -> dict:
        latch = getattr(self.env, "_failure_latch", None)
        return {} if latch is None else latch.metrics()

    def close(self) -> None:
        self.adapter.close()
