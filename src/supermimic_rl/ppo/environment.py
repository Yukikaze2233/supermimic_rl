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

from .policy import HISTORY_LENGTH
from ..chassis_observation import COMMAND_CONTRACTS, LEGACY_COMMANDS, chassis_frame

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
        self.chassis_command_contract = LEGACY_COMMANDS

    def set_chassis_command_contract(self, contract: str) -> None:
        if contract not in COMMAND_CONTRACTS:
            raise ValueError("Unknown chassis command contract")
        self.chassis_command_contract = contract

    @property
    def action_bounds(self) -> tuple[float, ...]:
        """Native execution clip; the actor's bounds must match it exactly."""
        return tuple(self.env._action_bounds.tolist())

    def _observation(self) -> Observation:
        frames = self.env.obs_buf["policy"].reshape(self.num_envs, HISTORY_LENGTH, NATIVE_FRAME_DIM)
        return Observation(chassis_frame(frames, self.chassis_command_contract).clone(), self.env.obs_buf["critic"].clone(),
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
            # Native moments already mask placeholder commands and integrate
            # all physical substeps. Keep their exposure denominator instead
            # of averaging per-interval RMS values or filling inactive rows.
            for name, count in getattr(result, "scoped_error_counts", {}).items():
                moments = (result.scoped_error_sums[name],
                           result.scoped_error_square_sums[name], count)
                measured = torch.stack(tuple(torch.isfinite(value) for value in moments)).all(0)
                for field, value in zip(("sum", "square_sum", "count"), moments):
                    diagnostics[f"tracking_scope/{field}/{name}"] = torch.where(
                        measured, value, float("nan"))
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

    def set_opened_course_gates(self, keys) -> None:
        """Apply performance openings, or None to use the authored height schedule."""
        self.adapter.curriculum.set_opened_gates(keys)

    def check_course_gates(self, configured_keys) -> None:
        """A difficulty ladder whose gates are not configured would silently stay at its first level."""
        missing = set(getattr(self.adapter.curriculum, "ladder_gate_keys", tuple)()) - set(configured_keys)
        if missing:
            raise ValueError(f"Report-mode course gates miss the course's ladder gates: {sorted(missing)}")

    def course_stage_starts(self) -> dict[str, int]:
        """First real-transition coordinate of each native course stage."""
        curriculum = self.adapter.curriculum
        starts, total = {}, 0
        for name, budget in zip(curriculum.stage_names, curriculum.stage_budgets):
            starts[name] = total
            total += budget
        return starts

    def course_progress(self) -> tuple[dict[str, float], dict[str, float]]:
        """Read native cumulative skill counters and current course gauges."""
        from ..course.metrics import course_progress

        return course_progress(self.env, self.adapter.curriculum)

    def termination_reasons(self) -> dict:
        latch = getattr(self.env, "_failure_latch", None)
        return {} if latch is None else latch.metrics()

    def close(self) -> None:
        self.adapter.close()
