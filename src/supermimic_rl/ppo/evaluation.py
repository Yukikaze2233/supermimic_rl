"""Deterministic native evaluation and ONNX export of the teacher actor.

Evaluation reports physical outcomes, not reward: failures per environment and
per second, termination reasons, completed episode lengths and RMS tracking
errors of forward speed, yaw rate and support height at policy endpoints.
"""
from __future__ import annotations

from pathlib import Path
from time import monotonic

import torch

from .policy import FRAME_DIM, HISTORY_LENGTH, TeacherActor
from .trainer import EpisodeStatistics
from ..chassis_observation import LEGACY_COMMANDS, annotate_onnx_commands

POLICY_DT_S = .02


@torch.no_grad()
def evaluate(environment, actor: TeacherActor, steps: int, *, decision=None,
             on_progress=None, progress_interval: int = 100) -> dict:
    if steps < 1 or progress_interval < 1:
        raise ValueError("Evaluation steps and progress interval must be positive")
    statistics = EpisodeStatistics(environment.num_envs, environment.device)
    if hasattr(environment, "set_chassis_command_contract"):
        environment.set_chassis_command_contract(getattr(actor, "chassis_command_contract", LEGACY_COMMANDS))
    observation = environment.reset()
    if actor is not None and actor.features.configuration is not None:
        actor.features.validate_reference(environment.env._terrain_course.tasks.reference.config)
    started = monotonic()
    for index in range(steps):
        if decision is None:
            estimate = actor.estimator(observation.history)
            action = actor.clip(actor(observation.history))
        else:
            action, estimate = decision(observation.history)
        environment.set_estimate(observation.history, estimate)
        result = environment.step(action)
        statistics.record(result)
        observation = result.observation
        if on_progress is not None and ((index + 1) % progress_interval == 0 or index + 1 == steps):
            on_progress({"completed_steps": index + 1, "requested_steps": steps,
                         "elapsed_s": monotonic() - started, **statistics.summary()})
    summary = statistics.summary()
    seconds = environment.num_envs * steps * POLICY_DT_S
    valid_seconds = summary["valid_tracking_samples"] * POLICY_DT_S
    summary.update({"steps": steps, "environments": environment.num_envs, "simulated_seconds": seconds,
                    "failures_per_env_minute": 60 * summary["terminations"] / seconds,
                    "valid_simulated_seconds": valid_seconds,
                    "failures_per_valid_env_minute": (60 * summary["terminations"] / valid_seconds
                                                     if valid_seconds else None),
                    "termination_reasons": environment.termination_reasons()})
    return summary


class _DeployedActor(torch.nn.Module):
    """[1, 5, 35] deployed history -> 6 clipped targets (estimator inside)."""

    def __init__(self, actor: TeacherActor):
        super().__init__()
        self.actor = actor

    def forward(self, history):
        return self.actor.clip(self.actor(history))


def export_onnx(actor: TeacherActor, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, HISTORY_LENGTH, FRAME_DIM)
    torch.onnx.export(_DeployedActor(actor.cpu().eval()), example, str(path), input_names=["history"],
                      output_names=["action"], opset_version=17, dynamo=False)
    annotate_onnx_commands(path, actor.chassis_command_contract)
    from ..actor_features import annotate_onnx_features
    annotate_onnx_features(path, actor.features.configuration)
    return path
