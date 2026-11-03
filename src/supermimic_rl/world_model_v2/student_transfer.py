"""Load V2 student actors and transfer the 5x35 chassis student into the PPO actor.

The chassis student and the PPO actor share their inputs (five native 35D
frames, oldest first), estimator layout (175 -> hidden -> 5), decoder layout
([175 history, 5 detached estimate] -> hidden -> 6) and output
``tanh(decoder) * action_bounds``. With equal bounds the copy is exact; it is
checked numerically before an initialization artifact is written. Only the history
refill at control handoffs differs: the student's collector refills there,
the native PPO history does not.

For other PPO bounds B (e.g. full-turn legs from a +-3 student) the final
layer is scaled by b/B, so ``B*tanh(bz/B)`` keeps the student's response at
small actions and departs from ``b*tanh(z)`` only near the student's saturation. Such a start is
approximate; PPO then imitates the exact student on its own states.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from ..ppo.policy import FRAME_DIM, HISTORY_LENGTH, TeacherActor
from ..ppo.trainer import ACTOR_INIT_FORMAT
from .policy import CHASSIS_FRAME_DIM, EstimatorActor

EQUIVALENCE_TOLERANCE = 1e-6


def load_student_actor(path: str | Path, device: str | torch.device = "cpu") -> tuple[EstimatorActor, dict]:
    """Current V2 student actor artifact and its saved metadata."""
    from .online import CHECKPOINT_FORMAT, OnlineConfig
    from .records import MODEL_CONTRACT

    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") != CHECKPOINT_FORMAT + "-actor" or saved["model_contract"] != MODEL_CONTRACT:
        raise ValueError("Requires a current V2 student actor artifact")
    config = OnlineConfig.from_dict(saved["config"])
    # Bounds are a saved buffer; build with the saved value so the load is exact.
    bounds = tuple(saved["actor"]["action_bounds"].tolist())
    actor = EstimatorActor(estimator_hidden=config.estimator_hidden, actor_hidden=config.actor_hidden,
                           frame_dim=config.actor_frame_dim, action_bounds=bounds).to(device)
    actor.load_state_dict(saved["actor"])
    return actor.eval(), saved


def _hidden(network: nn.Sequential) -> tuple[int, ...]:
    return tuple(layer.out_features for layer in network if isinstance(layer, nn.Linear))[:-1]


def ppo_actor_from_student(student: EstimatorActor,
                           action_bounds: tuple[float, ...] | None = None) -> TeacherActor:
    """PPO actor with ``action_bounds`` (default: the student's own bounds)."""
    if student.frame_dim != CHASSIS_FRAME_DIM or student.history_length != HISTORY_LENGTH:
        raise ValueError("Only the 5x35 chassis student matches the PPO actor interface")
    bounds = tuple(student.action_bounds.tolist()) if action_bounds is None else tuple(action_bounds)
    actor = TeacherActor(_hidden(student.decoder), _hidden(student.estimator), mean_squash="tanh",
                         action_bounds=bounds)
    actor.estimator.network.load_state_dict(student.estimator.state_dict())
    actor.network.load_state_dict(student.decoder.state_dict())
    with torch.no_grad():
        ratio = student.action_bounds.cpu() / actor.action_bounds
        actor.network[-1].weight.mul_(ratio[:, None])
        actor.network[-1].bias.mul_(ratio)
    return actor.to(student.action_bounds.device)


@torch.no_grad()
def max_action_difference(student: EstimatorActor, actor: TeacherActor, *, samples: int = 4096,
                          seed: int = 0) -> float:
    generator = torch.Generator().manual_seed(seed)
    history = torch.randn(samples, HISTORY_LENGTH, FRAME_DIM, generator=generator)
    history = history.to(student.action_bounds.device)
    return float((student(history) - actor(history)).abs().max())


def actor_init_payload(student_path: str | Path, action_bounds: tuple[float, ...] | None = None) -> dict:
    """PPO initialization artifact; with unchanged bounds any action change is refused."""
    path = Path(student_path).resolve(strict=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    student, saved = load_student_actor(path)
    actor = ppo_actor_from_student(student, action_bounds)
    difference = max_action_difference(student, actor)
    exact = torch.equal(actor.action_bounds, student.action_bounds.cpu())
    if exact and difference > EQUIVALENCE_TOLERANCE:
        raise ValueError(f"Transferred actor differs from the student by {difference}")
    return {"format": ACTOR_INIT_FORMAT, "actor": actor.state_dict(),
            "actor_hidden": list(_hidden(student.decoder)), "estimator_hidden": list(_hidden(student.estimator)),
            "mean_squash": "tanh",
            "action_bounds": (student.action_bounds.tolist() if action_bounds is None
                              else [float(value) for value in action_bounds]),
            "source": {"kind": "v2_chassis_student", "path": str(path), "sha256": digest,
                       "cycle": saved["cycle"], "updates": saved["updates"],
                       "model_contract": saved["model_contract"], "schema_fingerprint": saved["schema_fingerprint"]},
            "equivalence": {"exact": exact, "max_abs_action_difference": difference, "samples": 4096,
                            "inputs": "standard-normal 5x35 histories",
                            "scope": ("deterministic mean; PPO adds its own exploration std" if exact else
                                      "first-order start for wider bounds; imitate the exact student next")}}


def initialize_student_from_ppo(student: EstimatorActor, path: Path, expected_sha256: str) -> dict:
    """Copy a PPO actor (checkpoint or initialization artifact) into a fresh chassis student.

    Both networks share inputs, layers and the ``tanh * action_bounds`` output,
    so the copy is exact; it is refused for any other interface.
    """
    from ..ppo.trainer import load_actor

    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError("Initial PPO actor SHA256 differs from the configured value")
    device = student.action_bounds.device
    actor = load_actor(path, device)
    if (actor.mean_squash != "tanh" or student.frame_dim != CHASSIS_FRAME_DIM
            or not torch.equal(actor.action_bounds, student.action_bounds)
            or _hidden(actor.network) != _hidden(student.decoder)
            or _hidden(actor.estimator.network) != _hidden(student.estimator)):
        raise ValueError("PPO actor and chassis student interfaces differ")
    with torch.no_grad():
        student.estimator.load_state_dict(actor.estimator.network.state_dict())
        student.decoder.load_state_dict(actor.network.state_dict())
    difference = max_action_difference(student.eval(), actor.eval())
    if difference > EQUIVALENCE_TOLERANCE:
        raise ValueError(f"Initialized student differs from the PPO actor by {difference}")
    return {"path": str(path), "sha256": digest, "max_abs_action_difference": difference,
            "action_std": actor.log_std.detach().exp().tolist(),
            "scope": "student weights only; world model, critic and optimizers start fresh"}
