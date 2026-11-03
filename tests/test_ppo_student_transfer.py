"""Bounded PPO mean, legacy raw-mean checkpoints, student transfer and critic warm-up."""
from dataclasses import asdict
import hashlib
import json
import math

import pytest
import torch

from supermimic_rl.contracts import ACTION_BOUNDS
from supermimic_rl.ppo.policy import TeacherActor
from supermimic_rl.ppo.trainer import ACTOR_INIT_FORMAT, CHECKPOINT_FORMAT, PPOConfig, PPOTrainer, load_actor
from supermimic_rl.world_model_v2.online import CHECKPOINT_FORMAT as ONLINE_FORMAT, OnlineConfig
from supermimic_rl.world_model_v2.policy import EstimatorActor
from supermimic_rl.world_model_v2.records import MODEL_CONTRACT
from supermimic_rl.world_model_v2.student_transfer import actor_init_payload, ppo_actor_from_student

from support.ppo import TargetEnvironment, config


def test_tanh_mean_stays_inside_the_execution_bounds():
    actor = TeacherActor((16,), (8,))
    torch.nn.init.constant_(actor.network[-1].bias, 50.)
    mean = actor(torch.randn(32, 5, 35))
    bounds = torch.tensor(ACTION_BOUNDS)
    assert (mean.abs() <= bounds).all() and torch.allclose(mean, bounds.expand_as(mean))


def test_configs_without_squash_are_legacy_raw_mean(tmp_path):
    assert PPOConfig().mean_squash == "tanh"
    legacy = asdict(PPOConfig(actor_hidden=(16,), estimator_hidden=(8,), critic_hidden=(16,)))
    del legacy["mean_squash"], legacy["initial_actor"], legacy["initial_actor_sha256"]
    del legacy["actor_freeze_updates"]
    assert PPOConfig.from_dict(legacy).mean_squash == "none"
    actor = TeacherActor((16,), (8,), mean_squash="none")
    torch.nn.init.constant_(actor.network[-1].bias, 7.)
    torch.save({"format": CHECKPOINT_FORMAT, "config": legacy, "actor": actor.state_dict(),
                "critic": {}, "updates": 1, "transitions": 1}, tmp_path / "legacy.pt")
    history = torch.randn(4, 5, 35)
    loaded = load_actor(tmp_path / "legacy.pt")
    torch.testing.assert_close(loaded(history), actor(history))
    assert (loaded(history) > torch.tensor(ACTION_BOUNDS)).any()


def _student_artifact(path, actor_hidden=(16, 8), estimator_hidden=(8,)):
    torch.manual_seed(3)
    student = EstimatorActor(5, estimator_hidden, actor_hidden, frame_dim=35)
    for parameter in student.parameters():
        torch.nn.init.normal_(parameter, std=.3)
    online = OnlineConfig(actor_hidden=actor_hidden, estimator_hidden=estimator_hidden, actor_frame_dim=35)
    torch.save({"format": ONLINE_FORMAT + "-actor", "model_contract": MODEL_CONTRACT, "config": asdict(online),
                "cycle": 82, "updates": {"actor": 1}, "schema_fingerprint": "f" * 64,
                "actor": student.state_dict()}, path)
    return student


def test_student_transfer_reproduces_student_actions(tmp_path):
    student = _student_artifact(tmp_path / "student.pt").eval()
    history = torch.randn(64, 5, 35) * 3
    with torch.no_grad():
        torch.testing.assert_close(ppo_actor_from_student(student)(history), student(history), rtol=0, atol=0)
    payload = actor_init_payload(tmp_path / "student.pt")
    assert payload["format"] == ACTOR_INIT_FORMAT and payload["equivalence"]["max_abs_action_difference"] == 0
    assert payload["source"]["cycle"] == 82 and payload["mean_squash"] == "tanh"


def test_initialization_artifact_evaluates_through_the_ppo_loader(tmp_path):
    student, path, _ = _init_artifact(tmp_path)
    history = torch.randn(16, 5, 35)
    with torch.no_grad():
        torch.testing.assert_close(load_actor(path)(history), student.eval()(history), rtol=0, atol=0)


def test_student_with_gimbal_inputs_is_not_transferred():
    with pytest.raises(ValueError, match="5x35 chassis"):
        ppo_actor_from_student(EstimatorActor(5, (8,), (16,), frame_dim=45))


def _init_artifact(tmp_path):
    student = _student_artifact(tmp_path / "student.pt", actor_hidden=(64, 64), estimator_hidden=(32,))
    payload = actor_init_payload(tmp_path / "student.pt")
    path = tmp_path / "init.pt"
    torch.save(payload, path)
    return student, path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_initialized_actor_is_frozen_while_the_critic_warms_up(tmp_path):
    student, path, digest = _init_artifact(tmp_path)
    environment = TargetEnvironment(num_envs=8)
    trainer = PPOTrainer(environment, config(updates=3, initial_actor=str(path), initial_actor_sha256=digest,
                                             actor_freeze_updates=2, initial_std=.3), tmp_path / "run")
    history = torch.randn(16, 5, 35)
    with torch.no_grad():
        torch.testing.assert_close(trainer.actor(history), student.eval()(history), rtol=0, atol=0)
    torch.testing.assert_close(trainer.actor.log_std.exp(), torch.full((6,), .3))
    actor_before = {name: value.clone() for name, value in trainer.actor.state_dict().items()}
    critic_before = {name: value.clone() for name, value in trainer.critic.state_dict().items()}
    learning_rate = trainer.learning_rate
    trainer.train()
    records = [json.loads(line) for line in (tmp_path / "run/metrics.jsonl").read_text().splitlines()]
    assert [record["actor_frozen"] for record in records] == [True, True, False]
    assert [record["learning_rate"] for record in records[:2]] == [learning_rate, learning_rate]
    assert len(records[0]["action/mean_near_bound_fraction"]) == 6
    assert any(not torch.equal(actor_before[name], value) for name, value in trainer.actor.state_dict().items())
    assert any(not torch.equal(critic_before[name], value)
               for name, value in trainer.critic.state_dict().items() if "norm" not in name)
    assert trainer.state_dict()["initialization"]["sha256"] == digest


def test_frozen_updates_leave_actor_and_estimator_unchanged(tmp_path):
    _, path, digest = _init_artifact(tmp_path)
    trainer = PPOTrainer(TargetEnvironment(num_envs=8),
                         config(updates=2, initial_actor=str(path), initial_actor_sha256=digest,
                                actor_freeze_updates=2), tmp_path / "run")
    before = {name: value.clone() for name, value in trainer.actor.state_dict().items()}
    trainer.train()
    for name, value in trainer.actor.state_dict().items():
        assert torch.equal(before[name], value), name


def test_initial_actor_requires_matching_digest_and_interface(tmp_path):
    _, path, digest = _init_artifact(tmp_path)
    with pytest.raises(ValueError, match="SHA256"):
        PPOTrainer(TargetEnvironment(num_envs=4), config(initial_actor=str(path), initial_actor_sha256="0" * 64),
                   tmp_path / "a")
    with pytest.raises(ValueError, match="interface"):
        PPOTrainer(TargetEnvironment(num_envs=4), config(initial_actor=str(path), initial_actor_sha256=digest,
                                                         actor_hidden=(32,)), tmp_path / "b")
    with pytest.raises(ValueError, match="together"):
        config(initial_actor=str(path))


FULL_TURN = (4 * math.pi,) * 4 + (9., 9.)


class FullTurnEnvironment(TargetEnvironment):
    """Same task with the full-turn leg contract exposed like the native view."""

    action_bounds = FULL_TURN

    def step(self, action):
        assert (action.abs() <= torch.tensor(FULL_TURN) + 1e-5).all()
        return super().step(action.clamp(-3, 3))


def test_wider_bounds_keep_the_student_response_at_small_actions(tmp_path):
    student = _student_artifact(tmp_path / "student.pt").eval()
    with torch.no_grad():
        # Small decoder outputs: the student acts in the near-linear tanh range.
        student.decoder[-1].weight.mul_(.01)
        student.decoder[-1].bias.zero_()
    actor = ppo_actor_from_student(student, FULL_TURN)
    history = torch.randn(64, 5, 35)
    with torch.no_grad():
        expected = student(history)
        assert expected[:, :4].abs().max() < .3  # wheels keep their bound and stay exact
        torch.testing.assert_close(actor(history), expected, rtol=2e-2, atol=1e-4)
    payload = actor_init_payload(tmp_path / "student.pt", FULL_TURN)
    assert payload["action_bounds"] == list(FULL_TURN) and payload["equivalence"]["exact"] is False


def test_actor_bounds_must_match_the_native_contract(tmp_path):
    with pytest.raises(ValueError, match="native"):
        PPOTrainer(FullTurnEnvironment(num_envs=4), config(), tmp_path)
    trainer = PPOTrainer(FullTurnEnvironment(num_envs=4), config(action_bounds=FULL_TURN), tmp_path / "ok")
    assert trainer.actor.action_bounds.tolist() == pytest.approx(list(FULL_TURN))


def test_imitation_runs_on_learner_states_then_decays(tmp_path):
    _, teacher_path, teacher_digest = _init_artifact(tmp_path)
    environment = FullTurnEnvironment(num_envs=16)
    trainer = PPOTrainer(environment, config(updates=6, action_bounds=FULL_TURN, imitation_actor=str(teacher_path),
                                             imitation_actor_sha256=teacher_digest, imitation_updates=3,
                                             imitation_anneal_updates=2, initial_std=.2), tmp_path / "run")
    history = torch.randn(256, 5, 35) * .1
    with torch.no_grad():
        target = trainer.imitation_teacher(history)
        before = (trainer.actor(history) - target).square().sum(-1).mean()
    learning_rate = trainer.learning_rate
    trainer.train()
    records = [json.loads(line) for line in (tmp_path / "run/metrics.jsonl").read_text().splitlines()]
    assert [record["imitation_weight"] for record in records] == pytest.approx([1, 1, 1, 2 / 3, 1 / 3, 0])
    assert [record["surrogate_active"] for record in records] == [False] * 3 + [True] * 3
    assert all(record["learning_rate"] == learning_rate for record in records[:3])
    with torch.no_grad():
        after = (trainer.actor(history) - target).square().sum(-1).mean()
    assert after < before


def test_imitation_schedule_requires_a_teacher():
    with pytest.raises(ValueError, match="imitation_actor"):
        config(imitation_updates=2)


def _fake_native(bounds):
    from types import SimpleNamespace

    controller = SimpleNamespace(action_bounds=torch.tensor((3., 3., 3., 3., 9., 9.)))
    return SimpleNamespace(_action_bounds=torch.tensor(bounds), _control=SimpleNamespace(controller=controller))


def _restore_native_validation(monkeypatch):
    import sys
    from transformer_rl.formal.control import ownership

    for name, module in tuple(sys.modules.items()):
        if name.startswith("transformer_rl.") and hasattr(module, "validate_issued_control"):
            monkeypatch.setattr(module, "validate_issued_control", module.validate_issued_control)
    return ownership


def test_legacy_bound_leaves_the_native_contract_untouched(monkeypatch):
    from supermimic_rl.course.action_contract import install_action_bounds

    ownership = _restore_native_validation(monkeypatch)
    original = ownership.validate_issued_control
    native = _fake_native(ACTION_BOUNDS)
    assert install_action_bounds(native, ACTION_BOUNDS)["ownership_validation"] == "pinned native"
    assert ownership.validate_issued_control is original


def test_full_turn_bound_reaches_controller_and_native_validation(monkeypatch):
    from supermimic_rl.course.action_contract import install_action_bounds
    import transformer_rl.formal.control.actuation as actuation

    ownership = _restore_native_validation(monkeypatch)
    native = _fake_native(FULL_TURN)
    receipt = install_action_bounds(native, FULL_TURN)
    torch.testing.assert_close(native._control.controller.action_bounds, torch.tensor(FULL_TURN))
    assert "transformer_rl.formal.control.actuation" in receipt["patched_modules"]
    assert actuation.validate_issued_control is ownership.validate_issued_control
    policy = torch.tensor([[12., -12., 4., 0., 9., -9.]])
    source = torch.zeros(1, dtype=torch.long)
    ownership.validate_issued_control(policy, source, num_envs=1, device="cpu")
    with pytest.raises(ValueError, match="action bounds"):
        ownership.validate_issued_control(policy * 1.1, source, num_envs=1, device="cpu")
    hold = torch.tensor([[6 * math.pi - .1, 0., 0., 0., 0., 0.]])
    ownership.validate_issued_control(hold, torch.full((1,), 2, dtype=torch.long), num_envs=1, device="cpu")
    with pytest.raises(ValueError, match="task.action_bounds"):
        install_action_bounds(native, ACTION_BOUNDS)


def test_only_a_changed_action_contract_enters_the_schema_identity():
    from supermimic_rl.cli import _native_schema_source

    base = {"runtime_sha256": "r"}
    legacy = json.loads(_native_schema_source({**base, "action_bounds": list(ACTION_BOUNDS)}))
    assert "action_bounds" not in legacy and legacy == json.loads(_native_schema_source(base))
    assert "action_bounds" in json.loads(_native_schema_source({**base, "action_bounds": list(FULL_TURN)}))


def _rotate_left_leg(bounds, steps=40, increment=.3):
    """Feed the native decoder a target turning the rigid left leg by ``increment`` per step."""
    from transformer_rl.formal.control.actuation import OutputFeedbackController

    nominal = (-.42, .13742282595395358, .42, -.1374155762580851, 0., 0.)
    controller = OutputFeedbackController(nominal, 1, pair_domain=None)
    controller.action_bounds.copy_(torch.tensor(bounds))
    feedback = torch.tensor([nominal], dtype=torch.float32)
    clock = torch.zeros(1, dtype=torch.float64)
    angles = []
    for step in range(1, steps + 1):
        # The policy issues the wrapped offset; the decoder picks the turn nearest the feedback.
        offset = math.remainder(step * increment, 2 * math.pi)
        action = torch.zeros(1, 6)
        action[0, :2] = offset / .25
        targets = controller.decode(action, feedback, clock)
        feedback = feedback.clone()
        feedback[0, :4] = targets.admitted_leg_position_rad[0]
        angles.append(float(feedback[0, 0]) - nominal[0])
    return angles


def test_full_turn_bound_lets_the_leg_rotate_continuously():
    angles = _rotate_left_leg(FULL_TURN)
    assert angles == pytest.approx([.3 * step for step in range(1, 41)], abs=1e-5)
    assert angles[-1] > 2 * math.pi  # more than one full revolution


def test_legacy_bound_stalls_the_leg_at_43_degrees():
    angles = _rotate_left_leg(ACTION_BOUNDS)
    assert max(abs(angle) for angle in angles) == pytest.approx(.75, abs=1e-5)



def test_student_initializes_exactly_from_a_ppo_actor(tmp_path):
    from supermimic_rl.world_model_v2.student_transfer import initialize_student_from_ppo

    torch.manual_seed(5)
    actor = TeacherActor((16, 8), (8,), action_bounds=FULL_TURN)
    for parameter in actor.parameters():
        torch.nn.init.normal_(parameter, std=.3)
    payload = {"format": ACTOR_INIT_FORMAT, "actor": actor.state_dict(), "actor_hidden": [16, 8],
               "estimator_hidden": [8], "mean_squash": "tanh", "action_bounds": list(FULL_TURN),
               "source": {}, "equivalence": {}}
    torch.save(payload, tmp_path / "ppo.pt")
    digest = hashlib.sha256((tmp_path / "ppo.pt").read_bytes()).hexdigest()
    student = EstimatorActor(5, (8,), (16, 8), frame_dim=35, action_bounds=FULL_TURN)
    receipt = initialize_student_from_ppo(student, tmp_path / "ppo.pt", digest)
    history = torch.randn(32, 5, 35)
    with torch.no_grad():
        torch.testing.assert_close(student(history), actor(history), rtol=0, atol=0)
    assert receipt["max_abs_action_difference"] == 0 and receipt["sha256"] == digest
    with pytest.raises(ValueError, match="interfaces differ"):
        initialize_student_from_ppo(EstimatorActor(5, (8,), (16, 8), frame_dim=35), tmp_path / "ppo.pt", digest)
