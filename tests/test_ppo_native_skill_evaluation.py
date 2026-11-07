"""PPO-only physical domains must not pass through the WM replay layout."""
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.adapters.isaac import RawSnapshot, RawTransition
from supermimic_rl.adapters.packed import PackedEnvironment
from supermimic_rl.ppo.native_skill_evaluation import evaluate_native_policy
from supermimic_rl.compact_state import named_field
from supermimic_rl.chassis_observation import LEGACY_COMMANDS
from supermimic_rl.world_model_v2 import evaluation as old_evaluation
from supermimic_rl.world_model_v2.so3 import quaternion_to_matrix


class NativeAdapter:
    def __init__(self, *, new_domain=False, fail=False):
        self.new_domain, self.fail = new_domain, fail
        self.current, self.index = None, 0
        self.events, self.seen, self.estimates = [], [], []
        self._evaluation_configuration = {"fixed": True}
        self.env = SimpleNamespace(_numerical_invalid=torch.zeros(2, dtype=torch.bool), obs_buf={})
        self.env._sample = lambda: object()
        self.env._observations = SimpleNamespace(assembler=SimpleNamespace(measure=self.measure))
        self.env._commands = lambda: SimpleNamespace(admitted_body_command=torch.zeros(2, 4))

    def snapshot(self, marker, *, reasons=(0., 0.), episode=(0, 0)):
        angle = torch.tensor([.2, -.4])
        q = torch.stack((torch.zeros(2), (angle / 2).sin(), torch.zeros(2), (angle / 2).cos()), -1)
        return RawSnapshot(
            state={"root_orientation_xyzw": q, "failure.failure_reasons": torch.tensor(reasons)[:, None],
                   "sensors.queue": torch.full((2, 4 if self.new_domain else 3, 38), marker)},
            context=({"requested.wheel_aggregate_delay_s": torch.tensor([[.0055, .0035], [.004, .006]])}
                     if self.new_domain else {}),
            observation=torch.full((2, 45), marker), actor_observation=torch.full((2, 45), marker),
            goal=torch.zeros(2, 12), episode_id=torch.tensor(episode), control_epoch=torch.zeros(2, dtype=torch.long),
            policy_owned=torch.ones(2, dtype=torch.bool))

    def publish(self, snapshot):
        self.current = snapshot
        marker = snapshot.actor_observation[:, :1]
        self.env.obs_buf["policy"] = marker.expand(-1, 5 * 42).clone()

    def reset(self):
        self.publish(self.snapshot(.1))
        return self.current

    def evaluation_begin(self, schema):
        self.events.append("begin")
        self.publish(self.snapshot(.2))  # Must repack this fixed-plan reset.

    def evaluation_end(self):
        self.events.append("end")

    def evaluation_step(self, transition, sample):
        self.seen.append((transition, sample))
        self.events.append("evaluate_step")

    def evaluation_summary(self):
        return {"measured": len(self.seen)}

    def measure(self, _):
        self.events.append("measure")
        marker = self.current.actor_observation[:, 0]
        return SimpleNamespace(velocity_origin_body_m_s=torch.stack((marker, marker * 0, marker * 0), -1),
            angular_velocity_body_rad_s=torch.zeros(2, 3), gravity_body=torch.tensor([[0., 0., -1.]] * 2),
            height_above_ground_m=marker)

    def set_policy_estimate(self, frame, raw):
        self.events.append("estimate_hook")
        self.estimates.append((frame.clone(), raw.clone()))
        assert torch.equal(frame, self.current.actor_observation)
        assert torch.equal(raw[:, 3:6], torch.zeros(2, 3))

    def set_training_progress(self, **kwargs):
        raise AssertionError("Evaluation must not advance training exposure")

    def step(self, action):
        self.events.append("step")
        if self.fail:
            raise RuntimeError("simulated engine failure")
        before = self.current
        valid = torch.tensor([True, self.index != 0])
        terminated = torch.tensor([self.index == 0, self.index == 0])
        end = self.snapshot(.4, reasons=(5., 9.), episode=before.episode_id.tolist())
        self.env._numerical_invalid = ~valid
        self.publish(self.snapshot(.3 + .1 * self.index, episode=(1, 1)))
        self.index += 1
        return RawTransition(before, end, action, torch.tensor([1., 2.]), terminated,
                             torch.tensor([False, self.index == 2]), valid)


class NativeActor:
    def __init__(self, adapter):
        self.adapter = adapter
        self.features = SimpleNamespace(configuration=None)
        self.chassis_command_contract = LEGACY_COMMANDS
        self.log_std = torch.zeros(6)
        self.histories = []

    def eval(self):
        return self

    def estimator(self, frames):
        self.adapter.events.append("actor_estimate")
        self.histories.append(frames.clone())
        return frames[:, -1, :5] + torch.arange(5)

    def __call__(self, frames, estimate):
        self.adapter.events.append("actor")
        return frames[:, -1, :6]

    def clip(self, action):
        return action.clamp(-1., 1.)


def fixture(*, new_domain=False, fail=False):
    adapter = NativeAdapter(new_domain=new_domain, fail=fail)
    environment = PackedEnvironment(adapter, source='native-evaluation-test')
    environment.reset()
    actor = NativeActor(adapter)
    cls = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/evaluate_ppo_skills.py"))["NativeHistoryActor"]
    return environment, cls(actor, environment), actor


def test_new_wheel_delay_and_four_slot_ring_use_native_history_and_estimator_before_step():
    env, policy, actor = fixture(new_domain=True)
    from supermimic_rl.world_model_v2.layout import SnapshotLayout
    with pytest.raises(ValueError, match="episode delay DR"):
        SnapshotLayout(env.schema)  # Dataset compatibility remains rejected.
    progress = []
    report = evaluate_native_policy(env, policy, 2, on_progress=progress.append)
    assert report["observed_transitions"] == 4 and report["valid_tracking_samples"] == 3
    assert report["invalid_transitions"] == report["numerical_invalid_transitions"] == 1
    assert report["terminations"] == report["truncations"] == 1
    assert report["termination_masks"] == {"5": 1}
    assert report["task_suite"] == {"measured": 2}
    assert env.adapter.events == ["begin"] + ["measure", "actor_estimate", "actor", "estimate_hook", "step", "evaluate_step"] * 2 + ["end"]
    assert actor.histories[0].eq(.2).all() and actor.histories[1].eq(.3).all()
    assert env.adapter.seen[0][0].end.episode_id.eq(0).all()
    assert env.adapter.seen[1][0].start.episode_id.eq(1).all()
    torch.testing.assert_close(env.adapter.estimates[0][1], torch.tensor([[.2, 1.2, 2.2, 0., 0., 0., 3.2, 4.2]] * 2))
    assert progress[0]["steps"] == 2 and progress[0]["transitions"] == 4
    assert "dataset" not in report and "replay" in report["scope"]


def test_old_domain_statistics_match_existing_collector_loop(monkeypatch):
    # Drive identical real PackedEnvironment transitions through the historical
    # evaluator; only its unused WM encoder/replay is replaced by a CPU driver.
    class Collector:
        def __init__(self, environment, decision, **kwargs):
            self.environment, self.decision = environment, decision
            self.layout = SimpleNamespace(physical=lambda state: SimpleNamespace(
                gravity_body=-quaternion_to_matrix(named_field(state, environment.schema, "root_orientation_xyzw"))[:, 2]))

        def collect(self, steps, **kwargs):
            for index in range(steps):
                before = self.environment.current
                action, estimate = self.decision(None)
                raw = torch.cat((estimate[:, :3], torch.zeros_like(estimate[:, :3]), estimate[:, 3:]), -1)
                self.environment.set_policy_estimate(before.actor_observation, raw)
                kwargs["on_transition"](self.environment.step(action))
                kwargs["on_progress"]({"steps": index + 1})

    monkeypatch.setattr(old_evaluation, "Collector", Collector)
    old_env, old_policy, _ = fixture()
    expected = old_evaluation.evaluate_native(old_env, old_policy, 2)
    env, policy, _ = fixture()
    actual = evaluate_native_policy(env, policy, 2)
    assert set(actual) == set(expected)
    for key, value in expected.items():
        if key in ("scope", "elapsed_s"):
            continue
        if isinstance(value, float):
            assert actual[key] == pytest.approx(value, abs=1e-7), key
        else:
            assert actual[key] == value, key


def test_native_failure_restores_evaluation_hooks_without_advancing_training():
    env, policy, _ = fixture(new_domain=True, fail=True)
    with pytest.raises(RuntimeError, match="engine failure"):
        evaluate_native_policy(env, policy, 1)
    assert env.adapter.events[-1] == "end" and not env.adapter.seen


@pytest.mark.parametrize("mixed", [False, True])
def test_platform_report_keeps_actual_owner_and_distinguishes_flat_anchor_outcomes(mixed):
    env, policy, _ = fixture(new_domain=True)
    platform = SimpleNamespace(platform_rows=torch.tensor([False, True]),
        summary=lambda: {"phase": [0, 3], "success_rate": None})
    if mixed:
        platform.plan = SimpleNamespace(sampler=object())
    env.adapter.env.supermimic_platform_course = platform
    original_begin = env.adapter.evaluation_begin

    def begin(schema):
        original_begin(schema)
        platform.platform_rows = torch.tensor([True, False])

    original_step = env.adapter.evaluation_step

    def step(*args):
        original_step(*args)
        if mixed:
            platform.platform_rows = torch.tensor([False, True])

    suite = {"jump_diagnostics": {"totals": {"success": 0}}, "success_semantics": "Native pulses"}
    env.adapter.evaluation_begin, env.adapter.evaluation_step = begin, step
    env.adapter.evaluation_summary = lambda: suite
    result = evaluate_native_policy(env, policy, 2)
    assert result["platform_diagnostics"] == platform.summary()
    scope = result["platform_evaluation_scope"]
    assert scope["planned_env_ids"] == [0] and scope["recorded_physics_ticks"] is False
    assert "jump_diagnostics" in suite and "jump_diagnostics" not in result["task_suite"]
    diagnostic = result["task_suite"]["flat_anchor_jump_diagnostics"]
    assert diagnostic["totals"] == suite["jump_diagnostics"]["totals"]
    if mixed:
        assert "initial episode" in scope["planned_env_ids_scope"]
        assert diagnostic["platform_interpretation"]["includes_platform_rows"] is None
        assert "not a full-run" in diagnostic["platform_interpretation"]["planned_env_ids_scope"]
    else:
        assert diagnostic["platform_interpretation"]["includes_platform_rows"] is True


def test_platform_without_summary_does_not_fabricate_capability():
    env, policy, _ = fixture(new_domain=True)
    env.adapter.env.supermimic_platform_course = SimpleNamespace(platform_rows=torch.tensor([True, False]))
    report = evaluate_native_policy(env, policy, 1)
    assert report["platform_diagnostics"]["status"] == "unknown"
    assert "final snapshots are not a success rate" in report["platform_evaluation_scope"]["capability_source"]


@pytest.mark.parametrize("steps", [0, -1, True, 1.5])
def test_invalid_exposure_rejected_before_native_work(steps):
    env, policy, _ = fixture()
    with pytest.raises(ValueError, match="positive integer"):
        evaluate_native_policy(env, policy, steps)
    assert not env.adapter.events
