"""Run a pinned chassis actor in the full native articulation with active gimbal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import torch

from supermimic_rl.adapters.factory import create_isaac
from supermimic_rl.transferred_actor import TransferredActor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factory-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    adapter = None

    def write(name, value):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    try:
        if args.steps < 1:
            raise ValueError("probe steps must be positive")
        configuration = json.loads(Path(args.factory_config).read_text())
        configuration["native_log_directory"] = str(output)
        adapter = create_isaac(configuration)
        write("native-provenance.json", adapter.provenance)
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        actor = TransferredActor(payload["recipe"]["policy"], history_length=5)
        actor.load_initial_checkpoint(args.checkpoint, args.checkpoint_sha256)
        actor.to(adapter.device).eval().requires_grad_(False)
        write("actor-source.json", actor.source_metadata)
        current = adapter.reset()
        observer_error = None
        compact_width = None
        if configuration.get("compact_state", False):
            from supermimic_rl.adapters.packed import PackedEnvironment
            from supermimic_rl.compact_state import analytic_actor_observation
            packed = PackedEnvironment(adapter, source="full-vehicle-native-probe")
            snapshot = packed._pack(current)
            reconstructed = analytic_actor_observation(
                snapshot.state, snapshot.context, snapshot.goal, None, packed.schema)
            observer_error = float((reconstructed - snapshot.observation).abs().max())
            compact_width = packed.schema.state_dim
            if observer_error > 2e-4:
                raise RuntimeError(f"compact analytic observer differs from native clean35: {observer_error}")
        history = current.actor_observation[:, None].expand(-1, 5, -1).clone()
        count, device = adapter.num_envs, adapter.device
        failures = torch.zeros(count, device=device, dtype=torch.long)
        first_failure = torch.full_like(failures, args.steps + 1)
        error_sum = torch.zeros(3, device=device, dtype=torch.float64)
        gimbal_velocity_max = torch.zeros(2, device=device)
        gimbal_position_min = torch.full((2,), float("inf"), device=device)
        gimbal_position_max = -gimbal_position_min
        gimbal_effort_max = torch.zeros(2, device=device)
        masks = {}
        with torch.no_grad(), (output / "steps.jsonl").open("w") as stream:
            for step in range(1, args.steps + 1):
                native = adapter.env
                measured = native._observations.assembler.measure(native._sample())
                command = native._commands().admitted_body_command
                actual = torch.stack((measured.velocity_origin_body_m_s[:, 0],
                                      measured.angular_velocity_body_rad_s[:, 2],
                                      measured.height_above_ground_m), dim=-1)
                error_sum += (actual - command[:, [0, 2, 3]]).double().square().sum(0)
                transition = adapter.step(actor(history))
                if not transition.valid.all():
                    raise RuntimeError("native probe yielded an invalid interval")
                end = transition.end
                if "gimbal.position_rad" in end.state:
                    q = end.state["gimbal.position_rad"]
                else:
                    phase = end.state["gimbal.yaw_sin_cos"]
                    q = torch.cat((torch.atan2(phase[:, :1], phase[:, 1:]),
                                   end.state["gimbal.pitch_position_rad"]), dim=-1)
                dq = end.state["gimbal.velocity_rad_s"]
                effort = end.state["gimbal.effort_nm"]
                gimbal_velocity_max = torch.maximum(gimbal_velocity_max, dq.abs().amax(0))
                gimbal_effort_max = torch.maximum(gimbal_effort_max, effort.abs().amax(0))
                gimbal_position_min = torch.minimum(gimbal_position_min, q.amin(0))
                gimbal_position_max = torch.maximum(gimbal_position_max, q.amax(0))
                failures += transition.terminated.long()
                first_failure = torch.where(transition.terminated & (first_failure > args.steps), step, first_failure)
                bits = end.state["failure.failure_reasons"][transition.terminated]
                for mask, total in zip(*[v.cpu().tolist() for v in bits.unique(return_counts=True)]):
                    masks[str(int(mask))] = masks.get(str(int(mask)), 0) + total
                current = adapter.current
                reset = transition.terminated | transition.truncated
                history = torch.cat((history[:, 1:], current.actor_observation[:, None]), dim=1)
                history[reset] = current.actor_observation[reset, None].expand(-1, 5, -1)
                if step % 50 == 0 or step == args.steps:
                    status = {"step": step, "failures": int(failures.sum()),
                              "gimbal_velocity_max_rad_s": gimbal_velocity_max.cpu().tolist()}
                    stream.write(json.dumps(status) + "\n")
                    stream.flush()
                    write("progress.json", status)
        summary = {
            "status": "completed", "steps": args.steps, "num_envs": count,
            "physical_duration_per_env_s": args.steps * adapter.policy_dt_s,
            "failure_count": int(failures.sum()), "failure_reason_bitmasks": masks,
            "failure_free_envs": int((failures == 0).sum()),
            "first_failure_time_s": [None if x > args.steps else x * adapter.policy_dt_s
                                     for x in first_failure.cpu().tolist()],
            "body_axis_tracking_rmse_vx_wz_height": (error_sum / (count * args.steps)).sqrt().cpu().tolist(),
            "gimbal_position_min_rad": gimbal_position_min.cpu().tolist(),
            "gimbal_position_max_rad": gimbal_position_max.cpu().tolist(),
            "gimbal_position_representation": "yaw phase modulo 2*pi and native pitch" if compact_width else "native joint encoders",
            "gimbal_native_velocity_max_rad_s": gimbal_velocity_max.cpu().tolist(),
            "gimbal_native_effort_max_nm": gimbal_effort_max.cpu().tolist(),
            "compact_state_dim": compact_width,
            "compact_observer_max_error_at_reset": observer_error,
            "scope": "native integration and transferred actor probe; no SuperMimic training updates",
        }
        write("summary.json", summary)
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        write("failure.json", {"type": type(error).__name__, "message": str(error),
                               "traceback": traceback.format_exc()})
        raise
    finally:
        if adapter is not None:
            adapter.close()


if __name__ == "__main__":
    main()
