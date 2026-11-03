"""Per-skill populations and coverage of a compact replay, without a world model."""
import torch

from supermimic_rl.world_model_v2.data_report import coverage, populations, report
from supermimic_rl.world_model_v2.records import CONTACT_BODIES, RecordSpec, TICK_LABEL_DIM
from supermimic_rl.world_model_v2.physical import INCREMENT_DIM
from supermimic_rl.world_model_v2.replay import CompactReplay

SUPPORT = slice(INCREMENT_DIM, INCREMENT_DIM + 2)
BASE = INCREMENT_DIM + 2 + CONTACT_BODIES.index("base_link")


def replay_fixture():
    fields = RecordSpec(goal_dim=11, sim_context_dim=65).fields()
    replay = CompactReplay(fields, num_envs=4, capacity_steps=3, device="cpu")
    boundary = {name: torch.zeros(4, dtype=torch.bool) for name in ("terminated", "truncated")}
    boundary.update({name: torch.ones(4, dtype=torch.bool) for name in ("valid", "policy_owned", "end_policy_owned")})
    boundary.update({name: torch.zeros(4, dtype=torch.long) for name in
                     ("start_episode_id", "end_episode_id", "start_control_epoch", "end_control_epoch")})
    for step in range(2):
        record = {name: torch.zeros(4, *shape) for name, shape in fields.items()}
        record["physical"][:, 6] = 1.   # identity quaternion (xyzw)
        # Rows: flat ordinary, terrain, jump, recovery.
        record["reward_route"] = torch.tensor([[0., 0, 0], [0, 3, 0], [4, 0, 0], [3, 0, 0]])
        ticks = torch.zeros(4, 4, TICK_LABEL_DIM)
        ticks[0, :, SUPPORT] = 50.          # flat row fully supported
        ticks[2, :, SUPPORT] = 0.           # jump row airborne
        ticks[3, :, BASE] = 20.             # recovery row lies on its base
        ticks[1, :, SUPPORT] = torch.tensor([50., 0.])  # terrain row on one wheel
        record["ticks"] = ticks
        record["action"] = torch.tensor([[0., 0, 0, 0, 1, 1], [1, 1, 1, 1, 0, 0],
                                         [5, -5, 5, -5, 0, 0], [12, 12, -12, -12, 0, 0]])
        replay.add(record, boundary)
    return replay


def test_populations_follow_the_recorded_native_route():
    masks = populations(replay_fixture())
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    assert counts == {"all": 8, "flat_ordinary": 2, "gimbal_special": 0, "terrain": 2, "jump": 2,
                      "recovery": 2, "unrouted": 0}


def test_coverage_reports_contact_phases_and_wide_leg_actions():
    replay = replay_fixture()
    masks = populations(replay)
    flat, jump, recovery = (coverage(replay, masks[name]) for name in ("flat_ordinary", "jump", "recovery"))
    assert flat["tick_phase_fraction"]["both_wheels_supported"] == 1.
    assert jump["tick_phase_fraction"]["both_wheels_airborne"] == 1.
    assert coverage(replay, masks["terrain"])["tick_phase_fraction"]["one_wheel_supported"] == 1.
    assert recovery["tick_phase_fraction"]["base_contact"] == 1.
    assert recovery["leg_action_beyond_legacy_bound_fraction"] == 1.
    assert flat["leg_action_beyond_legacy_bound_fraction"] == 0.
    assert report(replay)["populations"]["gimbal_special"]["coverage"] == {"transitions": 0}
