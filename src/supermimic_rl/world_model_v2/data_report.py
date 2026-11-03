"""Per-skill data coverage and prediction quality of a compact world-model replay.

A teacher's data helps the world model when it covers the contact phases and
actions of a skill and the model predicts them; return or success rate alone
does not say that. Failed attempts are kept: they carry the contacts and
recoveries the imagination must also predict. Nothing here is a training gate.

Populations follow the native command mode and terrain kind recorded at the
interval start (``reward_route``): ordinary flat, gimbal special motion,
terrain, jump and recovery.
"""
from __future__ import annotations

import torch

from .model import Ticks
from .records import CONTACT_BODIES, FORBIDDEN_BODIES, decode_physical
from .validation import validate

# Native CommandMode: NORMAL 0, STAIR 1, SLOPE 2, RECOVERY 3, JUMP 4.
ORDINARY_MODES = (0, 1, 2)
SUPPORT_ON_N, SUPPORT_OFF_N, BODY_CONTACT_N = 5., 2., 3.
ACTION_QUANTILES = (.01, .05, .5, .95, .99)
LEGACY_LEG_BOUND = 3.


def populations(replay) -> dict[str, torch.Tensor]:
    """Boolean masks over physical replay slots ``[capacity, num_envs]``."""
    route = replay.data["reward_route"]
    mode, terrain, special = route[..., 0], route[..., 1], route[..., 2]
    recorded = (route >= 0).all(-1)
    ordinary = recorded & torch.isin(mode, mode.new_tensor(ORDINARY_MODES))
    filled = torch.zeros_like(recorded)
    filled[replay._chronological()] = True
    owned = filled & replay.data["valid"]
    return {"all": owned, "flat_ordinary": owned & ordinary & (terrain == 0) & (special == 0),
            "gimbal_special": owned & ordinary & (terrain == 0) & (special != 0),
            "terrain": owned & ordinary & (terrain != 0), "jump": owned & recorded & (mode == 4),
            "recovery": owned & recorded & (mode == 3), "unrouted": owned & ~recorded}


@torch.no_grad()
def coverage(replay, mask: torch.Tensor, *, chunk: int = 65536) -> dict:
    """Action and contact-phase coverage of the selected recorded intervals."""
    rows = mask.reshape(-1).nonzero().flatten()
    if not rows.numel():
        return {"transitions": 0}
    data = {name: replay.data[name].flatten(0, 1) for name in ("action", "ticks", "physical", "terminated")}
    actions, phases, terminated = [], torch.zeros(5, dtype=torch.float64), 0
    for part in rows.split(chunk):
        actions.append(data["action"][part].float())
        ticks = Ticks.decode(data["ticks"][part], decode_physical(data["physical"][part]))
        support = ticks.wheel_load
        on, off = (support >= SUPPORT_ON_N), (support <= SUPPORT_OFF_N)
        base = ticks.body_normal_load[..., CONTACT_BODIES.index("base_link")] > BODY_CONTACT_N
        forbidden = (ticks.forbidden_load > BODY_CONTACT_N).any(-1)
        for index, flag in enumerate((on.all(-1), off.all(-1), on.any(-1) & off.any(-1), base, forbidden)):
            phases[index] += flag.double().sum()
        terminated += int(data["terminated"][part].sum())
    action = torch.cat(actions)
    ticks_total = rows.numel() * 4
    quantiles = torch.quantile(action[torch.randperm(len(action))[:200000]],
                               torch.tensor(ACTION_QUANTILES), dim=0)
    return {"transitions": int(rows.numel()), "terminated_fraction": terminated / rows.numel(),
            "action_quantiles": {f"q{int(q * 100):02d}": values.tolist() for q, values in zip(ACTION_QUANTILES, quantiles)},
            "leg_action_beyond_legacy_bound_fraction": float((action[:, :4].abs() > LEGACY_LEG_BOUND).float().mean()),
            "tick_phase_fraction": dict(zip(("both_wheels_supported", "both_wheels_airborne", "one_wheel_supported",
                                             "base_contact", "forbidden_contact"),
                                            (phases / ticks_total).tolist())),
            "thresholds_n": {"support_on": SUPPORT_ON_N, "support_off": SUPPORT_OFF_N, "body_contact": BODY_CONTACT_N}}


def prediction(fit, replay, env_ids, mask, *, horizon, batch, generator, device,
               contact_floor=None, termination=None) -> dict:
    """Held-out multi-step errors and forbidden-contact confusion starting in the population."""
    threshold = BODY_CONTACT_N if termination is None else termination.limits.forbidden_body_load_n
    return validate(fit, replay, env_ids, horizon, batch, generator, device,
                    contact_threshold_n=threshold, start_rows=mask,
                    contact_floor=contact_floor, termination=termination)


def report(replay, *, fit=None, env_ids=None, horizon=16, batch=1024, generator=None, device="cpu",
           contact_floor=None, termination=None) -> dict:
    result = {"scope": "coverage of recorded intervals; prediction on held-out rows when a model is given",
              "forbidden_bodies": list(FORBIDDEN_BODIES), "populations": {},
              "contact_prediction": {
                  "saved_floor_applied": contact_floor is not None,
                  "calibration": "saved floors are not refitted; fresh sampled windows share the held-out environment population",
                  "confirmation_clock": "audited with pinned native limits" if termination is not None else
                      "not audited; provide the pinned runtime JSON",
                  "contact_threshold_source": "pinned native forbidden-body threshold" if termination is not None else
                      "3 N body-contact coverage threshold; native failure threshold unavailable",
                  "missing_positive_labels": "recall is null, never interpreted as perfect recall"}}
    if contact_floor is not None:
        result['contact_prediction']['body_floor_n'] = contact_floor[0].tolist()
        result['contact_prediction']['base_resultant_floor_n'] = float(contact_floor[1])
    for name, mask in populations(replay).items():
        entry = {"coverage": coverage(replay, mask)}
        if fit is not None and entry["coverage"]["transitions"]:
            entry["prediction"] = prediction(fit, replay, env_ids, mask, horizon=horizon, batch=batch,
                                             generator=generator, device=device,
                                             contact_floor=contact_floor, termination=termination)
        result["populations"][name] = entry
    return result
