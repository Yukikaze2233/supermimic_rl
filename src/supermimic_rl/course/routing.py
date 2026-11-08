"""Route explicit authored-step endpoints to actual immutable atlas assets.

The upstream router already provides 20% ceiling exploration. Its ordinary
route also limits rows by past proficiency, so a 0.20 m request alone does not
mean the robot was placed on an authored mesh. This wrapper makes that mapping
explicit from the authored stage, without changing any non-endpoint route or
the upstream RNG, terrain geometry, spawn placement or difficulty progression.
"""
from __future__ import annotations

import math

import torch


_AUTHORED = {
    "road_step": (3, (0., .20)),
    "central_steps": (7, (0., .20, .35, .30)),
}


def authored_tile_keys(router) -> dict[str, int]:
    """Resolve source asset names and profiles; never assume a numeric tile ID."""
    result = {}
    for index, tile in enumerate(router.manifest["tiles"]):
        name = tile.get("asset")
        if name not in _AUTHORED:
            continue
        kind, profile = _AUTHORED[name]
        actual = tile.get("profile_m")
        roi = tile.get("source_roi_m")
        if (name in result or tile.get("name") != name or tile.get("kind") != kind
                or not isinstance(actual, list) or len(actual) != len(profile)
                or any(not math.isfinite(v) or abs(v - expected) > 1e-7 for v, expected in zip(actual, profile))
                or not isinstance(roi, list) or len(roi) != 2
                or any(len(corner) != 2 for corner in roi)
                or any(not math.isfinite(value) for corner in roi for value in corner)
                or any(roi[0][axis] >= roi[1][axis] for axis in range(2))
                or not tile.get("source_manifest_sha256")
                or abs(float(router.step_table[index]) - profile[1]) > 1e-7):
            raise ValueError("Authored terrain asset/profile/ROI identity differs: " + name)
        result[name] = index
    if set(result) != set(_AUTHORED):
        raise ValueError("The actual atlas must contain both authored road_step and central_steps")
    return {name: result[name] for name in _AUTHORED}


class AuthoredEndpointRouter:
    """Transparent router wrapper with actual-placement counters.

    Explicit endpoint requests in the two authored kinds bypass proficiency
    only after ``minimum_stage`` and within the declared course ceiling. Other
    rows retain the upstream completed-episode levels and continuous requests.
    The opt-in foundation rehearsal additionally admits only its certified
    birth rows at the real asset endpoint, leaving the ordinary caps unchanged.
    Counts describe reset placement, never traversal, success or mastery.
    """

    _FORMAT = "supermimic_authored_endpoint_router_v1"

    def __init__(self, router, *, minimum_stage=4, rehearsal_curriculum=None):
        if type(minimum_stage) is not int or minimum_stage < 0:
            raise ValueError("Authored routing requires an explicit nonnegative minimum stage")
        self._router = router
        self.minimum_stage = minimum_stage
        self.authored_tile_keys = authored_tile_keys(router)
        self._actual_tile_counts = torch.zeros(len(router.kinds), dtype=torch.int64, device=router.device)
        self._forced_counts = torch.zeros(len(_AUTHORED), dtype=torch.int64, device=router.device)
        self._last_forced = torch.zeros(router.num_envs, dtype=torch.bool, device=router.device)
        self._rehearsal_curriculum = rehearsal_curriculum
        self._rehearsal_receipt = (None if rehearsal_curriculum is None else
                                  rehearsal_curriculum.foundation_authored_rehearsal_receipt())
        if rehearsal_curriculum is not None:
            if (self._rehearsal_receipt is None or rehearsal_curriculum.num_envs != router.num_envs
                    or rehearsal_curriculum.device != router.device
                    or any(abs(self._rehearsal_receipt['requested_step_up_m'] - profile[1]) > 1e-7
                           for _, profile in _AUTHORED.values())):
                raise ValueError("Foundation rehearsal must declare this atlas's actual endpoint")
            self._rehearsal_counts = torch.zeros(len(_AUTHORED), dtype=torch.int64, device=router.device)

    def __getattr__(self, name):
        return getattr(self._router, name)

    def reset(self, ids, kind, *, stage, command_forward, requested=None,
              level_cap=None, step_up_cap_m=None):
        if type(stage) is not int or stage < 0:
            raise ValueError("Terrain routing requires a nonnegative integer stage")
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        kind = torch.as_tensor(kind, dtype=torch.long, device=self.device)
        rehearsal = torch.zeros(len(ids), dtype=torch.bool, device=self.device)
        if self._rehearsal_receipt is not None and stage == self._rehearsal_receipt['stage']:
            rehearsal = self._rehearsal_curriculum.foundation_authored_rehearsal_rows(ids)
            allowed_kind = torch.isin(kind, kind.new_tensor([value[0] for value in _AUTHORED.values()]))
            if (requested is None or (rehearsal & ~allowed_kind).any()
                    or (rehearsal & ((requested['step_up_m']
                                     - self._rehearsal_receipt['requested_step_up_m']).abs() > 1e-7)).any()):
                raise ValueError("Authored rehearsal draw must retain its actual kind and endpoint request")
        # Parent validation and RNG consumption are unchanged, including its
        # explicit ceiling exploration. Only the selected endpoint tile changes.
        self._router.reset(
            ids, kind, stage=stage, command_forward=command_forward,
            requested=requested, level_cap=level_cap, step_up_cap_m=step_up_cap_m)
        forced = torch.zeros(len(ids), dtype=torch.bool, device=self.device)
        if requested is not None and step_up_cap_m is not None:
            for slot, (name, (expected_kind, profile)) in enumerate(_AUTHORED.items()):
                height = profile[1]
                eligible = ((kind == expected_kind) & (requested["step_up_m"] >= height - 1e-7)
                            & (((stage >= self.minimum_stage) & (step_up_cap_m >= height - 1e-7))
                               | rehearsal))
                selected_rows = ids[eligible]
                tile = self.authored_tile_keys[name]
                self._router.tile[selected_rows] = tile
                self._router.origins[selected_rows] = self._router.origins_table[tile]
                forced |= eligible
                self._forced_counts[slot] += eligible.sum()
                if self._rehearsal_receipt is not None:
                    self._rehearsal_counts[slot] += (eligible & rehearsal).sum()
        self._last_forced[ids] = forced
        selected = self._router.tile[ids]
        self._actual_tile_counts += torch.bincount(selected, minlength=len(self.kinds))
        return self._router.origins[ids], self._router.spawn_z[selected]

    def receipt(self) -> dict:
        result = {
            "contract": self._FORMAT, "atlas_sha256": self.identity,
            "minimum_stage": self.minimum_stage,
            "selection": "matching authored kind AND endpoint request AND sufficient explicit course ceiling",
            "ordinary_routes": "unchanged upstream request/proficiency/20-percent ceiling exploration",
            "extra_random_draws": 0,
            "authored_tiles": {
                name: {"tile_key": index, "kind": _AUTHORED[name][0],
                       "profile_m": list(_AUTHORED[name][1]),
                       "source_roi_m": self.manifest["tiles"][index]["source_roi_m"],
                       "source_manifest_sha256": self.manifest["tiles"][index]["source_manifest_sha256"]}
                for name, index in self.authored_tile_keys.items()
            },
        }
        if self._rehearsal_receipt is not None:
            result['foundation_rehearsal'] = dict(self._rehearsal_receipt)
            result['selection'] += " OR certified foundation authored-rehearsal birth at its declared endpoint"
        return result

    def actual_metrics(self) -> dict:
        counts = self._actual_tile_counts.tolist()
        forced = self._forced_counts.tolist()
        result = {
            "scope": "actual reset placements since installation; persisted across checkpoint restore",
            "actual_reset_count": sum(counts), "actual_tile_reset_counts": counts,
            "authored_reset_counts": {name: counts[index] for name, index in self.authored_tile_keys.items()},
            "explicit_endpoint_reset_counts": {name: forced[i] for i, name in enumerate(_AUTHORED)},
            "success_semantics": "none; traversal requires separate physical root/wheel profile evidence",
        }
        if self._rehearsal_receipt is not None:
            result['foundation_rehearsal_reset_counts'] = dict(zip(_AUTHORED, self._rehearsal_counts.tolist()))
        return result

    def stair_coverage_metrics(self) -> dict:
        return {
            "upstream_selection_before_endpoint_override": self._router.stair_coverage_metrics(),
            **self.actual_metrics(),
        }

    def state_dict(self) -> dict:
        result = {
            "format": self._FORMAT, "minimum_stage": self.minimum_stage,
            "authored_tile_keys": dict(self.authored_tile_keys), "base": self._router.state_dict(),
            "actual_tile_counts": self._actual_tile_counts.clone(),
            "forced_counts": self._forced_counts.clone(), "last_forced": self._last_forced.clone(),
        }
        if self._rehearsal_receipt is not None:
            result.update(foundation_rehearsal=dict(self._rehearsal_receipt),
                          rehearsal_counts=self._rehearsal_counts.clone())
        return result

    def load_state_dict(self, state) -> None:
        expected = self.state_dict()
        if (not isinstance(state, dict) or state.keys() != expected.keys()
                or any(state[key] != expected[key] for key in ("format", "minimum_stage", "authored_tile_keys"))):
            raise ValueError("Authored terrain routing checkpoint contract differs")
        if self._rehearsal_receipt is not None and state['foundation_rehearsal'] != self._rehearsal_receipt:
            raise ValueError("Foundation authored rehearsal checkpoint contract differs")
        keys = ("actual_tile_counts", "forced_counts", "last_forced")
        if self._rehearsal_receipt is not None:
            keys += ('rehearsal_counts',)
        for key in keys:
            value = state[key]
            if (not isinstance(value, torch.Tensor) or value.shape != expected[key].shape
                    or value.dtype != expected[key].dtype or (value < 0).any()):
                raise ValueError("Invalid authored routing placement counts")
        for slot, index in enumerate(self.authored_tile_keys.values()):
            if state["forced_counts"][slot] > state["actual_tile_counts"][index]:
                raise ValueError("Forced authored counts exceed actual tile placements")
        if (self._rehearsal_receipt is not None
                and (state['rehearsal_counts'] > state['forced_counts']).any()):
            raise ValueError("Rehearsal counts exceed forced authored placements")
        self._router.load_state_dict(state["base"])
        self._actual_tile_counts.copy_(state["actual_tile_counts"])
        self._forced_counts.copy_(state["forced_counts"])
        self._last_forced.copy_(state["last_forced"])
        if self._rehearsal_receipt is not None:
            self._rehearsal_counts.copy_(state['rehearsal_counts'])


def install_authored_routing(environment, *, minimum_stage=4, rehearsal_curriculum=None) -> AuthoredEndpointRouter:
    """Install after set_task_sampler creates the real LabTerrainCourse."""
    course = environment._terrain_course
    if course is None:
        raise ValueError("Install the native terrain course before authored routing")
    if isinstance(course.router, AuthoredEndpointRouter):
        if (course.router.minimum_stage != minimum_stage
                or course.router._rehearsal_curriculum is not rehearsal_curriculum):
            raise ValueError("Authored routing already has a different stage contract")
        return course.router
    router = AuthoredEndpointRouter(course.router, minimum_stage=minimum_stage,
                                    rehearsal_curriculum=rehearsal_curriculum)
    course.router = router
    environment.authored_routing_receipt = router.receipt()
    return router
