"""Preserve requested PhysX average-contact friction on shared Kamino terrain.

Kamino multiplies two Coulomb coefficients. A shared terrain cannot carry a
different coefficient for each environment, so normalize its native mu to one
and put the requested pair coefficient on each robot shape. This preserves
robot-ground friction, not robot-robot friction. Native self-collision filters
are therefore checked before applying the mapping. Geometry is never written.
"""
from __future__ import annotations

import hashlib

import torch


MATERIAL_SOURCE_PINS = {
    "contract": "5e2e64523bd1c3ed0a40b537b8fef7cfc48d9b8173860550d470e98da8272157",
    "contact_material.py": "991ba7efa3d2ee1ed4be00d19ff5630615cff1c60eaf162f00e258304ff273eb",
    "mechanism.json": "25a1e69b650f99416f40b2a3d1312c037b348c2d06d134bd9e5e5f19be28f6d8",
    "randomization_study.py": "80ec179c8af89f85f0439c0fd15e78815860a78c98ec2ab99eb6de6af187dec1",
}


def material_mapping_provenance():
    return {
        "schema": "requested_average_to_kamino_multiply_v1",
        "source_root": "/home/kaiser/workspace/isaac_sim-6-task-200hz",
        "source_contract": "contracts/wheel_leg_randomization_study_v2.json",
        "source_mechanism": "model/wheel_leg_v2.0.2/friction_v1/training/mechanism.json",
        "source_sha256": dict(MATERIAL_SOURCE_PINS),
        "source_combine_mode": "average", "native_combine_mode": "multiply",
        "source_ground_mu_range": [.3, .5], "source_wheel_mu": .1, "source_body_mu": .25,
        "sampling": "one CPU uniform ground draw per environment at startup; seed xor 0x434F4E54",
        "nominal_lanes": "ground still randomized, as in the source contact domain",
        "native_terrain_mu": 1., "native_wheel_mu": "(.1 + ground_mu) / 2",
        "native_other_body_mu": "(.25 + ground_mu) / 2",
        "terrain_geometry_changed": False,
        "terrain_query_mu": "authored metadata only; native contact material is normalized",
        "scope": "exact robot-ground Coulomb coefficients; requires native self collisions disabled",
        "classification": "requested simulation distribution, not measured hardware friction",
    }


def sample_ground_friction(num_envs, seed):
    """Match ContactMaterialDomain's equal static/dynamic band draw exactly."""
    generator = torch.Generator(device="cpu").manual_seed(seed ^ 0x434F4E54)
    return .3 + (.5 - .3) * torch.rand(num_envs, generator=generator)


def _digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _self_collision_receipt(model, worlds, bodies, groups, num_envs, shape_count):
    """Check actual group rules and explicit filters in bounded CPU chunks."""
    ids = ((worlds >= 0) & (bodies >= 0)).nonzero().flatten()
    selected_worlds = worlds[ids]
    if (selected_worlds.numel() != num_envs * shape_count
            or not torch.equal(torch.bincount(selected_worlds, minlength=num_envs),
                               torch.full((num_envs,), shape_count, dtype=torch.long))):
        raise ValueError("Contact mapping requires exactly one selected robot per Newton world")
    indices = ids[selected_worlds.argsort(stable=True)].reshape(num_envs, shape_count)
    left, right = torch.triu_indices(shape_count, shape_count, offset=1)
    candidate_counts, live_counts = [], []
    for start in range(0, num_envs, 128):
        a, b = indices[start:start + 128, left], indices[start:start + 128, right]
        different_body = bodies[a] != bodies[b]
        ga, gb = groups[a], groups[b]
        # Newton ModelBuilder._test_group_pair / broad_phase_common use these
        # rules. Equal negative groups and zero groups never collide.
        group_allows = ((ga != 0) & (gb != 0)
                        & torch.where(ga > 0, (ga == gb) | (gb < 0), ga != gb))
        possible = different_body & group_allows
        remaining = torch.zeros_like(possible)
        if possible.any():
            pairs = torch.stack((a[possible], b[possible]), -1).numpy()
            filtered = torch.as_tensor(model.shape_collision_filter_mask(pairs), dtype=torch.bool)
            if filtered.shape != (len(pairs),):
                raise ValueError("Newton collision filter mask has an invalid shape")
            remaining[possible] = ~filtered
        candidate_counts.append(different_body.sum(-1))
        live_counts.append(remaining.sum(-1))
    counts, live = torch.cat(candidate_counts), torch.cat(live_counts)
    result = {
        "source": "native shape_world/body/group and Model.shape_collision_filter_mask",
        "cross_body_pair_count_range": [int(counts.min()), int(counts.max())],
        "unfiltered_cross_body_pair_count_range": [int(live.min()), int(live.max())],
        "all_robot_self_collisions_disabled": bool((live == 0).all()),
    }
    if not result["all_robot_self_collisions_disabled"]:
        raise ValueError("Requested contact mapping requires disabled native robot self collisions: " + str(result))
    return result


def make_material_reader_type(base_type, randomization, *, solver_config=None):
    """Wrap the native reader during construction; never reapply its palette."""
    class RequestedMaterialReader(base_type):
        @torch.no_grad()
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            from transformer_rl.formal.lab_env.newton_properties import native_tensor

            config = solver_config
            if config is None:
                from isaaclab_newton.physics import NewtonManager
                # Newton 1.6.1's actual solver config has no public accessor.
                # Read its explicit configuration; never infer the mix mode.
                config = NewtonManager._solver._config
            if config.materials.friction_mix_mode != "multiply":
                raise ValueError("Requested contact mapping requires actual Kamino multiply friction")
            if self.num_envs != randomization.num_envs:
                raise ValueError("Requested material batch differs from randomization")
            if self._receipt["wheel_body_names"] != ["L_link3", "R_link3"]:
                raise ValueError("Requested material mapping requires named left/right wheel bodies")
            model = self._properties.model
            worlds = native_tensor(model.shape_world).detach().cpu().long()
            bodies = native_tensor(model.shape_body).detach().cpu().long()
            groups = native_tensor(model.shape_collision_group).detach().cpu().long()
            labels = tuple(model.shape_label)
            terrain = torch.tensor([label == "/World/ground" or label.startswith("/World/ground/")
                                    for label in labels], dtype=torch.bool)
            if (worlds.ndim != 1 or bodies.shape != worlds.shape or groups.shape != worlds.shape
                    or terrain.shape != worlds.shape or not terrain.any()
                    or (worlds[terrain] != -1).any() or (bodies[terrain] != -1).any()):
                raise ValueError("Contact mapping requires actual shared static /World/ground shapes")
            filters = _self_collision_receipt(model, worlds, bodies, groups,
                                             self.num_envs, self._properties.shape_count)
            native_mu = native_tensor(model.shape_material_mu)
            before_terrain = native_mu[terrain.to(native_mu.device)].clone()
            before_mu, before_e = self._read_native()
            ground = sample_ground_friction(self.num_envs, randomization.seed)
            wheel, body = (.1 + ground) / 2, (.25 + ground) / 2
            target = body.to(before_mu.device)[:, None].expand_as(before_mu).clone()
            for indices in self._shapes:
                target[:, indices] = wheel.to(target.device)[:, None]
            self._properties.write("shape_material_mu", target)
            native_mu[terrain.to(native_mu.device)] = 1.
            self._properties.publish()
            actual_mu, actual_e = self._read_native()
            actual_terrain = native_tensor(model.shape_material_mu)[terrain.to(native_mu.device)]
            if (not torch.equal(target, actual_mu) or not torch.equal(before_e, actual_e)
                    or not torch.equal(actual_terrain, torch.ones_like(actual_terrain))):
                raise RuntimeError("Requested contact material native readback differs")
            self._publish(actual_mu, actual_e)
            self._startup_applied = True
            self._receipt.update({
                "friction_mapping": "requested average pair coefficient on robot; shared terrain mu=1",
                "terrain_material_substitution": True,
                "requested_contact_mapping": {
                    **material_mapping_provenance(), "status": "native_readback_matched",
                    "actual_solver_config_source": "NewtonManager._solver._config.materials",
                    "seed": randomization.seed, "ground_draw_sha256": _digest(ground),
                    "robot_mu_sha256": _digest(actual_mu), "terrain_shape_count": int(terrain.sum()),
                    "terrain_authored_mu_range": [float(before_terrain.min()), float(before_terrain.max())],
                    "terrain_native_mu_range": [float(actual_terrain.min()), float(actual_terrain.max())],
                    "source_ground_draw_range": [float(ground.min()), float(ground.max())],
                    "effective_wheel_ground_mu_range": [float(wheel.min()), float(wheel.max())],
                    "effective_body_ground_mu_range": [float(body.min()), float(body.max())],
                    "restitution_unchanged": True, "self_collision": filters,
                },
            })
            device = randomization.device
            randomization.material_context = {
                "source_ground_friction": ground[:, None].to(device),
                "native_terrain_friction": torch.ones(self.num_envs, 1, device=device),
                "effective_wheel_ground_friction": self.features()[:, [0, 3]].to(device),
                "effective_body_ground_friction": actual_mu[:, next(
                    i for i in range(actual_mu.shape[1]) if all(i not in group for group in self._shapes)
                ), None].to(device),
            }

        def randomize_startup(self, *args, **kwargs):
            raise ValueError("Requested contact mapping owns startup materials; set legacy wheel_material to null")

    return RequestedMaterialReader
