"""Native contact-point attribution and request-bound platform landing history.

Patch positions certify surface contact; wheel-centre columns do not. Flight
and the support gap band additionally require a current whole-cylinder query.
The arbitrary-triangle atlas clearance provider is an explicit integration
dependency, not silently replaced by a point/rim sample in this observer.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor

from .jump_physics import PlatformLandingEvidence, PlatformLandingState, observe_jump_physics
from .platform_intent import PlatformIntentState


CLEARANCE_REFERENCE = 'whole_cylinder_minimum_world_z_gap_over_own_terrain_footprint'


@dataclass(frozen=True)
class PlatformLandingInterval:
    """Four actual physics ticks and their post-update landing histories.

    Heights retain native world Z, including unknown NaN values when support
    cannot be certified. The evidence is not inferred from compact load labels.
    """

    evidence: PlatformLandingEvidence
    state: PlatformLandingState

    def validate(self, count, device):
        self.evidence.validate((count, 4), device)
        self.state.validate((count, 4), device)


@dataclass(frozen=True)
class PlatformContactPatches:
    """Active native patches, retaining exact environment/body ownership."""

    row: Tensor
    body: Tensor
    position_world_m: Tensor
    normal_world: Tensor
    normal_load_n: Tensor

    @classmethod
    def from_native(cls, packet, loads):
        from transformer_rl.formal.simulation.contacts import NativeTerrainContacts

        if len(packet) != 6:
            raise ValueError('Platform observation requires the native six-field contact packet')
        force, points, normals, counts, starts = (loads._tensor(packet[i]) for i in (0, 1, 2, 4, 5))
        layout = NativeTerrainContacts._patch_layout(force, counts, starts, loads.capacity)
        if (counts.shape != (loads.view.sensor_count, loads.surface_count)
                or points.shape != (force.numel(), 3) or normals.shape != points.shape
                or points.dtype != torch.float32 or normals.dtype != torch.float32
                or points.device != force.device or normals.device != force.device):
            raise ValueError('Platform patch positions/directions must match the native force buffer')
        points, normals = points[layout.indices], normals[layout.indices]
        if (not torch.isfinite(points).all() or not torch.isfinite(normals).all()
                or (normals.norm(dim=-1) - 1).abs().gt(1e-3).any()):
            raise ValueError('Active platform contact points and unit normals must be finite')
        native = loads.sensor_to_native[layout.pairs // counts.shape[1]]
        bodies = loads.normal.shape[1]
        return cls(native // bodies, native % bodies, points, normals, layout.force[layout.indices])


class PlatformLandingEvidenceBuilder:
    """Tensor evidence definitions, reusable with an explicit contact producer.

    This builder consumes true native patches here. A world-model caller must
    identify its contact estimates as estimates rather than native measurements.
    Existing skill support/air-load/gap bounds are used unchanged. The pilot's
    departure definition is positive whole-wheel clearance and no body load
    above the existing 2 N airborne threshold, followed by the shared updater's
    two ticks and ascent test.
    """

    def __init__(self, geometry, wheel_body_indices, *, config=None):
        from transformer_rl.formal.tasks.skills import SkillEvidenceConfig

        self.geometry = geometry
        self.wheels = torch.as_tensor(wheel_body_indices, dtype=torch.long, device=geometry.device)
        if self.wheels.shape != (2,) or self.wheels.unique().numel() != 2:
            raise ValueError('Platform evidence needs the two distinct native wheel body indices')
        self.config = SkillEvidenceConfig() if config is None else config

    def build(self, *, patches, body_load_n, contact_valid, tile, selected, target_index,
              command_height_m, clearance=None):
        """Build one tick; optional clearance is a CylinderGap in tile coordinates.

        Without clearance, real contact attribution remains available. A quiet
        no-contact sample cannot certify flight, and complete support cannot be
        qualified until the finite-cylinder gap is available. Missing geometry
        never changes the measured wheel/non-wheel contact flags.
        """
        c, device = self.config, body_load_n.device
        n, bodies = body_load_n.shape
        if (body_load_n.dtype != torch.float32 or contact_valid.shape != (n,)
                or contact_valid.dtype != torch.bool or contact_valid.device != device
                or self.wheels.device != device or (self.wheels < 0).any()
                or (self.wheels >= bodies).any()):
            raise ValueError('Platform body loads and contact validity must follow native body order')
        for name, value, dtype in (('tile', tile, torch.long), ('target', target_index, torch.long),
                                   ('selected', selected, torch.bool), ('height', command_height_m, torch.float32)):
            if value.shape != (n,) or value.device != device or value.dtype != dtype:
                raise ValueError('Platform evidence has inconsistent ' + name + ' rows')
        valid = (contact_valid & torch.isfinite(body_load_n).all(-1)
                 & (body_load_n >= 0).all(-1) & torch.isfinite(command_height_m))
        wheel_load = body_load_n.index_select(1, self.wheels)
        wheel_touch = wheel_load > c.airborne_load_max_n
        other = torch.ones(bodies, dtype=torch.bool, device=device)
        other[self.wheels] = False
        nonwheel_contact = (body_load_n[:, other] > c.airborne_load_max_n).any(-1)
        target_normal, other_normal = body_load_n.new_zeros(n * 2), body_load_n.new_zeros(n * 2)
        support_upward, support_z = body_load_n.new_zeros(n * 2), body_load_n.new_zeros(n * 2)
        if patches is not None:
            is_wheel = patches.body[:, None] == self.wheels[None]
            keep = is_wheel.any(-1) & selected[patches.row]
            row = patches.row[keep]
            if row.numel():
                side = is_wheel[keep].long().argmax(-1)
                slot = row * 2 + side
                point = patches.position_world_m[keep]
                point_local = point - self.geometry.terrain.origins[tile[row]]
                surface = self.geometry.query(point_local, tile[row], target_index[row])
                ground = self.geometry.terrain.height_at(point_local, tile[row])
                upward = patches.normal_load_n[keep] * patches.normal_world[keep, 2].clamp_min(0)
                target_gap = point_local[:, 2] - surface.height
                target = (surface.inside & (target_gap >= -c.support_penetration_max_m)
                          & (target_gap <= c.support_gap_max_m) & (upward > 0))
                ground_gap = point_local[:, 2] - ground.height
                support = (ground.valid & (ground_gap >= -c.support_penetration_max_m)
                           & (ground_gap <= c.support_gap_max_m) & (upward > 0))
                target_normal.scatter_add_(0, slot, torch.where(target, patches.normal_load_n[keep], 0.))
                # A horizontal riser impact remains contact even beside target support.
                other_normal.scatter_add_(0, slot, torch.where(target, 0., patches.normal_load_n[keep]))
                support_upward.scatter_add_(0, slot, torch.where(support, upward, 0.))
                support_z.scatter_add_(0, slot, torch.where(support, upward * point[:, 2], 0.))
        else:
            valid = torch.zeros_like(valid)
        target_normal, other_normal = target_normal.view(n, 2), other_normal.view(n, 2)
        support_upward, support_z = support_upward.view(n, 2), support_z.view(n, 2)
        target_touch = ((target_normal > c.airborne_load_max_n)
                        & (other_normal <= c.airborne_load_max_n))
        gap_valid = torch.zeros_like(valid)
        gap = body_load_n.new_full((n, 2), torch.nan)
        if clearance is not None:
            if (clearance.vertical_gap_m.shape != (n, 2) or clearance.valid.shape != (n, 2)
                    or clearance.domain_covered.shape != (n, 2)
                    or clearance.vertical_gap_m.device != device):
                raise ValueError('Whole-cylinder clearance must match the native two-wheel rows')
            gap = clearance.vertical_gap_m
            gap_valid = (clearance.valid & clearance.domain_covered & torch.isfinite(gap)).all(-1)
        any_contact = wheel_touch.any(-1) | nonwheel_contact
        measured = valid & (gap_valid | any_contact)
        clear = valid & gap_valid & ~any_contact & (gap > 0).all(-1)
        supported = (valid & gap_valid & (support_upward >= c.grounded_load_min_n).all(-1)
                     & (gap >= -c.support_penetration_max_m).all(-1)
                     & (gap <= c.support_gap_max_m).all(-1))
        # Current, measured support may leave the selected short platform after
        # a certified both-wheel landing. No root column or remembered target Z.
        ground_world_z = (support_z / support_upward.clamp_min(torch.finfo(body_load_n.dtype).tiny)).mean(-1)
        landing_height = torch.where(supported, ground_world_z + command_height_m, torch.nan)
        return PlatformLandingEvidence(selected, measured, clear, wheel_touch, target_touch,
                                       nonwheel_contact, supported, landing_height)


class _ContactPacketView:
    """Observe the one existing backend read, leaving reduction and RNG alone."""

    def __init__(self, view, observer):
        self._view, self._observer = view, observer

    def __getattr__(self, name):
        return getattr(self._view, name)

    def get_contact_data(self, *args, **kwargs):
        packet = self._view.get_contact_data(*args, **kwargs)
        self._observer._packet = packet
        self._observer._packet_tick = self._observer.native._physical_tick
        return packet


class NativePlatformLanding:
    """Install after shared physics, before the outer first-flight latch.

    ``intent_capture`` returns PlatformIntentState or its typed record.
    ``clearance_capture`` optionally returns the current WheelClearanceSample;
    it must use actual body poses and the whole finite-cylinder footprint on
    the same atlas. No simulation, request or reward is created by this object.
    """

    def __init__(self, native, geometry, *, intent_capture, clearance_capture=None, wheel_pose_capture=None):
        self.native, self.geometry = native, geometry
        self.tasks = native._terrain_course.tasks
        self.loads = native._contact_loads
        if native._terrain_course.router.identity != geometry.terrain.manifest_sha256:
            raise ValueError('Native platform contacts and target geometry must use the same atlas')
        if (getattr(self.tasks.observe_physics, '__func__', None) is not observe_jump_physics
                or hasattr(self.tasks, 'supermimic_flight_latch')):
            raise ValueError('Install platform landing immediately after shared jump physics, before its latch')
        if not callable(intent_capture) or (clearance_capture is not None and not callable(clearance_capture)):
            raise ValueError('Platform landing needs explicit intent and optional clearance capture')
        if not all(hasattr(self.loads, name) for name in ('view', 'sensor_to_native', 'normal', '_tick')):
            raise ValueError('Native platform patch observation requires the PhysX LabContactLoads interface')
        if wheel_pose_capture is not None and not callable(wheel_pose_capture):
            raise ValueError('Platform reward wheel pose capture must be callable')
        self.intent_capture, self.clearance_capture = intent_capture, clearance_capture
        self.wheel_pose_capture = wheel_pose_capture
        self.builder = PlatformLandingEvidenceBuilder(geometry, native._wheel_body_indices)
        self.wheel_body_names = tuple(native._binding.native_body_names[index]
                                      for index in native._binding.wheel_body_indices)
        self.state = PlatformLandingState.empty(self.tasks.physical_id)
        self.last_evidence = None
        self._packet, self._packet_tick = None, None
        self._patches = []
        # No interval buffers or per-tick copies unless a collector opts in.
        self._interval = None

    def capture(self):
        return PlatformLandingState(**{field.name: getattr(self.state, field.name).clone()
                                      for field in fields(self.state)})

    def target_progress(self, bound):
        """Shape only the same request's current qualified flight/first target touch."""
        from .platform_target_reward import platform_target_progress

        if self.wheel_pose_capture is None:
            return None
        pose = self.wheel_pose_capture()
        intent = self.intent_capture()
        if not isinstance(intent, PlatformIntentState):
            intent = PlatformIntentState.decode(intent)
        return platform_target_progress(self.geometry, centres_local_m=pose.centres_local_m,
            bottoms_local_z_m=pose.bottoms_local_z_m, radii_m=pose.radii_m,
            terrain_tile=pose.terrain_tile, target_index=intent.target_index,
            requested_rise_m=self.tasks.active_rise, bound=bound, landing_phase=self.state.phase)

    def reset(self, mask):
        # Auto-reset happens before the outer collector receives the completed
        # transition. Its four pre-reset tick records must survive this reset.
        self.state = self.state.reset(mask)

    def begin_interval(self):
        """Start recording one four-tick policy step before native stepping."""
        if self._interval is not None:
            raise RuntimeError('Capture or abort the existing platform interval before beginning another')
        self._interval = []
        self._interval_start_tick = self.native._physical_tick

    def abort_interval(self):
        """Discard only an interrupted recording, without changing task state."""
        self._interval = None

    def capture_interval(self):
        """Consume the completed recording, including after native auto-reset."""
        if self._interval is None or len(self._interval) != 4:
            raise RuntimeError('Platform interval capture requires exactly four observed physics ticks')
        evidence = PlatformLandingEvidence(**{
            field.name: torch.stack([getattr(tick[0], field.name) for tick in self._interval], 1)
            for field in fields(PlatformLandingEvidence)})
        state = PlatformLandingState(**{
            field.name: torch.stack([getattr(tick[1], field.name) for tick in self._interval], 1)
            for field in fields(PlatformLandingState)})
        result = PlatformLandingInterval(evidence, state)
        result.validate(len(self.state.request_id), self.state.request_id.device)
        self._interval = None
        return result

    def _patch(self, owner, name, callback):
        self._patches.append((owner, name, name in vars(owner), vars(owner).get(name), callback))
        setattr(owner, name, callback)

    def install(self):
        if self._patches or hasattr(self.native, 'supermimic_platform_landing'):
            raise ValueError('Native platform landing is already installed')
        self._observe, self._reset = self.tasks.observe_physics, self.tasks.reset
        self._patch(self.loads, 'view', _ContactPacketView(self.loads.view, self))
        self._patch(self.tasks, 'observe_physics', self.observe)
        self._patch(self.tasks, 'reset', self._reset_tasks)
        self._patch(self.native, 'supermimic_platform_landing', self)
        return self

    def close(self):
        if any(getattr(owner, name) is not callback for owner, name, _, _, callback in self._patches):
            raise RuntimeError('Close outer native observers before platform landing')
        for owner, name, present, previous, _ in reversed(self._patches):
            if present:
                setattr(owner, name, previous)
            else:
                delattr(owner, name)
        self._patches.clear()

    def _reset_tasks(self, ids, *args, **kwargs):
        result = self._reset(ids, *args, **kwargs)
        mask = torch.zeros_like(self.tasks.physical_id, dtype=torch.bool)
        mask[ids] = True
        self.reset(mask)
        return result

    @torch.no_grad()
    def observe(self, *args, **kwargs):
        native, tick = self.native, self.native._physical_tick
        if self._interval is not None and (
                len(self._interval) >= 4 or tick != self._interval_start_tick + len(self._interval) + 1):
            raise RuntimeError('Platform recording requires one observation per consecutive physics tick')
        sample, measured_state = native._last_sample, native._last_state
        if sample is None or measured_state is None:
            raise RuntimeError('Platform physics must follow the current native substep sample')
        intent = self.intent_capture()
        if not isinstance(intent, PlatformIntentState):
            intent = PlatformIntentState.decode(intent)
        selected = (intent.bound_request_id >= 0) & (intent.bound_request_id == self.tasks.physical_id)
        fresh = self._packet_tick == tick and self.loads._tick == tick
        patches = PlatformContactPatches.from_native(self._packet, self.loads) if fresh else None
        clearance = None
        if self.clearance_capture is not None:
            captured = self.clearance_capture()
            if (captured is not None and captured.physical_tick == tick
                    and captured.reference == CLEARANCE_REFERENCE and captured.source_geometry_bound
                    and captured.body_names == self.wheel_body_names):
                clearance = captured.geometry
        self.last_evidence = self.builder.build(
            patches=patches, body_load_n=sample.body_normal_load_n,
            contact_valid=sample.contact_measurement_valid & measured_state.contact_measurement_valid,
            tile=native._terrain_course.router.tile, selected=selected, target_index=intent.target_index,
            command_height_m=native._command[:, 3], clearance=clearance)
        # Call the unique success/payment owner once, then preserve its history.
        self.state = self._observe(*args, **kwargs, platform_state=self.state,
                                   platform_evidence=self.last_evidence)
        if self._interval is not None:
            evidence = PlatformLandingEvidence(**{field.name: getattr(self.last_evidence, field.name).clone()
                                                  for field in fields(PlatformLandingEvidence)})
            self._interval.append((evidence, self.capture()))
        return self.state
