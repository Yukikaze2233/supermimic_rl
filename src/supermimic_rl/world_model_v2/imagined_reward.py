"""Shared native four-tick locomotion reward with explicit terrain/task routing.

Recorded task state advances dynamic commands and gimbal-mode eligibility.
Terrain columns follow predicted root/wheel positions. Optional jump records
advance the native physical evidence and reference before every tick's reward.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from transformer_rl.formal.lab_env.terrain_support import TerrainSupportConfig
from transformer_rl.formal.tasks.reward import RewardPhase, RewardReference, RewardTuning
from transformer_rl.formal.tasks.gimbal_course import GimbalRewardReference
from ..contracts import action_bounds_from_source
from ..reward_kernel import JumpProgress, RewardExtensions, RewardParameters
from .execution import WHEEL_ACTION_BOUND, LegExecution
from .geometry import RewardGeometry
from .records import CONTACT_BODIES, WHEEL_BODIES, decode_transforms
from .reward_assembly import RewardAssembly, RewardTick
from . import so3
from .motion_task import MotionTask
from .terrain import AtlasTerrain

if TYPE_CHECKING:
    from ..course.jump_physics import PlatformLandingEvidence, PlatformLandingState


@dataclass(frozen=True)
class LocomotionRewardContext:
    assembly: RewardAssembly
    geometry: RewardGeometry
    legs: LegExecution
    encoder_bias: torch.Tensor
    motor_strength: torch.Tensor
    ground_z: torch.Tensor
    body_contact_allowed: torch.Tensor
    gimbal_sign: torch.Tensor | None = None
    gimbal_zero: torch.Tensor | None = None
    terrain: AtlasTerrain | None = None
    terrain_tile: torch.Tensor | None = None
    terrain_kind: torch.Tensor | None = None
    task_family: torch.Tensor | None = None
    jump_task: object | None = None
    base_com_offset: torch.Tensor | None = None

    def with_conditions(self, conditions):
        return replace(self, body_contact_allowed=conditions[:, 1] > .5)

    def integrate(self, start, prediction, action, reward_memory, controller_memory, goal, step, *,
                  jump_state=None, base_mode=None,
                  platform_state: PlatformLandingState | None = None,
                  platform_evidence: PlatformLandingEvidence | None = None):
        """Integrate explicit task evidence alongside this interval's motion.

        Platform evidence comes from the rollout's prediction producer, or
        recorded truth only in a parity audit. Its ``landing_world_height``
        must share ``prediction.ticks.state.position`` coordinates (tile-local
        in the world model). No implicit translation or replay lookup occurs.
        """
        if (platform_state is None) != (platform_evidence is None):
            raise ValueError('Platform reward requires both landing history and interval evidence')
        if platform_state is not None and self.jump_task is None:
            raise ValueError('Platform reward requires the configured jump task')
        ticks = prediction.ticks
        n, device = len(action), action.device
        ids = torch.zeros(n, dtype=torch.long, device=device)
        valid = torch.ones(n, dtype=torch.bool, device=device)
        mode = goal[:, 4:9].argmax(-1)
        reference = RewardReference(
            torch.cat((goal[..., :3], goal[..., 3:4] / 5.), -1), mode,
            torch.full_like(ids, int(RewardPhase.SUPPORT)), torch.zeros_like(self.ground_z),
            torch.zeros_like(self.ground_z), torch.zeros_like(action[..., :2]),
            intentional_base_contact=self.body_contact_allowed)
        samples, heights = [], []
        special = goal[:, 12] > .5
        direction = torch.atan2(torch.where(special, goal[:, 14], 0.),
                                torch.where(special, goal[:, 15], 1.))
        geometry = self.geometry(ticks.state, passive_knee_position=ticks.passive_knee_position,
                                 passive_knee_velocity=ticks.passive_knee_velocity)
        count = ticks.leg_position.shape[-2]
        terrain_rows = torch.zeros_like(valid)
        approach = torch.zeros_like(valid)
        if self.terrain is None:
            ground = self.ground_z[:, None, None].expand(-1, count, 3)
            ground_valid = valid[:, None].expand(-1, count)
        else:
            points = torch.cat((ticks.state.position[..., None, :], geometry.centre_world), -2)
            sample = self.terrain.height_at(points, self.terrain_tile)
            ground, ground_valid = sample.height, sample.valid.all(-1)
            terrain_rows = (self.task_family == 3) & (self.terrain_kind != 0)
            approach = ((self.terrain_kind == 3) | (self.terrain_kind == 4)
                        | (self.terrain_kind == 5) | (self.terrain_kind == 7))
        scan_valid = getattr(prediction, 'tick_terrain_valid', None)
        if scan_valid is not None:
            ground_valid = ground_valid & scan_valid
        jump_interval = None
        if self.jump_task is not None:
            if jump_state is None:
                raise ValueError('Jump reward requires recorded request and physical task state')
            jump_interval = self.jump_task.observe_interval(jump_state, physical=ticks.state,
                wheel_support_load=ticks.wheel_load.clamp_min(0),
                base_com_offset=self.base_com_offset, base_mode=mode if base_mode is None else base_mode,
                command_height=goal[:, 3] / 5., ground_z=ground[:, :, 0],
                retain_peak_gradient=self.assembly.parameters.extensions.jump_peak_completion,
                **({} if platform_state is None else {
                    'platform_state': platform_state, 'platform_evidence': platform_evidence}))
        for tick in range(count):
            state = ticks.state.tick(tick)
            wheel = geometry.tick(tick)
            active = terrain_rows & ground_valid[:, tick]
            height = state.position[:, 2] - ground[:, tick, 0]
            heights.append(torch.where(active & approach, torch.maximum(height, wheel.extension_m), height))
            tick_reference = reference
            if jump_interval is not None:
                tick_reference = replace(tick_reference, phase=jump_interval.phase[:, tick],
                    jump_height_world_z_m=jump_interval.height_world_z_m[:, tick],
                    jump_velocity_world_z_m_s=jump_interval.velocity_world_z_m_s[:, tick])
            if self.gimbal_sign is not None:
                yaw = so3.yaw(state.rotation) + (state.joint_position[:, 4] - self.gimbal_zero[:, 0]) * self.gimbal_sign[:, 0]
                tick_reference = replace(tick_reference, gimbal=GimbalRewardReference(special, yaw, goal[:, 13], direction))
            elif special.any():
                raise ValueError('Gimbal-frame rewards require recorded native encoder signs and zeros')
            # Raw regressions remain unconstrained in fitting. Physical use of
            # force magnitudes must not turn a negative prediction into support
            # or a negative cost. Geometry/motion remain unclipped and auditable.
            samples.append(RewardTick(
                state, wheel, ground[:, tick, 0], ground[:, tick, 1:],
                ticks.body_normal_load[:, tick].clamp_min(0), ticks.wheel_load[:, tick].clamp_min(0),
                ticks.resultant_contact_force[:, tick].clamp_min(0), ground_valid[:, tick], terrain_rows, tick_reference,
                action.new_full((n,), (step * 4 + tick + 1) * self.assembly.parameters.physics_dt_s),
                ids, ids, valid,
                # Replays without the first-flight latch leave it None, so jump
                # flight terms fail closed instead of being inferred.
                jump_progress=None if jump_interval is None else JumpProgress(
                    jump_interval.reference_stage[:, tick], jump_interval.ascent_seen[:, tick],
                    None if jump_interval.flight_closed is None else jump_interval.flight_closed[:, tick],
                    None if jump_interval.landed_s is None else jump_interval.landed_s[:, tick],
                    None if jump_interval.flight_started is None else jump_interval.flight_started[:, tick],
                    None if jump_interval.peak_rise is None else jump_interval.peak_rise[:, tick],
                    None if jump_interval.requested_rise is None else jump_interval.requested_rise[:, tick],
                    None if jump_interval.physical_active is None else jump_interval.physical_active[:, tick])))
        result = self.assembly.integrate(start, tuple(samples), action, reward_memory,
            controller_memory.wheel_torque_history, self.legs, self.encoder_bias, self.motor_strength)
        return replace(result, termination_height=torch.stack(heights, -1)[..., None],
            jump_state=None if jump_interval is None else jump_interval.state,
            success_credited=None if jump_interval is None else jump_interval.success_credited,
            platform_state=None if jump_interval is None else jump_interval.platform_state)


class OrdinaryFlatReward:
    """Bind actual episode mass/controller parameters and pinned native tuning."""

    def __init__(self, source: dict, geometry: RewardGeometry, wheels, policy_dt_s: float, *, jump_task=None, recovery_task=None):
        config = source['reward_config']
        if source['asset_manifest_sha256'] != geometry.linkage.provenance['asset_manifest_sha256']:
            raise ValueError('Reward producer and collected data use different assets')
        if policy_dt_s != .02 or config['physics_dt_s'] != .005:
            raise ValueError('The four-tick reward requires the recorded 200/50 Hz clocks')
        # Native-collection extensions (if any) are part of the pinned source identity.
        self.extensions = RewardExtensions(**source.get('reward_kernel', {}).get('extensions', {}))
        self.tuning = RewardTuning(config['robot_mass_kg'], gamma=config['gamma'], policy_dt_s=policy_dt_s,
                                   profile=config['profile'],
                                   **({} if self.extensions.success_event is None
                                      else {'success_event': float(self.extensions.success_event)}))
        self.dt, self.terms = config['physics_dt_s'], tuple(config['available_terms'])
        self.support_config = TerrainSupportConfig(**source['terrain_support_config'])
        self.geometry, self.wheels = geometry, wheels
        self.jump_task, self.recovery_task = jump_task, recovery_task
        self.action_bounds = action_bounds_from_source(source)
        if self.action_bounds[4:] != (WHEEL_ACTION_BOUND, WHEEL_ACTION_BOUND):
            raise ValueError('Imagined wheel targets use the pinned wheel action bound')

    def _check_surface(self, start):
        if (start['reward_route'][:, :2] != 0).any():
            raise ValueError('Flat reward cannot advance a skill or nonflat terrain task')
        if 'motion_task' in start and (MotionTask.decode(start['motion_task']).route[:, [1, 3]] != 0).any():
            raise ValueError('Flat motion task contains a nonflat or skill command route')

    def bind(self, start, state, *, known_interval=False):
        """Bind an imagined start, or audit a single already-recorded interval.

        A known interval uses its recorded command for every physics tick; it
        needs no future command controller. Imagination must supply that state.
        """
        if not (start['reward_context_valid'] > .5).all():
            raise ValueError('Shared reward needs recorded native reward/support/controller memories')
        if (self.recovery_task is None and (start['goal'][:, 7] > .5).any()
                or self.jump_task is None and (start['goal'][:, 8] > .5).any()):
            raise ValueError('Locomotion reward cannot advance a jump or recovery task')
        if self.jump_task is not None and ('jump_record_valid' not in start
                                          or not (start['jump_record_valid'] > .5).all()):
            raise ValueError('Jump reward requires complete recorded native jump state')
        if self.recovery_task is not None and ('recovery_record_valid' not in start
                or not (start['recovery_record_valid'] > .5).all()):
            raise ValueError('Recovery reward requires complete recorded native lifecycle state')
        self._check_surface(start)
        if 'motion_task' not in start and not known_interval:
            if (start['horizon'][:, 1] > .5).any() or (start['reward_route'][:, 2] != 0).any():
                raise ValueError('Dynamic flat reward requires recorded command and heading state')
        wheels = [CONTACT_BODIES.index(name) for name in WHEEL_BODIES]
        parameters = RewardParameters(self.tuning, self.dt, start['sim_context'][:, :21].sum(-1), wheels,
            [i for i in range(len(CONTACT_BODIES)) if i not in wheels], CONTACT_BODIES.index('base_link'),
            self.extensions)
        assembly = RewardAssembly(parameters, self.terms, self.geometry.knee_limits, self.support_config, self.wheels)
        nominal, stiffness, damping, bounds, bias = start['controller_parameters'].split(6, -1)
        transforms = decode_transforms(start['transforms'])
        legs = LegExecution(nominal, stiffness, damping, bounds, leg_action_bound=self.action_bounds[0])
        return LocomotionRewardContext(assembly, self.geometry, legs, bias,
            start['motor_strength'], state.position[:, 2] - start['height'][:, 0],
            start['failure_conditions'][:, 1] > .5, transforms.gimbal_sign, transforms.gimbal_zero_rad,
            jump_task=self.jump_task, base_com_offset=start['sim_context'][:, 41:44])


class TerrainLocomotionReward(OrdinaryFlatReward):
    """Native ordinary-surface reward; geometry never implies a task family.

    Full terrain imagination additionally needs the deployable stair request
    producer. This class evaluates a supplied command interval and surface.
    """

    def __init__(self, source, geometry, wheels, policy_dt_s, *, terrain: AtlasTerrain, jump_task=None, recovery_task=None):
        super().__init__(source, geometry, wheels, policy_dt_s, jump_task=jump_task, recovery_task=recovery_task)
        if source['terrain_sha256'] != terrain.manifest_sha256:
            raise ValueError('Reward and native collection must use the same terrain atlas')
        self.terrain = terrain

    def _check_surface(self, start):
        modes = start['reward_route'][:, 0]
        allowed = ((modes <= 2) | ((modes == 4) & (self.jump_task is not None))
                   | ((modes == 3) & (self.recovery_task is not None)))
        if not allowed.all():
            raise ValueError('Terrain locomotion cannot advance a jump or recovery task')
        if 'terrain_tile' not in start or 'task_family' not in start:
            raise ValueError('Terrain reward needs the recorded native tile and task family')
        if 'motion_task' in start:
            base_mode = MotionTask.decode(start['motion_task']).route[:, 3]
            allowed_base = (base_mode <= 2) | ((base_mode == 3) & (self.recovery_task is not None))
            if not allowed_base.all():
                raise ValueError('Terrain motion task contains an unconfigured skill command route')

    def bind(self, start, state, *, known_interval=False):
        context = super().bind(start, state, known_interval=known_interval)
        return replace(context, terrain=self.terrain, terrain_tile=start['terrain_tile'][:, 0],
                       terrain_kind=start['reward_route'][:, 1], task_family=start['task_family'][:, 0])
