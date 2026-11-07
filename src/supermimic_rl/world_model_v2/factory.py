"""Construct the V2 learner from pinned native provenance and explicit network sizes."""
import json

from ..contracts import action_bounds_from_source
from ..chassis_observation import LEGACY_COMMANDS
from .execution import WheelExecution
from .features import MotionInputs
from .geometry import RewardGeometry
from .imagination import Critic
from .learning import Learner
from .leg_linkage import LegLinkage
from .model import SIM_CONTEXT_DIM, WorldModelV2
from .policy import EstimatorActor
from .termination import PoseContactLimits, PoseContactTermination
from .motion_task import MOTION_TASK_DIM


def make_learner(schema, runtime, config, *, linkage_path, asset_directory, identified_bundle, device,
                 model_hidden=(2048, 2048, 2048), actor_hidden=(512, 256, 128),
                 estimator_hidden=(128, 64), critic_hidden=(256, 128), record_tasks=False,
                 terrain=None, stair_task=None, stair_source=None, jump_task=None, recovery_task=None, actor_frame_dim=45,
                 chassis_command_contract=LEGACY_COMMANDS, actor_features=None,
                 platform_task=None, platform_contact=None, platform_geometry=None,
                 platform_contact_hidden=None):
    # Native task dependencies become available after the pinned course factory
    # has installed its upstream path; importing the package alone needs no Isaac.
    from .imagined_reward import OrdinaryFlatReward, TerrainLocomotionReward

    source = json.loads(schema.source)
    from ..course.randomization import RequestedRandomizationConfig

    disturbances = any(field.name == 'velocity_push_world' for field in schema.substep_fields)
    disturbance_config = (RequestedRandomizationConfig(**source['randomization']['configuration'])
                          if disturbances else None)
    dt = schema.policy_dt_s / schema.substeps_per_step
    linkage = LegLinkage.load(linkage_path, expected_asset_manifest_sha256=source['asset_manifest_sha256'])
    model = WorldModelV2(MotionInputs(linkage), WheelExecution.from_bundle(identified_bundle, dt),
                         hidden=tuple(model_hidden), physics_dt_s=dt, terrain=terrain,
                         disturbances=disturbances).to(device)
    actor = EstimatorActor(estimator_hidden=tuple(estimator_hidden), actor_hidden=tuple(actor_hidden),
                           frame_dim=actor_frame_dim, action_bounds=action_bounds_from_source(source),
                           chassis_command_contract=chassis_command_contract,
                           actor_features=actor_features).to(device)
    actor.features.validate_reference(None if jump_task is None else jump_task.config.reference)
    task_dim = MOTION_TASK_DIM if record_tasks else 0
    routes = {'terrain': terrain is not None, 'stairs': stair_source is not None, 'jumps': jump_task is not None,
              'recovery': recovery_task is not None,
              'disturbances': disturbances,
              'platforms': platform_task is not None,
              'jump_trigger': jump_task is not None and jump_task.config.trigger is not None}
    critic = Critic(Critic.feature_size(schema.goal_dim, SIM_CONTEXT_DIM, task_dim=task_dim, **routes),
                    tuple(critic_hidden), task_dim=task_dim, input_clip=config.critic_input_clip, **routes).to(device)
    reward_type = OrdinaryFlatReward if terrain is None else TerrainLocomotionReward
    geometry = RewardGeometry(linkage, asset_directory)
    if platform_contact_hidden is not None:
        from ..course.atlas_cylinder_gap import AtlasCylinderTerrainGap
        from .platform_contacts import PlatformContactModel

        if (platform_contact is not None or platform_task is None or platform_geometry is None
                or platform_geometry.terrain is not terrain):
            raise ValueError('Contact head construction requires one shared platform task and atlas')
        clearance = AtlasCylinderTerrainGap(terrain, tuple(float(s.radius) for s in geometry.sides),
            tuple(2 * float(s.half_length) for s in geometry.sides),
            tile_ids=tuple(int(tile) for tile in platform_geometry.tiles))
        platform_contact = PlatformContactModel(geometry, platform_geometry, clearance,
                                               hidden=tuple(platform_contact_hidden)).to(device)
    reward = reward_type(source, geometry, model.wheels, schema.policy_dt_s,
                         jump_task=jump_task, recovery_task=recovery_task, **({} if terrain is None else {'terrain': terrain}))
    termination = PoseContactTermination(PoseContactLimits.from_runtime(runtime), dt)
    return Learner(model, actor, critic, termination, config, reward=reward,
                    stair_task=stair_task, stair_source=stair_source, jump_task=jump_task, recovery_task=recovery_task,
                    disturbance_config=disturbance_config, platform_task=platform_task,
                    platform_contact=platform_contact, platform_geometry=platform_geometry)
