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
                 chassis_command_contract=LEGACY_COMMANDS):
    # Native task dependencies become available after the pinned course factory
    # has installed its upstream path; importing the package alone needs no Isaac.
    from .imagined_reward import OrdinaryFlatReward, TerrainLocomotionReward

    source = json.loads(schema.source)
    dt = schema.policy_dt_s / schema.substeps_per_step
    linkage = LegLinkage.load(linkage_path, expected_asset_manifest_sha256=source['asset_manifest_sha256'])
    model = WorldModelV2(MotionInputs(linkage), WheelExecution.from_bundle(identified_bundle, dt),
                         hidden=tuple(model_hidden), physics_dt_s=dt, terrain=terrain).to(device)
    actor = EstimatorActor(estimator_hidden=tuple(estimator_hidden), actor_hidden=tuple(actor_hidden),
                           frame_dim=actor_frame_dim, action_bounds=action_bounds_from_source(source),
                           chassis_command_contract=chassis_command_contract).to(device)
    task_dim = MOTION_TASK_DIM if record_tasks else 0
    routes = {'terrain': terrain is not None, 'stairs': stair_source is not None, 'jumps': jump_task is not None,
              'recovery': recovery_task is not None,
              'jump_trigger': jump_task is not None and jump_task.config.trigger is not None}
    critic = Critic(Critic.feature_size(schema.goal_dim, SIM_CONTEXT_DIM, task_dim=task_dim, **routes),
                    tuple(critic_hidden), task_dim=task_dim, input_clip=config.critic_input_clip, **routes).to(device)
    reward_type = OrdinaryFlatReward if terrain is None else TerrainLocomotionReward
    reward = reward_type(source, RewardGeometry(linkage, asset_directory), model.wheels, schema.policy_dt_s,
                         jump_task=jump_task, recovery_task=recovery_task, **({} if terrain is None else {'terrain': terrain}))
    termination = PoseContactTermination(PoseContactLimits.from_runtime(runtime), dt)
    return Learner(model, actor, critic, termination, config, reward=reward,
                    stair_task=stair_task, stair_source=stair_source, jump_task=jump_task, recovery_task=recovery_task)
