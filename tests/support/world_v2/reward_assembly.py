"""Native-equivalent reward assembly inputs on the pinned asset."""

import json
import torch
from transformer_rl.formal.control.observation import MechanicalBinding
from transformer_rl.formal.lab_env.formal_reward import LAB_AVAILABLE_TERMS
from transformer_rl.formal.lab_env.terrain_support import TerrainSupportConfig
from transformer_rl.formal.tasks.reward import PhysicalRewardComposer, RewardTuning
from supermimic_rl.reward_kernel import RewardParameters
from supermimic_rl.world_model_v2.execution import LegExecution, WheelExecution
from supermimic_rl.world_model_v2.geometry import RewardGeometry
from supermimic_rl.world_model_v2.leg_linkage import LegLinkage
from supermimic_rl.world_model_v2.physical import PhysicalState
from supermimic_rl.world_model_v2.reward_assembly import RewardAssembly, RewardMemory
from supermimic_rl.world_model_v2.support import SupportState
from support import REPO_ROOT


ROOT = REPO_ROOT


def setup(profile):
    asset = ROOT / 'assets/full_vehicle_v2p0p1_source_audited_v2'
    manifest = json.loads((asset / 'manifest.json').read_text())
    binding = MechanicalBinding.from_manifest(manifest,
        manifest['tree_joint_names'] + manifest['auxiliary_joint_names'],
        manifest['rigid_body_names'] + manifest['auxiliary_body_names'])
    native = PhysicalRewardComposer(binding, 2, RewardTuning(manifest['total_mass_kg'], policy_dt_s=.02, profile=profile))
    ids, mask = torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.bool)
    native.reset(mask, torch.zeros(2, dtype=torch.float64), ids, ids)
    wheels = WheelExecution.from_bundle(ROOT / 'configs/identified/20261031', .005)
    geometry = RewardGeometry(LegLinkage.load(ROOT / 'configs/linkage/leg_linkage_e726d23c.json'), asset)
    parameters = RewardParameters.from_composer(native)
    assembly = RewardAssembly(parameters, LAB_AVAILABLE_TERMS + ('joint_boundary',), geometry.knee_limits,
                              TerrainSupportConfig(profile=profile), wheels)
    q = torch.zeros(2, 6)
    for side in geometry.sides:
        table = geometry.linkage.sides[side.name]
        q[:, side.primary] = table.reference_primary_rad
        q[:, side.secondary] = table.reference_primary_rad + (table.phi_min + table.phi_max) / 2
    state = PhysicalState(torch.tensor([[0., 0., .35], [0., 0., .3]]), torch.eye(3).expand(2, 3, 3),
                          torch.zeros(2, 3), torch.zeros(2, 3), q, torch.zeros(2, 8))
    legs = LegExecution(q, torch.full((2, 6), 160.), torch.full((2, 6), 2.5), torch.full((2, 6), 40.))
    memory = RewardMemory(torch.zeros(2, 6), torch.zeros(2, 6), torch.zeros(2), SupportState.decode(torch.zeros(2, 10)))
    return native, assembly, geometry, state, legs, memory
