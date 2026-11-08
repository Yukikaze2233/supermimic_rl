"""Native reward fixtures shared by composer and telemetry tests."""
from dataclasses import replace

import torch

from transformer_rl.formal.control.observation import MechanicalBinding, CommandMode
from transformer_rl.formal.tasks.reward import (
    PhysicalRewardComposer, RewardPhysics, RewardReference, RewardTuning, RewardPhase,
    RoughDashRewardReference, TerrainRewardReference,
)
from transformer_rl.formal.tasks.gimbal_course import GimbalRewardReference


def setup(case):
    joints = ('L_joint1', 'LL_joint1', 'R_joint1', 'RR_joint1', 'L_joint3', 'R_joint3')
    binding = MechanicalBinding(joints, ('base_link', 'L_link3', 'R_link3', 'M_link1', 'M_link2'),
                                joints, tuple(range(6)), tuple(range(6)), (1, 2), (4, 5), (0.,) * 6)
    tuning = RewardTuning(21.44, policy_dt_s=.02, profile='rough_dash' if case.startswith('air') else 'legacy')
    composer = PhysicalRewardComposer(binding, 2, tuning)
    composer.reset(torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.float64),
                   torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long))
    mode = CommandMode.RECOVERY if case == 'recovery' else CommandMode.JUMP if case.startswith('jump') else CommandMode.NORMAL
    phase = {'jump_takeoff': RewardPhase.TAKEOFF, 'jump_flight': RewardPhase.FLIGHT,
             'jump_landing': RewardPhase.LANDING, 'recovery': RewardPhase.RECOVERY}.get(case, RewardPhase.SUPPORT)
    reference = RewardReference(torch.tensor([[0., 0., 0., .32], [.5, 0., -.5, .3]]),
                                torch.full((2,), int(mode)), torch.full((2,), int(phase)),
                                torch.tensor([.45, .38]), torch.tensor([.2, -.1]), torch.zeros(2, 2))
    if case == 'gimbal':
        reference = replace(reference, gimbal=GimbalRewardReference(
            torch.tensor([True, False]), torch.tensor([.7, -.4]), torch.tensor([.5, 0.]), torch.tensor([1., 0.])))
    if case == 'terrain':
        reference = replace(reference, terrain=TerrainRewardReference(
            torch.tensor([True, False]), torch.tensor([.31, .29]), torch.tensor([True, False]),
            torch.zeros(2, dtype=torch.bool)), intentional_base_contact=torch.tensor([True, False]))
    if case.startswith('air'):
        reference = replace(reference, rough_dash=RoughDashRewardReference(
            torch.tensor([True, False]), torch.tensor([.32, .3]),
            torch.full((2, 2), .03 if case == 'air_landing' else 0.), torch.zeros(2)))
    return composer, reference


def begin(composer, reference, action):
    composer.begin(action, composer._last_dq, now_s=composer._time,
                   episode_id=composer._episode, control_epoch=composer._control,
                   policy_owned=torch.ones(2, dtype=torch.bool),
                   admitted_body_command=reference.admitted_body_command, command_mode=reference.mode)


def sample(composer, tick):
    generator = torch.Generator().manual_seed(tick)
    rand = lambda *shape: torch.rand(*shape, generator=generator)
    window = composer._window
    gravity = torch.tensor([[.15, .1, -1.], [-.2, -.1, -1.]])
    gravity /= gravity.norm(dim=-1, keepdim=True)
    return RewardPhysics(
        window['start'] + tick * .005, window['episode'], window['control'], window['owned'],
        rand(2, 3), rand(2, 3), rand(2) + .3, rand(2) * .1 + .26,
        rand(2, 2), gravity, rand(2, 3), rand(2, 6), rand(2, 6),
        torch.tensor([[4., 150., 200., 0., 0.], [0., 30., 40., 4., 0.]]),
        rand(2, 2), rand(2, 6) * .02,
        wheel_position_base_m=torch.tensor([[[.1, .2, -.26], [0., -.2, -.24]]]).repeat(2, 1, 1),
        wheel_forward_velocity_rad_s=rand(2, 2) * 12, wheel_contact_force_n=rand(2, 2) * 300,
        passive_knee_position_rad=rand(2, 2), passive_knee_velocity_rad_s=rand(2, 2) * 15,
        passive_knee_limits_rad=torch.tensor([[0., 1.], [0., 1.]]))


def finish(composer, failed=False):
    return composer.finish(terminated=torch.tensor([failed, False]), truncated=torch.zeros(2, dtype=torch.bool),
                           physical_success_request_id=torch.full((2,), -1),
                           entire_skill_policy_owned=torch.ones(2, dtype=torch.bool))
