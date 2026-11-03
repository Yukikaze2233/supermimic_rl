"""Check the externally selected Transformer source without starting Isaac.

Run with the intended upstream checkout on PYTHONPATH (or installed in the
active interpreter). This script does not change import paths or source files.
"""

import torch

from transformer_rl.formal.lab_env import skill_fsm


def main():
    commands = torch.tensor([[0.2, 0.0, 0.1, 0.32], [-0.1, 0.0, -0.2, 0.28]])
    fsm = skill_fsm.SkillFSM(2, device="cpu")
    zeros = torch.zeros(2)

    def advance():
        return fsm.step(
            step_height_m=zeros,
            wheel_force_n=torch.full((2, 2), 100.0),
            body_height_m=torch.full((2,), 0.32),
            tilt_rad=zeros,
            base_contact_force_n=zeros,
            command_height_m=commands[:, 3],
            auto_enabled=torch.ones(2, dtype=torch.bool),
            jump_request=torch.zeros(2, dtype=torch.bool),
            jump_request_height_m=zeros,
            vertical_velocity_m_s=zeros,
        )

    advance()
    # Match WheeledLabEnv: publish the new command before resetting its FSM.
    commands[0, 3] = 0.30
    expected = commands.clone()
    fsm.reset(torch.tensor([0]))
    if not torch.equal(commands, expected):
        raise AssertionError(
            "Partial FSM reset mutated caller-owned commands: "
            f"expected={expected.tolist()}, actual={commands.tolist()}"
        )
    result = advance()
    if not torch.equal(result.height_command_m, expected[:, 3]):
        raise AssertionError("The next normal decision lost its positive height command")
    if fsm.active.any() or result.mode.ne(0).any() or not torch.equal(commands, expected):
        raise AssertionError("Flat normal-mode reset changed command ownership or mode")
    print(f"PASS: partial reset preserves command heights; source={skill_fsm.__file__}")


if __name__ == "__main__":
    main()
