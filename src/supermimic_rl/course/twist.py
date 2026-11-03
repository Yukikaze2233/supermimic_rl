"""Root COM twist correction for the IsaacLab-Newton (Kamino) backend.

``robot.data.root_com_vel_w`` binds the floating-base joint velocity, whose
linear part is the root link ORIGIN velocity, while ``body_com_vel_w`` holds
true COM twists. Recorded Kamino substeps confirm it: link-origin steps match
``dt * root_com_vel_w`` to ~1e-6 m and root-COM steps match ``dt *
body_com_vel_w[base]`` to ~1e-8 m. The pinned sample treats the former as a
COM velocity and subtracts ``omega x c`` once more, biasing every measured
body velocity (reward, estimator labels, vertical-velocity task logic) by
``omega x c``. This layer substitutes the root body's actual COM twist.
"""
from __future__ import annotations

from dataclasses import replace

RECEIPT = {
    "root_velocity_source": "body_com_vel_w[base_link] (Newton COM twist)",
    "replaced": "root_com_vel_w linear part, which is the floating-base origin velocity on Newton",
    "unchanged": "PhysX, whose root_com_vel_w is the COM velocity; direct reads outside _sample",
}


def make_root_twist_corrected_environment(base_type):
    class RootTwistCorrectedEnvironment(base_type):
        def _sample(self):
            sample = super()._sample()
            if not self._is_newton:
                return sample
            com = self._as_torch(self.robot.data.body_com_vel_w)[:, self._base_body_index, :3]
            return replace(sample, root_com_velocity_world_m_s=com.clone())

    return RootTwistCorrectedEnvironment
