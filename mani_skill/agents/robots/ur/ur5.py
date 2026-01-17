import numpy as np
import sapien
from copy import deepcopy

from mani_skill import ASSET_DIR, PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent


@register_agent()
class UR5(BaseAgent):
    uid = "ur5"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/ur/ur5.urdf"
    urdf_config = dict()

    keyframes = dict(
        rest=Keyframe(
            pose=sapien.Pose(p=[0, 0, 0]),
            qpos=np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0]),
        )
    )

    arm_joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    ee_link_name = "wrist_3_link"

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    @property
    def _controller_configs(
        self,
    ):
        return dict(
            pd_joint_pos=PDJointPosControllerConfig(
                self.arm_joint_names,
                lower=None,
                upper=None,
                stiffness=self.arm_stiffness,
                damping=self.arm_damping,
                force_limit=self.arm_force_limit,
                normalize_action=False,
            ),
            pd_joint_delta_pos=PDJointPosControllerConfig(
                self.arm_joint_names,
                lower=-0.1,
                upper=0.1,
                stiffness=self.arm_stiffness,
                damping=self.arm_damping,
                force_limit=self.arm_force_limit,
                use_delta=True,
            ),
            pd_ee_delta_pos=PDEEPosControllerConfig(
                joint_names=self.arm_joint_names,
                pos_lower=-0.1,
                pos_upper=0.1,
                stiffness=self.arm_stiffness,
                damping=self.arm_damping,
                force_limit=self.arm_force_limit,
                ee_link=self.ee_link_name,
                urdf_path=self.urdf_path,
            ),
            pd_ee_delta_pose=PDEEPoseControllerConfig(
                joint_names=self.arm_joint_names,
                pos_lower=-0.1,
                pos_upper=0.1,
                rot_lower=-0.1,
                rot_upper=0.1,
                stiffness=self.arm_stiffness,
                damping=self.arm_damping,
                force_limit=self.arm_force_limit,
                ee_link=self.ee_link_name,
                urdf_path=self.urdf_path,
            ),
            pd_ee_pose=PDEEPoseControllerConfig(
                joint_names=self.arm_joint_names,
                pos_lower=None,
                pos_upper=None,
                rot_lower=None,
                rot_upper=None,
                stiffness=self.arm_stiffness,
                damping=self.arm_damping,
                force_limit=self.arm_force_limit,
                ee_link=self.ee_link_name,
                urdf_path=self.urdf_path,
                use_delta=False,
                normalize_action=False,
            ),
            pd_joint_vel=PDJointVelControllerConfig(
                self.arm_joint_names,
                lower=-1.0,
                upper=1.0,
                damping=self.arm_damping,
                force_limit=self.arm_force_limit,
            ),
            pd_joint_pos_vel=PDJointPosVelControllerConfig(
                self.arm_joint_names,
                lower=None,
                upper=None,
                stiffness=self.arm_stiffness,
                damping=self.arm_damping,
                force_limit=self.arm_force_limit,
                normalize_action=False,
            ),
            pd_joint_delta_pos_vel=PDJointPosVelControllerConfig(
                self.arm_joint_names,
                lower=-0.1,
                upper=0.1,
                stiffness=self.arm_stiffness,
                damping=self.arm_damping,
                force_limit=self.arm_force_limit,
                normalize_action=True,
                use_delta=True,
            ),
        )
