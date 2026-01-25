from typing import Union

import torch
import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill import ASSET_DIR, PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, DictControllerConfig, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.controllers.base_controller import ControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor

@register_agent(asset_download_ids=["robotiq_2f"])
class FloatingRobotiq2F85Gripper(BaseAgent):
    uid = "floating_robotiq_2f_85_gripper"
    urdf_path = f"{ASSET_DIR}/robots/robotiq_2f/floating_robotiq_2f_85.urdf"
    disable_self_collisions = True
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            left_inner_finger_pad=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            right_inner_finger_pad=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )
    keyframes = dict(
        rest=Keyframe(
            qpos=[0.5, 0.0, 0.5, np.pi, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            pose=sapien.Pose(p=np.array([0.0, 0.0, 0.5]), q=euler2quat(np.pi, 0, 0)),
        ),
        open_facing_down=Keyframe(
            qpos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            pose=sapien.Pose(p=np.array([0.0, 0.0, 0.5]), q=euler2quat(np.pi, 0, 0)),
        ),
        open_facing_up=Keyframe(
            qpos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            pose=sapien.Pose(p=np.array([0.0, 0.0, 0.5])),
        ),
        open_facing_side=Keyframe(
            qpos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            pose=sapien.Pose(
                p=np.array([0.0, 0.0, 0.5]), q=np.array([0.7071, 0, 0.7071, 0])
            ),
        ),
    )
    root_joint_names = [
        "root_x_axis_joint",
        "root_y_axis_joint",
        "root_z_axis_joint",
        "root_x_rot_joint",
        "root_y_rot_joint",
        "root_z_rot_joint",
    ]
    ee_link_name = "eef"

    @property
    def _controller_configs(
        self,
    ) -> dict[str, Union[ControllerConfig, DictControllerConfig]]:

        # define a simple controller to control the floating base with XYZ/RPY control.
        base_pd_joint_pos = PDJointPosControllerConfig(
            joint_names=self.root_joint_names,
            lower=None,
            upper=None,
            stiffness=1e3,
            damping=1e2,
            force_limit=100,
            normalize_action=False,
        )
        base_pd_joint_delta_pos = PDJointPosControllerConfig(
            joint_names=self.root_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=1e3,
            damping=1e2,
            force_limit=100,
            use_delta=True,
        )
        base_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.root_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=1e3,
            damping=1e2,
            force_limit=100,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )

        # define a passive controller config to simply "turn off" other joints from being controlled and set their properties (damping/friction) to 0.
        # these joints are controlled passively by the mimic controller later on.
        passive_finger_joint_names = [
            "left_inner_knuckle_joint",
            "right_inner_knuckle_joint",
            "left_inner_finger_joint",
            "right_inner_finger_joint",
        ]
        passive_finger_joints = PassiveControllerConfig(
            joint_names=passive_finger_joint_names,
            damping=0,
            friction=0,
        )

        finger_joint_names = ["left_outer_knuckle_joint", "right_outer_knuckle_joint"]
        # use a mimic controller config to define one action to control both fingers
        mimic_config = dict(
            left_outer_knuckle_joint=dict(joint="right_outer_knuckle_joint", multiplier=1.0, offset=0.0),
        )
        finger_mimic_pd_joint_pos = PDJointPosMimicControllerConfig(
            joint_names=finger_joint_names,
            lower=None,
            upper=None,
            stiffness=1e5,
            damping=1e3,
            force_limit=0.1,
            friction=0.05,
            normalize_action=False,
            mimic=mimic_config,
        )
        finger_mimic_pd_joint_delta_pos = PDJointPosMimicControllerConfig(
            joint_names=finger_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=1e5,
            damping=1e3,
            force_limit=0.1,
            normalize_action=True,
            friction=0.05,
            use_delta=True,
            mimic=mimic_config,
        )
        return dict(
            pd_joint_pos=dict(
                arm=base_pd_joint_pos,
                gripper_active=finger_mimic_pd_joint_pos,
                gripper_passive=passive_finger_joints,
            ),
            pd_joint_delta_pos=dict(
                arm=base_pd_joint_delta_pos,
                gripper_active=finger_mimic_pd_joint_delta_pos,
                gripper_passive=passive_finger_joints,
            ),
            pd_ee_delta_pose=dict(
                arm=base_pd_ee_delta_pose,
                gripper_active=finger_mimic_pd_joint_pos,
                gripper_passive=passive_finger_joints,
            ),
        )

    def _after_loading_articulation(self):
        outer_finger = self.robot.active_joints_map["right_inner_finger_joint"]
        inner_knuckle = self.robot.active_joints_map["right_inner_knuckle_joint"]
        pad = outer_finger.get_child_link()
        lif = inner_knuckle.get_child_link()

        # the next 4 magic arrays come from https://github.com/haosulab/cvpr-tutorial-2022/blob/master/debug/robotiq.py which was
        # used to precompute these poses for drive creation
        p_f_right = [-1.6048949e-08, 3.7600022e-02, 4.3000020e-02]
        p_p_right = [1.3578170e-09, -1.7901104e-02, 6.5159947e-03]
        p_f_left = [-1.8080145e-08, 3.7600014e-02, 4.2999994e-02]
        p_p_left = [-1.4041154e-08, -1.7901093e-02, 6.5159872e-03]

        right_drive = self.scene.create_drive(
            lif, sapien.Pose(p_f_right), pad, sapien.Pose(p_p_right)
        )
        right_drive.set_limit_x(0, 0)
        right_drive.set_limit_y(0, 0)
        right_drive.set_limit_z(0, 0)

        outer_finger = self.robot.active_joints_map["left_inner_finger_joint"]
        inner_knuckle = self.robot.active_joints_map["left_inner_knuckle_joint"]
        pad = outer_finger.get_child_link()
        lif = inner_knuckle.get_child_link()

        left_drive = self.scene.create_drive(
            lif, sapien.Pose(p_f_left), pad, sapien.Pose(p_p_left)
        )
        left_drive.set_limit_x(0, 0)
        left_drive.set_limit_y(0, 0)
        left_drive.set_limit_z(0, 0)

    def _after_init(self):
        self.finger1_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "left_inner_finger_pad"
        )
        self.finger2_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "right_inner_finger_pad"
        )
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if the robot is grasping an object

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
        """
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # direction to open the gripper
        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-2]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """Build a grasp pose (panda_hand_tcp)."""
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)

@register_agent()
class FloatingRobotiq2F85GripperWristCamera(FloatingRobotiq2F85Gripper):
    uid = "floating_robotiq_2f_85_gripper_wristcam"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/robotiq_2f/floating_robotiq_2f_85_wristcam.urdf"

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="hand_camera",
                pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
                # width=256,
                # height=256,
                width=224,
                height=224,
                # fov=np.pi * 155 / 180,
                fov=np.pi * 120 / 180,
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link"],
            )
        ]
