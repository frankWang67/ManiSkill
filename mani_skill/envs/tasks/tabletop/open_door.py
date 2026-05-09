from typing import Any, Optional, Union

import numpy as np
import sapien
import sapien.render
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import (
    FloatingRobotiq2F85GripperWristCamera,
    PandaRobotiqWristCamera,
    UR5RobotiqWristCamera,
    XArm6RobotiqWristCamera,
    XArm7RobotiqWristCamera,
    IIwa7RobotiqWristCamera,
    Gen36DofRobotiqWristCamera,
    Gen37DofRobotiqWristCamera,
    Rizon4RobotiqWristCamera,
    SawyerRobotiqWristCamera,
)
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, io_utils, sapien_utils
from mani_skill.utils.building import actors, articulations
from mani_skill.utils.geometry.geometry import transform_points
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Articulation, Link, Pose
from mani_skill.utils.structs.types import Array


def create_colored_material(color):
    mat = sapien.render.RenderMaterial()
    mat.base_color = color
    mat.metallic = 0.0
    mat.roughness = 0.5
    mat.specular = 0.5
    return mat


@register_env(
    "OpenDoor-v1",
    max_episode_steps=120,
    asset_download_ids=["partnet_mobility_cabinet"],
)
class OpenDoorEnv(BaseEnv):
    """
    **Task Description:**
    Open a cabinet door mounted on a tabletop.

    **Difficulty Levels:**
    - default: only the cabinet door articulation.
    - harder: one horizontal kinematic platform is placed between the robot and
      the door handle to block direct opening trajectories.

    **Success Condition:**
    - door revolute joint angle exceeds a task threshold.
    """

    SUPPORTED_ROBOTS = [
        "panda_robotiq_wristcam",
        "ur5_robotiq_wristcam",
        "xarm6_robotiq_wristcam",
        "xarm7_robotiq_wristcam",
        "floating_robotiq_2f_85_gripper_wristcam",
        "iiwa7_robotiq_wristcam",
        "gen3_6dof_robotiq_wristcam",
        "gen3_7dof_robotiq_wristcam",
        "rizon4_robotiq_wristcam",
        "sawyer_robotiq_wristcam",
    ]
    agent: Union[
        PandaRobotiqWristCamera,
        UR5RobotiqWristCamera,
        XArm6RobotiqWristCamera,
        XArm7RobotiqWristCamera,
        FloatingRobotiq2F85GripperWristCamera,
        IIwa7RobotiqWristCamera,
        Gen36DofRobotiqWristCamera,
        Gen37DofRobotiqWristCamera,
        Rizon4RobotiqWristCamera,
        SawyerRobotiqWristCamera,
    ]

    TRAIN_JSON = PACKAGE_ASSET_DIR / "partnet_mobility/meta/info_cabinet_door_train.json"
    handle_types = ["revolute", "revolute_unwrapped"]

    def __init__(
        self,
        *args,
        robot_uids="panda_robotiq_wristcam",
        robot_init_qpos_noise=0.02,
        harder: bool = False,
        model_id: Optional[str] = None,
        reconfiguration_freq=None,
        num_envs=1,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.harder = harder
        self.model_id = str(model_id) if model_id is not None else None

        train_data = io_utils.load_json(self.TRAIN_JSON)
        # model 1006 is a compact cabinet door that is feasible for tabletop opening
        default_model = "1006"
        if self.model_id is not None:
            if self.model_id not in train_data:
                raise ValueError(
                    f"model_id={self.model_id} not found in {self.TRAIN_JSON}"
                )
            self.all_model_ids = np.array([self.model_id])
        else:
            if default_model in train_data:
                self.all_model_ids = np.array([default_model])
            else:
                self.all_model_ids = np.array(list(train_data.keys()))

        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0

        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

    @property
    def _default_sensor_configs(self):
        return self.agent._sensor_configs

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[-0.95, 1.05, 0.85], target=[0.02, 0.0, 0.16])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1.0, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.522, 0, 0]))

    def _build_obstacles(self):
        wall_mat = create_colored_material([0.85, 0.15, 0.15, 1.0])
        self.obstacle_half_sizes = [
            [0.12, 0.14, 0.25],
        ]
        self.obstacles = []
        for i, hs in enumerate(self.obstacle_half_sizes):
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=hs)
            builder.add_box_visual(half_size=hs, material=wall_mat)
            builder.initial_pose = sapien.Pose(p=[0, 0, -10])
            self.obstacles.append(builder.build_kinematic(name=f"obstacle_wall_{i}"))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        model_ids = self._batched_episode_rng.choice(self.all_model_ids)
        link_ids = self._batched_episode_rng.randint(0, 2**31)
        self._cabinets: list[Articulation] = []
        target_links: list[Link] = []
        target_handle_local_pos = []
        self.selected_model_ids = []

        for i, model_id in enumerate(model_ids):
            self.selected_model_ids.append(str(model_id))
            builder = articulations.get_articulation_builder(
                self.scene, f"partnet-mobility:{model_id}"
            )
            builder.set_scene_idxs(scene_idxs=[i])
            builder.initial_pose = sapien.Pose()
            cabinet = builder.build(name=f"{model_id}-{i}")
            self.remove_from_state_dict_registry(cabinet)
            for joint in cabinet.active_joints:
                joint.set_friction(0.1)
                joint.set_drive_properties(0.0, 0.2)
            self._cabinets.append(cabinet)

            links = []
            local_mesh_centers = []
            for link, joint in zip(cabinet.links, cabinet.joints):
                if joint.type[0] in self.handle_types:
                    links.append(link)
                    handle_meshes = link.generate_mesh(
                        filter=lambda _, render_shape: "handle"
                        in render_shape.name.lower(),
                        mesh_name="handle",
                    )
                    mesh = (
                        handle_meshes[0]
                        if len(handle_meshes) > 0 and handle_meshes[0] is not None
                        else link.generate_mesh(
                            filter=lambda *_: True, mesh_name="link_visual"
                        )[0]
                    )
                    local_mesh_centers.append(mesh.bounding_box.center_mass)

            if len(links) == 0:
                raise RuntimeError(
                    f"Model {model_id} has no revolute door links for OpenDoor-v1."
                )
            idx = link_ids[i] % len(links)
            target_links.append(links[idx])
            target_handle_local_pos.append(local_mesh_centers[idx])

        self.cabinet = Articulation.merge(self._cabinets, name="cabinet")
        self.add_to_state_dict_registry(self.cabinet)
        self.target_link = Link.merge(target_links, name="target_link")
        self.target_joint = self.target_link.joint
        self.handle_link_local_pos = common.to_tensor(
            np.array(target_handle_local_pos), device=self.device
        )

        self.handle_goal = actors.build_sphere(
            self.scene,
            radius=0.012,
            color=[0, 1, 0, 1],
            name="handle_goal",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.handle_goal)

        qlimits = self.target_joint.get_limits()
        qmin, qmax = qlimits[:, 0], qlimits[:, 1]
        qmin = torch.where(torch.isfinite(qmin), qmin, torch.zeros_like(qmin))
        qmax = torch.where(torch.isfinite(qmax), qmax, qmin + np.pi / 2)
        open_range = torch.clamp(qmax - qmin, min=0.2)
        self.closed_qpos = qmin

        deg10 = torch.full_like(qmin, np.deg2rad(10))
        deg30 = torch.full_like(qmin, np.deg2rad(30))
        deg40 = torch.full_like(qmin, np.deg2rad(40))
        deg55 = torch.full_like(qmin, np.deg2rad(55))
        deg70 = torch.full_like(qmin, np.deg2rad(70))
        deg80 = torch.full_like(qmin, np.deg2rad(80))
        deg8 = torch.full_like(qmin, np.deg2rad(8))
        success = torch.minimum(qmin + open_range * 0.50, qmin + deg30)
        success = torch.maximum(success, qmin + deg10)
        success = torch.minimum(success, qmax - 1e-3)
        self.success_qpos = success

        motionplan = torch.minimum(qmin + open_range * 0.70, qmin + deg55)
        motionplan = torch.maximum(motionplan, self.success_qpos + deg8)
        motionplan = torch.minimum(motionplan, qmax - 1e-3)
        self.motionplan_target_qpos = motionplan

        self._build_obstacles()

    def _after_reconfigure(self, options):
        self.cabinet_zs = []
        for cabinet in self._cabinets:
            collision_mesh = cabinet.get_first_collision_mesh()
            if collision_mesh is None:
                self.cabinet_zs.append(0.0)
            else:
                self.cabinet_zs.append(-collision_mesh.bounding_box.bounds[0, 2] + 1e-3)
        self.cabinet_zs = common.to_tensor(self.cabinet_zs, device=self.device)

    def target_handle_positions(self, env_idx: Optional[torch.Tensor] = None):
        if env_idx is None:
            return transform_points(
                self.target_link.pose.to_transformation_matrix().clone(),
                common.to_tensor(self.handle_link_local_pos, device=self.device),
            )
        return transform_points(
            self.target_link.pose[env_idx].to_transformation_matrix().clone(),
            common.to_tensor(self.handle_link_local_pos[env_idx], device=self.device),
        )

    def _update_task_kinematics(self):
        if self.gpu_sim_enabled:
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        self.handle_pos = self.target_handle_positions()
        self.handle_goal.set_pose(Pose.create_from_pq(p=self.handle_pos))
        joint_pose = self.target_joint.get_global_pose().to_transformation_matrix()
        self.hinge_pos = joint_pose[:, :3, 3]
        self.hinge_axis = joint_pose[:, :3, 0]
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            p = torch.zeros((b, 3), device=self.device)
            p[:, 0] = torch.rand((b), device=self.device) * 0.06 + 0.15
            p[:, 1] = (torch.rand((b), device=self.device) - 0.5) * 0.10 - 0.4
            p[:, 2] = self.cabinet_zs[env_idx]
            q = randomization.random_quaternions(
                n=b, lock_x=True, lock_y=True, bounds=(-torch.pi * 80 / 180, -torch.pi * 60 / 180)
            )
            self.cabinet.set_pose(Pose.create_from_pq(p, q))

            qlimits = self.cabinet.get_qlimits()
            qpos = qlimits[env_idx, :, 0].clone()
            target_joint_idx = self.target_joint.active_index[env_idx].to(torch.long)
            qpos[
                torch.arange(b, device=self.device), target_joint_idx
            ] += torch.rand((b), device=self.device) * 0.01
            self.cabinet.set_qpos(qpos)
            self.cabinet.set_qvel(self.cabinet.qpos[env_idx] * 0)

            hidden = torch.zeros((b, 3), device=self.device)
            hidden[:, 2] = -10
            if self.harder:
                self._update_task_kinematics()
                handle = self.handle_pos[env_idx]
                robot_pos = self.agent.robot.pose.p[env_idx]

                # Place a horizontal platform on the straight robot->handle corridor.
                robot_to_handle = handle - robot_pos
                robot_to_handle_dir = robot_to_handle / torch.linalg.norm(robot_to_handle, dim=1, keepdim=True)
                ortho_dir = torch.cross(robot_to_handle_dir, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(robot_to_handle_dir))
                wall_center = robot_pos * 0.60 + handle * 0.40 + ortho_dir * 0.4
                wall_center[:, 2] = self.obstacle_half_sizes[0][2] + 1e-3

                wall_q = torch.zeros((b, 4), device=self.device, dtype=torch.float32)
                wall_q[:, 0] = 1.0
                self.obstacles[0].set_pose(Pose.create_from_pq(wall_center, wall_q))
            else:
                for obs in self.obstacles:
                    obs.set_pose(Pose.create_from_pq(hidden))

            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene.px.step()
                self.scene._gpu_fetch_all()

            self._update_task_kinematics()

    def _after_control_step(self):
        self._update_task_kinematics()

    @property
    def current_angle(self):
        qpos = self.cabinet.get_qpos()
        idx = self.target_joint.active_index.to(torch.long)
        rows = torch.arange(self.num_envs, device=self.device)
        return qpos[rows, idx]

    def get_door_kinematics(self):
        self._update_task_kinematics()
        return dict(
            handle_pos=self.handle_pos,
            hinge_pos=self.hinge_pos,
            hinge_axis=self.hinge_axis,
        )

    def get_obstacles_info(self):
        obstacles_info = []
        if not self.harder:
            return obstacles_info
        robot_root_inv = self.agent.robot.get_pose().inv()
        for i, part in enumerate(self.obstacles):
            raw_pose = (robot_root_inv * part.pose).raw_pose
            obs_info = {
                "center": raw_pose[:, :3],
                "quat": raw_pose[:, 3:],
                "extent": torch.tensor(
                    self.obstacle_half_sizes[i],
                    dtype=torch.float32,
                    device=raw_pose.device,
                ).expand(raw_pose.shape[0], 3),
            }
            obstacles_info.append(obs_info)
        return obstacles_info

    def evaluate(self):
        self._update_task_kinematics()
        door_joint_pos = self.current_angle
        open_enough = door_joint_pos >= self.success_qpos
        return {
            "success": open_enough,
            "open_enough": open_enough,
            "door_joint_pos": door_joint_pos,
            "target_open_angle": self.success_qpos,
            "handle_pos": self.handle_pos,
        }

    def _get_obs_extra(self, info: dict):
        tcp_pose_world_frame = self.agent.tcp_pose
        tcp_pose_root_frame = self.agent.robot.get_pose().inv() * tcp_pose_world_frame
        obs = dict(
            tcp_pose=tcp_pose_root_frame.raw_pose,
            tcp_pose_world_frame=tcp_pose_world_frame.raw_pose,
            door_joint_pos=info["door_joint_pos"],
            target_open_angle=info["target_open_angle"],
            handle_pos=info["handle_pos"],
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                cabinet_pose=self.cabinet.pose.raw_pose,
                hinge_pos=self.hinge_pos,
                hinge_axis=self.hinge_axis,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        tcp_to_handle_dist = torch.linalg.norm(
            self.agent.tcp.pose.p - info["handle_pos"], axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_handle_dist)

        open_progress = (info["door_joint_pos"] - self.closed_qpos) / (
            self.success_qpos - self.closed_qpos + 1e-6
        )
        open_progress = torch.clamp(open_progress, 0.0, 1.0)
        reward = reaching_reward + 2.0 * open_progress
        reward[info["success"]] = 4.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 4.0
