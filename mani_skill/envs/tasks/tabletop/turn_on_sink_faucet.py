from typing import Any, Optional, Union

import numpy as np
import sapien
import sapien.render
import torch
from transforms3d.euler import euler2quat

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
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.geometry.geometry import transform_points
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.robocasa.fixtures.fixture import FixtureType
from mani_skill.utils.scene_builder.robocasa.scene_builder import (
    ROBOT_FRONT_FACING_SIZE,
    RoboCasaSceneBuilder,
)
from mani_skill.utils.scene_builder.robocasa.utils import object_utils as rc_object_utils
from mani_skill.utils.structs import Link, Pose
from mani_skill.utils.structs.types import Array

ROBOT_FRONT_FACING_SIZE.update(
    {
        "panda_robotiq_wristcam": 0.3,
        "ur5_robotiq_wristcam": 0.3,
        "xarm6_robotiq_wristcam": 0.3,
        "xarm7_robotiq_wristcam": 0.3,
        "floating_robotiq_2f_85_gripper_wristcam": 0.3,
        "iiwa7_robotiq_wristcam": 0.3,
        "gen3_6dof_robotiq_wristcam": 0.3,
        "gen3_7dof_robotiq_wristcam": 0.3,
        "rizon4_robotiq_wristcam": 0.3,
        "sawyer_robotiq_wristcam": 0.3,
    }
)


def create_colored_material(color, metallic=0.0, roughness=0.5, specular=0.5):
    mat = sapien.render.RenderMaterial()
    mat.base_color = color
    mat.metallic = metallic
    mat.roughness = roughness
    mat.specular = specular
    return mat


@register_env("TurnOnSinkFaucet-v1", max_episode_steps=120, asset_download_ids=["ycb"])
class TurnOnSinkFaucetEnv(BaseEnv):
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
    SUPPORTED_REWARD_MODES = ["sparse", "dense", "normalized_dense", "none"]
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

    harder_robot_obstacle_contact_force_thresh = 1e-2

    def __init__(
        self,
        *args,
        robot_uids="panda_robotiq_wristcam",
        robot_init_qpos_noise=0.05,
        harder: bool = False,
        reconfiguration_freq=None,
        num_envs=1,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.harder = harder
        self.layout_idx = 0
        self.style_idx = 0
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
        # base_pose = sapien_utils.look_at(
        #     eye=[1.42, -1.65, 1.52], target=[1.24, -0.42, 0.96]
        # )
        # return [
        #     *self.agent._sensor_configs,
        #     CameraConfig(
        #         "base_camera",
        #         pose=base_pose,
        #         width=128,
        #         height=128,
        #         fov=np.pi / 2,
        #         near=0.01,
        #         far=100,
        #     ),
        # ]
        return self.agent._sensor_configs

    @property
    def _default_human_render_camera_configs(self):
        # pose = sapien_utils.look_at(
        #     eye=[1.46, -1.60, 1.50], target=[1.24, -0.42, 0.96]
        # )
        pose = sapien_utils.look_at(
            eye=[2.06, -0.50, 1.20], target=[1.24, -0.42, 0.96]
        )
        # pose = sapien_utils.look_at(
        #     eye=[0.66, -0.50, 1.20], target=[1.24, -0.42, 0.96]
        # )
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1.0, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        ps = torch.zeros((self.num_envs, 3), device=self.device)
        super()._load_agent(options, Pose.create_from_pq(p=ps))

    def _initialize_robot(self, env_idx: torch.Tensor):
        b = len(env_idx)
        qpos = self.agent.keyframes["rest"].qpos
        # if isinstance(rest_qpos, torch.Tensor):
        #     qpos = rest_qpos.clone().to(self.device).repeat(b, 1)
        # else:
        #     qpos = torch.tensor(rest_qpos, dtype=torch.float32, device=self.device).repeat(
        #         b, 1
        #     )
        qpos = (
            self._episode_rng.normal(
                0, self.robot_init_qpos_noise, (b, len(qpos))
            )
            + qpos
        )
        self.agent.reset(qpos)
        # robot_p = torch.tensor(
        #     [[1.18, -0.80, 0.6]], dtype=torch.float32, device=self.device
        # ).repeat(b, 1)
        if self.robot_uids in ["sawyer_robotiq_wristcam"]:
            robot_p = torch.tensor(
                [[1.35, -0.90, 0.6]], dtype=torch.float32, device=self.device
            ).repeat(b, 1)
        elif self.robot_uids in ["iiwa7_robotiq_wristcam"]:
            robot_p = torch.tensor(
                [[1.3, -0.90, 0.6]], dtype=torch.float32, device=self.device
            ).repeat(b, 1)
        else:
            robot_p = torch.tensor(
                [[1.3, -0.80, 0.6]], dtype=torch.float32, device=self.device
            ).repeat(b, 1)
        robot_q = torch.tensor([[np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]], dtype=torch.float32, device=self.device).repeat(b, 1)
        self.agent.robot.set_pose(Pose.create_from_pq(p=robot_p, q=robot_q))

    def _cache_sink_scene_geometry(self):
        self.sink_fixtures = []
        self.harder_anchor_positions = []
        self.harder_anchor_quats = []
        self.handle_link_local_poss = []

        for scene_idx in range(self.num_envs):
            fixtures = self.scene_builder.scene_data[scene_idx]["fixtures"]
            sink = self.scene_builder.get_fixture(fixtures, FixtureType.SINK)
            self.sink_fixtures.append(sink)
            sink_quat = np.array(euler2quat(0, 0, sink.rot), dtype=np.float32)
            handle_mesh = sink.articulation.links_map["handle"].generate_mesh(
                filter=lambda *_: True, mesh_name="handle"
            )[0]
            self.handle_link_local_poss.append(
                handle_mesh.bounding_box.center_mass.astype(np.float32)
            )

            int_p0, int_px, int_py, int_pz = sink.get_int_sites(relative=True)
            ix_min, ix_max = int_p0[0], int_px[0]
            iy_min, iy_max = int_p0[1], int_py[1]
            iz_top = int_pz[2]
            harder_anchor_local = np.array(
                [(ix_min + ix_max) / 2, (iy_min + iy_max) / 2, iz_top], dtype=np.float32
            )
            harder_anchor_world = rc_object_utils.get_pos_after_rel_offset(
                sink, harder_anchor_local
            )
            self.harder_anchor_positions.append(harder_anchor_world.astype(np.float32))
            self.harder_anchor_quats.append(sink_quat)

    def _build_harder_obstacles(self):
        self.extra_obstacles = []
        self.extra_obstacle_half_sizes = []
        # ycb_ids = ["006_mustard_bottle", "004_sugar_box"]
        ycb_ids = ["004_sugar_box"]
        for i, model_id in enumerate(ycb_ids):
            builder = actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")
            builder.initial_pose = sapien.Pose(p=[0, 0, -10])
            self.extra_obstacles.append(builder.build_kinematic(name=f"harder_obj_{i}"))
            self.extra_obstacle_half_sizes.append(None)

    def _load_scene(self, options: dict):
        self.scene_builder = RoboCasaSceneBuilder(
            self, init_robot_base_pos=FixtureType.SINK
        )
        build_config_idx = self.layout_idx * 12 + self.style_idx
        self.scene_builder.build(build_config_idxs=[build_config_idx] * self.num_envs)
        self._cache_sink_scene_geometry()
        self.faucet = self.sink_fixtures[0].articulation
        self.target_link = self.faucet.links_map["handle"]
        self.target_joint = next(
            joint for joint in self.faucet.joints if joint.name == "handle_joint"
        )
        self.handle_link_local_pos = torch.tensor(
            np.stack(self.handle_link_local_poss),
            dtype=torch.float32,
            device=self.device,
        )

        self.handle_goal = actors.build_sphere(
            self.scene,
            radius=0.012,
            color=[0, 1, 0, 1],
            name="faucet_handle_goal",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.handle_goal)

        qlimits = self.target_joint.get_limits()
        qmin, qmax = qlimits[:, 0], qlimits[:, 1]
        self.closed_qpos = torch.where(torch.isfinite(qmin), qmin, torch.zeros_like(qmin))
        self.success_qpos = torch.clamp(
            torch.full_like(qmin, 0.40),
            min=self.closed_qpos,
            max=qmax - 1e-3,
        )
        self.motionplan_target_qpos = torch.clamp(
            self.closed_qpos + torch.full_like(qmin, 0.60),
            max=qmax - 1e-3,
        )

        self._build_harder_obstacles()

    def _after_reconfigure(self, options):
        for i, actor in enumerate(self.extra_obstacles):
            collision_mesh = actor.get_first_collision_mesh()
            # if collision_mesh is None:
            #     self.extra_obstacle_half_sizes[i] = [0.03, 0.03, 0.05]
            # else:
            #     self.extra_obstacle_half_sizes[i] = (
            #         collision_mesh.bounding_box.extents / 2
            #     ).tolist()
            self.extra_obstacle_half_sizes[i] = (
                collision_mesh.bounding_box.extents / 2
            ).tolist()

    def target_handle_positions(self, env_idx: Optional[torch.Tensor] = None):
        if env_idx is None:
            return transform_points(
                self.target_link.pose.to_transformation_matrix().clone(),
                self.handle_link_local_pos,
            )
        return transform_points(
            self.target_link.pose[env_idx].to_transformation_matrix().clone(),
            self.handle_link_local_pos[env_idx],
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
        self.handle_dir = self.target_link.pose.to_transformation_matrix()[:, :3, 1]
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.scene_builder.initialize(env_idx)
            self._initialize_robot(env_idx)
            if not hasattr(self, "episode_failed"):
                self.episode_failed = torch.zeros(
                    self.num_envs, device=self.device, dtype=torch.bool
                )
            self.episode_failed[env_idx] = False

            qpos = self.faucet.get_qpos()
            qpos[env_idx, 0] = 0.3
            qpos[env_idx, 1] = 0.0
            qpos[env_idx, 2] = 0.0
            self.faucet.set_qpos(qpos)
            self.faucet.set_qvel(torch.zeros_like(self.faucet.qvel))

            hidden = torch.zeros((b, 3), dtype=torch.float32, device=self.device)
            hidden[:, 2] = -10.0
            if self.harder:
                anchor_p = torch.tensor(
                    np.stack(
                        [self.harder_anchor_positions[i] for i in env_idx.cpu().tolist()]
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )
                anchor_q = torch.tensor(
                    np.stack(
                        [self.harder_anchor_quats[i] for i in env_idx.cpu().tolist()]
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )
                sink_rot_mats = Pose.create_from_pq(
                    p=torch.zeros_like(anchor_p), q=anchor_q
                ).to_transformation_matrix()[:, :3, :3]

                def _offset(local_vec):
                    local = torch.tensor(local_vec, dtype=torch.float32, device=self.device)
                    return torch.einsum("bij,j->bi", sink_rot_mats, local)

                box_half = torch.tensor(
                    self.extra_obstacle_half_sizes[0], dtype=torch.float32, device=self.device
                )
                # bottle_half = torch.tensor(
                #     self.extra_obstacle_half_sizes[1], dtype=torch.float32, device=self.device
                # )

                box_p = torch.zeros((b, 3), dtype=torch.float32, device=self.device)
                # Place close to the faucet to block direct approach trajectories.
                # box_p[:] = anchor_p + _offset([-0.12, 0.18, 0.0])
                box_p[:] = anchor_p + _offset([-0.20, -0.24, 0.0]) + torch.rand(b, 3, device=self.device) * torch.tensor([-0.05, 0.1, 0.0], device=self.device)
                box_p[:, 2] = anchor_p[:, 2] + box_half[2] + 0.01

                # bottle_p = torch.zeros((b, 3), dtype=torch.float32, device=self.device)
                # # Place on the left countertop side of the sink.
                # # bottle_p[:] = anchor_p + _offset([-0.08, 0.12, 0.0])
                # bottle_p[:] = anchor_p + _offset([-0.04, 0.25, 0.0]) + torch.rand(b, 3, device=self.device) * torch.tensor([-0.03, 0.0, 0.0], device=self.device)
                # bottle_p[:, 2] = anchor_p[:, 2] + bottle_half[2] + 0.01

                box_q = anchor_q.clone()
                # bottle_q = anchor_q.clone()
                self.extra_obstacles[0].set_pose(Pose.create_from_pq(box_p, box_q))
                # self.extra_obstacles[1].set_pose(Pose.create_from_pq(bottle_p, bottle_q))
            else:
                for actor in self.extra_obstacles:
                    actor.set_pose(Pose.create_from_pq(hidden))

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
        qpos = self.faucet.get_qpos()
        idx = self.target_joint.active_index.to(torch.long)
        rows = torch.arange(self.num_envs, device=self.device)
        return qpos[rows, idx]

    def get_faucet_kinematics(self):
        self._update_task_kinematics()
        return dict(
            handle_pos=self.handle_pos,
            hinge_pos=self.hinge_pos,
            hinge_axis=self.hinge_axis,
            handle_dir=self.handle_dir,
        )

    def get_obstacle_actors(self):
        return list(self.extra_obstacles)

    def get_obstacles_info(self):
        obstacles_info = []
        if self.harder:
            robot_root_inv = self.agent.robot.get_pose().inv()
            for i, actor in enumerate(self.extra_obstacles):
                raw_pose = (robot_root_inv * actor.pose).raw_pose
                extent = self.extra_obstacle_half_sizes[i]
                if extent is None:
                    continue
                obstacles_info.append(
                    {
                        "center": raw_pose[:, :3],
                        "quat": raw_pose[:, 3:],
                        "extent": torch.tensor(
                            extent, dtype=torch.float32, device=raw_pose.device
                        ).expand(raw_pose.shape[0], 3),
                    }
                )
        return obstacles_info
    
    def _compute_robot_harder_obstacle_collision(self):
        fail = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        max_contact_force = torch.zeros(self.num_envs, device=self.device)
        if not self.harder:
            return fail, max_contact_force

        for obstacle in self.extra_obstacles:
            for link in self.agent.robot.get_links():
                contact_force = torch.linalg.norm(
                    self.scene.get_pairwise_contact_forces(link, obstacle), axis=1
                )
                max_contact_force = torch.maximum(max_contact_force, contact_force)

        fail = max_contact_force > self.harder_robot_obstacle_contact_force_thresh
        return fail, max_contact_force

    def evaluate(self):
        self._update_task_kinematics()
        faucet_joint_pos = self.current_angle
        faucet_joint_pos_wrapped = torch.remainder(faucet_joint_pos, 2 * np.pi)
        faucet_on = torch.logical_and(
            faucet_joint_pos_wrapped > self.success_qpos,
            faucet_joint_pos_wrapped < torch.full_like(faucet_joint_pos_wrapped, np.pi),
        )
        current_fail, robot_obstacle_contact_force = (
            self._compute_robot_harder_obstacle_collision()
        )
        self.episode_failed = torch.logical_or(self.episode_failed, current_fail)
        fail = self.episode_failed
        success = torch.logical_and(faucet_on, ~fail)
        return {
            "success": success,
            "fail": fail,
            "current_fail": current_fail,
            "faucet_on": faucet_on,
            "faucet_joint_pos": faucet_joint_pos,
            "target_on_angle": self.success_qpos,
            "handle_pos": self.handle_pos,
        }

    def _get_obs_extra(self, info: dict):
        tcp_pose_world_frame = self.agent.tcp_pose
        tcp_pose_root_frame = self.agent.robot.get_pose().inv() * tcp_pose_world_frame
        obs = dict(
            tcp_pose=tcp_pose_root_frame.raw_pose,
            tcp_pose_world_frame=tcp_pose_world_frame.raw_pose,
            faucet_joint_pos=info["faucet_joint_pos"],
            target_on_angle=info["target_on_angle"],
            handle_pos=info["handle_pos"],
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                faucet_pose=self.faucet.pose.raw_pose,
                hinge_pos=self.hinge_pos,
                hinge_axis=self.hinge_axis,
                handle_dir=self.handle_dir,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        tcp_to_handle_dist = torch.linalg.norm(
            self.agent.tcp.pose.p - info["handle_pos"], axis=1
        )
        reaching_reward = 1 - torch.tanh(6.0 * tcp_to_handle_dist)

        open_progress = (info["faucet_joint_pos"] - self.closed_qpos) / (
            self.success_qpos - self.closed_qpos + 1e-6
        )
        open_progress = torch.clamp(open_progress, 0.0, 1.0)
        reward = reaching_reward + 2.5 * open_progress
        reward[info["success"]] = 4.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 4.0
