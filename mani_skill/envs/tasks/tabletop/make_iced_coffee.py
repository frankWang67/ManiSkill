from typing import Any, Union

import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.agents.robots import (
    FloatingRobotiq2F85GripperWristCamera,
    PandaRobotiqWristCamera,
    UR5RobotiqWristCamera,
    XArm6RobotiqWristCamera,
    XArm7RobotiqWristCamera,
)
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array


def create_colored_material(color):
    mat = sapien.render.RenderMaterial()
    mat.base_color = color
    mat.metallic = 0.0
    mat.roughness = 0.5
    mat.specular = 0.4
    return mat


@register_env("MakeIcedCoffee-v1", max_episode_steps=80, asset_download_ids=["ycb"])
class MakeIcedCoffeeEnv(BaseEnv):
    """
    **Task Description:**
    Pick one ice cube from a dish and place it into a cup.

    **Difficulty Levels:**
    - default: no obstacles beyond dish/cup.
    - harder: sampled combinations of tall transfer blockers, table objects, and overhead blockers.

    **Success Conditions:**
    - the ice cube is inside the cup.
    - the robot is no longer grasping the ice cube.
    """

    SUPPORTED_ROBOTS = [
        "panda_robotiq_wristcam",
        "ur5_robotiq_wristcam",
        "xarm6_robotiq_wristcam",
        "xarm7_robotiq_wristcam",
        "floating_robotiq_2f_85_gripper_wristcam",
    ]
    agent: Union[
        PandaRobotiqWristCamera,
        UR5RobotiqWristCamera,
        XArm6RobotiqWristCamera,
        XArm7RobotiqWristCamera,
        FloatingRobotiq2F85GripperWristCamera,
    ]

    dish_model_id = "024_bowl"
    cup_model_id = "025_mug"
    harder_table_obstacle_model_ids = [
        "006_mustard_bottle",
        "003_cracker_box",
        "021_bleach_cleanser",
    ]

    def __init__(
        self,
        *args,
        robot_uids="panda_robotiq_wristcam",
        robot_init_qpos_noise=0.02,
        harder: bool = False,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.harder = harder
        self.ice_cube_half_size = 0.012
        self.dish_inner_radius = 0.06
        self.dish_wall_thickness = 0.004
        self.dish_inner_height = 0.018
        self.dish_bottom_thickness = 0.004
        self.cup_inner_radius = 0.033
        self.cup_wall_thickness = 0.004
        self.cup_inner_height = 0.085
        self.cup_bottom_thickness = 0.004
        self.harder_transfer_half_size = np.array([0.028, 0.11, 0.18], dtype=np.float32)
        self.overhead_half_size = np.array([0.09, 0.08, 0.015], dtype=np.float32)
        self.overhead_bottom_z = 0.145

        ycb_meta = load_json(ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json")
        self._model_meta = dict()
        for model_id in [
            self.dish_model_id,
            self.cup_model_id,
            *self.harder_table_obstacle_model_ids,
        ]:
            info = ycb_meta[model_id]
            scale = float(info.get("scales", [1.0])[0])
            bbox_min = np.array(info["bbox"]["min"], dtype=np.float32) * scale
            bbox_max = np.array(info["bbox"]["max"], dtype=np.float32) * scale
            half_extents = (bbox_max - bbox_min) / 2.0
            self._model_meta[model_id] = dict(
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                half_extents=half_extents,
                bottom_z=float(-bbox_min[2]),
            )

        dish_meta = self._model_meta[self.dish_model_id]
        self.dish_spawn_radius = float(min(dish_meta["half_extents"][0], dish_meta["half_extents"][1]) * 0.35)
        self.dish_spawn_local_z = float(self.dish_bottom_thickness + self.ice_cube_half_size + 0.002)
        self.cup_success_radius = float(self.cup_inner_radius - self.ice_cube_half_size * 0.35)
        self.cup_success_z_min = float(self.cup_bottom_thickness + self.ice_cube_half_size * 0.2)
        self.cup_success_z_max = float(self.cup_bottom_thickness + self.cup_inner_height - self.ice_cube_half_size * 0.25)
        self.cup_drop_local_z = float(self.cup_bottom_thickness + self.cup_inner_height * 0.65)

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        return self.agent._sensor_configs

    @property
    def _default_human_render_camera_configs(self):
        # pose = sapien_utils.look_at(eye=[-0.95, 0.95, 0.78], target=[0.1, 0.02, 0.08])
        pose = sapien_utils.look_at(eye=[0.7, 0.7, 0.7], target=[0.1, 0.02, 0.08])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.522, 0, 0]))

    def _build_ycb_actor(
        self,
        model_id: str,
        name: str,
        body_type: str,
        add_collision: bool = True,
        add_visual: bool = True,
    ):
        builder = actors.get_actor_builder(
            self.scene, id=f"ycb:{model_id}", add_collision=add_collision, add_visual=add_visual
        )
        builder.initial_pose = sapien.Pose(p=[0, 0, -10])
        if body_type == "kinematic":
            return builder.build_kinematic(name=name)
        if body_type == "dynamic":
            return builder.build(name=name)
        raise ValueError(f"Unsupported body_type={body_type}")

    def _build_receptacle(
        self,
        name: str,
        inner_radius: float,
        wall_thickness: float,
        inner_height: float,
        bottom_thickness: float,
        color=None,
        add_visual: bool = True,
    ):
        builder = self.scene.create_actor_builder()
        outer_r = inner_radius + wall_thickness
        wall_center_z = bottom_thickness + inner_height / 2

        bottom_half = [outer_r, outer_r, bottom_thickness / 2]
        wall_x_half = [wall_thickness / 2, outer_r, inner_height / 2]
        wall_y_half = [inner_radius, wall_thickness / 2, inner_height / 2]

        builder.add_box_collision(
            pose=sapien.Pose([0, 0, bottom_thickness / 2]),
            half_size=bottom_half,
        )
        builder.add_box_collision(
            pose=sapien.Pose([inner_radius + wall_thickness / 2, 0, wall_center_z]),
            half_size=wall_x_half,
        )
        builder.add_box_collision(
            pose=sapien.Pose([-inner_radius - wall_thickness / 2, 0, wall_center_z]),
            half_size=wall_x_half,
        )
        builder.add_box_collision(
            pose=sapien.Pose([0, inner_radius + wall_thickness / 2, wall_center_z]),
            half_size=wall_y_half,
        )
        builder.add_box_collision(
            pose=sapien.Pose([0, -inner_radius - wall_thickness / 2, wall_center_z]),
            half_size=wall_y_half,
        )

        if add_visual:
            if color is None:
                color = [0.4, 0.4, 0.4, 0.2]
            mat = create_colored_material(color)
            builder.add_box_visual(
                pose=sapien.Pose([0, 0, bottom_thickness / 2]),
                half_size=bottom_half,
                material=mat,
            )
            builder.add_box_visual(
                pose=sapien.Pose([inner_radius + wall_thickness / 2, 0, wall_center_z]),
                half_size=wall_x_half,
                material=mat,
            )
            builder.add_box_visual(
                pose=sapien.Pose([-inner_radius - wall_thickness / 2, 0, wall_center_z]),
                half_size=wall_x_half,
                material=mat,
            )
            builder.add_box_visual(
                pose=sapien.Pose([0, inner_radius + wall_thickness / 2, wall_center_z]),
                half_size=wall_y_half,
                material=mat,
            )
            builder.add_box_visual(
                pose=sapien.Pose([0, -inner_radius - wall_thickness / 2, wall_center_z]),
                half_size=wall_y_half,
                material=mat,
            )
        builder.initial_pose = sapien.Pose([0, 0, -10])
        return builder.build_kinematic(name=name)

    def _build_box_obstacle(
        self,
        name: str,
        half_size,
        color=None,
        add_visual: bool = True,
    ):
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=half_size)
        if add_visual:
            if color is None:
                color = [0.5, 0.5, 0.5, 1.0]
            builder.add_box_visual(
                half_size=half_size, material=create_colored_material(color)
            )
        builder.initial_pose = sapien.Pose([0, 0, -10])
        return builder.build_kinematic(name=name)

    def _yaw_to_quat(self, yaw: torch.Tensor):
        q = torch.zeros((yaw.shape[0], 4), device=yaw.device, dtype=yaw.dtype)
        q[:, 0] = torch.cos(yaw / 2)
        q[:, 3] = torch.sin(yaw / 2)
        return q

    def _set_box_obstacle_pose(
        self,
        actor,
        center_xy: torch.Tensor,
        half_size: np.ndarray,
        yaw: torch.Tensor,
        active_mask: torch.Tensor,
        hidden_pos: torch.Tensor,
        center_z: float = None,
    ):
        b = center_xy.shape[0]
        pos = hidden_pos.clone()
        quat = torch.zeros((b, 4), device=center_xy.device, dtype=center_xy.dtype)
        quat[:, 0] = 1.0
        if torch.any(active_mask):
            posed = torch.zeros((b, 3), device=center_xy.device, dtype=center_xy.dtype)
            posed[:, :2] = center_xy
            posed[:, 2] = float(half_size[2] if center_z is None else center_z)
            pos = torch.where(active_mask[:, None], posed, pos)
            target_quat = self._yaw_to_quat(yaw)
            quat = torch.where(active_mask[:, None], target_quat, quat)
        actor.set_pose(Pose.create_from_pq(pos, quat))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.dish_visual = self._build_ycb_actor(
            self.dish_model_id,
            "ice_dish_visual",
            body_type="kinematic",
            add_collision=False,
            add_visual=True,
        )
        self.cup_visual = self._build_ycb_actor(
            self.cup_model_id,
            "coffee_cup_visual",
            body_type="kinematic",
            add_collision=False,
            add_visual=True,
        )
        self.dish = self._build_receptacle(
            name="ice_dish",
            inner_radius=self.dish_inner_radius,
            wall_thickness=self.dish_wall_thickness,
            inner_height=self.dish_inner_height,
            bottom_thickness=self.dish_bottom_thickness,
            add_visual=False,
        )
        self.cup = self._build_receptacle(
            name="coffee_cup",
            inner_radius=self.cup_inner_radius,
            wall_thickness=self.cup_wall_thickness,
            inner_height=self.cup_inner_height,
            bottom_thickness=self.cup_bottom_thickness,
            add_visual=False,
        )

        self.harder_transfer_obstacles = [
            self._build_box_obstacle(
                "harder_transfer_obstacle_0",
                half_size=self.harder_transfer_half_size,
                color=[0.42, 0.42, 0.48, 1.0],
                add_visual=True,
            ),
            self._build_box_obstacle(
                "harder_transfer_obstacle_1",
                half_size=self.harder_transfer_half_size,
                color=[0.42, 0.42, 0.48, 1.0],
                add_visual=True,
            ),
        ]
        self.harder_overhead_obstacles = {
            "dish": self._build_box_obstacle(
                "harder_overhead_dish",
                half_size=self.overhead_half_size,
                color=[0.58, 0.56, 0.52, 1.0],
                add_visual=True,
            ),
            "cup": self._build_box_obstacle(
                "harder_overhead_cup",
                half_size=self.overhead_half_size,
                color=[0.58, 0.56, 0.52, 1.0],
                add_visual=True,
            ),
        }

        self.harder_table_obstacles = []
        for i, model_id in enumerate(self.harder_table_obstacle_model_ids):
            obstacle = self._build_ycb_actor(
                model_id, f"harder_table_obstacle_{i}", body_type="kinematic"
            )
            self.harder_table_obstacles.append(obstacle)

        self.ice_cube = actors.build_cube(
            self.scene,
            half_size=self.ice_cube_half_size,
            color=[0.7, 0.92, 1.0, 1.0],
            name="ice_cube",
            initial_pose=sapien.Pose(p=[0, 0, self.ice_cube_half_size]),
        )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=0.012,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            dish_base_pos = torch.zeros((b, 3), device=self.device)
            dish_base_pos[:, 0] = torch.rand((b), device=self.device) * 0.04
            dish_base_pos[:, 1] = (torch.rand((b), device=self.device) - 0.5) * 0.12 - 0.18
            dish_base_pos[:, 2] = 0
            dish_q = randomization.random_quaternions(
                n=b, device=self.device, lock_x=True, lock_y=True
            )
            self.dish.set_pose(Pose.create_from_pq(dish_base_pos, dish_q))
            dish_visual_pos = dish_base_pos.clone()
            dish_visual_pos[:, 2] += self._model_meta[self.dish_model_id]["bottom_z"]
            self.dish_visual.set_pose(Pose.create_from_pq(dish_visual_pos, dish_q))

            cup_offset = torch.zeros((b, 2), device=self.device)
            cup_offset[:, 0] = torch.rand((b), device=self.device) * 0.04 + 0.03
            cup_offset[:, 1] = torch.rand((b), device=self.device) * 0.10 + 0.32
            cup_base_pos = torch.zeros((b, 3), device=self.device)
            cup_base_pos[:, :2] = dish_base_pos[:, :2] + cup_offset
            cup_base_pos[:, 0] = torch.clamp(cup_base_pos[:, 0], min=0.02, max=0.22)
            cup_base_pos[:, 1] = torch.clamp(cup_base_pos[:, 1], min=0.10, max=0.32)
            cup_base_pos[:, 2] = 0
            cup_q = randomization.random_quaternions(
                n=b, device=self.device, lock_x=True, lock_y=True
            )
            self.cup.set_pose(Pose.create_from_pq(cup_base_pos, cup_q))
            cup_visual_pos = cup_base_pos.clone()
            cup_visual_pos[:, 2] += self._model_meta[self.cup_model_id]["bottom_z"]
            self.cup_visual.set_pose(Pose.create_from_pq(cup_visual_pos, cup_q))

            r = torch.sqrt(torch.rand((b), device=self.device)) * self.dish_spawn_radius
            theta = torch.rand((b), device=self.device) * (2 * np.pi)
            local_ice_pos = torch.zeros((b, 3), device=self.device)
            local_ice_pos[:, 0] = r * torch.cos(theta)
            local_ice_pos[:, 1] = r * torch.sin(theta)
            local_ice_pos[:, 2] = self.dish_spawn_local_z
            ice_pos = rotation_conversions.quaternion_apply(dish_q, local_ice_pos) + dish_base_pos
            ice_q = randomization.random_quaternions(
                n=b, device=self.device, lock_x=True, lock_y=True
            )
            self.ice_cube.set_pose(Pose.create_from_pq(ice_pos, ice_q))

            local_goal = torch.zeros((b, 3), device=self.device)
            local_goal[:, 2] = self.cup_drop_local_z
            goal_pos = rotation_conversions.quaternion_apply(cup_q, local_goal) + cup_base_pos
            self.goal_site.set_pose(Pose.create_from_pq(goal_pos))

            hidden_pos = torch.zeros((b, 3), device=self.device)
            hidden_pos[:, 2] = -10
            all_hidable_obstacles = (
                self.harder_transfer_obstacles
                + self.harder_table_obstacles
                + list(self.harder_overhead_obstacles.values())
            )
            for obstacle in all_hidable_obstacles:
                obstacle.set_pose(Pose.create_from_pq(hidden_pos))

            path_xy = cup_base_pos[:, :2] - dish_base_pos[:, :2]
            path_norm = torch.linalg.norm(path_xy, axis=1, keepdims=True).clamp(min=1e-6)
            path_dir = path_xy / path_norm
            perp_dir = torch.stack([-path_dir[:, 1], path_dir[:, 0]], dim=1)
            path_yaw = torch.atan2(path_dir[:, 1], path_dir[:, 0])

            if self.harder:
                use_transfer = torch.rand((b), device=self.device) < 0.9
                transfer_count = torch.where(
                    torch.rand((b), device=self.device) < 0.5,
                    torch.ones((b), device=self.device, dtype=torch.int64),
                    torch.full((b,), 2, device=self.device, dtype=torch.int64),
                )
                use_table = torch.rand((b), device=self.device) < 0.8
                table_count = torch.randint(1, 4, (b,), device=self.device)
                use_overhead_dish = torch.rand((b), device=self.device) < 0.45
                use_overhead_cup = torch.rand((b), device=self.device) < 0.45

                any_active = (
                    use_transfer | use_table | use_overhead_dish | use_overhead_cup
                )
                force_transfer = ~any_active
                use_transfer = use_transfer | force_transfer
                transfer_count = torch.where(
                    force_transfer,
                    torch.ones_like(transfer_count),
                    transfer_count,
                )

                transfer_sign = torch.where(
                    torch.rand((b), device=self.device) < 0.5,
                    torch.ones((b), device=self.device),
                    -torch.ones((b), device=self.device),
                )

                transfer0_center = dish_base_pos[:, :2] + path_xy * (
                    0.44 + (torch.rand((b), device=self.device) - 0.5) * 0.08
                )[:, None]
                transfer0_center += perp_dir * (
                    transfer_sign * (0.03 + torch.rand((b), device=self.device) * 0.03)
                )[:, None]
                transfer0_center[:, 0] = torch.clamp(transfer0_center[:, 0], min=0.07, max=0.26)
                transfer0_active = use_transfer & (transfer_count >= 1)
                transfer0_yaw = path_yaw + (torch.rand((b), device=self.device) - 0.5) * 0.18
                self._set_box_obstacle_pose(
                    self.harder_transfer_obstacles[0],
                    transfer0_center,
                    self.harder_transfer_half_size,
                    transfer0_yaw,
                    transfer0_active,
                    hidden_pos,
                )

                transfer1_center = dish_base_pos[:, :2] + path_xy * (
                    0.70 + (torch.rand((b), device=self.device) - 0.5) * 0.08
                )[:, None]
                transfer1_center += perp_dir * (
                    -transfer_sign * (0.02 + torch.rand((b), device=self.device) * 0.03)
                )[:, None]
                transfer1_center[:, 0] = torch.clamp(transfer1_center[:, 0], min=0.07, max=0.26)
                transfer1_active = use_transfer & (transfer_count >= 2)
                transfer1_yaw = path_yaw + (torch.rand((b), device=self.device) - 0.5) * 0.18
                self._set_box_obstacle_pose(
                    self.harder_transfer_obstacles[1],
                    transfer1_center,
                    self.harder_transfer_half_size,
                    transfer1_yaw,
                    transfer1_active,
                    hidden_pos,
                )

                table_fractions = [0.30, 0.55, 0.80]
                table_lateral_base = [0.07, -0.06, 0.05]
                for i, obstacle in enumerate(self.harder_table_obstacles):
                    active_mask = use_table & (table_count > i)
                    obs_pos = hidden_pos.clone()
                    posed = torch.zeros((b, 3), device=self.device)
                    posed[:, :2] = dish_base_pos[:, :2] + path_xy * (
                        table_fractions[i] + (torch.rand((b), device=self.device) - 0.5) * 0.08
                    )[:, None]
                    posed[:, :2] += perp_dir * (
                        table_lateral_base[i] + (torch.rand((b), device=self.device) - 0.5) * 0.05
                    )[:, None]
                    posed[:, 0] = torch.clamp(posed[:, 0], min=0.07, max=0.28)
                    model_id = self.harder_table_obstacle_model_ids[i]
                    posed[:, 2] = self._model_meta[model_id]["bottom_z"]
                    obs_pos = torch.where(active_mask[:, None], posed, obs_pos)
                    obs_q = randomization.random_quaternions(
                        n=b, device=self.device, lock_x=True, lock_y=True
                    )
                    identity_q = torch.zeros_like(obs_q)
                    identity_q[:, 0] = 1.0
                    obs_q = torch.where(active_mask[:, None], obs_q, identity_q)
                    obstacle.set_pose(Pose.create_from_pq(obs_pos, obs_q))

                overhead_center_z = self.overhead_bottom_z + self.overhead_half_size[2]

                dish_overhead_center = dish_base_pos[:, :2]
                dish_overhead_center += path_dir * (
                    0.05 + (torch.rand((b), device=self.device) - 0.5) * 0.02
                )[:, None]
                dish_overhead_center += perp_dir * (
                    (torch.rand((b), device=self.device) - 0.5) * 0.06
                )[:, None]
                dish_overhead_yaw = path_yaw + (torch.rand((b), device=self.device) - 0.5) * 0.3
                self._set_box_obstacle_pose(
                    self.harder_overhead_obstacles["dish"],
                    dish_overhead_center,
                    self.overhead_half_size,
                    dish_overhead_yaw,
                    use_overhead_dish,
                    hidden_pos,
                    center_z=overhead_center_z,
                )

                cup_overhead_center = cup_base_pos[:, :2]
                cup_overhead_center += path_dir * (
                    -0.05 + (torch.rand((b), device=self.device) - 0.5) * 0.02
                )[:, None]
                cup_overhead_center += perp_dir * (
                    (torch.rand((b), device=self.device) - 0.5) * 0.06
                )[:, None]
                cup_overhead_yaw = path_yaw + (torch.rand((b), device=self.device) - 0.5) * 0.3
                self._set_box_obstacle_pose(
                    self.harder_overhead_obstacles["cup"],
                    cup_overhead_center,
                    self.overhead_half_size,
                    cup_overhead_yaw,
                    use_overhead_cup,
                    hidden_pos,
                    center_z=overhead_center_z,
                )
            else:
                # Default mode has no additional obstacles.
                pass

    def _compute_ice_in_cup(self):
        ice_pose_local_to_cup = self.cup.pose.inv() * self.ice_cube.pose
        local_pos = ice_pose_local_to_cup.p
        xy_dist = torch.linalg.norm(local_pos[:, :2], axis=1)
        inside_xy = xy_dist <= self.cup_success_radius
        inside_z = torch.logical_and(
            local_pos[:, 2] >= self.cup_success_z_min,
            local_pos[:, 2] <= self.cup_success_z_max,
        )
        return torch.logical_and(inside_xy, inside_z)

    def get_obstacles_info(self):
        if not self.harder:
            return []
        obstacles_info = []
        for actor in self.harder_transfer_obstacles:
            raw_pose = actor.pose.raw_pose
            obstacles_info.append(
                dict(
                    center=raw_pose[:, :3],
                    quat=raw_pose[:, 3:],
                    extent=torch.tensor(
                        self.harder_transfer_half_size,
                        dtype=torch.float32,
                        device=raw_pose.device,
                    ).expand(raw_pose.shape[0], 3),
                )
            )
        for actor in self.harder_overhead_obstacles.values():
            raw_pose = actor.pose.raw_pose
            obstacles_info.append(
                dict(
                    center=raw_pose[:, :3],
                    quat=raw_pose[:, 3:],
                    extent=torch.tensor(
                        self.overhead_half_size,
                        dtype=torch.float32,
                        device=raw_pose.device,
                    ).expand(raw_pose.shape[0], 3),
                )
            )
        for i, actor in enumerate(self.harder_table_obstacles):
            raw_pose = actor.pose.raw_pose
            half_extents = self._model_meta[self.harder_table_obstacle_model_ids[i]][
                "half_extents"
            ]
            obstacles_info.append(
                dict(
                    center=raw_pose[:, :3],
                    quat=raw_pose[:, 3:],
                    extent=torch.tensor(
                        half_extents, dtype=torch.float32, device=raw_pose.device
                    ).expand(raw_pose.shape[0], 3),
                )
            )
        return obstacles_info

    def evaluate(self):
        is_ice_in_cup = self._compute_ice_in_cup()
        is_grasped = self.agent.is_grasping(self.ice_cube)
        success = torch.logical_and(is_ice_in_cup, ~is_grasped)
        return {
            "success": success,
            "is_ice_in_cup": is_ice_in_cup,
            "is_grasped": is_grasped,
        }

    def _get_obs_extra(self, info: dict):
        tcp_pose_world_frame = self.agent.tcp_pose
        tcp_pose_root_frame = self.agent.robot.get_pose().inv() * tcp_pose_world_frame
        obs = dict(
            is_ice_in_cup=info["is_ice_in_cup"],
            is_grasped=info["is_grasped"],
            tcp_pose=tcp_pose_root_frame.raw_pose,
            tcp_pose_world_frame=tcp_pose_world_frame.raw_pose,
            cup_pos=self.cup.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                ice_cube_pose=self.ice_cube.pose.raw_pose,
                dish_pose=self.dish.pose.raw_pose,
                cup_pose=self.cup.pose.raw_pose,
                tcp_to_ice_cube_pos=self.ice_cube.pose.p - self.agent.tcp_pose.p,
                ice_cube_to_goal_pos=self.goal_site.pose.p - self.ice_cube.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        tcp_to_ice_dist = torch.linalg.norm(
            self.ice_cube.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_ice_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped.float()

        ice_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.ice_cube.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * ice_to_goal_dist)
        reward += place_reward * is_grasped.float()

        release_reward = info["is_ice_in_cup"].float() * (~is_grasped).float()
        reward += release_reward

        reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5.0
