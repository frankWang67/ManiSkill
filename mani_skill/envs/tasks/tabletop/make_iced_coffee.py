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
    IIwa7RobotiqWristCamera,
    Gen36DofRobotiqWristCamera,
    Gen37DofRobotiqWristCamera,
    Rizon4RobotiqWristCamera,
    SawyerRobotiqWristCamera,
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
    - harder: sampled combinations of transfer blockers and table clutter.

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

    dish_model_id = "024_bowl"
    cup_model_id = "025_mug"
    harder_pick_place_obstacle_model_ids = [
        "021_bleach_cleanser",
        "003_cracker_box",
        "006_mustard_bottle",
        "003_cracker_box",
        "021_bleach_cleanser",
    ]
    # Tall wall to block the lifted transport trajectory (TCP z ≥ 0.26).
    # Half-sizes: [thin along path, wide perpendicular to path, tall].
    taller_transport_half_sizes = [
        [0.015, 0.09, 0.15],
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

        ycb_meta = load_json(ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json")
        self._model_meta = dict()
        for model_id in [
            self.dish_model_id,
            self.cup_model_id,
            *self.harder_pick_place_obstacle_model_ids,
        ]:
            if model_id not in ycb_meta:
                continue
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

    def _yaw_to_quat(self, yaw: torch.Tensor):
        q = torch.zeros((yaw.shape[0], 4), device=yaw.device, dtype=yaw.dtype)
        q[:, 0] = torch.cos(yaw / 2)
        q[:, 3] = torch.sin(yaw / 2)
        return q

    def _set_ycb_obstacle_pose(
        self,
        actor,
        model_id: str,
        center_xy: torch.Tensor,
        yaw: torch.Tensor,
        active_mask: torch.Tensor,
        hidden_pos: torch.Tensor,
        bottom_height: float = None,
    ):
        b = center_xy.shape[0]
        pos = hidden_pos.clone()
        quat = torch.zeros((b, 4), device=center_xy.device, dtype=center_xy.dtype)
        quat[:, 0] = 1.0
        if torch.any(active_mask):
            posed = torch.zeros((b, 3), device=center_xy.device, dtype=center_xy.dtype)
            posed[:, :2] = center_xy
            base_bottom_z = self._model_meta[model_id]["bottom_z"]
            if bottom_height is None:
                posed[:, 2] = float(base_bottom_z)
            else:
                posed[:, 2] = float(bottom_height + base_bottom_z)
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

        self.harder_pick_place_obstacles = []
        for i, model_id in enumerate(self.harder_pick_place_obstacle_model_ids):
            self.harder_pick_place_obstacles.append(
                self._build_ycb_actor(
                    model_id,
                    f"harder_pick_place_obstacle_{i}",
                    body_type="kinematic",
                    add_collision=True,
                    add_visual=True,
                )
            )

        self.taller_transport_obstacles = []
        for i, half_size in enumerate(self.taller_transport_half_sizes):
            obstacle = actors.build_box(
                self.scene,
                half_sizes=half_size,
                color=[0.65, 0.35, 0.35, 1.0],
                name=f"taller_transport_obstacle_{i}",
                body_type="kinematic",
                initial_pose=sapien.Pose(p=[0, 0, -10]),
            )
            self.taller_transport_obstacles.append(obstacle)

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
            all_hidable_obstacles = self.harder_pick_place_obstacles + self.taller_transport_obstacles
            for obstacle in all_hidable_obstacles:
                obstacle.set_pose(Pose.create_from_pq(hidden_pos))

            path_xy = cup_base_pos[:, :2] - dish_base_pos[:, :2]
            path_norm = torch.linalg.norm(path_xy, axis=1, keepdims=True).clamp(min=1e-6)
            path_dir = path_xy / path_norm
            perp_dir = torch.stack([-path_dir[:, 1], path_dir[:, 0]], dim=1)
            path_yaw = torch.atan2(path_dir[:, 1], path_dir[:, 0])

            if self.harder:
                # Radii for clearance calculation (use visual model half-extents).
                dish_visual_r = float(max(self._model_meta[self.dish_model_id]["half_extents"][:2]))
                cup_visual_r = float(max(self._model_meta[self.cup_model_id]["half_extents"][:2]))
                margin = 0.05

                def _obs_radius(model_id: str) -> float:
                    he = self._model_meta[model_id]["half_extents"]
                    return float(max(he[0], he[1]))

                path_dist_1d = path_norm.squeeze(-1)

                # --- Pick/Place obstacles (YCB objects near dish/cup, robot side) ---
                # Alternate assignment: even indices near dish, odd indices near cup.
                # Robot base is at x ≈ -0.522, so the robot-facing side is negative-x.
                for i, (actor, model_id) in enumerate(
                    zip(self.harder_pick_place_obstacles, self.harder_pick_place_obstacle_model_ids)
                ):
                    near_dish = i % 2 == 0
                    active = torch.rand((b,), device=self.device) < 0.65

                    if near_dish:
                        ref_center = dish_base_pos[:, :2]
                        ref_r = dish_visual_r
                    else:
                        ref_center = cup_base_pos[:, :2]
                        ref_r = cup_visual_r

                    obs_r = _obs_radius(model_id)
                    posed = torch.zeros((b, 3), device=self.device)
                    # Place on robot side (negative x), clear of the receptacle.
                    posed[:, 0] = (
                        ref_center[:, 0] - ref_r - obs_r - margin
                        - torch.rand((b,), device=self.device) * 0.03
                    )
                    posed[:, 0] = torch.clamp(posed[:, 0], min=-0.25)
                    # Spread obstacles in y around the reference center.
                    y_offset = (torch.rand((b,), device=self.device) - 0.5) * 0.08
                    posed[:, 1] = ref_center[:, 1] + y_offset
                    posed[:, 2] = self._model_meta[model_id]["bottom_z"]

                    # Also ensure obstacle does not overlap the OTHER receptacle.
                    if near_dish:
                        other_center = cup_base_pos[:, :2]
                        other_r = cup_visual_r
                    else:
                        other_center = dish_base_pos[:, :2]
                        other_r = dish_visual_r
                    dist_to_other = torch.linalg.norm(posed[:, :2] - other_center, axis=1)
                    clear_of_other = dist_to_other > (other_r + obs_r + margin)

                    valid = clear_of_other
                    active = active & valid

                    obs_pos = torch.where(active[:, None], posed, hidden_pos)
                    obs_q = randomization.random_quaternions(
                        n=b, device=self.device, lock_x=True, lock_y=True
                    )
                    identity_q = torch.zeros_like(obs_q)
                    identity_q[:, 0] = 1.0
                    obs_q = torch.where(active[:, None], obs_q, identity_q)
                    actor.set_pose(Pose.create_from_pq(obs_pos, obs_q))

                # --- Tall transport wall (blocks the lifted trajectory) ---
                # The wall is thin along the path (half_x) and wide perpendicular
                # to it (half_y), oriented to span across the direct dish→cup line.
                # Only half_x matters for clearance along the path direction.
                for i, (actor, half_size) in enumerate(
                    zip(self.taller_transport_obstacles, self.taller_transport_half_sizes)
                ):
                    wall_thin_r = float(half_size[0])  # extent along the path
                    active = torch.rand((b,), device=self.device) < 0.70

                    min_f = (
                        (dish_visual_r + wall_thin_r + margin)
                        / path_dist_1d.clamp(min=1e-6)
                    )
                    max_f = 1.0 - (
                        (cup_visual_r + wall_thin_r + margin)
                        / path_dist_1d.clamp(min=1e-6)
                    )
                    valid = min_f < max_f
                    active = active & valid

                    # Wall sits near the midpoint of the path.
                    frac = torch.where(
                        valid,
                        min_f
                        + (max_f - min_f)
                        * torch.clamp(
                            0.45 + (torch.rand((b,), device=self.device) - 0.5) * 0.16,
                            min=0.0,
                            max=1.0,
                        ),
                        torch.full((b,), 0.50, device=self.device),
                    )

                    # Center the wall on the path (no lateral offset).
                    posed = torch.zeros((b, 3), device=self.device)
                    posed[:, :2] = dish_base_pos[:, :2] + path_xy * frac[:, None]
                    posed[:, 2] = half_size[2]

                    obs_pos = torch.where(active[:, None], posed, hidden_pos)
                    # Orient the wall so its thin side faces along the path
                    # and its wide side spans across the path.
                    wall_yaw = path_yaw + (torch.rand((b,), device=self.device) - 0.5) * 0.2
                    obs_q = self._yaw_to_quat(wall_yaw)
                    identity_q = torch.zeros_like(obs_q)
                    identity_q[:, 0] = 1.0
                    obs_q = torch.where(active[:, None], obs_q, identity_q)
                    actor.set_pose(Pose.create_from_pq(obs_pos, obs_q))
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
        robot_root_inv = self.agent.robot.get_pose().inv()
        obstacles_info = []
        for i, actor in enumerate(self.harder_pick_place_obstacles):
            raw_pose = (robot_root_inv * actor.pose).raw_pose
            model_id = self.harder_pick_place_obstacle_model_ids[i]
            obstacles_info.append(
                dict(
                    center=raw_pose[:, :3],
                    quat=raw_pose[:, 3:],
                    extent=torch.tensor(
                        self._model_meta[model_id]["half_extents"],
                        dtype=torch.float32,
                        device=raw_pose.device,
                    ).expand(raw_pose.shape[0], 3),
                )
            )
        for i, actor in enumerate(self.taller_transport_obstacles):
            raw_pose = (robot_root_inv * actor.pose).raw_pose
            half_sizes = torch.tensor(
                self.taller_transport_half_sizes[i],
                dtype=torch.float32,
                device=raw_pose.device,
            ).expand(raw_pose.shape[0], 3)
            obstacles_info.append(
                dict(
                    center=raw_pose[:, :3],
                    quat=raw_pose[:, 3:],
                    extent=half_sizes,
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
