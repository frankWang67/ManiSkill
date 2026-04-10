from pathlib import Path
from typing import Any, Union

import numpy as np
import sapien
import sapien.render
import torch
from transforms3d.euler import euler2quat

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
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.robocasa.utils.scene_utils import (
    ROBOCASA_ASSET_DIR,
)
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, SceneConfig, SimConfig


def create_colored_material(color, metallic=0.0, roughness=0.5, specular=0.5):
    mat = sapien.render.RenderMaterial()
    mat.base_color = color
    mat.metallic = metallic
    mat.roughness = roughness
    mat.specular = specular
    return mat


@register_env(
    "PickPlaceToasterToCounter-v1",
    max_episode_steps=90,
    asset_download_ids=["RoboCasa", "ycb"],
)
class PickPlaceToasterToCounterEnv(BaseEnv):
    """
    **Task Description:**
    Pick a slice of toast that starts inserted vertically in a toaster and place it
    onto a plate on the counter.

    **Difficulty Levels:**
    - default: only the counter, toaster, and plate are present.
    - harder: extra kinematic tabletop obstacles block the direct transfer path from
      toaster to plate.

    **Success Condition:**
    - the toast is resting on the plate and is no longer grasped.
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

    plate_model_id = "029_plate"
    harder_obstacle_model_ids = [
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

        self.toaster_mesh_scale = np.array([0.30, 0.30, 0.30], dtype=np.float32)
        self.toaster_half_extents = (
            np.array([0.26006575, 0.50, 0.32624074], dtype=np.float32)
            * self.toaster_mesh_scale
        )
        self.toaster_bottom_z = float(self.toaster_half_extents[2])

        # Two-slot toaster cavity approximation (outer walls + middle divider)
        # that keeps the toast upright while still allowing a top-down grasp.
        self.slot_wall_thickness = 0.008
        self.slot_bottom_thickness = 0.010
        self.slot_half_size = np.array([0.050, 0.090, 0.075], dtype=np.float32)
        self.slot_bottom_z = 0.040
        self.slot_lane_center_x = float(
            (self.slot_half_size[0] + self.slot_wall_thickness / 2) / 2
        )
        # Spawn in the lane closer to the robot for easier access.
        self.toast_slot_sign = -1.0

        # Scale the available RoboCasa bread slice up modestly so it behaves like
        # the sandwich-bread slice used by the reference task.
        # self.toast_object_scale = 1.35
        self.toast_object_scale = 2.0
        self.toast_visual_scale = np.array(
            [0.12 * self.toast_object_scale] * 3, dtype=np.float32
        )
        self.toast_upright_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        # Planning target near the upper half of the toast when it is upright.
        self.toast_grasp_local_offset = np.array([0.0, 0.0, 0.074], dtype=np.float32)

        self.plate_inner_radius = 0.085
        self.plate_wall_thickness = 0.004
        self.plate_inner_height = 0.012
        self.plate_bottom_thickness = 0.006
        self.plate_goal_local_z = self.plate_bottom_thickness + 0.018
        self.plate_success_radius = self.plate_inner_radius - 0.014
        self.plate_success_z_min = 0.008
        self.plate_success_z_max = 0.120

        self.table_spawn_bounds = dict(
            toaster_x=(0.10, 0.16),
            toaster_y=(-0.02, 0.05),
            plate_between_alpha=(0.10, 0.16),
        )

        ycb_meta = load_json(ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json")
        self._model_meta = {}
        for model_id in [self.plate_model_id, *self.harder_obstacle_model_ids]:
            info = ycb_meta[model_id]
            scale = float(info.get("scales", [1.0])[0])
            bbox_min = np.array(info["bbox"]["min"], dtype=np.float32) * scale
            bbox_max = np.array(info["bbox"]["max"], dtype=np.float32) * scale
            self._model_meta[model_id] = dict(
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                half_extents=(bbox_max - bbox_min) / 2.0,
                bottom_z=float(-bbox_min[2]),
            )

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        # Use stricter contact solving for thin toast-gripper contacts to reduce
        # visible pad/toast interpenetration during grasp.
        return SimConfig(
            scene_config=SceneConfig(
                contact_offset=0.008,
                rest_offset=0.0,
                solver_position_iterations=40,
                solver_velocity_iterations=16,
            )
        )

    @property
    def _default_sensor_configs(self):
        return self.agent._sensor_configs

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=[-0.92, 0.92, 0.78], target=[0.05, -0.06, 0.08]
        )
        return CameraConfig(
            "render_camera",
            pose=pose,
            width=512,
            height=512,
            fov=1.0,
            near=0.01,
            far=100,
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
            self.scene,
            id=f"ycb:{model_id}",
            add_collision=add_collision,
            add_visual=add_visual,
        )
        builder.initial_pose = sapien.Pose([0, 0, -10])
        if body_type == "dynamic":
            return builder.build(name=name)
        if body_type == "kinematic":
            return builder.build_kinematic(name=name)
        raise ValueError(f"Unsupported body_type={body_type}")

    def _build_plate_receptacle(self, name: str):
        builder = self.scene.create_actor_builder()
        outer_r = self.plate_inner_radius + self.plate_wall_thickness
        rim_center_z = self.plate_bottom_thickness + self.plate_inner_height / 2
        bottom_half = [outer_r, outer_r, self.plate_bottom_thickness / 2]
        wall_x_half = [
            self.plate_wall_thickness / 2,
            outer_r,
            self.plate_inner_height / 2,
        ]
        wall_y_half = [
            self.plate_inner_radius,
            self.plate_wall_thickness / 2,
            self.plate_inner_height / 2,
        ]

        builder.add_box_collision(
            pose=sapien.Pose([0, 0, self.plate_bottom_thickness / 2]),
            half_size=bottom_half,
        )
        for sx in [-1, 1]:
            builder.add_box_collision(
                pose=sapien.Pose(
                    [sx * (self.plate_inner_radius + self.plate_wall_thickness / 2), 0, rim_center_z]
                ),
                half_size=wall_x_half,
            )
        for sy in [-1, 1]:
            builder.add_box_collision(
                pose=sapien.Pose(
                    [0, sy * (self.plate_inner_radius + self.plate_wall_thickness / 2), rim_center_z]
                ),
                half_size=wall_y_half,
            )
        builder.initial_pose = sapien.Pose([0, 0, -10])
        return builder.build_kinematic(name=name)

    def _build_toaster_visual(self, name: str):
        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(
            filename=str(
                ROBOCASA_ASSET_DIR
                / "fixtures/toasters/basic_popup/visuals/model_0.obj"
            ),
            scale=self.toaster_mesh_scale.tolist(),
        )
        builder.initial_pose = sapien.Pose([0, 0, -10])
        return builder.build_kinematic(name=name)

    def _build_toast(self, name: str):
        builder = self.scene.create_actor_builder()
        # Triangle-mesh dynamic bodies can collapse to near-zero auto-computed
        # mass/inertia; set explicit values for stable contact resolution.
        toast_mass = 0.03
        builder.set_mass_and_inertia(
            toast_mass,
            sapien.Pose(),
            np.array(
                [
                    9.0e-05,  # Ixx
                    6.7e-05,  # Iyy
                    2.6e-05,  # Izz
                ],
                dtype=np.float32,
            ),
        )
        toast_material = sapien.pysapien.physx.PhysxMaterial(
            static_friction=3.0, dynamic_friction=2.5, restitution=0.0
        )
        toast_mesh_pose = sapien.Pose(q=euler2quat(0, np.pi / 2, 0))
        toast_mesh_file = (
            ROBOCASA_ASSET_DIR
            / "objects/objaverse/bread/bread_9/visual/model_normalized_0.obj"
        )
        # Keep collision geometry identical to the rendered mesh.
        builder.add_nonconvex_collision_from_file(
            filename=str(toast_mesh_file),
            pose=toast_mesh_pose,
            scale=self.toast_visual_scale.tolist(),
            material=toast_material,
        )
        builder.add_visual_from_file(
            filename=str(toast_mesh_file),
            scale=self.toast_visual_scale.tolist(),
            pose=toast_mesh_pose,
        )
        builder.initial_pose = sapien.Pose([0, 0, -10])
        return builder.build(name=name)

    def toast_grasp_center(self):
        grasp_center = self.toast.pose.p.clone()
        grasp_center[:, 2] += float(self.toast_grasp_local_offset[2])
        return grasp_center

    def _build_box_obstacle(
        self,
        name: str,
        half_size,
        color=None,
        add_visual: bool = False,
    ):
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=half_size)
        if add_visual:
            if color is None:
                color = [0.55, 0.55, 0.55, 1.0]
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

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.toaster_visual = self._build_toaster_visual("toaster_visual")
        self.plate_visual = self._build_ycb_actor(
            self.plate_model_id,
            "plate_visual",
            body_type="kinematic",
            add_collision=False,
            add_visual=True,
        )
        self.plate = self._build_plate_receptacle("plate")

        self.toaster_bottom = self._build_box_obstacle(
            "toaster_slot_bottom",
            half_size=[
                self.slot_half_size[0],
                self.slot_half_size[1],
                self.slot_bottom_thickness / 2,
            ],
        )
        self.toaster_left = self._build_box_obstacle(
            "toaster_slot_left",
            half_size=[
                self.slot_wall_thickness / 2,
                self.slot_half_size[1] + self.slot_wall_thickness,
                self.slot_half_size[2],
            ],
        )
        self.toaster_right = self._build_box_obstacle(
            "toaster_slot_right",
            half_size=[
                self.slot_wall_thickness / 2,
                self.slot_half_size[1] + self.slot_wall_thickness,
                self.slot_half_size[2],
            ],
        )
        self.toaster_middle = self._build_box_obstacle(
            "toaster_slot_middle",
            half_size=[
                self.slot_wall_thickness / 2,
                self.slot_half_size[1] + self.slot_wall_thickness,
                self.slot_half_size[2],
            ],
        )
        self.toaster_front = self._build_box_obstacle(
            "toaster_slot_front",
            half_size=[
                self.slot_half_size[0],
                self.slot_wall_thickness / 2,
                self.slot_half_size[2],
            ],
        )
        self.toaster_back = self._build_box_obstacle(
            "toaster_slot_back",
            half_size=[
                self.slot_half_size[0],
                self.slot_wall_thickness / 2,
                self.slot_half_size[2],
            ],
        )
        self.toaster_obstacles = [
            self.toaster_bottom,
            self.toaster_left,
            self.toaster_right,
            self.toaster_middle,
            self.toaster_front,
            self.toaster_back,
        ]
        self.toaster_obstacle_half_sizes = [
            np.array(
                [
                    self.slot_half_size[0],
                    self.slot_half_size[1],
                    self.slot_bottom_thickness / 2,
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    self.slot_wall_thickness / 2,
                    self.slot_half_size[1] + self.slot_wall_thickness,
                    self.slot_half_size[2],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    self.slot_wall_thickness / 2,
                    self.slot_half_size[1] + self.slot_wall_thickness,
                    self.slot_half_size[2],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    self.slot_wall_thickness / 2,
                    self.slot_half_size[1] + self.slot_wall_thickness,
                    self.slot_half_size[2],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    self.slot_half_size[0],
                    self.slot_wall_thickness / 2,
                    self.slot_half_size[2],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    self.slot_half_size[0],
                    self.slot_wall_thickness / 2,
                    self.slot_half_size[2],
                ],
                dtype=np.float32,
            ),
        ]

        self.toast = self._build_toast("toast")

        self.harder_obstacles = []
        for i, model_id in enumerate(self.harder_obstacle_model_ids):
            obstacle = self._build_ycb_actor(
                model_id,
                f"harder_obstacle_{i}",
                body_type="kinematic",
                add_collision=True,
                add_visual=True,
            )
            self.harder_obstacles.append(obstacle)

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

    def _after_reconfigure(self, options: dict):
        collision_mesh = self.toast.get_first_collision_mesh()
        if collision_mesh is not None:
            self.toast_half_extents = (
                collision_mesh.bounding_box.extents / 2
            ).astype(np.float32)
        else:
            self.toast_half_extents = np.array([0.045, 0.078, 0.081], dtype=np.float32)

        # Spawn height should use the full vertical span of the current toast
        # collision approximation. Using only planar extents can sink the toast
        # into the toaster floor at reset when collision geometry changes.
        self.toast_upright_height = float(np.max(self.toast_half_extents) * 2)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            toaster_base_pos = torch.zeros((b, 3), dtype=torch.float32, device=self.device)
            toaster_base_pos[:, 0] = (
                torch.rand((b), device=self.device)
                * (self.table_spawn_bounds["toaster_x"][1] - self.table_spawn_bounds["toaster_x"][0])
                + self.table_spawn_bounds["toaster_x"][0] - 0.05
            )
            toaster_base_pos[:, 1] = (
                torch.rand((b), device=self.device)
                * (self.table_spawn_bounds["toaster_y"][1] - self.table_spawn_bounds["toaster_y"][0])
                + self.table_spawn_bounds["toaster_y"][0]
            )
            toaster_visual_pos = toaster_base_pos.clone()
            toaster_visual_pos[:, 2] = self.toaster_bottom_z
            toaster_q = torch.zeros((b, 4), dtype=torch.float32, device=self.device)
            toaster_q[:, 0] = 1.0
            self.toaster_visual.set_pose(Pose.create_from_pq(toaster_visual_pos, toaster_q))

            slot_center_z = self.slot_bottom_z + self.slot_half_size[2]
            bottom_center_z = self.slot_bottom_z - self.slot_bottom_thickness / 2

            def _set_toaster_piece(actor, center_offset, quat=None):
                pose_p = toaster_base_pos + torch.tensor(
                    center_offset, dtype=torch.float32, device=self.device
                )
                if quat is None:
                    local_q = toaster_q
                else:
                    local_q = torch.tensor(quat, dtype=torch.float32, device=self.device).repeat(b, 1)
                actor.set_pose(Pose.create_from_pq(pose_p, q=local_q))

            _set_toaster_piece(self.toaster_bottom, [0.0, 0.0, bottom_center_z])
            _set_toaster_piece(
                self.toaster_left,
                [-(self.slot_half_size[0] + self.slot_wall_thickness / 2), 0.0, slot_center_z],
            )
            _set_toaster_piece(
                self.toaster_right,
                [(self.slot_half_size[0] + self.slot_wall_thickness / 2), 0.0, slot_center_z],
            )
            _set_toaster_piece(
                self.toaster_middle,
                [0.0, 0.0, slot_center_z],
            )
            _set_toaster_piece(
                self.toaster_front,
                [0.0, -(self.slot_half_size[1] + self.slot_wall_thickness / 2), slot_center_z],
            )
            _set_toaster_piece(
                self.toaster_back,
                [0.0, (self.slot_half_size[1] + self.slot_wall_thickness / 2), slot_center_z],
            )
            robot_base_pos = self.agent.robot.pose.p[env_idx].clone()
            plate_alpha = (
                torch.rand((b), device=self.device)
                * (
                    self.table_spawn_bounds["plate_between_alpha"][1]
                    - self.table_spawn_bounds["plate_between_alpha"][0]
                )
                + self.table_spawn_bounds["plate_between_alpha"][0]
            )
            plate_base_pos = torch.zeros((b, 3), dtype=torch.float32, device=self.device)
            plate_base_pos[:, :2] = toaster_base_pos[:, :2] + (
                robot_base_pos[:, :2] - toaster_base_pos[:, :2]
            ) * plate_alpha[:, None]
            plate_base_pos[:, 2] = 0.0
            # plate_base_pos[:, 0] = torch.clamp(
            #     plate_base_pos[:, 0], min=-0.02, max=0.14
            # )
            plate_base_pos[:, 0] = -torch.ones((b), device=self.device) * 0.20
            plate_base_pos[:, 1] = torch.clamp(plate_base_pos[:, 1], min=-0.02, max=0.05)
            plate_q = torch.zeros((b, 4), dtype=torch.float32, device=self.device)
            plate_q[:, 0] = 1.0
            self.plate.set_pose(Pose.create_from_pq(plate_base_pos, plate_q))
            plate_visual_pos = plate_base_pos.clone()
            plate_visual_pos[:, 2] += self._model_meta[self.plate_model_id]["bottom_z"]
            self.plate_visual.set_pose(Pose.create_from_pq(plate_visual_pos, plate_q))

            toast_pos = toaster_base_pos.clone()
            toast_pos[:, 0] += self.toast_slot_sign * self.slot_lane_center_x
            toast_pos[:, 2] = self.slot_bottom_z + self.toast_upright_height / 2
            toast_q = torch.tensor(
                self.toast_upright_quat, dtype=torch.float32, device=self.device
            ).repeat(b, 1)
            self.toast.set_pose(Pose.create_from_pq(toast_pos, toast_q))
            self.toast.set_linear_velocity(torch.zeros((b, 3), device=self.device))
            self.toast.set_angular_velocity(torch.zeros((b, 3), device=self.device))

            goal_pos = plate_base_pos.clone()
            goal_pos[:, 2] = self.plate_goal_local_z
            self.goal_site.set_pose(Pose.create_from_pq(goal_pos))

            hidden_pos = torch.zeros((b, 3), dtype=torch.float32, device=self.device)
            hidden_pos[:, 2] = -10.0
            for obstacle in self.harder_obstacles:
                obstacle.set_pose(Pose.create_from_pq(hidden_pos))

            if self.harder:
                path_xy = plate_base_pos[:, :2] - toaster_base_pos[:, :2]
                path_norm = torch.linalg.norm(path_xy, axis=1, keepdims=True).clamp(min=1e-6)
                path_dir = path_xy / path_norm
                plate_side_dir = torch.stack([-path_dir[:, 1], path_dir[:, 0]], dim=1)
                blocked_side_sign = torch.where(
                    torch.rand((b), device=self.device) < 0.5,
                    torch.ones((b), device=self.device),
                    -torch.ones((b), device=self.device),
                )
                blocked_side_offset = blocked_side_sign * (
                    0.20 + (torch.rand((b), device=self.device) - 0.5) * 0.02
                )
                side_center = (
                    plate_base_pos[:, :2] + plate_side_dir * blocked_side_offset[:, None]
                )
                longitudinal_offsets = torch.linspace(
                    -0.10,
                    0.10,
                    steps=len(self.harder_obstacles),
                    device=self.device,
                    dtype=torch.float32,
                )
                for i, obstacle in enumerate(self.harder_obstacles):
                    posed = hidden_pos.clone()
                    side_jitter = (torch.rand((b), device=self.device) - 0.5) * 0.01
                    posed[:, :2] = side_center + path_dir * (
                        longitudinal_offsets[i]
                        + (torch.rand((b), device=self.device) - 0.5) * 0.02
                    )[:, None]
                    posed[:, :2] += plate_side_dir * (
                        blocked_side_sign * side_jitter
                    )[:, None]
                    posed[:, 0] = torch.clamp(posed[:, 0], min=-0.32, max=0.02)
                    posed[:, 1] = torch.clamp(posed[:, 1], min=-0.28, max=0.18)
                    model_id = self.harder_obstacle_model_ids[i]
                    posed[:, 2] = self._model_meta[model_id]["bottom_z"]
                    obs_q = randomization.random_quaternions(
                        n=b, device=self.device, lock_x=True, lock_y=True
                    )
                    obstacle.set_pose(Pose.create_from_pq(posed, obs_q))

    def _compute_toast_on_plate(self):
        toast_local = self.plate.pose.inv() * self.toast.pose
        local_pos = toast_local.p
        xy_dist = torch.linalg.norm(local_pos[:, :2], axis=1)
        inside_xy = xy_dist <= self.plate_success_radius
        inside_z = torch.logical_and(
            local_pos[:, 2] >= self.plate_success_z_min,
            local_pos[:, 2] <= self.plate_success_z_max,
        )
        return torch.logical_and(inside_xy, inside_z)

    def evaluate(self):
        is_toast_on_plate = self._compute_toast_on_plate()
        is_grasped = self.agent.is_grasping(self.toast)
        success = torch.logical_and(is_toast_on_plate, ~is_grasped)
        return {
            "success": success,
            "is_toast_on_plate": is_toast_on_plate,
            "is_grasped": is_grasped,
        }

    def get_obstacles_info(self):
        obstacles_info = []
        for actor, half_extent in zip(
            self.toaster_obstacles, self.toaster_obstacle_half_sizes
        ):
            raw_pose = actor.pose.raw_pose
            obstacles_info.append(
                dict(
                    center=raw_pose[:, :3],
                    quat=raw_pose[:, 3:],
                    extent=torch.tensor(
                        half_extent, dtype=torch.float32, device=raw_pose.device
                    ).expand(raw_pose.shape[0], 3),
                )
            )
        if self.harder:
            for i, actor in enumerate(self.harder_obstacles):
                raw_pose = actor.pose.raw_pose
                obstacles_info.append(
                    dict(
                        center=raw_pose[:, :3],
                        quat=raw_pose[:, 3:],
                        extent=torch.tensor(
                            self._model_meta[self.harder_obstacle_model_ids[i]][
                                "half_extents"
                            ],
                            dtype=torch.float32,
                            device=raw_pose.device,
                        ).expand(raw_pose.shape[0], 3),
                    )
                )
        return obstacles_info

    def _get_obs_extra(self, info: dict):
        tcp_pose_world_frame = self.agent.tcp_pose
        tcp_pose_root_frame = self.agent.robot.get_pose().inv() * tcp_pose_world_frame
        obs = dict(
            is_toast_on_plate=info["is_toast_on_plate"],
            is_grasped=info["is_grasped"],
            tcp_pose=tcp_pose_root_frame.raw_pose,
            tcp_pose_world_frame=tcp_pose_world_frame.raw_pose,
            goal_pos=self.goal_site.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                toast_pose=self.toast.pose.raw_pose,
                toaster_pose=self.toaster_visual.pose.raw_pose,
                plate_pose=self.plate.pose.raw_pose,
                tcp_to_toast_pos=self.toast.pose.p - self.agent.tcp_pose.p,
                toast_to_goal_pos=self.goal_site.pose.p - self.toast.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        tcp_to_toast_dist = torch.linalg.norm(
            self.toast.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_toast_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped.float()

        toast_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.toast.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * toast_to_goal_dist)
        reward += place_reward * is_grasped.float()

        release_reward = info["is_toast_on_plate"].float() * (~is_grasped).float()
        reward += release_reward

        reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5.0
