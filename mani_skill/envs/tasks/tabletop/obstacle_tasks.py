from typing import Any, Union, List

import numpy as np
import sapien
import sapien.render
import torch
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.utils import randomization
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.sensors.camera import CameraConfig

GOAL_RADIUS = 0.10

# --- Helper Functions ---

def create_colored_material(color):
    """
    创建一个指定颜色的 RenderMaterial。
    """
    mat = sapien.render.RenderMaterial()
    mat.base_color = color
    mat.metallic = 0.0
    mat.roughness = 0.5
    mat.specular = 0.5
    return mat

# =============================================================================
# Task 1: PickFromDeepBox (Deep Box Picking)
# =============================================================================

@register_env("PickFromDeepBox-v1", max_episode_steps=50)
class PickFromDeepBoxEnv(PickCubeEnv):
    """
    Task Description:
    Target object is placed at the bottom of a deep box.
    Implementation: Using table as bottom, surrounded by 4 walls.
    Fixed: Increased inner width to avoid screw plan failures.
    """
    goal_radius = GOAL_RADIUS
    
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)
        self.goal_thresh = self.goal_radius

    @property
    def _default_human_render_camera_configs(self):
        self.human_cam_eye_pos[2] = 0.8
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        
        # Reduced wall height slightly to make planning easier
        self.wall_height = 0.15
        # Increased length to ensure coverage
        self.wall_length = 0.35
        self.wall_thickness = 0.02
        self.obstacle_half_sizes = []
        
        self.box_walls = []
        wall_names = ["box_front", "box_back", "box_left", "box_right"]
        
        wall_mat = create_colored_material([0.8, 0.6, 0.4, 1])
        
        for name in wall_names:
            builder = self.scene.create_actor_builder()
            half_size = [self.wall_thickness/2, self.wall_length/2, self.wall_height/2]
            builder.add_box_collision(half_size=half_size)
            builder.add_box_visual(half_size=half_size, material=wall_mat)
            builder.initial_pose = sapien.Pose(p=[0, 0, -10])
            part = builder.build_kinematic(name=name)
            self.box_walls.append(part)
            self.obstacle_half_sizes.append(half_size)

        self.goal_site = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_site",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0, 1e-3]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # --- 1. Randomize Box Parameters ---
            box_center_xy = torch.rand((b, 2)) * 0.2 - 0.15
            box_center_xy[:, 0] += 0.05 # Bias x towards front
            
            # FIX: Increased inner width range to [0.18, 0.24] to fit Panda gripper (~12cm)
            inner_width = torch.rand((b, 1)) * 0.06 + 0.25
            
            # Randomize depth/height [0.15, 0.22]
            target_depth = torch.rand((b, 1)) * 0.07 + 0.1
            
            # Wall Z position: CenterZ = TargetDepth - HalfHeight
            wall_z = target_depth - self.wall_height / 2

            # --- 2. Set Wall Poses ---
            offset = inner_width / 2 + self.wall_thickness / 2
            
            # Front Wall
            pos_front = torch.cat([box_center_xy[:, 0:1] + offset, box_center_xy[:, 1:2], wall_z], dim=1)
            q_front = torch.tensor([1, 0, 0, 0], dtype=torch.float, device=self.device).repeat(b, 1)
            self.box_walls[0].set_pose(Pose.create_from_pq(pos_front, q_front))

            # Back Wall
            pos_back = torch.cat([box_center_xy[:, 0:1] - offset, box_center_xy[:, 1:2], wall_z], dim=1)
            q_back = q_front 
            self.box_walls[1].set_pose(Pose.create_from_pq(pos_back, q_back))

            # Left Wall (Rotate 90 deg around Z)
            pos_left = torch.cat([box_center_xy[:, 0:1], box_center_xy[:, 1:2] + offset, wall_z], dim=1)
            q_side = torch.tensor([0.7071068, 0, 0, 0.7071068], dtype=torch.float, device=self.device).repeat(b, 1)
            self.box_walls[2].set_pose(Pose.create_from_pq(pos_left, q_side))

            # Right Wall
            pos_right = torch.cat([box_center_xy[:, 0:1], box_center_xy[:, 1:2] - offset, wall_z], dim=1)
            self.box_walls[3].set_pose(Pose.create_from_pq(pos_right, q_side))

            # --- 3. Place Cube ---
            cube_xyz = torch.zeros((b, 3))
            # Spawn inside box with margin
            spawn_range = inner_width - self.cube_half_size * 2 - 0.15
            cube_xyz[:, :2] = (torch.rand((b, 2)) - 0.5) * spawn_range
            cube_xyz[:, 0] += box_center_xy[:, 0]
            cube_xyz[:, 1] += box_center_xy[:, 1]
            cube_xyz[:, 2] = self.cube_half_size 
            
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(cube_xyz, qs))

            # --- 4. Goal ---
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, 0] = box_center_xy[:, 0] - 0.1
            goal_xyz[:, 1] = box_center_xy[:, 1] + 0.3
            goal_xyz[:, 2] = 1e-3
            # self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))
            self.goal_site.set_pose(
                Pose.create_from_pq(
                    p=goal_xyz,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )

    def get_obstacles_info(self):
        obstacles_info = []
        
        # 遍历环境中的架子部件 (Bottom, Top, Back)
        for i in range(len(self.box_walls)):
            part = self.box_walls[i]
            extent = self.obstacle_half_sizes[i]  # 预存的半尺寸列表
            
            current_pose = part.pose.raw_pose
            
            obs_info = {
                # 这里的 pose.p 和 pose.q 在 ManiSkill 封装下通常支持 Batch
                'center': current_pose[:, :3], # (B, 3)
                'quat': current_pose[:, 3:],   # (B, 4) [w, x, y, z]
                'extent': torch.tensor(extent, device=current_pose.device).expand(current_pose.shape[0], 3) # (B, 3) 动态读取的尺寸
            }
            obstacles_info.append(obs_info)
            
        return obstacles_info

# =============================================================================
# Task 2: PickFromShelf (Shelf Picking)
# =============================================================================

@register_env("PickFromShelf-v1", max_episode_steps=50)
class PickFromShelfEnv(PickCubeEnv):
    """
    Task Description:
    Target object on a shelf.
    Fixed: Increased shelf gap and reduced depth to facilitate motion planning.
    """
    goal_radius = GOAL_RADIUS
    
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)
        self.goal_thresh = self.goal_radius

    @property
    def _default_human_render_camera_configs(self):
        # registers a more high-definition (512x512) camera used just for rendering when render_mode="rgb_array" or calling env.render_rgb_array()
        pose = sapien_utils.look_at([-0.6, -0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        
        self.shelf_parts = []
        part_names = ["shelf_bottom", "shelf_top", "shelf_back"]
        
        shelf_mat = create_colored_material([0.5, 0.3, 0.1, 1])

        # FIX: Reduced shelf depth (half_size x) from 0.2 (0.4 total) to 0.12 (0.24 total)
        self.shelf_half_depth = 0.08
        self.shelf_half_width = 0.2
        self.obstacle_half_sizes = []

        for i, name in enumerate(part_names):
            builder = self.scene.create_actor_builder()
            
            if name == "shelf_back":
                # Back plate: x=thickness, y=width, z=height
                hs = [0.02, self.shelf_half_width, 0.3] 
            else:
                # Top/Bottom: x=depth, y=width, z=thickness
                hs = [self.shelf_half_depth, self.shelf_half_width, 0.02]
            self.obstacle_half_sizes.append(hs)
                
            builder.add_box_collision(half_size=hs)
            builder.add_box_visual(half_size=hs, material=shelf_mat)
            builder.initial_pose = sapien.Pose(p=[0, 0, -10])
            self.shelf_parts.append(builder.build_kinematic(name=name))

        self.goal_site = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_site",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0, 1e-3]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # --- 1. Randomize Parameters ---
            # Shelf Position
            shelf_x = torch.rand((b)) * 0.05 + 0.05
            shelf_y = (torch.rand((b)) - 0.5) * 0.15 + 0.25
            
            # Constrain Yaw: +/- 30 deg instead of 45 to make approach easier
            shelf_yaw = torch.rand((b)) * np.pi / 6
            shelf_yaw = np.pi / 2 - shelf_yaw
            
            # FIX: Increased gap range [0.14, 0.20]
            shelf_gap = torch.rand((b)) * 0.06 + 0.14
            bottom_z = 0.2
            
            # Rotations
            zeros = torch.zeros(b, device=self.device)
            euler_angles = torch.stack([zeros, zeros, shelf_yaw], dim=1)
            rot_mat = rotation_conversions.euler_angles_to_matrix(euler_angles, "XYZ")
            shelf_q = rotation_conversions.matrix_to_quaternion(rot_mat)
            
            shelf_pos = torch.stack([shelf_x, shelf_y, torch.zeros(b, device=self.device)], dim=1)

            # --- 2. Calculate Part Poses ---
            
            # Bottom Plate
            p_bot_local = torch.zeros((b, 3), device=self.device)
            p_bot_local[:, 2] = bottom_z
            p_bot_global = rotation_conversions.quaternion_apply(shelf_q, p_bot_local) + shelf_pos
            self.shelf_parts[0].set_pose(Pose.create_from_pq(p_bot_global, shelf_q))
            
            # Top Plate
            top_z = shelf_gap + bottom_z + 0.04
            p_top_local = torch.zeros((b, 3), device=self.device)
            p_top_local[:, 2] = top_z
            p_top_global = rotation_conversions.quaternion_apply(shelf_q, p_top_local) + shelf_pos
            self.shelf_parts[1].set_pose(Pose.create_from_pq(p_top_global, shelf_q))
            
            # Back Plate
            p_back_local = torch.zeros((b, 3), device=self.device)
            # Offset = half_depth + thickness
            p_back_local[:, 0] = self.shelf_half_depth + 0.02 
            p_back_local[:, 2] = (top_z + bottom_z) / 2 
            p_back_global = rotation_conversions.quaternion_apply(shelf_q, p_back_local) + shelf_pos
            self.shelf_parts[2].set_pose(Pose.create_from_pq(p_back_global, shelf_q))

            # --- 3. Place Cube ---
            cube_local = torch.zeros((b, 3), device=self.device)
            # Randomize within reduced depth
            cube_local[:, 0] = (torch.rand(b) - 0.5) * (self.shelf_half_depth * 0.8) 
            cube_local[:, 1] = (torch.rand(b) - 0.5) * 0.2 
            cube_local[:, 2] = bottom_z + 0.02 + self.cube_half_size
            
            cube_global = rotation_conversions.quaternion_apply(shelf_q, cube_local) + shelf_pos
            self.cube.set_pose(Pose.create_from_pq(cube_global, shelf_q))

            # --- 4. Goal ---
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, 0] = -0.1
            goal_xyz[:, 1] = 0.0
            goal_xyz[:, 2] = 1e-3
            # self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))
            self.goal_site.set_pose(
                Pose.create_from_pq(
                    p=goal_xyz,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )

    def get_obstacles_info(self):
        obstacles_info = []
        
        # 遍历环境中的架子部件 (Bottom, Top, Back)
        for i in range(len(self.shelf_parts)):
            part = self.shelf_parts[i]
            extent = self.obstacle_half_sizes[i]  # 预存的半尺寸列表
            
            current_pose = part.pose.raw_pose
            
            obs_info = {
                # 这里的 pose.p 和 pose.q 在 ManiSkill 封装下通常支持 Batch
                'center': current_pose[:, :3], # (B, 3)
                'quat': current_pose[:, 3:],   # (B, 4) [w, x, y, z]
                'extent': torch.tensor(extent, device=current_pose.device).expand(current_pose.shape[0], 3) # (B, 3) 动态读取的尺寸
            }
            obstacles_info.append(obs_info)
            
        return obstacles_info

# =============================================================================
# Task 3: PickBehindBarrier (Barrier Picking)
# =============================================================================

@register_env("PickBehindBarrier-v1", max_episode_steps=50)
class PickBehindBarrierEnv(PickCubeEnv):
    """
    Task Description:
    Obstacle between robot and object.
    Fixed: Moved barrier to positive X to prevent initial overlap with robot.
    """
    goal_radius = GOAL_RADIUS
    
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        if "harder" in kwargs:
            self.harder = kwargs["harder"]
            kwargs.pop("harder")
        else:
            self.harder = False
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)
        self.goal_thresh = self.goal_radius

    @property
    def _default_human_render_camera_configs(self):
        self.human_cam_eye_pos[0] = 0.1
        self.human_cam_eye_pos[1] = 0.8
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        
        self.barrier_half_height = 0.15
        self.barrier_half_size = [0.025, 0.3, self.barrier_half_height]
        barrier_mat = create_colored_material([0.3, 0.3, 0.3, 1])

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=self.barrier_half_size)
        builder.add_box_visual(half_size=self.barrier_half_size, material=barrier_mat)
        builder.initial_pose = sapien.Pose(p=[0, 0, -10])
        self.barrier = builder.build_kinematic(name="barrier")

        self.goal_site = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_site",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0, 1e-3]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # --- 1. Randomize Barrier ---
            # FIX: Move barrier to [0.0, 0.1] range (Center to slightly Front)
            # Previous was [-0.25, -0.15] which is too close to robot base (~-0.6)
            barrier_x = torch.rand(b) * 0.05 - 0.1
            
            if self.harder:
                target_height = torch.ones((b, 1)) * 0.32
            else:
                target_height = torch.rand((b, 1)) * 0.1 + 0.15
            barrier_z = target_height - self.barrier_half_height
            
            barrier_pos = torch.zeros((b, 3))
            barrier_pos[:, 0] = barrier_x
            barrier_pos[:, 1] = (torch.rand(b) - 0.5) * 0.2
            barrier_pos[:, 2] = barrier_z[:, 0]
            
            self.barrier.set_pose(Pose.create_from_pq(barrier_pos))

            # --- 2. Place Cube (Behind Barrier) ---
            cube_xyz = torch.zeros((b, 3))
            # Place cube 20-30cm behind the barrier
            if self.harder:
                cube_xyz[:, 0] = barrier_pos[:, 0] + 0.19
            else:
                cube_xyz[:, 0] = barrier_pos[:, 0] + 0.15 + torch.rand(b) * 0.05
            cube_xyz[:, 1] = (torch.rand(b) - 0.5) * 0.2
            cube_xyz[:, 2] = self.cube_half_size
            
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(cube_xyz, qs))

            # --- 3. Goal (In front of barrier) ---
            goal_xyz = torch.zeros((b, 3))
            # Goal close to robot side
            goal_xyz[:, 0] = barrier_pos[:, 0] - 0.15
            goal_xyz[:, 1] = 0
            goal_xyz[:, 2] = 1e-3
            # self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))
            self.goal_site.set_pose(
                Pose.create_from_pq(
                    p=goal_xyz,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )

    def get_obstacles_info(self):
        obstacles_info = []
        
        # 遍历环境中的障碍物 (Barrier)
        part = self.barrier
        extent = self.barrier_half_size  # 预存的半尺寸列表
        
        # # === WORLD FRAME ===
        # current_pose = part.pose.raw_pose
        # === ROOT FRAME ===
        current_pose_world_frame = part.pose
        current_pose_root_frame = self.agent.robot.get_pose().inv() * current_pose_world_frame
        current_pose = current_pose_root_frame.raw_pose  # (B, 7) [x, y, z, w, x, y, z]
        
        obs_info = {
            # 这里的 pose.p 和 pose.q 在 ManiSkill 封装下通常支持 Batch
            'center': current_pose[:, :3], # (B, 3)
            'quat': current_pose[:, 3:],   # (B, 4) [w, x, y, z]
            'extent': torch.tensor(extent, device=current_pose.device).expand(current_pose.shape[0], 3) # (B, 3) 动态读取的尺寸
        }
        obstacles_info.append(obs_info)
            
        return obstacles_info