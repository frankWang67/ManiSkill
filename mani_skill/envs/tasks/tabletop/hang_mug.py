from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import sapien
import torch
import trimesh

from mani_skill.agents.robots import (
    FloatingRobotiq2F85GripperWristCamera,
    PandaRobotiqWristCamera,
    UR5RobotiqWristCamera,
    XArm6RobotiqWristCamera,
    XArm7RobotiqWristCamera,
)
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.geometry.geometry import transform_points
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array


@register_env("HangMug-v1", max_episode_steps=120, asset_download_ids=["ycb"])
class HangMugEnv(BaseEnv):
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

    # RoboTwin initial poses used in hanging_mug.py
    mug_base_quat = np.array([0.7071068, 0.7071068, 0.0, 0.0], dtype=np.float32)
    rack_base_quat = np.array([-0.22, -0.22, 0.67, 0.67], dtype=np.float32)
    bottle_half_height = 0.06
    bottle_radius = 0.018
    blocker_half_size = [0.03, 0.05, 0.12]
    hang_support_inset = 0.006
    hang_position_tolerance = 0.03
    hang_success_position_tolerance = 0.075
    hang_elevation_tolerance = 0.06
    success_settle_steps = 12

    def __init__(
        self,
        *args,
        robot_uids="panda_robotiq_wristcam",
        robot_init_qpos_noise=0.02,
        harder: bool = False,
        reconfiguration_freq=None,
        num_envs=1,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.harder = harder
        self._rt_objects_root = Path("/home/wshf/RoboTwin/assets/objects")
        self._mug_model_id = 0
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
        pose = sapien_utils.look_at(eye=[0.62, 0.72, 0.6], target=[0.05, 0.0, 0.2])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.522, 0, 0]))

    def _asset_paths(self, modelname: str, model_id: int):
        model_dir = self._rt_objects_root / modelname
        if not model_dir.exists():
            raise FileNotFoundError(
                f"RoboTwin object assets not found: {model_dir}. "
                "Please extract from /home/wshf/RoboTwin/assets/objects.zip."
            )
        visual = model_dir / "visual" / f"base{model_id}.glb"
        collision = model_dir / "collision" / f"base{model_id}.glb"
        meta = model_dir / f"model_data{model_id}.json"
        if not visual.exists() or not collision.exists() or not meta.exists():
            raise FileNotFoundError(
                f"Missing files for {modelname} model_id={model_id}: "
                f"{visual}, {collision}, {meta}"
            )
        return visual, collision, meta

    def _build_mesh_actor(
        self,
        visual_file: Path,
        collision_file: Path,
        scale: list[float],
        name: str,
        dynamic: bool,
        density: float = 100.0,
    ):
        b = self.scene.create_actor_builder()
        if dynamic:
            b.set_physx_body_type("dynamic")
            b.add_multiple_convex_collisions_from_file(
                filename=str(collision_file),
                scale=scale,
                density=density,
            )
        else:
            b.set_physx_body_type("static")
            b.add_multiple_convex_collisions_from_file(
                filename=str(collision_file),
                scale=scale,
            )
        b.add_visual_from_file(filename=str(visual_file), scale=scale)
        b.initial_pose = sapien.Pose(p=[0, 0, -10])
        return b.build(name=name)

    def _infer_rack_branch_geometry_local(self, rack_visual_file: Path, branch_axis_local: np.ndarray):
        # The rack has a single short hanging branch. Use the annotated
        # functional point as the tip and infer the branch root by tracing a
        # short cylinder backward into the mesh along the branch axis.
        mesh = trimesh.load(str(rack_visual_file), force="scene").to_geometry()
        v = mesh.vertices
        tip = self._rack_function_local.cpu().numpy()
        axis = branch_axis_local / (np.linalg.norm(branch_axis_local) + 1e-8)
        rel = v - tip
        proj = rel @ axis
        radial = np.linalg.norm(rel - np.outer(proj, axis), axis=1)
        mask = radial < 0.02
        inward_proj = -0.012
        if np.any(mask):
            min_proj = float(proj[mask].min())
            max_proj = float(proj[mask].max())
            inward_proj = min_proj if abs(min_proj) >= abs(max_proj) else max_proj
        if abs(inward_proj) < 0.008:
            inward_proj = -0.012 if inward_proj <= 0 else 0.012
        root = tip + axis * inward_proj
        return (
            torch.tensor(tip, dtype=torch.float32),
            torch.tensor(root, dtype=torch.float32),
            torch.tensor(axis, dtype=torch.float32),
        )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        mug_visual, mug_collision, mug_meta_file = self._asset_paths(
            "039_mug", self._mug_model_id
        )
        rack_visual, rack_collision, rack_meta_file = self._asset_paths("040_rack", 0)
        self._mug_meta = load_json(mug_meta_file)
        self._rack_meta = load_json(rack_meta_file)

        self.mug = self._build_mesh_actor(
            mug_visual,
            mug_collision,
            self._mug_meta["scale"],
            name="mug",
            dynamic=True,
            density=120.0,
        )
        self.hanger = self._build_mesh_actor(
            rack_visual,
            rack_collision,
            self._rack_meta["scale"],
            name="hanger",
            dynamic=False,
        )

        # Mug handle functional point (id=0) from RoboTwin metadata.
        mug_handle_tf = np.array(self._mug_meta["functional_matrix"][0], dtype=np.float32)
        mug_scale = np.array(self._mug_meta["scale"], dtype=np.float32)
        self.mug_handle_local_pos = torch.tensor(
            mug_handle_tf[:3, 3] * mug_scale, dtype=torch.float32
        )
        self.mug_handle_local_rot = torch.tensor(mug_handle_tf[:3, :3], dtype=torch.float32)
        self.mug_handle_axis_local = torch.tensor(mug_handle_tf[:3, 0], dtype=torch.float32)

        mug_center = np.array(self._mug_meta["center"], dtype=np.float32)
        mug_extents = np.array(self._mug_meta["extents"], dtype=np.float32)
        mug_min_z = (mug_center[2] - mug_extents[2] * 0.5) * mug_scale[2]
        self.mug_bottom_to_com = float(-mug_min_z)

        # Rack functional point (id=0) from RoboTwin metadata.
        rack_fun_tf = np.array(self._rack_meta["functional_matrix"][0], dtype=np.float32)
        rack_scale = np.array(self._rack_meta["scale"], dtype=np.float32)
        self._rack_function_local = torch.tensor(
            rack_fun_tf[:3, 3] * rack_scale, dtype=torch.float32
        )
        self.rack_function_local_rot = torch.tensor(
            rack_fun_tf[:3, :3], dtype=torch.float32
        )
        self.branch_tip_local, self.branch_root_local, self.branch_axis_local = (
            self._infer_rack_branch_geometry_local(
                rack_visual,
                rack_fun_tf[:3, 0],
            )
        )

        rack_center = np.array(self._rack_meta["center"], dtype=np.float32)
        rack_extents = np.array(self._rack_meta["extents"], dtype=np.float32)
        rack_min_z = (rack_center[2] - rack_extents[2] * 0.5) * rack_scale[2]
        self.rack_bottom_to_com = float(-rack_min_z)

        self._build_harder_obstacles()
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=0.01,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _build_harder_obstacles(self):
        b = self.scene.create_actor_builder()
        b.add_box_collision(half_size=self.blocker_half_size)
        b.add_box_visual(half_size=self.blocker_half_size)
        b.initial_pose = sapien.Pose(p=[0, 0, -10])
        self.branch_blocker = b.build_kinematic(name="branch_blocker")

        b = self.scene.create_actor_builder()
        b.add_cylinder_collision(radius=self.bottle_radius, half_length=self.bottle_half_height)
        b.add_cylinder_visual(radius=self.bottle_radius, half_length=self.bottle_half_height)
        b.initial_pose = sapien.Pose(p=[0, 0, -10])
        self.blocker_bottle = b.build_kinematic(name="blocker_bottle")

    def _branch_target_points_world(self, env_idx: Optional[torch.Tensor] = None):
        if env_idx is None:
            T = self.hanger.pose.to_transformation_matrix()
            b = self.num_envs
        else:
            T = self.hanger.pose[env_idx].to_transformation_matrix()
            b = len(env_idx)
        tip = self.branch_tip_local.to(self.device)[None, :].repeat(b, 1)
        root = self.branch_root_local.to(self.device)[None, :].repeat(b, 1)
        return transform_points(T, tip), transform_points(T, root)

    def _branch_support_points_world(self, env_idx: Optional[torch.Tensor] = None):
        branch_tip, branch_root = self._branch_target_points_world(env_idx)
        inward_axis = branch_root - branch_tip
        inward_axis = inward_axis / (torch.linalg.norm(inward_axis, dim=1, keepdim=True) + 1e-8)
        support = branch_tip + inward_axis * self.hang_support_inset
        branch_dir = -inward_axis
        return support, branch_tip, branch_root, branch_dir

    def _mug_handle_center_world(self, env_idx: Optional[torch.Tensor] = None):
        if env_idx is None:
            T = self.mug.pose.to_transformation_matrix()
            b = self.num_envs
        else:
            T = self.mug.pose[env_idx].to_transformation_matrix()
            b = len(env_idx)
        p = self.mug_handle_local_pos.to(self.device)[None, :].repeat(b, 1)
        return transform_points(T, p)

    def _update_released_hang_counter(self, released_hang_candidate: torch.Tensor):
        if not hasattr(self, "released_hang_steps"):
            self.released_hang_steps = torch.zeros(
                (self.num_envs,), dtype=torch.int32, device=self.device
            )
            self._released_hang_last_step = torch.full(
                (self.num_envs,), -1, dtype=torch.int64, device=self.device
            )

        current_steps = self.elapsed_steps.to(device=self.device, dtype=torch.int64)
        needs_update = self._released_hang_last_step != current_steps
        if not torch.any(needs_update):
            return

        increment_mask = needs_update & released_hang_candidate
        reset_mask = needs_update & (~released_hang_candidate)
        self.released_hang_steps[increment_mask] += 1
        self.released_hang_steps[reset_mask] = 0
        self._released_hang_last_step[needs_update] = current_steps[needs_update]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Rack spawn (RoboTwin-style ranges).
            rack_xyz = torch.zeros((b, 3), device=self.device)
            rack_xyz[:, 0] = torch.rand((b), device=self.device) * 0.05 + 0.07
            rack_xyz[:, 1] = torch.rand((b), device=self.device) * 0.04 + 0.20
            rack_xyz[:, 2] = self.rack_bottom_to_com + 0.001

            base_q = common.to_tensor(
                np.tile(self.rack_base_quat[None, :], (b, 1)), device=self.device
            )
            euler = torch.zeros((b, 3), device=self.device)
            euler[:, 1] = torch.rand((b), device=self.device) * 0.4 - 0.2
            delta_q = rotation_conversions.matrix_to_quaternion(
                rotation_conversions.euler_angles_to_matrix(euler, "XYZ")
            )
            rack_q = rotation_conversions.quaternion_multiply(base_q, delta_q)
            self.hanger.set_pose(Pose.create_from_pq(rack_xyz, rack_q))

            # Mug spawn (RoboTwin-style ranges).
            mug_xyz = torch.zeros((b, 3), device=self.device)
            mug_xyz[:, 0] = torch.rand((b), device=self.device) * 0.10 - 0.20
            mug_xyz[:, 1] = torch.rand((b), device=self.device) * 0.10 - 0.05
            # mug_xyz[:, 2] = self.mug_bottom_to_com + 0.001

            mug_base_q = common.to_tensor(
                np.tile(self.mug_base_quat[None, :], (b, 1)), device=self.device
            )
            euler = torch.zeros((b, 3), device=self.device)
            euler[:, 1] = torch.rand((b), device=self.device) * 3.14 - 1.57
            delta_q = rotation_conversions.matrix_to_quaternion(
                rotation_conversions.euler_angles_to_matrix(euler, "XYZ")
            )
            mug_q = rotation_conversions.quaternion_multiply(mug_base_q, delta_q)
            self.mug.set_pose(Pose.create_from_pq(mug_xyz, mug_q))
            self.mug.set_linear_velocity(torch.zeros((b, 3), device=self.device))
            self.mug.set_angular_velocity(torch.zeros((b, 3), device=self.device))

            if not hasattr(self, "blocked_branch"):
                self.blocked_branch = torch.full(
                    (self.num_envs,), -1, dtype=torch.int64, device=self.device
                )
            blocked_branch = torch.full((b,), -1, dtype=torch.int64, device=self.device)

            hide_xyz = torch.zeros((b, 3), device=self.device)
            hide_xyz[:, 2] = -10
            if self.harder:
                branch_tip, branch_root = self._branch_target_points_world(env_idx)
                axis = branch_root - branch_tip
                axis = axis / (torch.linalg.norm(axis, dim=1, keepdim=True) + 1e-8)

                blocker_pos = branch_tip + axis * 0.012
                blocker_pos[:, 2] += 0.02
                blocker_q = common.to_tensor(
                    np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (b, 1)),
                    device=self.device,
                )
                self.branch_blocker.set_pose(Pose.create_from_pq(blocker_pos, blocker_q))

                bottle_pos = branch_tip + axis * 0.03
                bottle_pos[:, 2] = self.bottle_half_height + 0.002
                self.blocker_bottle.set_pose(Pose.create_from_pq(bottle_pos, blocker_q))
            else:
                self.branch_blocker.set_pose(Pose.create_from_pq(hide_xyz))
                self.blocker_bottle.set_pose(Pose.create_from_pq(hide_xyz))

            self.blocked_branch[env_idx] = blocked_branch
            if not hasattr(self, "released_hang_steps"):
                self.released_hang_steps = torch.zeros(
                    (self.num_envs,), dtype=torch.int32, device=self.device
                )
                self._released_hang_last_step = torch.full(
                    (self.num_envs,), -1, dtype=torch.int64, device=self.device
                )
            self.released_hang_steps[env_idx] = 0
            self._released_hang_last_step[env_idx] = -1

            # Goal marker is the actual hanging support point on the usable branch.
            target, _, _, _ = self._branch_support_points_world(env_idx)
            self.goal_site.set_pose(Pose.create_from_pq(target))

            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()
                self.scene.px.step()
                self.scene._gpu_fetch_all()

    def evaluate(self):
        mug_handle = self._mug_handle_center_world()
        branch_tip, branch_root = self._branch_target_points_world()
        _, _, _, branch_dir = self._branch_support_points_world()
        mug_com = self.mug.pose.p

        # Treat the whole branch segment (tip -> root) as valid hanging support.
        branch_vec = branch_root - branch_tip
        branch_len_sq = torch.sum(branch_vec * branch_vec, dim=1, keepdim=True).clamp_min(1e-8)
        proj = torch.sum((mug_handle - branch_tip) * branch_vec, dim=1, keepdim=True) / branch_len_sq
        proj = torch.clamp(proj, 0.0, 1.0)
        branch_support = branch_tip + proj * branch_vec
        best_dist = torch.linalg.norm(mug_handle - branch_support, axis=1)
        nearest_branch = torch.zeros((self.num_envs,), dtype=torch.int64, device=self.device)
        allowed = torch.ones_like(nearest_branch, dtype=torch.bool)
        # Success checking is intentionally looser than dense reward shaping:
        # a mug can be validly hung while the annotated handle point is not
        # exactly centered on the branch axis (e.g. when it settles near root).
        is_handle_aligned = best_dist < self.hang_success_position_tolerance

        contact_force = torch.linalg.norm(
            self.scene.get_pairwise_contact_forces(self.hanger, self.mug), axis=1
        )
        has_contact = contact_force > 0.02

        table_contact_force = torch.linalg.norm(
            self.scene.get_pairwise_contact_forces(self.table_scene.table, self.mug), axis=1
        )
        has_table_support = table_contact_force > 0.05

        obstacle_contact_force = torch.zeros_like(table_contact_force)
        if self.harder:
            blocker_contact_force = torch.linalg.norm(
                self.scene.get_pairwise_contact_forces(self.branch_blocker, self.mug),
                axis=1,
            )
            bottle_contact_force = torch.linalg.norm(
                self.scene.get_pairwise_contact_forces(self.blocker_bottle, self.mug),
                axis=1,
            )
            obstacle_contact_force = torch.maximum(
                blocker_contact_force, bottle_contact_force
            )
        has_obstacle_support = obstacle_contact_force > 0.05
        is_unobstructed = (~has_table_support) & (~has_obstacle_support)

        is_grasped = self.agent.is_grasping(self.mug)
        is_robot_static = self.agent.is_static(0.15)
        is_mug_static = self.mug.is_static(lin_thresh=0.01, ang_thresh=0.2)
        min_branch_z = torch.minimum(branch_tip[:, 2], branch_root[:, 2])
        is_elevated = mug_handle[:, 2] > (min_branch_z - self.hang_elevation_tolerance)
        table_top_z = self.table_scene.table.pose.p[:, 2] + self.table_scene.table_height
        is_mug_above_table = mug_com[:, 2] > (table_top_z + 0.08)
        is_mug_below_branch = mug_com[:, 2] < (branch_support[:, 2] - 0.02)
        # Contact force between hanger/mug can be numerically zero for some
        # stable hooked states, so success should primarily rely on geometry
        # and stability rather than force threshold alone.
        is_hung = (
            is_handle_aligned
            & allowed
            & is_elevated
            & is_unobstructed
            & is_mug_above_table
            & is_mug_below_branch
        )
        released_hang_candidate = is_hung & (~is_grasped) & is_robot_static & is_mug_static
        self._update_released_hang_counter(released_hang_candidate)
        is_release_stable = self.released_hang_steps >= self.success_settle_steps
        success = released_hang_candidate & is_release_stable

        return {
            "success": success,
            "is_hung": is_hung,
            "released_hang_candidate": released_hang_candidate,
            "is_release_stable": is_release_stable,
            "released_hang_steps": self.released_hang_steps,
            "is_handle_aligned": is_handle_aligned,
            "has_contact": has_contact,
            "allowed_branch": allowed,
            "blocked_branch": self.blocked_branch,
            "nearest_branch": nearest_branch,
            "is_grasped": is_grasped,
            "is_robot_static": is_robot_static,
            "is_mug_static": is_mug_static,
            "has_table_support": has_table_support,
            "has_obstacle_support": has_obstacle_support,
            "is_unobstructed": is_unobstructed,
            "is_mug_above_table": is_mug_above_table,
            "is_mug_below_branch": is_mug_below_branch,
            "is_elevated": is_elevated,
            "mug_handle_pos": mug_handle,
            "mug_com_pos": mug_com,
            "branch_neg_pos": branch_support,
            "branch_pos_pos": branch_root,
            "branch_neg_tip_pos": branch_tip,
            "branch_pos_tip_pos": branch_tip,
            "branch_dir": branch_dir,
            "branch_dist": best_dist,
            "table_contact_force": table_contact_force,
            "obstacle_contact_force": obstacle_contact_force,
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            mug_handle_pos=info["mug_handle_pos"],
            blocked_branch=info["blocked_branch"],
            nearest_branch=info["nearest_branch"],
            is_hung=info["is_hung"],
            branch_dist=info["branch_dist"],
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                mug_pose=self.mug.pose.raw_pose,
                hanger_pose=self.hanger.pose.raw_pose,
                target_branch_pos=self.goal_site.pose.p,
                branch_neg_pos=info["branch_neg_pos"],
                branch_pos_pos=info["branch_pos_pos"],
                branch_neg_tip_pos=info["branch_neg_tip_pos"],
                branch_pos_tip_pos=info["branch_pos_tip_pos"],
            )
        return obs

    def get_hang_pose_and_direction(self, env_idx: Optional[int] = None):
        idx = 0 if env_idx is None else int(env_idx)
        target, tip, root, branch_dir = self._branch_support_points_world()
        return {
            "target": target[idx],
            "tip": tip[idx],
            "root": root[idx],
            "approach": branch_dir[idx],
        }

    def get_hang_goal_pose(self, env_idx: Optional[int] = None):
        idx = 0 if env_idx is None else int(env_idx)
        target, tip, root, branch_axis = self._branch_support_points_world()
        target = target[idx]
        tip = tip[idx]
        root = root[idx]
        branch_axis = branch_axis[idx]
        hanger_rot = self.hanger.pose.to_transformation_matrix()[idx, :3, :3]
        rack_fun_rot = hanger_rot @ self.rack_function_local_rot.to(self.device)
        mug_rot = rack_fun_rot @ self.mug_handle_local_rot.to(self.device).transpose(0, 1)
        mug_q = rotation_conversions.matrix_to_quaternion(mug_rot[None, :, :])[0]

        handle_local = common.to_tensor(self.mug_handle_local_pos, device=self.device)
        handle_world_offset = mug_rot @ handle_local
        mug_p = target - handle_world_offset

        return {
            "mug_pos": mug_p,
            "mug_quat": mug_q,
            "branch_target": target,
            "branch_tip": tip,
            "branch_root": root,
            "branch_dir": branch_axis,
        }

    def get_obstacles_info(self):
        if not self.harder:
            return []
        out = []
        for actor, extent in [
            (self.branch_blocker, self.blocker_half_size),
            (
                self.blocker_bottle,
                [self.bottle_radius, self.bottle_radius, self.bottle_half_height],
            ),
        ]:
            raw_pose = actor.pose.raw_pose
            out.append(
                {
                    "center": raw_pose[:, :3],
                    "quat": raw_pose[:, 3:],
                    "extent": torch.tensor(
                        extent, dtype=torch.float32, device=raw_pose.device
                    ).expand(raw_pose.shape[0], 3),
                }
            )
        return out

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        tcp_to_mug = torch.linalg.norm(self.agent.tcp.pose.p - self.mug.pose.p, axis=1)
        reaching_reward = 1 - torch.tanh(4 * tcp_to_mug)
        reward = reaching_reward
        reward += info["is_grasped"].float()
        reward += (1 - torch.tanh(10 * info["branch_dist"])) * info["is_grasped"].float()
        reward += info["is_hung"].float() * 2.0
        reward += (~info["is_grasped"]).float() * info["is_hung"].float()
        reward[info["success"]] = 6.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 6.0
