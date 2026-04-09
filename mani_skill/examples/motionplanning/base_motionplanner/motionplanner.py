import mplib
import numpy as np
import sapien
import trimesh

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.pose import to_sapien_pose


class BaseMotionPlanningSolver:

    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,  # TODO mplib doesn't support robot base being anywhere but 0
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
    ):
        self.env = env
        self.base_env: BaseEnv = env.unwrapped
        self.env_agent: BaseAgent = self.base_env.agent
        self.robot = self.env_agent.robot
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits

        self.base_pose = to_sapien_pose(base_pose)

        self.planner = self.setup_planner()
        self.control_mode = self.base_env.control_mode
        self.active_joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        self.arm_joint_indices, self.non_arm_joint_indices = self._infer_arm_joint_indices()
        self.ik_reference_qpos = self.robot.get_qpos().cpu().numpy()[0].copy()

        self.debug = debug
        self.vis = vis
        self.print_env_info = print_env_info
        
        self.elapsed_steps = 0

        self.use_point_cloud = False
        self.collision_pts_changed = False
        self.all_collision_pts = None

    def _infer_arm_joint_indices(self):
        controller = getattr(self.env_agent, "controller", None)
        controllers = getattr(controller, "controllers", None)
        arm_joint_names = None
        if isinstance(controllers, dict) and "arm" in controllers:
            arm_config = getattr(controllers["arm"], "config", None)
            joint_names = getattr(arm_config, "joint_names", None)
            if joint_names is not None:
                arm_joint_names = set(joint_names)

        if arm_joint_names is None:
            arm_dof = len(self.planner.joint_vel_limits)
            return list(range(arm_dof)), []

        arm_joint_indices = [
            i for i, name in enumerate(self.active_joint_names) if name in arm_joint_names
        ]
        if not arm_joint_indices:
            arm_dof = len(self.planner.joint_vel_limits)
            return list(range(arm_dof)), []

        non_arm_joint_indices = [
            i for i in range(len(self.active_joint_names)) if i not in arm_joint_indices
        ]
        return arm_joint_indices, non_arm_joint_indices

    def _get_current_qpos(self):
        return self.robot.get_qpos().cpu().numpy()[0]

    def _get_current_arm_qpos(self):
        qpos = self._get_current_qpos()
        arm_dof = len(self.planner.joint_vel_limits)
        return qpos[:arm_dof]

    def _get_ik_seed_qpos(self):
        qpos = self._get_current_qpos().copy()
        if self.non_arm_joint_indices:
            qpos[self.non_arm_joint_indices] = self.ik_reference_qpos[
                self.non_arm_joint_indices
            ]
        return qpos

    def render_wait(self):
        if not self.vis or not self.debug:
            return
        print("Press [c] to continue")
        viewer = self.base_env.render_human()
        while True:
            if viewer.window.key_down("c"):
                break
            self.base_env.render_human()

    def setup_planner(self):
        move_group = self.MOVE_GROUP if hasattr(self, "MOVE_GROUP") else "eef"
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        planner = mplib.Planner(
            urdf=self.env_agent.urdf_path,
            srdf=self.env_agent.urdf_path.replace(".urdf", ".srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group=move_group,
        )
        planner.set_base_pose(np.hstack([self.base_pose.p, self.base_pose.q]))
        planner.joint_vel_limits = np.asarray(planner.joint_vel_limits) * self.joint_vel_limits
        planner.joint_acc_limits = np.asarray(planner.joint_acc_limits) * self.joint_acc_limits
        return planner

    def _update_grasp_visual(self, target: sapien.Pose) -> None:
        return None

    def _transform_pose_for_planning(self, target: sapien.Pose) -> sapien.Pose:
        return target

    def follow_path(self, result, refine_steps: int = 0):
        n_step = result["position"].shape[0]
        for i in range(n_step + refine_steps):
            qpos = result["position"][min(i, n_step - 1)]
            if self.control_mode == "pd_joint_pos_vel":
                qvel = result["velocity"][min(i, n_step - 1)]
                action = np.hstack([qpos, qvel])
            else:
                action = np.hstack([qpos])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

        
    def move_to_pose_with_RRTStar(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        pose = to_sapien_pose(pose)
        self._update_grasp_visual(pose)
        pose = self._transform_pose_for_planning(pose)
        result = self.planner.plan_qpos_to_pose(
            np.concatenate([pose.p, pose.q]),
            self._get_ik_seed_qpos(),
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
            rrt_range=0.0,
            planning_time=1,
            planner_name="RRTstar",
            wrt_world=True,
        )
        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def move_to_pose_with_RRTConnect(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        pose = to_sapien_pose(pose)
        self._update_grasp_visual(pose)
        pose = self._transform_pose_for_planning(pose)
        result = self.planner.plan_qpos_to_pose(
            np.concatenate([pose.p, pose.q]),
            self._get_ik_seed_qpos(),
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
            rrt_range=0.0,
            planning_time=1,
            planner_name="RRTConnect",
            wrt_world=True,
        )
        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def move_to_pose_with_screw(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        pose = to_sapien_pose(pose)
        # try screw two times before giving up
        self._update_grasp_visual(pose)
        pose = self._transform_pose_for_planning(pose)
        result = self.planner.plan_screw(
            np.concatenate([pose.p, pose.q]),
            self._get_current_arm_qpos(),
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
        )
        if result["status"] != "Success":
            result = self.planner.plan_screw(
                np.concatenate([pose.p, pose.q]),
                self._get_current_arm_qpos(),
                time_step=self.base_env.control_timestep,
                use_point_cloud=self.use_point_cloud,
            )
            if result["status"] != "Success":
                # print(result["status"])
                self.render_wait()
                return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def add_box_collision(self, extents: np.ndarray, pose: sapien.Pose):
        self.use_point_cloud = True
        box = trimesh.creation.box(extents, transform=pose.to_transformation_matrix())
        pts, _ = trimesh.sample.sample_surface(box, 256)
        if self.all_collision_pts is None:
            self.all_collision_pts = pts
        else:
            self.all_collision_pts = np.vstack([self.all_collision_pts, pts])
        self.planner.update_point_cloud(self.all_collision_pts)

    def add_collision_pts(self, pts: np.ndarray):
        if self.all_collision_pts is None:
            self.all_collision_pts = pts
        else:
            self.all_collision_pts = np.vstack([self.all_collision_pts, pts])
        self.planner.update_point_cloud(self.all_collision_pts)

    def clear_collisions(self):
        self.all_collision_pts = None
        self.use_point_cloud = False

    def close(self):
        pass
