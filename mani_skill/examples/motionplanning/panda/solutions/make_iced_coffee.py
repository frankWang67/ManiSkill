import numpy as np
import sapien

from mani_skill.envs.tasks import MakeIcedCoffeeEnv
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)

JOINT_VEL_LIMITS = 0.5
JOINT_ACC_LIMITS = 0.5
FINGER_LENGTH = 0.025
PRE_GRASP_Z = 0.08
LIFT_DELTA_Z = 0.12
MIN_LIFT_Z = 0.26
ABOVE_CUP_Z = 0.10

def _move(planner: PandaArmMotionPlanningSolver, pose: sapien.Pose):
    res = planner.move_to_pose_with_screw(pose)
    if res == -1:
        res = planner.move_to_pose_with_RRTConnect(pose)
    return res

def solve(env: MakeIcedCoffeeEnv, seed=None, debug: bool = False, vis: bool = False):
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode

    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        joint_vel_limits=JOINT_VEL_LIMITS,
        joint_acc_limits=JOINT_ACC_LIMITS,
    )
    env = env.unwrapped

    # Match planner gripper states to this env's actual action-space bounds.
    gripper_ctrl = env.agent.controller.controllers["gripper_active"]
    planner.OPEN = float(gripper_ctrl.single_action_space.low[0])
    planner.CLOSED = float(gripper_ctrl.single_action_space.high[0])

    home_pose = env.agent.tcp.pose.sp

    approaching = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    obb = get_actor_obb(env.ice_cube)
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    grasp_pose = env.agent.build_grasp_pose(
        approaching, grasp_info["closing"], grasp_info["center"]
    )

    pre_grasp_pose = grasp_pose * sapien.Pose([0.0, 0.0, -PRE_GRASP_Z])
    lift_pose = sapien.Pose(
        p=np.array(
            [
                grasp_pose.p[0],
                grasp_pose.p[1],
                max(float(grasp_pose.p[2]) + LIFT_DELTA_Z, MIN_LIFT_Z),
            ],
            dtype=np.float64,
        ),
        q=grasp_pose.q,
    )

    goal_pos = env.goal_site.pose.p[0].detach().cpu().numpy()

    try:
        # 1) Get ready above the ice cube.
        planner.open_gripper(t=6)
        if _move(planner, pre_grasp_pose) == -1:
            return -1

        # 2) Move down to grasp and close gripper.
        if _move(planner, grasp_pose) == -1:
            return -1
        planner.close_gripper(t=12)

        # 3) Lift and transfer above the cup.
        if _move(planner, lift_pose) == -1:
            return -1
        # Use the post-lift orientation as transfer orientation to avoid
        # unnecessary branch switching that can make RRT fallback detour.
        transfer_q = env.agent.tcp.pose.q[0].detach().cpu().numpy()
        above_cup_pose = sapien.Pose(
            p=goal_pos + np.array([0.0, 0.0, ABOVE_CUP_Z], dtype=np.float64),
            q=transfer_q,
        )
        place_pose = sapien.Pose(p=goal_pos, q=transfer_q)
        if _move(planner, above_cup_pose) == -1:
            return -1

        # 4) Lower into the cup and release.
        if _move(planner, place_pose) == -1:
            return -1
        planner.open_gripper(t=12)

        # 5) Leave the cup and return home.
        if _move(planner, above_cup_pose) == -1:
            return -1
        res = _move(planner, home_pose)
        return res
    finally:
        planner.close()
