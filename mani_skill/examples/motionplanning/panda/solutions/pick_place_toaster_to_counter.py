import numpy as np
import sapien

from mani_skill.envs.tasks import PickPlaceToasterToCounterEnv
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)

ROTATE_25_DEG = -25 * np.pi / 180.0

def _move(planner: PandaArmMotionPlanningSolver, pose: sapien.Pose):
    res = planner.move_to_pose_with_screw(pose)
    if res == -1:
        res = planner.move_to_pose_with_RRTConnect(pose)
    return res


def solve(
    env: PickPlaceToasterToCounterEnv, seed=None, debug: bool = False, vis: bool = False
):
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
        joint_vel_limits=0.5,
        joint_acc_limits=0.5,
    )
    env = env.unwrapped

    gripper_ctrl = env.agent.controller.controllers["gripper_active"]
    planner.OPEN = float(gripper_ctrl.single_action_space.low[0])
    planner.CLOSED = float(gripper_ctrl.single_action_space.high[0])

    home_pose = env.agent.tcp.pose.sp

    # -------------------------------------------------------------------------- #
    # 1) Approach and grasp toast in toaster
    # -------------------------------------------------------------------------- #
    toast_pos = env.toast.pose.p[0].detach().cpu().numpy()
    grasp_center = toast_pos + env.toast_grasp_local_offset.astype(np.float64)

    rng = np.random.default_rng(seed)
    side_sign = 1.0 if rng.random() < 0.5 else -1.0  # +y: left, -y: right

    grasp_approaching = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    grasp_closing = np.array([-side_sign, 0.0, 0.0], dtype=np.float64)
    grasp_closing /= np.linalg.norm(grasp_closing)
    grasp_pose = env.agent.build_grasp_pose(grasp_approaching, grasp_closing, grasp_center)

    pre_grasp_pose = grasp_pose * sapien.Pose([0.0, 0.0, -0.07])
    planner.open_gripper(t=8)
    if _move(planner, pre_grasp_pose) == -1:
        planner.close()
        return -1
    if _move(planner, grasp_pose) == -1:
        planner.close()
        return -1
    planner.close_gripper(t=16)

    # Lift high enough so the toast fully clears the toaster before transfer.
    lift_mid_height = max(float(grasp_pose.p[2]) + 0.1, 0.25)
    lift_height = max(float(grasp_pose.p[2]) + 0.16, 0.32)
    lift_mid_pose = sapien.Pose(
        [grasp_pose.p[0], grasp_pose.p[1], lift_mid_height], grasp_pose.q
    )
    lift_pose = sapien.Pose([grasp_pose.p[0], grasp_pose.p[1], lift_height], grasp_pose.q)
    if _move(planner, lift_mid_pose) == -1:
        planner.close()
        return -1
    planner.close_gripper(t=6)
    if _move(planner, lift_pose) == -1:
        planner.close()
        return -1

    # -------------------------------------------------------------------------- #
    # 2) Choose left/right side and place toast horizontally on plate
    # -------------------------------------------------------------------------- #

    goal_pos = env.goal_site.pose.p[0].detach().cpu().numpy().copy()
    place_center = goal_pos.copy()
    # place_center[1] -= 0.10 * side_sign
    # place_center[2] += 0.055
    place_center[1] -= 0.05 * side_sign
    place_center[2] += 0.1

    # Horizontal approach direction makes the toast lie down at release.
    place_approaching = np.array([0.0, side_sign * np.cos(ROTATE_25_DEG), np.sin(ROTATE_25_DEG)], dtype=np.float64)
    place_closing = np.array([0.0, np.sin(ROTATE_25_DEG), -side_sign * np.cos(ROTATE_25_DEG)], dtype=np.float64)
    place_pose = env.agent.build_grasp_pose(place_approaching, place_closing, place_center)

    carry_pose = sapien.Pose([place_center[0], place_center[1], lift_height], grasp_pose.q)

    quit_center = place_center.copy()
    quit_center -= 0.15 * place_approaching  # move backward from place pose after release
    quit_pose = env.agent.build_grasp_pose(place_approaching, place_closing, quit_center)

    if _move(planner, carry_pose) == -1:
        planner.close()
        return -1
    if _move(planner, place_pose) == -1:
        planner.close()
        return -1

    planner.open_gripper(t=10)

    if _move(planner, quit_pose) == -1:
        planner.close()
        return -1

    # -------------------------------------------------------------------------- #
    # 3) Return to initial pose
    # -------------------------------------------------------------------------- #
    res = _move(planner, home_pose)
    if res == -1:
        planner.close()
        return -1
    planner.close()
    return res
