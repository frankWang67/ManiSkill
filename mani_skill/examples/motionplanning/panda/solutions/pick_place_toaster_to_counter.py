import numpy as np
import sapien

from mani_skill.envs.tasks import PickPlaceToasterToCounterEnv
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)

ROTATE_25_DEG = -35 * np.pi / 180.0
RRT_REPLAN_ATTEMPTS = 8
MAX_RRT_PLAN_STEPS = 140
MAX_RRT_CUM_JOINT_TRAVEL_BASE = 1.8
MAX_RRT_CUM_JOINT_TRAVEL_PER_M = 8.0
MAX_RRT_CUM_JOINT_TRAVEL_PER_RAD = 1.8
MAX_RRT_MAX_STEP = 0.12
MAX_RRT_DETOUR_RATIO = 3.0


def _to_sapien_pose(pose):
    if hasattr(pose, "sp"):
        return pose.sp
    return pose


def _quat_angle_distance(q0: np.ndarray, q1: np.ndarray) -> float:
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 = q0 / (np.linalg.norm(q0) + 1e-12)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    cos_half = np.clip(np.abs(np.dot(q0, q1)), 0.0, 1.0)
    return float(2.0 * np.arccos(cos_half))


def _path_metrics(result):
    qpos = np.asarray(result["position"], dtype=np.float64)
    if qpos.shape[0] <= 1:
        return dict(
            steps=int(qpos.shape[0]),
            cum_joint=0.0,
            direct_joint=0.0,
            max_step=0.0,
            detour_ratio=1.0,
        )

    step_norms = np.linalg.norm(np.diff(qpos, axis=0), axis=1)
    cum_joint = float(step_norms.sum())
    direct_joint = float(np.linalg.norm(qpos[-1] - qpos[0]))
    return dict(
        steps=int(qpos.shape[0]),
        cum_joint=cum_joint,
        direct_joint=direct_joint,
        max_step=float(step_norms.max()),
        detour_ratio=float(cum_joint / max(direct_joint, 1e-6)),
    )


def _plan_rrt_result(planner: PandaArmMotionPlanningSolver, pose: sapien.Pose):
    pose = _to_sapien_pose(pose)
    planner._update_grasp_visual(pose)
    pose = planner._transform_pose_for_planning(pose)
    return planner.planner.plan_qpos_to_pose(
        np.concatenate([pose.p, pose.q]),
        planner._get_ik_seed_qpos(),
        time_step=planner.base_env.control_timestep,
        use_point_cloud=planner.use_point_cloud,
        rrt_range=0.0,
        planning_time=1,
        planner_name="RRTConnect",
        wrt_world=True,
    )


def _rrt_plan_is_reasonable(
    planner: PandaArmMotionPlanningSolver,
    pose: sapien.Pose,
    result,
):
    metrics = _path_metrics(result)

    current_tcp_pose = _to_sapien_pose(getattr(planner.env_agent, "tcp_pose", None))
    if current_tcp_pose is None:
        current_tcp_pose = _to_sapien_pose(planner.env_agent.tcp.pose)
    target_pose = _to_sapien_pose(pose)

    cart_delta = float(
        np.linalg.norm(
            np.asarray(target_pose.p, dtype=np.float64)
            - np.asarray(current_tcp_pose.p, dtype=np.float64)
        )
    )
    rot_delta = _quat_angle_distance(
        np.asarray(target_pose.q, dtype=np.float64),
        np.asarray(current_tcp_pose.q, dtype=np.float64),
    )
    max_cum_joint = (
        MAX_RRT_CUM_JOINT_TRAVEL_BASE
        + MAX_RRT_CUM_JOINT_TRAVEL_PER_M * cart_delta
        + MAX_RRT_CUM_JOINT_TRAVEL_PER_RAD * rot_delta
    )

    return (
        metrics["steps"] <= MAX_RRT_PLAN_STEPS
        and metrics["cum_joint"] <= max_cum_joint
        and metrics["max_step"] <= MAX_RRT_MAX_STEP
        and (
            metrics["detour_ratio"] <= MAX_RRT_DETOUR_RATIO
            or metrics["cum_joint"] <= 1.0
        )
    ), metrics


def _best_rrt_plan_without_loop(planner: PandaArmMotionPlanningSolver, pose: sapien.Pose):
    best_result = None
    best_metrics = None

    for _ in range(RRT_REPLAN_ATTEMPTS):
        result = _plan_rrt_result(planner, pose)
        if result["status"] != "Success":
            continue

        is_reasonable, metrics = _rrt_plan_is_reasonable(planner, pose, result)
        if not is_reasonable:
            continue

        if best_result is None or metrics["cum_joint"] < best_metrics["cum_joint"]:
            best_result = result
            best_metrics = metrics

    return best_result

def _move(planner: PandaArmMotionPlanningSolver, pose: sapien.Pose):
    res = planner.move_to_pose_with_screw(pose)
    if res != -1:
        return res

    rrt_result = _best_rrt_plan_without_loop(planner, pose)
    if rrt_result is None:
        if planner.debug:
            print("Rejecting RRTConnect plan: no non-looping trajectory found")
        return -1

    planner.render_wait()
    return planner.follow_path(rrt_result)


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
    grasp_closing = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
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
