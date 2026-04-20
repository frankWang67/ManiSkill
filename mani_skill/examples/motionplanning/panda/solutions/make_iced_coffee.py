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
PRE_GRASP_Z_RANGE = (0.06, 0.10)
LIFT_DELTA_Z_RANGE = (0.10, 0.18)
MIN_LIFT_Z = 0.26
ABOVE_CUP_Z_RANGE = (0.08, 0.14)
PLACE_XY_JITTER = 0.012
DETOUR_RADIUS_RANGE = (0.05, 0.12)
DETOUR_XY_NOISE = 0.015
CRUISE_EXTRA_Z_RANGE = (0.03, 0.08)
DETOUR_PROB = 0.85
RETREAT_DETOUR_PROB = 0.5
GRASP_CLOSING_YAW_JITTER_RANGE = (-0.35, 0.35)

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

    rng = np.random.default_rng(seed)

    home_pose = env.agent.tcp.pose.sp

    approaching = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    tcp_closing = (
        env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    ).astype(np.float64)
    # Keep the grasp closing direction close to the current TCP closing
    # direction to avoid unnecessary 180-degree wrist flips, while still
    # allowing a small planar yaw adjustment to improve grasp robustness.
    target_closing = tcp_closing.copy()
    target_closing[2] = 0.0
    grasp_closing_yaw_jitter = float(rng.uniform(*GRASP_CLOSING_YAW_JITTER_RANGE))
    if abs(grasp_closing_yaw_jitter) > 1e-8:
        c = np.cos(grasp_closing_yaw_jitter)
        s = np.sin(grasp_closing_yaw_jitter)
        target_closing[:2] = np.array(
            [
                c * target_closing[0] - s * target_closing[1],
                s * target_closing[0] + c * target_closing[1],
            ],
            dtype=np.float64,
        )
    target_closing_norm = np.linalg.norm(target_closing)
    if target_closing_norm < 1e-8:
        target_closing = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        target_closing /= target_closing_norm

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

    pre_grasp_z = float(rng.uniform(*PRE_GRASP_Z_RANGE))
    pre_grasp_pose = grasp_pose * sapien.Pose([0.0, 0.0, -pre_grasp_z])

    lift_delta_z = float(rng.uniform(*LIFT_DELTA_Z_RANGE))
    lift_pose = sapien.Pose(
        p=np.array(
            [
                grasp_pose.p[0],
                grasp_pose.p[1],
                max(float(grasp_pose.p[2]) + lift_delta_z, MIN_LIFT_Z),
            ],
            dtype=np.float64,
        ),
        q=grasp_pose.q,
    )

    goal_pos = env.goal_site.pose.p[0].detach().cpu().numpy().astype(np.float64)
    place_pos = goal_pos.copy()
    place_pos[:2] += rng.uniform(-PLACE_XY_JITTER, PLACE_XY_JITTER, size=2)

    try:
        # 1) Get ready above the ice cube.
        planner.open_gripper(t=int(rng.integers(5, 9)))
        if _move(planner, pre_grasp_pose) == -1:
            return -1

        # 2) Move down to grasp and close gripper.
        if _move(planner, grasp_pose) == -1:
            return -1
        planner.close_gripper(t=int(rng.integers(10, 17)))

        # 3) Lift and transfer above the cup.
        if _move(planner, lift_pose) == -1:
            return -1
        # Use the post-lift orientation as transfer orientation to avoid
        # unnecessary branch switching that can make RRT fallback detour.
        transfer_q = env.agent.tcp.pose.q[0].detach().cpu().numpy().astype(np.float64)
        above_cup_z = float(rng.uniform(*ABOVE_CUP_Z_RANGE))
        above_cup_pose = sapien.Pose(
            p=place_pos + np.array([0.0, 0.0, above_cup_z], dtype=np.float64),
            q=transfer_q,
        )
        place_pose = sapien.Pose(p=place_pos, q=transfer_q)

        lift_tcp_pos = env.agent.tcp.pose.p[0].detach().cpu().numpy().astype(np.float64)

        # Optional transfer waypoints add trajectory diversity; failures are
        # ignored so the solver can still fall back to direct transfer.
        if rng.random() < DETOUR_PROB:
            to_goal_xy = place_pos[:2] - lift_tcp_pos[:2]
            to_goal_norm = np.linalg.norm(to_goal_xy)
            if to_goal_norm < 1e-8:
                fwd_xy = np.array([1.0, 0.0], dtype=np.float64)
            else:
                fwd_xy = to_goal_xy / to_goal_norm
            side_xy = np.array([-fwd_xy[1], fwd_xy[0]], dtype=np.float64)

            detour_side = 1.0 if rng.random() < 0.5 else -1.0
            detour_radius = float(rng.uniform(*DETOUR_RADIUS_RANGE))
            detour_xy = 0.5 * (lift_tcp_pos[:2] + place_pos[:2])
            detour_xy += detour_side * detour_radius * side_xy
            detour_xy += rng.uniform(-DETOUR_XY_NOISE, DETOUR_XY_NOISE, size=2)
            detour_z = max(float(lift_tcp_pos[2]), float(above_cup_pose.p[2]))
            detour_z += float(rng.uniform(*CRUISE_EXTRA_Z_RANGE))
            detour_pose = sapien.Pose(
                p=np.array([detour_xy[0], detour_xy[1], detour_z], dtype=np.float64),
                q=transfer_q,
            )
            _move(planner, detour_pose)

        if rng.random() < DETOUR_PROB:
            orbit_radius = float(rng.uniform(0.03, 0.08))
            orbit_theta = float(rng.uniform(-np.pi, np.pi))
            orbit_xy = place_pos[:2] + orbit_radius * np.array(
                [np.cos(orbit_theta), np.sin(orbit_theta)], dtype=np.float64
            )
            orbit_z = float(above_cup_pose.p[2]) + float(
                rng.uniform(*CRUISE_EXTRA_Z_RANGE)
            )
            orbit_pose = sapien.Pose(
                p=np.array([orbit_xy[0], orbit_xy[1], orbit_z], dtype=np.float64),
                q=transfer_q,
            )
            _move(planner, orbit_pose)

        if _move(planner, above_cup_pose) == -1:
            return -1

        # 4) Lower into the cup and release.
        if _move(planner, place_pose) == -1:
            return -1
        planner.open_gripper(t=int(rng.integers(10, 17)))

        # 5) Leave the cup and return home.
        if _move(planner, above_cup_pose) == -1:
            return -1

        if rng.random() < RETREAT_DETOUR_PROB:
            retreat_radius = float(rng.uniform(*DETOUR_RADIUS_RANGE))
            retreat_theta = float(rng.uniform(-np.pi, np.pi))
            retreat_xy = place_pos[:2] + retreat_radius * np.array(
                [np.cos(retreat_theta), np.sin(retreat_theta)], dtype=np.float64
            )
            retreat_z = float(above_cup_pose.p[2]) + float(
                rng.uniform(*CRUISE_EXTRA_Z_RANGE)
            )
            retreat_pose = sapien.Pose(
                p=np.array([retreat_xy[0], retreat_xy[1], retreat_z], dtype=np.float64),
                q=transfer_q,
            )
            _move(planner, retreat_pose)

        res = _move(planner, home_pose)
        return res
    finally:
        planner.close()
