import numpy as np
import sapien

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
EDGE_INSET = 0.012
PRE_GRASP_DIST = 0.10
LIFT_EXTRA_Z = 0.16
MIN_CRUISE_Z = 0.34
HOME_CLEARANCE_EXTRA_Z = 0.10
LEFT_TARGET_BACKOFF_REACHABLE = 0.005
LEFT_TARGET_BACKOFF_UNREACHABLE = 0.01
LEFT_RELEASE_SETTLE_STEPS = 24
LEFT_VERTICAL_CLEARANCE_Z = 0.02
MIN_LIFTED_DELTA_Z = 0.03
RETREAT_BACKOFF = 0.10
RETREAT_UP = -0.03
OPEN_RELEASE_STEPS = 10
POST_RELEASE_SETTLE_STEPS = 12
RETURN_HOME_STEPS = 36
RETURN_HOME_SETTLE_STEPS = 18
FINAL_SETTLE_STEPS = 12
POSE_INTERP_SEGMENTS = (2, 4)
MAX_RRT_PLAN_STEPS = 120
MAX_RRT_CUM_JOINT_TRAVEL = 4.0
MAX_RRT_MAX_STEP = 0.08
GRASP_VERIFY_STEPS = 6
MAX_GRASP_VERIFY_POS_ERR = 0.08
MAX_GRASP_VERIFY_DROP = 0.04
LEFT_BRANCH_IDX = 0

# Stable hanger-relative mug pose estimate used to compute a placement TCP target.
HANGER_TO_MUG_REL_POSE = np.array(
    [
        [-0.61402, -0.10827, 0.78183, 0.10743],
        [-0.76995, -0.13576, -0.62349, 0.16088],
        [0.17365, -0.98481, 0.0, 0.07182],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _first_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    if x.ndim == 0:
        return np.asarray([float(x)], dtype=np.float64)
    if x.ndim == 1:
        return x.astype(np.float64)
    return x[0].astype(np.float64)


def _normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v
    return v / n


def _scalar_bool(x) -> bool:
    return bool(_first_np(x)[0])


def _to_sapien_pose(pose):
    if hasattr(pose, "sp"):
        return pose.sp
    return pose


def _nlerp_quat(q0: np.ndarray, q1: np.ndarray, alpha: float):
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    if np.dot(q0, q1) < 0.0:
        q1 = -q1
    q = (1.0 - alpha) * q0 + alpha * q1
    return _normalize(q)


def _plan_screw_result(planner, pose: sapien.Pose):
    pose = _to_sapien_pose(pose)
    planner._update_grasp_visual(pose)
    pose = planner._transform_pose_for_planning(pose)
    result = planner.planner.plan_screw(
        np.concatenate([pose.p, pose.q]),
        planner._get_current_arm_qpos(),
        time_step=planner.base_env.control_timestep,
        use_point_cloud=planner.use_point_cloud,
    )
    if result["status"] != "Success":
        result = planner.planner.plan_screw(
            np.concatenate([pose.p, pose.q]),
            planner._get_current_arm_qpos(),
            time_step=planner.base_env.control_timestep,
            use_point_cloud=planner.use_point_cloud,
        )
    return result


def _plan_rrt_result(planner, pose: sapien.Pose):
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
        planner_name="RRTstar",
        wrt_world=True,
    )


def _path_metrics(result):
    qpos = np.asarray(result["position"], dtype=np.float64)
    if qpos.shape[0] <= 1:
        return dict(steps=int(qpos.shape[0]), cum_joint=0.0, direct_joint=0.0, max_step=0.0)
    diffs = np.linalg.norm(np.diff(qpos, axis=0), axis=1)
    return dict(
        steps=int(qpos.shape[0]),
        cum_joint=float(diffs.sum()),
        direct_joint=float(np.linalg.norm(qpos[-1] - qpos[0])),
        max_step=float(diffs.max()),
    )


def _rrt_plan_is_reasonable(result) -> bool:
    metrics = _path_metrics(result)
    return (
        metrics["steps"] <= MAX_RRT_PLAN_STEPS
        and metrics["cum_joint"] <= MAX_RRT_CUM_JOINT_TRAVEL
        and metrics["max_step"] <= MAX_RRT_MAX_STEP
    )


def _move_with_interpolated_screw(planner, target_pose: sapien.Pose, num_segments: int):
    start_pose = _to_sapien_pose(planner.env_agent.tcp_pose)
    target_pose = _to_sapien_pose(target_pose)
    obs = reward = terminated = truncated = info = None
    for i in range(num_segments):
        alpha = (i + 1) / num_segments
        pose_i = sapien.Pose(
            p=(1.0 - alpha) * np.asarray(start_pose.p, dtype=np.float64)
            + alpha * np.asarray(target_pose.p, dtype=np.float64),
            q=_nlerp_quat(start_pose.q, target_pose.q, alpha),
        )
        result = _plan_screw_result(planner, pose_i)
        if result["status"] != "Success":
            return -1
        obs, reward, terminated, truncated, info = planner.follow_path(result)
    return obs, reward, terminated, truncated, info


def _move(planner, pose: sapien.Pose, dry_run: bool = False):
    screw = _plan_screw_result(planner, pose)
    if screw["status"] == "Success":
        if dry_run:
            return screw
        return planner.follow_path(screw)

    rrt = _plan_rrt_result(planner, pose)
    if rrt["status"] != "Success":
        return -1
    if dry_run:
        return rrt
    return planner.follow_path(rrt)


def _move_local(planner, pose: sapien.Pose, dry_run: bool = False):
    screw = _plan_screw_result(planner, pose)
    if screw["status"] == "Success":
        if dry_run:
            return screw
        return planner.follow_path(screw)

    if not dry_run:
        for num_segments in POSE_INTERP_SEGMENTS:
            res = _move_with_interpolated_screw(planner, pose, num_segments)
            if res != -1:
                return res

    rrt = _plan_rrt_result(planner, pose)
    if rrt["status"] != "Success":
        return -1
    if dry_run:
        return rrt
    if not _rrt_plan_is_reasonable(rrt):
        return -1
    return planner.follow_path(rrt)


def _can_plan_pose(planner, pose: sapien.Pose) -> bool:
    pose = _to_sapien_pose(pose)
    planner._update_grasp_visual(pose)
    pose = planner._transform_pose_for_planning(pose)
    target = np.concatenate([pose.p, pose.q])
    screw = planner.planner.plan_screw(
        target,
        planner._get_current_arm_qpos(),
        time_step=planner.base_env.control_timestep,
        use_point_cloud=planner.use_point_cloud,
    )
    if screw["status"] == "Success":
        return True
    rrt = planner.planner.plan_qpos_to_pose(
        target,
        planner._get_ik_seed_qpos(),
        time_step=planner.base_env.control_timestep,
        use_point_cloud=planner.use_point_cloud,
        rrt_range=0.0,
        planning_time=1,
        planner_name="RRTstar",
        wrt_world=True,
    )
    return rrt["status"] == "Success"


def _can_plan_screw_pose(planner, pose: sapien.Pose) -> bool:
    pose = _to_sapien_pose(pose)
    planner._update_grasp_visual(pose)
    pose = planner._transform_pose_for_planning(pose)
    target = np.concatenate([pose.p, pose.q])
    screw = planner.planner.plan_screw(
        target,
        planner._get_current_arm_qpos(),
        time_step=planner.base_env.control_timestep,
        use_point_cloud=planner.use_point_cloud,
    )
    return screw["status"] == "Success"


def _step_with_qpos(planner, qpos: np.ndarray):
    if planner.control_mode == "pd_joint_pos_vel":
        action = np.hstack([qpos, np.zeros_like(qpos), planner.gripper_state])
    else:
        action = np.hstack([qpos, planner.gripper_state])
    obs, reward, terminated, truncated, info = planner.env.step(action)
    planner.elapsed_steps += 1
    if planner.vis:
        planner.base_env.render_human()
    return obs, reward, terminated, truncated, info


def _move_to_qpos_linearly(planner, target_qpos: np.ndarray, steps: int):
    start_qpos = planner.robot.get_qpos()[0, : len(target_qpos)].cpu().numpy()
    obs = reward = terminated = truncated = info = None
    for i in range(steps):
        alpha = (i + 1) / steps
        qpos = (1.0 - alpha) * start_qpos + alpha * target_qpos
        obs, reward, terminated, truncated, info = _step_with_qpos(planner, qpos)
    return obs, reward, terminated, truncated, info


def _hold_current_qpos(planner, steps: int):
    qpos = planner.robot.get_qpos()[0, : len(planner.planner.joint_vel_limits)].cpu().numpy()
    obs = reward = terminated = truncated = info = None
    for _ in range(steps):
        obs, reward, terminated, truncated, info = _step_with_qpos(planner, qpos)
    return obs, reward, terminated, truncated, info


def _hold_target_qpos(planner, qpos: np.ndarray, steps: int):
    obs = reward = terminated = truncated = info = None
    for _ in range(steps):
        obs, reward, terminated, truncated, info = _step_with_qpos(planner, qpos)
    return obs, reward, terminated, truncated, info


def _verify_physical_grasp(
    planner,
    env,
    mug_in_tcp,
    steps: int = GRASP_VERIFY_STEPS,
    max_pos_err: float = MAX_GRASP_VERIFY_POS_ERR,
    max_drop: float = MAX_GRASP_VERIFY_DROP,
):
    mug_pose_before = _to_sapien_pose(env.mug.pose)
    _hold_current_qpos(planner, steps)
    tcp_pose_after = _to_sapien_pose(env.agent.tcp_pose)
    expected_mug_pose = tcp_pose_after * mug_in_tcp
    mug_pose_after = _to_sapien_pose(env.mug.pose)

    pos_err = np.linalg.norm(np.asarray(mug_pose_after.p) - np.asarray(expected_mug_pose.p))
    z_drop = float(mug_pose_before.p[2] - mug_pose_after.p[2])
    return (pos_err <= max_pos_err) and (z_drop <= max_drop)


def _compute_release_retreat_dir(
    branch_dir: np.ndarray, target_branch_idx: int, prefer_side_retreat: bool
):
    branch_dir = _normalize(branch_dir)
    if prefer_side_retreat:
        retreat_dir = np.cross(branch_dir, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        if target_branch_idx == LEFT_BRANCH_IDX:
            retreat_dir = -retreat_dir
    else:
        retreat_dir = -branch_dir
    retreat_dir[2] = 0.0
    if np.linalg.norm(retreat_dir) < 1e-8:
        retreat_dir = -branch_dir
        retreat_dir[2] = 0.0
    return _normalize(retreat_dir)


def _release_state(info):
    return (
        f"success={_scalar_bool(info['success'])} "
        f"is_hung={_scalar_bool(info['is_hung'])} "
        f"candidate={_scalar_bool(info['released_hang_candidate'])} "
        f"stable={_scalar_bool(info['is_release_stable'])} "
        f"steps={int(_first_np(info['released_hang_steps'])[0])}"
    )


def _build_retreat_pose(env, retreat_dir: np.ndarray, z_offset: float):
    tcp_pose = _to_sapien_pose(env.agent.tcp_pose)
    return sapien.Pose(
        p=tcp_pose.p
        + retreat_dir * RETREAT_BACKOFF
        + np.array([0.0, 0.0, z_offset], dtype=np.float64),
        q=tcp_pose.q,
    )


def _build_home_clearance_pose(tcp_pose: sapien.Pose, min_clearance_z: float):
    clearance_z = max(float(tcp_pose.p[2] + HOME_CLEARANCE_EXTRA_Z), min_clearance_z)
    return sapien.Pose(
        p=np.array([tcp_pose.p[0], tcp_pose.p[1], clearance_z], dtype=np.float64),
        q=tcp_pose.q,
    )


def _build_vertical_clearance_pose(env, z_offset: float):
    tcp_pose = _to_sapien_pose(env.agent.tcp_pose)
    return sapien.Pose(
        p=np.array([tcp_pose.p[0], tcp_pose.p[1], tcp_pose.p[2] + z_offset], dtype=np.float64),
        q=tcp_pose.q,
    )


def _select_closing_toward_handle(
    env,
    approaching: np.ndarray,
    closing: np.ndarray,
    grasp_center_base: np.ndarray,
    edge_shift: float,
):
    handle_world = None
    if hasattr(env, "_mug_handle_center_world"):
        handle_world = _first_np(env._mug_handle_center_world())
    elif hasattr(env, "mug_handle_local_pos"):
        mug_pose = _to_sapien_pose(env.mug.pose)
        handle_local = _first_np(env.mug_handle_local_pos)
        handle_world = np.asarray((mug_pose * sapien.Pose(p=handle_local)).p, dtype=np.float64)
    if handle_world is None:
        return closing

    camera_link = getattr(getattr(env.agent, "robot", None), "links_map", {}).get(
        "camera_link", None
    )
    if camera_link is None:
        return closing

    tcp_pose_now = _to_sapien_pose(env.agent.tcp_pose)
    cam_pose_now = _to_sapien_pose(camera_link.pose)
    tcp_to_cam = _to_sapien_pose(tcp_pose_now.inv() * cam_pose_now)
    robot_p = _first_np(env.agent.robot.pose.p)
    mug_center = _first_np(env.mug.pose.p)
    handle_vec = np.asarray(handle_world - mug_center, dtype=np.float64)
    if np.linalg.norm(handle_vec) < 1e-8:
        return closing

    def _camera_handle_score(closing_dir: np.ndarray):
        side_sign = 1.0 if np.dot(robot_p - grasp_center_base, closing_dir) >= 0 else -1.0
        grasp_center = grasp_center_base + side_sign * closing_dir * edge_shift
        grasp_pose = _to_sapien_pose(
            env.agent.build_grasp_pose(approaching, closing_dir, grasp_center)
        )
        cam_pose = _to_sapien_pose(grasp_pose * tcp_to_cam)
        cam_vec = _first_np(cam_pose.p) - mug_center
        return float(np.dot(cam_vec, handle_vec))

    pos_score = _camera_handle_score(closing)
    neg_score = _camera_handle_score(-closing)
    return -closing if neg_score > pos_score else closing


def solve(env, seed=None, debug=False, vis=False):
    if seed is not None:
        np.random.seed(int(seed))
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in ["pd_joint_pos", "pd_joint_pos_vel"]

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

    try:
        gripper_ctrl = env.agent.controller.controllers["gripper_active"]
        planner.OPEN = float(gripper_ctrl.single_action_space.low[0])
        planner.CLOSED = float(gripper_ctrl.single_action_space.high[0])

        dof = len(planner.planner.joint_vel_limits)
        home_qpos = env.agent.robot.get_qpos()[0, :dof].cpu().numpy()
        home_tcp_pose = _to_sapien_pose(env.agent.tcp_pose)
        mug_start_z = float(_first_np(env.mug.pose.p)[2])
        rng = np.random.default_rng(seed)
        hangers = env.hangers if hasattr(env, "hangers") else [env.hanger]
        target_branch_idx = int(rng.integers(0, len(hangers)))

        # ------------------------------------------------------------------ #
        # 1) Approach and grasp mug edge (single-side grasp)
        # ------------------------------------------------------------------ #
        obb = get_actor_obb(env.mug)
        approaching = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=FINGER_LENGTH,
        )

        closing = grasp_info["closing"]
        grasp_center_base = grasp_info["center"].copy()
        edge_half = grasp_info["extents"][1] * 0.5
        edge_shift = max(edge_half - EDGE_INSET, 0.0)
        closing = _select_closing_toward_handle(
            env,
            approaching=approaching,
            closing=closing,
            grasp_center_base=grasp_center_base,
            edge_shift=edge_shift,
        )

        robot_p = _first_np(env.agent.robot.pose.p)
        primary_side_sign = (
            1.0 if np.dot(robot_p - grasp_center_base, closing) >= 0 else -1.0
        )
        grasp_side_sign = primary_side_sign
        if target_branch_idx == LEFT_BRANCH_IDX:
            grasp_side_sign = -primary_side_sign

        planner.open_gripper(t=6)
        grasp_center = grasp_center_base + grasp_side_sign * closing * edge_shift
        grasp_pose = env.agent.build_grasp_pose(approaching, closing, grasp_center)
        pre_grasp_pose = grasp_pose * sapien.Pose([0.0, 0.0, -PRE_GRASP_DIST])
        if _move(planner, pre_grasp_pose) == -1:
            return -1
        if _move(planner, grasp_pose) == -1:
            return -1

        planner.close_gripper(t=14)

        # ------------------------------------------------------------------ #
        # 2) Lift mug up
        # ------------------------------------------------------------------ #
        tcp_pose = _to_sapien_pose(env.agent.tcp_pose)
        lift_z = max(float(tcp_pose.p[2] + LIFT_EXTRA_Z), MIN_CRUISE_Z)
        lift_pose = sapien.Pose(
            p=np.array([tcp_pose.p[0], tcp_pose.p[1], lift_z], dtype=np.float64),
            q=tcp_pose.q,
        )
        if _move(planner, lift_pose) == -1:
            return -1

        planner.close_gripper(t=6)
        mug_lifted_z = float(_first_np(env.mug.pose.p)[2])
        if mug_lifted_z < mug_start_z + MIN_LIFTED_DELTA_Z:
            return -1

        # Mug-TCP transform after successful lift.
        mug_to_tcp = _to_sapien_pose(env.mug.pose.inv() * env.agent.tcp_pose)
        mug_in_tcp = _to_sapien_pose(mug_to_tcp.inv())

        # ------------------------------------------------------------------ #
        # 3) Compute target pose from hanger pose
        # ------------------------------------------------------------------ #
        target_mug_pose = env.get_hang_goal_pose(branch_idx=target_branch_idx)
        target_mug_pose = sapien.Pose(
            p=_first_np(target_mug_pose["mug_pos"]).astype(np.float32),
            q=_first_np(target_mug_pose["mug_quat"]).astype(np.float32),
        )
        target_tcp_pose = target_mug_pose * mug_to_tcp

        hang_pose = env.get_hang_pose_and_direction(branch_idx=target_branch_idx)
        branch_dir = _normalize(_first_np(hang_pose["approach"]))

        # ------------------------------------------------------------------ #
        # 4) Rotate first, then directly transfer to target hang pose
        # ------------------------------------------------------------------ #
        current_tcp_pose = _to_sapien_pose(env.agent.tcp_pose)
        rotate_pose_current_xy = sapien.Pose(
            p=np.array(
                [current_tcp_pose.p[0], current_tcp_pose.p[1], max(lift_z, current_tcp_pose.p[2])],
                dtype=np.float64,
            ),
            q=target_tcp_pose.q,
        )
        rotate_pose_target_xy = sapien.Pose(
            p=np.array(
                [target_tcp_pose.p[0], target_tcp_pose.p[1], max(lift_z, current_tcp_pose.p[2])],
                dtype=np.float64,
            ),
            q=target_tcp_pose.q,
        )
        do_rotate = True
        if target_branch_idx == LEFT_BRANCH_IDX:
            if _can_plan_screw_pose(planner, rotate_pose_target_xy):
                rotate_pose = rotate_pose_target_xy
            else:
                left_target_backoff = (
                    LEFT_TARGET_BACKOFF_REACHABLE
                    if _can_plan_pose(planner, rotate_pose_target_xy)
                    else LEFT_TARGET_BACKOFF_UNREACHABLE
                )
                target_mug_pose = sapien.Pose(
                    p=np.asarray(target_mug_pose.p, dtype=np.float64)
                    - branch_dir * left_target_backoff,
                    q=target_mug_pose.q,
                )
                target_tcp_pose = target_mug_pose * mug_to_tcp
                rotate_pose = sapien.Pose(
                    p=np.array(
                        [current_tcp_pose.p[0], current_tcp_pose.p[1], max(lift_z, current_tcp_pose.p[2])],
                        dtype=np.float64,
                    ),
                    q=target_tcp_pose.q,
                )
        else:
            rotate_pose = rotate_pose_current_xy
            if not _can_plan_pose(planner, rotate_pose_current_xy):
                do_rotate = False
        if do_rotate and _move(planner, rotate_pose) == -1:
            return -1
        
        mug_to_tcp = _to_sapien_pose(env.mug.pose.inv() * env.agent.tcp_pose)
        target_tcp_pose = target_mug_pose * mug_to_tcp
        if _move(planner, target_tcp_pose) == -1:
            return -1

        if not _verify_physical_grasp(planner, env, mug_in_tcp):
            return -1

        # Release is purely physical from here.
        planner.open_gripper(t=OPEN_RELEASE_STEPS)
        initial_release_settle_steps = (
            LEFT_RELEASE_SETTLE_STEPS
            if target_branch_idx == LEFT_BRANCH_IDX
            else POST_RELEASE_SETTLE_STEPS
        )
        _hold_current_qpos(planner, initial_release_settle_steps)
        release_info = env.evaluate()
        if debug:
            print(
                f"[release/post_open_settle] branch={target_branch_idx} "
                f"{_release_state(release_info)}"
            )

        # ------------------------------------------------------------------ #
        # 5) Return to initial pose
        # ------------------------------------------------------------------ #
        if (
            target_branch_idx == LEFT_BRANCH_IDX
            and _scalar_bool(release_info["is_hung"])
            and (not _scalar_bool(release_info["is_release_stable"]))
        ):
            _hold_current_qpos(planner, POST_RELEASE_SETTLE_STEPS)
            release_info = env.evaluate()
            if debug:
                print(
                    f"[release/post_left_extra_settle] branch={target_branch_idx} "
                    f"{_release_state(release_info)}"
                )

        if _scalar_bool(release_info["released_hang_candidate"]) and (
            not _scalar_bool(release_info["is_release_stable"])
        ):
            remaining_settle = int(
                max(
                    env.success_settle_steps
                    - int(_first_np(release_info["released_hang_steps"])[0]),
                    0,
                )
            )
            if remaining_settle > 0:
                _hold_current_qpos(planner, remaining_settle)
                release_info = env.evaluate()
                if debug:
                    print(
                        f"[release/post_candidate_hold] branch={target_branch_idx} "
                        f"{_release_state(release_info)}"
                    )

        if (
            target_branch_idx == LEFT_BRANCH_IDX
            and _scalar_bool(release_info["is_hung"])
            and (not _scalar_bool(release_info["released_hang_candidate"]))
            and (not _scalar_bool(release_info["is_release_stable"]))
        ):
            vertical_clearance_pose = _build_vertical_clearance_pose(
                env, LEFT_VERTICAL_CLEARANCE_Z
            )
            if _can_plan_pose(planner, vertical_clearance_pose):
                _move(planner, vertical_clearance_pose)
                _hold_current_qpos(planner, POST_RELEASE_SETTLE_STEPS)
                release_info = env.evaluate()
                if debug:
                    print(
                        f"[release/post_vertical_clear] branch={target_branch_idx} "
                        f"{_release_state(release_info)}"
                    )

        if _scalar_bool(release_info["released_hang_candidate"]) and (
            not _scalar_bool(release_info["is_release_stable"])
        ):
            remaining_settle = int(
                max(
                    env.success_settle_steps
                    - int(_first_np(release_info["released_hang_steps"])[0]),
                    0,
                )
            )
            if remaining_settle > 0:
                _hold_current_qpos(planner, remaining_settle)
                release_info = env.evaluate()
                if debug:
                    print(
                        f"[release/post_candidate_hold_2] branch={target_branch_idx} "
                        f"{_release_state(release_info)}"
                    )

        release_stable = _scalar_bool(release_info["is_release_stable"])
        retreat_succeeded = False
        if not release_stable:
            if target_branch_idx == LEFT_BRANCH_IDX:
                retreat_candidates = [
                    ("side", RETREAT_UP),
                    ("axial", 0.0),
                    ("axial", RETREAT_UP),
                    ("side", 0.0),
                ]
            else:
                retreat_candidates = [("side", RETREAT_UP), ("side", 0.0)]

            for retreat_mode, retreat_z_offset in retreat_candidates:
                retreat_dir = _compute_release_retreat_dir(
                    branch_dir,
                    target_branch_idx,
                    prefer_side_retreat=(retreat_mode == "side"),
                )
                retreat_pose = _build_retreat_pose(env, retreat_dir, retreat_z_offset)
                reachable = _can_plan_pose(planner, retreat_pose)
                if debug:
                    print(
                        f"[release/retreat_dir] branch={target_branch_idx} "
                        f"mode={retreat_mode} dir={retreat_dir} "
                        f"z={retreat_z_offset} reachable={reachable}"
                    )
                if not reachable:
                    continue
                if _move_local(planner, retreat_pose) == -1:
                    continue
                retreat_succeeded = True
                break
            _hold_current_qpos(planner, POST_RELEASE_SETTLE_STEPS)
            release_info = env.evaluate()
            if debug:
                print(
                    f"[release/post_retreat] branch={target_branch_idx} "
                    f"moved={retreat_succeeded} {_release_state(release_info)}"
                )

        if _scalar_bool(release_info["is_hung"]) and (
            not _scalar_bool(release_info["is_release_stable"])
        ):
            remaining_settle = int(
                max(
                    env.success_settle_steps
                    - int(_first_np(release_info["released_hang_steps"])[0]),
                    0,
                )
            )
            if remaining_settle > 0:
                _hold_current_qpos(planner, remaining_settle)
                release_info = env.evaluate()
                if debug:
                    print(
                        f"[release/post_hold] branch={target_branch_idx} "
                        f"{_release_state(release_info)}"
                    )

        if _scalar_bool(release_info["is_release_stable"]) and _scalar_bool(
            release_info["is_hung"]
        ):
            detach_candidates = [("side", 0.0), ("side", RETREAT_UP)]

            detach_succeeded = False
            for detach_mode, detach_z_offset in detach_candidates:
                detach_dir = _compute_release_retreat_dir(
                    branch_dir,
                    target_branch_idx,
                    prefer_side_retreat=True,
                )
                detach_pose = _build_retreat_pose(env, detach_dir, detach_z_offset)
                reachable = _can_plan_pose(planner, detach_pose)
                if debug:
                    print(
                        f"[release/detach] branch={target_branch_idx} "
                        f"mode={detach_mode} z={detach_z_offset} reachable={reachable}"
                    )
                if not reachable:
                    continue
                if _move_local(planner, detach_pose) == -1:
                    continue
                detach_succeeded = True
                break
            if detach_succeeded:
                _hold_current_qpos(planner, POST_RELEASE_SETTLE_STEPS)
                release_info = env.evaluate()
                if debug:
                    print(
                        f"[release/post_detach] branch={target_branch_idx} "
                        f"{_release_state(release_info)}"
                    )

        if target_branch_idx == LEFT_BRANCH_IDX and _scalar_bool(release_info["is_hung"]):
            pre_home_clearance_pose = _build_vertical_clearance_pose(
                env, LEFT_VERTICAL_CLEARANCE_Z
            )
        else:
            pre_home_clearance_pose = _build_home_clearance_pose(
                _to_sapien_pose(env.agent.tcp_pose), min_clearance_z=lift_z
            )
        if _can_plan_pose(planner, pre_home_clearance_pose):
            _move(planner, pre_home_clearance_pose)

        returned_home_with_pose = False
        if _can_plan_pose(planner, home_tcp_pose):
            returned_home_with_pose = _move(planner, home_tcp_pose) != -1

        if not returned_home_with_pose:

            _move_to_qpos_linearly(planner, home_qpos, steps=RETURN_HOME_STEPS)

        _hold_target_qpos(planner, home_qpos, steps=RETURN_HOME_SETTLE_STEPS)
        release_info = env.evaluate()
        if _scalar_bool(release_info["released_hang_candidate"]) and (
            not _scalar_bool(release_info["is_release_stable"])
        ):
            remaining_settle = int(
                max(
                    env.success_settle_steps
                    - int(_first_np(release_info["released_hang_steps"])[0]),
                    0,
                )
            )
            if remaining_settle > 0:
                _hold_target_qpos(planner, home_qpos, steps=remaining_settle)
                release_info = env.evaluate()

        if debug:
            print(
                f"[release/post_home] branch={target_branch_idx} "
                f"{_release_state(release_info)}"
            )
        res = _hold_target_qpos(planner, home_qpos, FINAL_SETTLE_STEPS)
        return res
    finally:
        planner.close()
