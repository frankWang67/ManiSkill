import numpy as np
import sapien
import torch

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
MIN_LIFTED_DELTA_Z = 0.03
RETREAT_BACKOFF = 0.06
RETREAT_UP = 0.07
OPEN_RELEASE_STEPS = 10
POST_RELEASE_SETTLE_STEPS = 12
RETURN_HOME_STEPS = 36
FINAL_SETTLE_STEPS = 12
MAX_ATTACH_POS_ERR = 0.12
MAX_ATTACH_ROT_ERR = 3.2
GRASP_VERIFY_STEPS = 6
MAX_GRASP_VERIFY_POS_ERR = 0.08
MAX_GRASP_VERIFY_DROP = 0.04

# Stable hanger-relative mug pose estimate used to compute a placement TCP target.
HANGER_TO_MUG_REL_POSE = np.array(
    [
        [-0.96162, 0.15370, 0.22722, 0.06687-0.02],
        [-0.14493, 0.41862, -0.89652, 0.11050],
        [-0.23292, -0.89504, -0.38028, 0.04691-0.02],
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


def _to_sapien_pose(pose):
    if hasattr(pose, "sp"):
        return pose.sp
    return pose


def _quat_angle(q0, q1):
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 = q0 / (np.linalg.norm(q0) + 1e-8)
    q1 = q1 / (np.linalg.norm(q1) + 1e-8)
    dot = np.clip(np.abs(np.dot(q0, q1)), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


def _move(planner, pose: sapien.Pose, dry_run: bool = False):
    res = planner.move_to_pose_with_screw(pose, dry_run=dry_run)
    if res == -1:
        res = planner.move_to_pose_with_RRTStar(pose, dry_run=dry_run)
    return res


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


def _set_mug_pose_static(env, mug_pose: sapien.Pose):
    env.mug.set_pose(mug_pose)
    env.mug.set_linear_velocity(torch.zeros_like(env.mug.get_linear_velocity()))
    env.mug.set_angular_velocity(torch.zeros_like(env.mug.get_angular_velocity()))


def _follow_path_with_attachment(planner, result, env, mug_in_tcp):
    n_step = result["position"].shape[0]
    if n_step == 0:
        qpos = planner.robot.get_qpos()[0, : len(planner.planner.joint_vel_limits)].cpu().numpy()
        obs, reward, terminated, truncated, info = _step_with_qpos(planner, qpos)
        tcp_pose = _to_sapien_pose(env.agent.tcp_pose)
        expected_mug_pose = tcp_pose * mug_in_tcp
        current_mug_pose = _to_sapien_pose(env.mug.pose)
        pos_err = np.linalg.norm(np.asarray(current_mug_pose.p) - np.asarray(expected_mug_pose.p))
        rot_err = _quat_angle(current_mug_pose.q, expected_mug_pose.q)
        if pos_err > MAX_ATTACH_POS_ERR or rot_err > MAX_ATTACH_ROT_ERR:
            return -1
        _set_mug_pose_static(env, expected_mug_pose)
        return obs, reward, terminated, truncated, info

    for i in range(n_step):
        qpos = result["position"][i]
        obs, reward, terminated, truncated, info = _step_with_qpos(planner, qpos)
        tcp_pose = _to_sapien_pose(env.agent.tcp_pose)
        expected_mug_pose = tcp_pose * mug_in_tcp
        current_mug_pose = _to_sapien_pose(env.mug.pose)
        pos_err = np.linalg.norm(np.asarray(current_mug_pose.p) - np.asarray(expected_mug_pose.p))
        rot_err = _quat_angle(current_mug_pose.q, expected_mug_pose.q)
        if pos_err > MAX_ATTACH_POS_ERR or rot_err > MAX_ATTACH_ROT_ERR:
            return -1
        _set_mug_pose_static(env, expected_mug_pose)
    return obs, reward, terminated, truncated, info


def _move_with_attachment(planner, env, pose: sapien.Pose, mug_in_tcp):
    res = _move(planner, pose, dry_run=True)
    if res == -1:
        return -1
    return _follow_path_with_attachment(planner, res, env, mug_in_tcp)


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


def solve(env, seed=None, debug=False, vis=False):
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
        mug_start_z = float(_first_np(env.mug.pose.p)[2])

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

        robot_p = _first_np(env.agent.robot.pose.p)
        primary_side_sign = 1.0 if np.dot(robot_p - grasp_center_base, closing) >= 0 else -1.0

        planner.open_gripper(t=6)
        grasp_ok = False
        for side_sign in [primary_side_sign, -primary_side_sign]:
            grasp_center = grasp_center_base + side_sign * closing * edge_shift
            grasp_pose = env.agent.build_grasp_pose(approaching, closing, grasp_center)
            pre_grasp_pose = grasp_pose * sapien.Pose([0.0, 0.0, -PRE_GRASP_DIST])
            if _move(planner, pre_grasp_pose) == -1:
                continue
            if _move(planner, grasp_pose) == -1:
                continue
            grasp_ok = True
            break
        if not grasp_ok:
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
        rng = np.random.default_rng(seed)
        hangers = env.hangers if hasattr(env, "hangers") else [env.hanger]
        target_branch_idx = int(rng.integers(0, len(hangers)))

        hanger_tf = hangers[target_branch_idx].pose.to_transformation_matrix()[0].cpu().numpy()
        target_mug_pose = sapien.Pose(hanger_tf @ HANGER_TO_MUG_REL_POSE)
        target_tcp_pose = target_mug_pose * mug_to_tcp

        hang_pose = env.get_hang_pose_and_direction(branch_idx=target_branch_idx)
        branch_dir = _normalize(_first_np(hang_pose["approach"]))

        # ------------------------------------------------------------------ #
        # 4) Rotate first, then directly transfer to target hang pose
        # ------------------------------------------------------------------ #
        current_tcp_pose = _to_sapien_pose(env.agent.tcp_pose)
        rotate_pose = sapien.Pose(
            p=np.array(
                [current_tcp_pose.p[0], current_tcp_pose.p[1], max(lift_z, current_tcp_pose.p[2])],
                dtype=np.float64,
            ),
            q=target_tcp_pose.q,
        )
        if _move_with_attachment(planner, env, rotate_pose, mug_in_tcp) == -1:
            return -1
        if _move_with_attachment(planner, env, target_tcp_pose, mug_in_tcp) == -1:
            return -1

        if not _verify_physical_grasp(planner, env, mug_in_tcp):
            return -1

        # Release is purely physical from here: no mug pose forcing.
        planner.open_gripper(t=OPEN_RELEASE_STEPS)
        _hold_current_qpos(planner, POST_RELEASE_SETTLE_STEPS)

        # ------------------------------------------------------------------ #
        # 5) Return to initial pose
        # ------------------------------------------------------------------ #
        retreat_pose = sapien.Pose(
            p=_to_sapien_pose(env.agent.tcp_pose).p
            + branch_dir * RETREAT_BACKOFF
            + np.array([0.0, 0.0, RETREAT_UP], dtype=np.float64),
            q=_to_sapien_pose(env.agent.tcp_pose).q,
        )
        _move(planner, retreat_pose)
        _move_to_qpos_linearly(planner, home_qpos, steps=RETURN_HOME_STEPS)

        res = planner.open_gripper(t=FINAL_SETTLE_STEPS)
        return res
    finally:
        planner.close()
