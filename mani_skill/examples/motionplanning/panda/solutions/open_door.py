import numpy as np
import sapien
from transforms3d.quaternions import axangle2quat, qmult

from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)


RETURN_HOME_STEPS = 30
RETURN_HOME_OPEN_GRIPPER_STEPS = 8
POST_OPEN_RETREAT_DIST = 0.2


def _normalize(vec):
    vec = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm


def _first_batch_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    if x.ndim == 0:
        return np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return x.astype(np.float64)
    return x[0].astype(np.float64)


def _move(planner, pose: sapien.Pose, dry_run: bool = False):
    res = planner.move_to_pose_with_screw(pose, dry_run=dry_run)
    if res == -1:
        res = planner.move_to_pose_with_RRTStar(pose, dry_run=dry_run)
    return res


def _move_screw(planner, pose: sapien.Pose, dry_run: bool = False):
    return planner.move_to_pose_with_screw(pose, dry_run=dry_run)


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


def _retreat_tcp_backward(planner, distance: float):
    tcp_pose = planner.base_env.agent.tcp.pose.sp
    tcp_mat = tcp_pose.to_transformation_matrix()
    tcp_forward = _normalize(tcp_mat[:3, 2])
    retreat_pos = np.asarray(tcp_pose.p, dtype=np.float64) - tcp_forward * distance
    retreat_pose = sapien.Pose(p=retreat_pos, q=tcp_pose.q)
    res = _move_screw(planner, retreat_pose)
    if res == -1:
        res = _move(planner, retreat_pose)
    return res


def _auto_configure_gripper_targets(env, planner):
    gripper_ctrl = env.agent.controller.controllers["gripper_active"]
    planner.OPEN = float(gripper_ctrl.single_action_space.low[0])
    planner.CLOSED = float(gripper_ctrl.single_action_space.high[0])


def _apply_env_obstacles_to_planner(env, planner):
    planner.clear_collisions()
    if not hasattr(env, "get_obstacles_info"):
        return

    for obs in env.get_obstacles_info():
        center = _first_batch_np(obs["center"])
        quat = _first_batch_np(obs["quat"])
        extent = _first_batch_np(obs["extent"])
        if center[2] < -1.0 or np.any(extent <= 0):
            continue
        quat_norm = np.linalg.norm(quat)
        if quat_norm < 1e-8:
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            quat = quat / quat_norm
        planner.add_box_collision(
            extents=extent * 2.0, pose=sapien.Pose(p=center, q=quat)
        )


def _rotate_about_axis(vec, axis, angle):
    axis = _normalize(axis)
    vec = np.asarray(vec, dtype=np.float64)
    cos_t = np.cos(angle)
    sin_t = np.sin(angle)
    return (
        vec * cos_t
        + np.cross(axis, vec) * sin_t
        + axis * np.dot(axis, vec) * (1.0 - cos_t)
    )


def _rotate_pose_world(pose: sapien.Pose, axis, angle):
    world_q = axangle2quat(_normalize(axis), angle)
    return sapien.Pose(p=pose.p, q=qmult(world_q, pose.q))


def _compute_handle_grasp(env):
    # Frame definitions from OpenDoor-v1:
    # - handle_pos is the selected handle mesh center in world frame.
    # - hinge_pos is the target revolute joint origin in world frame.
    # - hinge_axis is the joint +X axis in world frame.
    # For Panda build_grasp_pose():
    # - TCP +Z = approaching direction.
    # - TCP +Y = finger closing direction.
    # - TCP +X = finger length direction = closing x approaching.
    kinematics = env.get_door_kinematics()
    handle_pos = _first_batch_np(kinematics["handle_pos"])
    hinge_pos = _first_batch_np(kinematics["hinge_pos"])
    hinge_axis = _normalize(_first_batch_np(kinematics["hinge_axis"]))

    radial = handle_pos - hinge_pos
    handle_vec = radial.copy()
    radial = radial - hinge_axis * np.dot(radial, hinge_axis)
    radius = np.linalg.norm(radial)
    radial = _normalize(radial)

    robot_anchor = _first_batch_np(env.agent.robot.pose.p)
    to_robot = robot_anchor - handle_pos
    to_robot = to_robot - hinge_axis * np.dot(to_robot, hinge_axis)

    # The door plane is spanned by hinge_axis and radial.
    # Its normal points to either the outside side or the inside side.
    outside_normal = _normalize(np.cross(hinge_axis, radial))
    if np.dot(outside_normal, to_robot) < 0:
        outside_normal = -outside_normal

    # True tangent grasp:
    # - TCP +Z approaches along the handle tangent in the door plane.
    #   For this cabinet handle, that tangent is the radial direction across the door width.
    # - TCP +Y closes along the outside/inside normal.
    # - TCP +X then aligns with the hinge axis / handle axis.
    approaching = -radial
    closing = outside_normal
    grasp_axis = closing.copy()

    grasp_pos = handle_pos + outside_normal * 0.01 - radial * 0.05
    # grasp_pos = handle_pos - outside_normal * 0.03
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, grasp_pos)
    grasp_pose = _rotate_pose_world(grasp_pose, [1.0, 0.0, 0.0], np.pi)
    # grasp_pose = _rotate_pose_world(grasp_pose, hinge_axis, np.deg2rad(-15.0))

    # # Add a small outward tilt around the handle axis so the gripper does not
    # # scrape the door face during the final approach.
    # tilt_angle = np.deg2rad(12.0)
    # tilt_candidates = [
    #     _rotate_pose_world(grasp_pose, hinge_axis, tilt_angle),
    #     _rotate_pose_world(grasp_pose, hinge_axis, -tilt_angle),
    # ]
    # best_pose = grasp_pose
    # best_score = -np.inf
    # for candidate in tilt_candidates:
    #     candidate_mat = candidate.to_transformation_matrix()
    #     candidate_approaching = candidate_mat[:3, 2]
    #     score = np.dot(candidate_approaching, outside_normal)
    #     if score > best_score:
    #         best_score = score
    #         best_pose = candidate
    # grasp_pose = best_pose

    grasp_mat = grasp_pose.to_transformation_matrix()
    tilted_approaching = grasp_mat[:3, 2]
    pre_grasp_pos = handle_pos - tilted_approaching * 0.06 + outside_normal * 0.04
    pre_grasp_pose = sapien.Pose(pre_grasp_pos, grasp_pose.q)

    return dict(
        handle_pos=handle_pos,
        hinge_pos=hinge_pos,
        hinge_axis=hinge_axis,
        grasp_axis=grasp_axis,
        handle_vec=handle_vec,
        radial=radial,
        radius=radius,
        approaching=approaching,
        outside_normal=outside_normal,
        grasp_pose=grasp_pose,
        pre_grasp_pose=pre_grasp_pose,
    )


def _open_door_smooth(env, planner, grasp):
    res = None
    current_q = float(_first_batch_np(env.current_angle))
    target_q = float(_first_batch_np(env.motionplan_target_qpos))
    delta_q = target_q - current_q
    if abs(delta_q) < 1e-4:
        return res

    # Keep the TCP on the handle arc with a fixed grasp orientation. Reorienting
    # the wrist after contact can trigger large IK branch changes and upward
    # detours, while a fixed orientation keeps the post-grasp motion smooth.
    fixed_q = grasp["grasp_pose"].q
    n_steps = 16
    for frac in np.linspace(1.0 / n_steps, 1.0, n_steps):
        angle = delta_q * frac
        rotated_handle_vec = _rotate_about_axis(
            grasp["handle_vec"], grasp["hinge_axis"], angle
        )
        target_handle = grasp["hinge_pos"] + rotated_handle_vec + grasp["outside_normal"] * 0.01 - grasp["radial"] * 0.05
        target_pose = sapien.Pose(p=target_handle, q=fixed_q)
        res = _move_screw(planner, target_pose)
        if res == -1:
            return -1
        if bool(env.evaluate()["success"][0].item()):
            return res
    return res


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
        joint_vel_limits=0.5,
        joint_acc_limits=0.5,
    )
    env = env.unwrapped

    _auto_configure_gripper_targets(env, planner)
    _apply_env_obstacles_to_planner(env, planner)
    home_pose = env.agent.tcp.pose.sp

    if planner.open_gripper(t=6) == -1:
        return -1

    grasp = _compute_handle_grasp(env)
    if _move(planner, grasp["pre_grasp_pose"]) == -1:
        return -1
    if _move(planner, grasp["grasp_pose"]) == -1:
        return -1

    res = planner.close_gripper(t=12)
    if res == -1:
        return -1

    open_res = _open_door_smooth(env, planner, grasp)
    if open_res == -1:
        return -1
    if not bool(env.evaluate()["success"][0].item()):
        return open_res
    planner.open_gripper(t=RETURN_HOME_OPEN_GRIPPER_STEPS)
    if _retreat_tcp_backward(planner, POST_OPEN_RETREAT_DIST) == -1:
        return -1
    quit_pos = np.array([-0.3, 0.0, 0.6], dtype=np.float64)
    quit_pose = sapien.Pose(p=quit_pos, q=home_pose.q)
    _move(planner, quit_pose)
    return open_res
