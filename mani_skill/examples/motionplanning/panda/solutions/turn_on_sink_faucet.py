import numpy as np
import sapien

from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)


JOINT_VEL_LIMITS = 0.5
JOINT_ACC_LIMITS = 0.5
PRE_GRASP_BACKOFF = 0.05
PRE_GRASP_LIFT = 0.02
HANDLE_SHAFT_HEIGHT = 0.02
ARC_STEPS = 10
RETURN_HOME_STEPS = 30
DETOUR_PROB = 0.85
DETOUR_ATTEMPTS = 6
DETOUR_BACKOFF_RANGE = (0.03, 0.07)
DETOUR_WORLD_X_RANGE = (-0.05, 0.25)
DETOUR_WORLD_Y_RANGE = (-0.15, 0.15)
DETOUR_LIFT_RANGE = (0.01, 0.04)


def _first_batch_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    if x.ndim == 0:
        return np.asarray([float(x)], dtype=np.float64)
    if x.ndim == 1:
        return x.astype(np.float64)
    return x[0].astype(np.float64)


def _normalize(vec):
    vec = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm


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


def _sample_uniform(rng, low, high):
    return float(rng.uniform(low, high))


def _move(planner, pose: sapien.Pose, dry_run: bool = False):
    res = planner.move_to_pose_with_screw(pose, dry_run=dry_run)
    if res == -1:
        res = planner.move_to_pose_with_RRTConnect(pose, dry_run=dry_run)
    return res


def _sample_detour_pose(turn_plan, rng):
    approaching = _normalize(turn_plan["approaching"])

    detour_center = (
        np.asarray(turn_plan["pre_grasp_pose"].p, dtype=np.float64)
        - approaching * _sample_uniform(rng, *DETOUR_BACKOFF_RANGE)
        + np.array(
            [
                _sample_uniform(rng, *DETOUR_WORLD_X_RANGE),
                _sample_uniform(rng, *DETOUR_WORLD_Y_RANGE),
                _sample_uniform(rng, *DETOUR_LIFT_RANGE),
            ],
            dtype=np.float64,
        )
    )
    return sapien.Pose(p=detour_center, q=turn_plan["grasp_pose"].q)


def _move_to_faucet_with_detour(planner, turn_plan, rng):
    if _sample_uniform(rng, 0.0, 1.0) < DETOUR_PROB:
        for _ in range(DETOUR_ATTEMPTS):
            detour_pose = _sample_detour_pose(turn_plan, rng)
            if _move(planner, detour_pose, dry_run=True) == -1:
                continue
            if _move(planner, detour_pose) != -1:
                break
    if _move(planner, turn_plan["pre_grasp_pose"]) == -1:
        return -1
    return _move(planner, turn_plan["grasp_pose"])


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


def _auto_configure_gripper_targets(env, planner):
    gripper_ctrl = env.agent.controller.controllers["gripper_active"]
    planner.OPEN = float(gripper_ctrl.single_action_space.low[0])
    planner.CLOSED = float(gripper_ctrl.single_action_space.high[0])

def _compute_turn_plan(env):
    kinematics = env.get_faucet_kinematics()
    handle_pos = _first_batch_np(kinematics["handle_pos"])
    hinge_pos = _first_batch_np(kinematics["hinge_pos"])
    hinge_axis = _normalize(_first_batch_np(kinematics["hinge_axis"]))

    approaching = hinge_axis
    approaching[2] = -0.5
    approaching = _normalize(approaching)

    radial = handle_pos - hinge_pos
    radial = radial - hinge_axis * np.dot(radial, hinge_axis)

    closing = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    grasp_center = handle_pos.copy()
    grasp_center[0] -= 0.02
    grasp_center[1] += 0.02
    grasp_center[2] += 0.04
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, grasp_center)
    pre_grasp_pose = sapien.Pose(
        p=grasp_center - np.array([0.0, 0.1, 0.0]),
        q=grasp_pose.q,
    )
    
    shaft_rel = grasp_center - handle_pos - np.array([0.0, 0.02, 0.0])

    return dict(
        handle_pos=handle_pos,
        hinge_pos=hinge_pos,
        hinge_axis=hinge_axis,
        shaft_rel=shaft_rel,
        approaching=approaching,
        closing=closing,
        grasp_pose=grasp_pose,
        pre_grasp_pose=pre_grasp_pose,
    )

def _turn_faucet(env, planner, turn_plan):
    current_angle = float(_first_batch_np(env.current_angle)[0])
    target_angle = float(_first_batch_np(env.motionplan_target_qpos)[0])
    delta_angle = target_angle - current_angle
    delta_angle *= 4

    res = None
    for frac in np.linspace(1.0 / ARC_STEPS, 1.0, ARC_STEPS):
        angle = delta_angle * frac
        target_center = (
            turn_plan["handle_pos"]
            + _rotate_about_axis(turn_plan["shaft_rel"], turn_plan["hinge_axis"], angle)
            + turn_plan["closing"] * 0.02
        )
        closing = _rotate_about_axis(turn_plan["closing"], turn_plan["hinge_axis"], angle * 0.5)
        target_pose = env.agent.build_grasp_pose(turn_plan["approaching"], closing, target_center)
        res = _move(planner, target_pose)
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
        joint_vel_limits=JOINT_VEL_LIMITS,
        joint_acc_limits=JOINT_ACC_LIMITS,
    )
    env = env.unwrapped

    try:
        rng = getattr(env, "_episode_rng", None)
        if rng is None:
            rng = np.random.default_rng(seed)
        _auto_configure_gripper_targets(env, planner)
        dof = len(planner.planner.joint_vel_limits)
        home_qpos = env.agent.robot.get_qpos()[0, :dof].cpu().numpy()

        turn_plan = _compute_turn_plan(env)

        if planner.close_gripper(t=8) == -1:
            return -1
        if _move_to_faucet_with_detour(planner, turn_plan, rng) == -1:
            return -1

        res = _turn_faucet(env, planner, turn_plan)
        if res == -1:
            return -1

        # retreat_pose = turn_plan["pre_grasp_pose"]
        # _move(planner, retreat_pose)
        _move_to_qpos_linearly(planner, home_qpos, steps=RETURN_HOME_STEPS)
        return planner.open_gripper(t=8)
    finally:
        planner.close()
