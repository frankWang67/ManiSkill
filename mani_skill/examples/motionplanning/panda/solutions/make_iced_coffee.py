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

# ---------------------------------------------------------------------------
# Optional instrumentation — set _INSTRUMENT = True before calling solve()
# to collect per-episode metrics (grasp retries, path quality, TCP trace).
# ---------------------------------------------------------------------------
_INSTRUMENT = False
_INSTRUMENT_DATA = None  # populated per call to solve()


def _instrument_reset():
    global _INSTRUMENT_DATA
    _INSTRUMENT_DATA = {
        "grasp_attempts": 0,
        "grasp_success": False,
        "screw_calls": 0,
        "rrt_calls": 0,
        "rrt_rejected": 0,
        "tcp_trace": [],        # per-step {"before", "after"}
        "move_segments": [],    # per-_move {"label", "method", "rrt_rejected", "trace_start", "trace_end"}
        "approaching": None,
        "place_yaw": 0.0,
    }


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
GRASP_CLOSING_YAW_JITTER_RANGE = (-0.60, 0.60)

# ---------------------------------------------------------------------------
# RRTConnect path quality thresholds (prevent large-loop trajectories)
# ---------------------------------------------------------------------------
RRT_REPLAN_ATTEMPTS = 4
MAX_RRT_PLAN_STEPS = 140
MAX_RRT_CUM_JOINT_TRAVEL_BASE = 1.8
MAX_RRT_CUM_JOINT_TRAVEL_PER_M = 8.0
MAX_RRT_CUM_JOINT_TRAVEL_PER_RAD = 1.8
MAX_RRT_MAX_STEP = 0.12
MAX_RRT_DETOUR_RATIO = 3.0

# ---------------------------------------------------------------------------
# Grasp retry
# ---------------------------------------------------------------------------
MAX_GRASP_RETRIES = 2
GRASP_TEST_LIFT = 0.025
GRASP_TEST_Z_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Rotation diversity
# ---------------------------------------------------------------------------
APPROACH_TILT_RANGE = 0.26
PLACE_YAW_JITTER = 0.35

# ---------------------------------------------------------------------------
# Episode time limit
# ---------------------------------------------------------------------------
MAX_EPISODE_STEPS = 500


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


def _rotate_vector(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate vector v around axis by angle (radians) using Rodrigues' formula."""
    v = np.asarray(v, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return cos_a * v + sin_a * np.cross(axis, v) + (1 - cos_a) * np.dot(axis, v) * axis


def _path_metrics(result):
    qpos = np.asarray(result["position"], dtype=np.float64)
    if qpos.shape[0] <= 1:
        return dict(
            steps=int(qpos.shape[0]), cum_joint=0.0, direct_joint=0.0,
            max_step=0.0, detour_ratio=1.0,
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
    current_tcp_pose = _to_sapien_pose(
        getattr(planner.env_agent, "tcp_pose", None)
    )
    if current_tcp_pose is None:
        current_tcp_pose = planner.env_agent.tcp.pose.sp
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


def _best_rrt_plan_without_loop(
    planner: PandaArmMotionPlanningSolver, pose: sapien.Pose
):
    best_result = None
    best_metrics = None
    rejected_count = 0
    for _ in range(RRT_REPLAN_ATTEMPTS):
        result = _plan_rrt_result(planner, pose)
        if result["status"] != "Success":
            continue
        is_reasonable, metrics = _rrt_plan_is_reasonable(planner, pose, result)
        if not is_reasonable:
            rejected_count += 1
            continue
        if best_result is None or metrics["cum_joint"] < best_metrics["cum_joint"]:
            best_result = result
            best_metrics = metrics
    if _INSTRUMENT and rejected_count > 0:
        _INSTRUMENT_DATA["rrt_rejected"] += rejected_count
    return best_result


def _check_step_limit(planner: PandaArmMotionPlanningSolver) -> bool:
    return planner.elapsed_steps >= MAX_EPISODE_STEPS


def _move(planner: PandaArmMotionPlanningSolver, pose: sapien.Pose, label: str = ""):
    if _check_step_limit(planner):
        return -1
    res = planner.move_to_pose_with_screw(pose)
    if res != -1:
        if _INSTRUMENT:
            _INSTRUMENT_DATA["screw_calls"] += 1
            _INSTRUMENT_DATA["move_segments"].append(
                {"label": label, "method": "screw", "rrt_rejected": 0,
                 "trace_start": len(_INSTRUMENT_DATA["tcp_trace"])})
        return res
    if _INSTRUMENT:
        _INSTRUMENT_DATA["rrt_calls"] += 1
    rrt_result = _best_rrt_plan_without_loop(planner, pose)
    if rrt_result is None:
        if _INSTRUMENT:
            _INSTRUMENT_DATA["move_segments"].append(
                {"label": label, "method": "rrtconnect", "rrt_rejected": -1,
                 "trace_start": len(_INSTRUMENT_DATA["tcp_trace"])})
        return -1
    if _INSTRUMENT:
        _INSTRUMENT_DATA["move_segments"].append(
            {"label": label, "method": "rrtconnect", "rrt_rejected": 0,
             "trace_start": len(_INSTRUMENT_DATA["tcp_trace"])})
    return planner.follow_path(rrt_result)


def _check_grasp(planner: PandaArmMotionPlanningSolver, env: MakeIcedCoffeeEnv,
                 ice_z_before: float, tcp_z_before: float, test_lift_z: float) -> bool:
    """Do a small lift and check if the ice cube moved with the TCP."""
    test_pose = sapien.Pose(
        p=np.array([
            float(env.agent.tcp.pose.p[0, 0].detach().cpu().numpy()),
            float(env.agent.tcp.pose.p[0, 1].detach().cpu().numpy()),
            test_lift_z,
        ], dtype=np.float64),
        q=env.agent.tcp.pose.q[0].detach().cpu().numpy().astype(np.float64),
    )
    if _move(planner, test_pose, "test_lift") == -1:
        return False
    tcp_z_after = float(env.agent.tcp.pose.p[0, 2].detach().cpu().numpy())
    ice_z_after = float(env.ice_cube.pose.p[0, 2].detach().cpu().numpy())
    tcp_dz = tcp_z_after - tcp_z_before
    ice_dz = ice_z_after - ice_z_before
    return ice_dz >= GRASP_TEST_Z_THRESHOLD * max(tcp_dz, 1e-6)


def solve(env: MakeIcedCoffeeEnv, seed=None, debug: bool = False, vis: bool = False):
    if _INSTRUMENT:
        _instrument_reset()

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

    # ---- Instrumentation: hook env.step for per-step TCP trace --------------
    if _INSTRUMENT:
        _orig_env_step = planner.env.step

        def _hooked_step(action):
            tcp_before = (
                env.agent.tcp.pose.p[0].detach().cpu().numpy()
                .astype(np.float64).copy()
            )
            result = _orig_env_step(action)
            tcp_after = (
                env.agent.tcp.pose.p[0].detach().cpu().numpy()
                .astype(np.float64).copy()
            )
            _INSTRUMENT_DATA["tcp_trace"].append(
                {"before": tcp_before, "after": tcp_after}
            )
            return result

        planner.env.step = _hooked_step

    # Match planner gripper states to this env's actual action-space bounds.
    gripper_ctrl = env.agent.controller.controllers["gripper_active"]
    planner.OPEN = float(gripper_ctrl.single_action_space.low[0])
    planner.CLOSED = float(gripper_ctrl.single_action_space.high[0])

    rng = np.random.default_rng(seed)

    home_pose = env.agent.tcp.pose.sp

    # ---- Rotation diversity: approach tilt ---------------------------------
    base_approach = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    tilt_angle = float(rng.uniform(0.0, APPROACH_TILT_RANGE))
    if tilt_angle > 1e-6:
        tilt_axis = np.array([rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), 0.0],
                             dtype=np.float64)
        tilt_axis = tilt_axis / (np.linalg.norm(tilt_axis) + 1e-12)
        approaching = _rotate_vector(base_approach, tilt_axis, tilt_angle)
    else:
        approaching = base_approach.copy()

    if _INSTRUMENT:
        _INSTRUMENT_DATA["approaching"] = approaching.copy()

    target_closing = (
        env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    ).astype(np.float64)
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

    # ---- Compute grasp pose ------------------------------------------------
    def _compute_grasp_pose(env, approaching, target_closing):
        obb = get_actor_obb(env.ice_cube)
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=FINGER_LENGTH,
        )
        return env.agent.build_grasp_pose(
            approaching, grasp_info["closing"], grasp_info["center"]
        )

    grasp_pose = _compute_grasp_pose(env, approaching, target_closing)

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
        # ---- Grasp with retry -----------------------------------------------
        grasped = False

        for attempt in range(MAX_GRASP_RETRIES + 1):
            if _INSTRUMENT:
                _INSTRUMENT_DATA["grasp_attempts"] = attempt + 1

            # On retry, the arm is at test_lift with the gripper closed.
            # Open the gripper at the current position first, then approach
            # so the gripper action overlaps with the pre-move preparation
            # instead of adding a separate stationary block at pre_grasp.
            if attempt == 0:
                planner.open_gripper(t=int(rng.integers(5, 9)))
            else:
                planner.open_gripper(t=int(rng.integers(3, 5)))
                planner.close_gripper(t=int(rng.integers(6, 10)))
            if _check_step_limit(planner):
                return -1

            if _move(planner, pre_grasp_pose, "approach") == -1:
                if attempt < MAX_GRASP_RETRIES:
                    continue
                return -1
            if attempt > 0:
                planner.open_gripper(t=int(rng.integers(5, 9)))

            if _move(planner, grasp_pose, "grasp") == -1:
                if attempt < MAX_GRASP_RETRIES:
                    continue
                return -1

            planner.close_gripper(t=int(rng.integers(6, 10)))
            if _check_step_limit(planner):
                return -1

            # Test-lift verification
            ice_z_before = float(env.ice_cube.pose.p[0, 2].detach().cpu().numpy())
            tcp_z_before = float(env.agent.tcp.pose.p[0, 2].detach().cpu().numpy())
            test_lift_z = max(tcp_z_before + GRASP_TEST_LIFT, float(grasp_pose.p[2]) + GRASP_TEST_LIFT)

            if _check_grasp(planner, env, ice_z_before, tcp_z_before, test_lift_z):
                grasped = True
                if _INSTRUMENT:
                    _INSTRUMENT_DATA["grasp_success"] = True
                break

            # Retry: re-randomize grasp for next attempt.
            # The arm is at test_lift; the next iterationʼs open_gripper + _move
            # will transition to the new pre_grasp pose.
            if attempt < MAX_GRASP_RETRIES:
                tilt_angle = float(rng.uniform(0.0, APPROACH_TILT_RANGE))
                if tilt_angle > 1e-6:
                    tilt_axis = np.array(
                        [rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), 0.0],
                        dtype=np.float64,
                    )
                    tilt_axis = tilt_axis / (np.linalg.norm(tilt_axis) + 1e-12)
                    approaching = _rotate_vector(base_approach, tilt_axis, tilt_angle)
                else:
                    approaching = base_approach.copy()

                grasp_closing_yaw_jitter = float(
                    rng.uniform(*GRASP_CLOSING_YAW_JITTER_RANGE)
                )
                tcp_closing_retry = (
                    env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1]
                    .cpu().numpy()
                    .astype(np.float64)
                )
                target_closing = tcp_closing_retry.copy()
                target_closing[2] = 0.0
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
                tcn = np.linalg.norm(target_closing)
                if tcn < 1e-8:
                    target_closing = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                else:
                    target_closing /= tcn

                grasp_pose = _compute_grasp_pose(env, approaching, target_closing)
                pre_grasp_z = float(rng.uniform(*PRE_GRASP_Z_RANGE))
                pre_grasp_pose = grasp_pose * sapien.Pose([0.0, 0.0, -pre_grasp_z])

        # ---- Lift and transfer ----------------------------------------------
        if not grasped:
            # Continue anyway — the episode will be a failure, providing
            # negative data for the diffusion policy to learn from.
            pass

        if _move(planner, lift_pose, "lift") == -1:
            return -1

        transfer_q = env.agent.tcp.pose.q[0].detach().cpu().numpy().astype(np.float64)

        # Rotation diversity: jitter place orientation
        place_yaw = rng.uniform(-PLACE_YAW_JITTER, PLACE_YAW_JITTER)
        if _INSTRUMENT:
            _INSTRUMENT_DATA["place_yaw"] = float(place_yaw)
        if abs(place_yaw) > 1e-8:
            # Apply yaw rotation to transfer_q
            half_yaw = place_yaw / 2.0
            yaw_quat = np.array(
                [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)], dtype=np.float64
            )
            # Quaternion multiplication: yaw_quat * transfer_q
            tw = yaw_quat[0] * transfer_q[0] - yaw_quat[3] * transfer_q[3]
            tx = yaw_quat[0] * transfer_q[1] + yaw_quat[3] * transfer_q[2]
            ty = yaw_quat[0] * transfer_q[2] - yaw_quat[3] * transfer_q[1]
            tz = yaw_quat[0] * transfer_q[3] + yaw_quat[3] * transfer_q[0]
            place_q = np.array([tw, tx, ty, tz], dtype=np.float64)
        else:
            place_q = transfer_q.copy()

        above_cup_z = float(rng.uniform(*ABOVE_CUP_Z_RANGE))
        above_cup_pose = sapien.Pose(
            p=place_pos + np.array([0.0, 0.0, above_cup_z], dtype=np.float64),
            q=place_q,
        )
        place_pose = sapien.Pose(p=place_pos, q=place_q)

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
                q=place_q,
            )
            _move(planner, detour_pose, "transfer_detour")

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
                q=place_q,
            )
            _move(planner, orbit_pose, "orbit")

        if _move(planner, above_cup_pose, "above_cup") == -1:
            return -1

        # 4) Lower into the cup and release.
        if _move(planner, place_pose, "place") == -1:
            return -1
        planner.open_gripper(t=int(rng.integers(5, 8)))
        if _check_step_limit(planner):
            return -1

        # 5) Leave the cup and return home.
        if _move(planner, above_cup_pose, "retreat_above") == -1:
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
                q=place_q,
            )
            _move(planner, retreat_pose, "retreat_detour")

        res = _move(planner, home_pose, "go_home")
        return res
    finally:
        if _INSTRUMENT:
            planner.env.step = _orig_env_step
        planner.close()
