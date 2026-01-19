import numpy as np
import sapien

from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb
)

# -------------------------------------------------------------------------- #
# Task 1: Deep Box (修复 Screw Plan Failed)
# -------------------------------------------------------------------------- #
def solveDeepBox(env, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    env = env.unwrapped
    
    FINGER_LENGTH = 0.025
    obb = get_actor_obb(env.cube)
    
    # -------------------------------------------------------------------------- #
    # 1. 计算抓取姿态
    # -------------------------------------------------------------------------- #
    
    # 抓取策略：严格垂直向下 (Z轴负方向)
    approaching = np.array([0, 0, -1])
    
    # 获取 TCP 当前的 closing 方向 (用于参考)
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    
    # 修正抓取中心 (Batch 0)
    grasp_center = grasp_info["center"]
    
    grasp_pose = env.agent.build_grasp_pose(approaching, grasp_info["closing"], grasp_center)

    # -------------------------------------------------------------------------- #
    # 2. 执行抓取 (Pick)
    # -------------------------------------------------------------------------- #

    # Pre-grasp: 大幅提高高度，确保飞越箱壁
    # 相对抬升 25cm (足以覆盖 25cm 高的墙壁)
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.25])

    # Step A: 飞到箱子正上方
    planner.move_to_pose_with_screw(reach_pose)

    # Step B: 垂直下潜
    # 为了避免太窄碰撞，先张开夹爪
    planner.open_gripper() 
    
    try:
        planner.move_to_pose_with_screw(grasp_pose)
    except Exception as e:
        if debug:
            print(f"Screw plan failed: {e}, trying to nudge grasp pose higher...")
        # 备选方案：抓高一点
        grasp_pose_higher = grasp_pose * sapien.Pose([0, 0, -0.01])
        planner.move_to_pose_with_screw(grasp_pose_higher)

    planner.close_gripper()

    # Step C: 垂直拔出 (回到箱子上方)
    planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # 3. 执行放置 (Safe Travel & Place) [核心修改]
    # -------------------------------------------------------------------------- #

    # 目标位置
    goal_p = env.goal_site.pose.p[0].cpu().numpy()
    # 目标姿态：保持抓取时的姿态
    goal_pose = sapien.Pose(goal_p, grasp_pose.q)
    
    # Step D: 平移到目标正上方 (Pre-place)
    # 这是一个高空过渡点，避免横向移动时撞到箱子
    # 高度设置为目标上方 25cm (与 reach_pose 保持一致的飞行高度)
    pre_place_pose = goal_pose * sapien.Pose([0, 0, -0.25])
    
    # 执行高空平移
    planner.move_to_pose_with_screw(pre_place_pose)

    # Step E: 垂直下降到目标
    res = planner.move_to_pose_with_screw(goal_pose)
    
    # Step F: 张开夹爪 (放置)
    planner.open_gripper()
    
    # Step G: 垂直抬起 (离场)
    planner.move_to_pose_with_screw(pre_place_pose)

    planner.close()
    return res


# -------------------------------------------------------------------------- #
# Task 2: Shelf (修复 Matrix Dimension Error)
# -------------------------------------------------------------------------- #
def solveShelf(env, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    env = env.unwrapped
    
    FINGER_LENGTH = 0.025
    obb = get_actor_obb(env.cube)
    
    # -------------------------------------------------------------------------- #
    # 1. 计算抓取姿态 (Horizontal Approach + Camera Up)
    # -------------------------------------------------------------------------- #
    
    # 1.1 计算水平接近方向
    # 获取 Cube 旋转矩阵 (Batch 0)
    cube_mat = env.cube.pose.to_transformation_matrix()[0].cpu().numpy()
    cube_rot = cube_mat[:3, :3]
    
    # 假设架子开口方向对应 Cube 的 X 轴
    local_x = np.array([1, 0, 0])
    approaching = cube_rot @ local_x
    approaching[2] = 0 # 强制抹平 Z 轴分量，确保水平接近
    if np.linalg.norm(approaching) < 1e-3:
        approaching = np.array([1, 0, 0])
    else:
        approaching = approaching / np.linalg.norm(approaching)
    
    # 1.2 计算闭合方向 (Closing Vector) 以强制相机朝上 [核心修改]
    # Panda 相机在 Hand +X 面。要让相机朝上，即 Hand X 指向 Global Z ([0, 0, 1])。
    # Hand Z (Approaching) 已知。
    # 根据右手定则：Hand Y (Closing) = Hand Z (Approaching) x Hand X (Global Z)
    global_z = np.array([0, 0, 1])
    camera_up_closing = np.cross(approaching, global_z)
    camera_up_closing /= np.linalg.norm(camera_up_closing)
    
    # 1.3 构建抓取姿态
    # 我们只需要 OBB 的 center，closing 方向完全由上面计算决定
    target_closing_dummy = camera_up_closing 
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing_dummy,
        depth=FINGER_LENGTH,
    )
    
    # 使用强制计算的 camera_up_closing 构建姿态
    grasp_pose = env.agent.build_grasp_pose(approaching, camera_up_closing, grasp_info["center"])

    # -------------------------------------------------------------------------- #
    # 2. 执行抓取 (Pick)
    # -------------------------------------------------------------------------- #
    
    # Pre-grasp: 沿接近方向退后 20cm
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.2])

    # Step A: 移动到架子外 (Pre-grasp)
    planner.move_to_pose_with_screw(reach_pose)
    
    # Step B: 水平插入 (Grasp Pose)
    planner.move_to_pose_with_screw(grasp_pose)
    
    # Step C: 闭合夹爪
    planner.close_gripper()
    
    # Step D: 水平拔出 (回到 Pre-grasp)
    planner.move_to_pose_with_screw(reach_pose)
    
    # -------------------------------------------------------------------------- #
    # 3. 执行放置 (Place Vertically) [核心修改]
    # -------------------------------------------------------------------------- #
    
    # 目标位置
    goal_p = env.goal_site.pose.p[0].cpu().numpy()
    
    # 3.1 构建“竖着放”的姿态 (Top-Down)
    # Approach 指向下方 (-Z)
    place_approaching = np.array([0, 0, -1])
    # Closing 可以是水平面任意方向，这里选 X 轴
    place_closing = np.array([1, 0, 0])
    
    # 构建放置姿态 (TCP 在目标位置夹着物体)
    goal_pose = env.agent.build_grasp_pose(place_approaching, place_closing, goal_p)
    
    # 3.2 高空过渡点 (Pre-place)
    # 在目标上方 15cm 处悬停，用于完成姿态旋转 (从横向变竖向) 并避免碰撞
    pre_place_pose = goal_pose * sapien.Pose([0, 0, -0.15])
    
    # Step E: 移动到目标上方 (螺旋运动会自动处理旋转)
    planner.move_to_pose_with_screw(pre_place_pose)
    
    # Step F: 垂直下降到目标
    res = planner.move_to_pose_with_screw(goal_pose)
    
    # Step G: 张开夹爪
    planner.open_gripper()
    
    # Step H: 垂直抬起 (离场)
    planner.move_to_pose_with_screw(pre_place_pose)

    planner.close()
    return res

# -------------------------------------------------------------------------- #
# Task 3: Barrier (修复 Index Error)
# -------------------------------------------------------------------------- #
def solveBarrier(env, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    env = env.unwrapped
    
    FINGER_LENGTH = 0.025
    obb = get_actor_obb(env.cube)
    
    # -------------------------------------------------------------------------- #
    # 1. 计算抓取姿态
    # -------------------------------------------------------------------------- #
    
    # 垂直抓取
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    grasp_pose = env.agent.build_grasp_pose(approaching, grasp_info["closing"], grasp_info["center"])

    # -------------------------------------------------------------------------- #
    # 2. 执行抓取 (Pick)
    # -------------------------------------------------------------------------- #

    # Pre-grasp: 位于物体上方 15cm
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.15])
    
    # Step A: 移动到预备点
    planner.move_to_pose_with_screw(reach_pose)
    
    # Step B: 抓取
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()
    
    # -------------------------------------------------------------------------- #
    # 3. 翻越障碍 (High Lift & Traverse) [核心修改]
    # -------------------------------------------------------------------------- #
    
    # 定义安全巡航高度 (Safe Z)
    # 障碍物最高约 0.35m，设定 0.5m 为安全高度
    safe_z = 0.3
    
    # Step C: 垂直抬升至安全高度 (High Lift)
    # 保持抓取时的 XY 位置，只提升 Z
    current_p = grasp_pose.p
    lift_high_pose = sapien.Pose([current_p[0], current_p[1], safe_z], grasp_pose.q)
    planner.move_to_pose_with_screw(lift_high_pose)

    # 目标位置 (Batch 0)
    goal_p = env.goal_site.pose.p[0].cpu().numpy()
    
    # Step D: 平移至目标正上方 (Traverse)
    # 保持高度 safe_z，移动到目标 XY
    pre_place_pose = sapien.Pose([goal_p[0], goal_p[1], safe_z], grasp_pose.q)
    planner.move_to_pose_with_screw(pre_place_pose)

    # Step E: 垂直下降至目标 (Descend)
    goal_pose = sapien.Pose(goal_p, grasp_pose.q)
    res = planner.move_to_pose_with_screw(goal_pose)
    
    # Step F: 张开夹爪 (放置)
    planner.open_gripper()

    # Step G: 抬起离场
    planner.move_to_pose_with_screw(pre_place_pose)

    planner.close()
    return res