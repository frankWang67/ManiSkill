import gymnasium as gym
import numpy as np
from transforms3d.quaternions import axangle2quat
import mani_skill.envs

def test_quaternion_absolute_pose():
    # 假设您的环境使用的是浮动夹爪或其他机械臂，并且配置为绝对位姿控制 (use_delta=False)
    env_id = "PickCube-v1"
    robot_uid = "floating_robotiq_2f_85_gripper_wristcam" # 请替换为您实际使用的机器人
    control_mode = "pd_ee_pose" # 绝对位姿控制
    
    print(f"初始化环境: {env_id}, 控制模式: {control_mode}")
    env = gym.make(env_id, robot_uids=robot_uid, control_mode=control_mode)
    obs, _ = env.reset()
    
    controller = env.unwrapped.agent.controller
    
    # 打印控制器的动作空间维度，验证您的修改是否生效
    print(f"当前动作空间维度: {controller.action_space.shape}")
    expected_dim = 3 + 4 + 1 # x,y,z(3) + quat(4) + gripper(1) = 8
    if controller.action_space.shape[0] != expected_dim:
        print(f"[警告] 动作维度为 {controller.action_space.shape[0]}，但包含四元数的绝对位姿+夹爪通常需要 {expected_dim} 维！请检查控制器的 _initialize_action_space。")

    # 获取初始位姿作为安全基准
    start_pose = controller.controllers["arm"].ee_pose
    start_pos = start_pose.p.cpu().numpy()[0] if hasattr(start_pose.p, 'cpu') else start_pose.p
    print(f"\n初始 EE 位置: {start_pos}")
    print(f"初始 EE 四元数 (w,x,y,z): {start_pose.q}")

    # ==========================================
    # 测试用例定义
    # ==========================================
    # SAPIEN 的四元数格式为 [w, x, y, z]
    quat_identity = np.array([1.0, 0.0, 0.0, 0.0]) # 保持原样不旋转
    quat_rot_z_90 = axangle2quat([0, 0, 1], np.pi / 2) # 绕Z轴旋转90度 [0.707, 0, 0, 0.707]
    quat_rot_x_90 = axangle2quat([1, 0, 0], np.pi / 2) # 绕X轴旋转90度
    
    test_cases = [
        {"name": "1. 保持初始位置，姿态设为单位四元数 (Identity)", "pos": start_pos, "quat": quat_identity},
        {"name": "2. 保持初始位置，绕Z轴旋转90度", "pos": start_pos, "quat": quat_rot_z_90},
        {"name": "3. 保持初始位置，绕X轴旋转90度", "pos": start_pos, "quat": quat_rot_x_90},
    ]

    for i, case in enumerate(test_cases):
        print(f"\n{'-'*50}")
        print(f"执行测试: {case['name']}")
        print(f"期望目标 - 位置: {case['pos']}, 四元数: {case['quat']}")
        
        # 组装动作 (假设格式为 [x, y, z, w, qx, qy, qz, gripper])
        # 注意：这里假设您的控制器取消了动作空间的 normalize_action，或者您输入的动作在[-1, 1]内
        gripper_action = [0.0]
        action = np.concatenate([case['pos'], case['quat'], gripper_action])
        
        # 给予足够的步数让 IK 和物理引擎收敛
        for step in range(50):
            env.step(action)
            
        # 获取最终实际位姿
        actual_pose = controller.controllers["arm"].ee_pose
        actual_pos = actual_pose.p.cpu().numpy()[0] if hasattr(actual_pose.p, 'cpu') else actual_pose.p
        actual_quat = actual_pose.q.cpu().numpy()[0] if hasattr(actual_pose.q, 'cpu') else actual_pose.q
        
        print(f"实际结果 - 位置: {actual_pos}")
        print(f"实际结果 - 四元数: {actual_quat}")
        
        # 计算误差
        pos_error = np.linalg.norm(actual_pos - case['pos'])
        # 四元数距离: 1 - <q1, q2>^2 (考虑 q 和 -q 表示同一旋转)
        dot_product = np.clip(np.dot(case['quat'], actual_quat), -1.0, 1.0)
        rot_error_rad = 2 * np.arccos(np.abs(dot_product))
        rot_error_deg = np.rad2deg(rot_error_rad)
        
        print(f"误差统计 - 位置误差: {pos_error:.4f} m, 角度误差: {rot_error_deg:.2f} 度")
        
        if pos_error > 0.05 or rot_error_deg > 5.0:
            print(">>> [失败] 实际位姿与四元数指令差异过大！")
            print(">>> 诊断建议:")
            print("    1. 检查 `agents/controllers/pd_ee_pose.py` 中的 `_clip_and_scale_action`，确保四元数没有被当成欧拉角截断 (例如被限制在 rot_lower 范围内)。")
            print("    2. 检查四元数是否传反了 (传入了 x,y,z,w 而不是 w,x,y,z)。")
            print("    3. 检查代码中是否仍有 `euler_angles_to_matrix` 的残留逻辑破坏了四元数。")
        else:
            print(">>> [成功] 姿态跟随准确！")

    env.close()

if __name__ == "__main__":
    test_quaternion_absolute_pose()