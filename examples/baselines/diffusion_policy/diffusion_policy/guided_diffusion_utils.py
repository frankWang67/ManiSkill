import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mani_skill.utils.geometry.rotation_conversions import axis_angle_to_matrix, euler_angles_to_matrix

def quat_conjugate(quat):
    """
    计算四元数的共轭 (用于逆旋转)
    假设格式: [w, x, y, z] (ManiSkill/Sapien 默认是 w 在前)
    共轭: [w, -x, -y, -z]
    """
    # 如果是 batched input (..., 4)
    w, x, y, z = quat.unbind(dim=-1)
    return torch.stack([w, -x, -y, -z], dim=-1)

def quat_apply(quat, vec):
    """
    将四元数旋转应用到向量上 (Standard vectorization implementation)
    quat: (..., 4) [w, x, y, z] or [x, y, z, w], 需注意你的模型输出格式
    vec: (..., 3)
    """
    # 假设 quaternion 格式为 [w, x, y, z] (实部在前)
    # 如果你的模型输出是 [x, y, z, w]，请在此处调整切片
    w, x, y, z = quat.unbind(dim=-1)
    
    # 临时变量辅助计算
    two_s = 2.0 / (quat * quat).sum(dim=-1) # normalization factor if not normalized
    two_s = 2.0 # 假设输出已归一化，通常直接取2
    
    # 简化的旋转逻辑 (PyTorch 官方常用写法)
    uv = torch.cross(quat[..., 1:], vec, dim=-1)
    uuv = torch.cross(quat[..., 1:], uv, dim=-1)
    return vec + two_s * (w.unsqueeze(-1) * uv + uuv)

def point_obb_distance(points, box_center, box_quat, box_extent):
    """
    计算点到旋转长方体 (OBB) 的距离。
    
    Args:
        points: (B, T, N_k, 3) 机器人关键点 (世界坐标)
        box_center: (B, 3) 障碍物中心
        box_quat: (B, 4) 障碍物旋转四元数
        box_extent: (B, 3) 障碍物半长轴 (half_size)
        
    Returns:
        dist: (B, T, N_k) 距离
    """
    # 1. 广播维度处理 (Broadcasting)
    # 假设 points 是 (B, T, N_k, 3)
    # 障碍物 pose 通常是 (B, 3)，需要扩充到 (B, 1, 1, 3) 以便对齐 T 和 N_k
    if box_center.ndim == 2: # (B, 3)
        center_expanded = box_center.unsqueeze(1).unsqueeze(1) # (B, 1, 1, 3)
        quat_expanded = box_quat.unsqueeze(1).unsqueeze(1)     # (B, 1, 1, 4)
        box_extent = box_extent.unsqueeze(1).unsqueeze(1) # (B, 1, 1, 3)
    else: # static single env
        center_expanded = box_center
        quat_expanded = box_quat
        box_extent = box_extent

    # 2. 平移变换 (Translate)
    # 得到相对于 Box 中心的向量 (世界坐标系方向)
    p_centered = points - center_expanded
    
    # 3. 旋转变换 (Rotate into Local Frame)
    # 使用 Box 的逆旋转 (共轭四元数)
    quat_inv = quat_conjugate(quat_expanded)
    p_local = quat_apply(quat_inv, p_centered)
    
    # 4. 在局部坐标系下计算 AABB 距离 (即之前的逻辑)
    # 此时 Box 在局部坐标系下是以原点为中心，轴对齐的
    q = torch.abs(p_local) - box_extent.to(points.device)
    
    # 计算外部距离
    dist_outside = torch.norm(torch.clamp(q, min=0.0), dim=-1)

    # 内部距离部分 (标量, 取最大的那个负分量，即离最近表面的距离)
    # max(q, dim=-1)[0] 找到了离得最近的那个面的距离（在内部都是负数）
    # min(..., 0.0) 确保只在内部生效
    dist_inside = torch.min(torch.max(q, dim=-1)[0], torch.tensor(0.0).to(points.device))
    
    return dist_outside + dist_inside

def integrate_trajectory(current_state, pred_deltas):
    """
    可微地将 Delta 动作序列积分为世界坐标系轨迹。
    
    Args:
        current_state: (B, 6) [x, y, z, ax, ay, az] (当前机器人状态)
        pred_deltas:   (B, T, 6) [dx, dy, dz, d_roll, d_pitch, d_yaw] (模型预测)
        
    Returns:
        abs_traj_pos: (B, T, 3) 世界坐标系下的位置序列
        abs_traj_rot: (B, T, 3, 3) 世界坐标系下的旋转矩阵序列
    """
    B, T, _ = pred_deltas.shape
    
    # 1. 初始化当前状态
    curr_pos = current_state[:, :3]  # (B, 3)
    curr_rot_axis = current_state[:, 3:6]
    curr_rot_mat = axis_angle_to_matrix(curr_rot_axis) # (B, 3, 3)
    
    all_positions = []
    all_rotations = []
    
    # 2. 逐步积分 (For Loop 是可微的，PyTorch 会自动记录计算图)
    for t in range(T):
        # 获取当前步的 Delta
        delta_pos = pred_deltas[:, t, :3] # (B, 3)
        delta_euler = pred_deltas[:, t, 3:6] # (B, 3)
        
        # --- 位置更新 (假设是累加) ---
        # next_pos = curr_pos + delta
        next_pos = curr_pos + delta_pos
        
        # --- 旋转更新 ---
        # 将 Delta Euler 转为矩阵
        delta_rot_mat = euler_angles_to_matrix(delta_euler, "XYZ") # (B, 3, 3)
        
        # 旋转叠加: R_next = R_delta @ R_curr (假设 Delta 是基于当前 Frame 的)
        # 或者 R_next = R_curr @ R_delta (取决于你的训练数据定义)
        # 通常机器人控制中:
        # 如果 delta 是 "在末端坐标系下的移动"，用 R_curr @ R_delta
        # 如果 delta 是 "世界坐标系下的增量"，用 R_delta_global (这就不是简单的 Euler 了)
        # **假设**: 这里的 Delta Euler 是描述末端相对于上一时刻的姿态变化 (Global Frame additivity is rare for Euler).
        # 下面按最常见的 "Global Position Delta + Local Orientation Delta" 实现：
        
        # 修正: 如果 delta euler 是 "世界坐标系欧拉角增量" (很少见但可能):
        # next_rot_mat = euler_angles_to_matrix(curr_euler + delta_euler, "XYZ")
        
        # 修正: 如果 delta euler 是 "相对上一时刻的旋转矩阵乘法" (常见):
        # next_rot_mat = torch.bmm(curr_rot_mat, delta_rot_mat) # Local Rotation
        next_rot_mat = torch.bmm(delta_rot_mat, curr_rot_mat) # Global Rotation
        
        # 存入列表
        all_positions.append(next_pos)
        all_rotations.append(next_rot_mat)
        
        # 更新 curr 状态用于下一步
        curr_pos = next_pos
        curr_rot_mat = next_rot_mat
        
    # Stack 成序列
    abs_traj_pos = torch.stack(all_positions, dim=1) # (B, T, 3)
    abs_traj_rot = torch.stack(all_rotations, dim=1) # (B, T, 3, 3)
    
    return abs_traj_pos, abs_traj_rot

def get_pred_x0(noise_pred, noisy_action_seq, alpha_prod_k):
    return (noisy_action_seq - torch.sqrt(1 - alpha_prod_k) * noise_pred) / torch.sqrt(alpha_prod_k)

def flatten_obstacle_info(obstacles):
    if isinstance(obstacles, list):
        return obstacles

    res = []
    n_envs = len(obstacles)
    n_obstacles = len(obstacles[0])
    for i in range(n_obstacles):
        res.append({
            'center': [],
            'quat': [],
            'extent': [],
        })
        for j in range(n_envs):
            res[i]['center'].append(obstacles[j][i]['center'])
            res[i]['quat'].append(obstacles[j][i]['quat'])
            res[i]['extent'].append(obstacles[j][i]['extent'])
        res[i]['center'] = torch.cat(res[i]['center'], dim=0)
        res[i]['quat'] = torch.cat(res[i]['quat'], dim=0)
        res[i]['extent'] = torch.cat(res[i]['extent'], dim=0)
    return res

def delta_action_obstacle_loss(
    pred_deltas,        # (B, T, 6) 需要求导的变量
    current_state,      # (B, 6) Constant
    robot_corners,      # (N, 3) 夹爪局部角点
    obstacles,     # 障碍物信息
    safety_margin=0.03
):
    # 1. 【积分还原】将 Delta 变为 Absolute Pose
    # 梯度会穿过这里，回传给 pred_deltas
    traj_pos, traj_rot = integrate_trajectory(current_state, pred_deltas)
    
    # 2. 【运动学变换】计算角点在世界坐标系的位置
    # traj_pos: (B, T, 3)
    # traj_rot: (B, T, 3, 3)
    # corners:  (N, 3)
    
    # Expand dims for broadcasting
    # (B, T, 1, 3, 3)
    rot_expanded = traj_rot.unsqueeze(2) 
    # (1, 1, N, 3, 1) -> 转置为列向量做矩阵乘法
    corners_expanded = robot_corners.view(1, 1, robot_corners.shape[0], 3, 1)
    
    # 旋转: R * p_local
    # result: (B, T, N, 3, 1) -> squeeze -> (B, T, N, 3)
    corners_rotated = torch.matmul(rot_expanded, corners_expanded).squeeze(-1)
    
    # 平移: + pos
    # (B, T, 1, 3) + (B, T, N, 3)
    corners_world = corners_rotated + traj_pos.unsqueeze(2)
    
    # 3. 【SDF Loss】计算距离场 Loss
    total_loss = 0
    obstacles = flatten_obstacle_info(obstacles)
    for obs in obstacles:
        # 调用之前定义的点到Box距离函数
        dists = point_obb_distance(
            corners_world, 
            obs['center'].to(pred_deltas.device),
            obs['quat'].to(pred_deltas.device), 
            obs['extent'].to(pred_deltas.device),
        )
        
        penetration = safety_margin - dists
        # cost = torch.relu(penetration).pow(2)
        cost = torch.relu(penetration)
        total_loss += cost.sum()
        
    return total_loss

def get_guidance_strength(k, num_diffusion_steps):
    h1 = 1.0
    h2 = 50.0
    h3 = 0.7
    
    t = k / num_diffusion_steps
    gamma = h1 / (1 + torch.exp(-h2 * (h3 - t)))

    return gamma

def visualize_grad(all_grads_np, b=0):
    # 1. 设置更美观的绘图风格
    plt.style.use('seaborn-v0_8-whitegrid') # 或者 'ggplot'

    # 2. 动态计算布局：假设你有16个步数，使用 2行8列 比较适合宽屏
    rows, cols = 2, 8  
    fig, axes = plt.subplots(rows, cols, figsize=(24, 10))
    axes = axes.flatten() # 展平方便遍历

    # 定义颜色列表，确保全局统一
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    dim_labels = [f'dim {d+1}' for d in range(7)]

    for i in range(len(axes)):
        # 确保不越界 (如果 all_grads_np 只有16个step)
        if i < all_grads_np.shape[1]: 
            ax = axes[i]
            
            # 3. 绘制每一维度的曲线
            # all_grads_np[b, i, :, :] shape 应该是 (TimeSteps, Dims)
            data = all_grads_np[b, i, :, :]
            
            for d in range(data.shape[1]):
                ax.plot(data[:, d], label=dim_labels[d], color=colors[d], alpha=0.8, linewidth=1.5)

            # 4. 标题优化：只用简单的 Step X
            ax.set_title(f"Step {i}", fontsize=12, fontweight='bold')
            
            # 5. 坐标轴美化
            # 只有特定子图显示标签，避免重复
            # 如果是最后一行，显示X标签
            if i >= (rows - 1) * cols:
                ax.set_xlabel("Time", fontsize=10)
            
            # 科学计数法：因为你的梯度值跨度很大 (从1e-5 到 8)
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))
            ax.grid(True, linestyle='--', alpha=0.6)

        else:
            # 隐藏多余的子图
            axes[i].axis('off')

    # 6. 全局图例 (Global Legend)
    # 只需要获取第一个子图的句柄即可
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, 
        labels, 
        loc='lower center',       # 以底部中心为锚点
        bbox_to_anchor=(0.5, 0.93), # 放在画布高度 93% 的位置
        ncol=7,                   # 7 列横排
        fontsize=14,              # 字体稍微大一点
        frameon=False)            # 无边框

    # 7. 全局大标题 (可选)
    fig.suptitle(f'Gradient Flow across Diffusion Steps (Batch {b})', fontsize=16, y=1.08)

    plt.show()
