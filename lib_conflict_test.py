import numpy as np
import torch
import scipy.spatial.transform as st
# 导入 ManiSkill 原生的转换函数 (等价于 PyTorch3D)
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    quaternion_to_matrix,
)

def main():
    # 设定显示精度，方便对比
    np.set_printoptions(precision=4, suppress=True)
    
    # 1. 定义一个测试的欧拉角 (Roll, Pitch, Yaw)
    # 比如：绕X转30度，绕Y转45度，绕Z转60度
    euler_deg = np.array([30.0, 45.0, 60.0])
    euler_rad = np.deg2rad(euler_deg)
    print(f"【初始设定】原始欧拉角 (度): {euler_deg}")
    print("=" * 60)

    # ---------------------------------------------------------
    # 实验一：将欧拉角转为旋转矩阵 —— 看看两个库的差异
    # ---------------------------------------------------------
    # [Scipy 路线] 
    # 假设你之前生成 6D 标签时用的是小写 'xyz' (内旋) 或大写 'XYZ' (外旋)
    # 这里以你常用的 'xyz' 为例
    mat_scipy = st.Rotation.from_euler('xyz', euler_rad).as_matrix()
    
    # [ManiSkill / PyTorch3D 路线]
    # ManiSkill 物理引擎底层使用的约定
    euler_tensor = torch.tensor(euler_rad, dtype=torch.float32)
    mat_ms = euler_angles_to_matrix(euler_tensor, "XYZ").numpy()

    print("\n【实验一：欧拉角 -> 旋转矩阵】")
    print("Scipy 生成的矩阵 (你旧数据集里的 6D 标签长这样):")
    print(mat_scipy)
    print("\nManiSkill(PyTorch3D) 生成的矩阵 (物理引擎真正期待的矩阵):")
    print(mat_ms)
    
    # 揭示差异：你会发现它们通常互为转置 (Transpose)，或者是完全不同的矩阵！
    diff = np.abs(mat_scipy - mat_ms).max()
    print(f"\n-> 两个矩阵的最大差异: {diff:.4f} (如果不为0，说明数学约定完全不同！)")
    if np.allclose(mat_scipy, mat_ms.T):
        print("-> 💡 发现规律：在这个特定约定下，Scipy的矩阵刚好是 ManiSkill矩阵的转置！")
    print("=" * 60)

    # ---------------------------------------------------------
    # 实验二：复现你最初的“负负得正”奇迹 (欧拉角控制器时期)
    # ---------------------------------------------------------
    # DP 网络预测出了 Scipy 约定的矩阵
    predicted_mat_scipy = mat_scipy 
    
    # 在旧代码中，你用 Scipy 把矩阵转回欧拉角发给环境
    recovered_euler_scipy = st.Rotation.from_matrix(predicted_mat_scipy).as_euler('xyz')
    recovered_deg_scipy = np.rad2deg(recovered_euler_scipy)
    
    print("\n【实验二：“负负得正”再现 (旧版欧拉角控制为什么能行)】")
    print("1. DP网络输出扭曲的 Scipy 矩阵")
    print("2. 你用 Scipy 把矩阵解析回欧拉角")
    print(f"-> 提取出的欧拉角 (度): {recovered_deg_scipy}")
    print("-> 结论：用解铃人还须系铃人的方式，你完美还原了初始角度，物理引擎拿到正确的角度，所以能动！")
    print("=" * 60)

    # ---------------------------------------------------------
    # 实验三：复现你换成“四元数”后的灾难 (Y轴反向/对称现象)
    # ---------------------------------------------------------
    print("\n【实验三：四元数灾难再现 (当前报错的原因)】")
    print("1. DP网络依然输出扭曲的 Scipy 矩阵")
    
    # 你现在的代码：把 Scipy 的矩阵直接变成了四元数 (哪怕你对齐了 wxyz)
    quat_from_scipy_mat = st.Rotation.from_matrix(predicted_mat_scipy).as_quat()
    quat_from_scipy_mat = np.array([quat_from_scipy_mat[3], quat_from_scipy_mat[0], quat_from_scipy_mat[1], quat_from_scipy_mat[2]]) # 转为 wxyz
    
    print(f"2. 生成发给物理引擎的四元数 (wxyz): {quat_from_scipy_mat}")
    
    # 物理引擎底层拿到这个四元数后，会用自己的 PyTorch3D 约定去理解它。
    # 我们来看看物理引擎“以为”你想转多少度：
    ms_quat_tensor = torch.tensor(quat_from_scipy_mat, dtype=torch.float32)
    # 物理引擎将其解析为 XYZ 欧拉角
    ms_understood_euler = matrix_to_euler_angles(quaternion_to_matrix(ms_quat_tensor), "XYZ")
    ms_understood_deg = torch.rad2deg(ms_understood_euler).numpy()

    print(f"3. ⚠️ 物理引擎实际执行的欧拉角 (度): {ms_understood_deg}")
    print(f"   (原指令应为: {euler_deg})")
    print("\n-> 结论：你会看到原本的 [30, 45, 60] 变成了完全不同的角度（例如符号反转或数值乱掉），这就解释了为什么机器人会做出类似于沿某轴对称、反向扭曲的奇怪动作！")
    print("=" * 60)

if __name__ == "__main__":
    main()