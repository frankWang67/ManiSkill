import os
import sys
import subprocess
from argparse import ArgumentParser

from mani_skill.trajectory.merge_trajectory import merge_trajectories

SCRIPT_PATH = "mani_skill/examples/motionplanning/panda/run.py"

parser = ArgumentParser(description="Multi-Robot Data Collection via Motion Planning and Replay")
parser.add_argument("--env", "-e", type=str, help="Environment ID")
parser.add_argument("--traj-num", "-n", type=int, default=20, help="Number of trajectories to collect per robot")
parser.add_argument("--obs-mode", "-o", type=str, default="rgb", help="Observation mode for replay")
parser.add_argument("--control-mode", "-c", type=str, default="pd_ee_delta_pose", help="Control mode for replay")
parser.add_argument("--output-filename", "-f", type=str, default="merged_multi_robot_data.h5", help="Final merged output filename")
parser.add_argument("--sim-backend-gen", type=str, default="physx_cpu", help="Simulation backend for trajectory generation")
parser.add_argument("--sim-backend-replay", type=str, default="physx_cpu", help="Simulation backend for trajectory replay")
parser.add_argument("--save-video", action="store_true", help="Whether to save video during trajectory generation")
args = parser.parse_args()

# ================= 配置区域 =================
# 在这里定义你想采集的所有机器人配置
# 注意：请确保 script 路径和 robot_uid 是正确的
tasks = [
    {
        "name": "panda",
        "env": args.env,
        "robot_uid": "panda_robotiq_wristcam",
        "script_path": SCRIPT_PATH
    },
    {
        "name": "xarm6",
        "env": args.env,
        "robot_uid": "xarm6_robotiq_wristcam",
        "script_path": SCRIPT_PATH
    },
    {
        "name": "xarm7",
        "env": args.env,
        "robot_uid": "xarm7_robotiq_wristcam",
        "script_path": SCRIPT_PATH
    },
    {
        "name": "ur5",
        "env": args.env,
        "robot_uid": "ur5_robotiq_wristcam",
        "script_path": SCRIPT_PATH
    },
    {
        "name": "floating_robotiq",
        "env": args.env,
        "robot_uid": "floating_robotiq_2f_85_gripper_wristcam",
        "script_path": SCRIPT_PATH
    },
]

# ===========================================

def run_command(cmd):
    print(f"Executing: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error executing command: {cmd}")
        sys.exit(1)

def main():
    replayed_files = []

    for task in tasks:
        print(f"\n=== Processing Robot: {task['name']} ===")
        
        # 1. 定义文件名
        traj_name = f"{task['name']}_{task['env']}"
        raw_traj_path = f"demos/{task['env']}/motionplanning/{traj_name}.h5"
        
        # 2. 运行 Motion Planning 生成轨迹 (run.py)
        # 对应: python run.py ... --only-count-success
        cmd_gen = (
            f"python {task['script_path']} "
            f"-e {task['env']} "
            f"-r {task['robot_uid']} "
            f"--traj-name=\"{traj_name}\" "
            f"-n {args.traj_num} "
            f"--sim-backend {args.sim_backend_gen} "
            f"--only-count-success "
        )
        if args.save_video:
            cmd_gen += "--save-video "
            
        run_command(cmd_gen)
        
        # 3. 运行 Replay 生成观测数据 (replay_trajectory.py)
        # 对应: python -m mani_skill.trajectory.replay_trajectory ...
        # 注意：回放后生成的文件名会自动加上后缀，我们需要构建这个文件名以便后续合并
        # 命名规则通常是: {traj_name}.{obs_mode}.{control_mode}.{backend}.h5
        suffix = f"{args.obs_mode}.{args.control_mode}.{args.sim_backend_replay}"
        replayed_traj_path = f"demos/{task['env']}/motionplanning/{traj_name}.{suffix}.h5"
        
        cmd_replay = (
            f"python -m mani_skill.trajectory.replay_trajectory "
            f"-b {args.sim_backend_replay} "
            f"--traj-path \"{raw_traj_path}\" "
            f"--use-first-env-state "
            f"-c {args.control_mode} "
            f"-o {args.obs_mode} "
            f"--save-traj" # 确保保存回放后的轨迹
        )
        
        run_command(cmd_replay)
        
        if os.path.exists(replayed_traj_path):
            replayed_files.append(replayed_traj_path)
        else:
            print(f"Warning: Expected output file not found: {replayed_traj_path}")

    # 4. 合并所有文件
    merged_file_path = os.path.join(os.path.dirname(replayed_files[0]), args.output_filename)
    print(f"\n=== Merging {len(replayed_files)} files into {merged_file_path} ===")
    if len(replayed_files) > 0:
        try:
            merge_trajectories(merged_file_path, replayed_files)
            print(f"Successfully created {merged_file_path}")
            
            # 可选：清理中间文件
            # for f in replayed_files:
            #     os.remove(f)
        except ValueError as e:
            print(f"Merge failed: {e}")
            print("Tip: If merging failed due to shape mismatch (e.g., different joint counts), "
                  "you cannot save them into a single standard H5 dataset. "
                  "You may need to keep them separate or unify the state space.")
    else:
        print("No files to merge.")

if __name__ == "__main__":
    main()