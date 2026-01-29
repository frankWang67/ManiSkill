import os
import sys
import subprocess
from argparse import ArgumentParser

SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "evaluate_model.py")

parser = ArgumentParser(description="Multi-Robot Data Collection via Motion Planning and Replay")
parser.add_argument("--exp-name", "-e", type=str, help="Environment ID")
parser.add_argument("--save-video", action="store_true", help="Whether to save video during trajectory generation")
args = parser.parse_args()

# ================= 配置区域 =================
# 在这里定义你想采集的所有机器人配置
# 注意：请确保 script 路径和 robot_uid 是正确的
tasks = [
    {
        "name": "panda",
        "robot_uid": "panda_robotiq_wristcam",
        "script_path": SCRIPT_PATH
    },
    {
        "name": "xarm6",
        "robot_uid": "xarm6_robotiq_wristcam",
        "script_path": SCRIPT_PATH
    },
    {
        "name": "xarm7",
        "robot_uid": "xarm7_robotiq_wristcam",
        "script_path": SCRIPT_PATH
    },
    {
        "name": "ur5",
        "robot_uid": "ur5_robotiq_wristcam",
        "script_path": SCRIPT_PATH
    },
    {
        "name": "floating_robotiq",
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
    for task in tasks:
        print(f"\n=== Evaluating Robot: {task['name']} ===")
        cmd = f"python {task['script_path']} --exp-name {args.exp_name} --robot-uids {task['robot_uid']}"
        if args.save_video:
            cmd += " --save-video"
        run_command(cmd)

if __name__ == "__main__":
    main()