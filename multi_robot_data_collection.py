import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*CUDA reports that you have.*")

import os
import sys
import subprocess
import json
from argparse import ArgumentParser

from mani_skill.trajectory.merge_trajectory import merge_trajectories

SCRIPT_PATH = "mani_skill/examples/motionplanning/panda/run.py"

ROBOT_TASKS = {
    "panda_robotiq_wristcam": "panda",
    "xarm6_robotiq_wristcam": "xarm6",
    "xarm7_robotiq_wristcam": "xarm7",
    "ur5_robotiq_wristcam": "ur5",
    "floating_robotiq_2f_85_gripper_wristcam": "floating_robotiq",
}
DEFAULT_ROBOT_UIDS = ["floating_robotiq_2f_85_gripper_wristcam"]

parser = ArgumentParser(description="Multi-Robot Data Collection via Motion Planning and Replay")
parser.add_argument("--env", "-e", type=str, help="Environment ID")
parser.add_argument("--output-filename", "-f", type=str, help="Final merged output filename")
parser.add_argument("--traj-num", "-n", type=int, default=20, help="Number of trajectories to collect per robot")
parser.add_argument(
    "--robot-uids",
    nargs="+",
    default=DEFAULT_ROBOT_UIDS,
    choices=tuple(ROBOT_TASKS),
    help="Robot UIDs used for demonstration collection (default: floating Robotiq only)",
)
parser.add_argument("--obs-mode", "-o", type=str, default="rgb", help="Observation mode for replay")
parser.add_argument("--control-mode", "-c", type=str, default="pd_ee_delta_pose", help="Control mode for replay")
parser.add_argument("--sim-backend-gen", type=str, default="physx_cpu", help="Simulation backend for trajectory generation")
parser.add_argument("--sim-backend-replay", type=str, default="physx_cpu", help="Simulation backend for trajectory replay")
parser.add_argument("--save-video", action="store_true", help="Whether to save video during trajectory generation")
parser.add_argument("--num-procs", type=int, default=1, help="Number of parallel processes for motion planning and trajectory replay")
args = parser.parse_args()

tasks = [
    {
        "name": ROBOT_TASKS[robot_uid],
        "env": args.env,
        "robot_uid": robot_uid,
        "script_path": SCRIPT_PATH,
    }
    for robot_uid in args.robot_uids
]

def run_command(cmd):
    print(f"Executing: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error executing command: {cmd}")
        sys.exit(1)


def load_episode_count(json_path):
    with open(json_path, "r") as f:
        return len(json.load(f)["episodes"])


def load_max_seed(json_path):
    with open(json_path, "r") as f:
        episodes = json.load(f)["episodes"]
    if not episodes:
        return -1
    return max(int(ep["episode_seed"]) for ep in episodes)

def main():
    replayed_files = []

    for task in tasks:
        print(f"\n=== Processing Robot: {task['name']} ===")

        suffix = f"{args.obs_mode}.{args.control_mode}.{args.sim_backend_replay}"
        robot_replayed_files = []
        remaining = args.traj_num
        next_seed_start = 0
        batch_idx = 0
        while remaining > 0:
            batch_idx += 1
            batch_num_procs = max(1, min(args.num_procs, remaining))
            # replay_trajectory strips everything after the first '.' in traj names,
            # so batch names must avoid dots to keep per-batch outputs distinct.
            batch_traj_name = f"{task['name']}_{task['env']}_batch_{batch_idx}"
            raw_traj_path = f"demos/{task['env']}/motionplanning/{batch_traj_name}.h5"
            raw_json_path = raw_traj_path.replace(".h5", ".json")
            replayed_traj_path = (
                f"demos/{task['env']}/motionplanning/{batch_traj_name}.{suffix}.h5"
            )
            replayed_json_path = replayed_traj_path.replace(".h5", ".json")

            print(
                f"Collecting batch {batch_idx} for {task['name']}: "
                f"need {remaining} more replay-success demos, seed_start={next_seed_start}"
            )

            cmd_gen = (
                f"python {task['script_path']} "
                f"-e {task['env']} "
                f"-r {task['robot_uid']} "
                f"--traj-name=\"{batch_traj_name}\" "
                f"-n {remaining} "
                f"--seed-start {next_seed_start} "
                f"--sim-backend {args.sim_backend_gen} "
                f"--num-procs {batch_num_procs} "
                f"--only-count-success "
            )
            if args.save_video:
                cmd_gen += "--save-video "
            run_command(cmd_gen)

            cmd_replay = (
                f"python -m mani_skill.trajectory.replay_trajectory "
                f"-b {args.sim_backend_replay} "
                f"--traj-path \"{raw_traj_path}\" "
                f"--use-first-env-state "
                f"-c {args.control_mode} "
                f"-o {args.obs_mode} "
                f"--num-envs {batch_num_procs} "
                f"--save-traj "
            )
            if args.save_video:
                cmd_replay += "--save-video "
            run_command(cmd_replay)

            if not os.path.exists(replayed_traj_path):
                print(f"Warning: Expected output file not found: {replayed_traj_path}")
                break

            batch_replay_count = load_episode_count(replayed_json_path)
            print(
                f"Replay kept {batch_replay_count}/{remaining} demos for {task['name']} "
                f"in batch {batch_idx}"
            )
            if batch_replay_count > 0:
                robot_replayed_files.append(replayed_traj_path)
                remaining -= batch_replay_count

            next_seed_start = load_max_seed(raw_json_path) + 1

        if len(robot_replayed_files) == 0:
            print(f"Warning: no replay-success files collected for {task['name']}")
            continue

        robot_merged_path = f"demos/{task['env']}/motionplanning/{task['name']}_{task['env']}.{suffix}.h5"
        if len(robot_replayed_files) == 1:
            replayed_files.append(robot_replayed_files[0])
        else:
            print(
                f"Merging {len(robot_replayed_files)} replay batches for {task['name']} "
                f"into {robot_merged_path}"
            )
            merge_trajectories(robot_merged_path, robot_replayed_files)
            replayed_files.append(robot_merged_path)

    # 4. 合并所有文件
    if len(replayed_files) > 0:
        merged_file_path = os.path.join(
            os.path.dirname(replayed_files[0]), args.output_filename
        )
        print(f"\n=== Merging {len(replayed_files)} files into {merged_file_path} ===")
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
