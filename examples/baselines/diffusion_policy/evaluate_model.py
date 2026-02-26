import tyro
from dataclasses import dataclass
from typing import Optional, Annotated

import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

import os
import yaml
import torch
import sapien
import numpy as np

from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from diffusion_policy.args import DiffusionPolicyArgs
from diffusion_policy.model import Agent
# from diffusion_policy.guided_diffusion_model import GuidedDiffusionAgent as Agent
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.evaluate import evaluate

@dataclass
class Args:
    exp_name: Optional[str] = None
    """Experiment name for locating config and checkpoint files"""

    num_eval_episodes: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 100
    """Number of evaluation episodes to run"""

    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "physx_cpu"
    """Which simulation backend to use. Can be 'physx_cpu', 'physx_gpu'"""

    num_envs: Annotated[int, tyro.conf.arg(aliases=["-ne"])] = 10
    """Number of parallel environments to use during evaluation"""

    robot_uids: Annotated[str, tyro.conf.arg(aliases=["-r"])] = "panda_wristcam"
    """Robot UID to use in the environment"""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """If using human render mode, auto pauses the simulation upon loading"""

    quiet: bool = True
    """Disable verbose output."""

    save_video: bool = False
    """Whether to save evaluation videos"""

def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    env_kwargs = dict()
    model_kwargs = dict()

    for k, v in config.items():
        if "wandb" in k:
            continue
        if k == "eval_env_cfg":
            env_kwargs = v["value"]
        else:
            model_kwargs[k] = v["value"]

    dp_args = DiffusionPolicyArgs(**model_kwargs)

    return env_kwargs, dp_args

def main(args: Args):
    # assert args.config_file is not None and args.checkpoint_path is not None
    assert args.exp_name is not None
    config_file = os.path.join(os.path.dirname(__file__), "evals", args.exp_name, "config.yaml")
    # checkpoint_path = os.path.join(os.path.dirname(__file__), "evals", args.exp_name, "best_eval_success_at_end.pt")
    checkpoint_path = os.path.join(os.path.dirname(__file__), "evals", args.exp_name, "ckpt_iteration_30000.pt")
    # checkpoint_path = os.path.join(os.path.dirname(__file__), "evals", args.exp_name, "ckpt_iteration_25000.pt")

    env_kwargs, dp_args = load_config(config_file)

    np.set_printoptions(suppress=True, precision=3)
    verbose = not args.quiet
    if isinstance(dp_args.seed, int):
        dp_args.seed = [dp_args.seed]
    if dp_args.seed is not None:
        np.random.seed(dp_args.seed[0])
    parallel_in_single_scene = env_kwargs["render_mode"] == "human"
    if env_kwargs["render_mode"] == "human" and env_kwargs["obs_mode"] in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
        parallel_in_single_scene = False
    if env_kwargs["render_mode"] == "human" and env_kwargs["num_envs"] == 1:
        parallel_in_single_scene = False

    env_kwargs["parallel_in_single_scene"] = parallel_in_single_scene
    env_kwargs.pop("env_id")
    env_kwargs.pop("env_horizon")
    env_kwargs.pop("num_envs")
    if "robot_uids" in env_kwargs:
        env_kwargs.pop("robot_uids")

    # create evaluation environment
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = args.robot_uids
    other_kwargs = dict(obs_horizon=dp_args.obs_horizon)

    video_dir = None
    if args.save_video:
        video_dir = os.path.join(os.path.dirname(__file__), "evals", args.exp_name, "eval_results", args.robot_uids, "videos")
        # video_dir = os.path.join(os.path.dirname(__file__), "evals", args.exp_name, "eval_results", "without_guidance", args.robot_uids, "videos")
        # video_dir = os.path.join(os.path.dirname(__file__), "evals", args.exp_name, "eval_results", "guided_diffusion", args.robot_uids, "videos")

    env = make_eval_envs(
        dp_args.env_id,
        args.num_envs,
        args.sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=video_dir,
        wrappers=[FlattenRGBDObservationWrapper],
    )

    device = torch.device("cuda" if torch.cuda.is_available() and dp_args.cuda else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent = Agent(env, dp_args, checkpoint["state_mean"], checkpoint["state_std"], checkpoint["action_mean"], checkpoint["action_std"], device).to(device)
    agent.load_state_dict(checkpoint["ema_agent"])
    agent.eval()

    if verbose:
        print("Observation space", env.observation_space)
        print("Action space", env.action_space)
        if env.unwrapped.agent is not None:
            print("Control mode", env.unwrapped.control_mode)
        print("Reward mode", env.unwrapped.reward_mode)

    # obs, _ = env.reset(seed=dp_args.seed, options=dict(reconfigure=True))
    # if dp_args.seed is not None and env.action_space is not None:
    #     env.action_space.seed(dp_args.seed[0])
    if env_kwargs["render_mode"] == "human":
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = args.pause
        env.render()

    eval_metrics = evaluate(
        args.num_eval_episodes, agent, env, device, dp_args.sim_backend
    )
    print("Evaluation results over {} episodes:".format(args.num_eval_episodes))
    # for k, v in eval_metrics.items():
    #     print(f"{k}: {v}")
    success_once_rate = np.mean(eval_metrics["success_once"])
    success_at_end_rate = np.mean(eval_metrics["success_at_end"])
    print(f"{success_once_rate=}")
    print(f"{success_at_end_rate=}")

    log_filename = os.path.join(os.path.dirname(__file__), "evals", args.exp_name, "eval_results", args.robot_uids, "eval_results.txt")
    # log_filename = os.path.join(os.path.dirname(__file__), "evals", args.exp_name, "eval_results", "without_guidance", args.robot_uids, "eval_results.txt")
    # log_filename = os.path.join(os.path.dirname(__file__), "evals", args.exp_name, "eval_results", "guided_diffusion", args.robot_uids, "eval_results.txt")
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    with open(log_filename, "a") as f:
        f.write(f"Success Once Rate: {success_once_rate}\n")
        f.write(f"Success At End Rate: {success_at_end_rate}\n")

if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    args = tyro.cli(Args)
    main(args)
