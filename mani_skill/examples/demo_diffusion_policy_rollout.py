import os
import yaml
import time
import torch
import sapien
import numpy as np
import gymnasium as gym

import warnings
warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated.*")

from mani_skill.utils.wrappers import FrameStack
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
import diffusion_policy
from diffusion_policy.model import Agent
# from diffusion_policy.guided_diffusion_model import GuidedDiffusionAgent as Agent
from diffusion_policy.args import DiffusionPolicyArgs

import tyro
from dataclasses import dataclass
from typing import Optional, Annotated

@dataclass
class Args:
    exp_name: Optional[str] = None
    """Experiment name for locating config and checkpoint files"""

    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "physx_cpu"
    """Which simulation backend to use. Can be 'physx_cpu', 'physx_gpu'"""

    robot_uids: Annotated[str, tyro.conf.arg(aliases=["-r"])] = "panda_wristcam"
    """Robot UID to use in the environment"""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """If using human render mode, auto pauses the simulation upon loading"""

    quiet: bool = True
    """Disable verbose output."""

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

    env_kwargs["render_mode"] = "human"
    dp_args = DiffusionPolicyArgs(**model_kwargs)

    return env_kwargs, dp_args

def main(args: Args):
    assert args.exp_name is not None
    config_file = os.path.join(os.path.dirname(os.path.dirname(diffusion_policy.__file__)), "evals", args.exp_name, "config.yaml")
    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(diffusion_policy.__file__)), "evals", args.exp_name, "best_eval_success_at_end.pt")
    # checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(diffusion_policy.__file__)), "evals", args.exp_name, "ckpt_iteration_30000.pt")
    env_kwargs, dp_args = load_config(config_file)

    np.set_printoptions(suppress=True, precision=3)
    verbose = not args.quiet
    if isinstance(dp_args.seed, int):
        dp_args.seed = [dp_args.seed]
    if dp_args.seed is not None:
        np.random.seed(dp_args.seed[0])
    parallel_in_single_scene = env_kwargs["render_mode"] == "human"
    env_kwargs["num_envs"] = 1 if env_kwargs["render_mode"] == "human" else dp_args.num_eval_envs
    if env_kwargs["render_mode"] == "human" and env_kwargs["obs_mode"] in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
        parallel_in_single_scene = False
    if env_kwargs["render_mode"] == "human" and env_kwargs["num_envs"] == 1:
        parallel_in_single_scene = False

    env_kwargs["parallel_in_single_scene"] = parallel_in_single_scene
    env_kwargs.pop("env_id")
    env_kwargs.pop("env_horizon")
    env_kwargs.pop("robot_uids")

    env = gym.make(id=dp_args.env_id, robot_uids=args.robot_uids, **env_kwargs)
    # record_dir = args.record_dir
    # if record_dir:
    #     record_dir = record_dir.format(env_id=args.env_id)
    #     env = RecordEpisode(env, record_dir, info_on_video=False, save_trajectory=False, max_steps_per_video=gym_utils.find_max_episode_steps_value(env))
    env = FlattenRGBDObservationWrapper(env)
    env = FrameStack(env, num_stack=dp_args.obs_horizon)
    env = ManiSkillVectorEnv(env, ignore_terminations=True, record_metrics=True)

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

    obs, _ = env.reset(seed=dp_args.seed, options=dict(reconfigure=True))
    if dp_args.seed is not None and env.action_space is not None:
        env.action_space.seed(dp_args.seed[0])
    if env_kwargs["render_mode"] == "human":
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = args.pause
        env.render()
    start_time = time.time()
    inference_cnt = 0
    while True:
        try:
            action_seq = agent.get_action(obs)
            inference_cnt += 1
            for i in range(action_seq.shape[1]):
                obs, reward, terminated, truncated, info = env.step(action_seq[:, i])
                if verbose:
                    print("reward", reward)
                    print("terminated", terminated)
                    print("truncated", truncated)
                    print("info", info)
                if env_kwargs["render_mode"] == "human":
                    if (terminated | truncated).any():
                        break
                    env.render()
                if env_kwargs["render_mode"] is None or env_kwargs["render_mode"] != "human":
                    if (terminated | truncated).any():
                        break
            if env_kwargs["render_mode"] is None or env_kwargs["render_mode"] != "human":
                if (terminated | truncated).any():
                    break
        except KeyboardInterrupt:
            break
    env.close()
    
    inference_hz = inference_cnt / (time.time() - start_time)
    print(f"Average inference Hz: {inference_hz}")

    # if record_dir:
    #     print(f"Saving video to {record_dir}")

if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)
