import gymnasium as gym
import numpy as np
import sapien
import yaml
import torch
import time

from mani_skill.utils.wrappers import FrameStack
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from flow_matching_policy.model import Agent
from flow_matching_policy.args import FlowMatchingPolicyArgs

import tyro
from dataclasses import dataclass
from typing import Optional, Annotated

@dataclass
class Args:
    config_file: Optional[str] = "/home/wshf/ManiSkill/examples/baselines/flow_matching_policy/evals/PushCube-v1-12-21-small-noise/config.yaml"
    """Parameter config file to load"""

    checkpoint_path: Optional[str] = "/home/wshf/ManiSkill/examples/baselines/flow_matching_policy/evals/PushCube-v1-12-21-small-noise/best_eval_success_at_end.pt"
    """Checkpoint path to load"""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """If using human render mode, auto pauses the simulation upon loading"""

    quiet: bool = True
    """Disable verbose output."""

def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    env_kwargs = dict()

    for k, v in config.items():
        if "wandb" in k:
            continue
        if k == "eval_env_cfg":
            env_kwargs = v["value"]

    model_args = FlowMatchingPolicyArgs()
    model_args.obs_horizon = config["obs_horizon"]["value"]
    model_args.act_horizon = config["act_horizon"]["value"]
    model_args.pred_horizon = config["pred_horizon"]["value"]
    model_args.diffusion_step_embed_dim = config["diffusion_step_embed_dim"]["value"]
    model_args.unet_dims = config["unet_dims"]["value"]
    model_args.n_groups = config["n_groups"]["value"]
    model_args.cuda = config["cuda"]["value"]
    model_args.seed = [config["seed"]["value"]]

    env_kwargs["render_mode"] = "human"

    return env_kwargs, model_args

def main(args: Args):
    assert args.config_file is not None and args.checkpoint_path is not None
    env_kwargs, model_args = load_config(args.config_file)

    np.set_printoptions(suppress=True, precision=3)
    verbose = not args.quiet
    if model_args.seed is not None:
        np.random.seed(model_args.seed)
    parallel_in_single_scene = env_kwargs["render_mode"] == "human"
    if env_kwargs["render_mode"] == "human" and env_kwargs["obs_mode"] in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
        parallel_in_single_scene = False
    if env_kwargs["render_mode"] == "human" and env_kwargs["num_envs"] == 1:
        parallel_in_single_scene = False

    env_kwargs["num_envs"] = 1
    env_kwargs["sim_backend"] = "gpu"
    env_kwargs["render_backend"] = "gpu"
    env_kwargs["enable_shadow"] = True
    env_kwargs["parallel_in_single_scene"] = parallel_in_single_scene
    env_id = env_kwargs.pop("env_id")
    env_kwargs.pop("env_horizon")

    env = gym.make(id=env_id, **env_kwargs)
    # record_dir = args.record_dir
    # if record_dir:
    #     record_dir = record_dir.format(env_id=args.env_id)
    #     env = RecordEpisode(env, record_dir, info_on_video=False, save_trajectory=False, max_steps_per_video=gym_utils.find_max_episode_steps_value(env))
    env = FlattenRGBDObservationWrapper(env)
    env = FrameStack(env, num_stack=model_args.obs_horizon)
    env = ManiSkillVectorEnv(env, ignore_terminations=True, record_metrics=True)

    device = torch.device("cuda" if torch.cuda.is_available() and model_args.cuda else "cpu")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    agent = Agent(env, model_args, checkpoint["action_std"], device).to(device)
    agent.load_state_dict(checkpoint["ema_agent"])
    agent.eval()

    if verbose:
        print("Observation space", env.observation_space)
        print("Action space", env.action_space)
        if env.unwrapped.agent is not None:
            print("Control mode", env.unwrapped.control_mode)
        print("Reward mode", env.unwrapped.reward_mode)

    obs, _ = env.reset(seed=model_args.seed, options=dict(reconfigure=True))
    if model_args.seed is not None and env.action_space is not None:
        env.action_space.seed(model_args.seed[0])
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
