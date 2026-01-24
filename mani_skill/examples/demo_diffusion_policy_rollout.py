import gymnasium as gym
import numpy as np
import sapien
import yaml
import torch
import torch.nn as nn
from gymnasium.vector.vector_env import VectorEnv
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import time

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import FrameStack
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.plain_conv import PlainConv

import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union

@dataclass
class Args:
    config_file: Optional[str] = "/home/wshf/ManiSkill/examples/baselines/diffusion_policy/evals/PickFromDeepBoxv1-DP-01-23/config.yaml"
    """Parameter config file to load"""

    checkpoint_path: Optional[str] = "/home/wshf/ManiSkill/examples/baselines/diffusion_policy/evals/PickFromDeepBoxv1-DP-01-23/best_eval_success_at_end.pt"
    """Checkpoint path to load"""

    robot: Optional[str] = "panda_wristcam"
    """Robot UID to use in the environment"""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """If using human render mode, auto pauses the simulation upon loading"""

    quiet: bool = True
    """Disable verbose output."""

class Agent(nn.Module):
    def __init__(self, env: VectorEnv, state_mean, state_std, action_mean, action_std, device, **kwargs):
        super().__init__()
        self.obs_horizon = kwargs["obs_horizon"]
        self.act_horizon = kwargs["act_horizon"]
        self.pred_horizon = kwargs["pred_horizon"]
        self.state_mean = state_mean
        self.state_std = state_std
        self.action_mean = action_mean
        self.action_std = action_std
        self.device = device
        assert (
            len(env.single_observation_space["state"].shape) == 2
        )  # (obs_horizon, obs_dim)
        assert len(env.single_action_space.shape) == 1  # (act_dim, )
        # assert (env.single_action_space.high == 1).all() and (
        #     env.single_action_space.low == -1
        # ).all()
        # denoising results will be clipped to [-1,1], so the action should be in [-1,1] as well
        self.act_dim = env.single_action_space.shape[0]
        obs_state_dim = env.single_observation_space["state"].shape[1]
        total_visual_channels = 0
        self.include_rgb = "rgb" in env.single_observation_space.keys()
        self.include_depth = "depth" in env.single_observation_space.keys()

        if self.include_rgb:
            total_visual_channels += env.single_observation_space["rgb"].shape[-1]
        if self.include_depth:
            total_visual_channels += env.single_observation_space["depth"].shape[-1]

        visual_feature_dim = 256
        self.visual_encoder = PlainConv(
            in_channels=total_visual_channels, out_dim=visual_feature_dim, pool_feature_map=True
        )
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim,  # act_horizon is not used (U-Net doesn't care)
            global_cond_dim=self.obs_horizon * (visual_feature_dim + obs_state_dim),
            diffusion_step_embed_dim=kwargs["diffusion_step_embed_dim"],
            down_dims=kwargs["unet_dims"],
            n_groups=kwargs["n_groups"],
        )
        self.num_diffusion_iters = kwargs["num_diffusion_iters"]
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",  # has big impact on performance, try not to change
            # clip_sample=True,  # clip output to [-1,1] to improve stability
            clip_sample=False,
            prediction_type="epsilon",  # predict noise (instead of denoised action)
        )

    def encode_obs(self, obs_seq, eval_mode):
        if self.include_rgb:
            rgb = obs_seq["rgb"].float() / 255.0  # (B, obs_horizon, 3*k, H, W)
            img_seq = rgb
        if self.include_depth:
            depth = obs_seq["depth"].float() / 1024.0  # (B, obs_horizon, 1*k, H, W)
            img_seq = depth
        if self.include_rgb and self.include_depth:
            img_seq = torch.cat([rgb, depth], dim=2)  # (B, obs_horizon, C, H, W), C=4*k
        batch_size = img_seq.shape[0]
        img_seq = img_seq.flatten(end_dim=1)  # (B*obs_horizon, C, H, W)
        if hasattr(self, "aug") and not eval_mode:
            img_seq = self.aug(img_seq)  # (B*obs_horizon, C, H, W)
        visual_feature = self.visual_encoder(img_seq)  # (B*obs_horizon, D)
        visual_feature = visual_feature.reshape(
            batch_size, self.obs_horizon, visual_feature.shape[1]
        )  # (B, obs_horizon, D)
        state = (obs_seq["state"].to(self.device) - self.state_mean) / self.state_std
        feature = torch.cat(
            (visual_feature, state), dim=-1
        )  # (B, obs_horizon, D+obs_state_dim)
        return feature.flatten(start_dim=1).to(self.device)  # (B, obs_horizon * (D+obs_state_dim))

    def get_action(self, obs_seq):
        # init scheduler
        # self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
        # set_timesteps will change noise_scheduler.timesteps is only used in noise_scheduler.step()
        # noise_scheduler.step() is only called during inference
        # if we use DDPM, and inference_diffusion_steps == train_diffusion_steps, then we can skip this

        # obs_seq['state']: (B, obs_horizon, obs_state_dim)
        B = obs_seq["state"].shape[0]
        with torch.no_grad():
            if self.include_rgb:
                obs_seq["rgb"] = obs_seq["rgb"].permute(0, 1, 4, 2, 3)
            if self.include_depth:
                obs_seq["depth"] = obs_seq["depth"].permute(0, 1, 4, 2, 3)

            obs_cond = self.encode_obs(
                obs_seq, eval_mode=True
            )  # (B, obs_horizon * obs_dim)

            # initialize action from Guassian noise
            noisy_action_seq = torch.randn(
                (B, self.pred_horizon, self.act_dim), device=self.device
            ) # * self.action_std + self.action_mean

            for k in self.noise_scheduler.timesteps:
                # k_pad = torch.full(
                #     (B,), k.item(), device=self.device, dtype=torch.long
                # )
                # alpha_prod = self.noise_scheduler.alphas_cumprod.to(self.device)[k_pad][:, None].unsqueeze(1)
                # alpha = self.noise_scheduler.alphas.to(self.device)[k_pad][:, None].unsqueeze(1)
                # beta = self.noise_scheduler.betas.to(self.device)[k_pad][:, None].unsqueeze(1)

                # predict noise
                noise_pred = self.noise_pred_net(
                    # sample=(noisy_action_seq - self.action_mean) / self.action_std,
                    sample=noisy_action_seq,
                    timestep=k,
                    global_cond=obs_cond,
                )

                # inverse diffusion step (remove noise)
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action_seq,
                ).prev_sample

                # cartesian_score = -1 / (1 - alpha_prod).sqrt() * noise_pred * self.action_std
                # noisy_action_seq = 1 / alpha.sqrt() * (noisy_action_seq + (1 - alpha) * cartesian_score) + beta.sqrt() * torch.randn_like(noisy_action_seq)
                # noisy_action_seq += 0.5 * beta * dk * cartesian_score + (beta * dk).sqrt() * torch.randn_like(noisy_action_seq) * self.action_std
                # noisy_action_seq += beta * cartesian_score # + beta.sqrt() * torch.randn_like(noisy_action_seq) * self.action_std
                # noisy_action_seq = 1 / alpha.sqrt() * (noisy_action_seq - self.action_mean + beta * cartesian_score) + self.action_mean # + beta.sqrt() * torch.randn_like(noisy_action_seq) * self.action_std
                # noisy_action_seq += 0.1 * cartesian_score

                # print(f"{beta * cartesian_score / noisy_action_seq=}")

        # denormalize action
        noisy_action_seq = noisy_action_seq * self.action_std + self.action_mean

        # only take act_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end]  # (B, act_horizon, act_dim)

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

    return env_kwargs, model_kwargs

def main(args: Args):
    assert args.config_file is not None and args.checkpoint_path is not None
    env_kwargs, model_kwargs = load_config(args.config_file)

    np.set_printoptions(suppress=True, precision=3)
    verbose = not args.quiet
    if isinstance(model_kwargs["seed"], int):
        model_kwargs["seed"] = [model_kwargs["seed"]]
    if model_kwargs["seed"] is not None:
        np.random.seed(model_kwargs["seed"][0])
    parallel_in_single_scene = env_kwargs["render_mode"] == "human"
    env_kwargs["num_envs"] = 1 if env_kwargs["render_mode"] == "human" else model_kwargs["num_eval_envs"]
    if env_kwargs["render_mode"] == "human" and env_kwargs["obs_mode"] in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
        parallel_in_single_scene = False
    if env_kwargs["render_mode"] == "human" and env_kwargs["num_envs"] == 1:
        parallel_in_single_scene = False

    env_kwargs["sim_backend"] = "gpu"
    env_kwargs["render_backend"] = "gpu"
    env_kwargs["enable_shadow"] = True
    env_kwargs["parallel_in_single_scene"] = parallel_in_single_scene
    env_kwargs.pop("env_id")
    env_kwargs.pop("env_horizon")
    env_kwargs.pop("robot_uids")

    env = gym.make(id=model_kwargs["env_id"], robot_uids=args.robot, **env_kwargs)
    # record_dir = args.record_dir
    # if record_dir:
    #     record_dir = record_dir.format(env_id=args.env_id)
    #     env = RecordEpisode(env, record_dir, info_on_video=False, save_trajectory=False, max_steps_per_video=gym_utils.find_max_episode_steps_value(env))
    env = FlattenRGBDObservationWrapper(env)
    env = FrameStack(env, num_stack=model_kwargs["obs_horizon"])
    env = ManiSkillVectorEnv(env, ignore_terminations=True, record_metrics=True)

    device = torch.device("cuda" if torch.cuda.is_available() and model_kwargs["cuda"] else "cpu")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    agent = Agent(env, checkpoint["state_mean"], checkpoint["state_std"], checkpoint["action_mean"], checkpoint["action_std"], device, **model_kwargs).to(device)
    agent.load_state_dict(checkpoint["ema_agent"])
    agent.eval()

    if verbose:
        print("Observation space", env.observation_space)
        print("Action space", env.action_space)
        if env.unwrapped.agent is not None:
            print("Control mode", env.unwrapped.control_mode)
        print("Reward mode", env.unwrapped.reward_mode)

    obs, _ = env.reset(seed=model_kwargs["seed"], options=dict(reconfigure=True))
    if model_kwargs["seed"] is not None and env.action_space is not None:
        env.action_space.seed(model_kwargs["seed"][0])
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
