import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gymnasium.vector.vector_env import VectorEnv
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torchvision.transforms import ColorJitter

from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.plain_conv import PlainConv
from diffusion_policy.timm_obs_encoder import TimmObsEncoder, SHAPE_META
from diffusion_policy.args import DiffusionPolicyArgs

class Agent(nn.Module):
    def __init__(self, env: VectorEnv, args: DiffusionPolicyArgs, state_mean, state_std, action_mean, action_std, device):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
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

        # visual_feature_dim = 256
        # self.visual_encoder = PlainConv(
        #     in_channels=total_visual_channels, out_dim=visual_feature_dim, pool_feature_map=True
        # )
        shape_meta = SHAPE_META
        shape_meta["obs"]["rgb"]["horizon"] = args.obs_horizon if self.include_rgb else 0
        transforms = [
            {"type": "RandomCrop", "ratio": 0.95},
            ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
        ]
        self.visual_encoder = TimmObsEncoder(
            shape_meta=shape_meta,
            model_name="vit_base_patch16_clip_224.openai",
            pretrained=True,
            frozen=False,
            global_pool="",
            transforms=transforms,
            feature_aggregation="attention_pool_2d",
            position_encording="sinusoidal",
            downsample_ratio=32,
            use_group_norm=True,
            share_rgb_model=False,
            imagenet_norm=True,
        )
        visual_feature_dim = np.prod(self.visual_encoder.output_shape())
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim,  # act_horizon is not used (U-Net doesn't care)
            global_cond_dim=self.obs_horizon * (visual_feature_dim + obs_state_dim),
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )
        self.num_diffusion_iters = args.num_diffusion_iters
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
        state = (obs_seq["state"] - self.state_mean) / self.state_std
        feature = torch.cat(
            (visual_feature, state), dim=-1
        )  # (B, obs_horizon, D+obs_state_dim)
        return feature.flatten(start_dim=1)  # (B, obs_horizon * (D+obs_state_dim))

    def compute_loss(self, obs_seq, action_seq):
        B = obs_seq["state"].shape[0]

        # observation as FiLM conditioning
        obs_cond = self.encode_obs(
            obs_seq, eval_mode=False
        )  # (B, obs_horizon * obs_dim)

        # sample noise to add to actions
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=self.device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device
        ).long()

        # add noise to the clean images(actions) according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        action_seq_norm = (action_seq - self.action_mean) / self.action_std
        noisy_action_seq = self.noise_scheduler.add_noise(action_seq_norm, noise, timesteps)

        # predict the noise residual
        noise_pred = self.noise_pred_net(
            noisy_action_seq, timesteps, global_cond=obs_cond
        )

        return F.mse_loss(noise_pred, noise)

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
                (B, self.pred_horizon, self.act_dim), device=obs_seq["state"].device
            )

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.noise_pred_net(
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

        # denormalize
        noisy_action_seq = noisy_action_seq * self.action_std + self.action_mean

        # only take act_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end]  # (B, act_horizon, act_dim)


def save_ckpt(ema, ema_agent, agent, dataset, run_name, tag):
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    ema.copy_to(ema_agent.parameters())
    torch.save(
        {
            "agent": agent.state_dict(),
            "ema_agent": ema_agent.state_dict(),
            "state_mean": dataset.state_mean,
            "state_std": dataset.state_std,
            "action_mean": dataset.action_mean,
            "action_std": dataset.action_std,
        },
        f"runs/{run_name}/checkpoints/{tag}.pt",
    )