import torch
from gymnasium.vector.vector_env import VectorEnv

from diffusion_policy.model import Agent
from diffusion_policy.args import DiffusionPolicyArgs
from diffusion_policy.guided_diffusion_utils import (
    get_pred_x0,
    delta_action_obstacle_loss,
    get_guidance_strength,
    visualize_grad,
)

import numpy as np
import matplotlib.pyplot as plt

class GuidedDiffusionAgent(Agent):
    eef_corner_pts = torch.tensor([
        [ 0.01,  0.043, 0.01],
        [ 0.01, -0.043, 0.01],
        [-0.01,  0.043, 0.01],
        [-0.01, -0.043, 0.01],

        [ 0.04,  0.0,  -0.15],
        [-0.04,  0.0,  -0.15],
        [ 0.0 ,  0.04, -0.15],
        [ 0.0 , -0.04, -0.15],
        
        # [0.0, 0.0, 0.0],
    ])

    def __init__(self, env: VectorEnv, args: DiffusionPolicyArgs, state_mean, state_std, action_mean, action_std, device):
        super().__init__(env, args, state_mean, state_std, action_mean, action_std, device)
        self.env = env
        self.eef_corner_pts = self.eef_corner_pts.to(device)

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
            )

            # obstacles = get_shelf_obstacles_info(self.env)
            obstacles = self.env.call("get_obstacles_info")

            # all_grads = []
            # all_losses = []
                
            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.noise_pred_net(
                    sample=noisy_action_seq,
                    timestep=k,
                    global_cond=obs_cond,
                )

                # ===============================================================
                # Guided Diffusion
                # ===============================================================
                # 1. 准备数据
                alpha_prod_k = self.noise_scheduler.alphas_cumprod[k]

                # 开启梯度计算
                with torch.enable_grad():
                    # 2. 预测 x0 (clean action prediction)
                    # 这里的 pred_action_delta 应该是反归一化后的物理值
                    x_in = noisy_action_seq.detach().clone().requires_grad_(True)
                    pred_action_delta_normalized = get_pred_x0(noise_pred, x_in, alpha_prod_k)
                    
                    # !重要!: 反归一化 (Un-Normalize)
                    # Loss 计算必须在物理空间 (米/弧度)
                    x_physical = pred_action_delta_normalized * self.action_std + self.action_mean
                    
                    # 计算 Loss
                    loss = delta_action_obstacle_loss(
                        pred_deltas=x_physical, 
                        current_state=obs_seq["state"][:, -1, :6].to(self.device), # 传入当前的真实机器人状态
                        robot_corners=self.eef_corner_pts,
                        obstacles=obstacles,
                    )
                    # all_losses.append(loss.item())
                    
                    # 计算梯度
                    grad = torch.autograd.grad(loss, x_in)[0]
                    
                # 3. Apply Guidance
                # 修改 noisy_action 或 epsilon
                # 注意: 如果是在 Delta 空间做引导，梯度往往会改变整条轨迹的形态
                gamma = get_guidance_strength(k, self.num_diffusion_iters)
                # noise_pred = noise_pred + torch.sqrt(1 - alpha_prod_k) * gamma * grad
                # ===============================================================

                # inverse diffusion step (remove noise)
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action_seq,
                ).prev_sample

                # all_grads.append(gamma * grad / noisy_action_seq)
                noisy_action_seq = noisy_action_seq - gamma * grad

            # all_grads = torch.stack(all_grads, dim=1)  # (B, num_diffusion_steps, H, 6)
            # all_grads_np = all_grads.cpu().numpy()
            # b = 0
            # visualize_grad(all_grads_np, b=b)

            # all_losses_np = np.array(all_losses)
            # plt.figure(figsize=(8, 4))
            # plt.plot(all_losses_np, label="SDF Loss")
            # plt.title("SDF Loss over Diffusion Steps")
            # plt.xlabel("Diffusion Step")
            # plt.ylabel("Loss")
            # plt.legend()
            # plt.grid()
            # plt.show()

        # denormalize
        noisy_action_seq = noisy_action_seq * self.action_std + self.action_mean

        # only take act_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end]  # (B, act_horizon, act_dim)