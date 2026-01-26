task_name=$1
exp_name=$2
seed=1
demos=100
CUDA_VISIBLE_DEVICES=$3 python train_rgbd.py --env-id ${task_name}-v1 --seed ${seed} \
  --demo-path /home/wshf/ManiSkill/demos/${task_name}-v1/motionplanning/${exp_name}.h5 \
  --control-mode "pd_ee_delta_pose" --sim-backend "physx_cpu" --num-demos ${demos} --max_episode_steps 500 \
  --total_iters 30000 --obs-mode "rgb" --batch_size 128 --num_diffusion_iters 16 \
  --robot_uids "floating_robotiq_2f_85_gripper_wristcam" --unet-dims 256 512 1024 \
  --exp-name diffusion_policy_${task_name}_${exp_name} \
  --track
  