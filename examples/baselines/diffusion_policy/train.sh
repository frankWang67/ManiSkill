task_name=$1
data_name=$2
exp_name=$3
cuda_device=$4
seed=1
demos=100
robot_uids=${5:-"floating_robotiq_2f_85_gripper_wristcam"}

export CUDA_VISIBLE_DEVICES=${cuda_device}
echo "Environment variable set to: $CUDA_VISIBLE_DEVICES"

python train_rgbd.py --env-id ${task_name}-v1 --seed ${seed} \
  --demo-path /home/wshf/ManiSkill/demos/${task_name}-v1/motionplanning/${data_name}.h5 \
  --control-mode "pd_ee_delta_pose" --sim-backend "physx_cpu" --num-demos ${demos} --max_episode_steps 500 \
  --total_iters 30000 --obs-mode "rgb" --batch_size 120 --num_diffusion_iters 16 \
  --robot_uids ${robot_uids} --unet-dims 256 512 1024 \
  --exp-name diffusion_policy_${task_name}_${exp_name} \
  --track
  