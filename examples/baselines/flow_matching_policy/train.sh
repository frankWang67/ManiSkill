seed=1
demos=100
python flow_matching_policy/train_rgbd.py --env-id PickFromDeepBox-v1 --seed ${seed} \
  --demo-path /home/wshf/ManiSkill/demos/PickFromDeepBox-v1/motionplanning/PandaRobotiq-fov120-224x224.h5 \
  --control-mode "pd_ee_delta_pose" --sim-backend "physx_cpu" --num-demos ${demos} --max_episode_steps 300 \
  --total_iters 30000 --obs-mode "rgb" --batch_size 128 \
  --robot_uids "panda_robotiq_wristcam" --unet-dims 256 512 1024 \
  --exp-name flow_matching_policy-wider-PandaRobotiq-fov120-timm-encoder-PickFromDeepBox-0123 \
  --track