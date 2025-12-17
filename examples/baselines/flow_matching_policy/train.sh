seed=1
demos=49
python flow_matching_policy/train_rgbd.py --env-id PushCube-v1 \
  --demo-path ~/ManiSkill/demos/PushCube-v1-panda/motionplanning/trajectory_gpu_1106.rgb.pd_ee_delta_pose.physx_cpu.h5 \
  --control-mode "pd_ee_delta_pose" --sim-backend "physx_cpu" --num-demos ${demos} --max_episode_steps 100 \
  --total_iters 30000 --obs-mode "rgb" \
  --exp-name flow_matching_policy-PushCube-v1-rgb-${demos}_motionplanning_demos-${seed} \
  --track