env="PushCube-v1"
robot="floating_panda_gripper_wristcam"
traj_name="pandahand_pushcube"
traj_num=10
sim_backend="gpu"

python mani_skill/examples/motionplanning/panda/run.py -e "$env" -r "$robot" --traj-name="$traj_name" -n "$traj_num" --sim-backend "$sim_backend" --save-video --only-count-success
python -m mani_skill.trajectory.replay_trajectory -b physx_cpu --traj-path ./demos/"$env"/motionplanning/"$traj_name".h5 --use-first-env-state -c pd_ee_delta_pose -o rgb --save-traj