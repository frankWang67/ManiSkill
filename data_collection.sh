env="PickBehindBarrier-v1"
robot="ur5_robotiq_wristcam"
traj_name="ur5_0127"
traj_num=100
sim_backend="physx_cpu"
num_procs=10

python mani_skill/examples/motionplanning/panda/run.py -e "$env" -r "$robot" --traj-name="$traj_name" -n "$traj_num" --sim-backend "$sim_backend" --only-count-success --num-procs ${num_procs}
python -m mani_skill.trajectory.replay_trajectory -b physx_cpu --traj-path ./demos/"$env"/motionplanning/"$traj_name".h5 --use-first-env-state -c pd_ee_delta_pose -o rgb --save-traj --save-video --num-envs ${num_procs}