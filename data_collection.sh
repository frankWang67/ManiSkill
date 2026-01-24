env="PickFromDeepBox-v1"
robot="panda_wristcam"
traj_name="panda_pick_from_deep_box_test"
traj_num=100
sim_backend="physx_cpu"

python mani_skill/examples/motionplanning/panda/run.py -e "$env" -r "$robot" --traj-name="$traj_name" -n "$traj_num" --sim-backend "$sim_backend" --only-count-success
python -m mani_skill.trajectory.replay_trajectory -b physx_cpu --traj-path ./demos/"$env"/motionplanning/"$traj_name".h5 --use-first-env-state -c pd_ee_delta_pose -o rgb --save-traj --save-video