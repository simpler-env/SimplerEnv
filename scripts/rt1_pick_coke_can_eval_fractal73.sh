export MS2_ASSET_DIR=/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data

declare -a arr=("/home/xuanlin/Real2Sim/rt_1_x_tf_trained_for_002272480_step/" \
                "/home/xuanlin/Real2Sim/rt1_xid45615428_000315000/" \
                "/home/xuanlin/Real2Sim/rt1poorearly_77467904_000010080/" \
                "/home/xuanlin/Real2Sim/xid77467904_000400120/" \
                "/home/xuanlin/Real2Sim/rt1poor_xid77467904_000058240/")

scene_name="fractal_traj73_table"
rgb_overlay_path="/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/real_impainting/fractal/73_table.png"
robot_rot_xy=0.015

for ckpt_path in "${arr[@]}"; do echo "$ckpt_path"; done

for ckpt_path in "${arr[@]}";

do gpu_id=0

CUDA_VISIBLE_DEVICES=${gpu_id} python real2sim/main_inference.py --policy-model rt1 --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name GraspSingleOpenedCokeCanInScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 ${robot_rot_xy} ${robot_rot_xy} 1 \
  --additional-env-build-kwargs lr_switch=True 

CUDA_VISIBLE_DEVICES=${gpu_id} python real2sim/main_inference.py --policy-model rt1 --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name GraspSingleOpenedCokeCanInScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 ${robot_rot_xy} ${robot_rot_xy} 1 \
  --additional-env-build-kwargs upright=True 

CUDA_VISIBLE_DEVICES=${gpu_id} python real2sim/main_inference.py --policy-model rt1 --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name GraspSingleOpenedCokeCanInScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 ${robot_rot_xy} ${robot_rot_xy} 1 \
  --additional-env-build-kwargs laid_vertically=True 

done