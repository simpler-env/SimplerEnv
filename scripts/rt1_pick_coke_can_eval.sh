export MS2_ASSET_DIR=/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data

# ckpt_path=/home/xuanlin/Real2Sim/rt_1_x_tf_trained_for_002272480_step/
# ckpt_path=/home/xuanlin/Real2Sim/rt1_xid45615428_000315000/
# ckpt_path=/home/xuanlin/Real2Sim/rt1poorearly_77467904_000010080/
ckpt_path=/home/xuanlin/Real2Sim/xid77467904_000400120/

gpu_id=0

python real2sim/main_inference.py --policy-model rt1 --ckpt-path ${ckpt_path} \
  --robot google_robot_static --gpu-id ${gpu_id} \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name GraspSingleOpenedCokeCanInScene-v0 --scene-name google_pick_coke_can_1_v4 \
  --rgb-overlay-path /home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/google_coke_can_real_eval_1.png \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --additional-env-build-kwargs lr_switch=True 

python real2sim/main_inference.py --policy-model rt1 --ckpt-path ${ckpt_path} \
  --robot google_robot_static --gpu-id ${gpu_id} \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name GraspSingleOpenedCokeCanInScene-v0 --scene-name google_pick_coke_can_1_v4 \
  --rgb-overlay-path /home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/google_coke_can_real_eval_1.png \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --additional-env-build-kwargs upright=True 

python real2sim/main_inference.py --policy-model rt1 --ckpt-path ${ckpt_path} \
  --robot google_robot_static --gpu-id ${gpu_id} \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name GraspSingleOpenedCokeCanInScene-v0 --scene-name google_pick_coke_can_1_v4 \
  --rgb-overlay-path /home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/google_coke_can_real_eval_2.png \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.03 0.03 1 \
  --additional-env-build-kwargs laid_vertically=True 



# debug

# python real2sim/main_inference.py --policy-model rt1 --ckpt-path ${ckpt_path} \
#   --robot google_robot_static --gpu-id ${gpu_id} \
#   --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
#   --env-name GraspSingleOpenedCokeCanInScene-v0 --scene-name google_pick_coke_can_1_v4 \
#   --rgb-overlay-path /home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/google_coke_can_real_eval_2.png \
#   --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.35 1 --obj-init-y -0.02 -0.02 1 \
#   --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.03 0.03 1 \
#   --additional-env-build-kwargs laid_vertically=True 


# python real2sim/main_inference.py --ckpt-path /home/xuanlin/Real2Sim/robotics_transformer/trained_checkpoints/rt1main/ \
