export MS2_ASSET_DIR=./ManiSkill2_real2sim/data
gpu_id=0

declare -a arr=("./checkpoints/rt_1_x_tf_trained_for_002272480_step/" \
                "./checkpoints/xid77467904_000400120/" \
                "./checkpoints/rt1poor_xid77467904_000058240/" \
                "./checkpoints/rt1new_77467904_000001120/")

env_name=MoveNearGoogleInScene-v0
scene_name=google_pick_coke_can_1_v4
rgb_overlay_path=./checkpoints/ManiSkill2_real2sim/data/real_impainting/google_move_near_real_eval_1.png

for ckpt_path in "${arr[@]}"; do echo "$ckpt_path"; done


for ckpt_path in "${arr[@]}";

do CUDA_VISIBLE_DEVICES=${gpu_id} python real2sim/main_inference.py --policy-model rt1 --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name ${env_name} --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1;

done





