export MS2_ASSET_DIR=/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data

gpu_id=0
policy_model=octo-base
scene_name=bridge_table_1_v1
rgb_overlay_path=/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/real_impainting/bridge_real_eval_1.png

CUDA_VISIBLE_DEVICES=${gpu_id} python real2sim/main_inference.py --policy-model ${policy_model} --ckpt-path ${policy_model} \
  --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 100 \
  --env-name PutSpoonOnTableClothInScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1;

CUDA_VISIBLE_DEVICES=${gpu_id} python real2sim/main_inference.py --policy-model ${policy_model} --ckpt-path ${policy_model} \
  --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 100 \
  --env-name PutCarrotOnPlateInScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1;

CUDA_VISIBLE_DEVICES=${gpu_id} python real2sim/main_inference.py --policy-model ${policy_model} --ckpt-path ${policy_model} \
  --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 100 \
  --env-name StackGreenCubeOnYellowCubeInScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1;




CUDA_VISIBLE_DEVICES=${gpu_id} python real2sim/main_inference.py --policy-model ${policy_model} --ckpt-path ${policy_model} \
  --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 100 \
  --env-name PutSpoonOnTableClothInScene-v0 --scene-name ${scene_name} \
  --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1;

CUDA_VISIBLE_DEVICES=${gpu_id} python real2sim/main_inference.py --policy-model ${policy_model} --ckpt-path ${policy_model} \
  --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 100 \
  --env-name PutCarrotOnPlateInScene-v0 --scene-name ${scene_name} \
  --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1;

CUDA_VISIBLE_DEVICES=${gpu_id} python real2sim/main_inference.py --policy-model ${policy_model} --ckpt-path ${policy_model} \
  --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 100 \
  --env-name StackGreenCubeOnYellowCubeInScene-v0 --scene-name ${scene_name} \
  --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1;



CUDA_VISIBLE_DEVICES=${gpu_id} python real2sim/main_inference.py --policy-model ${policy_model} --ckpt-path ${policy_model} \
  --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 100 \
  --env-name GraspSingleBridgeSpoonInScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --obj-init-x -0.25 -0.03 5 --obj-init-y -0.13 0.13 5



CUDA_VISIBLE_DEVICES=${gpu_id} python real2sim/main_inference.py --policy-model ${policy_model} --ckpt-path ${policy_model} \
  --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 100 \
  --env-name GraspSingleBridgeSpoonInScene-v0 --scene-name ${scene_name} \
  --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --obj-init-x -0.25 -0.03 5 --obj-init-y -0.13 0.13 5
