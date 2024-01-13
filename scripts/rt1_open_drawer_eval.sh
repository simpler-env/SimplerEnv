export MS2_ASSET_DIR=./ManiSkill2_real2sim/data

ckpt_path=./checkpoints/xid77467904_000400120/

gpu_id=0

python real2sim/main_inference.py --policy-model rt1 --ckpt-path ${ckpt_path} \
  --robot google_robot_static --gpu-id ${gpu_id} \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name OpenDrawerCustomInScene-v0 --scene-name frl_apartment_stage_simple \
  --robot-init-x 0.65 0.85 5 --robot-init-y -0.2 0.2 5 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.03 0.03 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --additional-env-build-kwargs scene_offset=[-1.8,-2.5,0.0]

# python real2sim/main_inference.py --policy-model rt1 --ckpt-path ${ckpt_path} \
#   --robot google_robot_static --gpu-id ${gpu_id} \
#   --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
#   --env-name OpenDrawerCustomInScene-v0 --scene-name frl_apartment_stage_simple \
#   --robot-init-x 0.65 0.65 1 --robot-init-y 0.0 0.0 1 \
#   --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.03 0.03 1 \
#   --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 --enable-raytracing
