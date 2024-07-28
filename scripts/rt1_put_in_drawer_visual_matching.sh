# shader_dir=rt means that we turn on ray-tracing rendering; this is quite crucial for the open / close drawer task as policies often rely on shadows to infer depth

declare -a ckpt_paths=(
"./checkpoints/rt_1_tf_trained_for_000400120/"
"./checkpoints/rt_1_tf_trained_for_000058240/"
"./checkpoints/rt_1_x_tf_trained_for_002272480_step/"
"./checkpoints/rt_1_tf_trained_for_000001120/"
)


declare -a env_names=(
PlaceIntoClosedTopDrawerCustomInScene-v0
# PlaceIntoClosedMiddleDrawerCustomInScene-v0
# PlaceIntoClosedBottomDrawerCustomInScene-v0
)


# URDF variations
declare -a urdf_version_arr=("recolor_cabinet_visual_matching_1" "recolor_tabletop_visual_matching_1" "recolor_tabletop_visual_matching_2" None)

for urdf_version in "${urdf_version_arr[@]}"; do

EXTRA_ARGS="--enable-raytracing --additional-env-build-kwargs station_name=mk_station_recolor light_mode=simple disable_bad_material=True urdf_version=${urdf_version} model_ids=baked_apple_v2"


EvalOverlay() {
# A0
python simpler_env/main_inference.py --policy-model rt1 --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 200 \
  --env-name ${env_name} --scene-name dummy_drawer \
  --robot-init-x 0.644 0.644 1 --robot-init-y -0.179 -0.179 1 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.03 -0.03 1 \
  --obj-init-x-range -0.08 -0.02 3 --obj-init-y-range -0.02 0.08 3 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/open_drawer_a0.png \
  ${EXTRA_ARGS}

# B0
python simpler_env/main_inference.py --policy-model rt1 --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 200 \
  --env-name ${env_name} --scene-name dummy_drawer \
  --robot-init-x 0.652 0.652 1 --robot-init-y 0.009 0.009 1 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --obj-init-x-range -0.08 -0.02 3 --obj-init-y-range -0.02 0.08 3 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/open_drawer_b0.png \
  ${EXTRA_ARGS}

# C0
python simpler_env/main_inference.py --policy-model rt1 --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 200 \
  --env-name ${env_name} --scene-name dummy_drawer \
  --robot-init-x 0.665 0.665 1 --robot-init-y 0.224 0.224 1 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --obj-init-x-range -0.08 -0.02 3 --obj-init-y-range -0.02 0.08 3 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/open_drawer_c0.png \
  ${EXTRA_ARGS}
}


for ckpt_path in "${ckpt_paths[@]}"; do
  for env_name in "${env_names[@]}"; do
    EvalOverlay
  done
done



done
