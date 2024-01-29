export MS2_ASSET_DIR=./ManiSkill2_real2sim/data

declare -a ckpt_paths=(
"./checkpoints/xid77467904_000400120/"
"./checkpoints/rt1poor_xid77467904_000058240/"
"./checkpoints/rt_1_x_tf_trained_for_002272480_step/"
)

declare -a env_names=(
OpenTopDrawerCustomInScene-v0
OpenMiddleDrawerCustomInScene-v0 
OpenBottomDrawerCustomInScene-v0
CloseTopDrawerCustomInScene-v0
CloseMiddleDrawerCustomInScene-v0 
CloseBottomDrawerCustomInScene-v0
)

declare -a scene_names=(
"modern_bedroom_no_roof"
"modern_office_no_roof"
)

EvalSim() {
  echo ${ckpt_path} ${env_name}

  python real2sim/main_inference.py --policy-model rt1 --ckpt-path ${ckpt_path} \
    --robot google_robot_static \
    --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
    --env-name ${env_name} --scene-name ${scene_name} \
    --robot-init-x 0.65 0.85 3 --robot-init-y -0.2 0.2 3 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
    --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
    ${EXTRA_ARGS}
}

for scene_name in "${scene_names[@]}"; do
  for ckpt_path in "${ckpt_paths[@]}"; do
    for env_name in "${env_names[@]}"; do
      EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt"
      EvalSim
    done
  done
done