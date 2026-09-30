model_name=octo-base
tasks=(
  bridge.sh
  drawer_variant_agg.sh
  drawer_visual_matching.sh
  move_near_variant_agg.sh
  move_near_visual_matching.sh
  pick_coke_can_variant_agg.sh
  pick_coke_can_visual_matching.sh
  put_in_drawer_variant_agg.sh
  put_in_drawer_visual_matching.sh
)

ckpts=(
  None
)

action_ensemble_temp=-0.8
for ckpt_path in ${ckpts[@]}; do
  base_dir=$(dirname $ckpt_path)

  # evaluation in simulator
  # logging_dir=$base_dir/simpler_env/$(basename $ckpt_path)${action_ensemble_temp}
  logging_dir=results/$(basename $ckpt_path)${action_ensemble_temp}
  mkdir -p $logging_dir
  for i in ${!tasks[@]}; do
    task=${tasks[$i]}
    echo "🚀 running $task ..."
    device=0
    session_name=CUDA${device}-$(basename $logging_dir)-${task}
    bash scripts/$task $ckpt_path $model_name $action_ensemble_temp $logging_dir $device
  done

  # statistics evalution results
  echo "🚀 all tasks DONE! Calculating metrics..."
  python tools/calc_metrics_evaluation_videos.py \
    --log-dir-root $logging_dir \
    >>$logging_dir/total.metrics
done
