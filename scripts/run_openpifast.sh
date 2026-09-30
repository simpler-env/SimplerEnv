XDG_RUNTIME_DIR=/usr/lib
LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

model_name=openpifast

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

)


# to start server, please refe to https://github.com/Physical-Intelligence/openpi
# e.g.,
# cd /cpfs01/openpi
# uv run scripts/serve_policy.py policy:checkpoint \
#     --policy.config=pi0_fast_bridge_lora_pt_tokenizer \
#     --policy.dir=/cpfs01/openpi/checkpoints/pi0_fast_bridge_lora_pt_tokenizer/pi0_fast_bridge_lora_pt_tokenizer/29999

action_ensemble_temp=-0.8

# client inference
for ckpt_path in ${ckpts[@]}; do    
    
    logging_dir=results/$(basename $ckpt_path)${action_ensemble_temp}
    mkdir -p $logging_dir
    for i in ${!tasks[@]}; do
        task=${tasks[$i]}
        cuda=$((i % 8))
        GPU_IDX=${cuda} bash scripts/$task $ckpt_path $model_name $action_ensemble_temp $logging_dir
    done

    # statistics evalution results
    echo "🚀 all tasks DONE! Calculating metrics..."
    python tools/calc_metrics_evaluation_videos.py \
        --log-dir-root $logging_dir \
        >>$logging_dir/total.metrics
done
