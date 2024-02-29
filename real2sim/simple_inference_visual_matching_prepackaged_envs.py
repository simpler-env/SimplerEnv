"""
Simple inference script for visual matching prepackaged environments
Example:
    cd {path_to_real2sim_repo_root}
    MS2_ASSET_DIR=./ManiSkill2_real2sim/data python real2sim/simple_inference_visual_matching_prepackaged_envs.py --policy rt1 \
        --ckpt-path ./checkpoints/xid77467904_000400120  --task google_robot_pick_horizontal_coke_can  --logging-dir ./results/
    MS2_ASSET_DIR=./ManiSkill2_real2sim/data python real2sim/simple_inference_visual_matching_prepackaged_envs.py --policy octo-small \
        --ckpt-path None --task widowx_spoon_on_towel  --logging-dir ./results/
"""

import argparse, os
import numpy as np
import tensorflow as tf
from real2sim.evaluation.argparse import create_prepackaged_sim_eval_args
from real2sim.evaluation.maniskill2_evaluator import maniskill2_evaluator
from copy import deepcopy

parser = argparse.ArgumentParser()

parser.add_argument('--policy', default='rt1', choices=['rt1', 'octo-base', 'octo-small'])
parser.add_argument('--ckpt-path', type=str, default='./checkpoints/rt_1_x_tf_trained_for_002272480_step/')
parser.add_argument('--task', default='google_robot_pick_horizontal_coke_can',
                    choices=['google_robot_pick_coke_can', 'google_robot_pick_horizontal_coke_can', 
                             'google_robot_pick_vertical_coke_can', 'google_robot_pick_standing_coke_can', 
                             'google_robot_move_near',
                             'google_robot_open_drawer', 'google_robot_open_top_drawer', 'google_robot_open_middle_drawer', 'google_robot_open_bottom_drawer', 
                             'google_robot_close_drawer', 'google_robot_close_top_drawer', 'google_robot_close_middle_drawer', 'google_robot_close_bottom_drawer',
                             'widowx_spoon_on_towel', 'widowx_carrot_on_plate', 'widowx_stack_cube'],)
parser.add_argument('--logging-dir', type=str, default='./results')
parser.add_argument('--tf-memory-limit', type=int, default=3072)

args = parser.parse_args()
if args.policy in ['octo-base', 'octo-small']:
    if args.ckpt_path in [None, 'None'] or 'rt_1_x' in args.ckpt_path:
        args.ckpt_path = args.policy
    
os.environ['DISPLAY'] = ''
# prevent a single jax process from taking up all the GPU memory
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    # prevent a single tf process from taking up all the GPU memory
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)])
    
# get the real-to-sim evaluation arguments
sim_eval_args = create_prepackaged_sim_eval_args(args.policy, args.ckpt_path, args.task,
                                                 logging_dir=args.logging_dir)
for i, sim_eval_arg in enumerate(sim_eval_args):
    print(i, ":", sim_eval_arg)
print("=" * 60)

if 'google_robot' in args.task:
    policy_setup = 'google_robot'
elif 'widowx' in args.task:
    policy_setup = 'widowx_bridge'
else:
    raise NotImplementedError()
    
success_arr = []
if args.policy == 'rt1':
    assert args.ckpt_path is not None
    from real2sim.rt1.rt1_model import RT1Inference
    # policy model creation
    model = RT1Inference(
        saved_model_path=args.ckpt_path, policy_setup=policy_setup
    )
    # run inference
    for sim_eval_arg in sim_eval_args:
        print(sim_eval_arg)
        success_arr.extend(maniskill2_evaluator(model, sim_eval_arg))
elif 'octo' in args.policy:
    from real2sim.octo.octo_model import OctoInference
    # for octo model, inference over 3 different seeds due to nondeterministic diffusion policy
    for rng in [0, 2, 4]:
        model = OctoInference(
            model_type=args.ckpt_path, policy_setup=policy_setup, init_rng=rng
        )
        for searg in sim_eval_args:
            sim_eval_arg = deepcopy(searg)
            sim_eval_arg.octo_init_rng = rng
            if not hasattr(sim_eval_arg, 'additional_env_save_tags'):
                sim_eval_arg.additional_env_save_tags = f'octo_init_rng_{i}'
            else:
                sim_eval_arg.additional_env_save_tags += f'_octo_init_rng_{i}'
            print(sim_eval_arg)
            success_arr.extend(maniskill2_evaluator(model, sim_eval_arg))
else:
    raise NotImplementedError()

print("**Overall Success**", np.mean(success_arr), f"({np.sum(success_arr)}/{len(success_arr)})")


