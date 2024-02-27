import numpy as np
import os
import tensorflow as tf

from real2sim.evaluation.argparse import get_args
from real2sim.evaluation.maniskill2_evaluator import maniskill2_evaluator
from real2sim.rt1.rt1_model import RT1Inference
from real2sim.octo.octo_server_model import OctoServerInference
try:
    from real2sim.octo.octo_model import OctoInference
except ImportError as e:
    print("Octo is not correctly imported.")
    print(e)


if __name__ == '__main__':
    args = get_args()
    
    os.environ['DISPLAY'] = ''
    # prevent a single jax process from taking up all the GPU memory
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        # prevent a single tf process from taking up all the GPU memory
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)])
      
    # policy model creation; update this if you are using a new policy model
    if args.policy_model == 'rt1':
        assert args.ckpt_path is not None
        model = RT1Inference(
            saved_model_path=args.ckpt_path, policy_setup=args.policy_setup, action_scale=args.action_scale
        )
    elif 'octo' in args.policy_model:
        args.ckpt_path = args.policy_model
        if 'server' in args.policy_model:
            model = OctoServerInference(
                model_type=args.policy_model, policy_setup=args.policy_setup, action_scale=args.action_scale)
        else:
            model = OctoInference(
                model_type=args.policy_model, policy_setup=args.policy_setup, init_rng=args.octo_init_rng, action_scale=args.action_scale,
            )
    else:
        raise NotImplementedError()
      
    # run real-to-sim evaluation
    maniskill2_evaluator(model, args)