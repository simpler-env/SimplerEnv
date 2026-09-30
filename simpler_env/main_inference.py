import os

import numpy as np
import tensorflow as tf

from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator

try:
    from simpler_env.policies.octo.octo_model import OctoInference
except ImportError as e:
    print("Octo is not correctly imported.")
    print(e)


if __name__ == "__main__":
    args = get_args()
    os.environ["DISPLAY"] = ""
    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        # prevent a single tf process from taking up all the GPU memory
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],
        )
    print(f"**** {args.policy_model} ****")
    # policy model creation; update this if you are using a new policy model
    if args.policy_model == "rt1":
        from simpler_env.policies.rt1.rt1_model import RT1Inference
        assert args.ckpt_path is not None
        model = RT1Inference(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
        )
    elif "octo" in args.policy_model:
        from simpler_env.policies.octo.octo_server_model import OctoServerInference
        if args.ckpt_path is None or args.ckpt_path == "None":
            args.ckpt_path = args.policy_model
        if "server" in args.policy_model:
            model = OctoServerInference(
                model_type=args.ckpt_path,
                policy_setup=args.policy_setup,
                action_scale=args.action_scale,
            )
        else:
            model = OctoInference(
                model_type=args.ckpt_path,
                policy_setup=args.policy_setup,
                init_rng=args.octo_init_rng,
                action_scale=args.action_scale,
            )
    elif args.policy_model == "openvla":
        assert args.ckpt_path is not None
        from simpler_env.policies.openvla.openvla_model import OpenVLAInference
        model = OpenVLAInference(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
        )
    elif args.policy_model == "cogact":
        from simpler_env.policies.sim_cogact import CogACTInference
        assert args.ckpt_path is not None
        model = CogACTInference(
            saved_model_path=args.ckpt_path,  # e.g., CogACT/CogACT-Base
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
            action_model_type='DiT-L',
            cfg_scale=1.5                     # cfg from 1.5 to 7 also performs well
        )
    elif args.policy_model == "spatialvla":
        assert args.ckpt_path is not None
        from simpler_env.policies.spatialvla.spatialvla_model import SpatialVLAInference
        model = SpatialVLAInference(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
        )
    elif args.policy_model == "openpifast":
        assert args.ckpt_path is not None
        from simpler_env.policies.openpi.pi0_or_fast import OpenPiFastInference
        model = OpenPiFastInference(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
        )
    elif args.policy_model == "lerobotpifast":
        assert args.ckpt_path is not None
        from simpler_env.policies.lerobotpi.pi0_or_fast import LerobotPiFastInference
        model = LerobotPiFastInference(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
        )
    elif args.policy_model == "gr00t":
        assert args.ckpt_path is not None
        from simpler_env.policies.gr00t.gr00t_model import Gr00tInference
        model = Gr00tInference(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
        )
    else:
        raise NotImplementedError()

    # run real-to-sim evaluation
    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
