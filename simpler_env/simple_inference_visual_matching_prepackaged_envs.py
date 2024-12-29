"""
Simple script for real-to-sim eval using the prepackaged visual matching setup in ManiSkill2.
Example:
    cd {path_to_simpler_env_repo_root}
    python simpler_env/simple_inference_visual_matching_prepackaged_envs.py --policy rt1 \
        --ckpt-path ./checkpoints/rt_1_tf_trained_for_000400120  --task google_robot_pick_coke_can  --logging-root ./results_simple_eval/  --n-trajs 10
    python simpler_env/simple_inference_visual_matching_prepackaged_envs.py --policy octo-small \
        --ckpt-path None --task widowx_spoon_on_towel  --logging-root ./results_simple_eval/  --n-trajs 10
"""

import argparse
import os

import mediapy as media
import numpy as np
import tensorflow as tf

import simpler_env
from simpler_env import ENVIRONMENTS
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

parser = argparse.ArgumentParser()

parser.add_argument("--policy", default="rt1", choices=["rt1", "octo-base", "octo-small"])
parser.add_argument(
    "--ckpt-path",
    type=str,
    default="./checkpoints/rt_1_x_tf_trained_for_002272480_step/",
)
parser.add_argument(
    "--task",
    default="google_robot_pick_horizontal_coke_can",
    choices=ENVIRONMENTS,
)
parser.add_argument("--logging-root", type=str, default="./results_simple_random_eval")
parser.add_argument("--tf-memory-limit", type=int, default=3072)
parser.add_argument("--n-trajs", type=int, default=10)

args = parser.parse_args()
if args.policy in ["octo-base", "octo-small"]:
    if args.ckpt_path in [None, "None"] or "rt_1_x" in args.ckpt_path:
        args.ckpt_path = args.policy
if args.ckpt_path[-1] == "/":
    args.ckpt_path = args.ckpt_path[:-1]
logging_dir = os.path.join(args.logging_root, args.task, args.policy, os.path.basename(args.ckpt_path))
os.makedirs(logging_dir, exist_ok=True)

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

# build environment
env = simpler_env.make(args.task)

# build policy
if "google_robot" in args.task:
    policy_setup = "google_robot"
elif "widowx" in args.task:
    policy_setup = "widowx_bridge"
else:
    raise NotImplementedError()

if args.policy == "rt1":
    from simpler_env.policies.rt1.rt1_model import RT1Inference

    model = RT1Inference(saved_model_path=args.ckpt_path, policy_setup=policy_setup)
elif "octo" in args.policy:
    from simpler_env.policies.octo.octo_model import OctoInference

    model = OctoInference(model_type=args.ckpt_path, policy_setup=policy_setup, init_rng=0)
else:
    raise NotImplementedError()

# run inference
success_arr = []
for ep_id in range(args.n_trajs):
    obs, reset_info = env.reset()
    instruction = env.get_language_instruction()
    # for long-horizon environments, we check if the current subtask is the final subtask
    is_final_subtask = env.is_final_subtask() 

    model.reset(instruction)
    print(instruction)

    image = get_image_from_maniskill2_obs_dict(env, obs)  # np.ndarray of shape (H, W, 3), uint8
    images = [image]
    predicted_terminated, success, truncated = False, False, False
    timestep = 0
    while not (predicted_terminated or truncated):
        # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
        raw_action, action = model.step(image, instruction)
        predicted_terminated = bool(action["terminate_episode"][0] > 0)
        if predicted_terminated:
            if not is_final_subtask:
                # advance the environment to the next subtask
                predicted_terminated = False
                env.advance_to_next_subtask()

        obs, reward, success, truncated, info = env.step(
            np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]),
        )
        print(timestep, info)
        new_instruction = env.get_language_instruction()
        if new_instruction != instruction:
            # update instruction for long horizon tasks
            instruction = new_instruction
            print(instruction)
        is_final_subtask = env.is_final_subtask() 
        # update image observation
        image = get_image_from_maniskill2_obs_dict(env, obs)
        images.append(image)
        timestep += 1

    episode_stats = info.get("episode_stats", {})
    success_arr.append(success)
    print(f"Episode {ep_id} success: {success}")
    media.write_video(f"{logging_dir}/episode_{ep_id}_success_{success}.mp4", images, fps=5)

print(
    "**Overall Success**",
    np.mean(success_arr),
    f"({np.sum(success_arr)}/{len(success_arr)})",
)
