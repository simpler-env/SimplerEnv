import numpy as np
import mediapy
import simpler_env
import sapien.core as sapien

from simpler_env.utils.env.observation_utils import (
    get_image_from_maniskill2_obs_dict
)

from simpler_env.policies.rt1.rt1_model import RT1Inference


task_name = "google_robot_pick_coke_can"

env = simpler_env.make(task_name)

sapien.render_config.rt_use_denoiser = False

model = RT1Inference(
    saved_model_path="./checkpoints/rt_1_x_tf_trained_for_002272480_step",
    policy_setup="google_robot"
)

obs, reset_info = env.reset()

instruction = env.get_language_instruction()

print("Instruction:", instruction)

model.reset(instruction)

image = get_image_from_maniskill2_obs_dict(env, obs)

frames = []

predicted_terminated = False
truncated = False
success = False

while not (predicted_terminated or truncated):

    raw_action, action = model.step(image)

    predicted_terminated = bool(
        action["terminate_episode"][0] > 0
    )

    obs, reward, success, truncated, info = env.step(
        np.concatenate([
            action["world_vector"],
            action["rot_axangle"],
            action["gripper"]
        ])
    )

    image = get_image_from_maniskill2_obs_dict(env, obs)

    frames.append(image)

print("Success:", success)

mediapy.write_video("rollout_1.mp4", frames, fps=10)

print("Video saved: rollout_1.mp4")
