import simpler_env
import numpy as np
import mediapy

from simpler_env.utils.env.observation_utils import (
    get_image_from_maniskill2_obs_dict
)

env = simpler_env.make(
    "google_robot_pick_coke_can"
)

obs, info = env.reset(
    options={
        "distractor_model_ids": [
            "chips_bag",
            "tetra_pak_carton",
            "tin_can"
        ]
    }
)

print("Distractors:", env.unwrapped.distractor_objs)
print("Count:", len(env.unwrapped.distractor_objs))

frames = []

for i in range(10):

    action = np.zeros(7)

    obs, reward, success, truncated, info = env.step(action)

    image = get_image_from_maniskill2_obs_dict(env, obs)

    frames.append(image)

mediapy.write_video("debug_scene.mp4", frames, fps=5)

print("saved")
