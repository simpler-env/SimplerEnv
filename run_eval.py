import os
import numpy as np
import mediapy
import simpler_env
import sapien.core as sapien

from simpler_env.utils.env.observation_utils import (
    get_image_from_maniskill2_obs_dict
)

from simpler_env.policies.rt1.rt1_model import RT1Inference



TASK_NAME = "google_robot_pick_coke_can"

# seeds для baseline и distractors
SEEDS = list(range(100, 120))

# 0 = baseline
# 1 = один distractor
# 3 = три distractors
DISTRACTOR_LEVEL = 3

OUTPUT_DIR = f"results/distractors_level_{DISTRACTOR_LEVEL}"

os.makedirs(OUTPUT_DIR, exist_ok=True)


sapien.render_config.rt_use_denoiser = False

env = simpler_env.make(TASK_NAME)
env.unwrapped.distractor_level = DISTRACTOR_LEVEL

model = RT1Inference(
    saved_model_path="./checkpoints/rt_1_x_tf_trained_for_002272480_step",
    policy_setup="google_robot"
)
success_count = 0

# MAIN

for episode_idx, seed in enumerate(SEEDS):

    print("\n" + "=" * 60)
    print(f"EPISODE {episode_idx}")
    print(f"SEED: {seed}")
    print("=" * 60)

    np.random.seed(seed)

    reset_options = {}

    if DISTRACTOR_LEVEL == 1:
        reset_options["distractor_model_ids"] = [
            "tetra_pak_carton"
        ]

    elif DISTRACTOR_LEVEL == 3:
        reset_options["distractor_model_ids"] = [
            "chips_bag",
            "tetra_pak_carton",
            "tin_can"
        ]

    obs, reset_info = env.reset(
        seed=seed,
        options=reset_options
    )

    instruction = env.get_language_instruction()
    print("Instruction:", instruction)

    model.reset(instruction)

    image = get_image_from_maniskill2_obs_dict(env, obs)

    frames = []

    predicted_terminated = False
    truncated = False
    success = False

    step_count = 0

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

        step_count += 1

    # сохраняем только первые 5 rollout
    video_path = os.path.join(
        OUTPUT_DIR,
        f"rollout_seed_{seed}.mp4"
    )

    mediapy.write_video(video_path, frames, fps=10)

    print("Video saved:", video_path)

    print("Success:", success)
    if success:
       success_count += 1
    print("Steps:", step_count)


sr = success_count / len(SEEDS)

print("\n\n")
print("FINAL RESULTS")
print("=" * 60)
print(f"Success Rate: {sr:.2f}")
