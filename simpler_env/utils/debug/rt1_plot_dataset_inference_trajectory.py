"""
Run RT-1 model on a dataset and plot the predicted and ground truth action trajectory.
"""

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from simpler_env.policies.rt1.rt1_model import RT1Inference
from simpler_env.utils.visualization import plot_pred_and_gt_action_trajectory

DATASETS = ["fractal20220817_data"]


def dataset2path(dataset_name):
    if dataset_name == "robo_net":
        version = "1.0.0"
    elif dataset_name == "language_table":
        version = "0.0.1"
    else:
        version = "0.1.0"
    return f"gs://gresearch/robotics/{dataset_name}/{version}"


def main(episode, model, model_type="rt1"):
    episode_steps = list(episode["steps"])
    pred_actions, gt_actions, images = [], [], []

    language_instruction = episode_steps[0]["observation"]["natural_language_instruction"]
    print(language_instruction)
    model.reset(language_instruction)

    for i in range(len(episode_steps) - 1):
        episode_step = episode_steps[i]

        raw_action, _ = model.step(episode_step["observation"]["image"])
        if model_type == "rt1":
            action_world_vector = raw_action["world_vector"]
            action_rotation_delta = raw_action["rotation_delta"]
            action_gripper_closedness_action = raw_action["gripper_closedness_action"]
            gt_action_world_vector = episode_step["action"]["world_vector"]
            gt_action_rotation_delta = episode_step["action"]["rotation_delta"]
            gt_action_gripper_closedness_action = episode_step["action"]["gripper_closedness_action"]
            print("**STEP**", i)
            print(
                "world pred",
                action_world_vector,
                "gt",
                gt_action_world_vector,
                "mse",
                np.mean((action_world_vector - gt_action_world_vector) ** 2),
            )
            print(
                "rotation pred",
                action_rotation_delta,
                "gt",
                gt_action_rotation_delta,
                "mse",
                np.mean((action_rotation_delta - gt_action_rotation_delta) ** 2),
            )
            print(
                "gripper pred",
                action_gripper_closedness_action,
                "gt",
                gt_action_gripper_closedness_action,
                "mse",
                np.mean((action_gripper_closedness_action - gt_action_gripper_closedness_action) ** 2),
            )
            print(
                "terminate pred",
                raw_action["terminate_episode"],
                "gt",
                episode_step["action"]["terminate_episode"],
            )
        else:
            raise NotImplementedError()

        pred_actions.append(raw_action)
        gt_actions.append(episode_step["action"])
        images.append(episode_step["observation"]["image"])

    plot_pred_and_gt_action_trajectory(
        pred_actions,
        gt_actions,
        tf.concat(tf.unstack(images[:: int(len(images) // 10)], axis=0), 1).numpy(),
    )


if __name__ == "__main__":
    ckpt_path = "checkpoints/rt_1_x_tf_trained_for_002272480_step/"
    # ckpt_path = "checkpoints/rt_1_tf_trained_for_000400120/"
    # ckpt_path = "checkpoints/rt_1_tf_trained_for_000058240/"
    # ckpt_path = 'checkpoints/rt_1_x_tf_trained_for_002272480_step'

    gpus = tf.config.list_physical_devices("GPU")
    tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])

    dataset_name = DATASETS[0]
    dset = tfds.builder_from_directory(builder_dir=dataset2path(dataset_name))
    dset = dset.as_dataset(split="train")
    dset_iter = iter(dset)

    tot_mse = 0.0
    model = RT1Inference(saved_model_path=ckpt_path)

    episode = next(dset_iter)
    main(episode, model, model_type="rt1")
