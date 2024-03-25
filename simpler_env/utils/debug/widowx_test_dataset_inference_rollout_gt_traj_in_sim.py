"""
Step ground-truth actions in a dataset in an open-loop manner and record the resulting observation video and robot qpos.
"""

import os

import cv2
import gymnasium as gym
import mani_skill2_real2sim.envs
import numpy as np
from sapien.core import Pose
import tensorflow_datasets as tfds
from transforms3d.euler import euler2axangle, euler2mat
from transforms3d.quaternions import mat2quat

from simpler_env.utils.visualization import write_video

DATASETS = ["fractal20220817_data", "bridge"]


def dataset2path(dataset_name):
    if dataset_name == "robo_net":
        version = "1.0.0"
    elif dataset_name == "language_table":
        version = "0.0.1"
    else:
        version = "0.1.0"
    return f"gs://gresearch/robotics/{dataset_name}/{version}"


def main(
    dset_iter,
    iter_num,
    episode_id,
    control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
    save_root="debug_logs/widowx_test_dataset_inference_rollout_gt_traj_in_sim/",
    overlay_camera="3rd_view_camera",
):

    for _ in range(iter_num):
        episode = next(dset_iter)
    print("episode tfds id", episode["tfds_id"])
    episode_steps = list(episode["steps"])

    language_instruction = episode_steps[0]["observation"]["natural_language_instruction"]
    print(language_instruction)

    sim_freq, control_freq = 510, 5
    action_scale = 1.0
    env = gym.make(
        "GraspSingleDummy-v0",
        control_mode=control_mode,
        obs_mode="rgbd",
        robot="widowx_bridge_dataset_camera_setup",
        sim_freq=sim_freq,
        control_freq=control_freq,
        max_episode_steps=50,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=f"ManiSkill2_real2sim/data/real_inpainting/bridge/bridge_{episode_id}_cleanup.png",
        rgb_overlay_cameras=[overlay_camera],
    )

    images = []
    gt_images = []
    qpos_arr = []
    obs, _ = env.reset()
    env.agent.robot.set_pose(Pose([0, 0, 1]))

    mat_transform = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64)
    # initialize robot end-effector pose to be the same as the first observation
    gt_tcp_pose_at_robot_base = Pose(
        p=episode_steps[0]["observation"]["state"][:3],
        q=mat2quat(
            euler2mat(*np.array(episode_steps[0]["observation"]["state"][3:6], dtype=np.float64)) @ mat_transform
        ),
    )
    controller = env.agent.controller.controllers["arm"]
    cur_qpos = env.agent.robot.get_qpos()
    init_arm_qpos = controller.compute_ik(gt_tcp_pose_at_robot_base)
    cur_qpos[controller.joint_indices] = init_arm_qpos
    env.agent.reset(cur_qpos)
    qpos_arr.append(env.agent.robot.get_qpos())

    obs = env.get_obs()
    images.append((obs["image"][overlay_camera]["Color"][..., :-1] * 255).astype(np.uint8))

    # step the environment using trajectory actions
    for i in range(len(episode_steps) - 1):
        episode_step = episode_steps[i]  # episode_step['observation']['base_pose_tool_reached'] = [xyz, quat xyzw]
        gt_images.append(episode_step["observation"]["image"])

        gt_action_world_vector = np.asarray(episode_step["action"]["world_vector"], dtype=np.float64)
        gt_action_rotation_delta = np.asarray(episode_step["action"]["rotation_delta"], dtype=np.float64)
        gt_action_rotation_ax, gt_action_rotation_angle = euler2axangle(*gt_action_rotation_delta)
        gt_action_rotation_axangle = gt_action_rotation_ax * gt_action_rotation_angle
        gt_action_gripper_closedness_action = 2.0 * (np.array(episode_step["action"]["open_gripper"])[None]) - 1.0

        action = np.concatenate(
            [
                gt_action_world_vector * action_scale,
                gt_action_rotation_axangle * action_scale,
                gt_action_gripper_closedness_action,
            ],
        ).astype(np.float64)

        obs, *_ = env.step(action)
        images.append(obs["image"][overlay_camera]["rgb"])
        qpos_arr.append(env.agent.robot.get_qpos())

    gt_images = [gt_images[np.clip(i, 0, len(gt_images) - 1)] for i in range(len(images))]
    for i in range(len(images)):
        images[i] = np.concatenate(
            [
                images[i],
                cv2.resize(np.asarray(gt_images[i]), (images[i].shape[1], images[i].shape[0])),
            ],
            axis=1,
        )

    os.makedirs(save_root, exist_ok=True)
    os.makedirs(f"{save_root}/save_qpos", exist_ok=True)
    write_video(f"{save_root}/{episode_id}.mp4", images, fps=5)
    np.save(f"{save_root}/save_qpos/{episode_id}_qpos.npy", np.array(qpos_arr))


if __name__ == "__main__":
    os.environ["DISPLAY"] = ""
    dataset_name = DATASETS[1]
    dset = tfds.builder_from_directory(builder_dir=dataset2path(dataset_name))
    save_root = "debug_logs/widowx_test_dataset_inference_rollout_gt_traj_in_sim/"

    dset = dset.as_dataset(split="train[:6]", read_config=tfds.ReadConfig(add_tfds_id=True))
    dset_iter = iter(dset)
    last_episode_id = 0
    for ep_idx in [1]:  # [0, 1, 2, 3, 4]: # [0, 1]:
        if last_episode_id == 0:
            main(dset_iter, ep_idx + 1 - last_episode_id, ep_idx, save_root=save_root)
        else:
            main(dset_iter, ep_idx - last_episode_id, ep_idx, save_root=save_root)
        last_episode_id = ep_idx
