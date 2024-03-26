"""
Uses:
1. Given a static (inpainting) image (with robot arm removed), query the Octo model to predict actions and visualize them in the environment,
    where the policy input is the inpainting image plus the robot arm rendered in it.
2. Given a video, feed the video frames as input to the Octo model, and visualize the resulting actions in the environment.
"""

import os

import cv2
import mediapy as media
import numpy as np
import tensorflow as tf

from simpler_env.policies.octo.octo_model import OctoInference
from simpler_env.policies.octo.octo_server_model import OctoServerInference
from simpler_env.utils.env.env_builder import build_maniskill2_env
from simpler_env.utils.visualization import write_video


def main(
    input_video,
    inpainting_img_path,
    instruction,
    gt_tcp_pose_at_robot_base=None,
    camera="3rd_view_camera",
    model_type="octo-base",
    policy_setup="widowx_bridge",
    robot="widowx",
    control_freq=5,
    max_episode_steps=90,
    control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
    save_root="./debug_logs/",
    save_name="debug_octo_inference",
    **kwargs,
):

    # Create environment
    env = build_maniskill2_env(
        "GraspSingleDummy-v0",
        control_mode=control_mode,
        obs_mode="rgbd",
        robot=robot,
        sim_freq=500,
        max_episode_steps=max_episode_steps,
        control_freq=control_freq,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=inpainting_img_path,
        rgb_overlay_cameras=[camera],
        **kwargs,
    )
    print(instruction)

    # Reset and initialize environment
    predicted_actions = []
    images = []
    qpos_arr = []
    qpos_arr.append(env.agent.robot.get_qpos())

    obs, _ = env.reset()

    if gt_tcp_pose_at_robot_base is not None:
        # reset robot's end-effector pose to be gt_tcp_pose_at_robot_base
        controller = env.agent.controller.controllers["arm"]
        cur_qpos = env.agent.robot.get_qpos()
        init_arm_qpos = controller.compute_ik(gt_tcp_pose_at_robot_base)
        cur_qpos[controller.joint_indices] = init_arm_qpos
        env.agent.reset(cur_qpos)

    image = (env.get_obs()["image"][camera]["Color"][..., :3] * 255).astype(np.uint8)
    images.append(image)

    if "server" in model_type:
        octo_model = OctoServerInference(model_type, policy_setup=policy_setup, action_scale=1.0)
    else:
        octo_model = OctoInference(model_type, policy_setup=policy_setup, action_scale=1.0)

    # Reset Octo model
    octo_model.reset(instruction)

    timestep = 0
    truncated = False
    if input_video is not None:
        loop_criterion = lambda timestep: timestep < len(input_video) - 1
    else:
        loop_criterion = lambda timestep: not truncated

    # Step the environment
    while loop_criterion(timestep):
        if input_video is not None:
            raw_action, action = octo_model.step(input_video[timestep])
        else:
            raw_action, action = octo_model.step(image)
        predicted_actions.append(raw_action)
        print(timestep, raw_action)

        obs, reward, terminated, truncated, info = env.step(
            np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
        )

        # debug
        # controller = env.agent.controller.controllers['arm']
        # cur_qpos = env.agent.robot.get_qpos()
        # cur_qpos[controller.joint_indices] = env.agent.controller.controllers['arm']._target_qpos
        # env.agent.reset(cur_qpos)

        image = obs["image"][camera]["rgb"]
        images.append(image)
        qpos_arr.append(env.agent.robot.get_qpos())
        timestep += 1

    if input_video is not None:
        for i in range(len(images)):
            images[i] = cv2.resize(images[i], (input_video[i].shape[1], input_video[i].shape[0]))
            images[i] = np.concatenate(
                [
                    images[i]
                    if inpainting_img_path is not None
                    else np.array(images[i] * 0.7 + input_video[i] * 0.3).astype(np.uint8),
                    input_video[i],
                ],
                axis=1,
            )

    os.makedirs(f"{save_root}", exist_ok=True)
    os.makedirs(f"{save_root}/save_qpos", exist_ok=True)

    octo_model.visualize_epoch(predicted_actions, images, save_path=f"{save_root}/{save_name}.png")
    video_path = f"{save_root}/{save_name}.mp4"
    write_video(video_path, images, fps=5)
    np.save(f"{save_root}/save_qpos/{save_name}_qpos.npy", np.array(qpos_arr))


if __name__ == "__main__":
    os.environ["DISPLAY"] = ""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=3072)])

    mp4_path = None
    inpainting_img_path = "ManiSkill2_real2sim/data/debug/bridge_real_stackcube_2.png"
    instruction = "stack the green block on the yellow block"
    gt_tcp_pose_at_robot_base = None
    camera = "3rd_view_camera"

    # mp4_path = 'ManiSkill2_real2sim/data/Octo_HF_Small_SanDiego_feb1/2-01-24/2024-02-01_15-52-27_hf-small_processed.mp4'
    # inpainting_img_path = None
    # instruction = 'stack the green block on the yellow block'
    # gt_tcp_pose_at_robot_base = None
    # camera = '3rd_view_camera'

    # inpainting_img_path = 'ManiSkill2_real2sim/data/debug/rt1_real_standing_coke_can_1_cleanup.png'
    # inpainting_img_path = 'ManiSkill2_real2sim/data/real_inpainting/pick_coke_can_real_misc/google_horizontal_coke_can_b0_cleanup.png'
    # instruction = 'pick coke can'
    # gt_tcp_pose_at_robot_base = None
    # camera = 'overhead_camera'

    # mp4_path = 'ManiSkill2_real2sim/data/octo_google_robot_inference_feb13/215953.mp4'
    # inpainting_img_path = None
    # instruction = 'pick coke can'
    # gt_tcp_pose_at_robot_base = None
    # camera = 'overhead_camera'

    # mp4_path = None
    # inpainting_img_path = 'ManiSkill2_real2sim/data/octo_google_robot_inference_feb13/215953_frame0.png'
    # instruction = 'pick coke can'
    # gt_tcp_pose_at_robot_base = None
    # camera = 'overhead_camera'

    if mp4_path is not None:
        input_video = media.read_video(mp4_path)
    else:
        input_video = None

    # main(input_video, inpainting_img_path, instruction, gt_tcp_pose_at_robot_base, camera)

    main(
        input_video,
        inpainting_img_path,
        instruction,
        gt_tcp_pose_at_robot_base,
        camera,
        model_type="octo-small",
        robot="widowx_camera_setup2",
        control_mode="arm_pd_ee_target_delta_pose_align_gripper_pd_joint_pos",
    )

    # main(input_video, inpainting_img_path, instruction, gt_tcp_pose_at_robot_base, camera,
    #      model_type='octo-base',
    #      policy_setup='google_robot', robot='google_robot_static',
    #      control_freq=3, max_episode_steps=60,
    #      control_mode='arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner',
    #      urdf_version="recolor_tabletop_visual_matching_1")
