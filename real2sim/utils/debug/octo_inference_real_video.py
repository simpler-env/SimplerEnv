"""
Given an impainting image, 
query the Octo model to predict actions and visualize the predicted actions in the environment,
where the policy input is the impainting image plus the robot arm rendered in it.
Another use: If a video is given, the video frames will be used as the input to the Octo model.
"""

import numpy as np
import os, cv2
import mediapy as media
import tensorflow as tf

from real2sim.octo.octo_model import OctoInference
from real2sim.utils.visualization import write_video
from real2sim.utils.env.env_builder import build_maniskill2_env
from sapien.core import Pose

def main(input_video, impainting_img_path, instruction,
         gt_tcp_pose_at_robot_base=None, camera='3rd_view_camera',
         model_type='octo-base', policy_setup='widowx_bridge', robot='widowx',
         control_freq=5,
         control_mode='arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos'):
    
    # Create environment
    env = build_maniskill2_env(
        'GraspSingleDummy-v0',
        control_mode=control_mode,
        obs_mode='rgbd',
        robot=robot,
        sim_freq=500,
        max_episode_steps=90,
        control_freq=control_freq,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=impainting_img_path,
        rgb_overlay_cameras=[camera],
    )
    print(instruction)
    
    # Reset and initialize environment
    predicted_actions = []
    images = []
    
    obs, _ = env.reset()
    
    if gt_tcp_pose_at_robot_base is not None:
        # reset robot's end-effector pose to be gt_tcp_pose_at_robot_base
        controller = env.agent.controller.controllers['arm']
        cur_qpos = env.agent.robot.get_qpos()
        init_arm_qpos = controller.compute_ik(gt_tcp_pose_at_robot_base)
        cur_qpos[controller.joint_indices] = init_arm_qpos
        env.agent.reset(cur_qpos)
    
    image = (env.get_obs()["image"][camera]['Color'][..., :3] * 255).astype(np.uint8)
    images.append(image)

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
            np.concatenate(
                [action['world_vector'], 
                action['rot_axangle'],
                action['gripper']
                ]
            )
        )
        
        image = obs['image'][camera]['rgb']
        images.append(image)
        timestep += 1

    octo_model.visualize_epoch(predicted_actions, images, save_path='debug_logs/debug_octo_inference.png')
    
    if input_video is not None:
        for i in range(len(images)):
            images[i] = np.concatenate(
                [cv2.resize(images[i], (input_video[i].shape[1], input_video[i].shape[0])), 
                input_video[i]], 
                axis=1
            )
    video_path = f'debug_logs/debug_octo_inference.mp4'
    write_video(video_path, images, fps=5)


if __name__ == '__main__':
    """
    MS2_ASSET_DIR=./ManiSkill2_real2sim/data python real2sim/utils/debug/octo_inference_real_video.py
    """
    
    os.environ['DISPLAY'] = ''
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
    
    mp4_path = None
    # impainting_img_path = 'ManiSkill2_real2sim/data/debug/bridge_real_stackcube_2.png'
    # instruction = 'stack the green block on the yellow block'
    # gt_tcp_pose_at_robot_base = None
    # camera = '3rd_view_camera'
    mp4_path = '/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/Octo_HF_Small_SanDiego_feb1/2-01-24/2024-02-01_15-52-27_hf-small_processed.mp4'
    impainting_img_path = None
    instruction = 'stack the green block on the yellow block'
    gt_tcp_pose_at_robot_base = None
    camera = '3rd_view_camera'
    
    # impainting_img_path = 'ManiSkill2_real2sim/data/debug/rt1_real_standing_coke_can_1_cleanup.png'
    # impainting_img_path = 'ManiSkill2_real2sim/data/real_impainting/pick_coke_can_real_misc/google_horizontal_coke_can_b0_cleanup.png'
    # instruction = 'pick coke can'
    # gt_tcp_pose_at_robot_base = None
    # camera = 'overhead_camera'
    
    if mp4_path is not None:
        input_video = media.read_video(mp4_path)
    else:
        input_video = None
    
    # main(input_video, impainting_img_path, instruction, gt_tcp_pose_at_robot_base, camera)
    
    main(input_video, impainting_img_path, instruction, gt_tcp_pose_at_robot_base, camera, model_type='octo-small')
    
    # main(input_video, impainting_img_path, instruction, gt_tcp_pose_at_robot_base, camera,
    #      model_type='./checkpoints/octo_feb1_24/octo_base_gpu_final',
    #      policy_setup='google_robot', robot='google_robot_static',
    #     #  control_freq=3,
    #     #  control_mode='arm_pd_ee_target_delta_pose_align_gripper_pd_joint_pos')
    #      control_freq=3,
    #      control_mode='arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner')