"""
Given an impainting image, 
query the rt1 model to predict actions and visualize the predicted actions in the environment,
where the policy input is the impainting image plus the robot arm rendered in it.
Another use: If a video is given, the video frames will be used as the input to the Octo model.
"""

import numpy as np
import os
import tensorflow as tf
import mediapy as media
import cv2

from real2sim.rt1.rt1_model import RT1Inference
from real2sim.utils.visualization import write_video
from real2sim.utils.env.env_builder import build_maniskill2_env
from sapien.core import Pose

def main(input_video, impainting_img_path, instruction, 
         ckpt_path='rt_1_x_tf_trained_for_002272480_step',
         control_freq=3,
         control_mode='arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner',
         policy_setup='google_robot',
         robot='google_robot_static',
         init_tcp_pose_at_robot_base=None,
         init_robot_base_pos=None,
         overlay_camera='overhead_camera'):
    
    # Build RT-1 Model
    rt1_model = RT1Inference(saved_model_path=ckpt_path, action_scale=1.0,
                             policy_setup=policy_setup)
    
    # Create environment
    env = build_maniskill2_env(
        'GraspSingleDummy-v0',
        control_mode=control_mode,
        obs_mode='rgbd',
        robot=robot,
        sim_freq=540,
        max_episode_steps=60,
        control_freq=control_freq,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=impainting_img_path,
        rgb_overlay_cameras=[overlay_camera],
    )
    
    # Reset and initialize environment
    predicted_actions = []
    images = []
    
    obs, _ = env.reset()
    if init_robot_base_pos is not None:
        env.agent.robot.set_pose(Pose(init_robot_base_pos))
    if init_tcp_pose_at_robot_base is not None:
        controller = env.agent.controller.controllers['arm']
        cur_qpos = env.agent.robot.get_qpos()
        init_arm_qpos = controller.compute_ik(init_tcp_pose_at_robot_base)
        cur_qpos[controller.joint_indices] = init_arm_qpos
        env.agent.reset(cur_qpos)
        
    image = (env.get_obs()["image"][overlay_camera]['Color'][..., :3] * 255).astype(np.uint8)
    images.append(image)
    truncated = False

    # Reset RT-1 model
    rt1_model.reset(instruction)

    timestep = 0
    if input_video is not None:
        loop_criterion = lambda timestep: timestep < len(input_video) - 1
    else:
        loop_criterion = lambda timestep: not truncated
        
    # Step the environment
    while loop_criterion(timestep):
        if input_video is not None:
            raw_action, action = rt1_model.step(input_video[timestep])
        else:
            raw_action, action = rt1_model.step(image)
        predicted_actions.append(raw_action)
        print(timestep, raw_action)
        predicted_terminated = bool(action['terminate_episode'][0] > 0)
        
        obs, reward, terminated, truncated, info = env.step(
            np.concatenate(
                [action['world_vector'], 
                action['rot_axangle'],
                action['gripper']
                ]
            )
        )
        
        image = obs['image']['overhead_camera']['rgb']
        images.append(image)
        timestep += 1

    if input_video is not None:
        for i in range(len(images)):
            images[i] = np.concatenate(
                [cv2.resize(images[i], (input_video[i].shape[1], input_video[i].shape[0])), 
                input_video[i]], 
                axis=1
            )
    video_path = f'debug_logs/debug_rt1_inference.mp4'
    write_video(video_path, images, fps=5)


if __name__ == '__main__':
    """
    MS2_ASSET_DIR=./ManiSkill2_real2sim/data python real2sim/utils/debug/rt1_inference_real_video.py
    """
    
    os.environ['DISPLAY'] = ''
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
    
    # mp4_path = 'ManiSkill2_real2sim/data/debug/rt1_real_vertical_coke_can_1.mp4'
    # impainting_img_path = 'ManiSkill2_real2sim/data/debug/rt1_real_vertical_coke_can_1_cleanup.png'
    mp4_path = None
    impainting_img_path = 'ManiSkill2_real2sim/data/real_impainting/move_near_real_obj_variants/move_near_real_1_1.png'
    instruction = 'move blue plastic bottle near pepsi can'
    ckpt_path = 'checkpoints/xid77467904_000400120/'
    # mp4_path = 'data/debug/bridge_real_1.mp4'
    # impainting_img_path = 'data/debug/bridge_real_1_cleanup.png'
    # instruction = 'Place the can to the left of the pot.'
    # init_tcp_pose_at_robot_base = Pose([0.298068, -0.114657, 0.10782], [0.750753, 0.115962, 0.642171, -0.102661])
    # camera = '3rd_view_camera_bridge'
    # ckpt_path = 'checkpoints/rt_1_x_tf_trained_for_002272480_step/'
    
    if mp4_path is not None:
        input_video = media.read_video(mp4_path)
    else:
        input_video = None
    
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])

    main(input_video, impainting_img_path, instruction, ckpt_path)
    # main(input_video, impainting_img_path, instruction, ckpt_path, 
    #     control_freq=5,
    #     control_mode='arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos',
    #     policy_setup='widowx_bridge', robot='widowx',
    #     init_tcp_pose_at_robot_base=init_tcp_pose_at_robot_base,
    #     init_robot_base_pos=[0,0,1], # move the widowx robot in the sky to avoid collision w/ ground
    #     overlay_camera='3rd_view_camera_bridge'
    # )