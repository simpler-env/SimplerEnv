import numpy as np
import os
import tensorflow as tf
import mediapy as media
import cv2

from real2sim.rt1.rt1_model import RT1Inference
from real2sim.utils.visualization import write_video
from real2sim.utils.env.env_builder import build_maniskill2_env

def main(input_video, impainting_img_path, instruction, 
         ckpt_path='rt_1_x_tf_trained_for_002272480_step',
         control_freq=3):
    # Build RT-1 Model
    rt1_model = RT1Inference(saved_model_path=ckpt_path, action_scale=1.0)
    
    # Create environment
    env, instruction = build_maniskill2_env(
        'PickCube-v0',
        control_mode='arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_pos',
        # control_mode='arm_pd_ee_delta_pose_align_gripper_pd_joint_target_pos',
        obs_mode='rgbd',
        robot='google_robot_static',
        sim_freq=540,
        max_episode_steps=60,
        control_freq=control_freq,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=impainting_img_path,
        rgb_overlay_cameras=['overhead_camera'],
        instruction=instruction
    )
    
    # Reset and initialize environment
    predicted_actions = []
    images = []
    
    obs, _ = env.reset()
    image = obs['image']['overhead_camera']['rgb']
    images.append(image)
    predicted_terminated, terminated, truncated = False, False, False

    # Reset RT-1 model
    rt1_model.reset(instruction)

    timestep = 0
    # Step the environment
    if input_video is not None:
        loop_criterion = lambda timestep: timestep < len(input_video) - 1
    else:
        loop_criterion = lambda timestep: not truncated
    while loop_criterion(timestep):
        cur_gripper_closedness = env.agent.get_gripper_closedness()
        
        if input_video is not None:
            raw_action, action = rt1_model.step(input_video[timestep], cur_gripper_closedness)
        else:
            raw_action, action = rt1_model.step(image, cur_gripper_closedness)
        predicted_actions.append(raw_action)
        print(timestep, raw_action)
        predicted_terminated = bool(action['terminate_episode'][0] > 0)
        
        obs, reward, terminated, truncated, info = env.step(
            np.concatenate(
                [action['world_vector'], 
                action['rot_axangle'],
                action['gripper_closedness_action']
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
    video_path = f'/home/xuanlin/Downloads/debug.mp4'
    write_video(video_path, images, fps=5)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['DISPLAY'] = ''
    # mp4_path = '/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/debug/rt1_real_vertical_coke_can_1.mp4'
    # impainting_img_path = '/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/debug/rt1_real_vertical_coke_can_1_cleanup.png'
    mp4_path = None
    impainting_img_path = '/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/real_impainting/google_vertical_coke_can_eval_1_cleanup.png'
    instruction = 'pick coke can'
    ckpt_path = '/home/xuanlin/Real2Sim/rt1_xid45615428_000315000/' # 'rt_1_x_tf_trained_for_002272480_step'
    
    if mp4_path is not None:
        input_video = media.read_video(mp4_path)
    else:
        input_video = None
    
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])

    main(input_video, impainting_img_path, instruction)