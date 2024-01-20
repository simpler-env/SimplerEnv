import numpy as np
import os
import tensorflow as tf
import mediapy as media
import cv2

from sapien.core import Pose
from real2sim.rt1.rt1_model import RT1Inference
from real2sim.utils.visualization import write_video
from real2sim.utils.env.env_builder import build_maniskill2_env

def main(input_video, impainting_img_path, instruction, gt_tcp_pose_at_robot_base=None,
         ckpt_path='rt_1_x_tf_trained_for_002272480_step', camera='3rd_view_camera',
         control_freq=5):
    # Build RT-1 Model
    rt1_model = RT1Inference(saved_model_path=ckpt_path, action_scale=1.0,
                             policy_setup="widowx_bridge")
    
    """
    rpy 5hz:
    arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos
    arm_pd_ee_delta_pose_align2_gripper_pd_joint_pos
    arm_pd_ee_delta_pose_align_gripper_pd_joint_pos
    """
    # Create environment
    env = build_maniskill2_env(
        'PickCube-v0',
        control_mode='arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos',
        # control_mode='arm_pd_ee_delta_pose_align_gripper_pd_joint_target_pos',
        obs_mode='rgbd',
        robot='widowx',
        sim_freq=500,
        max_episode_steps=90,
        control_freq=control_freq,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=impainting_img_path,
        rgb_overlay_cameras=[camera],
    )
    
    # Reset and initialize environment
    predicted_actions = []
    images = []
    
    obs, _ = env.reset()
    env.agent.robot.set_pose(Pose([0, 0, 1]))
    
    if gt_tcp_pose_at_robot_base is not None:
        controller = env.agent.controller.controllers['arm']
        cur_qpos = env.agent.robot.get_qpos()
        init_arm_qpos = controller.compute_ik(gt_tcp_pose_at_robot_base)
        cur_qpos[controller.joint_indices] = init_arm_qpos
        env.agent.reset(cur_qpos)
    
    image = (env.get_obs()["image"][camera]['Color'][..., :3] * 255).astype(np.uint8)
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
                action['gripper']
                ]
            )
        )
        
        image = obs['image'][camera]['rgb']
        images.append(image)
        timestep += 1

    if input_video is not None:
        for i in range(len(images)):
            images[i] = np.concatenate(
                [cv2.resize(images[i], (input_video[i].shape[1], input_video[i].shape[0])), 
                input_video[i]], 
                axis=1
            )
    video_path = f'/home/xuanlin/Downloads/debug_rt1_inference_widowx1.mp4'
    write_video(video_path, images, fps=5)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['DISPLAY'] = ''
    
    mp4_path = '/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/debug/bridge_real_1.mp4'
    # mp4_path = None
    impainting_img_path = '/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/debug/bridge_real_1_cleanup.png'
    instruction = 'Place the can to the left of the pot.'
    gt_tcp_pose_at_robot_base = Pose([0.298068, -0.114657, 0.10782], [0.750753, 0.115962, 0.642171, -0.102661])
    camera = '3rd_view_camera_bridge'
    ckpt_path = '/home/xuanlin/Real2Sim/rt_1_x_tf_trained_for_002272480_step/'
    
    if mp4_path is not None:
        input_video = media.read_video(mp4_path)
    else:
        input_video = None
    
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])

    main(input_video, impainting_img_path, instruction, gt_tcp_pose_at_robot_base, ckpt_path, camera)