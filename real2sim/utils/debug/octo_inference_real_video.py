import numpy as np
import os
import mediapy as media

from real2sim.octo.octo_model import OctoInference
from real2sim.utils.visualization import write_video
from real2sim.utils.env.env_builder import build_maniskill2_env
from sapien.core import Pose

def main(input_video, impainting_img_path, instruction,
         gt_tcp_pose_at_robot_base=None,
         model_type='octo-base',
         control_freq=5):
    
    # Create environment
    env, instruction = build_maniskill2_env(
        'PickCube-v0',
        control_mode='arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos',
        obs_mode='rgbd',
        robot='widowx',
        sim_freq=500,
        control_freq=control_freq,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=impainting_img_path,
        rgb_overlay_cameras=['3rd_view_camera'],
        instruction=instruction
    )
    print(instruction)
    
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
    
    image = (env.get_obs()["image"]["3rd_view_camera"]['Color'][..., :3] * 255).astype(np.uint8)
    images.append(image)

    octo_model = OctoInference(model_type, action_scale=1.0)
    # Reset Octo model
    octo_model.reset(instruction)

    timestep = 0
    # Step the environment
    while timestep < len(input_video) - 1:
        raw_action, action = octo_model.step(image)
        # raw_action, action = octo_model.step(input_video[timestep])
        predicted_actions.append(raw_action)
        print(timestep, raw_action)
        
        obs, reward, terminated, truncated, info = env.step(
            np.concatenate(
                [action['world_vector'], 
                action['rot_axangle'],
                action['open_gripper']
                ]
            )
        )
        
        image = obs['image']['3rd_view_camera']['rgb']
        images.append(image)
        timestep += 1

    octo_model.visualize_epoch(predicted_actions, images, save_path='/home/xuanlin/Downloads/debug.png')
    
    for i in range(len(images)):
        images[i] = np.concatenate([images[i], input_video[i]], axis=1)
    video_path = f'/home/xuanlin/Downloads/debug.mp4'
    write_video(video_path, images, fps=5)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['DISPLAY'] = ''
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    
    # mp4_path = '/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/debug/bridge_real_1.mp4'
    # impainting_img_path = '/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/debug/bridge_real_1_cleanup.png'
    # instruction = 'Place the can to the left of the pot.'
    # gt_tcp_pose_at_robot_base = Pose([0.298068, -0.114657, 0.10782], [0.750753, 0.115962, 0.642171, -0.102661])
    
    mp4_path = '/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/debug/bridge_real_2.mp4'
    impainting_img_path = '/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/debug/bridge_real_2_cleanup.png'
    instruction = 'Pick up the bowl.'
    # instruction = 'Move the kadai and place it at the right edge of the table.' # doesn't understand the instruction...
    gt_tcp_pose_at_robot_base = Pose([0.350166, -0.0610973, 0.157404], [0.73995, -0.377095, 0.438111, 0.343994])
    
    input_video = media.read_video(mp4_path)
    
    main(input_video, impainting_img_path, instruction, gt_tcp_pose_at_robot_base)