import numpy as np
from PIL import Image
from IPython import display
import tqdm, os
import mani_skill2.envs, gymnasium as gym
from transforms3d.euler import euler2axangle, euler2quat, euler2mat
from transforms3d.quaternions import quat2axangle, axangle2quat, mat2quat
from transforms3d.axangles import mat2axangle
from sapien.core import Pose
from copy import deepcopy
import pickle
import mediapy as media
import cv2

from real2sim.utils.visualization import write_video
from real2sim.octo.octo_model import OctoInference

DATASETS = ['fractal20220817_data', 'bridge']

def dataset2path(dataset_name):
    if dataset_name == 'robo_net':
        version = '1.0.0'
    elif dataset_name == 'language_table':
        version = '0.0.1'
    else:
        version = '0.1.0'
    return f'gs://gresearch/robotics/{dataset_name}/{version}'


def main(gt_images, actions, traj_name,
         control_mode='arm_pd_ee_delta_pose_align2_gripper_pd_joint_pos'):
         # control_mode='arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_target_pos_interpolate_by_planner'):
    sim_freq, control_freq = 510, 5
    action_scale = 1.0
    env = gym.make('PickCube-v0',
                        control_mode=control_mode,
                        obs_mode='rgbd',
                        robot='widowx',
                        sim_freq=sim_freq,
                        control_freq=control_freq,
                        max_episode_steps=100,
                        # camera_cfgs={"add_segmentation": True},
                        # rgb_overlay_path=f'/home/xuanlin/Downloads/{episode_id}_0_cleanup.png',
                        # rgb_overlay_cameras=['overhead_camera'],
            )
    
    
        
    images = []
    ee_poses_at_base = []
    obs, _ = env.reset()
    env.agent.robot.set_pose(Pose([0, 0, 1]))

    mat_transform = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]], dtype=np.float64)
    
    images.append(obs['image']['3rd_view_camera']['rgb'])
    ee_poses_at_base.append(env.agent.robot.pose.inv() * env.tcp.pose)
    
    for i in range(len(actions) - 1):
        current_pose_at_robot_base = env.agent.robot.pose.inv() * env.tcp.pose
        
        gt_action_world_vector = np.asarray(actions[i][:3], dtype=np.float64)
        gt_action_rotation_delta = np.asarray(actions[i][3:6], dtype=np.float64)
        gt_action_rotation_ax, gt_action_rotation_angle = euler2axangle(*gt_action_rotation_delta)
        gt_action_rotation_axangle = gt_action_rotation_ax * gt_action_rotation_angle
        gt_action_gripper_closedness_action = 2.0 * actions[i][6:] - 1.0
        action = np.concatenate(
                            [gt_action_world_vector * action_scale, 
                            gt_action_rotation_axangle * action_scale,
                            gt_action_gripper_closedness_action,
                            ],
                        ).astype(np.float64)
        
        obs, reward, terminated, truncated, info = env.step(action) 
        images.append(obs['image']['3rd_view_camera']['rgb'])
        ee_poses_at_base.append(env.agent.robot.pose.inv() * env.tcp.pose)
        arm_ctrl_mode, gripper_ctrl_mode = control_mode.split('gripper')
        
    for i in range(len(images)):
        images[i] = np.concatenate([images[i], cv2.resize(np.asarray(gt_images[i]), (images[i].shape[1], images[i].shape[0]))], axis=1)
    write_video(f'/home/xuanlin/Downloads/tmp_widowx_debug/{traj_name}.mp4', images, fps=5)
    

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['DISPLAY'] = ''
    root_traj_path = '/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/octo_eval_data_langcondition_Jan12/stack_blocks'
    traj_name = '2024-01-12_14-52-20'
    mp4_path = f'{root_traj_path}/{traj_name}_hf.mp4'
    video = media.read_video(mp4_path)
    video = video[:, video.shape[1] // 2:, :, :]
    
    pkl_path = f'{root_traj_path}/actions_and_obs_{traj_name}.pkl'
    with open(pkl_path, 'rb') as f:
        actions_and_obs = pickle.load(f)
    actions = np.array(actions_and_obs['actions'])[:, 0, :]
    gt_images = np.array([x[-1]['images'][0] for x in actions_and_obs['obs']])
    gt_images = gt_images[-len(actions):]
    
    # model = OctoInference(model_type='octo-base', action_scale=1.0)
    # action_mean, action_std = model.action_mean, model.action_std
    # assert action_mean.ndim ==1 and action_std.ndim == 1
    action_mean = np.array([ 0.00021161,  0.00012614, -0.00017022, -0.00015062, -0.00023831,
        0.00025646,  0.        ])
    action_std = np.array([0.00963721, 0.0135066 , 0.01251861, 0.02806791, 0.03016905,
       0.07632624, 1.        ])
    actions = actions * action_std[None] + action_mean[None]
    
    main(gt_images, actions, traj_name)