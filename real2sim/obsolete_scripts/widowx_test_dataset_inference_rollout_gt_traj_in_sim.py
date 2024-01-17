import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from PIL import Image
from IPython import display
import tqdm, os
import mani_skill2.envs, gymnasium as gym
from transforms3d.euler import euler2axangle, euler2quat, euler2mat
from transforms3d.quaternions import quat2axangle, axangle2quat, mat2quat
from transforms3d.axangles import mat2axangle
from sapien.core import Pose
from copy import deepcopy
import cv2

from real2sim.utils.visualization import write_video

DATASETS = ['fractal20220817_data', 'bridge']

def dataset2path(dataset_name):
    if dataset_name == 'robo_net':
        version = '1.0.0'
    elif dataset_name == 'language_table':
        version = '0.0.1'
    else:
        version = '0.1.0'
    return f'gs://gresearch/robotics/{dataset_name}/{version}'


def main(dset_iter, iter_num, episode_id, set_actual_reached=False, 
         control_mode='arm_pd_ee_delta_pose_base_gripper_pd_joint_pos'):
    for _ in range(iter_num):
        episode = next(dset_iter)
    print("episode tfds id", episode['tfds_id'])
    episode_steps = list(episode['steps'])
    
    language_instruction = episode_steps[0]['observation']['natural_language_instruction']
    print(language_instruction)
    
    sim_freq, control_freq, action_repeat = 510, 3, 5
    action_scale = 1.0
    env = gym.make('PickCube-v0',
                        control_mode=control_mode,
                        obs_mode='rgbd',
                        robot='widowx',
                        sim_freq=sim_freq,
                        control_freq=control_freq * action_repeat,
                        max_episode_steps=50 * action_repeat,
                        # camera_cfgs={"add_segmentation": True},
                        # rgb_overlay_path=f'/home/xuanlin/Downloads/{episode_id}_0_cleanup.png',
                        # rgb_overlay_cameras=['overhead_camera'],
            )
    
    images = []
    ee_poses_at_base = []
    gt_images = []
    obs, _ = env.reset()
    env.agent.robot.set_pose(Pose([0, 0, 1]))

    mat_transform = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]], dtype=np.float64)
    # move tcp pose in environment to the gt tcp pose wrt robot base
    for _ in range(50):
        gt_tcp_pose_at_robot_base = Pose(
            p=episode_steps[0]['observation']['state'][:3],
            q=mat2quat(euler2mat(*np.array(episode_steps[0]['observation']['state'][3:6], dtype=np.float64)) @ mat_transform),
        )
        current_pose_at_robot_base = env.agent.robot.pose.inv() * env.tcp.pose
        # print("current tcp pose", current_pose_at_robot_base, "gt tcp pose @ timestep=0", gt_tcp_pose_at_robot_base)
        # delta_tcp_pose = Pose(
        #     p=gt_tcp_pose_at_robot_base.p - current_pose_at_robot_base.p,
        #     q=(gt_tcp_pose_at_robot_base * current_pose_at_robot_base.inv()).q,
        # ) # if align
        delta_tcp_pose = gt_tcp_pose_at_robot_base * current_pose_at_robot_base.inv()
        action_translation = delta_tcp_pose.p
        action_rot_ax, action_rot_angle = quat2axangle(np.array(delta_tcp_pose.q, dtype=np.float64))
        if np.abs(action_rot_angle) > np.abs(action_rot_angle - 2 * np.pi):
            action_rot_angle = action_rot_angle - 2 * np.pi
        action_rotation = action_rot_ax * action_rot_angle
        action = np.concatenate([action_translation, action_rotation, [0]])
        obs, *_ = env.step(action)
        # print(env.agent.robot.get_qpos())
    
    images.append(obs['image']['3rd_view_camera_bridge']['rgb'])
    ee_poses_at_base.append(env.agent.robot.pose.inv() * env.tcp.pose)
    
    for i in range(len(episode_steps) - 1):
        episode_step = episode_steps[i] # episode_step['observation']['base_pose_tool_reached'] = [xyz, quat xyzw]
        gt_images.append(episode_step['observation']['image'])
        next_episode_step = episode_steps[i + 1]
        current_pose_at_robot_base = env.agent.robot.pose.inv() * env.tcp.pose
        
        # # uncomment to debug
        # cur_xyz = np.asarray(episode_step['observation']['state'][:3], dtype=np.float64)
        # cur_rpy = np.asarray(episode_step['observation']['state'][3:6], dtype=np.float64)
        # cur_ax, cur_angle = mat2axangle(euler2mat(*cur_rpy) @ mat_transform)
        # current_pose_at_robot_base = Pose(p=np.array(cur_xyz), q=axangle2quat(cur_ax, cur_angle))
                        
        if not set_actual_reached and False:
            gt_action_world_vector = np.asarray(episode_step['action']['world_vector'], dtype=np.float64)
            gt_action_rotation_delta = np.asarray(episode_step['action']['rotation_delta'], dtype=np.float64)
            gt_action_rotation_ax, gt_action_rotation_angle = euler2axangle(*gt_action_rotation_delta)
            gt_action_rotation_axangle = gt_action_rotation_ax * gt_action_rotation_angle
            gt_action_gripper_closedness_action = 2.0 * (np.array(episode_step['action']['open_gripper'])[None]) - 1.0
            # target_tcp_pose_at_base = Pose(
            #     p=gt_action_rotation_delta, q=axangle2quat(gt_action_rotation_ax, gt_action_rotation_angle)
            # ) * current_pose_at_robot_base
            # target_tcp_pose_at_base = (
            #     (
            #     Pose(p=current_pose_at_robot_base.p) 
            #     * Pose(p=gt_action_world_vector, q=axangle2quat(gt_action_rotation_ax, gt_action_rotation_angle)) 
            #     * Pose(p=current_pose_at_robot_base.p).inv()
            #     ) 
            #     * current_pose_at_robot_base
            # )
            target_tcp_pose_at_base = Pose(p=current_pose_at_robot_base.p + gt_action_world_vector * action_scale,
                                           q=(Pose(q=axangle2quat(gt_action_rotation_ax, gt_action_rotation_angle)) 
                                              * Pose(q=current_pose_at_robot_base.q)).q
            )
            target_delta_pose_at_robot_base = target_tcp_pose_at_base * current_pose_at_robot_base.inv()
            gt_action_world_vector = target_delta_pose_at_robot_base.p
            gt_action_rotation_ax, gt_action_rotation_angle = quat2axangle(target_delta_pose_at_robot_base.q)
            gt_action_rotation_axangle = gt_action_rotation_ax * gt_action_rotation_angle
            
            next_xyz = np.asarray(next_episode_step['observation']['state'][:3], dtype=np.float64)
            next_rpy = np.asarray(next_episode_step['observation']['state'][3:6], dtype=np.float64)
            next_ax, next_angle = mat2axangle(euler2mat(*next_rpy) @ mat_transform)
            next_pose_at_robot_base = Pose(p=np.array(next_xyz), q=axangle2quat(next_ax, next_angle))
            print("cur_pose", current_pose_at_robot_base, "gt action world vector", episode_step['action']['world_vector'], 
                  "gt action rotation delta", episode_step['action']['rotation_delta'])
            print("target tcp pose at base", target_tcp_pose_at_base)
            print("actual target tcp pose at base", next_pose_at_robot_base)
            print("*" * 20)
        else:
            # assert control_mode == 'arm_pd_ee_target_delta_pose_base_gripper_pd_joint_pos'
            next_xyz = np.asarray(next_episode_step['observation']['state'][:3], dtype=np.float64)
            next_rpy = np.asarray(next_episode_step['observation']['state'][3:6], dtype=np.float64)
            next_ax, next_angle = mat2axangle(euler2mat(*next_rpy) @ mat_transform)
            next_pose_at_robot_base = Pose(p=np.array(next_xyz), q=axangle2quat(next_ax, next_angle))
            print("current tcp pose at base", current_pose_at_robot_base, "next tcp pose at base", next_pose_at_robot_base)
            target_delta_pose_at_robot_base = next_pose_at_robot_base * current_pose_at_robot_base.inv()
                        
            gt_action_world_vector = target_delta_pose_at_robot_base.p
            gt_action_rotation_ax, gt_action_rotation_angle = quat2axangle(np.array(target_delta_pose_at_robot_base.q, dtype=np.float64))
            gt_action_rotation_axangle = gt_action_rotation_ax * gt_action_rotation_angle
            gt_action_gripper_closedness_action = 2.0 * (np.array(next_episode_step['observation']['state'][6:])) - 1.0
            target_gripper_closedness_action = gt_action_gripper_closedness_action
            
            target_tcp_pose_at_base = next_pose_at_robot_base
            
        # print(i, "gripper", env.agent.get_gripper_closedness(), episode_step['action']['gripper_closedness_action'])
        action = np.concatenate(
                            [gt_action_world_vector * action_scale, 
                            gt_action_rotation_axangle * action_scale,
                            gt_action_gripper_closedness_action,
                            ],
                        ).astype(np.float64)
        
        obs, reward, terminated, truncated, info = env.step(action) 
        images.append(obs['image']['3rd_view_camera_bridge']['rgb'])
        ee_poses_at_base.append(env.agent.robot.pose.inv() * env.tcp.pose)
        arm_ctrl_mode, gripper_ctrl_mode = control_mode.split('gripper')
        for _ in range(action_repeat - 1):
            interp_action = action.copy()
            if 'target' in arm_ctrl_mode:
                interp_action[:6] *= 0
            else:
                cur_tcp_pose_at_base = env.agent.robot.pose.inv() * env.tcp.pose
                delta_tcp_pose_at_base = target_tcp_pose_at_base * cur_tcp_pose_at_base.inv()
                interp_action[:3] = delta_tcp_pose_at_base.p
                interp_rot_ax, interp_rot_angle = quat2axangle(np.array(delta_tcp_pose_at_base.q, dtype=np.float64))
                interp_action[3:6] = interp_rot_ax * interp_rot_angle
                
            if 'target' in gripper_ctrl_mode:
                interp_action[6:] *= 0
            obs, reward, terminated, truncated, info = env.step(interp_action)
            images.append(obs['image']['3rd_view_camera_bridge']['rgb'])
            ee_poses_at_base.append(env.agent.robot.pose.inv() * env.tcp.pose)
            
    # for i, ee_pose in enumerate(ee_poses_at_base):
    #     print(i, "ee pose wrt robot base", ee_pose)
    gt_images = [gt_images[np.clip((i - 1) // action_repeat + 1, 0, len(gt_images) - 1)] for i in range(len(images))]
    for i in range(len(images)):
        images[i] = np.concatenate([images[i], cv2.resize(np.asarray(gt_images[i]), (images[i].shape[1], images[i].shape[0]))], axis=1)
    if not set_actual_reached:
        write_video(f'/home/xuanlin/Downloads/tmp_widowx_debug/{episode_id}.mp4', images, fps=5)
    else:
        write_video(f'/home/xuanlin/Downloads/tmp_widowx_debug/{episode_id}_actual_reached.mp4', images, fps=5)
    write_video(f'/home/xuanlin/Downloads/tmp_widowx_debug/{episode_id}_gt.mp4', gt_images, fps=5)
    

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['DISPLAY'] = ''
    dataset_name = DATASETS[1]
    dset = tfds.builder_from_directory(builder_dir=dataset2path(dataset_name))
    
    dset = dset.as_dataset(split='train[:6]', read_config=tfds.ReadConfig(add_tfds_id=True))
    dset_iter = iter(dset)
    last_episode_id = 0
    for ep_idx in [0, 1, 2, 3, 4]: # [0, 1]:
        if last_episode_id == 0:
            main(dset_iter, ep_idx + 1 - last_episode_id, ep_idx, False)
        else:
            main(dset_iter, ep_idx - last_episode_id, ep_idx, False)
        last_episode_id = ep_idx