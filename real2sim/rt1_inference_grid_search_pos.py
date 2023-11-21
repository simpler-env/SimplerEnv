import numpy as np
import os

import mani_skill2.envs, gymnasium as gym
from transforms3d.euler import euler2axangle

from real2sim.rt1.rt1_model import RT1Inference
from real2sim.utils.visualization import write_video

def main(env_name):
    action_repeat = 5 # render intermediate steps and possibly delay gripper execution
    robot_init_x_range = np.linspace(0.30, 0.50, 5)
    robot_init_y_range = np.linspace(0.0, 0.4, 5)
    obj_init_x_range = np.linspace(-0.4, -0.1, 7)
    obj_init_y_range = np.linspace(0.0, 0.4, 9)
    
    # Build RT-1 Model
    rt1_model = RT1Inference()
    
    for robot_init_x in robot_init_x_range:
        for robot_init_y in robot_init_y_range:
            for obj_init_x in obj_init_x_range:
                for obj_init_y in obj_init_y_range:
                    # Create environment
                    env = gym.make(env_name,
                                control_mode='arm_pd_ee_target_delta_pose_base_gripper_finger_pd_joint_delta_pos',
                                # control_mode='arm_pd_ee_target_delta_pose_base_gripper_finger_pd_joint_target_delta_pos',
                                obs_mode='rgbd',
                                robot='google_robot_static',
                                sim_freq=510,
                                control_freq=3 * action_repeat,
                                max_episode_steps=50 * action_repeat,
                                asset_root='/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/mani_skill2_ycb/',
                                scene_root='/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/hab2_bench_assets/',
                                obj_init_rand_rot_z_enabled=False,
                                obj_init_rand_rot_range=0,
                                obj_init_fixed_xy_pos=np.array([obj_init_x, obj_init_y]),
                                obj_init_fixed_z_rot=None,
                                robot_init_fixed_xy_pos=np.array([robot_init_x, robot_init_y]),
                    )
                    
                    # Reset and initialize environment
                    predicted_actions = []
                    images = []
                    
                    obs, _ = env.reset()
                    image = obs['image']['overhead_camera']['rgb']
                    images.append(image)
                    predicted_terminated, terminated, truncated = False, False, False
                
                    obj_name = ' '.join(env.obj.name.split('_')[1:])
                    if env_name == 'PickSingleYCBIntoBowl-v0':
                        task_description = f"place {obj_name} into red bowl"
                    elif env_name in ['GraspSingleYCBInScene-v0', 'GraspSingleYCBSomeInScene-v0']:
                        task_description = f"pick {obj_name}"
                    elif env_name in ['GraspSingleYCBFruitInScene-v0']:
                        task_description = "pick fruit"
                    elif env_name in ['GraspSingleYCBCanInScene-v0', 'GraspSingleYCBTomatoCanInScene-v0']:
                        task_description = "pick can"
                    elif env_name == 'GraspSingleYCBBoxInScene-v0':
                        task_description = "pick box"
                    elif env_name == 'KnockSingleYCBBoxOverInScene-v0':
                        task_description = "knock box over"
                    else:
                        raise NotImplementedError()
                    print(task_description)
                
                    # Reset RT-1 model
                    rt1_model.reset(task_description)
                
                    timestep = 0
                    success = "failure"
                    # Step the environment
                    while not (predicted_terminated or truncated):
                        if timestep % action_repeat == 0:
                            action = rt1_model.step(image)
                            predicted_actions.append(action)
                            print(timestep, action)
                            predicted_terminated = bool(action['terminate_episode'][0] > 0)
                            
                            # If action['rotation_delta'] is already in axis-angle:
                            # obs, reward, terminated, truncated, info = env.step(
                            #     np.concatenate(
                            #         [action['world_vector'], 
                            #         action['rotation_delta'], 
                            #         action['gripper_closedness_action']
                            #         ]
                            #     )
                            # )
                            
                            # If action['rotation_delta'] is in Euler rpy:
                            action_rotation_euler = action['rotation_delta']
                            action_rotation_ax, action_rotation_angle = euler2axangle(*action_rotation_euler, axes='sxyz')
                            action_rotation_axangle = action_rotation_ax * action_rotation_angle
                            obs, reward, terminated, truncated, info = env.step(
                                np.concatenate(
                                    [action['world_vector'], 
                                    action_rotation_axangle,
                                    action['gripper_closedness_action']
                                    ]
                                )
                            )
                            if terminated:
                                # For now, if at any step the episode is successful, we consider it a success
                                success = "success"
                        else:
                            obs, reward, terminated, truncated, info = env.step(
                                np.concatenate(
                                    [np.zeros(3), # same target as previous step
                                    np.zeros(3), # same target as previous step
                                    action['gripper_closedness_action']
                                    # np.zeros(1) if timestep % action_repeat < (action_repeat // 2) else action['gripper_closedness_action'],
                                    ]
                                )
                            )
                        # print(env.agent.robot.get_qvel())
                        image = obs['image']['overhead_camera']['rgb']
                        images.append(image)
                        timestep += 1
                        
                    write_video(f'results/{env_name}/rob_{robot_init_x}_{robot_init_y}/{success}_obj_{obj_init_x}_{obj_init_y}.mp4', images, fps=5)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['DISPLAY'] = ''
    main('GraspSingleYCBTomatoCanInScene-v0')