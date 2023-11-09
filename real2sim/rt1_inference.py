import numpy as np
import os

import mani_skill2.envs, gymnasium as gym

from real2sim.rt1.rt1_model import RT1Inference
from real2sim.utils.visualization import write_video

def main():
    
    # Create environment
    env = gym.make('PickSingleYCBIntoBowl-v0',
                   control_mode='arm_pd_ee_target_delta_pose_base_gripper_finger_pd_joint_delta_pos',
                   obs_mode='rgbd',
                   robot='google_robot_static',
                   sim_freq=500,
                   control_freq=3,
                   max_episode_steps=50,
                   asset_root='/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/mani_skill2_ycb/',
                   scene_root='/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/hab2_bench_assets/')
    
    # Build RT-1 Model
    rt1_model = RT1Inference()
    
    for epoch_id in range(10):
        # Reset and initialize environment
        predicted_actions = []
        images = []
        
        obs, _ = env.reset()
        image = obs['image']['overhead_camera']['rgb']
        images.append(image)
        terminated, truncated = False, False
    
        obj_name = ' '.join(env.obj.name.split('_')[1:])
        task_description = f"Place the {obj_name} into the bowl."
        print(task_description)
    
        # Reset RT-1 model
        rt1_model.reset(task_description)
    
        timestep = 0
        # Step the environment
        while not (terminated or truncated):
            action = rt1_model.step(image)
            predicted_actions.append(action)
            print(timestep, action)
            obs, reward, terminated, truncated, info = env.step(np.concatenate([action['world_vector'], action['rotation_delta'], action['gripper_closedness_action']]))
            
            image = obs['image']['overhead_camera']['rgb']
            images.append(image)
            timestep += 1
            
        write_video(f'results/vis_{epoch_id}.mp4', images, fps=5)


if __name__ == '__main__':
    main()