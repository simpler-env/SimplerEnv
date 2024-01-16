import mani_skill2.envs, gymnasium as gym
import numpy as np
import os
import time
from pathlib import Path
from matplotlib import pyplot as plt
from transforms3d.euler import euler2quat
from sapien.core import Pose

def main():
    env_name = 'GraspSingleCokeCanInScene-v0'
    if 'YCB' in env_name:
        asset_root='/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/mani_skill2_ycb/'
    else:
        asset_root='/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/custom/'
    scene_name = 'Baked_sc1_staging_table_616385'
    # scene_name = 'Baked_sc1_staging_objaverse_cabinet1'
    robot_init_x = 0.32 # hardcoded
    robot_init_y = 0.188 # hardcoded
    obj_init_x = -0.25
    obj_init_y = 0.20
    
    env = gym.make(env_name,
                control_mode='arm_pd_ee_target_delta_pose_base_gripper_pd_joint_target_delta_pos',
                obs_mode='rgbd',
                robot='google_robot_static',
                sim_freq=510,
                control_freq=3,
                max_episode_steps=50,
                asset_root=asset_root,
                scene_root='/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/hab2_bench_assets/',
                scene_name=scene_name,
                obj_init_rand_rot_z_enabled=False,
                obj_init_rand_rot_range=0,
                obj_init_fixed_xy_pos=np.array([obj_init_x, obj_init_y]),
                obj_init_fixed_z_rot=None,
                robot_init_fixed_xy_pos=np.array([robot_init_x, robot_init_y]),
                camera_cfgs={"add_segmentation": True},
                rgb_overlay_path='/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/real_impainting/google_table_top_1.png',
                rgb_overlay_cameras=['overhead_camera'],
                robot_init_fixed_rot_quat=(Pose(q=[0, 0, 0, 1]) * Pose(q=euler2quat(0, 0, -0.01))).q,
                # shader_dir="rt",
                # render_config={"rt_samples_per_pixel": 8, "rt_use_denoiser": True},
    )
    
    obs, _ = env.reset()
    # env.agent.robot.set_pose(env.agent.robot.pose * Pose([0, 0, 0], euler2quat(0, 0, -0.01)))
    # obs, *_ = env.step(env.action_space.sample())
    plt.figure()
    plt.imshow(obs['image']['overhead_camera']['rgb'])
    plt.savefig('test.png')
    plt.show()
    plt.close()
    
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['DISPLAY'] = ''
    main()