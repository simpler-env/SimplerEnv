import numpy as np
import argparse
import pickle
import os
import multiprocessing as mp

from transforms3d.quaternions import quat2axangle, axangle2quat, quat2mat
from simulated_annealing import sa

def calc_pose_err_single_ep(episode, arm_stiffness, arm_damping):
    from sapien.core import Pose
    import mani_skill2.envs, gymnasium as gym
    
    # append dummy stiffness & damping for the camera links in google robot, which do not affect the results
    arm_stiffness = np.concatenate([arm_stiffness, [2000, 2000]])
    arm_damping = np.concatenate([arm_damping, [600, 600]])
    
    sim_freq, control_freq, action_repeat = 510, 3, 5
    env = gym.make('PickCube-v0',
                    control_mode='arm_pd_ee_delta_pose_align_interpolate_gripper_pd_joint_pos',
                    obs_mode='rgbd',
                    robot='google_robot_static',
                    sim_freq=sim_freq,
                    control_freq=control_freq * action_repeat,
                    max_episode_steps=50 * action_repeat,
    )
    env.agent.controller.controllers['arm'].config.stiffness = arm_stiffness
    env.agent.controller.controllers['arm'].config.damping = arm_damping
    env.agent.controller.controllers['arm'].set_drive_property()
        
    tcp_poses_at_base = []
    gt_tcp_poses_at_base = []
    
    _ = env.reset()

    for step_id, episode_step in enumerate(episode):
        gt_tcp_xyz_at_base = episode_step['base_pose_tool_reached'][:3]
        gt_tcp_xyzw_at_base = episode_step['base_pose_tool_reached'][3:]
        gt_tcp_pose_at_robot_base = Pose(p=np.array(gt_tcp_xyz_at_base), q=np.concatenate([gt_tcp_xyzw_at_base[-1:], gt_tcp_xyzw_at_base[:-1]]))
        tcp_pose_at_robot_base = env.agent.robot.pose.inv() * env.tcp.pose
        
        if step_id == 0:
            # move tcp pose in environment to the gt tcp pose wrt robot base
            for _ in range(4 * action_repeat):
                delta_tcp_pose = Pose(
                    p=gt_tcp_pose_at_robot_base.p - tcp_pose_at_robot_base.p,
                    q=(gt_tcp_pose_at_robot_base * tcp_pose_at_robot_base.inv()).q,
                )
                action_translation = delta_tcp_pose.p
                action_rot_ax, action_rot_angle = quat2axangle(np.array(delta_tcp_pose.q, dtype=np.float64))
                action_rotation = action_rot_ax * action_rot_angle
                action = np.concatenate([action_translation, action_rotation, [0]])
                env.step(action)
                tcp_pose_at_robot_base = env.agent.robot.pose.inv() * env.tcp.pose
                
        tcp_poses_at_base.append(tcp_pose_at_robot_base)
        gt_tcp_poses_at_base.append(gt_tcp_pose_at_robot_base)
            
        gt_action_world_vector = episode_step['action_world_vector']
        gt_action_rotation_delta = np.asarray(episode_step['action_rotation_delta'], dtype=np.float64)
        gt_action_rotation_angle = np.linalg.norm(gt_action_rotation_delta)
        gt_action_rotation_ax = gt_action_rotation_delta / gt_action_rotation_angle if gt_action_rotation_angle > 1e-6 else np.array([0., 1., 0.])
        gt_action_rotation_axangle = gt_action_rotation_ax * gt_action_rotation_angle
        target_tcp_pose_at_base = Pose(p=tcp_pose_at_robot_base.p + gt_action_world_vector,
                                        q=(Pose(q=axangle2quat(gt_action_rotation_ax, gt_action_rotation_angle)) 
                                            * Pose(q=tcp_pose_at_robot_base.q)).q
        )
        action = np.concatenate(
            [gt_action_world_vector, 
            gt_action_rotation_axangle,
            np.array([0]),
            ],
        ).astype(np.float64)
        
        _ = env.step(action)
        for _ in range(action_repeat - 1):
            interp_action = action.copy()
            cur_tcp_pose_at_base = env.agent.robot.pose.inv() * env.tcp.pose
            delta_tcp_pose_at_base = target_tcp_pose_at_base * cur_tcp_pose_at_base.inv()
            interp_action[:3] = target_tcp_pose_at_base.p - cur_tcp_pose_at_base.p
            interp_rot_ax, interp_rot_angle = quat2axangle(np.array(delta_tcp_pose_at_base.q, dtype=np.float64))
            interp_action[3:6] = interp_rot_ax * interp_rot_angle
            _ = env.step(interp_action)
            
    # calculate trajectory error
    this_traj_err = []
    for (tcp_pose_at_base, gt_tcp_pose_at_base) in zip(tcp_poses_at_base, gt_tcp_poses_at_base):
        err = 2 * np.linalg.norm(tcp_pose_at_base.p - gt_tcp_pose_at_base.p)
        R_pred = quat2mat(tcp_pose_at_base.q)
        R_gt = quat2mat(gt_tcp_pose_at_base.q)
        err = err + 2 * np.arcsin(np.clip(1 / (2 * np.sqrt(2)) * np.sqrt(np.trace((R_pred - R_gt).T @ (R_pred - R_gt))), 0.0, 1.0))
        this_traj_err.append(err)
        
    return np.mean(this_traj_err)


def calc_pose_err(dset, arm_stiffness, arm_damping, log_path):
    
    errs = []
    processes = []
    
    pool = mp.Pool(16)
    for episode in dset:
        processes.append(pool.apply_async(calc_pose_err_single_ep, args=(episode, arm_stiffness, arm_damping)))
    pool.close()
    for process in processes:
        errs.append(process.get())
    pool.join()
            
    avg_err = np.mean(errs)
    print_info = f'arm_stiffness: {list(arm_stiffness)}, arm_damping: {list(arm_damping)}, avg_err: {avg_err}, per_traj_err: {errs}'
    with open(log_path, 'a') as f:
        print(print_info, file=f)
    print(print_info)
    
    return avg_err
            
    

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['DISPLAY'] = ''
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='/home/xuanlin/Downloads/sysid_dataset.pkl')
    parser.add_argument('--log-path', type=str, default='/home/xuanlin/Downloads/opt_results.txt')
    args = parser.parse_args()
    
    with open(args.dataset_path, 'rb') as f:
        dset = pickle.load(f)
    
    stiffness_high = np.array([2500, 2500, 1500, 1500, 1000, 800, 800])
    stiffness_low = np.array([1000, 1000, 500, 500, 300, 200, 200])
    damping_high = np.array([1000, 1000, 800, 800, 600, 400, 400])
    damping_low = np.array([400, 400, 300, 300, 200, 100, 100])
    # stiffness_high = np.array([2500, 2500, 2000, 2000, 2000, 1000, 1000])
    # stiffness_low = np.array([400, 400, 400, 400, 400, 200, 200])
    # damping_high = np.array([1200, 1200, 1200, 1200, 1200, 700, 700])
    # damping_low = np.array([200, 200, 200, 200, 200, 100, 100])
    
    raw_action_to_stiffness = lambda x: stiffness_low + (stiffness_high - stiffness_low) * x[:7]
    raw_action_to_damping = lambda x: damping_low + (damping_high - damping_low) * x[7:]
    init_stiffness = np.array([2000, 1800, 1200, 1000, 650, 500, 500])
    init_damping = np.array([850, 810, 500, 480, 460, 190, 250])
    init_action = np.concatenate(
        [(init_stiffness - stiffness_low) / (stiffness_high - stiffness_low), 
         (init_damping - damping_low) / (damping_high - damping_low)]
    )
    
    opt_fxn = lambda x: calc_pose_err(dset, raw_action_to_stiffness(x), raw_action_to_damping(x), log_path=args.log_path)
    opt = sa.minimize(opt_fxn, init_action, opt_mode='continuous', step_max=1000, t_max=1, t_min=0,
                      bounds=[[0, 1]] * 14)