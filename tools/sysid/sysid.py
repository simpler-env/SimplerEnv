import numpy as np
import argparse
import pickle
import os
import multiprocessing as mp

from transforms3d.quaternions import quat2axangle, axangle2quat, quat2mat
from transforms3d.euler import euler2axangle
from simulated_annealing import sa

def calc_pose_err_single_ep(episode, arm_stiffness, arm_damping, robot, control_mode):
    from sapien.core import Pose
    import mani_skill2.envs, gymnasium as gym
    
    assert robot in ['google_robot_static', 'widowx']
    if robot == 'google_robot_static':
        # append dummy stiffness & damping for the camera links in google robot, which do not affect the results
        arm_stiffness = np.concatenate([arm_stiffness, [2000, 2000]])
        arm_damping = np.concatenate([arm_damping, [600, 600]])
    
    if robot == 'google_robot_static':
        sim_freq, control_freq = 252, 3
    elif robot == 'widowx':
        sim_freq, control_freq = 500, 5
    env = gym.make('PickCube-v0',
                    control_mode=control_mode,
                    obs_mode='rgbd',
                    robot=robot,
                    sim_freq=sim_freq,
                    control_freq=control_freq,
                    max_episode_steps=50,
    )
    env.agent.controller.controllers['arm'].config.stiffness = arm_stiffness
    env.agent.controller.controllers['arm'].config.damping = arm_damping
    env.agent.controller.controllers['arm'].set_drive_property()
        
    tcp_poses_at_base = []
    gt_tcp_poses_at_base = []
    
    _ = env.reset()
    env.agent.robot.set_pose(Pose([0, 0, 1]))

    def get_tcp_pose_at_robot_base():
        tcp_pose_at_robot_base = env.agent.robot.pose.inv() * env.tcp.pose
        return tcp_pose_at_robot_base
        
    for step_id, episode_step in enumerate(episode):
        gt_tcp_xyz_at_base = episode_step['base_pose_tool_reached'][:3]
        gt_tcp_wxyz_at_base = episode_step['base_pose_tool_reached'][3:]
        gt_tcp_pose_at_robot_base = Pose(p=np.array(gt_tcp_xyz_at_base), q=np.array(gt_tcp_wxyz_at_base))
        tcp_pose_at_robot_base = get_tcp_pose_at_robot_base()
                
        if step_id == 0:
            # move tcp pose in environment to the gt_tcp_pose_at_robot_base
            controller = env.agent.controller.controllers['arm']
            cur_qpos = env.agent.robot.get_qpos()
            init_arm_qpos = controller.compute_ik(gt_tcp_pose_at_robot_base)
            cur_qpos[controller.joint_indices] = init_arm_qpos
            env.agent.reset(cur_qpos)
            tcp_pose_at_robot_base = get_tcp_pose_at_robot_base()
                
        tcp_poses_at_base.append(tcp_pose_at_robot_base)
        gt_tcp_poses_at_base.append(gt_tcp_pose_at_robot_base)
            
        gt_action_world_vector = episode_step['action_world_vector']
        gt_action_rotation_delta = np.asarray(episode_step['action_rotation_delta'], dtype=np.float64)
        if robot == 'google_robot_static':
            gt_action_rotation_angle = np.linalg.norm(gt_action_rotation_delta)
            gt_action_rotation_ax = gt_action_rotation_delta / gt_action_rotation_angle if gt_action_rotation_angle > 1e-6 else np.array([0., 1., 0.])
            gt_action_rotation_axangle = gt_action_rotation_ax * gt_action_rotation_angle
        elif robot == 'widowx':
            gt_action_rotation_ax, gt_action_rotation_angle = euler2axangle(*gt_action_rotation_delta)
            gt_action_rotation_axangle = gt_action_rotation_ax * gt_action_rotation_angle
        
        action = np.concatenate(
            [gt_action_world_vector, 
            gt_action_rotation_axangle,
            np.array([0]),
            ],
        ).astype(np.float64)
        
        _ = env.step(action)
            
    # calculate trajectory error
    this_traj_err = []
    for (tcp_pose_at_base, gt_tcp_pose_at_base) in zip(tcp_poses_at_base, gt_tcp_poses_at_base):
        err = 2 * np.linalg.norm(tcp_pose_at_base.p - gt_tcp_pose_at_base.p)
        R_pred = quat2mat(tcp_pose_at_base.q)
        R_gt = quat2mat(gt_tcp_pose_at_base.q)
        err = err + 2 * np.arcsin(np.clip(1 / (2 * np.sqrt(2)) * np.sqrt(np.trace((R_pred - R_gt).T @ (R_pred - R_gt))), 0.0, 1.0))
        this_traj_err.append(err)
        
    if np.mean(this_traj_err) > 0.25:
        for (tcp_pose_at_base, gt_tcp_pose_at_base) in zip(tcp_poses_at_base, gt_tcp_poses_at_base):
            print(tcp_pose_at_base, gt_tcp_pose_at_base)
        print("*" * 10)
        print(this_traj_err)
        print("-" * 20)
        
    if this_traj_err[0] > 0.02:
        print("WARNING: The robot is not initialized to have the same pose as the first step of the episode. Error is: ", this_traj_err[0])
    
    return np.mean(this_traj_err)


def calc_pose_err(dset, arm_stiffness, arm_damping, robot, control_mode, log_path):
    
    errs = []
    processes = []
    
    pool = mp.Pool(min(len(dset), 18))
    for episode in dset:
        processes.append(
            pool.apply_async(
                calc_pose_err_single_ep, 
                args=(episode, arm_stiffness, arm_damping, robot, control_mode)
            )
        )
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
    """
    python tools/sysid/sysid.py --dataset-path /home/xuanlin/Downloads/sysid_dataset.pkl \
        --log-path /home/xuanlin/Downloads/opt_results.txt --robot google_robot_static
    python tools/sysid/sysid.py --dataset-path /home/xuanlin/Downloads/sysid_dataset_bridge.pkl \
        --log-path /home/xuanlin/Downloads/opt_results_bridge.txt --robot widowx
    """
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['DISPLAY'] = ''
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='/home/xuanlin/Downloads/sysid_dataset.pkl')
    parser.add_argument('--log-path', type=str, default='/home/xuanlin/Downloads/opt_results.txt')
    parser.add_argument('--robot', type=str, default='google_robot_static')
    args = parser.parse_args()
    
    with open(args.dataset_path, 'rb') as f:
        dset = pickle.load(f)
    
    if args.robot == 'google_robot_static':
        control_mode = 'arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_pos'
        # control_mode = 'arm_pd_ee_delta_pose_align_interpolate_gripper_pd_joint_pos'
        # control_mode = 'arm_pd_ee_delta_pose_align_gripper_pd_joint_pos'
        # control_mode = 'arm_pd_ee_target_delta_pose_align_interpolate_by_planner_gripper_pd_joint_pos'
                
        # stiffness_high = np.array([1600, 1950, 1680, 1500, 730, 860, 680])
        # stiffness_low = np.array([1400, 1750, 1430, 1250, 530, 630, 480])
        # damping_high = np.array([630, 680, 610, 560, 380, 240, 220])
        # damping_low = np.array([380, 400, 380, 360, 210, 120, 90])
        # init_stiffness = np.array([1542.4844516168355, 1906.9938992819923, 1605.8611345378665, 1400.0, 630.0, 730.0, 583.6446104792196])
        # init_damping = np.array([513.436152107585, 574.0051814405743, 505.6134557131383, 458.36436883104705, 293.9497910839597, 186.7912085424362, 158.8619324972991])
        
        # stiffness_high = np.array([1800, 1750, 1050, 950, 1230, 510, 540])
        # stiffness_low = np.array([1700, 1650, 950, 850, 1130, 430, 480])
        # damping_high = np.array([1180, 1180, 880, 660, 830, 330, 400])
        # damping_low = np.array([900, 900, 680, 430, 660, 200, 240])
        # # init_stiffness = np.array([1932.2991678390808, 1826.4991049680966, 1172.273714250636, 882.4814756272485, 1397.5148682131537, 699.3489562744397, 660.0])
        # # init_damping = np.array([884.6002482794984, 1000.0, 631.0539484239861, 509.9225285931856, 753.8467217080913, 329.60720242099455, 441.4206687847951])
        # init_stiffness = np.array([1755.5802337759733, 1700.0, 1000.0, 896.1427073141074, 1181.0596023097614, 460.0, 518.7478307141772])
        # init_damping = np.array([1039.3004397057607, 997.7609238661106, 781.9120300040199, 533.1406757667885, 763.5690552485103, 247.37299930493683, 330.0])
        
        # stiffness_high = np.array([1000, 1250, 1050, 950, 450, 480, 230])
        # stiffness_low = np.array([800, 1100, 850, 850, 330, 380, 170])
        # damping_high = np.array([800, 960, 760, 660, 460, 280, 220])
        # damping_low = np.array([600, 780, 600, 520, 360, 170, 150])
        # init_stiffness = [900.0, 1200.0, 956.2135773317326, 924.9766943646305, 400.0, 430.0, 200.0]
        # init_damping = [743.4368714060297, 900.0, 700.0, 607.9657371455355, 432.91780633579725, 237.57304830746313, 205.07035572526215]
        
        stiffness_high = np.array([1800, 1750, 1050, 960, 1230, 450, 480])
        stiffness_low = np.array([1700, 1650, 950, 930, 1180, 410, 450])
        damping_high = np.array([1110, 1100, 800, 750, 730, 310, 360])
        damping_low = np.array([980, 900, 700, 610, 610, 220, 250])
        # stiffness_high = np.array([1900, 1850, 1150, 1000, 1280, 500, 530])
        # stiffness_low = np.array([1500, 1550, 850, 800, 1030, 380, 380])
        # damping_high = np.array([1200, 1200, 850, 780, 780, 350, 380])
        # damping_low = np.array([830, 830, 630, 550, 500, 190, 230])
        init_stiffness = np.array([1700.0, 1737.0471680861954, 979.975871856535, 930.0, 1212.154500274304, 432.96500923932535, 468.0013365498738])
        init_damping = np.array([1059.9791902443303, 1010.4720585373592, 767.2803161582076, 680.0, 674.9946964336588, 274.613381336198, 340.532560578637])
        
        # stiffness_high = np.array([1800, 1800, 1050, 950, 1300, 630, 550])
        # stiffness_low = np.array([1700, 1700, 950, 850, 1200, 580, 500])
        # damping_high = np.array([1300, 1300, 760, 660, 650, 350, 530])
        # damping_low = np.array([300, 300, 230, 230, 160, 100, 180])
        # init_stiffness = np.array([1735.8948480824674, 1754.3342187522323, 1007.9762036720238, 872.5638913272953, 1277.700676022463, 608.0856938168192, 530.0])
        # init_damping = np.array([1000.0, 1042.8696312830125, 606.8732757029185, 552.2718719738202, 528.0029778895791, 275.6999553621622, 530.0])
        
        # stiffness_high = np.array([2000, 2200, 1700, 1400, 860, 730, 730])
        # stiffness_low = np.array([1200, 1400, 1300, 1000, 630, 530, 530])
        # damping_high = np.array([400, 200, 150, 100, 100, 100, 100])
        # damping_low = np.array([40, 40, 30, 20, 20, 20, 20])
        # init_stiffness = np.array([1522.6925826441493, 2158.4544756749015, 1400.1676094551071, 1142.6986700565294, 730.659637818336, 669.7021044436542, 628.821295716587])
        # init_damping = np.array([293.9747942850573, 103.83092695838668, 85.29843663304095, 40.0, 30.0, 30.0, 74.07034288138254])
        
        # stiffness_high = np.array([1600, 2200, 1430, 1200, 780, 700, 680])
        # stiffness_low = np.array([1500, 2100, 1360, 1100, 680, 600, 580])
        # damping_high = np.array([90, 90, 80, 80, 60, 60, 60])
        # damping_low = np.array([30, 30, 20, 20, 15, 15, 15])
        # init_stiffness = np.array([1522.6925826441493, 2158.4544756749015, 1400.1676094551071, 1142.6986700565294, 730.659637818336, 669.7021044436542, 628.821295716587])
        # init_damping = np.array([30, 30, 20, 20, 15, 15, 15])
        
    elif args.robot == 'widowx':
        control_mode = 'arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos'
        stiffness_high = np.array([1200, 780, 860, 1230, 1430, 1080])
        stiffness_low = np.array([1110, 680, 730, 1010, 1180, 930])
        # damping_high = np.array([100, 50, 50, 50, 120, 120])
        # damping_high = np.array([300, 200, 200, 200, 250, 250])
        # damping_high = np.array([430, 310, 310, 310, 400, 400])
        # damping_high = np.array([330, 180, 180, 310, 280, 360])
        damping_high = np.array([500, 350, 210, 400, 260, 330])
        # damping_high = np.array([260, 130, 100, 140, 140, 170])
        # damping_low = np.array([30, 20, 20, 20, 30, 30])
        # damping_low = np.array([15, 8, 8, 8, 10, 10])
        # damping_low = np.array([100, 40, 40, 70, 70, 90])
        # damping_low = np.array([150, 90, 86, 130, 110, 150])
        damping_low = np.array([250, 150, 100, 240, 150, 200])
        # init_stiffness = [1193.2765654645982, 800.0, 784.3309604605763, 1250.3737197881153, 1392.0546244178072, 1038.3774360126893]
        # init_damping = [75.5250991585983, 20.0, 23.646570105574618, 23.825760721440837, 67.97737990215525, 78.14407359073823]
        # init_stiffness = [1214.6340906847158, 804.5146660467828, 801.9841311029891, 1110.0, 1310.0, 988.4499396558518]
        # init_damping = [175.18652498291488, 73.04563998424553, 62.47429885911165, 104.4069151351231, 108.3230540691408, 136.87526713617873]
        # init_stiffness = [1215.4327150032293, 730.0, 860.0, 1133.9675494142102, 1413.3815895525422, 930.0]
        # init_damping = [285.5831564748846, 118.83365148810542, 126.05256283235089, 142.0533158479584, 142.85223328752122, 96.00503592486184]
        init_stiffness = np.array([1169.7891719504198, 730.0, 808.4601346394447, 1229.1299089624076, 1272.2760456418862, 1056.3326605132252])
        init_damping = np.array([330.0, 180.0, 152.12036565582588, 309.6215302722146, 201.04998711007383, 269.51458932695414])
    else:
        raise NotImplementedError()
          
    raw_action_to_stiffness = lambda x: stiffness_low + (stiffness_high - stiffness_low) * x[: len(stiffness_high)]
    raw_action_to_damping = lambda x: damping_low + (damping_high - damping_low) * x[len(stiffness_high) : 2 * len(stiffness_high)]
        
    init_action = np.concatenate(
        [(init_stiffness - stiffness_low) / (stiffness_high - stiffness_low), 
         (init_damping - damping_low) / (damping_high - damping_low)]
    )
    
    opt_fxn = lambda x: calc_pose_err(
        dset, raw_action_to_stiffness(x), raw_action_to_damping(x), 
        args.robot, control_mode, log_path=args.log_path
    )
    opt = sa.minimize(opt_fxn, init_action, opt_mode='continuous', step_max=2000, t_max=1.5, t_min=0,
                      bounds=[[0, 1]] * (len(init_stiffness) * 2))
    
    
    
"""
# log_stiffness_high = np.log(stiffness_high)
# log_stiffness_low = np.log(stiffness_low)
# log_init_stiffness = np.log(init_stiffness)
# log_damping_high = np.log(damping_high)
# log_damping_low = np.log(damping_low)
# log_init_damping = np.log(init_damping)
# raw_action_to_stiffness = lambda x: np.exp(log_stiffness_low + (log_stiffness_high - log_stiffness_low) * x[: len(x) // 2])
# raw_action_to_damping = lambda x: np.exp(log_damping_low + (log_damping_high - log_damping_low) * x[len(x) // 2 :])
    
# init_action = np.concatenate(
#     [(log_init_stiffness - log_stiffness_low) / (log_stiffness_high - log_stiffness_low), 
#      (log_init_damping - log_damping_low) / (log_damping_high - log_damping_low)]
# )
"""