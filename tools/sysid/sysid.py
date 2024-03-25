"""
Use Simulated annealing to optimize the stiffness and damping of the robot arm and minimize 6d pose errors when open-loop unrolling demonstration trajectories.
"""

import argparse
import multiprocessing as mp
import os
import pickle

import numpy as np
from simulated_annealing import sa
from transforms3d.euler import euler2axangle
from transforms3d.quaternions import axangle2quat, quat2axangle, quat2mat


def calc_pose_err_single_ep(episode, arm_stiffness, arm_damping, robot, control_mode):
    import gymnasium as gym
    import mani_skill2_real2sim.envs
    from sapien.core import Pose

    assert robot in ["google_robot_static", "widowx"]
    if robot == "google_robot_static":
        # append dummy stiffness & damping for the camera links in google robot, which do not affect the results
        arm_stiffness = np.concatenate([arm_stiffness, [2000, 2000]])
        arm_damping = np.concatenate([arm_damping, [600, 600]])

    if robot == "google_robot_static":
        sim_freq, control_freq = 252, 3
    elif robot == "widowx":
        sim_freq, control_freq = 500, 5
    env = gym.make(
        "GraspSingleDummy-v0",
        control_mode=control_mode,
        obs_mode="rgbd",
        robot=robot,
        sim_freq=sim_freq,
        control_freq=control_freq,
        max_episode_steps=50,
    )
    # set arm stiffness and damping
    env.agent.controller.controllers["arm"].config.stiffness = arm_stiffness
    env.agent.controller.controllers["arm"].config.damping = arm_damping
    env.agent.controller.controllers["arm"].set_drive_property()

    tcp_poses_at_base = []
    gt_tcp_poses_at_base = []

    _ = env.reset()
    env.agent.robot.set_pose(Pose([0, 0, 1]))

    def get_tcp_pose_at_robot_base():
        tcp_pose_at_robot_base = env.agent.robot.pose.inv() * env.tcp.pose
        return tcp_pose_at_robot_base

    # unroll the episode and record the tcp (end-effector tool-center-point) poses
    for step_id, episode_step in enumerate(episode):
        gt_tcp_xyz_at_base = episode_step["base_pose_tool_reached"][:3]
        gt_tcp_wxyz_at_base = episode_step["base_pose_tool_reached"][3:]
        gt_tcp_pose_at_robot_base = Pose(p=np.array(gt_tcp_xyz_at_base), q=np.array(gt_tcp_wxyz_at_base))
        tcp_pose_at_robot_base = get_tcp_pose_at_robot_base()

        if step_id == 0:
            # At the beginning of episode, set the end-effector pose to be the same as the first observation
            controller = env.agent.controller.controllers["arm"]
            cur_qpos = env.agent.robot.get_qpos()
            init_arm_qpos = controller.compute_ik(gt_tcp_pose_at_robot_base)
            cur_qpos[controller.joint_indices] = init_arm_qpos
            env.agent.reset(cur_qpos)
            tcp_pose_at_robot_base = get_tcp_pose_at_robot_base()

        tcp_poses_at_base.append(tcp_pose_at_robot_base)
        gt_tcp_poses_at_base.append(gt_tcp_pose_at_robot_base)

        gt_action_world_vector = episode_step["action_world_vector"]
        gt_action_rotation_delta = np.asarray(episode_step["action_rotation_delta"], dtype=np.float64)
        if robot == "google_robot_static":
            # the recorded demonstration actions are in the form of axis-angle representation
            gt_action_rotation_angle = np.linalg.norm(gt_action_rotation_delta)
            gt_action_rotation_ax = (
                gt_action_rotation_delta / gt_action_rotation_angle
                if gt_action_rotation_angle > 1e-6
                else np.array([0.0, 1.0, 0.0])
            )
            gt_action_rotation_axangle = gt_action_rotation_ax * gt_action_rotation_angle
        elif robot == "widowx":
            # the recorded demonstration actions are in the form of raw, pitch, yaw euler angles
            gt_action_rotation_ax, gt_action_rotation_angle = euler2axangle(*gt_action_rotation_delta)
            gt_action_rotation_axangle = gt_action_rotation_ax * gt_action_rotation_angle

        action = np.concatenate(
            [
                gt_action_world_vector,
                gt_action_rotation_axangle,
                np.array([0]),
            ],
        ).astype(np.float64)

        _ = env.step(action)

    # calculate trajectory error
    this_traj_err = []
    this_traj_raw_transl_err = []
    this_traj_raw_rot_err = []
    for (tcp_pose_at_base, gt_tcp_pose_at_base) in zip(tcp_poses_at_base, gt_tcp_poses_at_base):
        raw_transl_err = np.linalg.norm(tcp_pose_at_base.p - gt_tcp_pose_at_base.p)
        err = raw_transl_err
        this_traj_raw_transl_err.append(raw_transl_err)

        R_pred = quat2mat(tcp_pose_at_base.q)
        R_gt = quat2mat(gt_tcp_pose_at_base.q)
        raw_rot_err = np.arcsin(
            np.clip(
                1 / (2 * np.sqrt(2)) * np.sqrt(np.trace((R_pred - R_gt).T @ (R_pred - R_gt))),
                0.0,
                1.0,
            )
        )
        err = err + raw_rot_err
        this_traj_raw_rot_err.append(raw_rot_err)

        this_traj_err.append(err)

    if np.mean(this_traj_err) > 0.15:
        for (tcp_pose_at_base, gt_tcp_pose_at_base) in zip(tcp_poses_at_base, gt_tcp_poses_at_base):
            print(tcp_pose_at_base, gt_tcp_pose_at_base)
        print("*" * 10)
        print(this_traj_err)
        print("-" * 20)

    if this_traj_err[0] > 0.02:
        print(
            "WARNING: The robot is not initialized to have the same pose as the first step of the episode. Error is: ",
            this_traj_err[0],
        )

    return (
        np.mean(this_traj_err),
        np.mean(this_traj_raw_transl_err),
        np.mean(this_traj_raw_rot_err),
    )


def calc_pose_err(dset, arm_stiffness, arm_damping, robot, control_mode, log_path):

    errs = []
    raw_transl_errs = []
    raw_rot_errs = []
    processes = []

    # calculate the pose error for each episode in the dataset in parallel
    pool = mp.Pool(min(len(dset), 18))
    for episode in dset:
        processes.append(
            pool.apply_async(
                calc_pose_err_single_ep,
                args=(episode, arm_stiffness, arm_damping, robot, control_mode),
            )
        )
    pool.close()
    for process in processes:
        result = process.get()
        errs.append(result[0])
        raw_transl_errs.append(result[1])
        raw_rot_errs.append(result[2])
    pool.join()

    avg_err = np.mean(errs)
    avg_raw_transl_err = np.mean(raw_transl_errs)
    avg_raw_rot_err = np.mean(raw_rot_errs)

    # log the results
    print_info = f"arm_stiffness: {list(arm_stiffness)}, arm_damping: {list(arm_damping)}, avg_raw_transl_err: {avg_raw_transl_err}, avg_raw_rot_err: {avg_raw_rot_err}, avg_err: {avg_err}, per_traj_err: {errs}"
    with open(log_path, "a") as f:
        print(print_info, file=f)
    print(print_info)

    return avg_err


if __name__ == "__main__":
    """
    python tools/sysid/sysid.py --dataset-path /home/xuanlin/Downloads/sysid_dataset.pkl \
        --log-path /home/xuanlin/Downloads/opt_results.txt --robot google_robot_static
    python tools/sysid/sysid.py --dataset-path /home/xuanlin/Downloads/sysid_dataset_bridge.pkl \
        --log-path /home/xuanlin/Downloads/opt_results_bridge.txt --robot widowx
    """

    os.environ["DISPLAY"] = ""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="sysid_log/sysid_dataset.pkl")
    parser.add_argument("--log-path", type=str, default="sysid_log/opt_results_google_robot.txt")
    parser.add_argument("--robot", type=str, default="google_robot_static")
    args = parser.parse_args()

    with open(args.dataset_path, "rb") as f:
        dset = pickle.load(f)

    if args.robot == "google_robot_static":
        control_mode = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_pos"

        # these are just examples of the stiffness / damping ranges and initial values;

        stiffness_high = np.array([2000, 2000, 1300, 1300, 1300, 600, 600])
        stiffness_low = np.array([1400, 1400, 900, 900, 900, 380, 380])
        damping_high = np.array([1200, 1200, 850, 850, 850, 350, 360])
        damping_low = np.array([700, 700, 580, 580, 580, 200, 200])
        init_stiffness = np.array([1700, 1700, 1100, 1000, 1100, 500, 500])
        init_damping = np.array([950, 950, 700, 700, 700, 300, 300])

        # stiffness_high = np.array([1800, 1750, 1050, 960, 1230, 450, 480])
        # stiffness_low = np.array([1700, 1650, 950, 930, 1180, 410, 450])
        # damping_high = np.array([1110, 1100, 800, 750, 730, 310, 360])
        # damping_low = np.array([980, 900, 700, 610, 610, 220, 250])
        # # stiffness_high = np.array([1900, 1850, 1150, 1000, 1280, 500, 530])
        # # stiffness_low = np.array([1500, 1550, 850, 800, 1030, 380, 380])
        # # damping_high = np.array([1200, 1200, 850, 780, 780, 350, 360])
        # # damping_low = np.array([830, 830, 630, 550, 500, 190, 230])
        # init_stiffness = np.array([1700.0, 1737.0471680861954, 979.975871856535, 930.0, 1212.154500274304, 432.96500923932535, 468.0013365498738])
        # init_damping = np.array([1059.9791902443303, 1010.4720585373592, 767.2803161582076, 680.0, 674.9946964336588, 274.613381336198, 340.532560578637])

    elif args.robot == "widowx":
        control_mode = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
        stiffness_high = np.array([1400, 1400, 1400, 1400, 1400, 1400])
        stiffness_low = np.array([600, 600, 600, 600, 600, 600])
        damping_high = np.array([500, 400, 400, 400, 400, 400])
        damping_low = np.array([150, 130, 100, 100, 100, 100])
        init_stiffness = np.array([1000, 1000, 1000, 1000, 1000, 1000])
        init_damping = np.array([300, 250, 200, 200, 200, 200])

        # stiffness_high = np.array([1200, 780, 860, 1230, 1430, 1080])
        # stiffness_low = np.array([1110, 680, 730, 1010, 1180, 930])
        # damping_high = np.array([500, 350, 210, 400, 260, 330])
        # damping_low = np.array([250, 150, 100, 240, 150, 200])
        # init_stiffness = np.array([1169.7891719504198, 730.0, 808.4601346394447, 1229.1299089624076, 1272.2760456418862, 1056.3326605132252])
        # init_damping = np.array([330.0, 180.0, 152.12036565582588, 309.6215302722146, 201.04998711007383, 269.51458932695414])
    else:
        raise NotImplementedError()

    raw_action_to_stiffness = lambda x: stiffness_low + (stiffness_high - stiffness_low) * x[: len(stiffness_high)]
    raw_action_to_damping = (
        lambda x: damping_low + (damping_high - damping_low) * x[len(stiffness_high) : 2 * len(stiffness_high)]
    )

    init_action = np.concatenate(
        [
            (init_stiffness - stiffness_low) / (stiffness_high - stiffness_low),
            (init_damping - damping_low) / (damping_high - damping_low),
        ]
    )

    opt_fxn = lambda x: calc_pose_err(
        dset,
        raw_action_to_stiffness(x),
        raw_action_to_damping(x),
        args.robot,
        control_mode,
        log_path=args.log_path,
    )
    opt = sa.minimize(
        opt_fxn,
        init_action,
        opt_mode="continuous",
        step_max=2000,
        t_max=1.5,
        t_min=0,
        bounds=[[0, 1]] * (len(init_stiffness) * 2),
    )


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
