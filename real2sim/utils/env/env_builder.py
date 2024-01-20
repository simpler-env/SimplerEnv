import mani_skill2.envs, gymnasium as gym

def build_maniskill2_env(env_name, **kwargs):
    if kwargs.get('rgb_overlay_path', None) is not None:
        if kwargs.get('rgb_overlay_cameras', None) is None:
            if kwargs['robot'] == 'google_robot_static':
                kwargs['rgb_overlay_cameras'] = ['overhead_camera']
            elif kwargs['robot'] == 'widowx':
                kwargs['rgb_overlay_cameras'] = ['3rd_view_camera']
            else:
                raise NotImplementedError()
    env = gym.make(env_name, **kwargs)
    
    return env


def get_maniskill2_env_instruction(env, env_name, **kwargs):
    task_description = env.get_language_instruction()
    print(task_description)
    return task_description


def get_robot_control_mode(robot_name, policy_name):
    if robot_name == 'google_robot_static':
        control_mode = 'arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner'
    elif robot_name == 'widowx':
        if 'rt1' in policy_name:
            # control_mode = 'arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner'
            control_mode = 'arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos'
            # control_mode = 'arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_target_pos_interpolate_by_planner'
            # control_mode = 'arm_pd_ee_delta_pose_align2_gripper_pd_joint_pos'
        elif 'octo' in policy_name:
            # control_mode = 'arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_target_pos_interpolate_by_planner'
            control_mode = 'arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos'
            # control_mode = 'arm_pd_ee_delta_pose_align2_gripper_pd_joint_pos'
        else:
            raise NotImplementedError()
    return control_mode
