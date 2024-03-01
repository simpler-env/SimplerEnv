import mani_skill2.envs, gymnasium as gym

def build_prepackaged_maniskill2_env(task_name):
    # maps prepackaged (alias) task name to maniskill2 environment name and configuration kwargs
    task_map_dict =  {
        'google_robot_pick_coke_can': ('GraspSingleOpenedCokeCanInScene-v0', {}),
        'google_robot_pick_horizontal_coke_can': ('GraspSingleOpenedCokeCanInScene-v0', {'lr_switch': True}),
        'google_robot_pick_vertical_coke_can': ('GraspSingleOpenedCokeCanInScene-v0', {'laid_vertically': True}),
        'google_robot_pick_standing_coke_can': ('GraspSingleOpenedCokeCanInScene-v0', {'upright': True}),
        'google_robot_move_near': ('MoveNearGoogleBakedTexInScene-v0', {}),
        'google_robot_open_drawer': ('OpenDrawerCustomInScene-v0', {}),
        'google_robot_open_top_drawer': ('OpenTopDrawerCustomInScene-v0', {}),
        'google_robot_open_middle_drawer': ('OpenMiddleDrawerCustomInScene-v0', {}),
        'google_robot_open_bottom_drawer': ('OpenBottomDrawerCustomInScene-v0', {}),
        'google_robot_close_drawer': ('CloseDrawerCustomInScene-v0', {}),
        'google_robot_close_top_drawer': ('CloseTopDrawerCustomInScene-v0', {}),
        'google_robot_close_middle_drawer': ('CloseMiddleDrawerCustomInScene-v0', {}),
        'google_robot_close_bottom_drawer': ('CloseBottomDrawerCustomInScene-v0', {}),
        'widowx_spoon_on_towel': ('PutSpoonOnTableClothInScene-v0', {}),
        'widowx_carrot_on_plate': ('PutCarrotOnPlateInScene-v0', {}),
        'widowx_stack_cube': ('StackGreenCubeOnYellowCubeBakedTexInScene-v0', {}),
    }
    env_name, kwargs = task_map_dict[task_name]
    kwargs['prepackaged_config'] = True
    env = gym.make(env_name, obs_mode='rgbd', **kwargs)
    return env
    

def build_maniskill2_env(env_name, **kwargs):
    # Create environment
    if kwargs.get('rgb_overlay_path', None) is not None:
        if kwargs.get('rgb_overlay_cameras', None) is None:
            # Set the default camera to overlay real images for the visual-matching evaluation setting
            if 'google_robot_static' in kwargs['robot']:
                kwargs['rgb_overlay_cameras'] = ['overhead_camera']
            elif 'widowx' in kwargs['robot']:
                kwargs['rgb_overlay_cameras'] = ['3rd_view_camera']
            else:
                raise NotImplementedError()
    env = gym.make(env_name, **kwargs)
    
    return env


def get_maniskill2_env_instruction(env, **kwargs):
    # Get task description
    task_description = env.get_language_instruction()
    print(task_description)
    return task_description


def get_robot_control_mode(robot_name, policy_name):
    if 'google_robot_static' in robot_name:
        control_mode = 'arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner'
    elif 'widowx' in robot_name:
        control_mode = 'arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos'
        # control_mode = 'arm_pd_ee_delta_pose_align2_gripper_pd_joint_pos'
    else:
        raise NotImplementedError()
    print("Control mode: ", control_mode)
    return control_mode
