import mani_skill2.envs, gymnasium as gym

def build_maniskill2_env(env_name, instruction=None, **kwargs):
    if kwargs.get('rgb_overlay_path', None) is not None:
        if kwargs['robot'] == 'google_robot_static':
            kwargs['rgb_overlay_cameras'] = ['overhead_camera']
        elif kwargs['robot'] == 'widowx':
            kwargs['rgb_overlay_cameras'] = ['3rd_view_camera']
        else:
            raise NotImplementedError()
    env = gym.make(env_name, **kwargs)
    
    # Get task description
    obj_name = ' '.join(env.obj.name.split('_')[1:])
    if instruction is not None:
        task_description = instruction
    elif env_name in ['GraspSingleYCBInScene-v0', 'GraspSingleYCBSomeInScene-v0']:
        task_description = f"pick {obj_name}"
    elif env_name in ['GraspSingleYCBFruitInScene-v0']:
        task_description = "pick fruit"
    elif env_name in ['GraspSingleYCBCanInScene-v0', 'GraspSingleYCBTomatoCanInScene-v0']:
        task_description = "pick can"
    elif 'CokeCan' in env_name:
        task_description = "pick coke can"
    elif env_name in ['GraspSinglePepsiCanInScene-v0', 'GraspSingleUpRightPepsiCanInScene-v0']:
        task_description = "pick pepsi can"
    elif env_name == 'GraspSingleYCBBoxInScene-v0':
        task_description = "pick box"
    elif env_name == 'KnockSingleYCBBoxOverInScene-v0':
        task_description = "knock box over"
    elif env_name == 'OpenDrawerCustomInScene-v0':
        # TODO: add other drawers
        task_description = "open top drawer"
    else:
        raise NotImplementedError()
    
    print(task_description)
    
    return env, task_description

def get_robot_control_mode(robot_name):
    if robot_name == 'google_robot_static':
        control_mode = 'arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner'
    elif robot_name == 'widowx':
        control_mode = 'arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos'
    return control_mode
