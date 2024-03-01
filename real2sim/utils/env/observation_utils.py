def get_image_from_maniskill2_obs_dict(env, obs, camera_name=None):
    # obtain image from observation dictionary returned by ManiSkill2 environment
    if camera_name is None:
        if 'google_robot' in env.robot_uid:
            camera_name = 'overhead_camera'
        elif 'widowx' in env.robot_uid:
            camera_name = '3rd_view_camera'
        else:
            raise NotImplementedError()
    return obs['image'][camera_name]['rgb']

def obtain_truncation_step_success(env_name, episode_stats, info):
    # obtain success indicator if policy never outputs termination
    if 'GraspSingle' in env_name:
        return (info['lifted_object_significantly'] or (episode_stats['n_lift_significant'] >= 5))
    else:
        return info['success']