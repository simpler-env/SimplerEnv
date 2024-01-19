def get_image_from_maniskill2_obs_dict(obs, robot_name):
    if 'google_robot' in robot_name:
        camera_name = 'overhead_camera'
    elif robot_name == 'widowx':
        camera_name = '3rd_view_camera'
    else:
        raise NotImplementedError()
    return obs['image'][camera_name]['rgb']