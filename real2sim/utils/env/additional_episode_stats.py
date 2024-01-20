def initialize_additional_episode_stats(env_name):
    if 'GraspSingle' in env_name:
        episode_stats = {
            'n_lift_significant': 0,
            'consec_grasp': False,
            'grasped': False,
        }
    elif 'MoveNear' in env_name:
        episode_stats = {
            'all_obj_keep_height': False,
            'moved_correct_obj': False,
            'moved_wrong_obj': False,
            'near_tgt_obj': False,
            'is_closest_to_tgt': False,
        }
    elif 'OpenDrawer' in env_name:
        episode_stats = {
            'qpos': 0,
        }
    elif ('Put' in env_name or 'Stack' in env_name) and ('On' in env_name):
        episode_stats = {
            'moved_correct_obj': False,
            'moved_wrong_obj': False,
            'is_src_obj_grasped': False,
            'src_on_target': False,
            'num_success': 0,
        }
    else:
        raise NotImplementedError()
    
    return episode_stats

def update_additional_episode_stats(env_name, episode_stats, info):
    if 'GraspSingle' in env_name:
        episode_stats['n_lift_significant'] += int(info['lifted_object_significantly'])
        episode_stats['consec_grasp'] = episode_stats['consec_grasp'] or info['consecutive_grasp']
        episode_stats['grasped'] = episode_stats['grasped'] or info['is_grasped']
    elif 'MoveNear' in env_name:
        for k in episode_stats.keys():
            episode_stats[k] = info[k] # requires success at the final step
    elif 'OpenDrawer' in env_name:
        episode_stats['qpos'] = '{:.3f}'.format(info['qpos'])
    elif ('Put' in env_name or 'Stack' in env_name) and ('On' in env_name):
        for k in ['moved_correct_obj', 'moved_wrong_obj', 'src_on_target']:
            episode_stats[k] = info[k]
        episode_stats['is_src_obj_grasped'] = episode_stats['is_src_obj_grasped'] or info['is_src_obj_grasped']
        episode_stats['num_success'] += int(info['success'])
    else:
        raise NotImplementedError()
    
    return episode_stats

def obtain_truncation_step_success(env_name, episode_stats, info):
    # obtain success indicator if policy never terminates
    if 'GraspSingle' in env_name:
        return (info['lifted_object_significantly'] or (episode_stats['n_lift_significant'] >= 5))
    elif ('Put' in env_name or 'Stack' in env_name) and ('On' in env_name):
        return info['success'] or (episode_stats['num_success'] >= 5)
    else:
        return info['success']