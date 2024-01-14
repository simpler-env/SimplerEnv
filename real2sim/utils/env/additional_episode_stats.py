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
            'near_tgt_obj': False,
            'is_closest_to_tgt': False,
        }
    elif 'OpenDrawer' in env_name:
        episode_stats = {
            'qpos': 0,
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
    else:
        raise NotImplementedError()
    
    return episode_stats

def obtain_truncation_step_success(env_name, episode_stats, info):
    # obtain success indicator if policy never terminates
    if 'GraspSingle' in env_name:
        return (info['lifted_object_significantly'] or (episode_stats['n_lift_significant'] >= 5))
    elif 'MoveNear' in env_name:
        return info['success']
    elif 'OpenDrawer' in env_name:
        return info['success']
    else:
        raise NotImplementedError()