def initialize_additional_episode_stats(env_name):
    if 'GraspSingle' in env_name:
        episode_stats = {
            'n_lift_significant': 0,
            'consec_grasp': False,
            'grasped': False,
        }
    else:
        raise NotImplementedError()
    
    return episode_stats

def update_additional_episode_stats(env_name, episode_stats, info):
    if 'GraspSingle' in env_name:
        episode_stats['n_lift_significant'] += int(info['lifted_object_significantly'])
        episode_stats['consec_grasp'] = episode_stats['consec_grasp'] or info['consecutive_grasp']
        episode_stats['grasped'] = episode_stats['grasped'] or info['is_grasped']
    else:
        raise NotImplementedError()
    
    return episode_stats

def obtain_truncation_step_success(env_name, episode_stats, info):
    # obtain success indicator if policy never terminates
    if 'GraspSingle' in env_name:
        return (info['lifted_object_significantly'] or (episode_stats['n_lift_significant'] >= 10))
    else:
        raise NotImplementedError()