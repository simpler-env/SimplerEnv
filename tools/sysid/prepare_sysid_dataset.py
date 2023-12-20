import numpy as np
import tensorflow_datasets as tfds
import pickle
import argparse

DATASETS = ['fractal20220817_data', 'bridge']


def dataset2path(dataset_name):
    if dataset_name == 'robo_net':
        version = '1.0.0'
    elif dataset_name == 'language_table':
        version = '0.0.1'
    else:
        version = '0.1.0'
    return f'gs://gresearch/robotics/{dataset_name}/{version}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path', type=str, default='/home/xuanlin/Downloads/sysid_dataset.pkl')
    args = parser.parse_args()
    
    dataset_name = DATASETS[0]
    dset = tfds.builder_from_directory(builder_dir=dataset2path(dataset_name))
    
    dset = dset.as_dataset(split='train', read_config=tfds.ReadConfig(add_tfds_id=True))
    dset_iter = iter(dset)
    iter_episode_id = -1
    episode_ids = [2, 4, 5, 9, 11, 805, 1257, 1495, 1539, 1991, 2398]
    
    save = []
    while iter_episode_id <= max(episode_ids):
        iter_episode_id += 1
        episode = next(dset_iter)
        if iter_episode_id not in episode_ids:
            continue
        
        to_save = []
        episode_steps = list(episode['steps'])
        for j, episode_step in enumerate(episode_steps):
            save_episode_step = {
                'base_pose_tool_reached': np.array(episode_step['observation']['base_pose_tool_reached'], dtype=np.float64),
                'action_world_vector': np.array(episode_step['action']['world_vector'], dtype=np.float64),
                'action_rotation_delta': np.array(episode_step['action']['rotation_delta'], dtype=np.float64),
                'gripper_closedness_action': np.array(episode_step['action']['gripper_closedness_action'], dtype=np.float64),
            }
            to_save.append(save_episode_step)
        save.append(to_save)
            
    with open(args.save_path, 'wb') as f:
        pickle.dump(save, f)