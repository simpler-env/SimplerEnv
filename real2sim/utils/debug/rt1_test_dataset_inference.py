import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from PIL import Image
from IPython import display
import tqdm, os

from real2sim.rt1.rt1_model import RT1Inference
from real2sim.utils.visualization import plot_pred_and_gt_action_trajectory

DATASETS = ['fractal20220817_data']

def dataset2path(dataset_name):
    if dataset_name == 'robo_net':
        version = '1.0.0'
    elif dataset_name == 'language_table':
        version = '0.0.1'
    else:
        version = '0.1.0'
    return f'gs://gresearch/robotics/{dataset_name}/{version}'


def main(dset, ckpt_path, model_type='rt1'):
    episode = next(iter(dset))
    episode_steps = list(episode['steps'])
    pred_actions, gt_actions, images = [], [], []
    
    if model_type == 'rt1':
        model = RT1Inference(saved_model_path=ckpt_path)
    else:
        raise NotImplementedError()
    
    language_instruction = episode_steps[0]['observation']['natural_language_instruction']
    print(language_instruction)
    model.reset(language_instruction)
    
    for i in range(len(episode_steps) - 1):
        episode_step = episode_steps[i]
        next_episode_step = episode_steps[i + 1]
        
        action = model.step(episode_step['observation']['image'])
        if model_type == 'rt1':
            action_world_vector = action['world_vector']
            action_rotation_delta = action['rotation_delta']
            action_gripper_closedness_action = action['gripper_closedness_action']
            gt_action_world_vector = episode_step['action']['world_vector']
            gt_action_rotation_delta = episode_step['action']['rotation_delta']
            gt_action_gripper_closedness_action = episode_step['action']['gripper_closedness_action']
            print("**STEP**", i)
            print("world pred", action_world_vector, "gt", gt_action_world_vector, "mse", np.mean((action_world_vector - gt_action_world_vector) ** 2))
            print("rotation pred", action_rotation_delta, "gt", gt_action_rotation_delta, "mse", np.mean((action_rotation_delta - gt_action_rotation_delta) ** 2))
            print("gripper pred", action_gripper_closedness_action, "gt", gt_action_gripper_closedness_action, "mse", np.mean((action_gripper_closedness_action - gt_action_gripper_closedness_action) ** 2))
            print("terminate pred", action['terminate_episode'], "gt", episode_step['action']['terminate_episode'])
        else:
            raise NotImplementedError()
        
        pred_actions.append(action)
        gt_actions.append(episode_step['action'])
        images.append(episode_step['observation']['image'])
    
    plot_pred_and_gt_action_trajectory(pred_actions, gt_actions, 
                                       tf.concat(tf.unstack(images[::int(len(images) // 10)], axis=0), 1).numpy())

if __name__ == '__main__':
    ckpt_path = '/home/xuanlin/Real2Sim/robotics_transformer/trained_checkpoints/rt1main/'
    # ckpt_path = 'rt_1_x_tf_trained_for_002272480_step'
    
    dataset_name = DATASETS[0]
    dset = tfds.builder_from_directory(builder_dir=dataset2path(dataset_name))
    dset = dset.as_dataset(split='train[:10]')
    
    main(dset, ckpt_path, model_type='rt1')