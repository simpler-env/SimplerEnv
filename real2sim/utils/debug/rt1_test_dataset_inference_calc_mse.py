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


def main(episode, model, model_type='rt1'):
    episode_steps = list(episode['steps'])
    pred_actions, gt_actions, images = [], [], []
        
    language_instruction = episode_steps[0]['observation']['natural_language_instruction']
    print(language_instruction)
    model.reset(language_instruction)
    
    tot_mse = 0.0
    for i in range(len(episode_steps) - 1):
        episode_step = episode_steps[i]
        next_episode_step = episode_steps[i + 1]
        
        raw_action, _ = model.step(episode_step['observation']['image'], cur_gripper_closedness=0.0)
        if model_type == 'rt1':
            action_world_vector = raw_action['world_vector']
            action_rotation_delta = raw_action['rotation_delta']
            action_gripper_closedness_action = raw_action['gripper_closedness_action']
            gt_action_world_vector = episode_step['action']['world_vector']
            gt_action_rotation_delta = episode_step['action']['rotation_delta']
            gt_action_gripper_closedness_action = episode_step['action']['gripper_closedness_action']
            print("**STEP**", i)
            print("world pred", action_world_vector, "gt", gt_action_world_vector, "mse", np.mean((action_world_vector - gt_action_world_vector) ** 2))
            print("rotation pred", action_rotation_delta, "gt", gt_action_rotation_delta, "mse", np.mean((action_rotation_delta - gt_action_rotation_delta) ** 2))
            print("gripper pred", action_gripper_closedness_action, "gt", gt_action_gripper_closedness_action, "mse", np.mean((action_gripper_closedness_action - gt_action_gripper_closedness_action) ** 2))
            print("terminate pred", raw_action['terminate_episode'], "gt", episode_step['action']['terminate_episode'])
            action_concat = np.concatenate([action_world_vector, action_rotation_delta, action_gripper_closedness_action], axis=-1)
            gt_action_concat = np.concatenate([gt_action_world_vector, gt_action_rotation_delta, gt_action_gripper_closedness_action], axis=-1)
            tot_mse += np.mean((action_concat - gt_action_concat) ** 2)
        else:
            raise NotImplementedError()
        
        pred_actions.append(raw_action)
        gt_actions.append(episode_step['action'])
        images.append(episode_step['observation']['image'])
    
    # plot_pred_and_gt_action_trajectory(pred_actions, gt_actions, 
    #                                    tf.concat(tf.unstack(images[::int(len(images) // 10)], axis=0), 1).numpy())
    tot_mse /= (len(episode_steps) - 1)
    return tot_mse

if __name__ == '__main__':
    # ckpt_path = "/home/xuanlin/Real2Sim/rt_1_x_tf_trained_for_002272480_step/" # 30 traj coke can mse: 0.010757321488493518; 30 traj move near mse: 0.04338108899866212
    # ckpt_path = "/home/xuanlin/Real2Sim/xid77467904_000400120/" # 30 traj coke can mse: 0.0056369849062154085; 30 traj move near mse: 0.005759397293250814
    ckpt_path = "/home/xuanlin/Real2Sim/rt1poor_xid77467904_000058240/" # 30 traj coke can mse: 0.007822104350586; 30 traj move near mse: 0.011442187409432612
    # ckpt_path = "/home/xuanlin/Real2Sim/rt1poorearly_77467904_000010080/" # 
    # ckpt_path = "/home/xuanlin/Real2Sim/rt1_xid45615428_000315000/" # 
    # ckpt_path = '/home/xuanlin/Real2Sim/robotics_transformer/trained_checkpoints/rt1main/'
    # ckpt_path = 'rt_1_x_tf_trained_for_002272480_step'
    
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
        
    dataset_name = DATASETS[0]
    dset = tfds.builder_from_directory(builder_dir=dataset2path(dataset_name))
    dset = dset.as_dataset(split='train')
    dset_iter = iter(dset)
    
    tot_mse = 0.0
    model = RT1Inference(saved_model_path=ckpt_path)
    
    num_samples = 0
    tot_samples = 30
    while num_samples < tot_samples:
        episode = next(dset_iter)
        first_episode_step = next(iter(episode['steps']))
        # if first_episode_step['observation']['natural_language_instruction'] != 'pick coke can':
        lang = first_episode_step['observation']['natural_language_instruction'].numpy().decode('utf-8')
        if ('move' not in lang) or ('near' not in lang):
            continue
        tot_mse += main(episode, model, model_type='rt1')
        num_samples += 1
    print("tot mse", tot_mse / tot_samples)