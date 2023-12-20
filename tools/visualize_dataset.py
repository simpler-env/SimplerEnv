import numpy as np
import tensorflow_datasets as tfds

from real2sim.utils.visualization import write_video

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
    dataset_name = DATASETS[0]
    dset = tfds.builder_from_directory(builder_dir=dataset2path(dataset_name))
    
    dset = dset.as_dataset(split='train[:20]', read_config=tfds.ReadConfig(add_tfds_id=True))
    dset = list(dset)
    for i, episode in enumerate(dset):
        gt_images = []
        episode_steps = list(episode['steps'])
        for j in range(len(episode_steps) - 1):
            gt_images.append(episode_steps[j]['observation']['image'])
        write_video(f'/home/xuanlin/Downloads/tmp/{i}_gt.mp4', gt_images, fps=5)