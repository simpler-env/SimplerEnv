import numpy as np
import tensorflow_datasets as tfds

from simpler_env.utils.visualization import write_video

DATASETS = ["fractal20220817_data", "bridge"]


def dataset2path(dataset_name):
    if dataset_name == "robo_net":
        version = "1.0.0"
    elif dataset_name == "language_table":
        version = "0.0.1"
    else:
        version = "0.1.0"
    return f"gs://gresearch/robotics/{dataset_name}/{version}"


if __name__ == "__main__":
    dataset_name = DATASETS[0]
    dset = tfds.builder_from_directory(builder_dir=dataset2path(dataset_name))

    dset = dset.as_dataset(split="train[:50]", read_config=tfds.ReadConfig(add_tfds_id=True))
    dset = list(dset)
    for i, episode in enumerate(dset):
        gt_images = []
        episode_steps = list(episode["steps"])
        for j in range(len(episode_steps) - 1):
            gt_images.append(episode_steps[j]["observation"]["image"])
        write_video(f"{dataset_name}_vis/{i}_gt.mp4", gt_images, fps=5)

        # from matplotlib import pyplot as plt

        # images = gt_images
        # ACTION_DIM_LABELS = ['x', 'y', 'z', 'yaw', 'pitch', 'roll', 'grasp']

        # img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # # set up plt figure
        # figure_layout = [
        #     ['image'] * len(ACTION_DIM_LABELS),
        #     ACTION_DIM_LABELS
        # ]
        # plt.rcParams.update({'font.size': 12})
        # fig, axs = plt.subplot_mosaic(figure_layout)
        # fig.set_size_inches([45, 10])

        # # plot actions
        # pred_actions = np.array([np.concatenate([episode_step['action']['world_vector'], episode_step['action']['rotation_delta'], episode_step['action']['open_gripper'][None]], axis=-1) for episode_step in episode_steps])
        # for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
        #     # actions have batch, horizon, dim, in this example we just take the first action for simplicity
        #     axs[action_label].plot(pred_actions[:, action_dim], label='predicted action')
        #     axs[action_label].set_title(action_label)
        #     axs[action_label].set_xlabel('Time in one episode')

        # axs['image'].imshow(img_strip)
        # axs['image'].set_xlabel('Time in one episode (subsampled)')
        # plt.legend()
        # plt.show()
