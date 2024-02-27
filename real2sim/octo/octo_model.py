import numpy as np
import os
import jax
import tensorflow as tf
import matplotlib.pyplot as plt

from octo.model.octo_model import OctoModel

from transforms3d.euler import euler2axangle
from collections import deque
from real2sim.utils.action.action_ensemble import ActionEnsembler
from transformers import AutoTokenizer

class OctoInference:
    def __init__(
        self,
        model_type="octo-base",
        policy_setup='widowx_bridge',
        horizon=2,
        pred_action_horizon=4,
        exec_horizon=1,
        image_size=256,
        action_scale=1.0,
        init_rng=0,
    ):
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        if policy_setup == 'widowx_bridge':
            dataset_id = 'bridge_dataset'
            action_ensemble = True
            action_ensemble_temp = 0.0
            self.sticky_gripper_num_repeat = 1
        elif policy_setup == 'google_robot':
            dataset_id = 'fractal20220817_data'
            action_ensemble = True
            action_ensemble_temp = 0.0
            self.sticky_gripper_num_repeat = 15
        else:
            raise NotImplementedError(f"Policy setup {policy_setup} not supported for octo models.")
        self.policy_setup = policy_setup
            
        if model_type in ['octo-base', 'octo-small']:
            # released huggingface octo models
            self.model_type = f"hf://rail-berkeley/{model_type}"
            self.tokenizer, self.tokenizer_kwargs = None, None
            self.model = OctoModel.load_pretrained(self.model_type)
            self.action_mean = self.model.dataset_statistics[dataset_id]['action']['mean']
            self.action_std = self.model.dataset_statistics[dataset_id]['action']['std']
            self.automatic_task_creation = True
        else:
            # custom model path
            self.model_type = model_type
            self.tokenizer = AutoTokenizer.from_pretrained("t5-base")
            self.tokenizer_kwargs = {
                "max_length": 16,
                "padding": "max_length",
                "truncation": True,
                "return_tensors": "np",
            }
            self.model = tf.saved_model.load(self.model_type)
            if dataset_id == 'bridge_dataset':
                self.action_mean = np.array([
                    0.00021161, 0.00012614, -0.00017022, -0.00015062, -0.00023831, 0.00025646, 0.0
                ])
                self.action_std = np.array([
                    0.00963721, 0.0135066, 0.01251861, 0.02806791, 0.03016905, 0.07632624, 1.0
                ])
            elif dataset_id == 'fractal20220817_data':
                self.action_mean = np.array([ 0.00696389,  0.00627008, -0.01263256,  0.04330839, -0.00570499, 0.00089247, 0.])
                self.action_std = np.array([0.06925472, 0.06019009, 0.07354742, 0.15605888, 0.1316399, 0.14593437, 1.])
            else:
                raise NotImplementedError(f"{dataset_id} not supported yet for custom octo model checkpoints.")
            self.automatic_task_creation = False
        
        self.image_size = image_size
        self.action_scale = action_scale
        self.horizon = horizon
        self.pred_action_horizon = pred_action_horizon
        self.exec_horizon = exec_horizon
        self.action_ensemble = action_ensemble
        self.action_ensemble_temp = action_ensemble_temp
        self.rng = jax.random.PRNGKey(init_rng)
        for _ in range(5):
            # to match octo server's inference seeds
            self.rng, _key = jax.random.split(self.rng) # each shape [2,]
        
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        # self.gripper_is_closed = False
        self.previous_gripper_action = None
        
        self.task = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = ActionEnsembler(self.pred_action_horizon, self.action_ensemble_temp)
        else:
            self.action_ensembler = None
        self.num_image_history = 0
        self.time_step = 0

    def _resize_image(self, image):
        image = tf.image.resize(
            image, size=(self.image_size, self.image_size), method="lanczos3", antialias=True
        )
        image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
        return image
        
    def _add_image_to_history(self, image):
        self.image_history.append(image)
        # if self.num_image_history == 0:
        #     self.image_history.extend([image] * self.horizon)
        # else:
        #     self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)
        
    def _obtain_image_history_and_mask(self):
        images = np.stack(self.image_history, axis=0)
        horizon = len(self.image_history)
        pad_mask = np.ones(horizon, dtype=np.float64) # note: this is not np.bool
        pad_mask[:horizon - min(horizon, self.num_image_history)] = 0
        # pad_mask = np.ones(self.horizon, dtype=np.float64) # note: this is not np.bool
        # pad_mask[:self.horizon - self.num_image_history] = 0
        return images, pad_mask
        
    def reset(self, task_description):
        if self.automatic_task_creation:
            self.task = self.model.create_tasks(texts=[task_description])
        else:
            self.task = self.tokenizer(task_description, **self.tokenizer_kwargs)
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.time_step = 0
        
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        # self.gripper_is_closed = False
        self.previous_gripper_action = None

    def step(self, image, *args, **kwargs):
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        assert image.dtype == np.uint8
        image = self._resize_image(image)
        self._add_image_to_history(image)
        images, pad_mask = self._obtain_image_history_and_mask()
        images, pad_mask = images[None], pad_mask[None]
        
        # we need use a different rng key for each model forward step; this has a large impact on model performance
        self.rng, key = jax.random.split(self.rng) # each shape [2,]
        # print("octo local rng", self.rng, key)
        
        if self.automatic_task_creation:
            input_observation = {
                'image_primary': images,
                'pad_mask': pad_mask
            }
            norm_raw_actions = self.model.sample_actions(input_observation, self.task, rng=key)
        else:
            input_observation = {
                'image_primary': images,
                'timestep_pad_mask': pad_mask
            }
            input_observation = {
                'observations': input_observation,
                'tasks': {
                    'language_instruction': self.task
                },
                'rng': np.concatenate([self.rng, key])
            }
            norm_raw_actions = self.model.lc_ws2(input_observation)[:, :, :7]
        norm_raw_actions = norm_raw_actions[0]   # remove batch, becoming (action_pred_horizon, action_dim)
        assert norm_raw_actions.shape == (self.pred_action_horizon, 7)
        
        if self.action_ensemble:
            norm_raw_actions = self.action_ensembler.ensemble_action(norm_raw_actions)
            norm_raw_actions = norm_raw_actions[None] # [1, 7]
            
        raw_actions = norm_raw_actions * self.action_std[None] + self.action_mean[None]
        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]), # range [0, 1]; 1 = open; 0 = close
        }
        
        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action['world_vector'] = raw_action['world_vector'] * self.action_scale
        action_rotation_delta = np.asarray(raw_action['rotation_delta'], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action['rot_axangle'] = action_rotation_axangle * self.action_scale
        
        if self.policy_setup == 'google_robot':
            current_gripper_action = raw_action['open_gripper']
            
            # gripper_close_commanded = (current_gripper_action < 0.5)
            # relative_gripper_action = 1 if gripper_close_commanded else -1 # google robot 1 = close; -1 = open

            # # if action represents a change in gripper state and gripper is not already sticky, trigger sticky gripper
            # if gripper_close_commanded != self.gripper_is_closed and not self.sticky_action_is_on:
            #     self.sticky_action_is_on = True
            #     self.sticky_gripper_action = relative_gripper_action

            # if self.sticky_action_is_on:
            #     self.gripper_action_repeat += 1
            #     relative_gripper_action = self.sticky_gripper_action

            # if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
            #     self.gripper_is_closed = (self.sticky_gripper_action > 0)
            #     self.sticky_action_is_on = False
            #     self.gripper_action_repeat = 0
            
            # action['gripper'] = np.array([relative_gripper_action])
            
            # alternative implementation
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action # google robot 1 = close; -1 = open
            self.previous_gripper_action = current_gripper_action
        
            if np.abs(relative_gripper_action) > 0.5 and self.sticky_action_is_on is False:
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0
        
            action['gripper'] = relative_gripper_action
            
        elif self.policy_setup == 'widowx_bridge':
            action['gripper'] = 2.0 * (raw_action['open_gripper'] > 0.5) - 1.0 # binarize gripper action to 1 (open) and -1 (close)
            # self.gripper_is_closed = (action['gripper'] < 0.0)
        
        action['terminate_episode'] = np.array([0.0])
        
        self.time_step += 1
        
        return raw_action, action
    
    def visualize_epoch(self, predicted_raw_actions, images, save_path):
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'grasp']

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [
            ['image'] * len(ACTION_DIM_LABELS),
            ACTION_DIM_LABELS
        ]
        plt.rcParams.update({'font.size': 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array([np.concatenate([a['world_vector'], a['rotation_delta'], a['open_gripper']], axis=-1) for a in predicted_raw_actions])
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label='predicted action')
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel('Time in one episode')

        axs['image'].imshow(img_strip)
        axs['image'].set_xlabel('Time in one episode (subsampled)')
        plt.legend()
        plt.savefig(save_path)
        



"""
Bridge:
self.action_mean = np.array([
    0.00021161, 0.00012614, -0.00017022, -0.00015062, -0.00023831, 0.00025646, 0.0
])
self.action_std = np.array([
    0.00963721, 0.0135066, 0.01251861, 0.02806791, 0.03016905, 0.07632624, 1.0
])
"""