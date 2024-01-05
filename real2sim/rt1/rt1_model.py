from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds
import tf_agents
from tf_agents.policies import py_tf_eager_policy
from tf_agents.trajectories import time_step as ts
import tensorflow_hub as hub

from sapien.core import Pose
from transforms3d.quaternions import axangle2quat, quat2axangle

class RT1Inference:
    def __init__(
        self,
        saved_model_path="rt_1_x_tf_trained_for_002272480_step",
        lang_embed_model_path="https://tfhub.dev/google/universal-sentence-encoder-large/5",
        image_width=320,
        image_height=256,
        action_scale=1.0,
    ):
        self.lang_embed_model = hub.load(lang_embed_model_path)
        self.tfa_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
            model_path=saved_model_path,
            load_specs_from_pbtxt=True,
            use_tf_function=True,
        )
        self.image_width = image_width
        self.image_height = image_height
        self.action_scale = action_scale

        self.observation = None
        self.tfa_time_step = None
        self.policy_state = None
        self.task_description = None
        self.task_description_embedding = None
        
        self.goal_gripper_closedness = np.array([0.0])
        self.time_step = 0

    def _initialize_model(self):
        # Perform one step of inference using dummy input to trace the tensoflow graph
        # Obtain a dummy observation, where the features are all 0
        self.observation = tf_agents.specs.zero_spec_nest(
            tf_agents.specs.from_spec(self.tfa_policy.time_step_spec.observation)
        )  # "natural_language_embedding": [512], "image", [256,320,3], "natural_language_instruction": <tf.Tensor: shape=(), dtype=string, numpy=b''>
        # Construct a tf_agents time_step from the dummy observation
        self.tfa_time_step = ts.transition(
            self.observation, reward=np.zeros((), dtype=np.float32)
        )
        # Initialize the state of the policy
        self.policy_state = self.tfa_policy.get_initial_state(
            batch_size=1
        )  # {'seq_idx': shape (1,1,1,1,1), 'action_tokens': shape [1,15,11,1,1], 'context_image_tokens' (1,15,81,1,512)}
        # Run inference using the policy
        action = self.tfa_policy.action(self.tfa_time_step, self.policy_state)
        
        self.time_step = 0
        self.goal_gripper_closedness = np.array([0.0])

    def _resize_image(self, image):
        image = tf.image.resize_with_pad(
            image, target_width=self.image_width, target_height=self.image_height
        )
        image = tf.cast(image, tf.uint8)
        return image

    def reset(self, task_description):
        self._initialize_model()
        if task_description is not None:
            self.task_description = task_description
            self.task_description_embedding = self.lang_embed_model([task_description])[0]
        else:
            self.task_description = ''
            self.task_description_embedding = tf.zeros((512,), dtype=tf.float32)

    def step(self, image, cur_gripper_closedness):
        image = self._resize_image(image)
        self.observation["image"] = image
        self.observation["natural_language_embedding"] = self.task_description_embedding
        # self.observation["natural_language_instruction"] = tf.constant(self.task_description, dtype=tf.string)

        self.tfa_time_step = ts.transition(
            self.observation, reward=np.zeros((), dtype=np.float32)
        )
        policy_step = self.tfa_policy.action(self.tfa_time_step, self.policy_state)
        raw_action = policy_step.action # keys: ['base_displacement_vector', 'rotation_delta', 'world_vector', 'base_displacement_ve...l_rotation', 'gripper_closedness_action', 'terminate_episode']
        if tf.abs(raw_action['gripper_closedness_action'][0]) < 0.01:
            raw_action['gripper_closedness_action'][0] = 0.0 # regard small gripper actions as 0
        
        # process raw_action to obtain the action to be sent to the environment
        action = {}
        action['world_vector'] = raw_action['world_vector'] * self.action_scale
        action_rotation_delta = np.asarray(raw_action['rotation_delta'], dtype=np.float64) # this rotation_delta is in fact in axis-angle representation (not rpy)
        action_rotation_angle = np.linalg.norm(action_rotation_delta)
        action_rotation_ax = action_rotation_delta / action_rotation_angle if action_rotation_angle > 1e-6 else np.array([0., 1., 0.])
        action['rot_axangle'] = action_rotation_ax * action_rotation_angle * self.action_scale
        if np.abs(raw_action['gripper_closedness_action'][0]) > 0:
            action['gripper_closedness_action'] = cur_gripper_closedness + raw_action['gripper_closedness_action']
            self.goal_gripper_closedness = action['gripper_closedness_action'] # update gripper joint position goal
        else:
            action['gripper_closedness_action'] = self.goal_gripper_closedness # repeat last target gripper joint position
            
        action['terminate_episode'] = raw_action['terminate_episode']
                
        self.policy_state = policy_step.state
        self.time_step += 1
        
        return raw_action, action
    
    def visualize_epoch(self, predicted_raw_actions, images, save_path):
        images = [self._resize_image(image) for image in images]
        predicted_action_name_to_values_over_time = defaultdict(list)
        figure_layout = [
            "terminate_episode_0",
            "terminate_episode_1",
            "terminate_episode_2",
            "world_vector_0",
            "world_vector_1",
            "world_vector_2",
            "rotation_delta_0",
            "rotation_delta_1",
            "rotation_delta_2",
            "gripper_closedness_action_0",
        ]
        action_order = [
            "terminate_episode",
            "world_vector",
            "rotation_delta",
            "gripper_closedness_action",
        ]

        for i, action in enumerate(predicted_raw_actions):
            for action_name in action_order:
                for action_sub_dimension in range(action[action_name].shape[0]):
                    # print(action_name, action_sub_dimension)
                    title = f"{action_name}_{action_sub_dimension}"
                    predicted_action_name_to_values_over_time[title].append(
                        predicted_raw_actions[i][action_name][action_sub_dimension]
                    )

        figure_layout = [["image"] * len(figure_layout), figure_layout]

        plt.rcParams.update({"font.size": 12})

        stacked = tf.concat(tf.unstack(images[::3], axis=0), 1)

        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        for i, (k, v) in enumerate(predicted_action_name_to_values_over_time.items()):
            axs[k].plot(
                predicted_action_name_to_values_over_time[k], label="predicted action"
            )
            axs[k].set_title(k)
            axs[k].set_xlabel("Time in one episode")

        axs["image"].imshow(stacked.numpy())
        axs["image"].set_xlabel("Time in one episode (subsampled)")

        plt.legend()
        plt.savefig(save_path)