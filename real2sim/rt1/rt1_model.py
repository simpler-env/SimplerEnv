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
from transforms3d.euler import euler2axangle
from transforms3d.quaternions import axangle2quat, quat2axangle
from real2sim.utils.action.action_ensemble import ActionEnsembler

class RT1Inference:
    def __init__(
        self,
        saved_model_path="rt_1_x_tf_trained_for_002272480_step",
        lang_embed_model_path="https://tfhub.dev/google/universal-sentence-encoder-large/5",
        image_width=320,
        image_height=256,
        action_scale=1.0,
        policy_setup='google_robot',
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
        
        self.policy_setup = policy_setup
        if self.policy_setup == 'google_robot':
            self.unnormalize_action = False
            self.unnormalize_action_fxn = None
            self.invert_gripper_action = False
            self.action_rotation_mode = 'axis_angle'
            self.action_ensembler = None
        elif self.policy_setup == 'widowx_bridge':
            self.unnormalize_action = True
            self.unnormalize_action_fxn = self._unnormalize_action_widowx_bridge
            self.invert_gripper_action = True
            self.action_rotation_mode = 'rpy'
            self.action_ensembler = None
            # self.action_ensembler = ActionEnsembler(4, 0.0)
        else:
            raise NotImplementedError()
        
        self.goal_gripper_closedness = np.array([0.0])
        self.time_step = 0

    @staticmethod
    def _rescale_action_with_bound(
        actions: np.ndarray,
        low: float,
        high: float,
        safety_margin: float = 0,
        post_scaling_max: float = 1.0,
        post_scaling_min: float = -1.0,
    ) -> np.ndarray:
        """Formula taken from https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range."""
        resc_actions = (actions - low) / (high - low) * (
            post_scaling_max - post_scaling_min
        ) + post_scaling_min
        return np.clip(
            resc_actions,
            post_scaling_min + safety_margin,
            post_scaling_max - safety_margin,
        )
    
    def _unnormalize_action_widowx_bridge(self, action):
        action['world_vector'] = self._rescale_action_with_bound(
            action['world_vector'],
            low=-1.75,
            high=1.75,
            post_scaling_max=0.05,
            post_scaling_min=-0.05,
        )
        action['rotation_delta'] = self._rescale_action_with_bound(
            action['rotation_delta'],
            low=-1.4,
            high=1.4,
            post_scaling_max=0.25,
            post_scaling_min=-0.25,
        )

        return action
    
    # def _unnormalize_action_widowx_bridge(self, action):
    #     # https://github.com/Asap7772/rt1_eval/blob/2fad77e9bf4def2ef82604d445270f83475e9726/kitchen_eval/rt1_wrapper.py
    #     # the last dimension is -1.0 because maniskill2 widowx gripper action range normalizes to [-1, 1]
    #     # see ManiSkill2_real2sim/mani_skill2/agents/configs/widowx/defaults.py
    #     rescaled_action_min = np.array([-0.05, -0.05, -0.05, -0.25, -0.25, -0.25, -1.0])
    #     rescaled_action_max = np.array([0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 1.0])
    #     action_concat = np.concatenate([action['world_vector'], action['rotation_delta'], action['gripper_closedness_action']])
    #     action_rescaled = (action_concat + 1.0) / 2.0 * (rescaled_action_max - rescaled_action_min) + rescaled_action_min
        
    #     if np.any(action_rescaled > rescaled_action_max):
    #         print('action bounds violated: ', action_rescaled)
    #     if np.any(action_rescaled < rescaled_action_min):
    #         print('action bounds violated: ', action_rescaled)

    #     action['world_vector'] = action_rescaled[:3]
    #     action['rotation_delta'] = action_rescaled[3:6]
    #     action['gripper_closedness_action'] = action_rescaled[6:]
        
    #     return action

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

    @staticmethod
    def _small_action_filter_google_robot(raw_action, arm_movement=False, gripper=True):
        # small action filtering
        if arm_movement:
            raw_action['world_vector'] = tf.where(
                tf.abs(raw_action['world_vector']) < 5e-3, tf.zeros_like(raw_action['world_vector']), raw_action['world_vector']
            )
            raw_action['rotation_delta'] = tf.where(
                tf.abs(raw_action['rotation_delta']) < 5e-3, tf.zeros_like(raw_action['rotation_delta']), raw_action['rotation_delta']
            )
            raw_action['base_displacement_vector'] = tf.where(
                raw_action['base_displacement_vector'] < 5e-3, 
                tf.zeros_like(raw_action['base_displacement_vector']), 
                raw_action['base_displacement_vector']
            )
            raw_action['base_displacement_vertical_rotation'] = tf.where(
                raw_action['base_displacement_vertical_rotation'] < 1e-2,
                tf.zeros_like(raw_action['base_displacement_vertical_rotation']),
                raw_action['base_displacement_vertical_rotation']
            )
        if gripper:
            raw_action['gripper_closedness_action'] = tf.where(
                tf.abs(raw_action['gripper_closedness_action']) < 1e-2,
                tf.zeros_like(raw_action['gripper_closedness_action']),
                raw_action['gripper_closedness_action']
            )
        return raw_action
        
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
        if self.policy_setup == 'google_robot':
            raw_action = self._small_action_filter_google_robot(raw_action, arm_movement=False, gripper=True)
        
        if self.unnormalize_action:
            raw_action = self.unnormalize_action_fxn(raw_action)
            
        # ensemble action if needed
        ensembled_raw_action = raw_action.copy()
        if self.action_ensembler is not None:
            if self.policy_setup == 'widowx_bridge':
                raw_action_concat = np.concatenate([raw_action['world_vector'], raw_action['rotation_delta'], raw_action['gripper_closedness_action']])
                raw_action_concat = self.action_ensembler.ensemble_action(raw_action_concat)
                ensembled_raw_action['world_vector'] = raw_action_concat[:3]
                ensembled_raw_action['rotation_delta'] = raw_action_concat[3:6]
                ensembled_raw_action['gripper_closedness_action'] = raw_action_concat[6:]
        
        # process raw_action to obtain the action to be sent to the environment
        action = {}
        action['world_vector'] = np.asarray(ensembled_raw_action['world_vector'], dtype=np.float64) * self.action_scale
        if self.action_rotation_mode == 'axis_angle':
            action_rotation_delta = np.asarray(ensembled_raw_action['rotation_delta'], dtype=np.float64)
            action_rotation_angle = np.linalg.norm(action_rotation_delta)
            action_rotation_ax = action_rotation_delta / action_rotation_angle if action_rotation_angle > 1e-6 else np.array([0., 1., 0.])
            action['rot_axangle'] = action_rotation_ax * action_rotation_angle * self.action_scale
        elif self.action_rotation_mode in ['rpy', 'ypr', 'pry']:
            if self.action_rotation_mode == 'rpy':
                roll, pitch, yaw = np.asarray(ensembled_raw_action['rotation_delta'], dtype=np.float64)
            elif self.action_rotation_mode == 'ypr':
                yaw, pitch, roll = np.asarray(ensembled_raw_action['rotation_delta'], dtype=np.float64)
            elif self.action_rotation_mode == 'pry':
                pitch, roll, yaw = np.asarray(ensembled_raw_action['rotation_delta'], dtype=np.float64)
            action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
            action['rot_axangle'] = action_rotation_ax * action_rotation_angle * self.action_scale
        else:
            raise NotImplementedError()
        
        raw_gripper_closedness = ensembled_raw_action['gripper_closedness_action']
        if self.invert_gripper_action:
            # for some embodiments, -1 is open gripper, 1 is closed gripper; we uniformize to -1 is closed gripper, 1 is open gripper
            raw_gripper_closedness = -raw_gripper_closedness
        
        # gripper_pd_joint_target_pos:
        # if np.abs(ensembled_raw_action['gripper_closedness_action'][0]) > 0:
        #     action['gripper_closedness_action'] = cur_gripper_closedness + raw_gripper_closedness
        #     self.goal_gripper_closedness = action['gripper_closedness_action'] # update gripper joint position goal
        # else:
        #     action['gripper_closedness_action'] = self.goal_gripper_closedness # repeat last target gripper joint position
        
        if self.policy_setup == 'google_robot':
            # gripper_pd_joint_target_delta_pos_interpolate_by_planner
            if np.abs(ensembled_raw_action['gripper_closedness_action'][0]) > 0:
                action['gripper'] = np.asarray(raw_gripper_closedness, dtype=np.float64) # update gripper joint position goal
            else:
                action['gripper'] = np.array([0.0]) # repeat last target gripper joint position
        elif self.policy_setup == 'widowx_bridge':
            # gripper_pd_joint_pos; input raw_gripper_closedness has range of [-1, 1]
            action['gripper'] = np.asarray(raw_gripper_closedness, dtype=np.float64)
            # binarize gripper action to be -1 or 1
            action['gripper'] = 2.0 * (action['gripper'] > 0.0) - 1.0
        else:
            raise NotImplementedError()
            
        action['terminate_episode'] = ensembled_raw_action['terminate_episode']
                
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