import numpy as np
import os
import tensorflow as tf
import argparse

from transforms3d.euler import euler2axangle, euler2quat, quat2euler
from sapien.core import Pose

from real2sim.rt1.rt1_model import RT1Inference
try:
    from real2sim.octo.octo_model import OctoInference
except ImportError as e:
    print("Octo is not correctly imported.")
    print(e)
from real2sim.utils.visualization import write_video
from real2sim.utils.env.env_builder import build_maniskill2_env, get_maniskill2_env_instruction, get_robot_control_mode
from real2sim.utils.env.observation_utils import get_image_from_maniskill2_obs_dict, obtain_truncation_step_success
from real2sim.utils.io import DictAction

def main(model, ckpt_path, robot_name, env_name, scene_name, 
         robot_init_x, robot_init_y, robot_init_quat, 
         control_mode,
         obj_init_x=None, obj_init_y=None, obj_episode_id=None,
         additional_env_build_kwargs=None,
         rgb_overlay_path=None,
         control_freq=3, sim_freq=513, max_episode_steps=80,
         instruction=None,
         action_scale=1.0, enable_raytracing=False,
         additional_env_save_tags=None):
    
    if additional_env_build_kwargs is None:
        additional_env_build_kwargs = {}
    
    # Create environment
    kwargs = dict(
        obs_mode='rgbd',
        robot=robot_name,
        sim_freq=sim_freq,
        control_mode=control_mode,
        control_freq=control_freq,
        max_episode_steps=max_episode_steps,
        scene_name=scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=rgb_overlay_path,
    )
    if enable_raytracing:
        kwargs['shader_dir'] = 'rt'
        kwargs['render_config'] = {"rt_samples_per_pixel": 128, "rt_use_denoiser": True}
    env_reset_options = {
        'robot_init_options': {
            'init_xy': np.array([robot_init_x, robot_init_y]),
            'init_rot_quat': robot_init_quat,
        }
    }
    if obj_init_x is not None:
        assert obj_init_y is not None
        obj_variation_mode = 'xy'
        env_reset_options['obj_init_options'] = {
            'init_xy': np.array([obj_init_x, obj_init_y]),
        }
    else:
        assert obj_episode_id is not None
        obj_variation_mode = 'episode'
        env_reset_options['obj_init_options'] = {
            'episode_id': obj_episode_id,
        }
        
    # Build and initialize environment
    env = build_maniskill2_env(
        env_name,
        **additional_env_build_kwargs,
        **kwargs,
    )
    obs, _ = env.reset(options=env_reset_options)
    # Obtain language instruction
    if instruction is not None:
        task_description = instruction
    else:
        task_description = get_maniskill2_env_instruction(env, env_name)
    # Initialize logging
    image = get_image_from_maniskill2_obs_dict(obs, robot_name)
    images = [image]
    predicted_actions = []
    predicted_terminated, done, truncated = False, False, False
    
    # Initialize model
    model.reset(task_description)
        
    timestep = 0
    success = "failure"
    
    # Step the environment
    while not (predicted_terminated or truncated):
        raw_action, action = model.step(image)
        predicted_actions.append(raw_action)
        print(timestep, raw_action)
        predicted_terminated = bool(action['terminate_episode'][0] > 0)
        
        obs, reward, done, truncated, info = env.step(
            np.concatenate(
                [action['world_vector'], 
                action['rot_axangle'],
                action['gripper']
                ]
            )
        )
        if predicted_terminated and info['success']:
            success = "success"
        
        image = get_image_from_maniskill2_obs_dict(obs, robot_name)
        images.append(image)
        timestep += 1
        print(info)

    episode_stats = info.get("episode_stats", {})
    # if policy never outputs terminate throughout a trajectory, obtain an episode's success status based on episode stats and last step's info
    if obtain_truncation_step_success(env_name, episode_stats, info):
        success = "success"
    
    # save video
    env_save_name = env_name
    for k, v in additional_env_build_kwargs.items():
        env_save_name = env_save_name + f'_{k}_{v}'
    if additional_env_save_tags is not None:
        env_save_name = env_save_name + f'_{additional_env_save_tags}'
    ckpt_path_basename = ckpt_path if ckpt_path[-1] != '/' else ckpt_path[:-1]
    ckpt_path_basename = ckpt_path_basename.split('/')[-1]
    if obj_variation_mode == 'xy':
        video_name = f'{success}_obj_{obj_init_x}_{obj_init_y}'
    elif obj_variation_mode == 'episode':
        video_name = f'{success}_obj_episode_{obj_episode_id}'
    for k, v in episode_stats.items():
        video_name = video_name + f'_{k}_{v}'
    video_name = video_name + '.mp4'
    if rgb_overlay_path is not None:
        rgb_overlay_path_str = os.path.splitext(os.path.basename(rgb_overlay_path))[0]
    else:
        rgb_overlay_path_str = 'None'
    r, p, y = quat2euler(robot_init_quat)
    video_path = f'{ckpt_path_basename}/{scene_name}/{control_mode}/{env_save_name}/rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_rgb_overlay_{rgb_overlay_path_str}/{video_name}'
    video_path = 'results/' + video_path
    write_video(video_path, images, fps=5)
    
    # save action trajectory
    action_path = video_path.replace('.mp4', '.png')
    action_root = os.path.dirname(action_path) + '/actions/'
    os.makedirs(action_root, exist_ok=True)
    action_path = action_root + os.path.basename(action_path)
    model.visualize_epoch(predicted_actions, images, save_path=action_path)
    

def parse_range_tuple(t):
    return np.linspace(t[0], t[1], int(t[2]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy-model', type=str, default='rt1', help="Policy model type; e.g., 'rt1', 'octo-base', 'octo-small'")
    parser.add_argument('--policy-setup', type=str, default='google_robot', choices=['google_robot', 'widowx_bridge'])
    parser.add_argument('--ckpt-path', type=str, default=None)
    parser.add_argument('--env-name', type=str, required=True)
    parser.add_argument('--additional-env-save-tags', type=str, default=None, help='Additional tags to save the environment eval results')
    parser.add_argument('--scene-name', type=str, default='google_pick_coke_can_1_v4')
    parser.add_argument('--enable-raytracing', action='store_true')
    parser.add_argument('--robot', type=str, default='google_robot_static')
    parser.add_argument('--action-scale', type=float, default=1.0)
    
    parser.add_argument('--control-freq', type=int, default=3)
    parser.add_argument('--sim-freq', type=int, default=513)
    parser.add_argument('--max-episode-steps', type=int, default=80)
    parser.add_argument('--rgb-overlay-path', type=str, default=None)
    parser.add_argument('--robot-init-x-range', type=float, nargs=3, default=[0.35, 0.35, 1], help="[xmin, xmax, num]")
    parser.add_argument('--robot-init-y-range', type=float, nargs=3, default=[0.20, 0.20, 1], help="[ymin, ymax, num]")
    parser.add_argument('--robot-init-rot-quat-center', type=float, nargs=4, default=[1, 0, 0, 0], help="[x, y, z, w]")
    parser.add_argument('--robot-init-rot-rpy-range', type=float, nargs=9, default=[0, 0, 1, 0, 0, 1, 0, 0, 1], 
                        help="[rmin, rmax, rnum, pmin, pmax, pnum, ymin, ymax, ynum]")
    parser.add_argument('--obj-variation-mode', type=str, default='xy', choices=['xy', 'episode'], help="Whether to vary the xy position of a single object, or to vary predetermined episodes")
    parser.add_argument('--obj-episode-range', type=int, nargs=2, default=[0, 60], help="[start, end]")
    parser.add_argument('--obj-init-x-range', type=float, nargs=3, default=[-0.35, -0.12, 5], help="[xmin, xmax, num]")
    parser.add_argument('--obj-init-y-range', type=float, nargs=3, default=[-0.02, 0.42, 5], help="[ymin, ymax, num]")
    
    parser.add_argument("--additional-env-build-kwargs", nargs="+", action=DictAction,
        help="Additional env build kwargs in xxx=yyy format. If the value "
        'is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument("--tmp-exp", action='store_true', help="debug flag")
    
    args = parser.parse_args()
    
    os.environ['DISPLAY'] = ''
    
    if args.policy_model == 'rt1':
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) > 0:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
      
    # env args
    control_mode = get_robot_control_mode(args.robot, args.policy_model)
    control_freq, sim_freq, max_episode_steps = args.control_freq, args.sim_freq, args.max_episode_steps
    robot_init_xs = parse_range_tuple(args.robot_init_x_range)
    robot_init_ys = parse_range_tuple(args.robot_init_y_range)
    if args.obj_variation_mode == 'xy':
        obj_init_xs = parse_range_tuple(args.obj_init_x_range)
        obj_init_ys = parse_range_tuple(args.obj_init_y_range)
    robot_init_quats = []
    for r in parse_range_tuple(args.robot_init_rot_rpy_range[:3]):
        for p in parse_range_tuple(args.robot_init_rot_rpy_range[3:6]):
            for y in parse_range_tuple(args.robot_init_rot_rpy_range[6:]):
                robot_init_quats.append((Pose(q=euler2quat(r, p, y)) * Pose(q=args.robot_init_rot_quat_center)).q)
    additional_env_build_kwargs = args.additional_env_build_kwargs
    
    # policy
    if args.policy_model == 'rt1':
        assert args.ckpt_path is not None
        model = RT1Inference(saved_model_path=args.ckpt_path, action_scale=args.action_scale,
                             policy_setup=args.policy_setup)
    elif 'octo' in args.policy_model:
        args.ckpt_path = args.policy_model
        model = OctoInference(model_type=args.policy_model, action_scale=args.action_scale,
                              policy_setup=args.policy_setup)
    else:
        raise NotImplementedError()
    
    # run inference
    for robot_init_x in robot_init_xs:
        for robot_init_y in robot_init_ys:
            for robot_init_quat in robot_init_quats:
                kwargs = dict(
                    additional_env_build_kwargs=additional_env_build_kwargs,
                    rgb_overlay_path=args.rgb_overlay_path,
                    control_freq=control_freq, sim_freq=sim_freq, max_episode_steps=max_episode_steps,
                    action_scale=args.action_scale, 
                    enable_raytracing=args.enable_raytracing,
                    additional_env_save_tags=args.additional_env_save_tags
                )
                if args.obj_variation_mode == 'xy':
                    for obj_init_x in obj_init_xs:
                        for obj_init_y in obj_init_ys:
                            main(model, args.ckpt_path, args.robot, args.env_name, args.scene_name, 
                                robot_init_x, robot_init_y, robot_init_quat, 
                                control_mode,
                                obj_init_x=obj_init_x, obj_init_y=obj_init_y,
                                **kwargs)
                elif args.obj_variation_mode == 'episode':
                    for obj_episode_id in range(args.obj_episode_range[0], args.obj_episode_range[1]):
                        main(model, args.ckpt_path, args.robot, args.env_name, args.scene_name, 
                            robot_init_x, robot_init_y, robot_init_quat, 
                            control_mode,
                            obj_episode_id=obj_episode_id,
                            **kwargs)
                else:
                    raise NotImplementedError()