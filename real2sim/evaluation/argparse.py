import argparse
import numpy as np
from real2sim.utils.io import DictAction
from sapien.core import Pose
from transforms3d.euler import euler2quat
from copy import deepcopy

def parse_range_tuple(t):
    return np.linspace(t[0], t[1], int(t[2]))

def get_args():
    # parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy-model', type=str, default='rt1', help="Policy model type; e.g., 'rt1', 'octo-base', 'octo-small'")
    parser.add_argument('--policy-setup', type=str, default='google_robot', help="Policy model setup; e.g., 'google_robot', 'widowx_bridge'")
    parser.add_argument('--ckpt-path', type=str, default=None)
    parser.add_argument('--env-name', type=str, required=True)
    parser.add_argument('--additional-env-save-tags', type=str, default=None, help='Additional tags to save the environment eval results')
    parser.add_argument('--scene-name', type=str, default='google_pick_coke_can_1_v4')
    parser.add_argument('--enable-raytracing', action='store_true')
    parser.add_argument('--robot', type=str, default='google_robot_static')
    parser.add_argument('--obs-camera-name', type=str, default=None, help='Obtain image observation from this camera for policy input. None = default')
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
    parser.add_argument('--logging-dir', type=str, default='./results')
    parser.add_argument("--tf-memory-limit", type=int, default=3072, help="Tensorflow memory limit")
    parser.add_argument("--octo-init-rng", type=int, default=0, help="Octo init rng seed")
    
    args = parser.parse_args()
    
    # env args: robot pose
    args.robot_init_xs = parse_range_tuple(args.robot_init_x_range)
    args.robot_init_ys = parse_range_tuple(args.robot_init_y_range)
    args.robot_init_quats = []
    for r in parse_range_tuple(args.robot_init_rot_rpy_range[:3]):
        for p in parse_range_tuple(args.robot_init_rot_rpy_range[3:6]):
            for y in parse_range_tuple(args.robot_init_rot_rpy_range[6:]):
                args.robot_init_quats.append((Pose(q=euler2quat(r, p, y)) * Pose(q=args.robot_init_rot_quat_center)).q)
    # env args: object position
    if args.obj_variation_mode == 'xy':
        args.obj_init_xs = parse_range_tuple(args.obj_init_x_range)
        args.obj_init_ys = parse_range_tuple(args.obj_init_y_range)
    # update logging info (args.additional_env_save_tags) if using a different camera from default
    if args.obs_camera_name is not None:
        if args.additional_env_save_tags is None:
            args.additional_env_save_tags = f'obs_camera_{args.obs_camera_name}'
        else:
            args.additional_env_save_tags = args.additional_env_save_tags + f'_obs_camera_{args.obs_camera_name}'
    
    return args
    
    
def create_prepackaged_sim_eval_args(policy_model, ckpt_path, task_name, 
                                     logging_dir='./results'):
    # Create a list of arguments for prepackaged sim eval settings based on a high-level task name, for user-friendliness and simplicity, 
    # instead of manually specifying the sim eval args like in ./scripts/*.sh
    # Here, task_name is a wrapping high-level task name (e.g., 'google_robot_pick_horizontal_coke_can') that does not equal to maniskill2 environment names
    
    args = argparse.Namespace()
    
    # these args here are just for logging purposes
    args.policy_model = policy_model
    args.ckpt_path = ckpt_path
    args.logging_dir = logging_dir
    args.obs_camera_name = None
    if args.policy_model == 'rt1':
        assert args.ckpt_path is not None
    elif 'octo' in args.policy_model:
        if args.ckpt_path is None or args.ckpt_path == 'None':
            args.ckpt_path = args.policy_model
    
    # policy and robot setup, simulation & control frequency
    if 'google_robot' in task_name:
        args.policy_setup = 'google_robot'
        args.robot = 'google_robot_static'
        args.sim_freq, args.control_freq = 513, 3 # 3hz control for google robot
    elif 'widowx' in args.task:
        args.policy_setup = 'widowx_bridge'
        args.robot = 'widowx'
        args.sim_freq, args.control_freq = 500, 5 # 5hz control for widowx under Bridge setup
    else:
        raise NotImplementedError()
    
    args = [args]
    
    # environment names
    task_map_dict =  {
        'google_robot_pick_coke_can': 'GraspSingleOpenedCokeCanInScene-v0',
        'google_robot_pick_horizontal_coke_can': 'GraspSingleOpenedCokeCanInScene-v0',
        'google_robot_pick_vertical_coke_can': 'GraspSingleOpenedCokeCanInScene-v0',
        'google_robot_pick_standing_coke_can': 'GraspSingleOpenedCokeCanInScene-v0',
        'google_robot_move_near': 'MoveNearGoogleBakedTexInScene-v0',
        'google_robot_open_drawer': ['OpenTopDrawerCustomInScene-v0', 'OpenMiddleDrawerCustomInScene-v0', 'OpenBottomDrawerCustomInScene-v0'],
        'google_robot_open_top_drawer': 'OpenTopDrawerCustomInScene-v0',
        'google_robot_open_middle_drawer': 'OpenMiddleDrawerCustomInScene-v0',
        'google_robot_open_bottom_drawer': 'OpenBottomDrawerCustomInScene-v0',
        'google_robot_close_drawer': ['CloseTopDrawerCustomInScene-v0', 'CloseMiddleDrawerCustomInScene-v0', 'CloseBottomDrawerCustomInScene-v0'],
        'google_robot_close_top_drawer': 'CloseTopDrawerCustomInScene-v0',
        'google_robot_close_middle_drawer': 'CloseMiddleDrawerCustomInScene-v0',
        'google_robot_close_bottom_drawer': 'CloseBottomDrawerCustomInScene-v0',
        'widowx_spoon_on_towel': 'PutSpoonOnTableClothInScene-v0',
        'widowx_carrot_on_plate': 'PutCarrotOnPlateInScene-v0',
        'widowx_stack_cube': 'StackGreenCubeOnYellowCubeBakedTexInScene-v0',
    }
    env_names = task_map_dict[task_name]
    if not isinstance(env_names, list):
        env_names = [env_names]
        
    tmp = []
    for a in args:
        for env_name in env_names:
            arg = deepcopy(a)
            arg.env_name = env_name
            tmp.append(arg)
    args = tmp
    
    # scene names
    if 'coke_can' in task_name or 'move_near' in task_name:
        scene_name = 'google_pick_coke_can_1_v4'
    elif 'drawer' in task_name:
        scene_name = 'dummy_drawer'
    elif task_name in ['widowx_spoon_on_towel', 'widowx_carrot_on_plate', 'widowx_stack_cube']:
        scene_name = 'bridge_table_1_v1'
    else:
        raise NotImplementedError()
    for arg in args:
        arg.scene_name = scene_name
        
    # ray-tracing
    for arg in args:
        if 'drawer' in task_name:
            arg.enable_raytracing = True
        else:
            arg.enable_raytracing = False
        
    # robot and object settings, along with overlay image for visual matching evaluation setting
    rgb_overlay_root = './ManiSkill2_real2sim/data/real_inpainting/'
    if 'coke_can' in task_name:
        for arg in args:
            arg.obj_variation_mode = 'xy'
            arg.rgb_overlay_path = f'{rgb_overlay_root}/google_coke_can_real_eval_1.png'
            arg.robot_init_xs = parse_range_tuple([0.35, 0.35, 1])
            arg.robot_init_ys = parse_range_tuple([0.20, 0.20, 1])
            arg.robot_init_quats = [(Pose(q=euler2quat(0, 0, 0)) * Pose(q=[0, 0, 0, 1])).q]
            arg.obj_init_xs = parse_range_tuple([-0.35, -0.12, 5])
            arg.obj_init_ys = parse_range_tuple([-0.02, 0.42, 5])
            arg.additional_env_save_tags = None
    elif 'move_near' in task_name:
        for arg in args:
            arg.obj_variation_mode = 'episode'
            arg.rgb_overlay_path = f'{rgb_overlay_root}/google_move_near_real_eval_1.png'
            arg.robot_init_xs = parse_range_tuple([0.35, 0.35, 1])
            arg.robot_init_ys = parse_range_tuple([0.21, 0.21, 1])
            arg.robot_init_quats = [(Pose(q=euler2quat(0, 0, -0.09)) * Pose(q=[0, 0, 0, 1])).q]
            arg.obj_episode_range = [0, 60]
            arg.additional_env_save_tags = 'baked_except_bpb_orange'
    elif 'drawer' in task_name:
        overlay_ids = ['a0', 'a1', 'a2', 'b0', 'b1', 'b2', 'c0', 'c1', 'c2']
        rgb_overlay_paths = [f'{rgb_overlay_root}/open_drawer_{i}.png' for i in overlay_ids]
        robot_init_xs = [0.644, 0.765, 0.889, 0.652, 0.752, 0.851, 0.665, 0.765, 0.865]
        robot_init_ys = [-0.179, -0.182, -0.203, 0.009, 0.009, 0.035, 0.224, 0.222, 0.222]
        robot_init_rotzs = [-0.03, -0.02, -0.06, 0, 0, 0, 0, -0.025, -0.025]
        tmp = []
        for a in args:
            for (robot_init_x, robot_init_y, robot_init_rotz, rgb_overlay_path) in zip(robot_init_xs, robot_init_ys, robot_init_rotzs, rgb_overlay_paths):
                arg = deepcopy(a)
                arg.obj_variation_mode = 'xy'
                arg.rgb_overlay_path = rgb_overlay_path
                arg.robot_init_xs = parse_range_tuple([robot_init_x, robot_init_x, 1])
                arg.robot_init_ys = parse_range_tuple([robot_init_y, robot_init_y, 1])
                arg.robot_init_quats = [(Pose(q=euler2quat(0, 0, robot_init_rotz)) * Pose(q=[0, 0, 0, 1])).q]
                arg.obj_init_xs = parse_range_tuple([0, 0, 1])
                arg.obj_init_ys = parse_range_tuple([0, 0, 1])
                arg.additional_env_save_tags = None
                tmp.append(arg)
        args = tmp
    elif task_name in ['widowx_spoon_on_towel', 'widowx_carrot_on_plate', 'widowx_stack_cube']:
        for arg in args:
            arg.obj_variation_mode = 'episode'
            arg.rgb_overlay_path = f'{rgb_overlay_root}/bridge_real_eval_1.png'
            arg.robot_init_xs = parse_range_tuple([0.147, 0.147, 1])
            arg.robot_init_ys = parse_range_tuple([0.028, 0.028, 1])
            arg.robot_init_quats = [(Pose(q=euler2quat(0, 0, 0)) * Pose(q=[0, 0, 0, 1])).q]
            arg.obj_episode_range = [0, 24]
            arg.additional_env_save_tags = None
    
    # additional env build kwargs
    if 'coke_can' in task_name:
        if task_name == 'google_robot_pick_coke_can':
            opts = [{'lr_switch': True}, {'upright': True}, {'laid_vertically': True}]
        elif task_name == 'google_robot_pick_horizontal_coke_can':
            opts = [{'lr_switch': True}]
        elif task_name == 'google_robot_pick_standing_coke_can':
            opts = [{'upright': True}]
        elif task_name == 'google_robot_pick_vertical_coke_can':
            opts = [{'laid_vertically': True}]
        tmp = []
        for a in args:
            for opt in opts:
                arg = deepcopy(a)
                arg.additional_env_build_kwargs = opt
                tmp.append(arg)
        args = tmp
    elif 'move_near' in task_name:
        pass
    elif 'drawer' in task_name:
        for arg in args:
            arg.additional_env_build_kwargs = {
                'shader_dir': 'rt',
                'station_name': 'mk_station_recolor',
                'light_mode': 'simple',
                'disable_bad_material': True,
            }
    elif task_name in ['widowx_spoon_on_towel', 'widowx_carrot_on_plate', 'widowx_stack_cube']:
        pass
    
    # urdfs
    tmp = []
    for a in args:
        for urdf_version in ['None', "recolor_tabletop_visual_matching_1", "recolor_tabletop_visual_matching_2", "recolor_cabinet_visual_matching_1"]:
            if 'drawer' in task_name and urdf_version == 'None':
                continue
            arg = deepcopy(a)
            arg.additional_env_build_kwargs.update({'urdf_version': urdf_version})
            tmp.append(arg)
    args = tmp
        
    # max episode steps
    if 'google_robot' in task_name:
        if 'drawer' not in task_name:
            max_episode_steps = 80
        else:
            max_episode_steps = 113
    elif 'widowx' in task_name:
        max_episode_steps = 60
    else:
        raise NotImplementedError()
    for arg in args:
        arg.max_episode_steps = max_episode_steps
    
    return args