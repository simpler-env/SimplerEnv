import numpy as np
import os
import tensorflow as tf

from transforms3d.euler import euler2axangle, euler2quat
from sapien.core import Pose

from real2sim.rt1.rt1_model import RT1Inference
from real2sim.utils.visualization import write_video
from real2sim.utils.env.env_builder import build_maniskill2_env

def main(env_name, scene_name, ckpt_path='rt_1_x_tf_trained_for_002272480_step',
         additional_env_build_kwargs={},
         robot_init_x=None, robot_init_y=None, robot_init_quat=[0, 0, 0, 1], 
         obj_init_x_range=None, obj_init_y_range=None,
         rgb_overlay_path=None, tmp_exp=False,
         control_freq=3, sim_freq=513, max_episode_steps=60,
         instruction=None,
         action_scale=1.0,
         env_save_name=None):
    robot_init_x = robot_init_x if robot_init_x is not None else 0.31 # hardcoded
    robot_init_y = robot_init_y if robot_init_y is not None else 0.188 # hardcoded
    obj_init_x_range = obj_init_x_range if obj_init_x_range is not None else np.linspace(-0.35, -0.1, 10)
    obj_init_y_range = obj_init_y_range if obj_init_y_range is not None else np.linspace(0.0, 0.4, 10)
    
    if 'YCB' in env_name:
        asset_root='/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/mani_skill2_ycb/'
    else:
        asset_root='/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/custom/'
    
    # Build RT-1 Model
    rt1_model = RT1Inference(saved_model_path=ckpt_path, action_scale=action_scale)
    
    for obj_init_x in obj_init_x_range:
        for obj_init_y in obj_init_y_range:
            # Create environment
            env, task_description = build_maniskill2_env(
                        env_name,
                        # control_mode='arm_pd_ee_delta_pose_align_interpolate_gripper_pd_joint_pos',
                        # control_mode='arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_pos',
                        # control_mode='arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_pos_interpolate_by_planner',
                        control_mode='arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner',
                        # control_mode='arm_pd_ee_delta_pose_align_gripper_pd_joint_target_pos',
                        # control_mode='arm_pd_ee_delta_pose_align_interpolate_gripper_pd_joint_target_pos',
                        # control_mode='arm_pd_ee_target_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_pos',
                        # control_mode='arm_pd_ee_target_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner',
                        obs_mode='rgbd',
                        robot='google_robot_static',
                        sim_freq=sim_freq,
                        control_freq=control_freq,
                        max_episode_steps=max_episode_steps,
                        asset_root=asset_root,
                        scene_root='/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/hab2_bench_assets/',
                        scene_name=scene_name,
                        camera_cfgs={"add_segmentation": True},
                        rgb_overlay_path=rgb_overlay_path,
                        rgb_overlay_cameras=['overhead_camera'],
                        instruction=instruction,
                        **additional_env_build_kwargs,
                        # Enable Ray Tracing
                        # shader_dir="rt",
                        # render_config={"rt_samples_per_pixel": 8, "rt_use_denoiser": True},
            )
            env_reset_options = {
                'obj_init_options': {
                    'init_xy': np.array([obj_init_x, obj_init_y]),
                },
                'robot_init_options': {
                    'init_xy': np.array([robot_init_x, robot_init_y]),
                    'init_rot_quat': robot_init_quat,
                }
            }
            
            # Reset and initialize environment
            predicted_actions = []
            images = []
            obs, _ = env.reset(options=env_reset_options)
            rt1_model.reset(task_description)
            
            image = obs['image']['overhead_camera']['rgb']
            images.append(image)
            predicted_terminated, done, truncated = False, False, False
               
            timestep = 0
            n_lift_significant = 0
            success = "failure"
            if 'Grasp' in env_name:
                consecutive_grasp = False
                grasped = False
            # Step the environment
            while not (predicted_terminated or truncated):
                cur_gripper_closedness = env.agent.get_gripper_closedness()
                
                raw_action, action = rt1_model.step(image, cur_gripper_closedness)
                predicted_actions.append(raw_action)
                print(timestep, raw_action)
                predicted_terminated = bool(action['terminate_episode'][0] > 0)
                
                obs, reward, done, truncated, info = env.step(
                    np.concatenate(
                        [action['world_vector'], 
                        action['rot_axangle'],
                        action['gripper_closedness_action']
                        ]
                    )
                )
                
                n_lift_significant += int(info['lifted_object_significantly'])
                if predicted_terminated and info['success']:
                    success = "success"
                if 'Grasp' in env_name:
                    if info['consecutive_grasp']:
                        consecutive_grasp = True
                    if info['is_grasped']:
                        grasped = True
                image = obs['image']['overhead_camera']['rgb']
                images.append(image)
                timestep += 1
                print(info)

            # obtain success indicator if policy never terminates
            if info['lifted_object_significantly'] or (n_lift_significant >= 10):
                success = "success"
            
            # save video
            if env_save_name is None:
                env_save_name = env_name
            ckpt_path_basename = ckpt_path if ckpt_path[-1] != '/' else ckpt_path[:-1]
            ckpt_path_basename = ckpt_path_basename.split('/')[-1]
            video_name = f'{success}_obj_{obj_init_x}_{obj_init_y}'
            if 'Grasp' in env_name:
                video_name = video_name + f'_consgrasp{consecutive_grasp}_grasp{grasped}'
            video_name = video_name + '.mp4'
            # video_path = f'{ckpt_path_basename}/{scene_name}/scanned_coke_can_dec27/delta_pose_align_interpolate_by_planner_contvel_v2/{env_save_name}_fix_robot_grid_search_obj/rob_{robot_init_x}_{robot_init_y}_rgb_overlay_{rgb_overlay_path is not None}/{video_name}'
            video_path = f'{ckpt_path_basename}/{scene_name}/scanned_coke_can_dec27_overlay1png/delta_pose_align_interpolate_by_planner_contvel_v2/{env_save_name}_fix_robot_grid_search_obj/rob_{robot_init_x}_{robot_init_y}_rgb_overlay_{rgb_overlay_path is not None}/{video_name}'
            # video_path = f'{ckpt_path_basename}/{scene_name}/delta_pose_align_interpolate_by_planner_contvel_v2/{env_save_name}_fix_robot_grid_search_obj/rob_{robot_init_x}_{robot_init_y}_rgb_overlay_{rgb_overlay_path is not None}/{video_name}'
            # video_path = f'{ckpt_path_basename}/{scene_name}/very_long_coke_can_debug/delta_pose_align_interpolate_by_planner_contvel_v2/{env_save_name}_fix_robot_grid_search_obj/rob_{robot_init_x}_{robot_init_y}_rgb_overlay_{rgb_overlay_path is not None}/{video_name}'
            if not tmp_exp:
                video_path = 'results/' + video_path
            else:
                video_path = 'results_tmp/' + video_path 
            write_video(video_path, images, fps=5)
            
            # save action trajectory
            action_path = video_path.replace('.mp4', '.png')
            action_root = os.path.dirname(action_path) + '/actions/'
            os.makedirs(action_root, exist_ok=True)
            action_path = action_root + os.path.basename(action_path)
            rt1_model.visualize_epoch(predicted_actions, images, save_path=action_path)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['DISPLAY'] = ''
    rt1_x_ckpt_path = '/home/xuanlin/Real2Sim/rt_1_x_tf_trained_for_002272480_step'
    rt1_main_ckpt_path = '/home/xuanlin/Real2Sim/robotics_transformer/trained_checkpoints/rt1main/'
    rt1_best_ckpt_path = '/home/xuanlin/Real2Sim/rt1_xid45615428_000315000/'
    rt1_poor_ckpt_path = '/home/xuanlin/Real2Sim/rt1poor_xid77467904_000058240'
    
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])

    # Baked_sc1_staging_table_616385
    # robot_init_x, robot_init_y = 0.32, 0.188
    # rob_init_quat = (Pose(q=[0, 0, 0, 1]) * Pose(q=euler2quat(0, 0, -0.01))).q
    # obj_init_x_range = np.linspace(-0.35, -0.1, 5)
    # obj_init_y_range = np.linspace(0.0, 0.4, 5)
    # rgb_overlay_path = '/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/google_table_top_1.png'
    # for env_name in ['GraspSingleVerticalCokeCanInScene-v0', 'GraspSingleCokeCanInScene-v0', 'GraspSingleUpRightOpenedCokeCanInScene-v0']:
    #     main(env_name, 'Baked_sc1_staging_table_616385', rgb_overlay_path=rgb_overlay_path,
    #          obj_init_x_range=obj_init_x_range, obj_init_y_range=obj_init_y_range,
    #          robot_init_x=robot_init_x, robot_init_y=robot_init_y, robot_init_quat=rob_init_quat)
        
    # google robot pick coke can reproduce real
    control_freq, sim_freq, max_episode_steps = 3, 513, 80
    robot_init_x, robot_init_y = 0.35, 0.20 # 0.188
    scene_name = 'google_pick_coke_can_1_v4'
    
    rob_init_quat = Pose(q=[0, 0, 0, 1]).q
    obj_init_x_range = np.linspace(-0.35, -0.12, 5)
    obj_init_y_range = np.linspace(-0.02, 0.42, 5)
    rgb_overlay_path = '/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/google_coke_can_real_eval_1.png' # 1.png / 1.jpg
    
    # rob_init_quat = (Pose(q=euler2quat(0, 0, 0.03)) * Pose(q=[0, 0, 0, 1])).q
    # obj_init_x_range = np.linspace(-0.35, -0.12, 5)
    # obj_init_y_range = np.linspace(-0.02, 0.42, 5)
    # rgb_overlay_path = '/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/google_coke_can_real_eval_2.png'
    
    env_names = ['GraspSingleOpenedCokeCanInScene-v0'] * 3
    save_env_names = ['GraspSingleUpRightOpenedCokeCanInScene-v0', 'GraspSingleVerticalOpenedCokeCanInScene-v0', 'GraspSingleLRSwitchOpenedCokeCanInScene-v0']
    additional_kwargs_list = [
        {'upright': True},
        {'laid_vertically': True},
        {'lr_switch': True},
    ]
    for ckpt_path in [rt1_x_ckpt_path, rt1_best_ckpt_path, rt1_poor_ckpt_path]:
    # for ckpt_path in [rt1_best_ckpt_path, rt1_poor_ckpt_path]:
    # for ckpt_path in [rt1_x_ckpt_path]:
        for env_name, save_env_name, additional_kwargs in zip(env_names, save_env_names, additional_kwargs_list):           
            main(env_name, scene_name, 
                additional_env_build_kwargs=additional_kwargs,
                control_freq=control_freq, sim_freq=sim_freq, max_episode_steps=max_episode_steps,
                ckpt_path=ckpt_path,
                rgb_overlay_path=rgb_overlay_path,
                obj_init_x_range=obj_init_x_range, obj_init_y_range=obj_init_y_range,
                robot_init_x=robot_init_x, robot_init_y=robot_init_y, robot_init_quat=rob_init_quat,
                env_save_name=save_env_name)
            
            main(env_name, scene_name, 
                additional_env_build_kwargs=additional_kwargs,
                control_freq=control_freq, sim_freq=sim_freq, max_episode_steps=max_episode_steps,
                ckpt_path=ckpt_path,
                obj_init_x_range=obj_init_x_range, obj_init_y_range=obj_init_y_range,
                robot_init_x=robot_init_x, robot_init_y=robot_init_y, robot_init_quat=rob_init_quat,
                env_save_name=save_env_name)
            
    
    # debug
    # env_names = ['GraspSingleOpenedCokeCanInScene-v0']
    # save_env_names = ['GraspSingleUpRightOpenedCokeCanInScene-v0']
    # additional_kwargs_list = [
    #     {'upright': True},
    # ]
    # for ckpt_path in [rt1_best_ckpt_path]:
    # # for ckpt_path in [rt1_x_ckpt_path]:
    #     for env_name, save_env_name, additional_kwargs in zip(env_names, save_env_names, additional_kwargs_list):
    #         main(env_name, 'google_pick_coke_can_1_v4', 
    #             additional_env_build_kwargs=additional_kwargs,
    #             control_freq=control_freq, sim_freq=sim_freq, max_episode_steps=max_episode_steps,
    #             ckpt_path=ckpt_path,
    #             rgb_overlay_path=rgb_overlay_path,
    #             obj_init_x_range=[-0.12], obj_init_y_range=[0.42],
    #             robot_init_x=robot_init_x, robot_init_y=robot_init_y, robot_init_quat=rob_init_quat,
    #             env_save_name=save_env_name)