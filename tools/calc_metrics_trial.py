import numpy as np
import glob
import os

from scipy.stats import kruskal

def get_dir_stats(d):
    if d[-1] == '/':
        d = d[:-1]
    
    results = []
    fnames = glob.glob(d + '/*')
    for fname in fnames:
        if fname[-4:] != '.mp4':
            continue
        fname = fname.split('/')[-1].replace('.mp4', '')
        if 'success' in fname:
            results.append(1)
        elif 'failure' in fname:
            results.append(0)
            
    return results

def get_kruskal_results(x, y, name, n_ckpt=4):
    assert len(x) == len(y)
    x, y = np.array(x), np.array(y)
    print(name)
    print("overall kruskal", kruskal(x, y))
    len_x = len(x)
    len_per_ckpt = len_x // n_ckpt
    for i in range(n_ckpt):
        this_x = x[i*len_per_ckpt:(i+1)*len_per_ckpt]
        this_y = y[i*len_per_ckpt:(i+1)*len_per_ckpt]
        if np.all(this_x == this_y):
            print("all same, 1.0")
        else:
            print(kruskal(this_x, this_y))
    print("=" * 60)
    
def get_mean_diff(x, y, name):
    assert len(x) == len(y)
    print(name)
    print("meandiff", np.mean(np.abs(np.array(x) - np.array(y))))
    print("=" * 60)
    
def get_pearson(x, y, name):
    assert len(x) == len(y)
    print(name)
    x, y = np.array(x), np.array(y)
    x = x - np.mean(x)
    y = y - np.mean(y)
    print("pearson", np.sum(x * y) / (np.sqrt(np.sum(x ** 2) * np.sum(y ** 2)) + 1e-8))
    print("=" * 60)
    
def get_all_metrics(x, y, name, n_ckpt=4):
    get_kruskal_results(x, y, name, n_ckpt=n_ckpt)
    get_mean_diff(x, y, name)
    get_pearson(x, y, name)
    


task_path = 'google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner'
results_lr_switch = []
results_laid_vertically = []
results_upright = []
results_real_lr_switch = []
results_real_laid_vertically = []
results_real_upright = []


ckpt_path = '/home/xuanlin/Real2Sim/results/xid77467904_000400120/'
subtask_path = 'GraspSingleOpenedCokeCanInScene-v0_lr_switch_True/rob_0.35_0.2_rot_0.000_-0.000_3.142_rgb_overlay_google_coke_can_real_eval_1'
results_lr_switch.extend(get_dir_stats(os.path.join(ckpt_path, task_path, subtask_path)))
subtask_path = 'GraspSingleOpenedCokeCanInScene-v0_laid_vertically_True/rob_0.35_0.2_rot_0.000_-0.000_3.142_rgb_overlay_google_coke_can_real_eval_1'
results_laid_vertically.extend(get_dir_stats(os.path.join(ckpt_path, task_path, subtask_path)))
subtask_path = 'GraspSingleOpenedCokeCanInScene-v0_upright_True/rob_0.35_0.2_rgb_overlay_google_coke_can_real_eval_1'
results_upright.extend(get_dir_stats(os.path.join(ckpt_path, task_path, subtask_path)))
results_real_lr_switch.extend([1,1,1,1,1, 0,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1])
results_real_laid_vertically.extend([0,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 0,0,1,1,1,])
results_real_upright.extend([0,1,1,1,1, 0,1,1,1,0, 1,1,1,1,1, 0,1,1,0,1, 0,1,1,1,0])



ckpt_path = '/home/xuanlin/Real2Sim/results/rt1poor_xid77467904_000058240/'
subtask_path = 'GraspSingleOpenedCokeCanInScene-v0_lr_switch_True/rob_0.35_0.2_rot_0.000_-0.000_3.142_rgb_overlay_google_coke_can_real_eval_1'
results_lr_switch.extend(get_dir_stats(os.path.join(ckpt_path, task_path, subtask_path)))
subtask_path = 'GraspSingleOpenedCokeCanInScene-v0_laid_vertically_True/rob_0.35_0.2_rot_0.000_-0.000_3.142_rgb_overlay_google_coke_can_real_eval_1'
results_laid_vertically.extend(get_dir_stats(os.path.join(ckpt_path, task_path, subtask_path)))
subtask_path = 'GraspSingleOpenedCokeCanInScene-v0_upright_True/rob_0.35_0.2_rot_0.000_-0.000_3.142_rgb_overlay_google_coke_can_real_eval_1'
results_upright.extend(get_dir_stats(os.path.join(ckpt_path, task_path, subtask_path)))
results_real_lr_switch.extend([1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1])
results_real_laid_vertically.extend([0,0,1,0,0, 0,0,0,0,1, 1,0,0,1,0, 0,0,1,1,1, 1,0,1,1,1])
results_real_upright.extend([1,1,1,1,0, 1,1,1,1,1, 1,1,1,1,1, 1,0,1,0,1, 0,0,1,1,1])



ckpt_path = '/home/xuanlin/Real2Sim/results/rt_1_x_tf_trained_for_002272480_step/'
subtask_path = 'GraspSingleOpenedCokeCanInScene-v0_lr_switch_True/rob_0.35_0.2_rgb_overlay_google_coke_can_real_eval_1'
results_lr_switch.extend(get_dir_stats(os.path.join(ckpt_path, task_path, subtask_path)))
subtask_path = 'GraspSingleOpenedCokeCanInScene-v0_laid_vertically_True/rob_0.35_0.2_rot_0.000_-0.000_3.142_rgb_overlay_google_coke_can_real_eval_1'
results_laid_vertically.extend(get_dir_stats(os.path.join(ckpt_path, task_path, subtask_path)))
subtask_path = 'GraspSingleOpenedCokeCanInScene-v0_upright_True/rob_0.35_0.2_rgb_overlay_google_coke_can_real_eval_1'
results_upright.extend(get_dir_stats(os.path.join(ckpt_path, task_path, subtask_path)))
results_real_lr_switch.extend([1,1,1,1,1, 1,1,1,1,0, 1,1,1,1,1, 0,1,1,1,1, 0,1,1,1,1])
results_real_laid_vertically.extend([0,0,1,1,1, 0,0,1,1,1, 1,0,1,1,1, 0,1,1,0,1, 0,0,1,0,0,])
results_real_upright.extend([1,1,1,1,0, 1,1,1,1,1, 1,1,0,1,1, 1,1,1,1,1, 0,1,1,1,0])


ckpt_path = '/home/xuanlin/Real2Sim/results/rt1new_77467904_000001120/'
subtask_path = 'GraspSingleOpenedCokeCanInScene-v0_lr_switch_True/rob_0.35_0.2_rot_0.000_-0.000_3.142_rgb_overlay_google_coke_can_real_eval_1'
results_lr_switch.extend(get_dir_stats(os.path.join(ckpt_path, task_path, subtask_path)))
subtask_path = 'GraspSingleOpenedCokeCanInScene-v0_laid_vertically_True/rob_0.35_0.2_rot_0.000_-0.000_3.142_rgb_overlay_google_coke_can_real_eval_1'
results_laid_vertically.extend(get_dir_stats(os.path.join(ckpt_path, task_path, subtask_path)))
subtask_path = 'GraspSingleOpenedCokeCanInScene-v0_upright_True/rob_0.35_0.2_rot_0.000_-0.000_3.142_rgb_overlay_google_coke_can_real_eval_1'
results_upright.extend(get_dir_stats(os.path.join(ckpt_path, task_path, subtask_path)))
results_real_lr_switch.extend([0,0,0,0,0, 0,0,0,1,0, 1,1,0,0,0, 0,1,0,0,0, 0,1,0,0,0])
results_real_laid_vertically.extend([0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0])
results_real_upright.extend([0,0,1,0,0, 0,0,0,1,1, 0,0,0,0,1, 0,0,0,0,0, 1,0,0,0,0])




get_all_metrics(results_lr_switch, results_real_lr_switch, "coke can horizontal")
get_all_metrics(results_laid_vertically, results_real_laid_vertically, "coke can vertical")
get_all_metrics(results_upright, results_real_upright, "coke can upright")
results = np.array_split(np.stack([results_lr_switch, results_laid_vertically, results_upright], axis=0), 3, axis=-1)
results = np.concatenate([x.flatten() for x in results])
results_real = np.array_split(np.stack([results_real_lr_switch, results_real_laid_vertically, results_real_upright]), 3, axis=-1)
results_real = np.concatenate([x.flatten() for x in results_real])
get_all_metrics(results, results_real, "coke can all")


# Move Near
results = []
results_real = []

task_path = 'google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner'

ckpt_path = '/home/xuanlin/Real2Sim/results/xid77467904_000400120/'
subtask_path = 'MoveNearGoogleBakedTexInScene-v0_baked_except_bpb_orange_color_adjust_redbull/rob_0.35_0.21_rot_0.000_-0.000_3.052_rgb_overlay_google_move_near_real_eval_1'
results.extend(get_dir_stats(os.path.join(ckpt_path, task_path, subtask_path)))
results_real.extend([1,1,1,1,1,0,1,1,1,1,1,0, 0,0,1,1,1,1,1,1,1,1,0,1, 0,0,1,1,0,0,0,1,1,1,1,1, 0,1,0,0,0,0,1,1,0,0,1,1, 0,1,1,0,0,1,1,0,1,0,1,1])

ckpt_path = '/home/xuanlin/Real2Sim/results/rt1poor_xid77467904_000058240/'
subtask_path = 'MoveNearGoogleBakedTexInScene-v0_baked_except_bpb_orange_color_adjust_redbull/rob_0.35_0.21_rot_0.000_-0.000_3.052_rgb_overlay_google_move_near_real_eval_1'
results.extend(get_dir_stats(os.path.join(ckpt_path, task_path, subtask_path)))
results_real.extend([1,1,0,0,1,0,1,1,1,0,1,1, 0,0,1,1,1,1,1,1,1,1,1,1, 0,1,1,0,1,1,0,1,0,1,1,0, 1,1,0,0,1,0,0,1,0,1,0,1, 0,1,0,0,0,0,1,1,0,1,0,0])

ckpt_path = '/home/xuanlin/Real2Sim/results/rt_1_x_tf_trained_for_002272480_step/'
subtask_path = 'MoveNearGoogleBakedTexInScene-v0_baked_except_bpb_orange_color_adjust_redbull/rob_0.35_0.21_rot_0.000_-0.000_3.052_rgb_overlay_google_move_near_real_eval_1'
results.extend(get_dir_stats(os.path.join(ckpt_path, task_path, subtask_path)))
results_real.extend([1,1,0,1,1,1,1,0,0,0,0,1, 0,0,1,1,1,0,1,0,1,1,1,1, 0,1,0,0,1,0,1,0,0,0,0,0, 0,1,0,1,0,1,0,1,1,1,0,0, 1,0,0,0,1,0,0,1,0,0,0,0])

ckpt_path = '/home/xuanlin/Real2Sim/results/rt1new_77467904_000001120/'
subtask_path = 'MoveNearGoogleBakedTexInScene-v0_baked_except_bpb_orange_color_adjust_redbull/rob_0.35_0.21_rot_0.000_-0.000_3.052_rgb_overlay_google_move_near_real_eval_1'
results.extend(get_dir_stats(os.path.join(ckpt_path, task_path, subtask_path)))
results_real.extend([0] * 60)

get_all_metrics(results, results_real, "move near all")



def construct_trial_results(n, a):
    # kruskal is invariant to orderings within a checkpoint
    a = np.round(n * np.array(a)).astype(np.int32)
    result = []
    for x in a:
        result.extend([1] * x + [0] * (n - x))
    return result

get_all_metrics(construct_trial_results(27, [0.815,0.704,0.519,0.0]), construct_trial_results(27, [0.667,0.519,0.481,0.0]), 'Drawer open')
get_all_metrics(construct_trial_results(27, [0.926,0.889,0.741,0.0]), construct_trial_results(27, [0.889,0.556,0.815,0.333]), 'Drawer close')
get_all_metrics(construct_trial_results(27, [0.870,0.796,0.630,0.0]), construct_trial_results(27, [0.778,0.537,0.648,0.167]), 'Drawer all')



get_all_metrics(construct_trial_results(24, [0.167, 0.500, 0.208]), construct_trial_results(24, [0.167, 0.333, 0.125]), "Bridge carrot on plate partial", n_ckpt=3)
get_all_metrics(construct_trial_results(24, [0.000, 0.250, 0.083]), construct_trial_results(24, [0.000, 0.083, 0.000]), "Bridge carrot on plate ", n_ckpt=3)
get_all_metrics(construct_trial_results(24, [0.000, 0.292, 0.583]), construct_trial_results(24, [0.083, 0.208, 0.042]), "Bridge stack cube partial", n_ckpt=3)
get_all_metrics(construct_trial_results(24, [0.000, 0.000, 0.125]), construct_trial_results(24, [0.000, 0.000, 0.000]), "Bridge stack cube", n_ckpt=3)

# ckpt_path = '/home/xuanlin/Real2Sim/results/rt_1_x_tf_trained_for_002272480_step/'
# task_path = 'google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner'

# results = []
# subtask_path = 'GraspSingleOpenedCokeCanInScene-v0_lr_switch_True/rob_0.35_0.2_rgb_overlay_google_coke_can_real_eval_1'
# results.extend(get_dir_stats(os.path.join(ckpt_path, task_path, subtask_path)))
# subtask_path = 'GraspSingleOpenedCokeCanInScene-v0_laid_vertically_True/rob_0.35_0.2_rot_0.000_-0.000_3.142_rgb_overlay_google_coke_can_real_eval_1'
# results.extend(get_dir_stats(os.path.join(ckpt_path, task_path, subtask_path)))
# subtask_path = 'GraspSingleOpenedCokeCanInScene-v0_upright_True/rob_0.35_0.2_rgb_overlay_google_coke_can_real_eval_1'
# results.extend(get_dir_stats(os.path.join(ckpt_path, task_path, subtask_path)))

# results_real = [1,1,1,1,1, 1,1,1,1,0, 1,1,1,1,1, 0,1,1,1,1, 0,1,1,1,1,
#                 0,0,1,1,1, 0,0,1,1,1, 1,0,1,1,1, 0,1,1,0,1, 0,0,1,0,0,
#                 1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,0]

# get_all_metrics(results, results_real, "rt-1-x coke can")


































# Alternative control
for ctrl_args in ['_worse_control_2', '_worse_control_4', '_worse_control_5']:
    task_path = 'google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner'
    results_lr_switch = []
    results_laid_vertically = []
    results_upright = []
    results_real_lr_switch = []
    results_real_laid_vertically = []
    results_real_upright = []

    ckpt_path = '/home/xuanlin/Real2Sim/results/xid77467904_000400120/'
    subtask_path = f'GraspSingleOpenedCokeCanInScene-v0_lr_switch_True{ctrl_args}/rob_0.35_0.2_rot_0.000_-0.000_3.142_rgb_overlay_google_coke_can_real_eval_1'
    results_lr_switch.extend(get_dir_stats(os.path.join(ckpt_path, task_path, subtask_path)))
    subtask_path = f'GraspSingleOpenedCokeCanInScene-v0_laid_vertically_True{ctrl_args}/rob_0.35_0.2_rot_0.000_-0.000_3.142_rgb_overlay_google_coke_can_real_eval_1'
    results_laid_vertically.extend(get_dir_stats(os.path.join(ckpt_path, task_path, subtask_path)))
    subtask_path = f'GraspSingleOpenedCokeCanInScene-v0_upright_True{ctrl_args}/rob_0.35_0.2_rot_0.000_-0.000_3.142_rgb_overlay_google_coke_can_real_eval_1'
    results_upright.extend(get_dir_stats(os.path.join(ckpt_path, task_path, subtask_path)))
    results_real_lr_switch.extend([1,1,1,1,1, 0,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1])
    results_real_laid_vertically.extend([0,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 0,0,1,1,1,])
    results_real_upright.extend([0,1,1,1,1, 0,1,1,1,0, 1,1,1,1,1, 0,1,1,0,1, 0,1,1,1,0])

    ckpt_path = '/home/xuanlin/Real2Sim/results/rt1poor_xid77467904_000058240/'
    subtask_path = f'GraspSingleOpenedCokeCanInScene-v0_lr_switch_True{ctrl_args}/rob_0.35_0.2_rot_0.000_-0.000_3.142_rgb_overlay_google_coke_can_real_eval_1'
    results_lr_switch.extend(get_dir_stats(os.path.join(ckpt_path, task_path, subtask_path)))
    subtask_path = f'GraspSingleOpenedCokeCanInScene-v0_laid_vertically_True{ctrl_args}/rob_0.35_0.2_rot_0.000_-0.000_3.142_rgb_overlay_google_coke_can_real_eval_1'
    results_laid_vertically.extend(get_dir_stats(os.path.join(ckpt_path, task_path, subtask_path)))
    subtask_path = f'GraspSingleOpenedCokeCanInScene-v0_upright_True{ctrl_args}/rob_0.35_0.2_rot_0.000_-0.000_3.142_rgb_overlay_google_coke_can_real_eval_1'
    results_upright.extend(get_dir_stats(os.path.join(ckpt_path, task_path, subtask_path)))
    results_real_lr_switch.extend([1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1])
    results_real_laid_vertically.extend([0,0,1,0,0, 0,0,0,0,1, 1,0,0,1,0, 0,0,1,1,1, 1,0,1,1,1])
    results_real_upright.extend([1,1,1,1,0, 1,1,1,1,1, 1,1,1,1,1, 1,0,1,0,1, 0,0,1,1,1])

    ckpt_path = '/home/xuanlin/Real2Sim/results/rt_1_x_tf_trained_for_002272480_step/'
    subtask_path = f'GraspSingleOpenedCokeCanInScene-v0_lr_switch_True{ctrl_args}/rob_0.35_0.2_rot_0.000_-0.000_3.142_rgb_overlay_google_coke_can_real_eval_1'
    results_lr_switch.extend(get_dir_stats(os.path.join(ckpt_path, task_path, subtask_path)))
    subtask_path = f'GraspSingleOpenedCokeCanInScene-v0_laid_vertically_True{ctrl_args}/rob_0.35_0.2_rot_0.000_-0.000_3.142_rgb_overlay_google_coke_can_real_eval_1'
    results_laid_vertically.extend(get_dir_stats(os.path.join(ckpt_path, task_path, subtask_path)))
    subtask_path = f'GraspSingleOpenedCokeCanInScene-v0_upright_True{ctrl_args}/rob_0.35_0.2_rot_0.000_-0.000_3.142_rgb_overlay_google_coke_can_real_eval_1'
    results_upright.extend(get_dir_stats(os.path.join(ckpt_path, task_path, subtask_path)))
    results_real_lr_switch.extend([1,1,1,1,1, 1,1,1,1,0, 1,1,1,1,1, 0,1,1,1,1, 0,1,1,1,1])
    results_real_laid_vertically.extend([0,0,1,1,1, 0,0,1,1,1, 1,0,1,1,1, 0,1,1,0,1, 0,0,1,0,0,])
    results_real_upright.extend([1,1,1,1,0, 1,1,1,1,1, 1,1,0,1,1, 1,1,1,1,1, 0,1,1,1,0])
    
    ckpt_path = '/home/xuanlin/Real2Sim/results/rt1new_77467904_000001120/'
    subtask_path = 'GraspSingleOpenedCokeCanInScene-v0_lr_switch_True/rob_0.35_0.2_rot_0.000_-0.000_3.142_rgb_overlay_google_coke_can_real_eval_1'
    results_lr_switch.extend(get_dir_stats(os.path.join(ckpt_path, task_path, subtask_path)))
    subtask_path = 'GraspSingleOpenedCokeCanInScene-v0_laid_vertically_True/rob_0.35_0.2_rot_0.000_-0.000_3.142_rgb_overlay_google_coke_can_real_eval_1'
    results_laid_vertically.extend(get_dir_stats(os.path.join(ckpt_path, task_path, subtask_path)))
    subtask_path = 'GraspSingleOpenedCokeCanInScene-v0_upright_True/rob_0.35_0.2_rot_0.000_-0.000_3.142_rgb_overlay_google_coke_can_real_eval_1'
    results_upright.extend(get_dir_stats(os.path.join(ckpt_path, task_path, subtask_path)))
    results_real_lr_switch.extend([0,0,0,0,0, 0,0,0,1,0, 1,1,0,0,0, 0,1,0,0,0, 0,1,0,0,0])
    results_real_laid_vertically.extend([0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0])
    results_real_upright.extend([0,0,1,0,0, 0,0,0,1,1, 0,0,0,0,1, 0,0,0,0,0, 1,0,0,0,0])
    

    get_all_metrics(results_lr_switch, results_real_lr_switch, f"{ctrl_args} coke can horizontal")
    get_all_metrics(results_laid_vertically, results_real_laid_vertically, f"{ctrl_args} coke can vertical")
    get_all_metrics(results_upright, results_real_upright, f"{ctrl_args} coke can upright")
    results = np.array_split(np.stack([results_lr_switch, results_laid_vertically, results_upright], axis=0), 3, axis=-1)
    results = np.concatenate([x.flatten() for x in results])
    results_real = np.array_split(np.stack([results_real_lr_switch, results_real_laid_vertically, results_real_upright]), 3, axis=-1)
    results_real = np.concatenate([x.flatten() for x in results_real])
    get_all_metrics(results, results_real, f"{ctrl_args} coke can all")
    
    

    # # Move Near
    # results = []
    # results_real = []

    # task_path = 'google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner'

    # ckpt_path = '/home/xuanlin/Real2Sim/results/xid77467904_000400120/'
    # subtask_path = f'MoveNearGoogleInScene-v0{ctrl_args}/rob_0.35_0.21_rot_0.000_-0.000_3.052_rgb_overlay_google_move_near_real_eval_1'
    # results.extend(get_dir_stats(os.path.join(ckpt_path, task_path, subtask_path)))
    # results_real.extend([1,1,1,1,1,0,1,1,1,1,1,0, 0,0,1,1,1,1,1,1,1,1,0,1, 0,0,1,1,0,0,0,1,1,1,1,1, 0,1,0,0,0,0,1,1,0,0,1,1, 0,1,1,0,0,1,1,0,1,0,1,1])

    # ckpt_path = '/home/xuanlin/Real2Sim/results/rt1poor_xid77467904_000058240/'
    # subtask_path = f'MoveNearGoogleInScene-v0{ctrl_args}/rob_0.35_0.21_rot_0.000_-0.000_3.052_rgb_overlay_google_move_near_real_eval_1'
    # results.extend(get_dir_stats(os.path.join(ckpt_path, task_path, subtask_path)))
    # results_real.extend([1,1,0,0,1,0,1,1,1,0,1,1, 0,0,1,1,1,1,1,1,1,1,1,1, 0,1,1,0,1,1,0,1,0,1,1,0, 1,1,0,0,1,0,0,1,0,1,0,1, 0,1,0,0,0,0,1,1,0,1,0,0])

    # ckpt_path = '/home/xuanlin/Real2Sim/results/rt_1_x_tf_trained_for_002272480_step/'
    # subtask_path = f'MoveNearGoogleInScene-v0{ctrl_args}/rob_0.35_0.21_rot_0.000_-0.000_3.052_rgb_overlay_google_move_near_real_eval_1'
    # results.extend(get_dir_stats(os.path.join(ckpt_path, task_path, subtask_path)))
    # results_real.extend([1,1,0,1,1,1,1,0,0,0,0,1, 0,0,1,1,1,0,1,0,1,1,1,1, 0,1,0,0,1,0,1,0,0,0,0,0, 0,1,0,1,0,1,0,1,1,1,0,0, 1,0,0,0,1,0,0,1,0,0,0,0])

    # get_all_metrics(results, results_real, f"{ctrl_args} move near all")