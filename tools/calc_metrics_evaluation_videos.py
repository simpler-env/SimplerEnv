import argparse
import glob
from pathlib import Path
import pandas as pd

import numpy as np
from scipy.stats import kruskal

from simpler_env.utils.metrics import (
    construct_unordered_trial_results,
    get_dir_stats,
    mean_maximum_rank_violation,
    pearson_correlation,
    print_all_kruskal_results,
)

from simpler_env.utils.metrics import REF
# Calculate metrics for each task


def calc_pick_coke_can_stats(root_result_dir):
    print("***Pick coke can results***")
    # If you use a new checkpoint, please update the real evaluation results here
    coke_can_real_success = {
        "horizontal": {
            # "rt-2-x": 0.92,
            # "rt-1-converged": 0.96,
            # "rt-1-15pct": 1.0,
            # "rt-1-x": 0.88,
            # "rt-1-begin": 0.20,
            # "octo-base": 0.44,
            # "openvla-7b": 0.44,
            # "internvla-1_8b": 0.44,
            # "internvla_diff-1_8b": 0.44,
            "ours": 0.0,
        },
        "vertical": {
            # "rt-2-x": 0.80,
            # "rt-1-converged": 0.88,
            # "rt-1-15pct": 0.96,
            # "rt-1-x": 0.56,
            # "rt-1-begin": 0.00,
            # "octo-base": 0.20,
            # "openvla-7b": 0.20,
            # "internvla-1_8b": 0.44,
            # "internvla_diff-1_8b": 0.44,
            "ours": 0.0,
        },
        "standing": {
            # "rt-2-x": 1.00,
            # "rt-1-converged": 0.72,
            # "rt-1-15pct": 0.80,
            # "rt-1-x": 0.84,
            # "rt-1-begin": 0.20,
            # "octo-base": 0.24,
            # "openvla-7b": 0.24,
            # "internvla-1_8b": 0.44,
            # "internvla_diff-1_8b": 0.44,
            "ours": 0.0,
        },
    }

    ckpt_alias_keys = list(coke_can_real_success["horizontal"].keys())
    coke_can_orientation_map_dict = {
        "horizontal": "lr_switch",
        "vertical": "laid_vertically",
        "standing": "upright",
    }
    n_orientations = len(
        coke_can_orientation_map_dict
    )  # number of coke can orientations
    n_trials_per_ckpt_per_orientation = 25  # number of trials per checkpoint per coke can orientation; update if it is different
    # extra patterns required in file name; if you are using different visual matching overlay image, please update here
    extra_pattern_require_sim_variants = ["rgb_overlay_None"]
    extra_pattern_require_visual_matching = ["rgb_overlay_google_coke_can_real_eval_1"]

    # get simulation variant success
    coke_can_sim_variant_success = {
        k1: {k2: [] for k2 in ckpt_alias_keys}
        for k1 in coke_can_orientation_map_dict.keys()
    }

    # hardcoded variant aggregation result dirs; if you have new variants, please update here
    base_variants = [
        "google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/GraspSingleOpenedCokeCanInScene-v0_{}_True",
    ]
    base_visual_matching_variants = []
    for visual_matching_urdf_version in [
        "None",
        "recolor_tabletop_visual_matching_1",
        "recolor_tabletop_visual_matching_2",
        "recolor_cabinet_visual_matching_1",
    ]:
        base_visual_matching_variants.append(
            "google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/GraspSingleOpenedCokeCanInScene-v0_{}_True"
            + f"_urdf_version_{visual_matching_urdf_version}"
        )

    background_variants = [
        "google_pick_coke_can_1_v4_alt_background/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/GraspSingleOpenedCokeCanInScene-v0_{}_True",
        "google_pick_coke_can_1_v4_alt_background_2/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/GraspSingleOpenedCokeCanInScene-v0_{}_True",
    ]
    lighting_variants = [
        "google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/GraspSingleOpenedCokeCanInScene-v0_{}_True_slightly_brighter_lighting_True",
        "google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/GraspSingleOpenedCokeCanInScene-v0_{}_True_slightly_darker_lighting_True",
    ]
    distractor_variants = [
        "google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/GraspSingleOpenedCokeCanDistractorInScene-v0_{}_True",
        "google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/GraspSingleOpenedCokeCanDistractorInScene-v0_{}_True_distractor_config_more",
    ]
    table_texture_variants = [
        "Baked_sc1_staging_objaverse_cabinet1_h870/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/GraspSingleOpenedCokeCanInScene-v0_{}_True",
        "Baked_sc1_staging_objaverse_cabinet2_h870/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/GraspSingleOpenedCokeCanInScene-v0_{}_True",
    ]
    # camera_variants = [
    #     'google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/GraspSingleOpenedCokeCanAltGoogleCameraInScene-v0_{}_True',
    #     'google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/GraspSingleOpenedCokeCanAltGoogleCamera2InScene-v0_{}_True',
    # ] # we only consider camera variants when investigating the effect of different robustness factors on the same policy's performance

    for coke_can_orientation in coke_can_orientation_map_dict.keys():
        for ckpt_alias in ckpt_alias_keys:
            for variant in (
                base_variants
                + background_variants
                + lighting_variants
                + distractor_variants
                + table_texture_variants
            ):
                variant = variant.format(
                    coke_can_orientation_map_dict[coke_can_orientation]
                )
                variant = f"{root_result_dir}/{CKPT_MAPPING[ckpt_alias]}/{variant}"
                avg_sim_success = np.mean(
                    get_dir_stats(
                        variant,
                        extra_pattern_require=extra_pattern_require_sim_variants,
                    )
                )
                if np.isnan(avg_sim_success):
                    print(f"WARNING: avg_sim_success is nan for {variant}")
                coke_can_sim_variant_success[coke_can_orientation][ckpt_alias].append(
                    avg_sim_success
                )

            coke_can_sim_variant_success[coke_can_orientation][ckpt_alias] = np.mean(
                coke_can_sim_variant_success[coke_can_orientation][ckpt_alias]
            )

    print("-" * 20)
    for coke_can_orientation in coke_can_orientation_map_dict.keys():
        print(
            f"{coke_can_orientation} sim variant avg success",
            coke_can_sim_variant_success[coke_can_orientation],
        )
        print(
            f"{coke_can_orientation} real success",
            coke_can_real_success[coke_can_orientation],
        )
        print(
            f"{coke_can_orientation} MMRV",
            mean_maximum_rank_violation(
                list(coke_can_sim_variant_success[coke_can_orientation].values()),
                list(coke_can_real_success[coke_can_orientation].values()),
            ),
        )
        print(
            f"{coke_can_orientation} pearson correlation",
            pearson_correlation(
                list(coke_can_sim_variant_success[coke_can_orientation].values()),
                list(coke_can_real_success[coke_can_orientation].values()),
            ),
        )

    avg_orientation_sim_variant_results = []
    avg_orientation_real_results = []
    for ckpt_alias in ckpt_alias_keys:
        avg_orientation_sim_variant_results.append([])
        avg_orientation_real_results.append([])
        for coke_can_orientation in coke_can_orientation_map_dict.keys():
            avg_orientation_sim_variant_results[-1].append(
                coke_can_sim_variant_success[coke_can_orientation][ckpt_alias]
            )
            avg_orientation_real_results[-1].append(
                coke_can_real_success[coke_can_orientation][ckpt_alias]
            )
        avg_orientation_sim_variant_results[-1] = np.mean(
            avg_orientation_sim_variant_results[-1]
        )
        avg_orientation_real_results[-1] = np.mean(avg_orientation_real_results[-1])
    print("avg_orientation_sim_variant_results", avg_orientation_sim_variant_results)
    print("avg_orientation_real_results", avg_orientation_real_results)
    print(
        "mean_maximum_rank_violation(avg_orientation_sim_variant_results, avg_orientation_real_results)",
        mean_maximum_rank_violation(
            avg_orientation_sim_variant_results, avg_orientation_real_results
        ),
    )
    print(
        "pearson_correlation(avg_orientation_sim_variant_results, avg_orientation_real_results)",
        pearson_correlation(
            avg_orientation_sim_variant_results, avg_orientation_real_results
        ),
    )
    print("-" * 20)

    # get visual matching success
    coke_can_sim_visual_matching_success = {
        k1: {k2: [] for k2 in ckpt_alias_keys}
        for k1 in coke_can_orientation_map_dict.keys()
    }
    for coke_can_orientation in coke_can_orientation_map_dict.keys():
        for ckpt_alias in ckpt_alias_keys:
            for variant in base_visual_matching_variants:
                variant = variant.format(
                    coke_can_orientation_map_dict[coke_can_orientation]
                )
                variant = f"{root_result_dir}/{CKPT_MAPPING[ckpt_alias]}/{variant}"
                avg_sim_success = np.mean(
                    get_dir_stats(
                        variant,
                        extra_pattern_require=extra_pattern_require_visual_matching,
                    )
                )
                if np.isnan(avg_sim_success):
                    print(f"WARNING: avg_sim_success is nan for {variant}")
                coke_can_sim_visual_matching_success[coke_can_orientation][
                    ckpt_alias
                ].append(avg_sim_success)
            print(
                f"Orientation {coke_can_orientation}, ckpt {ckpt_alias} all robot arm visual matching success: {coke_can_sim_visual_matching_success[coke_can_orientation][ckpt_alias]}"
            )
            coke_can_sim_visual_matching_success[coke_can_orientation][ckpt_alias] = (
                np.mean(
                    coke_can_sim_visual_matching_success[coke_can_orientation][
                        ckpt_alias
                    ]
                )
            )

    for coke_can_orientation in coke_can_orientation_map_dict.keys():
        print(
            f"{coke_can_orientation} visual matching sim success",
            coke_can_sim_visual_matching_success[coke_can_orientation],
        )
        print(
            f"{coke_can_orientation} real success",
            coke_can_real_success[coke_can_orientation],
        )
        print(
            f"{coke_can_orientation} MMRV",
            mean_maximum_rank_violation(
                list(
                    coke_can_sim_visual_matching_success[coke_can_orientation].values()
                ),
                list(coke_can_real_success[coke_can_orientation].values()),
            ),
        )
        print(
            f"{coke_can_orientation} pearson correlation",
            pearson_correlation(
                list(
                    coke_can_sim_visual_matching_success[coke_can_orientation].values()
                ),
                list(coke_can_real_success[coke_can_orientation].values()),
            ),
        )
        print_all_kruskal_results(
            construct_unordered_trial_results(
                n_trials_per_ckpt_per_orientation,
                list(
                    coke_can_sim_visual_matching_success[coke_can_orientation].values()
                ),
            ),
            construct_unordered_trial_results(
                n_trials_per_ckpt_per_orientation,
                list(coke_can_real_success[coke_can_orientation].values()),
            ),
            f"{coke_can_orientation} kruskal:",
        )

    avg_orientation_sim_visual_matching_results = []
    avg_orientation_real_results = []
    for ckpt_alias in ckpt_alias_keys:
        avg_orientation_sim_visual_matching_results.append([])
        avg_orientation_real_results.append([])
        for coke_can_orientation in coke_can_orientation_map_dict.keys():
            avg_orientation_sim_visual_matching_results[-1].append(
                coke_can_sim_visual_matching_success[coke_can_orientation][ckpt_alias]
            )
            avg_orientation_real_results[-1].append(
                coke_can_real_success[coke_can_orientation][ckpt_alias]
            )
        avg_orientation_sim_visual_matching_results[-1] = np.mean(
            avg_orientation_sim_visual_matching_results[-1]
        )
        avg_orientation_real_results[-1] = np.mean(avg_orientation_real_results[-1])
    print(
        "avg_orientation_sim_visual_matching_results",
        avg_orientation_sim_visual_matching_results,
    )
    print("avg_orientation_real_results", avg_orientation_real_results)
    print(
        "mean_maximum_rank_violation(avg_orientation_sim_visual_matching_results, avg_orientation_real_results)",
        mean_maximum_rank_violation(
            avg_orientation_sim_visual_matching_results, avg_orientation_real_results
        ),
    )
    print(
        "pearson_correlation(avg_orientation_sim_visual_matching_results, avg_orientation_real_results)",
        pearson_correlation(
            avg_orientation_sim_visual_matching_results, avg_orientation_real_results
        ),
    )
    print_all_kruskal_results(
        construct_unordered_trial_results(
            n_trials_per_ckpt_per_orientation * n_orientations,
            avg_orientation_sim_visual_matching_results,
        ),
        construct_unordered_trial_results(
            n_trials_per_ckpt_per_orientation * n_orientations,
            avg_orientation_real_results,
        ),
        "avg kruskal:",
    )

    print("*" * 20)
    for _ in range(3):
        print()
    # process results and return
    ret_dict = {}
    ret_dict["coke_can/matching_avg"] = avg_orientation_sim_visual_matching_results
    ret_dict["coke_can/variant_avg"] = avg_orientation_sim_variant_results
    for key, val in coke_can_sim_visual_matching_success.items():
        ret_dict[f"coke_can/matching/{key}"] = [succ for _, succ in val.items()]
    for key, val in coke_can_sim_variant_success.items():
        ret_dict[f"coke_can/variant/{key}"] = [succ for _, succ in val.items()]
    return ret_dict


def calc_move_near_stats(root_result_dir):
    print("***Move Near results***")
    # If you use a new checkpoint, please update the real evaluation results here
    move_near_real_success = {
        # "rt-2-x": 0.733,
        # "rt-1-converged": 0.633,
        # "rt-1-15pct": 0.583,
        # "rt-1-x": 0.45,
        # "rt-1-begin": 0.017,
        # "octo-base": 0.35,
        # "openvla-7b": 0.35,
        # "internvla-1_8b": 0.44,
        # "internvla_diff-1_8b": 0.44,
        "ours": 0.0,
    }

    ckpt_alias_keys = list(move_near_real_success.keys())
    n_trials_per_ckpt = 60  # number of trials per checkpoint; update if it's different

    # extra patterns required in file name; if you are using different visual matching overlay image, please update here
    extra_pattern_require_sim_variants = ["rgb_overlay_None"]
    extra_pattern_require_visual_matching = ["rgb_overlay_google_move_near_real_eval_1"]

    # get simulation variant success
    move_near_sim_variant_success = {k: [] for k in ckpt_alias_keys}

    # hardcoded variant aggregation result dirs; if you have new variants, please update here
    base_variants = [
        "google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/MoveNearGoogleInScene-v0",
    ]
    base_visual_matching_variants = []
    for visual_matching_urdf_version in [
        "None",
        "recolor_tabletop_visual_matching_1",
        "recolor_tabletop_visual_matching_2",
        "recolor_cabinet_visual_matching_1",
    ]:
        base_visual_matching_variants.append(
            f"google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/MoveNearGoogleBakedTexInScene-v0_urdf_version_{visual_matching_urdf_version}_baked_except_bpb_orange"
        )

    background_variants = [
        "google_pick_coke_can_1_v4_alt_background/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/MoveNearGoogleInScene-v0",
        "google_pick_coke_can_1_v4_alt_background_2/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/MoveNearGoogleInScene-v0",
    ]
    lighting_variants = [
        "google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/MoveNearGoogleInScene-v0_slightly_brighter_lighting_True",
        "google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/MoveNearGoogleInScene-v0_slightly_darker_lighting_True",
    ]
    distractor_variants = [
        "google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/MoveNearGoogleInScene-v0_no_distractor_True",
    ]
    table_texture_variants = [
        "Baked_sc1_staging_objaverse_cabinet1_h870/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/MoveNearGoogleInScene-v0",
        "Baked_sc1_staging_objaverse_cabinet2_h870/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/MoveNearGoogleInScene-v0",
    ]
    # camera_variants = [
    #     'google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/MoveNearAltGoogleCameraInScene-v0',
    #     'google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/MoveNearAltGoogleCamera2InScene-v0',
    # ] # we only consider camera variants when investigating the effect of different robustness factors on the same policy's performance

    for ckpt_alias in ckpt_alias_keys:
        for variant in (
            base_variants
            + background_variants
            + lighting_variants
            + distractor_variants
            + table_texture_variants
        ):
            variant = f"{root_result_dir}/{CKPT_MAPPING[ckpt_alias]}/{variant}"
            avg_sim_success = np.mean(
                get_dir_stats(
                    variant, extra_pattern_require=extra_pattern_require_sim_variants
                )
            )
            if np.isnan(avg_sim_success):
                print(f"WARNING: avg_sim_success is nan for {variant}")
            move_near_sim_variant_success[ckpt_alias].append(avg_sim_success)
        move_near_sim_variant_success[ckpt_alias] = np.mean(
            move_near_sim_variant_success[ckpt_alias]
        )

    print("-" * 20)
    print("sim variant avg success", move_near_sim_variant_success)
    print("real success", move_near_real_success)
    print(
        "MMRV",
        mean_maximum_rank_violation(
            list(move_near_sim_variant_success.values()),
            list(move_near_real_success.values()),
        ),
    )
    print(
        "pearson correlation",
        pearson_correlation(
            list(move_near_sim_variant_success.values()),
            list(move_near_real_success.values()),
        ),
    )
    print("-" * 20)

    # get visual matching success
    move_near_sim_visual_matching_success = {k: [] for k in ckpt_alias_keys}
    for ckpt_alias in ckpt_alias_keys:
        for variant in base_visual_matching_variants:
            variant = f"{root_result_dir}/{CKPT_MAPPING[ckpt_alias]}/{variant}"
            avg_sim_success = np.mean(
                get_dir_stats(
                    variant, extra_pattern_require=extra_pattern_require_visual_matching
                )
            )
            if np.isnan(avg_sim_success):
                print(f"WARNING: avg_sim_success is nan for {variant}")
            move_near_sim_visual_matching_success[ckpt_alias].append(avg_sim_success)

        print(
            f"Ckpt {ckpt_alias} all robot arm visual matching success: {move_near_sim_visual_matching_success[ckpt_alias]}"
        )
        move_near_sim_visual_matching_success[ckpt_alias] = np.mean(
            move_near_sim_visual_matching_success[ckpt_alias]
        )

    print("sim visual matching success", move_near_sim_visual_matching_success)
    print("real success", move_near_real_success)
    print(
        "visual matching MMRV",
        mean_maximum_rank_violation(
            list(move_near_sim_visual_matching_success.values()),
            list(move_near_real_success.values()),
        ),
    )
    print(
        "visual matching pearson correlation",
        pearson_correlation(
            list(move_near_sim_visual_matching_success.values()),
            list(move_near_real_success.values()),
        ),
    )
    print_all_kruskal_results(
        construct_unordered_trial_results(
            n_trials_per_ckpt, list(move_near_sim_visual_matching_success.values())
        ),
        construct_unordered_trial_results(
            n_trials_per_ckpt, list(move_near_real_success.values())
        ),
        "avg kruskal:",
    )

    print("*" * 20)
    for _ in range(3):
        print()

    # TODO: process ret results
    ret_dict = {}
    for key, val in move_near_sim_variant_success.items():
        ret_dict["move_near/variant"] = [
            succ for _, succ in move_near_sim_variant_success.items()
        ]
    for key, val in move_near_sim_visual_matching_success.items():
        ret_dict["move_near/matching"] = [
            succ for _, succ in move_near_sim_visual_matching_success.items()
        ]
    return ret_dict


def calc_drawer_stats(root_result_dir):
    print("***Drawer results***")
    # If you use a new checkpoint, please update the real evaluation results here
    drawer_real_success = {
        "open": {
            # "rt-2-x": 0.333,
            # "rt-1-converged": 0.815,
            # "rt-1-15pct": 0.704,
            # "rt-1-x": 0.519,
            # "rt-1-begin": 0.000,
            # "octo-base": 0.148,
            # "openvla-7b": 0.148,
            # "internvla-1_8b": 0.44,
            # "internvla_diff-1_8b": 0.44,
            "ours": 0.0,
        },
        "close": {
            # "rt-2-x": 0.630,
            # "rt-1-converged": 0.926,
            # "rt-1-15pct": 0.889,
            # "rt-1-x": 0.741,
            # "rt-1-begin": 0.000,
            # "octo-base": 0.519,
            # "openvla-7b": 0.519,
            # "internvla-1_8b": 0.44,
            # "internvla_diff-1_8b": 0.44,
            "ours": 0.0,
        },
    }

    ckpt_alias_keys = list(drawer_real_success["open"].keys())
    drawer_task_map_dict = {
        "open": [
            "OpenTopDrawerCustomInScene-v0",
            "OpenMiddleDrawerCustomInScene-v0",
            "OpenBottomDrawerCustomInScene-v0",
        ],
        "close": [
            "CloseTopDrawerCustomInScene-v0",
            "CloseMiddleDrawerCustomInScene-v0",
            "CloseBottomDrawerCustomInScene-v0",
        ],
    }
    n_tasks = len(drawer_task_map_dict)
    n_trials_per_ckpt_per_task = 27  # number of trials per checkpoint for all open / all close tasks; update if it is different
    # extra patterns required in file name; if you are using different visual matching overlay image, please update here
    extra_pattern_require_sim_variants = ["rgb_overlay_None"]
    extra_pattern_require_visual_matching = ["rgb_overlay_open_drawer"]

    # get simulation variant success
    drawer_sim_variant_success = {
        k1: {k2: [] for k2 in ckpt_alias_keys} for k1 in drawer_task_map_dict.keys()
    }

    # hardcoded variant aggregation result dirs; if you have new variants, please update here
    base_variants = [
        "frl_apartment_stage_simple/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/{}_shader_dir_rt",
    ]
    base_visual_matching_variants = []
    for visual_matching_urdf_version in [
        "None",
        "recolor_tabletop_visual_matching_1",
        "recolor_tabletop_visual_matching_2",
        "recolor_cabinet_visual_matching_1",
    ]:
        urdf_version_str = f"urdf_version_{visual_matching_urdf_version}"
        base_visual_matching_variants.append(
            "dummy_drawer/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/{}_shader_dir_rt_station_name_mk_station_recolor_light_mode_simple_disable_bad_material_True_"
            + urdf_version_str
        )

    background_variants = [
        "modern_bedroom_no_roof/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/{}_shader_dir_rt",
        "modern_office_no_roof/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/{}_shader_dir_rt",
    ]
    lighting_variants = [
        "frl_apartment_stage_simple/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/{}_shader_dir_rt_light_mode_brighter",
        "frl_apartment_stage_simple/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/{}_shader_dir_rt_light_mode_darker",
    ]
    table_texture_variants = [
        "frl_apartment_stage_simple/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/{}_shader_dir_rt_station_name_mk_station2",
        "frl_apartment_stage_simple/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/{}_shader_dir_rt_station_name_mk_station3",
    ]

    for drawer_task in drawer_task_map_dict.keys():
        for ckpt_alias in ckpt_alias_keys:
            for specific_task in drawer_task_map_dict[drawer_task]:
                for variant in (
                    base_variants
                    + background_variants
                    + lighting_variants
                    + table_texture_variants
                ):
                    variant = variant.format(specific_task)
                    variant = f"{root_result_dir}/{CKPT_MAPPING[ckpt_alias]}/{variant}"
                    avg_sim_success = np.mean(
                        get_dir_stats(
                            variant,
                            extra_pattern_require=extra_pattern_require_sim_variants,
                        )
                    )
                    if np.isnan(avg_sim_success):
                        print(f"WARNING: avg_sim_success is nan for {variant}")
                    drawer_sim_variant_success[drawer_task][ckpt_alias].append(
                        avg_sim_success
                    )
            drawer_sim_variant_success[drawer_task][ckpt_alias] = np.mean(
                drawer_sim_variant_success[drawer_task][ckpt_alias]
            )

    print("-" * 20)
    for drawer_task in drawer_task_map_dict.keys():
        print(
            f"{drawer_task} sim variant avg success",
            drawer_sim_variant_success[drawer_task],
        )
        print(f"{drawer_task} real success", drawer_real_success[drawer_task])
        print(
            f"{drawer_task} MMRV",
            mean_maximum_rank_violation(
                list(drawer_sim_variant_success[drawer_task].values()),
                list(drawer_real_success[drawer_task].values()),
            ),
        )
        print(
            f"{drawer_task} pearson correlation",
            pearson_correlation(
                list(drawer_sim_variant_success[drawer_task].values()),
                list(drawer_real_success[drawer_task].values()),
            ),
        )

    avg_sim_variant_results = []
    avg_real_results = []
    for ckpt_alias in ckpt_alias_keys:
        avg_sim_variant_results.append([])
        avg_real_results.append([])
        for drawer_task in drawer_task_map_dict.keys():
            avg_sim_variant_results[-1].append(
                drawer_sim_variant_success[drawer_task][ckpt_alias]
            )
            avg_real_results[-1].append(drawer_real_success[drawer_task][ckpt_alias])
        avg_sim_variant_results[-1] = np.mean(avg_sim_variant_results[-1])
        avg_real_results[-1] = np.mean(avg_real_results[-1])
    print("avg_sim_variant_results", avg_sim_variant_results)
    print("avg_real_results", avg_real_results)
    print(
        "mean_maximum_rank_violation(avg_sim_variant_results, avg_real_results)",
        mean_maximum_rank_violation(avg_sim_variant_results, avg_real_results),
    )
    print(
        "pearson_correlation(avg_sim_variant_results, avg_real_results)",
        pearson_correlation(avg_sim_variant_results, avg_real_results),
    )
    print("-" * 20)

    # get visual matching success
    drawer_sim_visual_matching_success = {
        k1: {k2: [] for k2 in ckpt_alias_keys} for k1 in drawer_task_map_dict.keys()
    }
    for drawer_task in drawer_task_map_dict.keys():
        for ckpt_alias in ckpt_alias_keys:
            for specific_task in drawer_task_map_dict[drawer_task]:
                for variant in base_visual_matching_variants:
                    variant = variant.format(specific_task)
                    variant = f"{root_result_dir}/{CKPT_MAPPING[ckpt_alias]}/{variant}"
                    avg_sim_success = np.mean(
                        get_dir_stats(
                            variant,
                            extra_pattern_require=extra_pattern_require_visual_matching,
                        )
                    )
                    if np.isnan(avg_sim_success):
                        print(f"WARNING: avg_sim_success is nan for {variant}")
                    drawer_sim_visual_matching_success[drawer_task][ckpt_alias].append(
                        avg_sim_success
                    )
            tmp_variant_avg_each_robot_arm = []
            for i in range(len(base_visual_matching_variants)):
                tmp_variant_avg_each_robot_arm.append(
                    np.mean(
                        drawer_sim_visual_matching_success[drawer_task][ckpt_alias][
                            i :: len(drawer_task_map_dict[drawer_task])
                        ]
                    )
                )
            print(
                f"Drawer task {drawer_task}, ckpt {ckpt_alias} all robot arm visual matching success: {tmp_variant_avg_each_robot_arm}"
            )
            drawer_sim_visual_matching_success[drawer_task][ckpt_alias] = np.mean(
                drawer_sim_visual_matching_success[drawer_task][ckpt_alias]
            )

    for drawer_task in drawer_task_map_dict.keys():
        print(
            f"{drawer_task} visual matching sim success",
            drawer_sim_visual_matching_success[drawer_task],
        )
        print(f"{drawer_task} real success", drawer_real_success[drawer_task])
        print(
            f"{drawer_task} MMRV",
            mean_maximum_rank_violation(
                list(drawer_sim_visual_matching_success[drawer_task].values()),
                list(drawer_real_success[drawer_task].values()),
            ),
        )
        print(
            f"{drawer_task} pearson correlation",
            pearson_correlation(
                list(drawer_sim_visual_matching_success[drawer_task].values()),
                list(drawer_real_success[drawer_task].values()),
            ),
        )
        print_all_kruskal_results(
            construct_unordered_trial_results(
                n_trials_per_ckpt_per_task,
                list(drawer_sim_visual_matching_success[drawer_task].values()),
            ),
            construct_unordered_trial_results(
                n_trials_per_ckpt_per_task,
                list(drawer_real_success[drawer_task].values()),
            ),
            f"{drawer_task} kruskal:",
        )

    avg_sim_visual_matching_results = []
    avg_real_results = []
    for ckpt_alias in ckpt_alias_keys:
        avg_sim_visual_matching_results.append([])
        avg_real_results.append([])
        for drawer_task in drawer_task_map_dict.keys():
            avg_sim_visual_matching_results[-1].append(
                drawer_sim_visual_matching_success[drawer_task][ckpt_alias]
            )
            avg_real_results[-1].append(drawer_real_success[drawer_task][ckpt_alias])
        avg_sim_visual_matching_results[-1] = np.mean(
            avg_sim_visual_matching_results[-1]
        )
        avg_real_results[-1] = np.mean(avg_real_results[-1])
    print("avg_sim_visual_matching_results", avg_sim_visual_matching_results)
    print("avg_real_results", avg_real_results)
    print(
        "mean_maximum_rank_violation(avg_sim_visual_matching_results, avg_real_results)",
        mean_maximum_rank_violation(avg_sim_visual_matching_results, avg_real_results),
    )
    print(
        "pearson_correlation(avg_sim_visual_matching_results, avg_real_results)",
        pearson_correlation(avg_sim_visual_matching_results, avg_real_results),
    )
    print_all_kruskal_results(
        construct_unordered_trial_results(
            n_trials_per_ckpt_per_task * n_tasks, avg_sim_visual_matching_results
        ),
        construct_unordered_trial_results(
            n_trials_per_ckpt_per_task * n_tasks, avg_real_results
        ),
        "avg kruskal:",
    )

    print("*" * 20)
    for _ in range(3):
        print()

    # TODO: process ret dict
    ret_dict = {
        "drawer/matching_avg": avg_sim_visual_matching_results,
        "drawer/variant_avg": avg_sim_variant_results,
    }
    for key, val in drawer_sim_visual_matching_success.items():
        ret_dict[f"drawer/matching/{key}"] = [succ for _, succ in val.items()]
    for key, val in drawer_sim_variant_success.items():
        ret_dict[f"drawer/variant/{key}"] = [succ for _, succ in val.items()]
    return ret_dict


def calc_long_horizon_apple_in_drawer_stats(root_result_dir):
    print("***Drawer results***")
    # If you use a new checkpoint, please update the real evaluation results here
    drawer_real_success = {
        "put_apple_into_top_drawer": {
            # "rt-2-x": 0.074,
            # "rt-1-converged": 0.185,
            # "rt-1-15pct": 0.185,
            # "rt-1-x": 0.407,
            # "rt-1-begin": 0.000,
            # "octo-base": 0.000,
            # "openvla-7b": 0.000,
            # "internvla-1_8b": 0.44,
            # "internvla_diff-1_8b": 0.44,
            "ours": 0.0,
        },
    }

    ckpt_alias_keys = list(drawer_real_success["put_apple_into_top_drawer"].keys())
    drawer_task_map_dict = {
        "put_apple_into_top_drawer": [
            "PlaceIntoClosedTopDrawerCustomInScene-v0",
        ],
    }
    n_tasks = len(drawer_task_map_dict)
    n_trials_per_ckpt_per_task = 27  # number of trials per checkpoint for each key in drawer_task_map_dict; update if it is different
    # extra patterns required in file name; if you are using different visual matching overlay image, please update here
    extra_pattern_require_sim_variants = ["rgb_overlay_None", "apple"]
    extra_pattern_require_visual_matching = ["rgb_overlay_open_drawer", "apple"]
    extra_log_str_variant_agg = "model_ids_apple"
    extra_log_str_visual_matching = "model_ids_baked_apple_v2"

    # get simulation variant success
    drawer_sim_variant_success = {
        k1: {k2: [] for k2 in ckpt_alias_keys} for k1 in drawer_task_map_dict.keys()
    }

    # hardcoded variant aggregation result dirs; if you have new variants, please update here
    base_variants = [
        "frl_apartment_stage_simple/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/{}_shader_dir_rt",
    ]
    base_visual_matching_variants = []
    for visual_matching_urdf_version in [
        "None",
        "recolor_tabletop_visual_matching_1",
        "recolor_tabletop_visual_matching_2",
        "recolor_cabinet_visual_matching_1",
    ]:
        urdf_version_str = f"urdf_version_{visual_matching_urdf_version}"
        base_visual_matching_variants.append(
            "dummy_drawer/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/{}_shader_dir_rt_station_name_mk_station_recolor_light_mode_simple_disable_bad_material_True_"
            + urdf_version_str
        )

    background_variants = [
        "modern_bedroom_no_roof/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/{}_shader_dir_rt",
        "modern_office_no_roof/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/{}_shader_dir_rt",
    ]
    lighting_variants = [
        "frl_apartment_stage_simple/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/{}_shader_dir_rt_light_mode_brighter",
        "frl_apartment_stage_simple/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/{}_shader_dir_rt_light_mode_darker",
    ]
    table_texture_variants = [
        "frl_apartment_stage_simple/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/{}_shader_dir_rt_station_name_mk_station2",
        "frl_apartment_stage_simple/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/{}_shader_dir_rt_station_name_mk_station3",
    ]

    for drawer_task in drawer_task_map_dict.keys():
        for ckpt_alias in ckpt_alias_keys:
            for specific_task in drawer_task_map_dict[drawer_task]:
                for variant in (
                    base_variants
                    + background_variants
                    + lighting_variants
                    + table_texture_variants
                ):
                    variant = (
                        variant.format(specific_task) + f"_{extra_log_str_variant_agg}"
                    )
                    variant = f"{root_result_dir}/{CKPT_MAPPING[ckpt_alias]}/{variant}"
                    avg_sim_success = np.mean(
                        get_dir_stats(
                            variant,
                            extra_pattern_require=extra_pattern_require_sim_variants,
                        )
                    )
                    if np.isnan(avg_sim_success):
                        print(f"WARNING: avg_sim_success is nan for {variant}")
                    drawer_sim_variant_success[drawer_task][ckpt_alias].append(
                        avg_sim_success
                    )
            drawer_sim_variant_success[drawer_task][ckpt_alias] = np.mean(
                drawer_sim_variant_success[drawer_task][ckpt_alias]
            )

    print("-" * 20)
    for drawer_task in drawer_task_map_dict.keys():
        print(
            f"{drawer_task} sim variant avg success",
            drawer_sim_variant_success[drawer_task],
        )
        print(f"{drawer_task} real success", drawer_real_success[drawer_task])
        print(
            f"{drawer_task} MMRV",
            mean_maximum_rank_violation(
                list(drawer_sim_variant_success[drawer_task].values()),
                list(drawer_real_success[drawer_task].values()),
            ),
        )
        print(
            f"{drawer_task} pearson correlation",
            pearson_correlation(
                list(drawer_sim_variant_success[drawer_task].values()),
                list(drawer_real_success[drawer_task].values()),
            ),
        )

    avg_sim_variant_results = []
    avg_real_results = []
    for ckpt_alias in ckpt_alias_keys:
        avg_sim_variant_results.append([])
        avg_real_results.append([])
        for drawer_task in drawer_task_map_dict.keys():
            avg_sim_variant_results[-1].append(
                drawer_sim_variant_success[drawer_task][ckpt_alias]
            )
            avg_real_results[-1].append(drawer_real_success[drawer_task][ckpt_alias])
        avg_sim_variant_results[-1] = np.mean(avg_sim_variant_results[-1])
        avg_real_results[-1] = np.mean(avg_real_results[-1])
    print("avg_sim_variant_results", avg_sim_variant_results)
    print("avg_real_results", avg_real_results)
    print(
        "mean_maximum_rank_violation(avg_sim_variant_results, avg_real_results)",
        mean_maximum_rank_violation(avg_sim_variant_results, avg_real_results),
    )
    print(
        "pearson_correlation(avg_sim_variant_results, avg_real_results)",
        pearson_correlation(avg_sim_variant_results, avg_real_results),
    )
    print("-" * 20)

    # get visual matching success
    drawer_sim_visual_matching_success = {
        k1: {k2: [] for k2 in ckpt_alias_keys} for k1 in drawer_task_map_dict.keys()
    }
    for drawer_task in drawer_task_map_dict.keys():
        for ckpt_alias in ckpt_alias_keys:
            for specific_task in drawer_task_map_dict[drawer_task]:
                for variant in base_visual_matching_variants:
                    variant = (
                        variant.format(specific_task)
                        + f"_{extra_log_str_visual_matching}"
                    )
                    variant = f"{root_result_dir}/{CKPT_MAPPING[ckpt_alias]}/{variant}"
                    avg_sim_success = np.mean(
                        get_dir_stats(
                            variant,
                            extra_pattern_require=extra_pattern_require_visual_matching,
                        )
                    )
                    if np.isnan(avg_sim_success):
                        print(f"WARNING: avg_sim_success is nan for {variant}")
                    drawer_sim_visual_matching_success[drawer_task][ckpt_alias].append(
                        avg_sim_success
                    )
            tmp_variant_avg_each_robot_arm = []
            for i in range(len(base_visual_matching_variants)):
                tmp_variant_avg_each_robot_arm.append(
                    np.mean(
                        drawer_sim_visual_matching_success[drawer_task][ckpt_alias][
                            i :: len(drawer_task_map_dict[drawer_task])
                        ]
                    )
                )
            print(
                f"Drawer task {drawer_task}, ckpt {ckpt_alias} all robot arm visual matching success: {tmp_variant_avg_each_robot_arm}"
            )
            drawer_sim_visual_matching_success[drawer_task][ckpt_alias] = np.mean(
                drawer_sim_visual_matching_success[drawer_task][ckpt_alias]
            )

    for drawer_task in drawer_task_map_dict.keys():
        print(
            f"{drawer_task} visual matching sim success",
            drawer_sim_visual_matching_success[drawer_task],
        )
        print(f"{drawer_task} real success", drawer_real_success[drawer_task])
        print(
            f"{drawer_task} MMRV",
            mean_maximum_rank_violation(
                list(drawer_sim_visual_matching_success[drawer_task].values()),
                list(drawer_real_success[drawer_task].values()),
            ),
        )
        print(
            f"{drawer_task} pearson correlation",
            pearson_correlation(
                list(drawer_sim_visual_matching_success[drawer_task].values()),
                list(drawer_real_success[drawer_task].values()),
            ),
        )
        print_all_kruskal_results(
            construct_unordered_trial_results(
                n_trials_per_ckpt_per_task,
                list(drawer_sim_visual_matching_success[drawer_task].values()),
            ),
            construct_unordered_trial_results(
                n_trials_per_ckpt_per_task,
                list(drawer_real_success[drawer_task].values()),
            ),
            f"{drawer_task} kruskal:",
        )

    avg_sim_visual_matching_results = []
    avg_real_results = []
    for ckpt_alias in ckpt_alias_keys:
        avg_sim_visual_matching_results.append([])
        avg_real_results.append([])
        for drawer_task in drawer_task_map_dict.keys():
            avg_sim_visual_matching_results[-1].append(
                drawer_sim_visual_matching_success[drawer_task][ckpt_alias]
            )
            avg_real_results[-1].append(drawer_real_success[drawer_task][ckpt_alias])
        avg_sim_visual_matching_results[-1] = np.mean(
            avg_sim_visual_matching_results[-1]
        )
        avg_real_results[-1] = np.mean(avg_real_results[-1])
    print("avg_sim_visual_matching_results", avg_sim_visual_matching_results)
    print("avg_real_results", avg_real_results)
    print(
        "mean_maximum_rank_violation(avg_sim_visual_matching_results, avg_real_results)",
        mean_maximum_rank_violation(avg_sim_visual_matching_results, avg_real_results),
    )
    print(
        "pearson_correlation(avg_sim_visual_matching_results, avg_real_results)",
        pearson_correlation(avg_sim_visual_matching_results, avg_real_results),
    )
    print_all_kruskal_results(
        construct_unordered_trial_results(
            n_trials_per_ckpt_per_task * n_tasks, avg_sim_visual_matching_results
        ),
        construct_unordered_trial_results(
            n_trials_per_ckpt_per_task * n_tasks, avg_real_results
        ),
        "avg kruskal:",
    )

    print("*" * 20)
    for _ in range(3):
        print()
 
    # TODO: process ret dict
    ret_dict = {
        "apple_in_drawer/matching_avg": avg_sim_visual_matching_results,
        "apple_in_drawer/variant_avg": avg_sim_variant_results,
    }
    # for key, val in drawer_sim_visual_matching_success.items():
    #     ret_dict[f"drawer/matching/{key}"] = [succ for _, succ in val.items()]
    # for key, val in drawer_sim_variant_success.items():
    #     ret_dict[f"drawer/variant/{key}"] = [succ for _, succ in val.items()]
    return ret_dict


def calc_bridge_put_on_env_stats(root_result_dir):
    print("***Bridge Put On Env results***")
    # If you use a new checkpoint, please update the real evaluation results here
    real_partial_success_dict = {
        "put_spoon_on_tablecloth": {
            # "rt-1-x": 0.042,
            # "octo-base": 0.500,
            # "openvla-7b": 0.500,
            # "internvla-1_8b": 0.44,
            # "internvla_diff-1_8b": 0.44,
            "ours": 0.0,
            # "octo-small": 0.542,
        },
        "put_carrot_on_plate": {
            # "rt-1-x": 0.167,
            # "octo-base": 0.500,
            # "openvla-7b": 0.500,
            # "internvla-1_8b": 0.44,
            # "internvla_diff-1_8b": 0.44,
            "ours": 0.0,
            # "octo-small": 0.208,
        },
        "stack_green_block_on_yellow_block": {
            # "rt-1-x": 0.000,
            # "octo-base": 0.292,
            # "openvla-7b": 0.292,
            # "internvla-1_8b": 0.44,
            # "internvla_diff-1_8b": 0.44,
            "ours": 0.0,
            # "octo-small": 0.583,
        },
        "put_eggplant_in_basket": {
            # "rt-1-x": 0.000,
            # "octo-base": 0.400,
            # "openvla-7b": 0.400,
            # "internvla-1_8b": 0.44,
            # "internvla_diff-1_8b": 0.44,
            "ours": 0.0,
            # "octo-small": 0.600,
        },
    }
    real_success_dict = {
        "put_spoon_on_tablecloth": {
            # "rt-1-x": 0.000,
            # "octo-base": 0.333,
            # "openvla-7b": 0.333,
            # "internvla-1_8b": 0.44,
            # "internvla_diff-1_8b": 0.44,
            "ours": 0.0,
            # "octo-small": 0.417,
        },
        "put_carrot_on_plate": {
            # "rt-1-x": 0.00,
            # "octo-base": 0.25,
            # "octo-small": 0.083,
            # "openvla-7b": 0,
            # "internvla-1_8b": 0.44,
            # "internvla_diff-1_8b": 0.44,
            "ours": 0.0,
        },
        "stack_green_block_on_yellow_block": {
            # "rt-1-x": 0.000,
            # "octo-base": 0.000,
            # "openvla-7b": 0.000,
            # "internvla-1_8b": 0.44,
            # "internvla_diff-1_8b": 0.44,
            "ours": 0.0,
            # "octo-small": 0.125,
        },
        "put_eggplant_in_basket": {
            # "rt-1-x": 0.000,
            # "octo-base": 0.250,
            # "openvla-7b": 0.250,
            # "internvla-1_8b": 0.44,
            # "internvla_diff-1_8b": 0.44,
            "ours": 0.0,
            # "octo-small": 0.400,
        },
    }

    tasks = list(real_success_dict.keys())
    ckpt_alias_keys = list(real_success_dict[tasks[0]].keys())
    n_trials_per_ckpt = 24  # number of trials per checkpoint; update if it's different
    octo_seed_range = [
        0,
        2,
        4,
    ]  # we average octo performance over different random seeds to reduce variance

    # extra patterns required in file name; if you are using different visual matching overlay image, please update here
    extra_pattern_require_visual_matching = {
        "put_spoon_on_tablecloth": "rgb_overlay_bridge_real_eval_1",
        "put_carrot_on_plate": "rgb_overlay_bridge_real_eval_1",
        "stack_green_block_on_yellow_block": "rgb_overlay_bridge_real_eval_1",
        "put_eggplant_in_basket": "rgb_overlay_bridge_sink",
    }

    # hardcoded; if you have new variants, please update here
    base_visual_matching_variants_dict = {
        "put_spoon_on_tablecloth": [
            "bridge_table_1_v1/arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos/PutSpoonOnTableClothInScene-v0",
        ],
        "put_carrot_on_plate": [
            "bridge_table_1_v1/arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos/PutCarrotOnPlateInScene-v0",
        ],
        "stack_green_block_on_yellow_block": [
            "bridge_table_1_v1/arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos/StackGreenCubeOnYellowCubeBakedTexInScene-v0",
        ],
        "put_eggplant_in_basket": [
            "bridge_table_1_v2/arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos/PutEggplantInBasketScene-v0",
        ],
    }

    # success pattern; if different, please update here
    succ_fail_pattern = ["success_obj_episode", "failure_obj_episode"]
    # partial success pattern; if different, please update here
    partial_succ_fail_pattern = ["is_src_obj_grasped_True", "is_src_obj_grasped_False"]
    # partial_succ_fail_pattern = ['consecutive_grasp_True', 'consecutive_grasp_False']

    # get visual matching success
    ret_dict = {}
    for task in tasks:
        real_success = real_success_dict[task]
        real_partial_success = real_partial_success_dict[task]
        print("*" * 10, f"Results for {task}", "*" * 10)

        sim_visual_matching_success = {k: [] for k in ckpt_alias_keys}
        sim_visual_matching_partial_success = {k: [] for k in ckpt_alias_keys}
        for ckpt_alias in ckpt_alias_keys:
            base_visual_matching_variants = base_visual_matching_variants_dict[task]
            if "octo" in ckpt_alias:
                # we average octo performance over different random seeds
                tmp = []
                for seed in octo_seed_range:
                    tmp.extend(
                        [
                            f"{variant}_octo_init_rng_{seed}"
                            for variant in base_visual_matching_variants
                        ]
                    )
                base_visual_matching_variants = tmp
            for variant in base_visual_matching_variants:
                variant = f"{root_result_dir}/{CKPT_MAPPING[ckpt_alias]}/{variant}"
                avg_sim_success = np.mean(
                    get_dir_stats(
                        variant,
                        extra_pattern_require=extra_pattern_require_visual_matching[
                            task
                        ],
                        succ_fail_pattern=succ_fail_pattern,
                    )
                )
                avg_sim_partial_success = np.mean(
                    get_dir_stats(
                        variant,
                        extra_pattern_require=extra_pattern_require_visual_matching[
                            task
                        ],
                        succ_fail_pattern=partial_succ_fail_pattern,
                    )
                )
                if np.isnan(avg_sim_success) or np.isnan(avg_sim_partial_success):
                    print(f"WARNING: avg_sim_success is nan for {variant}")
                sim_visual_matching_success[ckpt_alias].append(avg_sim_success)
                sim_visual_matching_partial_success[ckpt_alias].append(
                    avg_sim_partial_success
                )
            sim_visual_matching_success[ckpt_alias] = np.mean(
                sim_visual_matching_success[ckpt_alias]
            )
            sim_visual_matching_partial_success[ckpt_alias] = np.mean(
                sim_visual_matching_partial_success[ckpt_alias]
            )

        print(
            "sim visual matching partial success", sim_visual_matching_partial_success
        )
        print("real partial success", real_partial_success)
        print(
            "visual matching MMRV (partial success)",
            mean_maximum_rank_violation(
                list(sim_visual_matching_partial_success.values()),
                list(real_partial_success.values()),
            ),
        )
        print(
            "visual matching pearson correlation (partial success) ",
            pearson_correlation(
                list(sim_visual_matching_partial_success.values()),
                list(real_partial_success.values()),
            ),
        )
        print_all_kruskal_results(
            construct_unordered_trial_results(
                n_trials_per_ckpt, list(sim_visual_matching_partial_success.values())
            ),
            construct_unordered_trial_results(
                n_trials_per_ckpt, list(real_partial_success.values())
            ),
            "avg kruskal (partial success):",
        )

        print("sim visual matching success", sim_visual_matching_success)
        print("real success", real_success)
        print(
            "visual matching MMRV",
            mean_maximum_rank_violation(
                list(sim_visual_matching_success.values()), list(real_success.values())
            ),
        )
        print(
            "visual matching pearson correlation",
            pearson_correlation(
                list(sim_visual_matching_success.values()), list(real_success.values())
            ),
        )
        print_all_kruskal_results(
            construct_unordered_trial_results(
                n_trials_per_ckpt, list(sim_visual_matching_success.values())
            ),
            construct_unordered_trial_results(
                n_trials_per_ckpt, list(real_success.values())
            ),
            "avg kruskal:",
        )

        print("*" * 20)
        for _ in range(3):
            print()

        ret_dict[f"{task}/matching_partial"] = [
            succ for _, succ in sim_visual_matching_partial_success.items()
        ]
        ret_dict[f"{task}/matching_entire"] = [
            succ for _, succ in sim_visual_matching_success.items()
        ]

    return ret_dict


# Define checkpoint alias-to-directory mapping; If you use a new checkpoint, please update the dict

CKPT_MAPPING = {
    # "rt-2-x": "rt_2_x",
    # "rt-1-converged": "rt_1_tf_trained_for_000400120",
    # "rt-1-15pct": "rt_1_tf_trained_for_000058240",
    # "rt-1-x": "rt_1_x_tf_trained_for_002272480_step",
    # "rt-1-begin": "rt_1_tf_trained_for_000001120",
    # "octo-base": "octo-base",
    # "openvla-7b": "openvla-7b",
    # "internvla-1_8b": "internvla-1_8b",
    # "internvla_diff-1_8b": "internvla_diff-1_8b",
    "ours": "TODO",
    # "octo-small": "octo-small",
    # "octo-server": "octo-server",
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--log-dir-root", type=str, default="./results/", help="log directory"
)
args = parser.parse_args()

# NOTE: replace the CKPT_MAPPING with the actual checkpoint directories
CKPT_MAPPING["ours"] = Path(args.log_dir_root).name

pick_coke_can_results = calc_pick_coke_can_stats(str(Path(args.log_dir_root).parent))
move_near_real_results = calc_move_near_stats(str(Path(args.log_dir_root).parent))
drawer_results = calc_drawer_stats(str(Path(args.log_dir_root).parent))
long_horizon_apple_in_drawer_results = calc_long_horizon_apple_in_drawer_stats(str(Path(args.log_dir_root).parent))
bridge_put_on_results = calc_bridge_put_on_env_stats(
    str(Path(args.log_dir_root).parent)
)

results = {
    **pick_coke_can_results,
    **move_near_real_results,
    **drawer_results,
    **bridge_put_on_results,
    **long_horizon_apple_in_drawer_results,
    "ckpt_name": ["ours"],
}

df = pd.DataFrame(results)
df_ref = pd.DataFrame(REF)
results_df = pd.concat([df, df_ref], axis=0, ignore_index=True)
results_df = results_df.transpose()
markdown = results_df.to_markdown(f"{args.log_dir_root}/results.md", index=True)


df.loc[0, "ckpt_name"] = CKPT_MAPPING["ours"]
df.to_csv(f"{args.log_dir_root}/results.csv", index=False, float_format="%.3f")
print(df)
exit(0)

"""
octo-base variant aggregation:
pick coke can ([horizontal, vertical, standing, avg]): default urdf [0.00, 0.00, 0.00, 0.00]; recolor_sim urdf [0.009, 0.00, 0.0267, 0.012]
move near: default urdf 0.03125; recolor_sim urdf 0.033
drawer ([open, close, avg]): default urdf [0.00, 0.021, 0.011]; recolor_sim urdf [0.00, 0.016, 0.008]
"""
