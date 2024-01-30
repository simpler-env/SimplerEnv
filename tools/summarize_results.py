from pathlib import Path
import pandas as pd


TASK_NAMES = [
    "OpenTopDrawerCustomInScene-v0",
    "OpenMiddleDrawerCustomInScene-v0",
    "OpenBottomDrawerCustomInScene-v0",
    "CloseTopDrawerCustomInScene-v0",
    "CloseMiddleDrawerCustomInScene-v0",
    "CloseBottomDrawerCustomInScene-v0",
]

LAYOUT_IDS = ["a0", "a1", "a2", "b0", "b1", "b2", "c0", "c1", "c2"]


def parse_open_drawer_results_overlay(result_dir, task_suffix="shader_dir_rt"):
    result_dir = Path(result_dir)
    # for task_dir in sorted(result_dir.iterdir()):
    #     if not task_dir.is_dir():
    #         continue
    #     task_name = task_dir.name.split("CustomInScene")[0]
    df = pd.DataFrame()
    for task_name in TASK_NAMES:
        dirname = task_name + "_" + task_suffix
        task_dir = result_dir / dirname
        print(task_name)
        results = []
        for video_dir in sorted(task_dir.iterdir()):
            print(video_dir.name)

            layout_id = video_dir.name.split("_")[-1]
            if layout_id not in LAYOUT_IDS:
                continue

            video_paths = list(video_dir.glob("./*.mp4"))
            assert len(video_paths) == 1, video_paths
            for video_path in video_paths:
                success = "success" in video_path.stem
                qpos = float(video_path.stem.split("_")[-1])
                results.append([task_name, layout_id, success, qpos])
        _df = pd.DataFrame(
            results, columns=["task_name", "layout_id", "success", "qpos"]
        )
        _df.sort_values(["layout_id"], inplace=True)
        df = pd.concat([df, _df], ignore_index=True)

    output_fname = task_suffix
    df.to_csv(result_dir / f"{output_fname}.csv")

TASK_SUFFIX = "shader_dir_rt"
# TASK_SUFFIX = "shader_dir_rt_station_name_mk_station2"

# parse_open_drawer_results_overlay(
#     "results/xid77467904_000400120/dummy/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner",
#     task_suffix=TASK_SUFFIX
# )
# parse_open_drawer_results_overlay(
#     "results/rt1poor_xid77467904_000058240/dummy/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner",
#     task_suffix=TASK_SUFFIX
# )
# parse_open_drawer_results_overlay(
#     "results/rt_1_x_tf_trained_for_002272480_step/dummy/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner",
#     task_suffix=TASK_SUFFIX
# )


def parse_open_drawer_results_sim(result_dir, task_suffix="shader_dir_rt"):
    result_dir = Path(result_dir)
    df = pd.DataFrame()

    for task_name in TASK_NAMES:
        dirname = task_name + "_" + task_suffix
        task_dir = result_dir / dirname
        print(task_name)

        results = []
        for video_dir in sorted(task_dir.iterdir()):
            print(video_dir.name)

            elems = video_dir.name.split("_")
            # hardcoded
            robot_init_x = round(float(elems[1]), 3)
            robot_init_y = round(float(elems[2]), 3)

            video_paths = list(video_dir.glob("./*.mp4"))
            assert len(video_paths) == 1, video_paths
            for video_path in video_paths:
                success = "success" in video_path.stem
                qpos = float(video_path.stem.split("_")[-1])
                results.append([task_name, (robot_init_y, robot_init_x), success, qpos])
        _df = pd.DataFrame(
            results, columns=["task_name", "layout_id", "success", "qpos"]
        )
        _df.sort_values(["layout_id"], inplace=True)
        df = pd.concat([df, _df], ignore_index=True)

    output_fname = task_suffix
    df.to_csv(result_dir / f"{output_fname}.csv")


TASK_SUFFIX = "shader_dir_rt"
# TASK_SUFFIX = "shader_dir_rt_light_mode_brighter"
# TASK_SUFFIX = "shader_dir_rt_light_mode_darker"
# TASK_SUFFIX = "shader_dir_rt_station_name_mk_station2"
# TASK_SUFFIX = "shader_dir_rt_station_name_mk_station3"

# parse_open_drawer_results_sim(
#     "results/xid77467904_000400120/frl_apartment_stage_simple/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner",
#     task_suffix=TASK_SUFFIX,
# )
# parse_open_drawer_results_sim(
#     "results/rt1poor_xid77467904_000058240/frl_apartment_stage_simple/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner",
#     task_suffix=TASK_SUFFIX,
# )
# parse_open_drawer_results_sim(
#     "results/rt_1_x_tf_trained_for_002272480_step/frl_apartment_stage_simple/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner",
#     task_suffix=TASK_SUFFIX,
# )
