from pathlib import Path
import pandas as pd
import os

# Specify if you have a subdirectory structure
ROOT_DIR = os.getenv("RESULT_ROOT", ".")

TASK_NAMES = [
    "OpenTopDrawerCustomInScene-v0",
    "OpenMiddleDrawerCustomInScene-v0",
    "OpenBottomDrawerCustomInScene-v0",
    "CloseTopDrawerCustomInScene-v0",
    "CloseMiddleDrawerCustomInScene-v0",
    "CloseBottomDrawerCustomInScene-v0",
]

LAYOUT_IDS = ["a0", "a1", "a2", "b0", "b1", "b2", "c0", "c1", "c2"]

DEFAULT_CONTROLLER = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"


def parse_open_drawer_results_overlay(result_dir, env_suffix):
    result_dir = Path(result_dir)
    df = pd.DataFrame()

    for task_name in TASK_NAMES:
        env_name = task_name + "_" + env_suffix
        task_dir = result_dir / env_name
        # print(task_name)

        results = []
        for video_dir in sorted(task_dir.iterdir()):
            # print(video_dir.name)

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

    csv_path = result_dir / f"{env_suffix}.csv"
    print(csv_path)
    df.to_csv(csv_path)


# Overlay main
for ckpt_name in [
    "xid77467904_000400120",
    "rt_1_x_tf_trained_for_002272480_step",
    "rt1poor_xid77467904_000058240",
    "rt1new_77467904_000001120",
    # "rt1new_77467904_000003080",
]:
    for env_suffix in [
        "shader_dir_rt_station_name_mk_station_recolor_light_mode_simple_disable_bad_material_True_urdf_version_recolor",
    ]:
        result_dir = f"{ROOT_DIR}/results/{ckpt_name}/dummy_drawer/{DEFAULT_CONTROLLER}"
        parse_open_drawer_results_overlay(result_dir, env_suffix)


# Ablations
for ckpt_name in [
    "xid77467904_000400120",
    "rt_1_x_tf_trained_for_002272480_step",
    "rt1poor_xid77467904_000058240",
]:
    for env_suffix in [
        "shader_dir_rt_station_name_mk_station_recolor_light_mode_simple_disable_bad_material_True",
        "shader_dir_rt_station_name_mk_station_recolor_light_mode_simple_disable_bad_material_True_urdf_version_recolor2",
    ]:
        result_dir = f"{ROOT_DIR}/results/{ckpt_name}/dummy_drawer/{DEFAULT_CONTROLLER}"
        parse_open_drawer_results_overlay(result_dir, env_suffix)


def parse_open_drawer_results_sim(result_dir, env_suffix):
    result_dir = Path(result_dir)
    df = pd.DataFrame()

    for task_name in TASK_NAMES:
        env_name = task_name + "_" + env_suffix
        env_dir = result_dir / env_name
        # print(task_name)

        results = []
        for video_dir in sorted(env_dir.iterdir()):
            # print(video_dir.name)

            elems = video_dir.name.split("_")
            # Hardcoded for drawer tasks
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

    csv_path = result_dir / f"{env_suffix}.csv"
    print(csv_path)
    df.to_csv(csv_path)


# Sim variants
for ckpt_name in [
    "xid77467904_000400120",
    "rt_1_x_tf_trained_for_002272480_step",
    "rt1poor_xid77467904_000058240",
    "rt1new_77467904_000001120",
    # "rt1new_77467904_000003080",
]:
    for env_suffix in [
        "shader_dir_rt_station_name_mk_station_recolor_light_mode_simple_disable_bad_material_True_urdf_version_recolor",
        "shader_dir_rt_station_name_mk_station_recolor_light_mode_simple_brighter_disable_bad_material_True_urdf_version_recolor",
        "shader_dir_rt_station_name_mk_station_recolor_light_mode_simple_darker_disable_bad_material_True_urdf_version_recolor",
        "shader_dir_rt_station_name_mk_station2_light_mode_simple_disable_bad_material_True_urdf_version_recolor",
        "shader_dir_rt_station_name_mk_station3_light_mode_simple_disable_bad_material_True_urdf_version_recolor",
    ]:
        result_dir = f"{ROOT_DIR}/results/{ckpt_name}/frl_apartment_stage_simple/{DEFAULT_CONTROLLER}"
        parse_open_drawer_results_sim(result_dir, env_suffix)

    env_suffix = "shader_dir_rt_station_name_mk_station_recolor_light_mode_simple_disable_bad_material_True_urdf_version_recolor"
    for scene_name in [
        "modern_bedroom_no_roof",
        "modern_office_no_roof",
    ]:
        result_dir = f"{ROOT_DIR}/results/{ckpt_name}/{scene_name}/{DEFAULT_CONTROLLER}"
        parse_open_drawer_results_sim(result_dir, env_suffix)


# Ablations
for ckpt_name in [
    "xid77467904_000400120",
    "rt_1_x_tf_trained_for_002272480_step",
    "rt1poor_xid77467904_000058240",
]:
    for env_suffix in [
        "shader_dir_rt_station_name_mk_station2_light_mode_simple_disable_bad_material_True",
        "shader_dir_rt_station_name_mk_station_recolor_light_mode_simple_disable_bad_material_True",
    ]:
        result_dir = f"{ROOT_DIR}/results/{ckpt_name}/frl_apartment_stage_simple/{DEFAULT_CONTROLLER}"
        parse_open_drawer_results_sim(result_dir, env_suffix)
