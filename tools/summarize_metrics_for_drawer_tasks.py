import os

import numpy as np
import pandas as pd
import scipy.stats

ROOT_DIR = os.getenv("RESULT_ROOT", ".")


# ---------------------------------------------------------------------------- #
# Global
# ---------------------------------------------------------------------------- #
def ranking_violation(x, y):
    # assuming x is sim result and y is real result
    x, y = np.array(x), np.array(y)
    assert x.shape == y.shape
    rank_violation = 0.0
    for i in range(len(x) - 1):
        for j in range(i + 1, len(x)):
            if (x[i] > x[j]) != (y[i] > y[j]):
                rank_violation += np.abs(y[i] - y[j])
    return rank_violation / (len(x) * (len(x) - 1) / 2)


# ---------------------------------------------------------------------------- #
# Local
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Summarize
# ---------------------------------------------------------------------------- #
CKPT_MAPPING = {
    "rt-1-400k": "xid77467904_000400120",
    "rt-1-58k": "rt1poor_xid77467904_000058240",
    "rt-1-x": "rt_1_x_tf_trained_for_002272480_step",
    "rt-1-1k": "rt1new_77467904_000001120",
}

TASK_NAMES = [
    "OpenTopDrawer",
    "OpenMiddleDrawer",
    "OpenBottomDrawer",
    "CloseTopDrawer",
    "CloseMiddleDrawer",
    "CloseBottomDrawer",
]


def get_results(csv_path, task_name):
    df = pd.read_csv(csv_path)
    if task_name is not None:
        df = df[df["task_name"].str.contains(task_name)]
    return df["success"]


def compute_metrics_real_vs_sim():
    # scene_name = "dummy2"
    scene_name = "frl_apartment_stage_simple"
    DEFAULT_CONTROLLER = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
    # model_names = ["rt-1-400k", "rt-1-58k", "rt-1-x", "rt-1-1k"]
    model_names = ["rt-1-400k", "rt-1-58k", "rt-1-x"]
    # env_suffix = "shader_dir_rt_station_name_mk_station_recolor_light_mode_simple_disable_bad_material_True_urdf_version_recolor"
    # env_suffix = "shader_dir_rt_station_name_mk_station_recolor_light_mode_simple_disable_bad_material_True_urdf_version_recolor2"
    # env_suffix = "shader_dir_rt_station_name_mk_station_recolor_light_mode_simple_disable_bad_material_True"
    # env_suffix = "shader_dir_rt_station_name_mk_station2_light_mode_simple_disable_bad_material_True_urdf_version_recolor"
    # env_suffix = "shader_dir_rt_station_name_mk_station2"
    # env_suffix = "shader_dir_rt_station_name_mk_station_recolor_light_mode_simple_disable_bad_material_True"
    env_suffix = "shader_dir_rt_station_name_mk_station2_light_mode_simple_disable_bad_material_True"

    df_all_tasks = pd.DataFrame()
    for task_name in ["Open", "Close", "Drawer"]:
        success_real_all_ckpts = {}
        success_sim_all_ckpts = {}

        for model_name in model_names:
            csv_path_real = f"results/real/{model_name}.csv"
            ckpt_name = CKPT_MAPPING[model_name]
            csv_path_sim = f"{ROOT_DIR}/results/{ckpt_name}/{scene_name}/{DEFAULT_CONTROLLER}/{env_suffix}.csv"

            sucess_real = get_results(csv_path_real, task_name)
            sucess_sim = get_results(csv_path_sim, task_name)
            success_real_all_ckpts[model_name] = sucess_real
            success_sim_all_ckpts[model_name] = sucess_sim

        # Success rates
        success_rates_real = []
        success_rates_sim = []
        for model_name in model_names:
            success_real = success_real_all_ckpts[model_name]
            success_sim = success_sim_all_ckpts[model_name]
            success_rates_real.append(np.mean(success_real))
            success_rates_sim.append(np.mean(success_sim))
        df = pd.DataFrame(
            [success_rates_real, success_rates_sim],
            columns=model_names,
        )
        df_all_tasks = pd.concat([df_all_tasks, df], axis=1, ignore_index=True)

        rank_loss = ranking_violation(success_rates_real, success_rates_sim)
        print(task_name, rank_loss)

        # print("Pearson:", scipy.stats.pearsonr(success_rates_real, success_rates_sim))
        # print(
        #     "Kruskal:",
        #     scipy.stats.kruskal(
        #         np.concatenate(all_trials_real), np.concatenate(all_trials_sim)
        #     ),
        # )
        # for i, ckpt_name in enumerate(ckpt_names):
        #     print(
        #         ckpt_name,
        #         "Kruskal:",
        #         scipy.stats.kruskal(all_trials_real[i], all_trials_sim[i]),
        #     )

        # print(df.to_csv(sep=',', index=False))

    print(df_all_tasks.to_csv(sep=',', index=False))


def compute_sim_variants():
    TASK_NAMES = [
        "OpenTopDrawer",
        "OpenMiddleDrawer",
        "OpenBottomDrawer",
        "CloseTopDrawer",
        "CloseMiddleDrawer",
        "CloseBottomDrawer",
    ]

    ckpt_names = ["rt1-late", "rt1-early", "rt1-x"]
    variant_names = [
        "frl_apartment_stage_simple/shader_dir_rt",
        "frl_apartment_stage_simple/shader_dir_rt_light_mode_brighter",
        "frl_apartment_stage_simple/shader_dir_rt_light_mode_darker",
        "frl_apartment_stage_simple/shader_dir_rt_station_name_mk_station2",
        "frl_apartment_stage_simple/shader_dir_rt_station_name_mk_station3",
        "modern_bedroom/shader_dir_rt",
        "modern_office/shader_dir_rt",
    ]
    for task_name in ["Open", "Close", "Drawer"]:
        # for task_name in TASK_NAMES:
        success_rates_real = []
        success_rates_sim = []
        success_rates_sim_all_variants = []
        for ckpt_name in ckpt_names:
            df_real = pandas.read_csv("results/csv/" + ckpt_name + "/real.csv")
            trials_real = df_real[df_real["task_name"].str.contains(task_name)][
                "success"
            ]
            success_rates_real.append(trials_real.mean())

            success_rates_sim_per_ckpt = []
            for variant in variant_names:
                df_sim = pandas.read_csv("results/csv/" + ckpt_name + f"/{variant}.csv")
                trials_sim = df_sim[df_sim["task_name"].str.contains(task_name)][
                    "success"
                ]
                success_rates_sim_per_ckpt.append(trials_sim.mean())

            success_rates_sim_all_variants.append(success_rates_sim_per_ckpt)
            success_rates_sim.append(np.mean(success_rates_sim_per_ckpt))

        print(task_name + ":" + str(ckpt_names))
        print("success rate (real)", success_rates_real)
        print("success rate (sim)", success_rates_sim)
        for ckpt_name, success_rates_sim_per_ckpt in zip(
            ckpt_names, success_rates_sim_all_variants
        ):
            print(
                "success rate (sim) all variants for",
                ckpt_name,
                ":",
                success_rates_sim_per_ckpt,
            )
        print(
            "normalized rank loss",
            ranking_violation(success_rates_sim, success_rates_real),
        )


if __name__ == "__main__":
    compute_metrics_real_vs_sim()
    # compute_sim_variants()
