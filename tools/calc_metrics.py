"""
Computes metrics for evaluating simulated evaluation pipelines.

Usage:
    from simpler_env.utils.metrics import mean_maximum_rank_violation, pearson_correlation

    sim_eval_perf = [
        your_sim_eval(task="google_robot_move_near", policy=p) 
        for p in ["rt-1-x", "octo", ...]
    ]
    real_eval_perf = [
        REAL_PERF["google_robot_move_near"][p] for p in ["rt-1-x", "octo", ...]
    ]
    mmrv = mean_maximum_rank_violation(real_eval_perf, sim_eval_perf)
    pearson = pearson_correlation(real_eval_perf, sim_eval_perf)

"""


from simpler_env.utils.metrics import mean_maximum_rank_violation, pearson_correlation



REAL_PERF = {    # Real robot eval performance --> extract via: REAL_PERF[task][policy]
    "google_robot_pick_coke_can": {
        "rt-2-x": 0.907,
        "rt-1-converged": 0.853,
        "rt-1-15pct": 0.920,
        "rt-1-x": 0.760,
        "rt-1-begin": 0.133,
        "octo-base": 0.293,
    },
    "google_robot_move_near": {
        "rt-2-x": 0.733,
        "rt-1-converged": 0.633,
        "rt-1-15pct": 0.583,
        "rt-1-x": 0.450,
        "rt-1-begin": 0.017,
        "octo-base": 0.350,
    },
    "google_robot_open_drawer": {
        "rt-2-x": 0.333,
        "rt-1-converged": 0.815,
        "rt-1-15pct": 0.704,
        "rt-1-x": 0.519,
        "rt-1-begin": 0.000,
        "octo-base": 0.148,
    },
    "google_robot_close_drawer": {
        "rt-2-x": 0.630,
        "rt-1-converged": 0.926,
        "rt-1-15pct": 0.889,
        "rt-1-x": 0.741,
        "rt-1-begin": 0.000,
        "octo-base": 0.519,
    },
    "google_robot_place_apple_in_closed_top_drawer": {
        "rt-2-x": 0.074,
        "rt-1-converged": 0.185,
        "rt-1-15pct": 0.185,
        "rt-1-x": 0.407,
        "rt-1-begin": 0.000,
        "octo-base": 0.000,
    },
    "widowx_spoon_on_towel": {
        "rt-1-x": 0.000,
        "octo-base": 0.333,
        "octo-small": 0.417,
    },
    "widowx_carrot_on_plate": {
        "rt-1-x": 0.000,
        "octo-base": 0.250,
        "octo-small": 0.083,
    },
    "widowx_stack_cube": {
        "rt-1-x": 0.000,
        "octo-base": 0.000,
        "octo-small": 0.125,
    },
    "widowx_put_eggplant_in_basket": {
        "rt-1-x": 0.000,
        "octo-base": 0.250,
        "octo-small": 0.400,
    },
}


SIMPLER_PERF = {    # SIMPLER simulated eval performance --> extract via: SIMPLER_PERF[task][policy]
    "google_robot_pick_coke_can": {
        "rt-2-x": 0.787,
        "rt-1-converged": 0.857,
        "rt-1-15pct": 0.710,
        "rt-1-x": 0.567,
        "rt-1-begin": 0.027,
        "octo-base": 0.170,
    },
    "google_robot_move_near": {
        "rt-2-x": 0.779,
        "rt-1-converged": 0.442,
        "rt-1-15pct": 0.354,
        "rt-1-x": 0.317,
        "rt-1-begin": 0.050,
        "octo-base": 0.042,
    },
    "google_robot_open_drawer": {
        "rt-2-x": 0.157,
        "rt-1-converged": 0.601,
        "rt-1-15pct": 0.463,
        "rt-1-x": 0.296,
        "rt-1-begin": 0.000,
        "octo-base": 0.009,
    },
    "google_robot_close_drawer": {
        "rt-2-x": 0.343,
        "rt-1-converged": 0.861,
        "rt-1-15pct": 0.667,
        "rt-1-x": 0.891,
        "rt-1-begin": 0.278,
        "octo-base": 0.444,
    },
    "google_robot_place_apple_in_closed_top_drawer": {
        "rt-2-x": 0.037,
        "rt-1-converged": 0.065,
        "rt-1-15pct": 0.130,
        "rt-1-x": 0.213,
        "rt-1-begin": 0.000,
        "octo-base": 0.000,
    },
    "widowx_spoon_on_towel": {
        "rt-1-x": 0.000,
        "octo-base": 0.125,
        "octo-small": 0.472,
    },
    "widowx_carrot_on_plate": {
        "rt-1-x": 0.042,
        "octo-base": 0.083,
        "octo-small": 0.097,
    },
    "widowx_stack_cube": {
        "rt-1-x": 0.000,
        "octo-base": 0.000,
        "octo-small": 0.042,
    },
    "widowx_put_eggplant_in_basket": {
        "rt-1-x": 0.000,
        "octo-base": 0.431,
        "octo-small": 0.569,
    },
}


if __name__ == "__main__":
    print("======= SIMPLER Evaluation =======\n")

    for k in SIMPLER_PERF.keys():
        print(f"{k}:")
        mmrv = mean_maximum_rank_violation(
            list(SIMPLER_PERF[k].values()), list(REAL_PERF[k].values())
        )
        pearson = pearson_correlation(
            list(SIMPLER_PERF[k].values()), list(REAL_PERF[k].values())
        )
        print(f"MMRV: {mmrv}, Pearson: {pearson}\n")
