"""
Computes metrics for evaluating simulated evaluation pipelines.

Usage:
    from simpler_env.utils.metrics import mean_maximum_rank_violation, pearson_correlation, REAL_PERF

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


from simpler_env.utils.metrics import mean_maximum_rank_violation, pearson_correlation, REAL_PERF, SIMPLER_PERF


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
