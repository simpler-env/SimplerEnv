import glob
from pathlib import Path
from typing import Sequence, Optional

import numpy as np
from pandas.core.frame import frame_sub_kwargs

REAL_PERF = {  # Real robot eval performance --> extract via: REAL_PERF[task][policy]
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


SIMPLER_PERF = {  # SIMPLER simulated eval performance --> extract via: SIMPLER_PERF[task][policy]
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

REF = {
    "ckpt_name": [
        "RT-1(Converged)",
        "RT-1(15%)",
        "RT-1-X",
        "RT-2-X",
        "Octo-Base",
        "Octo-Small",
        "RT-1(begin)",
        "OpenVLA",
        "RoboVLM",
    ],
    "coke_can/matching_avg": [0.857, 0.710, 0.567, 0.787, 0.170, "nan", 0.027, 0.163, 0.727],
    "coke_can/variant_avg": [0.898, 0.813, 0.490, 0.823, 0.006, "nan", 0.022, 0.545, "nan"],
    "coke_can/matching/horizontal": [
        0.960,
        0.860,
        0.820,
        0.740,
        0.210,
        "nan",
        0.050,
        0.270,
        0.850,
    ],
    "coke_can/matching/vertical": [
        0.900,
        0.790,
        0.330,
        0.740,
        0.210,
        "nan",
        0.000,
        0.030,
        0.430,
    ],
    "coke_can/matching/standing": [
        0.710,
        0.480,
        0.550,
        0.880,
        0.090,
        "nan",
        0.030,
        0.190,
        0.900,
    ],
    "coke_can/variant/horizontal": [
        0.969,
        0.920,
        0.569,
        0.822,
        0.005,
        "nan",
        0.022,
        0.711,
        "nan",
    ],
    "coke_can/variant/vertical": [
        0.760,
        0.704,
        0.204,
        0.754,
        0.000,
        "nan",
        0.013,
        0.271,
        "nan",
    ],
    "coke_can/variant/standing": [
        0.964,
        0.813,
        0.698,
        0.893,
        0.013,
        "nan",
        0.031,
        0.653,
        "nan",
    ],
    "move_near/variant": [0.500, 0.446, 0.323, 0.792, 0.031, "nan", 0.040, 0.477, "nan"],
    "move_near/matching": [0.442, 0.354, 0.317, 0.779, 0.042, "nan", 0.050, 0.462, 0.663],
    "drawer/matching_avg": [0.730, 0.565, 0.597, 0.250, 0.227, "nan", 0.139, 0.356, 0.268],
    "drawer/variant_avg": [0.323, 0.267, 0.294, 0.353, 0.011, "nan", 0.069, 0.177, "nan"],
    "drawer/matching/open": [0.601, 0.463, 0.296, 0.157, 0.009, "nan", 0.000, 0.194, 0.287],
    "drawer/matching/close": [0.861, 0.667, 0.891, 0.343, 0.444, "nan", 0.278, 0.518, 0.25],
    "drawer/variant/open": [0.270, 0.212, 0.069, 0.333, 0.000, "nan", 0.005, 0.158, "nan"],
    "drawer/variant/close": [0.376, 0.323, 0.519, 0.372, 0.021, "nan", 0.132, 0.195, "nan"],
    "apple_in_drawer/matching_avg": [0.065, 0.130, 0.213, 0.037, 0.000, 0.000, 0.000, "nan", 0.361],
    "apple_in_drawer/variant_avg": [0.026, 0.021, 0.101, 0.206, 0.000, 0.000, 0.000, "nan", "nan"],
    "put_spoon_on_tablecloth/matching_partial": [
        "nan",
        "nan",
        0.167,
        "nan",
        0.347,
        0.778,
        "nan",
        0.041,
        0.375
    ],
    "put_spoon_on_tablecloth/matching_entire": [
        "nan",
        "nan",
        0.000,
        "nan",
        0.125,
        0.472,
        "nan",
        0.000,
        0.208
    ],
    "put_carrot_on_plate/matching_partial": [
        "nan",
        "nan",
        0.208,
        "nan",
        0.528,
        0.278,
        "nan",
        0.333,
        0.333
    ],
    "put_carrot_on_plate/matching_entire": [
        "nan",
        "nan",
        0.042,
        "nan",
        0.083,
        0.097,
        "nan",
        0.000,
        0.25
    ],
    "stack_green_block_on_yellow_block/matching_partial": [
        "nan",
        "nan",
        0.083,
        "nan",
        0.319,
        0.403,
        "nan",
        0.125,
        0.083
    ],
    "stack_green_block_on_yellow_block/matching_entire": [
        "nan",
        "nan",
        0.000,
        "nan",
        0.000,
        0.042,
        "nan",
        0.000,
        0.083
    ],
    "put_eggplant_in_basket/matching_partial": [
        "nan",
        "nan",
        0.000,
        "nan",
        0.667,
        0.875,
        "nan",
        0.083,
        0.000
    ],
    "put_eggplant_in_basket/matching_entire": [
        "nan",
        "nan",
        0.000,
        "nan",
        0.431,
        0.569,
        "nan",
        0.041,
        0.000
    ],
}


def pearson_correlation(perf_sim: Sequence[float], perf_real: Sequence[float]) -> float:
    perf_sim, perf_real = np.array(perf_sim), np.array(perf_real)
    assert perf_sim.shape == perf_real.shape
    perf_sim = perf_sim - np.mean(perf_sim)
    perf_real = perf_real - np.mean(perf_real)
    if np.all(perf_sim == perf_real):
        pearson = 1
    else:
        pearson = np.sum(perf_sim * perf_real) / (
            np.sqrt(np.sum(perf_sim**2) * np.sum(perf_real**2)) + 1e-8
        )
    return pearson


def mean_maximum_rank_violation(
    perf_sim: Sequence[float], perf_real: Sequence[float]
) -> float:
    perf_sim, perf_real = np.array(perf_sim), np.array(perf_real)
    assert perf_sim.shape == perf_real.shape
    rank_violations = []
    for i in range(len(perf_sim)):
        rank_violation = 0.0
        for j in range(len(perf_sim)):
            if (perf_sim[i] > perf_sim[j]) != (perf_real[i] > perf_real[j]):
                rank_violation = max(
                    rank_violation, np.abs(perf_real[i] - perf_real[j])
                )
        rank_violations.append(rank_violation)
    rank_violation = np.mean(rank_violations)
    return rank_violation


def print_all_kruskal_results(
    sim: Sequence[Sequence[float]], real: Sequence[Sequence[float]], title: str
) -> None:
    """
    sim, real: shape [n_ckpt, n_trials]
        The trial-by-trial success indicator of each checkpoint
        (within each checkpoint, the ordering doesn't matter)
    Prints out the Kruskal-Wallis test for each checkpoint
    """
    from scipy.stats import kruskal

    sim, real = np.array(sim), np.array(real)
    assert sim.shape == real.shape
    print(title)
    # print(" " * 6, "overall kruskal", kruskal(sim.reshape(-1), real.reshape(-1)))
    print(" " * 6, "each checkpoint kruskal:")
    for i in range(sim.shape[0]):
        if np.all(sim[i] == real[i]):
            # handle a bug of scipy.kruskal; in this case p-value should be 1.0
            print(" " * 12, "all same, 1.0")
        else:
            print(" " * 12, kruskal(sim[i], real[i]))


def construct_unordered_trial_results(
    n_trials_per_ckpt: int, success: Sequence[float]
) -> np.ndarray:
    success = np.array(success)
    success = np.where(np.isnan(success), 0, success)
    n_success_trials = np.round(n_trials_per_ckpt * success).astype(np.int32)
    results = []
    for nst in n_success_trials:
        results.append([1] * nst + [0] * (n_trials_per_ckpt - nst))
    return np.array(results)


# util to get success / failure results from a directory
def get_dir_stats(
    dir_name: str,
    extra_pattern_require: Optional[Sequence[str]] = [],
    succ_fail_pattern: Sequence[str] = ["success", "failure"],
) -> Sequence[int]:
    if dir_name[-1] == "/":
        dir_name = dir_name[:-1]

    results = []
    fnames = glob.glob(dir_name + "/**/*.mp4", recursive=True)
    for fname in fnames:
        flag = True
        for pattern in extra_pattern_require:
            if pattern not in fname:
                flag = False
                break
        if not flag:
            continue
        fname = Path(fname)
        if fname.suffix != ".mp4":
            continue
        fname = fname.stem
        if succ_fail_pattern[0] in fname:
            results.append(1)
        elif succ_fail_pattern[1] in fname:
            results.append(0)
    return results
