import glob
from pathlib import Path
from typing import Sequence, Optional

import numpy as np


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
