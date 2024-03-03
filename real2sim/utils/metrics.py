import glob
from pathlib import Path

import numpy as np
from scipy.stats import kruskal


def pearson_correlation(x, y):
    x, y = np.array(x), np.array(y)
    assert x.shape == y.shape
    x = x - np.mean(x)
    y = y - np.mean(y)
    if np.all(x == y):
        pearson = 1
    else:
        pearson = np.sum(x * y) / (np.sqrt(np.sum(x**2) * np.sum(y**2)) + 1e-8)
    return pearson


def normalized_rank_loss(x, y):
    # assuming x is sim result and y is real result
    x, y = np.array(x), np.array(y)
    assert x.shape == y.shape
    rank_violation = 0.0
    for i in range(len(x) - 1):
        for j in range(i + 1, len(x)):
            if (x[i] > x[j]) != (y[i] > y[j]):
                rank_violation = max(rank_violation, np.abs(y[i] - y[j]))
    return rank_violation


def print_all_kruskal_results(sim, real, title):
    """
    sim, real: shape [n_ckpt, n_trials]
        The trial-by-trial success indicator of each checkpoint
        (within each checkpoint, the ordering doesn't matter)
    Prints out the Kruskal-Wallis test for each checkpoint
    """
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


def construct_unordered_trial_results(n_trials_per_ckpt, success):
    success = np.array(success)
    success = np.where(np.isnan(success), 0, success)
    n_success_trials = np.round(n_trials_per_ckpt * success).astype(np.int32)
    results = []
    for nst in n_success_trials:
        results.append([1] * nst + [0] * (n_trials_per_ckpt - nst))
    return np.array(results)


# util to get success / failure results from a directory
def get_dir_stats(
    dir_name, extra_pattern_require=[], succ_fail_pattern=["success", "failure"]
):
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
