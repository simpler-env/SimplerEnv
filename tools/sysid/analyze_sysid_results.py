"""
Parse the results of the system identification logs and print the top 10 results.
"""

import argparse
import re


def obtain_arr(s):
    assert s[0] == "[" and s[-1] == "]", s
    s = s[1:-1]
    s = re.split(",? +", s)
    return [float(x) for x in s]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=str, default="sysid_log/opt_results_google_robot.txt")
    args = parser.parse_args()

    with open(args.log_file, "r") as f:
        lines = f.readlines()

    results = []
    for line in lines:
        line = line.strip()
        line = line.split(":")
        line = [x.strip() for x in line]
        arm_stiffness = obtain_arr(",".join(line[1].split(",")[:-1]))
        arm_damping = obtain_arr(",".join(line[2].split(",")[:-1]))
        misc = [eval(line[i].split(",")[0].strip()) for i in range(3, len(line) - 2)]
        avg_err = eval(line[-2].split(",")[0].strip())
        per_traj_err = obtain_arr(line[-1].strip())
        results.append((avg_err, (arm_stiffness, arm_damping), misc, per_traj_err))

    results = sorted(results, key=lambda x: x[0])
    for result in results[:10]:
        print(
            f"Avg error: {result[0]}; Arm stiffness: {result[1][0]}; Arm damping: {result[1][1]}; Misc: {result[2]}; Per traj error: {result[3]}"
        )
