import argparse
import re

def obtain_arr(s):
    assert s[0] == '[' and s[-1] == ']', s
    s = s[1:-1]
    s = re.split(',? +', s)
    return [float(x) for x in s]
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-file', type=str, default='/home/xuanlin/Downloads/opt_results.txt')
    args = parser.parse_args()
    
    with open(args.log_file, 'r') as f:
        lines = f.readlines()
        
    results = []
    for line in lines:
        line = line.strip()
        line = line.split(':')
        line = [x.strip() for x in line]
        arm_stiffness = obtain_arr(','.join(line[1].split(',')[:-1]))
        arm_damping = obtain_arr(','.join(line[2].split(',')[:-1]))
        avg_err = eval(line[3].split(',')[0].strip())
        per_traj_err = obtain_arr(line[4].strip())
        results.append((avg_err, (arm_stiffness, arm_damping), per_traj_err))
        
    results = sorted(results, key=lambda x: x[0])
    for result in results[:10]:
        print(result)
        