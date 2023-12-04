from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import glob
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()
    
    if args.dir[-1] == '/':
        args.dir = args.dir[:-1]
    robot_x, robot_y = [float(num) for num in args.dir.split('/')[-1].split('_')[1:3]]
    
    success = []
    failure = []
    
    fnames = glob.glob(args.dir + '/*')
    for fname in fnames:
        fname = fname.split('/')[-1].replace('.mp4', '')
        x, y = [float(num) for num in fname.split('_')[-2:]]
        x, y = abs(x - robot_x), y - robot_y
        if 'success' in fname:
            success.append([x, y])
        elif 'failure' in fname:
            failure.append([x, y])
    
    print("**Success Rate: {}**".format(len(success) / (len(success) + len(failure))))
    
    plt.figure()
    if len(success) > 0:
        plt.plot(np.array(success)[:, 1], np.array(success)[:, 0], 'o', color='green', label='success')
    if len(failure) > 0:
        plt.plot(np.array(failure)[:, 1], np.array(failure)[:, 0], 'o', color='red', label='failure')
    plt.xlabel('object y from robot')
    plt.ylabel('object x from robot')
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()