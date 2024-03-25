"""
Save a particular frame from a video.
"""

import argparse
from pathlib import Path
from typing import List

import cv2
import mediapy as media
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-video", type=Path, required=True)
    parser.add_argument("-o", "--output-path", type=str, required=True)
    parser.add_argument("-n", "--frame", type=int, default=0, help="frame number")

    args = parser.parse_args()
    return args


def save_video_frame(input_video: Path, output_path: str = None, frame: int = 0):
    video = media.read_video(input_video)
    frame = video[frame]
    Image.fromarray(frame).save(output_path)


def main():
    args = parse_args()
    save_video_frame(args.input_video, args.output_path, args.frame)


if __name__ == "__main__":
    main()

"""
python tools/save_video_frame.py -i /home/xuanlin/simpler_env/results/rt_1_x_tf_trained_for_002272480_step/google_pick_coke_can_1_v4_alt_background/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/MoveNearGoogleInScene-v0/rob_0.35_0.21_rot_0.000_-0.000_3.052_rgb_overlay_None/failure_obj_episode_0_all_obj_keep_height_True_moved_correct_obj_False_moved_wrong_obj_True_near_tgt_obj_False_is_closest_to_tgt_False.mp4 \
    -o /home/xuanlin/simpler_env/ManiSkill2_real2sim/data/robustness_visualization/move_near_tab4_alt_bg_frame0.png
"""
