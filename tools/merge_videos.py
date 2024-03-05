"""
Merge videos into a single video with multiple subvideos.
e.g.,
python -m pdb tools/merge_videos.py \
    --input-dir results/rt_1_tf_trained_for_000400120/google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/GraspSingleOpenedCokeCanInScene-v0_upright_True_urdf_version_None/rob_0.35_0.2_rot_0.000_-0.000_3.142_rgb_overlay_google_coke_can_real_eval_1 \
    --output-path results/rt_1_tf_trained_for_000400120/google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/GraspSingleOpenedCokeCanInScene-v0_upright_True_urdf_version_None/merged.mp4
"""

import argparse
from functools import partial
from pathlib import Path
from typing import List

import cv2
import moviepy
from moviepy.editor import clips_array, ColorClip, VideoFileClip
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", type=Path, required=True)
    parser.add_argument("-o", "--output-path", type=str)
    parser.add_argument("--no-text-on-video", action="store_true")

    args = parser.parse_args()
    return args


def put_text_on_image(
    image: np.ndarray,
    lines: List[str],
    font_size=1,
    font_thickness=1,
    color=(255, 255, 255),
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 0
    for line in lines:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        x = 10
        cv2.putText(
            image,
            line,
            (x, y),
            font,
            font_size,
            color,
            font_thickness,
            lineType=cv2.LINE_AA,
        )


def merge_videos(input_dir: Path, output_path: str = None, no_text_on_video: bool = False):
    video_paths = []
    for video_path in input_dir.glob("**/*.mp4"):
        video_paths.append(video_path)
        print(video_path)
    n_videos = len(video_paths)

    video_clips = {}
    example_video_path_elems = video_paths[0].stem.split("_")
    if "qpos" in example_video_path_elems:
        # cabinet qpos
        pos_variation_mode = "qpos"
    elif "episode" in example_video_path_elems:
        # episode-based object pos variation
        pos_variation_mode = "episode"
    elif "obj" in example_video_path_elems:
        # position-based object variation
        pos_variation_mode = "obj"
    else:
        pos_variation_mode = "robot"

    init_xs = set()
    init_ys = set()
    max_duration = 0
    video_clip_size = None
    for video_path in video_paths:
        # robot initial poses
        dirname = video_path.parent
        dirname_elems = dirname.name.split("_")

        rob_idx = dirname_elems.index("rob")
        robot_init_x = round(float(dirname_elems[rob_idx + 1]), 3)
        robot_init_y = round(float(dirname_elems[rob_idx + 2]), 3)

        basename = video_path.stem
        basename_elems = basename.split("_")
        # hardcoded
        success = basename_elems[0]
        if pos_variation_mode == "qpos":
            qpos_idx = basename_elems.index("qpos")
            qpos = round(float(basename_elems[qpos_idx + 1]), 3)
            video_clip_additional_info = f"qpos: {qpos}"
        elif pos_variation_mode == "obj":
            obj_idx = basename_elems.index("obj")
            obj_init_x = round(float(basename_elems[obj_idx + 1]), 3)
            obj_init_y = round(float(basename_elems[obj_idx + 2]), 3)
            video_clip_additional_info = f"obj_init: {(obj_init_x, obj_init_y)}"
        elif pos_variation_mode == "episode":
            episode_idx = basename_elems.index("episode")
            episode = int(basename_elems[episode_idx + 1])
            video_clip_additional_info = f"episode: {episode}"
        else:
            video_clip_additional_info = None

        if pos_variation_mode in ["robot", "qpos"]:
            init_x, init_y = robot_init_x, robot_init_y
            init_xs.add(robot_init_x)
            init_ys.add(robot_init_y)
        elif pos_variation_mode == "obj":
            init_x, init_y = obj_init_x, obj_init_y
            init_xs.add(obj_init_x)
            init_ys.add(obj_init_y)
        elif pos_variation_mode == "episode":
            episode_idx = basename_elems.index("episode")
            episode = int(basename_elems[episode_idx + 1])
            merged_video_side_length = int(np.ceil(np.sqrt(n_videos)))
            init_x, init_y = (
                episode // merged_video_side_length,
                episode % merged_video_side_length,
            )
            init_xs.add(init_x)
            init_ys.add(init_y)
        else:
            raise NotImplementedError()

        video_clip = VideoFileClip(str(video_path))
        if video_clip_size is None:
            video_clip_size = video_clip.size
        max_duration = max(max_duration, video_clip.duration)
        video_clips[(init_x, init_y)] = (
            video_clip,
            success,
            video_clip_additional_info,
        )

    def add_text_to_clip(get_frame, t, text_fn, start=0):
        frame = np.array(get_frame(t))
        if t >= start:
            text_fn(frame)
        return frame

    final_clip_array = []
    for init_x in sorted(init_xs):
        final_clip_array.append([])
        for init_y in sorted(init_ys):
            if (pos_variation_mode == "episode") and (
                init_x,
                init_y,
            ) not in video_clips:
                # for episode-based object pos variation, we may have empty slots since the number of episodes might not be a perfect square
                pad_video_clip = ColorClip(size=video_clip_size, color=(0, 0, 0)).set_duration(max_duration)
                final_clip_array[-1].append(pad_video_clip)
                continue
            video_clip: VideoFileClip = video_clips[(init_x, init_y)][0]
            video_clip = video_clip.set_duration(max_duration)

            success = video_clips[(init_x, init_y)][1]
            video_clip_additional_info = video_clips[(init_x, init_y)][2]
            if not no_text_on_video:
                text_fn = partial(
                    put_text_on_image,
                    lines=[
                        "success: " + success,
                        video_clip_additional_info if video_clip_additional_info is not None else "",
                    ],
                    color=(255, 0, 0) if success != "success" else (0, 255, 0),
                )
                video_clip = video_clip.fl(partial(add_text_to_clip, text_fn=text_fn))

            final_clip_array[-1].append(video_clip)

    if output_path is None:
        output_path = input_dir.parent / "{}_merged.mp4".format(input_dir.name)
        output_path = str(output_path)

    final_clip = clips_array(final_clip_array)
    final_clip.resize(width=1920).write_videofile(output_path)


def main():
    args = parse_args()
    merge_videos(args.input_dir, args.output_path, args.no_text_on_video)


if __name__ == "__main__":
    main()
