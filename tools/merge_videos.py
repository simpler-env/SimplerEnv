import argparse
from functools import partial
from pathlib import Path
from typing import List

import cv2
import moviepy
import numpy as np
from moviepy.editor import VideoFileClip, clips_array


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", type=Path, required=True)
    parser.add_argument("-o", "--output-path", type=str)

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


def merge_videos(input_dir: Path, output_path:str= None):
    video_paths = []
    for video_path in input_dir.glob("**/*.mp4"):
        video_paths.append(video_path)
        # print(video_path)

    # video_clips = []
    video_clips = {}
    if 'obj' in video_paths[0].stem.split('_'):
        pos_variation_mode = 'obj'
    else:
        pos_variation_mode = 'robot'
    init_xs = set()
    init_ys = set()
    max_duration = 0
    for video_path in video_paths:
        # robot initial poses
        dirname = video_path.parent
        dirname_elems = dirname.name.split("_")
        # hardcoded
        robot_init_x = round(float(dirname_elems[1]), 3)
        robot_init_y = round(float(dirname_elems[2]), 3)
        
        basename = video_path.stem
        basename_elems = basename.split("_")
        # hardcoded
        success = basename_elems[0]
        if pos_variation_mode == 'obj':
            obj_init_x = round(float(basename_elems[2]), 3)
            obj_init_y = round(float(basename_elems[3]), 3)
        try:
            qpos = round(float(basename_elems[-1]), 3)
        except:
            if pos_variation_mode == 'obj':
                qpos = (obj_init_x, obj_init_y)
            else:
                qpos = None
            
        if pos_variation_mode == 'robot':
            init_x, init_y = robot_init_x, robot_init_y
            init_xs.add(robot_init_x)
            init_ys.add(robot_init_y)
        elif pos_variation_mode == 'obj':
            init_x, init_y = obj_init_x, obj_init_y
            init_xs.add(obj_init_x)
            init_ys.add(obj_init_y)
        else:
            raise NotImplementedError()

        video_clip = VideoFileClip(str(video_path))
        max_duration = max(max_duration, video_clip.duration)
        # video_clips.append(video_clip)
        video_clips[(init_x, init_y)] = (video_clip, success, qpos)

    def add_text_to_clip(get_frame, t, text_fn, start=0):
        frame = np.array(get_frame(t))
        if t >= start:
            text_fn(frame)
        return frame

    final_clip_array = []
    for init_x in sorted(init_xs):
        final_clip_array.append([])
        for init_y in sorted(init_ys):
            video_clip: VideoFileClip = video_clips[(init_x, init_y)][0]
            video_clip = video_clip.set_duration(max_duration)

            success = video_clips[(init_x, init_y)][1]
            qpos = video_clips[(init_x, init_y)][2]
            text_fn = partial(
                put_text_on_image,
                lines=["success: " + success, "qpos: " + str(qpos)],
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
    merge_videos(args.input_dir, args.output_path)


if __name__ == "__main__":
    main()
