import os
import cv2
import numpy as np
from PIL import Image, ExifTags
import piexif
import sys
import argparse

# python video_to_image.py --folder /hdd/lightstage/lightstage.nrp-nautilus.io/xuanlin/coke --video-frame 0
parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, required=True)
parser.add_argument('--video-frame', type=int, default=-1, help="the specific frame to extract from video; -1 = all frames")
parser.add_argument('--skip-frame', type=int, default=1, help="skip every n frames if video_frame==-1")
parser.add_argument('--average-video', action='store_true', help="average all frames in video")
parser.add_argument('--remove-video', action='store_true')
args = parser.parse_args()

folder_path = args.folder
if "calibration" in folder_path:
    images = os.path.join(folder_path, "images")
else:
    images = folder_path
os.makedirs(images, exist_ok = True)

def align(s):
    while (len(s) < 4):
        s = '0' + s
    return s.lower()



def extract_video_to_single_image(folder, name, video_frame, average):
    cap = cv2.VideoCapture(os.path.join(folder, name))
    ave = None

    if video_frame >= 0:
        for i in range(video_frame + 1):
            success, frame = cap.read()
        assert success
        cv2.imwrite(os.path.join(images, align(name[: -4] + ".jpg")), frame.astype(np.uint8))
    else:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(total_frames):
            success, frame = cap.read()
            if ave is None:
                ave = np.zeros(frame.shape, dtype=np.uint32)
            if success:
                ave += frame
        ave = (ave / total_frames).astype(np.uint8)
        if average:
            cv2.imwrite(os.path.join(images, align(name[: -4] + ".jpg")), ave)
    
    cap.release()
        
    if args.remove_video:
        os.remove(os.path.join(folder, name))
        
def merge_all_videos_to_images(video_folder, image_folder, video_names, skip_frame):
    frame_count = 0
    os.makedirs(image_folder, exist_ok=True)
    for video_name in video_names:
        cap = cv2.VideoCapture(os.path.join(video_folder, video_name))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(total_frames):
            success, frame = cap.read()
            if success and i % skip_frame == 0:
                if frame.shape[0] > frame.shape[1]:
                    frame = cv2.transpose(frame)
                cv2.imwrite(os.path.join(image_folder, align(f"{frame_count}.jpg")), frame.astype(np.uint8))
                frame_count += 1
        cap.release()
        if args.remove_video:
            os.remove(os.path.join(video_folder, video_name))

for image in os.listdir(folder_path):
    if ("jpg" in image.lower()):
        # delete_exif_and_resize(folder_path, image)
        pass
    elif ("mp4" in image.lower() or "mov" in image.lower()):
        if args.video_frame > 0 or args.average_video:
            extract_video_to_single_image(folder_path, image, args.video_frame, args.average_video)
            
if args.video_frame < 0 and not args.average_video:
    # extract all video frames to images
    video_list = sorted(os.listdir(folder_path))
    video_list = [video for video in video_list if "mp4" in video.lower() or "mov" in video.lower()]
    merge_all_videos_to_images(
        folder_path,
        os.path.join(folder_path, 'extracted_images'), 
        video_list,
        args.skip_frame,
    )

